from dataclasses import replace

import numpy as np
import pandas as pd

from src.adaptive_portfolio import PortfolioConfig, PortfolioSimulator
from src.minute_learning import minute_features


def raw(n=1000):
    index = pd.date_range('2026-01-01', periods=n, freq='min', tz='UTC')
    close = 100 + np.arange(n)*.02
    return {'KRW-BTC': pd.DataFrame({'open': close, 'high': close+.01, 'low': close-.01,
                                    'close': close, 'volume': 1000}, index=index)}


def test_context_features_never_see_incomplete_higher_timeframe():
    data = raw()
    cfg = PortfolioConfig(context_minutes=15, edge_cost_ratio=1.5)
    for length in (601, 607, 614, 615):
        prefix = minute_features({'KRW-BTC': data['KRW-BTC'].iloc[:length]}, cfg)
        full = minute_features(data, cfg)
        pd.testing.assert_frame_equal(prefix['KRW-BTC'], full['KRW-BTC'].iloc[:length])


def test_cost_gate_rejects_tiny_expected_move():
    data = raw()
    cfg = PortfolioConfig(entry_mode='trend')
    unfiltered = minute_features(data, cfg)['KRW-BTC']
    filtered = minute_features(data, replace(cfg, edge_cost_ratio=2.))['KRW-BTC']
    assert unfiltered.buy.sum() > 0
    assert filtered.buy.sum() == 0


def test_minimum_stop_distance_and_exit_confirmation():
    cfg = PortfolioConfig(min_stop_fraction=.01)
    sim = PortfolioSimulator(cfg)
    sim.quote({'KRW-BTC': 100.}, day='2026-01-01',
              signals={'KRW-BTC': {'buy': True, 'sell': False, 'atr': .01, 'high': 100., 'score': 1.}})
    assert sim.stops['KRW-BTC'] == 99.
    data = raw()
    frame = minute_features(data, PortfolioConfig(exit_confirm_bars=3))['KRW-BTC']
    original = minute_features(data, PortfolioConfig())['KRW-BTC']
    expected = original.sell.rolling(3, min_periods=3).sum().eq(3)
    pd.testing.assert_series_equal(frame.sell, expected, check_names=False)

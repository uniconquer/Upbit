import numpy as np
import pandas as pd
import pytest

from src import strategy_validation as validation


def candles():
    close = np.linspace(100, 200, 240)
    return pd.DataFrame({'open': close, 'high': close + 1, 'low': close - 1,
                         'close': close, 'volume': 1000},
                        index=pd.date_range('2024-01-01', periods=240, freq='D', tz='UTC'))


def test_selection_never_sees_or_changes_with_test_prices(monkeypatch):
    observed = []

    def builder(raw, *, strategy_name):
        observed.append((strategy_name, len(raw)))
        # Enough independent completed trades on training to clear the training gate.
        offsets = np.arange(len(raw))
        return raw.assign(buy_signal=offsets % 8 == 0, sell_signal=offsets % 8 == 6)

    monkeypatch.setattr(validation, 'build_strategy_frame', builder)
    data = candles()
    first = validation.validate_strategies({'KRW-A': data}, train_bars=180, warmup_bars=20,
                                          candidates=('first', 'second'))
    changed = data.copy()
    changed.iloc[180:, :4] *= .1
    second = validation.validate_strategies({'KRW-A': changed}, train_bars=180, warmup_bars=20,
                                           candidates=('first', 'second'))
    assert first['selected_strategy'] == second['selected_strategy'] == 'first'
    assert first['training_leaderboard'] == second['training_leaderboard']
    assert observed == [('first', 180), ('second', 180), ('first', 240)] * 2
    assert first['test'] != second['test']
    assert not first['live_enabled']


def test_cash_fallback_for_inactive_training(monkeypatch):
    monkeypatch.setattr(validation, 'build_strategy_frame',
                        lambda raw, **kw: raw.assign(buy_signal=False, sell_signal=False))
    report = validation.validate_strategies({'KRW-A': candles()}, train_bars=180,
                                           candidates=('inactive',))
    assert report['selected_strategy'] == 'cash'
    assert report['test']['return_pct'] == 0
    assert report['status'] == 'INSUFFICIENT_EVIDENCE'


def test_rejects_unaligned_or_invalid_data():
    data = candles()
    with pytest.raises(ValueError, match='Aligned'):
        validation.validate_strategies({'A': data, 'B': data.iloc[1:]}, train_bars=180)
    data.iloc[-1, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid candle'):
        validation.validate_strategies({'A': data}, train_bars=180)


@pytest.mark.parametrize('name', validation.DEFAULT_CANDIDATES)
def test_validation_candidates_have_causal_signals(name):
    raw = candles()
    # Future extreme prices must never rewrite historical strategy signals.
    raw.iloc[180:, :4] *= 5
    prefix = validation.build_strategy_frame(raw.iloc[:180], strategy_name=name)
    full = validation.build_strategy_frame(raw, strategy_name=name)
    pd.testing.assert_frame_equal(prefix[['buy_signal', 'sell_signal']],
                                  full.iloc[:180][['buy_signal', 'sell_signal']])

import copy

import pandas as pd
import pytest

from src.adaptive_portfolio import PortfolioConfig, PortfolioSimulator, build_features, replay


def signals(buy=True, atr=2., high=100., sell=False):
    return {'KRW-BTC': {'buy': buy, 'sell': sell, 'atr': atr, 'high': high, 'score': 1.}}


def test_cash_position_and_risk_caps():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    position = sim.trader.get_position('KRW-BTC')
    assert position is not None
    assert position.cost <= 40000
    assert sim.cash >= 60000
    loss = -sim.costs.simulate_exit(price=sim.stops['KRW-BTC'], qty=position.qty,
                                    cost_basis=position.cost)['pnl_value']
    assert loss <= 750 + 1e-6


def test_gap_stop_fills_at_observed_price_not_obsolete_stop():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    sim.quote({'KRW-BTC': 80.}, day='2025-01-02', signals=signals())
    sell = next(e for e in sim.events if e['side'] == 'SELL')
    assert sell['price'] < 80
    assert sell['reason'] == 'stop'
    assert not sim.trader.has_position('KRW-BTC')


def test_repeated_daily_signal_and_restart_do_not_duplicate_entry():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    restored = PortfolioSimulator.from_state(sim.to_state())
    restored.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    assert len(restored.events) == 1
    assert restored.cash == sim.cash
    assert restored.to_state()['positions'] == sim.to_state()['positions']


def test_stops_never_move_down_and_exits_work_when_entries_paused():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    stop = sim.stops['KRW-BTC']
    sim.quote({'KRW-BTC': 101.}, day='2025-01-02', signals=signals(atr=20, high=90))
    assert sim.stops['KRW-BTC'] >= stop
    sim.halted = True
    sim.quote({'KRW-BTC': 101.}, day='2025-01-03', signals=signals(sell=True))
    assert not sim.trader.has_position('KRW-BTC')


def test_minimum_order_and_invalid_quotes():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=1000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    assert not sim.events
    before = copy.deepcopy(sim.to_state())
    with pytest.raises(ValueError):
        sim.quote({'KRW-BTC': float('nan')}, day='2025-01-02', signals=signals())
    assert sim.to_state() == before


def raw_frame(n=300):
    close = pd.Series([100 + i * .5 for i in range(n)],
                      index=pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC'))
    return pd.DataFrame({'open': close, 'high': close+.1, 'low': close-.1, 'close': close, 'volume': 1000})


@pytest.mark.parametrize('mode', ['breakout', 'trend', 'pullback'])
def test_features_are_causal(mode):
    raw = raw_frame()
    cfg = PortfolioConfig(entry_mode=mode)
    prefix = build_features({'KRW-BTC': raw.iloc[:250]}, cfg)['KRW-BTC']
    raw.iloc[250:, :4] *= 10
    full = build_features({'KRW-BTC': raw}, cfg)['KRW-BTC']
    pd.testing.assert_frame_equal(prefix, full.iloc[:250])


def test_replay_matches_quote_path_when_no_intraday_stop():
    raw = raw_frame()
    cfg = PortfolioConfig()
    features = build_features({'KRW-BTC': raw}, cfg)
    result = replay({'KRW-BTC': raw}, cfg, start=220, end=230, features=features)
    assert result['buy_trades'] > 0
    sim = PortfolioSimulator(cfg, initial_cash=100000)
    for i in range(220, 230):
        day = str(raw.index[i].date())
        sig = {'KRW-BTC': features['KRW-BTC'].iloc[i-1].to_dict()}
        sim.quote({'KRW-BTC': float(raw.open.iloc[i])}, day=day, signals=sig)
        sim.mark({'KRW-BTC': float(raw.close.iloc[i])})
    assert result['final_equity'] == pytest.approx(sim.equity({'KRW-BTC': float(raw.close.iloc[229])}))


def test_current_bar_cannot_generate_its_own_entry():
    raw = raw_frame()
    features = build_features({'KRW-BTC': raw}, PortfolioConfig())
    features['KRW-BTC']['buy'] = False
    features['KRW-BTC'].iloc[220, features['KRW-BTC'].columns.get_loc('buy')] = True
    result = replay({'KRW-BTC': raw}, PortfolioConfig(), start=220, end=221, features=features)
    assert result['buy_trades'] == 0


def test_daily_gap_loss_latches_new_entry_block():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    sim.quote({'KRW-BTC': 70.}, day='2025-01-02', signals=signals())
    assert sim.daily_halted
    assert not sim.trader.positions
    restored = PortfolioSimulator.from_state(sim.to_state())
    assert restored.daily_halted
    restored.quote({'KRW-BTC': 100.}, day='2025-01-02', signals=signals())
    assert not restored.trader.positions


def test_drawdown_halt_survives_restart_and_new_day():
    sim = PortfolioSimulator(PortfolioConfig(drawdown_limit=.01), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2025-01-01', signals=signals())
    sim.quote({'KRW-BTC': 70.}, day='2025-01-02', signals=signals())
    assert sim.halted
    restored = PortfolioSimulator.from_state(sim.to_state())
    restored.quote({'KRW-BTC': 100.}, day='2025-01-03', signals=signals())
    assert restored.halted
    assert not restored.trader.positions


@pytest.mark.parametrize('low,expected_sells', [(99., 0), (90., 1)])
def test_intraday_stop_ignores_current_high_until_next_day(low, expected_sells):
    raw = raw_frame()
    raw.iloc[220] = [100., 200., low, 100., 1000.]
    features = build_features({'KRW-BTC': raw}, PortfolioConfig())
    features['KRW-BTC'].iloc[219] = [True, False, 2., 100., 1.]
    result = replay({'KRW-BTC': raw}, PortfolioConfig(), start=220, end=221, features=features)
    assert result['buy_trades'] == 1
    assert result['closed_trades'] == expected_sells

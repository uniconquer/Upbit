import pandas as pd

from src.adaptive_portfolio import PortfolioConfig, PortfolioSimulator
from src.paper_learning import new_learning_state, advance
from src.minute_learning import regular_minutes


def signals(buy=True, sell=False):
    return {'KRW-BTC': {'buy': buy, 'sell': sell, 'atr': 2., 'high': 100., 'score': 1.}}


def test_multiple_minute_signals_same_day_and_restart():
    sim = PortfolioSimulator(PortfolioConfig(), initial_cash=100000)
    sim.quote({'KRW-BTC': 100.}, day='2026-09-06', signals=signals(), signal_key='00:01')
    sim.quote({'KRW-BTC': 102.}, day='2026-09-06', signals=signals(False, True), signal_key='00:02')
    assert len(sim.events) == 2
    sim = PortfolioSimulator.from_state(sim.to_state())
    sim.quote({'KRW-BTC': 100.}, day='2026-09-06', signals=signals(), signal_key='00:02')
    assert len(sim.events) == 2
    sim.quote({'KRW-BTC': 100.}, day='2026-09-06', signals=signals(), signal_key='00:03')
    assert len(sim.events) == 3
    assert sim.day_start == 100000


def test_minute_learning_refreshes_signals_each_minute():
    state = new_learning_state([PortfolioConfig()], now='2026-09-06T00:00:00Z', timeframe='minute1')
    calls = []

    def builder(*a):
        calls.append(1)
        return {'KRW-BTC': pd.DataFrame({'buy': [True], 'sell': [False], 'atr': [2.], 'high': [100.], 'score': [1.]})}

    advance(state, {}, {'KRW-BTC': 100.}, now='2026-09-06T00:00:00Z', builder=builder)
    advance(state, {}, {'KRW-BTC': 100.}, now='2026-09-06T00:00:30Z', builder=builder)
    advance(state, {}, {'KRW-BTC': 101.}, now='2026-09-06T00:01:00Z', builder=builder)
    assert len(calls) == 2
    assert state['observations'] == 2
    assert state['coverage'] == 1


def test_missing_minute_is_flat_zero_volume_and_forming_candle_excluded():
    index = pd.to_datetime(['2026-09-06T00:00Z', '2026-09-06T00:02Z', '2026-09-06T00:03Z'])
    raw = pd.DataFrame({'open': [100., 102., 900.], 'high': [101., 103., 999.],
                        'low': [99., 101., 800.], 'close': [100., 102., 950.], 'volume': [10., 20., 999.]}, index=index)
    result = regular_minutes(raw, end=pd.Timestamp('2026-09-06T00:03Z'), count=3)
    assert list(result.close) == [100., 100., 102.]
    assert result.volume.iloc[1] == 0
    assert result.open.iloc[1] == result.high.iloc[1] == result.low.iloc[1] == 100


def test_replay_trades_more_than_once_per_day_at_bar_timestamps():
    from src.adaptive_portfolio import replay
    index = pd.date_range('2026-09-06', periods=5, freq='min', tz='UTC')
    raw = pd.DataFrame({'open': [100., 100., 103., 100., 103.],
                        'close': [100., 100., 103., 100., 103.],
                        'high': [101., 101., 104., 101., 104.],
                        'low': [99., 99., 102., 99., 102.], 'volume': 100.}, index=index)
    features = pd.DataFrame({'buy': [True, False, True, False, False],
                             'sell': [False, True, False, True, False],
                             'atr': 2., 'high': 100., 'score': 1.}, index=index)
    result = replay({'KRW-BTC': raw}, PortfolioConfig(), start=1, end=5, features={'KRW-BTC': features})
    assert result['closed_trades'] == 2
    assert [e['ts'] for e in result['state']['events']] == [x.timestamp() for x in index[1:]]

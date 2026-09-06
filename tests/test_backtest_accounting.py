import pandas as pd
import pytest

from src.strategy import backtest_signal_frame, extract_backtest_trade_events
from src.paper_trader import PaperTrader


def frame(closes, buys, sells):
    return pd.DataFrame({'close': closes, 'buy_signal': buys, 'sell_signal': sells},
                        index=pd.date_range('2025-01-01', periods=len(closes), freq='D'))


def test_open_loss_and_intratrade_drawdown_are_visible():
    data = frame([100, 50, 110], [True, False, False], [False, False, False])
    result = backtest_signal_frame(data, fee=0, slippage_bps=0)
    assert result['max_drawdown_pct'] == pytest.approx(-50)
    assert result['total_return_pct'] == pytest.approx(10)
    assert result['trades'] == 0


def test_missing_prices_preserve_timestamps_and_missing_signals_do_not_trade():
    data = frame([100, float('nan'), 50], [pd.NA, True, False], [False]*3)
    result = backtest_signal_frame(data, fee=0, slippage_bps=0)
    assert list(result['equity'].index) == list(data.index)
    assert result['total_return_pct'] == 0


def test_conflicting_signals_match_chart_and_do_not_round_trip():
    data = frame([100, 110], [True, False], [True, True])
    result = backtest_signal_frame(data, fee=0, slippage_bps=0)
    assert result['trades'] == 1
    assert result['total_return_pct'] == pytest.approx(10)
    assert [e['side'] for e in extract_backtest_trade_events(data)] == ['BUY', 'SELL']


def test_next_open_uses_prior_signal_and_actual_gap_price():
    data = frame([100, 210, 180], [True, False, True], [False, True, False])
    data['open'] = [90, 200, 180]
    result = backtest_signal_frame(data, fee=0, slippage_bps=0, execution_mode='next_open')
    assert result['total_return_pct'] == pytest.approx(-10)
    assert result['trades'] == 1
    assert extract_backtest_trade_events(data, execution_mode='next_open') == [
        {'ts': data.index[1], 'side': 'BUY', 'price': 200.0},
        {'ts': data.index[2], 'side': 'SELL', 'price': 180.0},
    ]


def test_marked_equity_matches_paper_liquidation_with_costs():
    data = frame([100, 105], [True, False], [False, False])
    trader = PaperTrader()
    trader.enter_long(market='test', price=100, cost=1, strategy='test', fee_rate=.001, slippage_bps=10)
    sold = trader.exit_long(market='test', price=105, reason='test', fee_rate=.001, slippage_bps=10)
    result = backtest_signal_frame(data, fee=.001, slippage_bps=10)
    assert result['equity'].iloc[-1] == pytest.approx(sold['net_proceeds'])


def test_next_open_requires_open_prices():
    with pytest.raises(ValueError, match='open'):
        backtest_signal_frame(frame([100], [True], [False]), execution_mode='next_open')

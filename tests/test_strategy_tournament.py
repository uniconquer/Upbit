from __future__ import annotations

import pandas as pd

from src.strategy_tournament import backtest_portfolio_signal_frames, compare_portfolio_strategies


def _portfolio_signal_frames() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2026-01-01", periods=6, freq="1h")
    market_a = pd.DataFrame(
        {
            "close": [100.0, 100.0, 110.0, 110.0, 110.0, 110.0],
            "buy_signal": [False, True, False, False, False, False],
            "sell_signal": [False, False, True, False, False, False],
            "strategy_score": [0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        },
        index=index,
    )
    market_b = pd.DataFrame(
        {
            "close": [50.0, 50.0, 50.0, 50.0, 60.0, 60.0],
            "buy_signal": [False, False, False, True, False, False],
            "sell_signal": [False, False, False, False, True, False],
            "strategy_score": [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        },
        index=index,
    )
    return {"KRW-A": market_a, "KRW-B": market_b}


def _trend_market() -> pd.DataFrame:
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 113, 111, 109, 110, 112, 115, 118, 121, 124, 126]
    frame = pd.DataFrame(
        {
            "open": [99] + closes[:-1],
            "high": [value + 1.0 for value in closes],
            "low": [value - 1.0 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 40 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-03-01", periods=len(closes), freq="1h"),
    )
    return frame


def _breakout_market() -> pd.DataFrame:
    closes = [100.0, 100.1, 99.9, 100.0, 100.1, 100.0, 99.95, 100.05, 100.0, 100.1, 100.15, 100.2, 100.1, 100.25, 101.4, 102.6, 103.8, 105.0]
    frame = pd.DataFrame(
        {
            "open": [100.0] + closes[:-1],
            "high": [value + 0.4 for value in closes],
            "low": [value - 0.4 for value in closes],
            "close": closes,
            "volume": [900 + idx * 20 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-03-10", periods=len(closes), freq="1h"),
    )
    return frame


def test_backtest_portfolio_signal_frames_uses_shared_cash_and_liquidates():
    result = backtest_portfolio_signal_frames(
        _portfolio_signal_frames(),
        strategy_name="test",
        initial_cash=10000.0,
        max_positions=1,
        fee=0.0,
        slippage_bps=0.0,
        liquidate_at_end=True,
    )

    assert result["trades"] == 2
    assert result["buy_trades"] == 2
    assert result["sell_trades"] == 2
    assert result["open_positions"] == 0
    assert float(result["final_equity"]) == 13200.0
    assert round(float(result["total_return_pct"]), 6) == 32.0


def test_compare_portfolio_strategies_returns_rows_for_new_strategies():
    raw_by_market = {
        "KRW-TREND": _trend_market(),
        "KRW-SQZ": _breakout_market(),
    }

    results = compare_portfolio_strategies(
        raw_by_market,
        strategies=[
            {"strategy_name": "ema_pullback", "params": {"fast_ema": 3, "slow_ema": 6, "rsi_window": 5, "volume_window": 4}},
            {"strategy_name": "squeeze_breakout", "params": {"bb_len": 5, "squeeze_window": 5, "breakout_window": 5, "trend_ema_window": 6, "volume_window": 4}},
        ],
        initial_cash=10000.0,
        max_positions=1,
        allocation_pct=1.0,
        fee=0.0,
        slippage_bps=0.0,
    )

    assert len(results) == 2
    assert {"strategy_name", "strategy_label", "final_equity", "return_pct", "max_drawdown_pct", "trades"}.issubset(results.columns)
    assert set(results["strategy_name"]) == {"ema_pullback", "squeeze_breakout"}

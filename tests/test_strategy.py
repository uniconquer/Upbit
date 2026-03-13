from __future__ import annotations

import pandas as pd

from src.paper_trader import PaperTrader
from src.risk_manager import ensure_daily_metrics, evaluate_entry, risk_config_from_dict
from src.strategy import (
    backtest_signal_frame,
    build_research_trend_signals,
    extract_backtest_trade_events,
    rsi_signals,
    sma_cross_signals,
)


def _sample_ohlcv() -> pd.DataFrame:
    closes = [
        100, 101, 102, 103, 104, 105, 106, 107, 109, 111,
        114, 118, 121, 124, 128, 132, 136, 140, 145, 149,
        152, 156, 161, 167, 172, 177, 181, 186, 192, 198,
        203, 209, 214, 220, 228, 235, 241, 248, 255, 263,
    ]
    frame = pd.DataFrame(
        {
            "open": [value - 1 for value in closes],
            "high": [value + 2 for value in closes],
            "low": [value - 3 for value in closes],
            "close": closes,
            "volume": [1000 + (idx * 40) for idx, _ in enumerate(closes)],
        }
    )
    frame.index = pd.date_range("2026-01-01", periods=len(frame), freq="4h")
    return frame


def test_no_signals_if_short_ge_long():
    try:
        sma_cross_signals([1, 2, 3, 4, 5, 6], short=5, long=5)
    except ValueError:
        return
    assert False, "expected ValueError when long == short"


def test_rsi_no_signals_small_series():
    prices = [1, 2, 3, 4, 5]
    signals = rsi_signals(prices, period=14)
    assert signals == []


def test_research_trend_frame_contains_expected_columns():
    frame = build_research_trend_signals(_sample_ohlcv())
    expected = {
        "ema_fast",
        "ema_slow",
        "atr",
        "adx",
        "strategy_score",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert frame.index.equals(_sample_ohlcv().index)
    assert frame["ema_fast"].notna().sum() > 0
    assert frame["ema_slow"].notna().sum() > 0


def test_backtest_signal_frame_runs_on_research_trend():
    frame = build_research_trend_signals(_sample_ohlcv())
    result = backtest_signal_frame(frame)
    assert isinstance(result["trades"], int)
    assert "total_return_pct" in result


def test_backtest_slippage_reduces_return():
    frame = build_research_trend_signals(_sample_ohlcv())
    base = backtest_signal_frame(frame, fee=0.0005, slippage_bps=0.0)
    slipped = backtest_signal_frame(frame, fee=0.0005, slippage_bps=10.0)
    assert float(slipped["total_return_pct"]) <= float(base["total_return_pct"])


def test_extract_backtest_trade_events_pairs_entries_and_exits():
    frame = pd.DataFrame(
        {
            "close": [100, 102, 104, 103, 101, 99],
            "buy_signal": [False, True, False, False, False, False],
            "sell_signal": [False, False, False, True, False, False],
        },
        index=pd.date_range("2026-01-01", periods=6, freq="1h"),
    )
    events = extract_backtest_trade_events(frame)
    assert events == [
        {"ts": frame.index[1], "side": "BUY", "price": 102.0},
        {"ts": frame.index[3], "side": "SELL", "price": 103.0},
    ]


def test_paper_trader_round_trip():
    trader = PaperTrader()
    buy_event = trader.enter_long(market="KRW-BTC", price=100.0, cost=10000.0, strategy="research_trend")
    assert buy_event["side"] == "BUY"
    assert trader.has_position("KRW-BTC")

    sell_event = trader.exit_long(market="KRW-BTC", price=110.0, reason="signal")
    assert sell_event is not None
    assert sell_event["side"] == "SELL"
    assert sell_event["pnl_value"] > 0
    assert not trader.has_position("KRW-BTC")


def test_risk_manager_blocks_daily_loss_limit():
    config = risk_config_from_dict(
        {
            "max_trade_krw": 100000,
            "daily_loss_limit_krw": 5000,
            "include_unrealized_loss": True,
        }
    )
    metrics = ensure_daily_metrics({"day_date": "2026-03-12", "realized_pnl": -2000, "daily_buy": 0}, day="2026-03-12", day_start_equity=1000000)
    positions = {
        "KRW-BTC": {"qty": 1000.0, "entry": 100.0, "cost": 100000.0},
    }
    price_map = {"KRW-BTC": 96.0}
    decision = evaluate_entry(
        config=config,
        metrics=metrics,
        positions=positions,
        price_map=price_map,
        market="KRW-ETH",
        day_start_equity=1000000,
    )
    assert not decision.allowed
    assert decision.blocked_reason is not None

from __future__ import annotations

import pandas as pd

from src.paper_trader import PaperTrader
from src.risk_manager import ensure_daily_metrics, evaluate_entry, risk_config_from_dict
from src.strategy import (
    backtest_signal_frame,
    build_ema_pullback_signals,
    build_regime_blend_signals,
    build_regime_blend_guard_signals,
    build_research_trend_signals,
    build_relative_strength_rotation_signals,
    build_relative_strength_guard_signals,
    build_rsi_bb_double_bottom_signals,
    build_rsi_trend_guard_signals,
    build_squeeze_breakout_signals,
    build_volatility_reset_breakout_signals,
    extract_backtest_trade_events,
    parameter_grid_size,
    rsi_signals,
    sma_cross_signals,
    sweep_research_trend_parameters,
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


def _double_bottom_ohlcv() -> pd.DataFrame:
    closes = [
        120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        109, 107, 105, 103, 101, 99, 97, 95, 93, 91,
        89, 87, 86, 88, 92, 95, 93, 91, 90, 92,
        95, 98, 101, 105, 109, 113, 117, 121, 125, 129,
    ]
    opens = [121] + closes[:-1]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 1 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 1 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1000 + (idx * 25) for idx, _ in enumerate(closes)],
        }
    )
    frame.index = pd.date_range("2026-02-01", periods=len(frame), freq="1h")
    return frame


def _pullback_trend_ohlcv() -> pd.DataFrame:
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 113, 111, 109, 110, 112, 115, 118, 121, 124, 126]
    opens = [99] + closes[:-1]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 1.2 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 1.2 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [1000, 1020, 1040, 1060, 1080, 1100, 1120, 1140, 1180, 1220, 1260, 1400, 1520, 1660, 1800, 1940, 2080, 2200],
        }
    )
    frame.index = pd.date_range("2026-03-01", periods=len(frame), freq="1h")
    return frame


def _squeeze_breakout_ohlcv() -> pd.DataFrame:
    closes = [100.0, 100.1, 99.9, 100.0, 100.1, 100.0, 99.95, 100.05, 100.0, 100.1, 100.15, 100.2, 100.1, 100.25, 101.4, 102.6, 103.8, 105.0, 106.2, 107.0]
    opens = [100.0] + closes[:-1]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 0.4 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 0.4 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [900, 880, 910, 905, 915, 920, 910, 925, 930, 940, 950, 960, 955, 970, 1600, 1850, 2100, 2200, 2300, 2400],
        }
    )
    frame.index = pd.date_range("2026-03-10", periods=len(frame), freq="1h")
    return frame


def _volatility_reset_breakout_ohlcv() -> pd.DataFrame:
    closes = [
        100, 101, 102, 103, 104, 105, 106, 107, 109, 111,
        113, 116, 118, 120, 123, 125, 127, 129, 128, 126,
        131, 124, 135, 127, 136, 130, 137, 141, 145, 150,
    ]
    opens = [99] + closes[:-1]
    highs: list[float] = []
    lows: list[float] = []
    for idx, (open_, close) in enumerate(zip(opens, closes, strict=False)):
        if 18 <= idx <= 23:
            highs.append(max(open_, close) + 8.0)
            lows.append(min(open_, close) - 8.0)
        elif 24 <= idx <= 26:
            highs.append(max(open_, close) + 2.0)
            lows.append(min(open_, close) - 2.0)
        else:
            highs.append(max(open_, close) + 1.1)
            lows.append(min(open_, close) - 1.1)
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [900 + (idx * 30) for idx, _ in enumerate(closes)],
        }
    )
    frame.loc[22:, "volume"] = [1800, 2100, 2400, 2500, 2600, 2800, 3000, 3200]
    frame.index = pd.date_range("2026-04-10", periods=len(frame), freq="1h")
    return frame


def _rotation_rebound_ohlcv() -> pd.DataFrame:
    closes = [
        220, 218, 216, 214, 212, 210, 208, 206, 204, 202,
        200, 198, 196, 194, 192, 190, 188, 186, 184, 182,
        180, 178, 176, 174, 172, 170, 168, 166, 164, 162,
        160, 162, 164, 166, 168, 170, 172, 174, 176, 178,
        181, 184, 187, 191, 195, 200, 206, 212, 218, 224,
    ]
    opens = [221] + closes[:-1]
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": [max(o, c) + 1.2 for o, c in zip(opens, closes)],
            "low": [min(o, c) - 1.2 for o, c in zip(opens, closes)],
            "close": closes,
            "volume": [900 + idx * 25 for idx, _ in enumerate(closes)],
        }
    )
    frame.index = pd.date_range("2026-04-01", periods=len(frame), freq="1h")
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


def test_relative_strength_rotation_frame_contains_expected_columns():
    frame = build_relative_strength_rotation_signals(
        _sample_ohlcv(),
        rs_short_window=4,
        rs_mid_window=8,
        rs_long_window=12,
        trend_ema_window=10,
        breakout_window=6,
        atr_window=5,
        volume_window=6,
        entry_score=1.0,
        exit_score=-1.0,
    )
    expected = {
        "rs_short",
        "rs_mid",
        "rs_long",
        "trend_ema",
        "atr_stop",
        "strategy_score",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool


def test_relative_strength_guard_adds_regime_columns_and_blocks_bear_entries():
    base = build_relative_strength_rotation_signals(
        _rotation_rebound_ohlcv(),
        rs_short_window=2,
        rs_mid_window=4,
        rs_long_window=8,
        trend_ema_window=6,
        breakout_window=3,
        atr_window=4,
        volume_window=4,
        volume_threshold=0.8,
        entry_score=0.1,
        exit_score=-2.0,
    )
    guard = build_relative_strength_guard_signals(
        _rotation_rebound_ohlcv(),
        rs_short_window=2,
        rs_mid_window=4,
        rs_long_window=8,
        trend_ema_window=6,
        breakout_window=3,
        atr_window=4,
        volume_window=4,
        volume_threshold=0.8,
        entry_score=0.1,
        exit_score=-2.0,
        guard_fast_ema=8,
        guard_slow_ema=21,
        guard_buffer_pct=0.5,
        guard_adx_window=4,
        guard_adx_floor=8.0,
        guard_rs_floor=0.0,
    )

    expected = {
        "guard_fast_ema",
        "guard_slow_ema",
        "guard_adx",
        "guard_slow_slope",
        "bearish_regime",
        "risk_on_regime",
        "base_buy_signal",
        "base_sell_signal",
    }
    assert expected.issubset(guard.columns)
    assert int(base["buy_signal"].sum()) >= 1
    assert int(guard["buy_signal"].sum()) == 0
    assert guard["bearish_regime"].any()


def test_ema_pullback_frame_contains_expected_columns_and_signal():
    frame = build_ema_pullback_signals(
        _pullback_trend_ohlcv(),
        fast_ema=3,
        slow_ema=6,
        rsi_window=5,
        rsi_floor=40.0,
        rsi_ceiling=75.0,
        pullback_tolerance_pct=1.5,
        volume_window=4,
        volume_threshold=0.95,
        exit_rsi=82.0,
    )

    expected = {
        "ema_fast",
        "ema_slow",
        "rsi",
        "atr",
        "pullback_band",
        "atr_stop",
        "strategy_score",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert int(frame["buy_signal"].sum()) >= 1


def test_rsi_bb_double_bottom_frame_contains_expected_columns_and_cycle():
    frame = build_rsi_bb_double_bottom_signals(
        _double_bottom_ohlcv(),
        oversold=35.0,
        bb_mult=1.5,
        max_setup_bars=15,
        confirm_bars=8,
        use_macd_filter=False,
    )
    expected = {
        "rsi",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "macd_line",
        "macd_signal",
        "rebound_marker",
        "second_bottom_marker",
        "trade_stop",
        "take_profit",
        "strategy_score",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert int(frame["buy_signal"].sum()) == 1
    assert int(frame["sell_signal"].sum()) == 1
    buy_index = frame.index[frame["buy_signal"]][0]
    sell_index = frame.index[frame["sell_signal"]][0]
    assert buy_index < sell_index
    assert frame.loc[buy_index, "trade_stop"] < frame.loc[buy_index, "close"]
    assert frame.loc[buy_index, "take_profit"] > frame.loc[buy_index, "close"]


def test_squeeze_breakout_frame_contains_expected_columns_and_signal():
    frame = build_squeeze_breakout_signals(
        _squeeze_breakout_ohlcv(),
        bb_len=5,
        squeeze_window=5,
        breakout_window=5,
        trend_ema_window=6,
        atr_window=4,
        atr_mult=1.8,
        volume_window=4,
        volume_threshold=1.05,
        squeeze_quantile=0.5,
    )

    expected = {
        "trend_ema",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "bandwidth",
        "breakout_high",
        "squeeze_on",
        "atr_stop",
        "strategy_score",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert int(frame["buy_signal"].sum()) >= 1


def test_volatility_reset_breakout_frame_contains_expected_columns_and_signal():
    frame = build_volatility_reset_breakout_signals(
        _volatility_reset_breakout_ohlcv(),
        fast_ema=3,
        slow_ema=6,
        bb_len=5,
        breakout_window=4,
        reset_window=3,
        atr_window=3,
        volume_window=4,
        volume_threshold=0.9,
        spike_window=5,
        spike_quantile=0.6,
    )

    expected = {
        "ema_fast",
        "ema_slow",
        "atr",
        "bb_basis",
        "bb_upper",
        "bb_lower",
        "bandwidth",
        "atr_ratio",
        "spike_recent",
        "cooling_recent",
        "reclaim_high",
        "strategy_score",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert int(frame["buy_signal"].sum()) >= 1


def test_regime_blend_frame_contains_expected_columns_and_both_modes():
    frame = build_regime_blend_signals(
        _double_bottom_ohlcv(),
        trend_fast_ema=4,
        trend_slow_ema=8,
        trend_breakout_window=6,
        trend_exit_window=4,
        trend_atr_window=5,
        trend_atr_mult=2.0,
        trend_adx_window=5,
        trend_adx_threshold=10.0,
        trend_momentum_window=4,
        trend_volume_window=4,
        trend_volume_threshold=0.8,
        rsi_len=8,
        oversold=38.0,
        bb_len=10,
        bb_mult=1.4,
        min_down_bars=2,
        low_tolerance_pct=1.5,
        max_setup_bars=12,
        confirm_bars=8,
        use_macd_filter=False,
        risk_reward=1.2,
        regime_adx_floor=8.0,
    )

    expected = {
        "trend_regime",
        "trend_score",
        "range_score",
        "strategy_score",
        "ema_fast",
        "ema_slow",
        "adx",
        "rsi",
        "bb_lower",
        "trend_buy_signal",
        "range_buy_signal",
        "entry_mode",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert frame["trend_regime"].dtype == bool
    assert frame["trend_regime"].any()
    assert (~frame["trend_regime"]).any()
    assert int(frame["buy_signal"].sum()) >= 1


def test_regime_blend_guard_adds_bearish_filter_columns():
    frame = build_regime_blend_guard_signals(
        _double_bottom_ohlcv(),
        trend_fast_ema=4,
        trend_slow_ema=8,
        trend_breakout_window=6,
        trend_exit_window=4,
        trend_atr_window=5,
        trend_atr_mult=2.0,
        trend_adx_window=5,
        trend_adx_threshold=10.0,
        trend_momentum_window=4,
        trend_volume_window=4,
        trend_volume_threshold=0.8,
        rsi_len=8,
        oversold=38.0,
        bb_len=10,
        bb_mult=1.4,
        min_down_bars=2,
        low_tolerance_pct=1.5,
        max_setup_bars=12,
        confirm_bars=8,
        use_macd_filter=False,
        risk_reward=1.2,
        regime_adx_floor=8.0,
        bear_guard_buffer_pct=0.5,
        bear_guard_adx_floor=8.0,
        bear_guard_score_floor=0.0,
    )

    expected = {
        "bearish_regime",
        "risk_on_regime",
        "bear_guard_slow_slope",
        "base_buy_signal",
        "base_sell_signal",
        "base_entry_mode",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert frame["risk_on_regime"].dtype == bool


def test_rsi_trend_guard_adds_bearish_filter_columns():
    frame = build_rsi_trend_guard_signals(
        _double_bottom_ohlcv(),
        rsi_len=8,
        oversold=38.0,
        bb_len=10,
        bb_mult=1.4,
        max_setup_bars=12,
        confirm_bars=8,
        use_macd_filter=False,
        trend_fast_ema=4,
        trend_slow_ema=12,
        trend_buffer_pct=2.0,
        bearish_adx_floor=12.0,
        adx_window=5,
    )

    expected = {
        "ema_fast",
        "ema_slow",
        "adx",
        "bearish_regime",
        "trend_filter",
        "base_buy_signal",
        "base_sell_signal",
        "buy_signal",
        "sell_signal",
    }
    assert expected.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool
    assert frame["trend_filter"].dtype == bool
    assert frame["buy_signal"].sum() <= frame["base_buy_signal"].sum()


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


def test_parameter_grid_size_multiplies_candidates():
    assert parameter_grid_size({"fast_ema": [12, 21, 34], "slow_ema": [55, 89]}) == 6


def test_sweep_research_trend_parameters_returns_ranked_rows():
    results = sweep_research_trend_parameters(
        _sample_ohlcv(),
        base_params={"exit_window": 10, "atr_window": 14, "momentum_window": 20, "volume_window": 20, "volume_threshold": 0.9},
        candidate_grid={
            "fast_ema": [12, 21],
            "slow_ema": [34, 55],
            "breakout_window": [10, 20],
            "atr_mult": [2.0],
            "adx_threshold": [16.0],
        },
    )

    assert len(results) == 8
    assert {"fast_ema", "slow_ema", "breakout_window", "total_return_pct", "max_drawdown_pct"}.issubset(results.columns)
    assert results["total_return_pct"].iloc[0] >= results["total_return_pct"].iloc[-1]


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

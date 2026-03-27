from __future__ import annotations

import pandas as pd

from src.strategy_engine import build_strategy_frame, compare_strategy_backtests, sweep_strategy_parameters


def _sample_ohlcv() -> pd.DataFrame:
    closes = [100, 101, 102, 104, 103, 101, 99, 100, 102, 105]
    frame = pd.DataFrame(
        {
            "open": [value - 1 for value in closes],
            "high": [value + 2 for value in closes],
            "low": [value - 2 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 10 for idx, _ in enumerate(closes)],
        }
    )
    frame.index = pd.date_range("2026-01-01", periods=len(frame), freq="1h")
    return frame


def _fake_flux_indicator(
    raw: pd.DataFrame,
    *,
    ltf_mult: float = 2.0,
    ltf_length: int = 20,
    htf_mult: float = 2.25,
    htf_length: int = 20,
    htf_rule: str = "60T",
) -> pd.DataFrame:
    close = raw["close"].astype(float)
    base = close.rolling(2, min_periods=1).mean()
    buy_signal = close > base.shift(1).fillna(close.iloc[0] - 1)
    sell_signal = close < base.shift(1).fillna(close.iloc[0] + 1)
    return pd.DataFrame(
        {
            "ltf_basis": base,
            "ltf_upper": base + ltf_mult,
            "ltf_lower": base - ltf_mult,
            "htf_upper": base + htf_mult,
            "htf_lower": base - htf_mult,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
        },
        index=raw.index,
    )


def _fake_flux_indicator_with_ema(
    raw: pd.DataFrame,
    *,
    ltf_mult: float = 2.0,
    ltf_length: int = 20,
    htf_mult: float = 2.25,
    htf_length: int = 20,
    htf_rule: str = "60T",
    sensitivity: int = 3,
    atr_period: int = 2,
    trend_ema_length: int = 240,
    confirm_window: int = 8,
    use_heikin_ashi: bool = False,
) -> pd.DataFrame:
    frame = _fake_flux_indicator(
        raw,
        ltf_mult=ltf_mult,
        ltf_length=ltf_length,
        htf_mult=htf_mult,
        htf_length=htf_length,
        htf_rule=htf_rule,
    ).copy()
    ema_buy = pd.Series([False, True, False, True, False, False, False, True, False, False], index=raw.index)
    ema_sell = pd.Series([False, False, False, False, True, False, True, False, False, True], index=raw.index)
    frame["ema_buy"] = ema_buy
    frame["ema_sell"] = ema_sell
    frame["atr_stop"] = raw["close"] - sensitivity
    frame["trend_ema"] = raw["close"].rolling(2, min_periods=1).mean()
    frame["strength"] = float(sensitivity) + float(atr_period) + float(confirm_window) / 10.0 + (0.5 if use_heikin_ashi else 0.0)
    frame["combo_buy"] = frame["buy_signal"] & ema_buy
    frame["combo_sell"] = frame["sell_signal"] & ema_sell
    return frame


def test_build_strategy_frame_flux_uses_indicator():
    frame = build_strategy_frame(
        _sample_ohlcv(),
        strategy_name="flux_trend",
        params={"ltf_len": 14, "ltf_mult": 1.5, "htf_len": 20, "htf_mult": 2.0, "htf_rule": "120T"},
        flux_indicator=_fake_flux_indicator,
    )

    assert {"ltf_basis", "ltf_upper", "htf_upper", "buy_signal", "sell_signal"}.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool


def test_build_strategy_frame_relative_strength_rotation_uses_shared_builder():
    frame = build_strategy_frame(
        _sample_ohlcv(),
        strategy_name="relative_strength_rotation",
        params={
            "rs_short_window": 2,
            "rs_mid_window": 3,
            "rs_long_window": 5,
            "trend_ema_window": 4,
            "breakout_window": 3,
            "atr_window": 3,
            "atr_mult": 1.8,
            "volume_window": 3,
            "entry_score": 0.5,
            "exit_score": -0.5,
        },
    )

    assert {"rs_short", "rs_mid", "rs_long", "trend_ema", "atr_stop", "strategy_score"}.issubset(frame.columns)
    assert frame["buy_signal"].dtype == bool
    assert frame["sell_signal"].dtype == bool


def test_build_strategy_frame_ema_pullback_uses_shared_builder():
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 113, 111, 109, 110, 112, 115, 118, 121]
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

    result = build_strategy_frame(
        frame,
        strategy_name="ema_pullback",
        params={
            "fast_ema": 3,
            "slow_ema": 6,
            "rsi_window": 5,
            "rsi_floor": 40.0,
            "rsi_ceiling": 68.0,
            "pullback_tolerance_pct": 1.5,
            "volume_window": 4,
        },
    )

    assert {"ema_fast", "ema_slow", "rsi", "pullback_band", "atr_stop", "strategy_score"}.issubset(result.columns)
    assert result["buy_signal"].dtype == bool
    assert result["sell_signal"].dtype == bool


def test_build_strategy_frame_rsi_bb_double_bottom_uses_shared_builder():
    closes = [
        120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        109, 107, 105, 103, 101, 99, 97, 95, 93, 91,
        89, 87, 86, 88, 92, 95, 93, 91, 90, 92,
        95, 98, 101, 105, 109, 113, 117, 121, 125, 129,
    ]
    frame = pd.DataFrame(
        {
            "open": [121] + closes[:-1],
            "high": [value + 1.5 for value in closes],
            "low": [value - 1.5 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 20 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-02-01", periods=len(closes), freq="1h"),
    )

    result = build_strategy_frame(
        frame,
        strategy_name="rsi_bb_double_bottom",
        params={
            "oversold": 35.0,
            "bb_mult": 1.5,
            "max_setup_bars": 15,
            "confirm_bars": 8,
            "use_macd_filter": False,
        },
    )

    assert {"rsi", "bb_lower", "trade_stop", "take_profit", "rebound_marker", "second_bottom_marker"}.issubset(result.columns)
    assert result["buy_signal"].dtype == bool
    assert result["sell_signal"].dtype == bool
    assert int(result["buy_signal"].sum()) >= 1


def test_build_strategy_frame_rsi_trend_guard_uses_shared_builder():
    closes = [
        120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        109, 107, 105, 103, 101, 99, 97, 95, 93, 91,
        89, 87, 86, 88, 92, 95, 93, 91, 90, 92,
        95, 98, 101, 105, 109, 113, 117, 121, 125, 129,
    ]
    frame = pd.DataFrame(
        {
            "open": [121] + closes[:-1],
            "high": [value + 1.5 for value in closes],
            "low": [value - 1.5 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 20 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-02-01", periods=len(closes), freq="1h"),
    )

    result = build_strategy_frame(
        frame,
        strategy_name="rsi_trend_guard",
        params={
            "rsi_len": 8,
            "oversold": 38.0,
            "bb_len": 10,
            "bb_mult": 1.4,
            "max_setup_bars": 12,
            "confirm_bars": 8,
            "use_macd_filter": False,
            "trend_fast_ema": 4,
            "trend_slow_ema": 12,
            "trend_buffer_pct": 2.0,
            "bearish_adx_floor": 12.0,
            "adx_window": 5,
        },
    )

    assert {"ema_fast", "ema_slow", "adx", "bearish_regime", "trend_filter", "base_buy_signal", "base_sell_signal"}.issubset(result.columns)
    assert result["buy_signal"].dtype == bool
    assert result["sell_signal"].dtype == bool


def test_build_strategy_frame_squeeze_breakout_uses_shared_builder():
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

    result = build_strategy_frame(
        frame,
        strategy_name="squeeze_breakout",
        params={
            "bb_len": 5,
            "squeeze_window": 5,
            "breakout_window": 5,
            "trend_ema_window": 6,
            "atr_window": 4,
            "volume_window": 4,
            "volume_threshold": 1.0,
        },
    )

    assert {"trend_ema", "bb_basis", "bandwidth", "breakout_high", "squeeze_on", "atr_stop"}.issubset(result.columns)
    assert result["buy_signal"].dtype == bool
    assert result["sell_signal"].dtype == bool


def test_build_strategy_frame_regime_blend_uses_shared_builder():
    closes = [
        120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        109, 107, 105, 103, 101, 99, 97, 95, 93, 91,
        89, 87, 86, 88, 92, 95, 93, 91, 90, 92,
        95, 98, 101, 105, 109, 113, 117, 121, 125, 129,
    ]
    frame = pd.DataFrame(
        {
            "open": [121] + closes[:-1],
            "high": [value + 1.5 for value in closes],
            "low": [value - 1.5 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 20 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-02-01", periods=len(closes), freq="1h"),
    )

    result = build_strategy_frame(
        frame,
        strategy_name="regime_blend",
        params={
            "trend_fast_ema": 4,
            "trend_slow_ema": 8,
            "trend_breakout_window": 6,
            "trend_exit_window": 4,
            "trend_atr_window": 5,
            "trend_atr_mult": 2.0,
            "trend_adx_window": 5,
            "trend_adx_threshold": 10.0,
            "trend_momentum_window": 4,
            "trend_volume_window": 4,
            "trend_volume_threshold": 0.8,
            "rsi_len": 8,
            "oversold": 38.0,
            "bb_len": 10,
            "bb_mult": 1.4,
            "max_setup_bars": 12,
            "confirm_bars": 8,
            "use_macd_filter": False,
            "regime_adx_floor": 8.0,
        },
    )

    assert {"trend_regime", "trend_score", "range_score", "ema_fast", "ema_slow", "adx", "rsi", "trend_buy_signal", "range_buy_signal", "entry_mode"}.issubset(result.columns)
    assert result["buy_signal"].dtype == bool
    assert result["sell_signal"].dtype == bool
    assert result["trend_regime"].dtype == bool


def test_build_strategy_frame_flux_ema_filter_uses_combo_signals():
    frame = build_strategy_frame(
        _sample_ohlcv(),
        strategy_name="flux_ema_filter",
        params={
            "ltf_len": 14,
            "ltf_mult": 1.5,
            "htf_len": 20,
            "htf_mult": 2.0,
            "htf_rule": "120T",
            "sensitivity": 4,
            "atr_period": 3,
            "trend_ema_length": 180,
            "confirm_window": 6,
            "use_heikin_ashi": True,
        },
        flux_indicator_with_ema=_fake_flux_indicator_with_ema,
    )

    assert {"combo_buy", "combo_sell", "ema_buy", "ema_sell", "trend_ema", "strength"}.issubset(frame.columns)
    assert {"flux_buy_signal", "flux_sell_signal"}.issubset(frame.columns)
    assert frame["buy_signal"].equals(frame["combo_buy"])
    assert frame["sell_signal"].equals(frame["combo_sell"])
    assert frame["strategy_score"].dtype.kind in {"f", "i"}


def test_sweep_strategy_parameters_supports_flux():
    results = sweep_strategy_parameters(
        _sample_ohlcv(),
        strategy_name="flux_trend",
        candidate_grid={
            "ltf_len": [14, 20],
            "ltf_mult": [1.5, 2.0],
            "htf_len": [20],
            "htf_mult": [2.0],
            "htf_rule": ["60T", "120T"],
        },
        flux_indicator=_fake_flux_indicator,
    )

    assert len(results) == 8
    assert {"ltf_len", "ltf_mult", "htf_rule", "total_return_pct", "buy_signals", "sell_signals"}.issubset(results.columns)


def test_sweep_strategy_parameters_supports_relative_strength_rotation():
    results = sweep_strategy_parameters(
        _sample_ohlcv(),
        strategy_name="relative_strength_rotation",
        candidate_grid={
            "rs_short_window": [2, 3],
            "rs_mid_window": [4],
            "rs_long_window": [6],
            "trend_ema_window": [4],
            "breakout_window": [3],
            "entry_score": [0.5, 1.0],
        },
    )

    assert len(results) == 4
    assert {"rs_short_window", "trend_ema_window", "entry_score", "total_return_pct"}.issubset(results.columns)


def test_sweep_strategy_parameters_supports_rsi_bb_double_bottom():
    closes = [
        120, 119, 118, 117, 116, 115, 114, 113, 112, 111,
        109, 107, 105, 103, 101, 99, 97, 95, 93, 91,
        89, 87, 86, 88, 92, 95, 93, 91, 90, 92,
        95, 98, 101, 105, 109, 113, 117, 121, 125, 129,
    ]
    frame = pd.DataFrame(
        {
            "open": [121] + closes[:-1],
            "high": [value + 1.5 for value in closes],
            "low": [value - 1.5 for value in closes],
            "close": closes,
            "volume": [1000 + idx * 20 for idx, _ in enumerate(closes)],
        },
        index=pd.date_range("2026-02-01", periods=len(closes), freq="1h"),
    )

    results = sweep_strategy_parameters(
        frame,
        strategy_name="rsi_bb_double_bottom",
        base_params={"use_macd_filter": False, "max_setup_bars": 15, "confirm_bars": 8},
        candidate_grid={
            "rsi_len": [12, 14],
            "oversold": [30.0, 35.0],
            "bb_len": [20],
            "bb_mult": [1.5],
        },
    )

    assert len(results) == 4
    assert {"rsi_len", "oversold", "bb_len", "bb_mult", "total_return_pct", "buy_signals", "sell_signals"}.issubset(results.columns)


def test_sweep_strategy_parameters_supports_flux_ema_filter():
    results = sweep_strategy_parameters(
        _sample_ohlcv(),
        strategy_name="flux_ema_filter",
        candidate_grid={
            "ltf_len": [14],
            "ltf_mult": [1.5, 2.0],
            "htf_len": [20],
            "htf_mult": [2.0],
            "htf_rule": ["60T"],
            "sensitivity": [2, 3],
            "atr_period": [2],
            "trend_ema_length": [180],
            "confirm_window": [8],
        },
        flux_indicator_with_ema=_fake_flux_indicator_with_ema,
    )

    assert len(results) == 4
    assert {"sensitivity", "atr_period", "trend_ema_length", "confirm_window", "total_return_pct"}.issubset(results.columns)


def test_compare_strategy_backtests_ranks_multiple_strategies():
    results = compare_strategy_backtests(
        _sample_ohlcv(),
        strategies=[
            {
                "strategy_name": "relative_strength_rotation",
                "params": {
                    "rs_short_window": 2,
                    "rs_mid_window": 3,
                    "rs_long_window": 5,
                    "trend_ema_window": 4,
                    "breakout_window": 3,
                    "entry_score": 0.5,
                    "exit_score": -0.5,
                },
            },
            {"strategy_name": "flux_trend", "params": {"ltf_len": 14, "ltf_mult": 1.5, "htf_len": 20, "htf_mult": 2.0, "htf_rule": "60T"}},
            {
                "strategy_name": "flux_ema_filter",
                "params": {
                    "ltf_len": 14,
                    "ltf_mult": 1.5,
                    "htf_len": 20,
                    "htf_mult": 2.0,
                    "htf_rule": "60T",
                    "sensitivity": 3,
                    "atr_period": 2,
                    "trend_ema_length": 180,
                    "confirm_window": 8,
                },
            },
        ],
        flux_indicator=_fake_flux_indicator,
        flux_indicator_with_ema=_fake_flux_indicator_with_ema,
    )

    assert len(results) == 3
    assert {"strategy_name", "strategy_label", "params", "return_pct", "max_drawdown_pct", "last_signal"}.issubset(results.columns)
    assert set(results["strategy_name"]) == {"relative_strength_rotation", "flux_trend", "flux_ema_filter"}

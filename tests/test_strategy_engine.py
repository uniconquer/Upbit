from __future__ import annotations

import pandas as pd

from src.strategy_engine import build_strategy_frame, sweep_strategy_parameters


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
    frame["strength"] = float(sensitivity) + float(atr_period) + (0.5 if use_heikin_ashi else 0.0)
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
        },
        flux_indicator_with_ema=_fake_flux_indicator_with_ema,
    )

    assert len(results) == 4
    assert {"sensitivity", "atr_period", "trend_ema_length", "total_return_pct"}.issubset(results.columns)

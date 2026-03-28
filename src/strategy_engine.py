"""Shared strategy engine used by backtest and live views."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd

try:
    from strategy import (
        backtest_signal_frame,
        build_ema_pullback_signals,
        build_regime_blend_signals,
        build_regime_blend_guard_signals,
        build_relative_strength_rotation_signals,
        build_relative_strength_guard_signals,
        build_research_trend_signals,
        build_rsi_bb_double_bottom_signals,
        build_rsi_trend_guard_signals,
        build_squeeze_breakout_signals,
        build_volatility_reset_breakout_signals,
        VOLATILITY_RESET_BREAKOUT_DEFAULTS,
    )
except ImportError:
    from src.strategy import (
        backtest_signal_frame,
        build_ema_pullback_signals,
        build_regime_blend_signals,
        build_regime_blend_guard_signals,
        build_relative_strength_rotation_signals,
        build_relative_strength_guard_signals,
        build_research_trend_signals,
        build_rsi_bb_double_bottom_signals,
        build_rsi_trend_guard_signals,
        build_squeeze_breakout_signals,
        build_volatility_reset_breakout_signals,
        VOLATILITY_RESET_BREAKOUT_DEFAULTS,
    )

FluxCallable = Callable[..., pd.DataFrame] | None

STRATEGY_LABELS = {
    "research_trend": "연구형 추세 돌파",
    "rsi_bb_double_bottom": "RSI+BB 더블바텀 롱",
    "relative_strength_rotation": "상대강도 로테이션",
    "relative_strength_guard": "상대강도 로테이션 가드",
    "ema_pullback": "EMA 눌림목 추세",
    "squeeze_breakout": "스퀴즈 돌파 추세",
    "volatility_reset_breakout": "변동성 재정비 돌파",
    "rsi_trend_guard": "RSI 반등 + 추세 가드",
    "regime_blend": "장세 적응 혼합",
    "regime_blend_guard": "장세 적응 혼합 가드",
    "flux_trend": "플럭스 추세 밴드",
    "flux_ema_filter": "플럭스 + EMA 필터",
}

STRATEGY_DESCRIPTIONS = {
    "research_trend": "EMA 정배열, 돌파, ADX 추세 강도, ATR 이탈 손절을 함께 보는 추세 전략입니다.",
    "rsi_bb_double_bottom": "RSI 과매도와 볼린저 하단 세척 이후 첫 반등과 두 번째 바닥을 확인한 뒤 진입하는 반등 전략입니다.",
    "relative_strength_rotation": "여러 기간 수익률과 추세 강도를 합친 점수로 강한 종목을 따라가는 회전형 모멘텀 전략입니다.",
    "relative_strength_guard": "상대강도 로테이션에 약세장 현금 보유 필터를 더해, 긴 하락 추세에서는 진입을 막고 빠르게 쉬는 보호형 회전 전략입니다.",
    "ema_pullback": "상승 추세 중 EMA 눌림과 RSI 재정비 뒤 반등 봉이 나올 때만 진입하는 추세-눌림목 전략입니다.",
    "squeeze_breakout": "볼린저 밴드 폭이 눌린 뒤 거래량과 함께 돌파가 나올 때만 따라가는 압축 돌파 전략입니다.",
    "volatility_reset_breakout": "급격한 변동성 확대가 지나간 뒤 재정비 구간을 거쳐 다시 돌파할 때만 진입하는 복원형 돌파 전략입니다.",
    "rsi_trend_guard": "RSI 이중 바닥 반등은 유지하되, 강한 약세 추세에서는 매수를 막고 조기 이탈하는 보호형 반등 전략입니다.",
    "regime_blend": "추세가 강하면 추세 돌파를, 그렇지 않으면 RSI 반등을 쓰는 장세 적응형 혼합 전략입니다.",
    "regime_blend_guard": "장세 적응 혼합 전략에 약세장 회피 필터를 더해, 추세도 반등도 불리한 구간에서는 현금 보유를 우선하는 보호형 혼합 전략입니다.",
    "flux_trend": "다중 시간대 밴드와 중심선 변화를 함께 보는 반전형 밴드 전략입니다.",
    "flux_ema_filter": "플럭스 신호 뒤에 EMA와 ATR 확인을 추가한 혼합형 추세-반전 전략입니다.",
}


BACKTEST_EXTRA_STRATEGY_OPTIONS = [
    "relative_strength_guard",
    "regime_blend_guard",
]


def strategy_options(
    flux_available: bool,
    flux_ema_available: bool = False,
    *,
    include_backtest_extras: bool = False,
) -> list[str]:
    options = ["research_trend", "rsi_bb_double_bottom", "rsi_trend_guard", "relative_strength_rotation"]
    if include_backtest_extras:
        options.extend(BACKTEST_EXTRA_STRATEGY_OPTIONS)
    if flux_available:
        options.append("flux_trend")
    if flux_ema_available:
        options.append("flux_ema_filter")
    return options


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def strategy_description(name: str) -> str:
    return STRATEGY_DESCRIPTIONS.get(name, name)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _finalize_indicator_frame(
    raw: pd.DataFrame,
    indicator_frame: pd.DataFrame,
    *,
    use_combo_signals: bool = False,
) -> pd.DataFrame:
    result = raw.join(indicator_frame, how="left")

    if "buy_signal" in result.columns:
        result["buy_signal"] = result["buy_signal"].fillna(False).astype(bool)
    if "sell_signal" in result.columns:
        result["sell_signal"] = result["sell_signal"].fillna(False).astype(bool)

    if use_combo_signals:
        if "buy_signal" in result.columns:
            result["flux_buy_signal"] = result["buy_signal"]
        if "sell_signal" in result.columns:
            result["flux_sell_signal"] = result["sell_signal"]
        result["buy_signal"] = result.get("combo_buy", pd.Series(False, index=result.index)).fillna(False).astype(bool)
        result["sell_signal"] = result.get("combo_sell", pd.Series(False, index=result.index)).fillna(False).astype(bool)

    if "strategy_score" not in result.columns:
        if "strength" in result.columns:
            strength = pd.to_numeric(result["strength"], errors="coerce")
            result["strategy_score"] = strength.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        else:
            result["strategy_score"] = 0.0
    return result


def build_strategy_frame(
    raw: pd.DataFrame,
    *,
    strategy_name: str,
    params: dict[str, Any] | None = None,
    flux_indicator: FluxCallable = None,
    flux_indicator_with_ema: FluxCallable = None,
) -> pd.DataFrame:
    params = params or {}

    if strategy_name == "research_trend":
        return build_research_trend_signals(
            raw,
            fast_ema=int(params.get("fast_ema", 21)),
            slow_ema=int(params.get("slow_ema", 55)),
            breakout_window=int(params.get("breakout_window", 20)),
            exit_window=int(params.get("exit_window", 10)),
            atr_window=int(params.get("atr_window", 14)),
            atr_mult=float(params.get("atr_mult", 2.5)),
            adx_window=int(params.get("adx_window", 14)),
            adx_threshold=float(params.get("adx_threshold", 18.0)),
            momentum_window=int(params.get("momentum_window", 20)),
            volume_window=int(params.get("volume_window", 20)),
            volume_threshold=float(params.get("volume_threshold", 0.9)),
        )

    if strategy_name == "rsi_bb_double_bottom":
        return build_rsi_bb_double_bottom_signals(
            raw,
            rsi_len=int(params.get("rsi_len", 14)),
            oversold=float(params.get("oversold", 30.0)),
            bb_len=int(params.get("bb_len", 20)),
            bb_mult=float(params.get("bb_mult", 2.0)),
            min_down_bars=int(params.get("min_down_bars", 2)),
            low_tolerance_pct=float(params.get("low_tolerance_pct", 1.0)),
            max_setup_bars=int(params.get("max_setup_bars", 12)),
            confirm_bars=int(params.get("confirm_bars", 4)),
            use_macd_filter=_as_bool(params.get("use_macd_filter"), True),
            macd_lookback=int(params.get("macd_lookback", 5)),
            risk_reward=float(params.get("risk_reward", 2.0)),
            stop_buffer_ticks=int(params.get("stop_buffer_ticks", 2)),
        )

    if strategy_name == "relative_strength_rotation":
        return build_relative_strength_rotation_signals(
            raw,
            rs_short_window=int(params.get("rs_short_window", 10)),
            rs_mid_window=int(params.get("rs_mid_window", 30)),
            rs_long_window=int(params.get("rs_long_window", 90)),
            trend_ema_window=int(params.get("trend_ema_window", 55)),
            breakout_window=int(params.get("breakout_window", 20)),
            atr_window=int(params.get("atr_window", 14)),
            atr_mult=float(params.get("atr_mult", 2.2)),
            volume_window=int(params.get("volume_window", 20)),
            volume_threshold=float(params.get("volume_threshold", 0.9)),
            entry_score=float(params.get("entry_score", 8.0)),
            exit_score=float(params.get("exit_score", 2.0)),
        )

    if strategy_name == "relative_strength_guard":
        return build_relative_strength_guard_signals(
            raw,
            rs_short_window=int(params.get("rs_short_window", 10)),
            rs_mid_window=int(params.get("rs_mid_window", 30)),
            rs_long_window=int(params.get("rs_long_window", 90)),
            trend_ema_window=int(params.get("trend_ema_window", 55)),
            breakout_window=int(params.get("breakout_window", 28)),
            atr_window=int(params.get("atr_window", 14)),
            atr_mult=float(params.get("atr_mult", 2.2)),
            volume_window=int(params.get("volume_window", 20)),
            volume_threshold=float(params.get("volume_threshold", 0.9)),
            entry_score=float(params.get("entry_score", 9.0)),
            exit_score=float(params.get("exit_score", 3.0)),
            guard_fast_ema=int(params.get("guard_fast_ema", 13)),
            guard_slow_ema=int(params.get("guard_slow_ema", 144)),
            guard_buffer_pct=float(params.get("guard_buffer_pct", 1.0)),
            guard_adx_window=int(params.get("guard_adx_window", 14)),
            guard_adx_floor=float(params.get("guard_adx_floor", 10.0)),
            guard_rs_floor=float(params.get("guard_rs_floor", -3.0)),
        )

    if strategy_name == "ema_pullback":
        return build_ema_pullback_signals(
            raw,
            fast_ema=int(params.get("fast_ema", 21)),
            slow_ema=int(params.get("slow_ema", 55)),
            rsi_window=int(params.get("rsi_window", 14)),
            rsi_floor=float(params.get("rsi_floor", 42.0)),
            rsi_ceiling=float(params.get("rsi_ceiling", 62.0)),
            pullback_tolerance_pct=float(params.get("pullback_tolerance_pct", 0.6)),
            atr_window=int(params.get("atr_window", 14)),
            atr_mult=float(params.get("atr_mult", 2.0)),
            volume_window=int(params.get("volume_window", 20)),
            volume_threshold=float(params.get("volume_threshold", 0.9)),
            exit_rsi=float(params.get("exit_rsi", 68.0)),
        )

    if strategy_name == "rsi_trend_guard":
        return build_rsi_trend_guard_signals(
            raw,
            rsi_len=int(params.get("rsi_len", 10)),
            oversold=float(params.get("oversold", 35.0)),
            bb_len=int(params.get("bb_len", 20)),
            bb_mult=float(params.get("bb_mult", 1.5)),
            min_down_bars=int(params.get("min_down_bars", 2)),
            low_tolerance_pct=float(params.get("low_tolerance_pct", 1.0)),
            max_setup_bars=int(params.get("max_setup_bars", 6)),
            confirm_bars=int(params.get("confirm_bars", 3)),
            use_macd_filter=_as_bool(params.get("use_macd_filter"), True),
            macd_lookback=int(params.get("macd_lookback", 5)),
            risk_reward=float(params.get("risk_reward", 1.5)),
            stop_buffer_ticks=int(params.get("stop_buffer_ticks", 2)),
            trend_fast_ema=int(params.get("trend_fast_ema", 13)),
            trend_slow_ema=int(params.get("trend_slow_ema", 89)),
            trend_buffer_pct=float(params.get("trend_buffer_pct", 2.0)),
            bearish_adx_floor=float(params.get("bearish_adx_floor", 14.0)),
            adx_window=int(params.get("adx_window", 14)),
        )

    if strategy_name == "squeeze_breakout":
        return build_squeeze_breakout_signals(
            raw,
            bb_len=int(params.get("bb_len", 20)),
            bb_mult=float(params.get("bb_mult", 2.0)),
            squeeze_window=int(params.get("squeeze_window", 20)),
            breakout_window=int(params.get("breakout_window", 20)),
            trend_ema_window=int(params.get("trend_ema_window", 55)),
            atr_window=int(params.get("atr_window", 14)),
            atr_mult=float(params.get("atr_mult", 2.0)),
            volume_window=int(params.get("volume_window", 20)),
            volume_threshold=float(params.get("volume_threshold", 1.1)),
            squeeze_quantile=float(params.get("squeeze_quantile", 0.35)),
        )

    if strategy_name == "volatility_reset_breakout":
        return build_volatility_reset_breakout_signals(
            raw,
            fast_ema=int(params.get("fast_ema", VOLATILITY_RESET_BREAKOUT_DEFAULTS["fast_ema"])),
            slow_ema=int(params.get("slow_ema", VOLATILITY_RESET_BREAKOUT_DEFAULTS["slow_ema"])),
            bb_len=int(params.get("bb_len", VOLATILITY_RESET_BREAKOUT_DEFAULTS["bb_len"])),
            bb_mult=float(params.get("bb_mult", VOLATILITY_RESET_BREAKOUT_DEFAULTS["bb_mult"])),
            breakout_window=int(params.get("breakout_window", VOLATILITY_RESET_BREAKOUT_DEFAULTS["breakout_window"])),
            reset_window=int(params.get("reset_window", VOLATILITY_RESET_BREAKOUT_DEFAULTS["reset_window"])),
            atr_window=int(params.get("atr_window", VOLATILITY_RESET_BREAKOUT_DEFAULTS["atr_window"])),
            atr_mult=float(params.get("atr_mult", VOLATILITY_RESET_BREAKOUT_DEFAULTS["atr_mult"])),
            volume_window=int(params.get("volume_window", VOLATILITY_RESET_BREAKOUT_DEFAULTS["volume_window"])),
            volume_threshold=float(params.get("volume_threshold", VOLATILITY_RESET_BREAKOUT_DEFAULTS["volume_threshold"])),
            spike_window=int(params.get("spike_window", VOLATILITY_RESET_BREAKOUT_DEFAULTS["spike_window"])),
            spike_quantile=float(params.get("spike_quantile", VOLATILITY_RESET_BREAKOUT_DEFAULTS["spike_quantile"])),
        )

    if strategy_name == "regime_blend":
        return build_regime_blend_signals(
            raw,
            trend_fast_ema=int(params.get("trend_fast_ema", 21)),
            trend_slow_ema=int(params.get("trend_slow_ema", 55)),
            trend_breakout_window=int(params.get("trend_breakout_window", 20)),
            trend_exit_window=int(params.get("trend_exit_window", 10)),
            trend_atr_window=int(params.get("trend_atr_window", 14)),
            trend_atr_mult=float(params.get("trend_atr_mult", 2.5)),
            trend_adx_window=int(params.get("trend_adx_window", 14)),
            trend_adx_threshold=float(params.get("trend_adx_threshold", 18.0)),
            trend_momentum_window=int(params.get("trend_momentum_window", 20)),
            trend_volume_window=int(params.get("trend_volume_window", 20)),
            trend_volume_threshold=float(params.get("trend_volume_threshold", 0.9)),
            rsi_len=int(params.get("rsi_len", 10)),
            oversold=float(params.get("oversold", 35.0)),
            bb_len=int(params.get("bb_len", 20)),
            bb_mult=float(params.get("bb_mult", 2.0)),
            min_down_bars=int(params.get("min_down_bars", 2)),
            low_tolerance_pct=float(params.get("low_tolerance_pct", 1.0)),
            max_setup_bars=int(params.get("max_setup_bars", 12)),
            confirm_bars=int(params.get("confirm_bars", 5)),
            use_macd_filter=_as_bool(params.get("use_macd_filter"), True),
            macd_lookback=int(params.get("macd_lookback", 5)),
            risk_reward=float(params.get("risk_reward", 1.5)),
            stop_buffer_ticks=int(params.get("stop_buffer_ticks", 2)),
            regime_adx_floor=float(params.get("regime_adx_floor", 16.0)),
        )

    if strategy_name == "regime_blend_guard":
        return build_regime_blend_guard_signals(
            raw,
            trend_fast_ema=int(params.get("trend_fast_ema", 21)),
            trend_slow_ema=int(params.get("trend_slow_ema", 55)),
            trend_breakout_window=int(params.get("trend_breakout_window", 20)),
            trend_exit_window=int(params.get("trend_exit_window", 10)),
            trend_atr_window=int(params.get("trend_atr_window", 14)),
            trend_atr_mult=float(params.get("trend_atr_mult", 2.5)),
            trend_adx_window=int(params.get("trend_adx_window", 14)),
            trend_adx_threshold=float(params.get("trend_adx_threshold", 18.0)),
            trend_momentum_window=int(params.get("trend_momentum_window", 20)),
            trend_volume_window=int(params.get("trend_volume_window", 20)),
            trend_volume_threshold=float(params.get("trend_volume_threshold", 0.9)),
            rsi_len=int(params.get("rsi_len", 10)),
            oversold=float(params.get("oversold", 35.0)),
            bb_len=int(params.get("bb_len", 20)),
            bb_mult=float(params.get("bb_mult", 2.0)),
            min_down_bars=int(params.get("min_down_bars", 2)),
            low_tolerance_pct=float(params.get("low_tolerance_pct", 1.0)),
            max_setup_bars=int(params.get("max_setup_bars", 12)),
            confirm_bars=int(params.get("confirm_bars", 5)),
            use_macd_filter=_as_bool(params.get("use_macd_filter"), True),
            macd_lookback=int(params.get("macd_lookback", 5)),
            risk_reward=float(params.get("risk_reward", 1.5)),
            stop_buffer_ticks=int(params.get("stop_buffer_ticks", 2)),
            regime_adx_floor=float(params.get("regime_adx_floor", 16.0)),
            bear_guard_buffer_pct=float(params.get("bear_guard_buffer_pct", 1.5)),
            bear_guard_adx_floor=float(params.get("bear_guard_adx_floor", 14.0)),
            bear_guard_score_floor=float(params.get("bear_guard_score_floor", -2.0)),
        )

    if strategy_name == "flux_trend":
        if flux_indicator is None:
            raise RuntimeError("Flux strategy selected but flux indicator is unavailable")
        indicator_frame = flux_indicator(
            raw,
            ltf_mult=float(params.get("ltf_mult", 2.0)),
            ltf_length=int(params.get("ltf_len", 20)),
            htf_mult=float(params.get("htf_mult", 2.25)),
            htf_length=int(params.get("htf_len", 20)),
            htf_rule=str(params.get("htf_rule", "30T")),
        )
        return _finalize_indicator_frame(raw, indicator_frame)

    if strategy_name == "flux_ema_filter":
        if flux_indicator_with_ema is None:
            raise RuntimeError("Flux EMA strategy selected but the extended flux indicator is unavailable")
        indicator_frame = flux_indicator_with_ema(
            raw,
            ltf_mult=float(params.get("ltf_mult", 2.0)),
            ltf_length=int(params.get("ltf_len", 20)),
            htf_mult=float(params.get("htf_mult", 2.25)),
            htf_length=int(params.get("htf_len", 20)),
            htf_rule=str(params.get("htf_rule", "30T")),
            sensitivity=int(params.get("sensitivity", 3)),
            atr_period=int(params.get("atr_period", 2)),
            trend_ema_length=int(params.get("trend_ema_length", 240)),
            confirm_window=int(params.get("confirm_window", 8)),
            use_heikin_ashi=_as_bool(params.get("use_heikin_ashi"), False),
        )
        return _finalize_indicator_frame(raw, indicator_frame, use_combo_signals=True)

    raise ValueError(f"unknown strategy: {strategy_name}")


def sweep_strategy_parameters(
    raw: pd.DataFrame,
    *,
    strategy_name: str,
    base_params: dict[str, Any] | None = None,
    candidate_grid: dict[str, list[Any]] | None = None,
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
    flux_indicator: FluxCallable = None,
    flux_indicator_with_ema: FluxCallable = None,
) -> pd.DataFrame:
    base = dict(base_params or {})
    grid = {key: list(values) for key, values in dict(candidate_grid or {}).items() if list(values)}
    if not grid:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    keys = list(grid.keys())
    for combo in product(*(grid[key] for key in keys)):
        combo_params = dict(zip(keys, combo, strict=False))
        merged = {**base, **combo_params}
        frame = build_strategy_frame(
            raw,
            strategy_name=strategy_name,
            params=merged,
            flux_indicator=flux_indicator,
            flux_indicator_with_ema=flux_indicator_with_ema,
        )
        bt = backtest_signal_frame(frame, fee=fee, slippage_bps=slippage_bps)
        rows.append(
            {
                **combo_params,
                "trades": int(bt["trades"]),
                "buy_signals": int(frame["buy_signal"].sum()) if "buy_signal" in frame else 0,
                "sell_signals": int(frame["sell_signal"].sum()) if "sell_signal" in frame else 0,
                "total_return_pct": float(bt["total_return_pct"]),
                "win_rate_pct": float(bt["win_rate_pct"]),
                "max_drawdown_pct": float(bt["max_drawdown_pct"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    results = pd.DataFrame(rows)
    return results.sort_values(
        ["total_return_pct", "max_drawdown_pct", "win_rate_pct", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _last_signal_code(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "WAIT"
    last_row = frame.iloc[-1]
    if bool(last_row.get("buy_signal")):
        return "BUY"
    if bool(last_row.get("sell_signal")):
        return "SELL"
    return "WAIT"


def compare_strategy_backtests(
    raw: pd.DataFrame,
    *,
    strategies: list[dict[str, Any]],
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
    flux_indicator: FluxCallable = None,
    flux_indicator_with_ema: FluxCallable = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in strategies:
        strategy_name = str(spec.get("strategy_name") or "")
        if not strategy_name:
            continue
        params = dict(spec.get("params") or {})
        frame = build_strategy_frame(
            raw,
            strategy_name=strategy_name,
            params=params,
            flux_indicator=flux_indicator,
            flux_indicator_with_ema=flux_indicator_with_ema,
        )
        bt = backtest_signal_frame(frame, fee=fee, slippage_bps=slippage_bps)
        last_row = frame.iloc[-1] if not frame.empty else pd.Series(dtype=object)
        rows.append(
            {
                "strategy_name": strategy_name,
                "strategy_label": strategy_label(strategy_name),
                "params": params,
                "score": float(last_row.get("strategy_score", 0.0)),
                "price": float(last_row.get("close", 0.0)),
                "trades": int(bt["trades"]),
                "buy_signals": int(frame["buy_signal"].sum()) if "buy_signal" in frame else 0,
                "sell_signals": int(frame["sell_signal"].sum()) if "sell_signal" in frame else 0,
                "return_pct": float(bt["total_return_pct"]),
                "win_rate_pct": float(bt["win_rate_pct"]),
                "max_drawdown_pct": float(bt["max_drawdown_pct"]),
                "last_signal": _last_signal_code(frame),
            }
        )
    if not rows:
        return pd.DataFrame()
    results = pd.DataFrame(rows)
    return results.sort_values(
        ["return_pct", "max_drawdown_pct", "win_rate_pct", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

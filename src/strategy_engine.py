"""Shared strategy engine used by backtest and live views."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

try:
    from strategy import build_research_trend_signals
except ImportError:
    from src.strategy import build_research_trend_signals

FluxCallable = Callable[..., pd.DataFrame] | None

STRATEGY_LABELS = {
    "research_trend": "연구형 추세 돌파",
    "flux_trend": "플럭스 추세 밴드",
}

STRATEGY_DESCRIPTIONS = {
    "research_trend": "EMA 정배열, 거래량 동반 돌파, ADX 추세 강도, ATR 이탈 손절을 함께 보는 추세 전략입니다.",
    "flux_trend": "다중 시간대 밴드와 추세 기준선을 함께 보는 추세 추종 전략입니다.",
}


def strategy_options(flux_available: bool) -> list[str]:
    options = ["research_trend"]
    if flux_available:
        options.append("flux_trend")
    return options


def strategy_label(name: str) -> str:
    return STRATEGY_LABELS.get(name, name)


def strategy_description(name: str) -> str:
    return STRATEGY_DESCRIPTIONS.get(name, name)


def build_strategy_frame(
    raw: pd.DataFrame,
    *,
    strategy_name: str,
    params: dict[str, Any] | None = None,
    flux_indicator: FluxCallable = None,
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
        result = raw.join(indicator_frame, how="left")
        if "strategy_score" not in result.columns:
            result["strategy_score"] = 0.0
        return result

    raise ValueError(f"unknown strategy: {strategy_name}")

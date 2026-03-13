"""Shared strategy engine used by backtest and live views."""

from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pandas as pd

try:
    from strategy import backtest_signal_frame, build_research_trend_signals
except ImportError:
    from src.strategy import backtest_signal_frame, build_research_trend_signals

FluxCallable = Callable[..., pd.DataFrame] | None

STRATEGY_LABELS = {
    "research_trend": "연구형 추세 돌파",
    "flux_trend": "플럭스 추세 밴드",
}

STRATEGY_DESCRIPTIONS = {
    "research_trend": "EMA 정배열, 거래량 동반 돌파, ADX 추세 강도, ATR 이탈 손절을 함께 보는 추세 전략입니다.",
    "flux_trend": "다중 시간대 볼린저 밴드와 칼만 기준선을 함께 보는 반전·재진입형 전략입니다.",
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


def sweep_strategy_parameters(
    raw: pd.DataFrame,
    *,
    strategy_name: str,
    base_params: dict[str, Any] | None = None,
    candidate_grid: dict[str, list[Any]] | None = None,
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
    flux_indicator: FluxCallable = None,
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

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
    "flux_ema_filter": "플럭스 + EMA 필터",
}

STRATEGY_DESCRIPTIONS = {
    "research_trend": "EMA 정배열, 거래량 동반 돌파, ADX 추세 강도, ATR 이탈 손절을 함께 보는 추세 전략입니다.",
    "flux_trend": "다중 시간대 볼린저 밴드와 칼만 기준선을 함께 보는 반전·재진입형 전략입니다.",
    "flux_ema_filter": "플럭스 신호 뒤에 EMA·ATR 확인이 일정 캔들 안에 따라올 때만 진입하는 확인형 추세-반전 혼합 전략입니다.",
}


def strategy_options(flux_available: bool, flux_ema_available: bool = False) -> list[str]:
    options = ["research_trend"]
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

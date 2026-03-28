"""Backtest-only strategy lab for iterative improvement and invention rounds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from hashlib import sha1
from typing import Any, Callable, Literal, Mapping, Sequence
import random

import numpy as np
import pandas as pd

try:
    from strategy import (
        average_directional_index,
        average_true_range,
        bollinger_bands,
        ema,
        relative_strength_index,
        VOLATILITY_RESET_BREAKOUT_DEFAULTS,
    )
    from strategy_engine import build_strategy_frame, strategy_label
    from strategy_tournament import backtest_portfolio_signal_frames
except ImportError:
    from src.strategy import (
        average_directional_index,
        average_true_range,
        bollinger_bands,
        ema,
        relative_strength_index,
        VOLATILITY_RESET_BREAKOUT_DEFAULTS,
    )
    from src.strategy_engine import build_strategy_frame, strategy_label
    from src.strategy_tournament import backtest_portfolio_signal_frames


TrackName = Literal["improve", "invent"]
CandidateKind = Literal["engine", "lab"]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    track: TrackName
    kind: CandidateKind
    strategy_name: str
    params: dict[str, Any] = field(default_factory=dict)
    generation: int = 0
    parent_id: str | None = None
    notes: str = ""

    @property
    def display_name(self) -> str:
        return strategy_label(self.strategy_name) if self.kind == "engine" else self.strategy_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "track": self.track,
            "kind": self.kind,
            "strategy_name": self.strategy_name,
            "params": dict(self.params),
            "generation": self.generation,
            "parent_id": self.parent_id,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SplitMetrics:
    initial_cash: float
    final_equity: float
    return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    trades: int
    buy_trades: int
    sell_trades: int
    open_positions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateResult:
    candidate: CandidateSpec
    train: SplitMetrics
    validation: SplitMetrics
    holdout: SplitMetrics
    holdout_weighted_score: float
    overfit_gap_pct: float
    rank: int = 0

    def to_row(self) -> dict[str, Any]:
        row = {
            "rank": self.rank,
            "candidate_id": self.candidate.candidate_id,
            "track": self.candidate.track,
            "kind": self.candidate.kind,
            "strategy_name": self.candidate.strategy_name,
            "display_name": self.candidate.display_name,
            "generation": self.candidate.generation,
            "parent_id": self.candidate.parent_id,
            "holdout_weighted_score": self.holdout_weighted_score,
            "overfit_gap_pct": self.overfit_gap_pct,
        }
        row.update({f"train_{key}": value for key, value in self.train.to_dict().items()})
        row.update({f"validation_{key}": value for key, value in self.validation.to_dict().items()})
        row.update({f"holdout_{key}": value for key, value in self.holdout.to_dict().items()})
        return row


@dataclass(frozen=True)
class RoundSummary:
    round_index: int
    candidates: list[CandidateResult]
    leaderboard: pd.DataFrame
    survivors: list[CandidateSpec]

    def top_result(self) -> CandidateResult | None:
        return self.candidates[0] if self.candidates else None


@dataclass(frozen=True)
class EvaluationWindow:
    name: str
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp
    validation_end: pd.Timestamp | None
    holdout_end: pd.Timestamp


@dataclass(frozen=True)
class CampaignCandidateResult:
    candidate: CandidateSpec
    window_results: list[CandidateResult]
    campaign_score: float
    avg_validation_return_pct: float
    avg_holdout_return_pct: float
    min_validation_return_pct: float
    min_holdout_return_pct: float
    worst_holdout_drawdown_pct: float
    rank: int = 0

    def to_row(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate.candidate_id,
            "track": self.candidate.track,
            "kind": self.candidate.kind,
            "strategy_name": self.candidate.strategy_name,
            "display_name": self.candidate.display_name,
            "generation": self.candidate.generation,
            "parent_id": self.candidate.parent_id,
            "campaign_score": self.campaign_score,
            "avg_validation_return_pct": self.avg_validation_return_pct,
            "avg_holdout_return_pct": self.avg_holdout_return_pct,
            "min_validation_return_pct": self.min_validation_return_pct,
            "min_holdout_return_pct": self.min_holdout_return_pct,
            "worst_holdout_drawdown_pct": self.worst_holdout_drawdown_pct,
            "windows": len(self.window_results),
        }


@dataclass(frozen=True)
class LabConfig:
    initial_cash: float = 10_000.0
    max_positions: int | None = 0
    allocation_pct: float = 1.0
    min_trade_krw: float = 0.0
    fee: float = 0.0005
    slippage_bps: float = 3.0
    train_weight: float = 0.05
    validation_weight: float = 0.45
    holdout_weight: float = 0.50
    drawdown_weight: float = 0.50
    overfit_weight: float = 0.60
    trade_penalty_weight: float = 0.015
    train_floor_return_pct: float = -25.0
    train_floor_penalty_weight: float = 0.80
    validation_loss_penalty_weight: float = 0.50
    validation_drawdown_guard_pct: float = 18.0
    validation_drawdown_penalty_weight: float = 0.30
    holdout_loss_penalty_weight: float = 0.75
    holdout_drawdown_guard_pct: float = 20.0
    holdout_drawdown_penalty_weight: float = 0.40
    parents_per_track: int = 2
    offspring_per_parent: int = 4
    inventions_per_round: int = 4
    random_seed: int = 7


def _stable_id(kind: str, track: TrackName, strategy_name: str, params: Mapping[str, Any], generation: int) -> str:
    payload = repr((kind, track, strategy_name, sorted((str(key), repr(value)) for key, value in params.items()), generation))
    return sha1(payload.encode("utf-8")).hexdigest()[:12]


def _slice_frame(frame: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    result = frame.sort_index()
    if start is not None:
        result = result.loc[result.index >= start]
    if end is not None:
        result = result.loc[result.index < end]
    return result.copy()


def split_market_frames(
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp | None,
    holdout_end: pd.Timestamp,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    train = {market: _slice_frame(frame, train_start, train_end) for market, frame in raw_by_market.items()}
    if validation_end is None:
        validation = {market: _slice_frame(frame, train_end, holdout_end) for market, frame in raw_by_market.items()}
        holdout = {market: _slice_frame(frame, holdout_end, holdout_end) for market, frame in raw_by_market.items()}
    else:
        validation = {market: _slice_frame(frame, train_end, validation_end) for market, frame in raw_by_market.items()}
        holdout = {market: _slice_frame(frame, validation_end, holdout_end) for market, frame in raw_by_market.items()}
    return train, validation, holdout


def _split_metrics(result: Mapping[str, Any]) -> SplitMetrics:
    return SplitMetrics(
        initial_cash=float(result.get("initial_cash", 0.0)),
        final_equity=float(result.get("final_equity", 0.0)),
        return_pct=float(result.get("total_return_pct", 0.0)),
        max_drawdown_pct=float(result.get("max_drawdown_pct", 0.0)),
        win_rate_pct=float(result.get("win_rate_pct", 0.0)),
        trades=int(result.get("trades", 0)),
        buy_trades=int(result.get("buy_trades", 0)),
        sell_trades=int(result.get("sell_trades", 0)),
        open_positions=int(result.get("open_positions", 0)),
    )


def _empty_backtest_result(initial_cash: float) -> dict[str, Any]:
    return {
        "initial_cash": initial_cash,
        "final_equity": initial_cash,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "win_rate_pct": 0.0,
        "trades": 0,
        "buy_trades": 0,
        "sell_trades": 0,
        "open_positions": 0,
    }


def _score(train: SplitMetrics, validation: SplitMetrics, holdout: SplitMetrics, config: LabConfig) -> tuple[float, float]:
    overfit_gap = max(0.0, train.return_pct - min(validation.return_pct, holdout.return_pct))
    train_floor_penalty = max(0.0, config.train_floor_return_pct - train.return_pct) * config.train_floor_penalty_weight
    validation_loss_penalty = max(0.0, -validation.return_pct) * config.validation_loss_penalty_weight
    validation_drawdown_penalty = (
        max(0.0, abs(validation.max_drawdown_pct) - config.validation_drawdown_guard_pct)
        * config.validation_drawdown_penalty_weight
    )
    holdout_loss_penalty = max(0.0, -holdout.return_pct) * config.holdout_loss_penalty_weight
    holdout_drawdown_penalty = (
        max(0.0, abs(holdout.max_drawdown_pct) - config.holdout_drawdown_guard_pct)
        * config.holdout_drawdown_penalty_weight
    )
    score = (
        (validation.return_pct * config.validation_weight)
        + (holdout.return_pct * config.holdout_weight)
        + (train.return_pct * config.train_weight)
        - (abs(validation.max_drawdown_pct) * (config.drawdown_weight * 0.85))
        - (abs(holdout.max_drawdown_pct) * config.drawdown_weight)
        - (abs(train.max_drawdown_pct) * (config.drawdown_weight * 0.25))
        - (overfit_gap * config.overfit_weight)
        - (max(0, validation.trades - 240) * (config.trade_penalty_weight * 0.5))
        - (max(0, holdout.trades - 240) * config.trade_penalty_weight)
        - train_floor_penalty
        - validation_loss_penalty
        - validation_drawdown_penalty
        - holdout_loss_penalty
        - holdout_drawdown_penalty
    )
    return score, overfit_gap


def _mutate_numeric(value: Any, rng: random.Random, *, lower: float, upper: float, integer: bool) -> Any:
    if integer:
        base = int(round(float(value)))
        delta = rng.choice([-6, -4, -2, -1, 1, 2, 4, 6])
        return max(int(lower), min(int(upper), base + delta))
    base = float(value)
    delta = rng.uniform(-0.25, 0.25)
    mutated = base + (abs(base) * delta if base else delta)
    return max(lower, min(upper, mutated))


def _mutate_boolean(value: Any, rng: random.Random) -> bool:
    return not bool(value) if rng.random() < 0.35 else bool(value)


def _bounded_mutation(name: str, value: Any, rng: random.Random) -> Any:
    key = name.lower()
    if isinstance(value, bool):
        return _mutate_boolean(value, rng)
    if isinstance(value, int) or key.endswith(("window", "length", "ema", "bars", "ticks", "period", "lookback")):
        lower, upper = (1, 400)
        if "rsi" in key:
            lower, upper = (2, 60)
        if "adx" in key:
            lower, upper = (1, 60)
        if "score" in key:
            lower, upper = (-20, 40)
        return _mutate_numeric(value, rng, lower=float(lower), upper=float(upper), integer=True)
    lower, upper = (-20.0, 40.0)
    if "quantile" in key:
        lower, upper = (0.05, 0.99)
    if any(token in key for token in ("mult", "threshold", "pct", "buffer", "quantile")):
        lower, upper = (0.0, 5.0)
    if "oversold" in key or "floor" in key:
        lower, upper = (-30.0, 50.0)
    if "adx" in key:
        lower, upper = (1.0, 60.0)
    if "quantile" in key:
        lower, upper = (0.05, 0.99)
    return _mutate_numeric(value, rng, lower=lower, upper=upper, integer=False)


def _dedupe_candidates(candidates: Sequence[CandidateSpec]) -> list[CandidateSpec]:
    unique: dict[str, CandidateSpec] = {}
    for candidate in candidates:
        unique[candidate.candidate_id] = candidate
    return list(unique.values())


def seed_candidates() -> list[CandidateSpec]:
    seeds: list[tuple[TrackName, CandidateKind, str, dict[str, Any]]] = [
        ("improve", "engine", "relative_strength_guard", {
            "breakout_window": 28,
            "entry_score": 9.0,
            "exit_score": 3.0,
            "guard_fast_ema": 13,
            "guard_slow_ema": 144,
            "guard_buffer_pct": 1.0,
            "guard_adx_floor": 10.0,
        }),
        ("improve", "engine", "relative_strength_rotation", {"entry_score": 8.0, "exit_score": 2.0}),
        ("improve", "engine", "regime_blend_guard", {
            "regime_adx_floor": 16.0,
            "bear_guard_buffer_pct": 1.5,
            "bear_guard_adx_floor": 14.0,
            "bear_guard_score_floor": -2.0,
        }),
        ("improve", "engine", "squeeze_breakout", {
            "bb_len": 20,
            "bb_mult": 2.0,
            "squeeze_window": 20,
            "breakout_window": 20,
            "trend_ema_window": 55,
            "volume_threshold": 1.1,
        }),
        ("invent", "engine", "volatility_reset_breakout", {
            **dict(VOLATILITY_RESET_BREAKOUT_DEFAULTS),
        }),
        ("invent", "lab", "lab_breakout_reversion_v1", {
            "trend_fast_ema": 21,
            "trend_slow_ema": 55,
            "breakout_window": 20,
            "squeeze_window": 20,
            "bb_len": 20,
            "bb_mult": 2.0,
            "atr_window": 14,
            "atr_mult": 2.0,
            "volume_window": 20,
            "volume_threshold": 1.0,
        }),
        ("invent", "lab", "lab_range_rebound_v1", {
            "rsi_len": 10,
            "oversold": 35.0,
            "bb_len": 20,
            "bb_mult": 1.8,
            "atr_window": 14,
            "atr_mult": 2.0,
            "adx_window": 14,
            "adx_ceiling": 18.0,
            "volume_window": 20,
            "volume_threshold": 0.9,
        }),
        ("invent", "lab", "lab_regime_switch_v1", {
            "trend_fast_ema": 21,
            "trend_slow_ema": 55,
            "rsi_len": 10,
            "oversold": 35.0,
            "bb_len": 20,
            "bb_mult": 1.8,
            "breakout_window": 20,
            "volume_window": 20,
            "volume_threshold": 1.0,
            "regime_threshold": 0.0,
        }),
    ]
    result: list[CandidateSpec] = []
    for track, kind, strategy_name, params in seeds:
        result.append(
            CandidateSpec(
                candidate_id=_stable_id(kind, track, strategy_name, params, 0),
                track=track,
                kind=kind,
                strategy_name=strategy_name,
                params=dict(params),
                generation=0,
                notes="seed",
            )
        )
    return result


def mutate_candidate(candidate: CandidateSpec, *, rng: random.Random | None = None, generation: int | None = None) -> CandidateSpec:
    rng = rng or random.Random()
    params = {key: _bounded_mutation(key, value, rng) for key, value in candidate.params.items()}
    next_generation = candidate.generation + 1 if generation is None else generation
    return CandidateSpec(
        candidate_id=_stable_id(candidate.kind, candidate.track, candidate.strategy_name, params, next_generation),
        track=candidate.track,
        kind=candidate.kind,
        strategy_name=candidate.strategy_name,
        params=params,
        generation=next_generation,
        parent_id=candidate.candidate_id,
        notes=f"mutated_from:{candidate.candidate_id}",
    )


def invent_candidate(
    parent: CandidateSpec | None = None,
    *,
    rng: random.Random | None = None,
    generation: int = 0,
) -> CandidateSpec:
    rng = rng or random.Random()
    template = rng.choice(["lab_breakout_reversion_v1", "lab_range_rebound_v1", "lab_regime_switch_v1"])
    base = dict(parent.params) if parent is not None else {}
    if template == "lab_breakout_reversion_v1":
        params = {
            "trend_fast_ema": int(base.get("trend_fast_ema", 21)),
            "trend_slow_ema": int(base.get("trend_slow_ema", 55)),
            "breakout_window": int(base.get("breakout_window", 20)),
            "squeeze_window": int(base.get("squeeze_window", 20)),
            "bb_len": int(base.get("bb_len", 20)),
            "bb_mult": float(base.get("bb_mult", 2.0)),
            "atr_window": int(base.get("atr_window", 14)),
            "atr_mult": float(base.get("atr_mult", 2.0)),
            "volume_window": int(base.get("volume_window", 20)),
            "volume_threshold": float(base.get("volume_threshold", 1.0)),
        }
    elif template == "lab_range_rebound_v1":
        params = {
            "rsi_len": int(base.get("rsi_len", 10)),
            "oversold": float(base.get("oversold", 35.0)),
            "bb_len": int(base.get("bb_len", 20)),
            "bb_mult": float(base.get("bb_mult", 1.8)),
            "atr_window": int(base.get("atr_window", 14)),
            "atr_mult": float(base.get("atr_mult", 2.0)),
            "adx_window": int(base.get("adx_window", 14)),
            "adx_ceiling": float(base.get("adx_ceiling", 18.0)),
            "volume_window": int(base.get("volume_window", 20)),
            "volume_threshold": float(base.get("volume_threshold", 0.9)),
        }
    else:
        params = {
            "trend_fast_ema": int(base.get("trend_fast_ema", 21)),
            "trend_slow_ema": int(base.get("trend_slow_ema", 55)),
            "rsi_len": int(base.get("rsi_len", 10)),
            "oversold": float(base.get("oversold", 35.0)),
            "bb_len": int(base.get("bb_len", 20)),
            "bb_mult": float(base.get("bb_mult", 1.8)),
            "breakout_window": int(base.get("breakout_window", 20)),
            "volume_window": int(base.get("volume_window", 20)),
            "volume_threshold": float(base.get("volume_threshold", 1.0)),
            "regime_threshold": float(base.get("regime_threshold", 0.0)),
        }
    return CandidateSpec(
        candidate_id=_stable_id("lab", "invent", template, params, generation),
        track="invent",
        kind="lab",
        strategy_name=template,
        params=params,
        generation=generation,
        parent_id=parent.candidate_id if parent is not None else None,
        notes="invented",
    )


def _build_lab_breakout_reversion_signals(raw: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    df = raw.copy()
    high = df["high"].astype(float)
    close = df["close"].astype(float)
    trend_fast_ema = int(params.get("trend_fast_ema", 21))
    trend_slow_ema = int(params.get("trend_slow_ema", 55))
    breakout_window = int(params.get("breakout_window", 20))
    squeeze_window = int(params.get("squeeze_window", 20))
    bb_len = int(params.get("bb_len", 20))
    bb_mult = float(params.get("bb_mult", 2.0))
    atr_window = int(params.get("atr_window", 14))
    atr_mult = float(params.get("atr_mult", 2.0))
    volume_window = int(params.get("volume_window", 20))
    volume_threshold = float(params.get("volume_threshold", 1.0))

    df["ema_fast"] = ema(close, trend_fast_ema)
    df["ema_slow"] = ema(close, trend_slow_ema)
    df["atr"] = average_true_range(df, atr_window)
    bb_basis, bb_upper, bb_lower = bollinger_bands(close, bb_len, bb_mult)
    df["bb_basis"] = bb_basis
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["breakout_high"] = high.rolling(breakout_window).max().shift(1)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)
    df["atr_stop"] = df["ema_slow"] - (atr_mult * df["atr"])
    df["bandwidth"] = ((bb_upper - bb_lower) / bb_basis.replace(0.0, np.nan)) * 100.0
    squeeze_min_periods = min(squeeze_window, max(5, squeeze_window // 2))
    squeeze_ref = df["bandwidth"].rolling(squeeze_window, min_periods=squeeze_min_periods).quantile(0.35).shift(1)
    squeeze_on = (df["bandwidth"] <= squeeze_ref).fillna(False)
    recent_squeeze = squeeze_on.rolling(4, min_periods=1).max().astype(bool)
    trend_ok = (close > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"] > df["ema_fast"].shift(max(2, trend_fast_ema // 5)))
    breakout = close > df["breakout_high"]
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold
    momentum = close.pct_change(max(2, breakout_window // 3)) * 100.0
    breakout_gap = ((close / df["breakout_high"].replace(0.0, np.nan)) - 1.0) * 100.0
    df["strategy_score"] = (
        breakout_gap.fillna(0.0) * 18.0
        + momentum.fillna(0.0) * 1.2
        + (df["volume_ratio"].fillna(1.0) - 1.0) * 8.0
        + (squeeze_ref - df["bandwidth"]).replace([np.inf, -np.inf], np.nan).fillna(0.0) * 2.0
        - df["bandwidth"].fillna(0.0) * 0.15
    )
    raw_buy = recent_squeeze & breakout & trend_ok & volume_ok
    raw_sell = (close < df["ema_slow"]) | (close < df["atr_stop"]) | ((close < df["bb_basis"]) & (df["bandwidth"] > df["bandwidth"].shift(1)))
    df["buy_signal"] = raw_buy & (~raw_buy.shift(1, fill_value=False))
    df["sell_signal"] = raw_sell & (~raw_sell.shift(1, fill_value=False))
    df["squeeze_on"] = squeeze_on
    return df


def _build_lab_range_rebound_signals(raw: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    df = raw.copy()
    open_ = df["open"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    rsi_len = int(params.get("rsi_len", 10))
    oversold = float(params.get("oversold", 35.0))
    bb_len = int(params.get("bb_len", 20))
    bb_mult = float(params.get("bb_mult", 1.8))
    atr_window = int(params.get("atr_window", 14))
    atr_mult = float(params.get("atr_mult", 2.0))
    adx_window = int(params.get("adx_window", 14))
    adx_ceiling = float(params.get("adx_ceiling", 18.0))
    volume_window = int(params.get("volume_window", 20))
    volume_threshold = float(params.get("volume_threshold", 0.9))

    df["rsi"] = relative_strength_index(close, rsi_len)
    bb_basis, bb_upper, bb_lower = bollinger_bands(close, bb_len, bb_mult)
    df["bb_basis"] = bb_basis
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["atr"] = average_true_range(df, atr_window)
    df["adx"] = average_directional_index(df, adx_window)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)
    df["atr_stop"] = close - (atr_mult * df["atr"])
    sideways_ok = (df["adx"].fillna(0.0) <= adx_ceiling) | (close.pct_change(4).abs().fillna(0.0) < 0.01)
    setup = (df["rsi"] <= oversold) & (low <= df["bb_lower"]) & sideways_ok
    rebound = (close > open_) & (close > close.shift(1))
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold
    exit_rsi = max(45.0, oversold + 12.0)
    df["strategy_score"] = (
        (oversold - df["rsi"]).clip(lower=0.0).fillna(0.0)
        + (((df["bb_lower"] - close) / close.replace(0.0, np.nan)) * 100.0).clip(lower=0.0).fillna(0.0) * 1.5
        + (df["volume_ratio"].fillna(1.0) - 1.0) * 4.0
    )
    raw_buy = setup & rebound & volume_ok
    raw_sell = (df["rsi"] >= exit_rsi) | (close < df["atr_stop"]) | (close < close.shift(1))
    df["buy_signal"] = raw_buy & (~raw_buy.shift(1, fill_value=False))
    df["sell_signal"] = raw_sell & (~raw_sell.shift(1, fill_value=False))
    return df


def _build_lab_regime_switch_signals(raw: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    trend = _build_lab_breakout_reversion_signals(raw, params)
    range_ = _build_lab_range_rebound_signals(raw, params)
    close = raw["close"].astype(float)
    regime_threshold = float(params.get("regime_threshold", 0.0))
    trend_score = pd.to_numeric(trend["strategy_score"], errors="coerce").fillna(0.0)
    range_score = pd.to_numeric(range_["strategy_score"], errors="coerce").fillna(0.0)
    regime_score = (
        (pd.to_numeric(trend.get("ema_fast"), errors="coerce").fillna(close) / pd.to_numeric(trend.get("ema_slow"), errors="coerce").fillna(close) - 1.0) * 100.0
    ).fillna(0.0)
    regime_score += pd.to_numeric(trend.get("volume_ratio"), errors="coerce").fillna(1.0) - 1.0
    regime_score += pd.to_numeric(range_.get("adx"), errors="coerce").fillna(0.0) / 100.0

    df = raw.copy()
    df["trend_score"] = trend_score
    df["range_score"] = range_score
    df["regime_score"] = regime_score
    df["trend_buy_signal"] = trend["buy_signal"].astype(bool)
    df["trend_sell_signal"] = trend["sell_signal"].astype(bool)
    df["range_buy_signal"] = range_["buy_signal"].astype(bool)
    df["range_sell_signal"] = range_["sell_signal"].astype(bool)
    df["ema_fast"] = trend.get("ema_fast")
    df["ema_slow"] = trend.get("ema_slow")
    df["rsi"] = range_.get("rsi")
    df["bb_lower"] = range_.get("bb_lower")
    df["bb_upper"] = range_.get("bb_upper")
    df["atr_stop"] = pd.to_numeric(trend.get("atr_stop"), errors="coerce").fillna(pd.to_numeric(range_.get("atr_stop"), errors="coerce"))

    trend_regime = regime_score >= regime_threshold
    buy_signal = pd.Series(False, index=df.index, dtype=bool)
    sell_signal = pd.Series(False, index=df.index, dtype=bool)
    active_mode: str | None = None
    mode_trace: list[str] = []
    for ts in df.index:
        if active_mode == "trend" and bool(df.at[ts, "trend_sell_signal"]):
            sell_signal.at[ts] = True
            active_mode = None
        elif active_mode == "range" and bool(df.at[ts, "range_sell_signal"]):
            sell_signal.at[ts] = True
            active_mode = None
        if active_mode is None:
            if bool(trend_regime.loc[ts]) and bool(df.at[ts, "trend_buy_signal"]):
                buy_signal.at[ts] = True
                active_mode = "trend"
            elif (not bool(trend_regime.loc[ts])) and bool(df.at[ts, "range_buy_signal"]):
                buy_signal.at[ts] = True
                active_mode = "range"
        mode_trace.append(active_mode or "")
    df["trend_regime"] = trend_regime.astype(bool)
    df["entry_mode"] = pd.Series(mode_trace, index=df.index, dtype="string")
    df["buy_signal"] = buy_signal
    df["sell_signal"] = sell_signal
    df["strategy_score"] = trend_score.where(df["trend_regime"], range_score)
    return df


LAB_BUILDERS: dict[str, Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]] = {
    "lab_breakout_reversion_v1": _build_lab_breakout_reversion_signals,
    "lab_range_rebound_v1": _build_lab_range_rebound_signals,
    "lab_regime_switch_v1": _build_lab_regime_switch_signals,
}


def build_lab_strategy_frame(raw: pd.DataFrame, *, strategy_name: str, params: Mapping[str, Any] | None = None) -> pd.DataFrame:
    builder = LAB_BUILDERS.get(strategy_name)
    if builder is None:
        raise ValueError(f"unknown lab strategy: {strategy_name}")
    return builder(raw, dict(params or {}))


def _portfolio_backtest_for_candidate(
    raw_by_market: Mapping[str, pd.DataFrame],
    candidate: CandidateSpec,
    *,
    config: LabConfig,
) -> dict[str, Any]:
    if candidate.kind == "engine":
        frames_by_market = {
            market: build_strategy_frame(raw, strategy_name=candidate.strategy_name, params=dict(candidate.params))
            for market, raw in raw_by_market.items()
            if not raw.empty
        }
    else:
        frames_by_market = {
            market: build_lab_strategy_frame(raw, strategy_name=candidate.strategy_name, params=candidate.params)
            for market, raw in raw_by_market.items()
            if not raw.empty
        }
    if not frames_by_market:
        return _empty_backtest_result(config.initial_cash)
    return backtest_portfolio_signal_frames(
        frames_by_market,
        strategy_name=candidate.strategy_name,
        initial_cash=config.initial_cash,
        max_positions=config.max_positions,
        allocation_pct=config.allocation_pct,
        min_trade_krw=config.min_trade_krw,
        fee=config.fee,
        slippage_bps=config.slippage_bps,
    )


def evaluate_candidate(
    candidate: CandidateSpec,
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp | None = None,
    holdout_end: pd.Timestamp,
    config: LabConfig | None = None,
) -> CandidateResult:
    config = config or LabConfig()
    train_raw, validation_raw, holdout_raw = split_market_frames(
        raw_by_market,
        train_start=train_start,
        train_end=train_end,
        validation_end=validation_end,
        holdout_end=holdout_end,
    )
    train_result = _portfolio_backtest_for_candidate(train_raw, candidate, config=config)
    validation_result = _portfolio_backtest_for_candidate(validation_raw, candidate, config=config)
    holdout_result = _portfolio_backtest_for_candidate(holdout_raw, candidate, config=config)
    train = _split_metrics(train_result)
    validation = _split_metrics(validation_result)
    holdout = _split_metrics(holdout_result)
    score, gap = _score(train, validation, holdout, config)
    return CandidateResult(
        candidate=candidate,
        train=train,
        validation=validation,
        holdout=holdout,
        holdout_weighted_score=score,
        overfit_gap_pct=gap,
    )


def rank_candidates(results: Sequence[CandidateResult]) -> list[CandidateResult]:
    ranked = sorted(
        results,
        key=lambda item: (
            item.holdout_weighted_score,
            item.holdout.final_equity,
            item.validation.final_equity,
            -abs(item.holdout.max_drawdown_pct),
            -abs(item.validation.max_drawdown_pct),
            item.holdout.return_pct,
            item.validation.return_pct,
            -item.holdout.trades,
        ),
        reverse=True,
    )
    return [replace(result, rank=index + 1) for index, result in enumerate(ranked)]


def select_survivors(
    ranked: Sequence[CandidateResult],
    *,
    config: LabConfig | None = None,
) -> list[CandidateSpec]:
    config = config or LabConfig()

    def passes_survival_floor(result: CandidateResult) -> bool:
        return (
            result.train.return_pct >= config.train_floor_return_pct
            and result.validation.return_pct > -20.0
            and result.holdout.return_pct > -20.0
        )

    selected: list[CandidateSpec] = []
    seen: set[str] = set()
    for track in ("improve", "invent"):
        track_pool = [result for result in ranked if result.candidate.track == track]
        preferred = [result for result in track_pool if passes_survival_floor(result)]
        track_results = (preferred or track_pool)[: max(1, config.parents_per_track)]
        for result in track_results:
            if result.candidate.candidate_id not in seen:
                selected.append(result.candidate)
                seen.add(result.candidate.candidate_id)
    total_target = max(2, config.parents_per_track * 2)
    for result in ranked:
        if len(selected) >= total_target:
            break
        if result.candidate.candidate_id in seen:
            continue
        if not passes_survival_floor(result):
            continue
        selected.append(result.candidate)
        seen.add(result.candidate.candidate_id)
    if len(selected) < total_target:
        for result in ranked:
            if len(selected) >= total_target:
                break
            if result.candidate.candidate_id in seen:
                continue
            selected.append(result.candidate)
            seen.add(result.candidate.candidate_id)
    return selected


def make_offspring(
    parents: Sequence[CandidateSpec],
    *,
    config: LabConfig | None = None,
    rng: random.Random | None = None,
) -> list[CandidateSpec]:
    config = config or LabConfig()
    rng = rng or random.Random(config.random_seed)
    offspring: list[CandidateSpec] = []
    for parent in parents:
        for _ in range(max(1, config.offspring_per_parent)):
            offspring.append(mutate_candidate(parent, rng=rng, generation=parent.generation + 1))
    inventor_parents = [parent for parent in parents if parent.track == "invent"]
    for _ in range(max(0, config.inventions_per_round)):
        base_parent = rng.choice(inventor_parents) if inventor_parents else None
        offspring.append(
            invent_candidate(
                base_parent if (base_parent and base_parent.kind == "lab") else None,
                rng=rng,
                generation=(base_parent.generation + 1) if base_parent else 0,
            )
        )
    return _dedupe_candidates(offspring)


def run_round(
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp | None = None,
    holdout_end: pd.Timestamp,
    candidates: Sequence[CandidateSpec] | None = None,
    config: LabConfig | None = None,
    rng: random.Random | None = None,
) -> RoundSummary:
    config = config or LabConfig()
    rng = rng or random.Random(config.random_seed)
    population = list(candidates or seed_candidates())
    results = [
        evaluate_candidate(
            candidate,
            raw_by_market,
            train_start=train_start,
            train_end=train_end,
            validation_end=validation_end,
            holdout_end=holdout_end,
            config=config,
        )
        for candidate in population
    ]
    ranked = rank_candidates(results)
    leaderboard = pd.DataFrame([result.to_row() for result in ranked])
    survivors = select_survivors(ranked, config=config)
    return RoundSummary(round_index=0, candidates=ranked, leaderboard=leaderboard, survivors=survivors)


def run_evolution(
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    train_start: pd.Timestamp | None,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp | None = None,
    holdout_end: pd.Timestamp,
    rounds: int = 3,
    config: LabConfig | None = None,
) -> list[RoundSummary]:
    config = config or LabConfig()
    rng = random.Random(config.random_seed)
    population = seed_candidates()
    history: list[RoundSummary] = []
    for round_index in range(max(1, rounds)):
        round_summary = run_round(
            raw_by_market,
            train_start=train_start,
            train_end=train_end,
            validation_end=validation_end,
            holdout_end=holdout_end,
            candidates=population,
            config=config,
            rng=rng,
        )
        history.append(replace(round_summary, round_index=round_index))
        population = _dedupe_candidates([
            *round_summary.survivors,
            *make_offspring(round_summary.survivors, config=config, rng=rng),
            *seed_candidates()[:2],
        ])
    return history


def best_candidate(history: Sequence[RoundSummary]) -> CandidateResult | None:
    best: CandidateResult | None = None
    for round_summary in history:
        top = round_summary.top_result()
        if top is None:
            continue
        if best is None or top.holdout_weighted_score > best.holdout_weighted_score:
            best = top
    return best


def evaluate_candidate_campaign(
    candidate: CandidateSpec,
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    windows: Sequence[EvaluationWindow],
    config: LabConfig | None = None,
) -> CampaignCandidateResult:
    config = config or LabConfig()
    window_results = [
        evaluate_candidate(
            candidate,
            raw_by_market,
            train_start=window.train_start,
            train_end=window.train_end,
            validation_end=window.validation_end,
            holdout_end=window.holdout_end,
            config=config,
        )
        for window in windows
    ]
    scores = [result.holdout_weighted_score for result in window_results]
    validation_returns = [result.validation.return_pct for result in window_results]
    holdout_returns = [result.holdout.return_pct for result in window_results]
    worst_holdout_drawdown = min(result.holdout.max_drawdown_pct for result in window_results)
    avg_score = float(np.mean(scores)) if scores else float("-inf")
    min_score = min(scores) if scores else float("-inf")
    avg_validation_return = float(np.mean(validation_returns)) if validation_returns else 0.0
    avg_holdout_return = float(np.mean(holdout_returns)) if holdout_returns else 0.0
    min_validation_return = min(validation_returns) if validation_returns else 0.0
    min_holdout_return = min(holdout_returns) if holdout_returns else 0.0
    campaign_score = (
        (avg_score * 0.55)
        + (min_score * 0.45)
        - (max(0.0, -min_validation_return) * 0.25)
        - (max(0.0, -min_holdout_return) * 0.35)
        - (max(0.0, abs(worst_holdout_drawdown) - 20.0) * 0.20)
    )
    return CampaignCandidateResult(
        candidate=candidate,
        window_results=window_results,
        campaign_score=campaign_score,
        avg_validation_return_pct=avg_validation_return,
        avg_holdout_return_pct=avg_holdout_return,
        min_validation_return_pct=min_validation_return,
        min_holdout_return_pct=min_holdout_return,
        worst_holdout_drawdown_pct=worst_holdout_drawdown,
    )


def rank_campaign_candidates(results: Sequence[CampaignCandidateResult]) -> list[CampaignCandidateResult]:
    ranked = sorted(
        results,
        key=lambda item: (
            item.campaign_score,
            item.min_holdout_return_pct,
            item.avg_holdout_return_pct,
            item.min_validation_return_pct,
            -abs(item.worst_holdout_drawdown_pct),
        ),
        reverse=True,
    )
    return [replace(result, rank=index + 1) for index, result in enumerate(ranked)]

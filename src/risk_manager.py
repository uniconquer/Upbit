"""Risk-rule helpers for simulation and guarded live-trading flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(slots=True)
class RiskConfig:
    max_trade_krw: float = 0.0
    max_trade_pct: float = 0.0
    per_asset_max_pct: float = 0.0
    daily_buy_limit: float = 0.0
    daily_loss_limit_krw: float = 0.0
    daily_loss_limit_pct: float = 0.0
    include_unrealized_loss: bool = False


@dataclass(slots=True)
class EntryDecision:
    allowed: bool
    trade_cost: float = 0.0
    blocked_reason: str | None = None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def risk_config_from_dict(raw: Mapping[str, Any] | None) -> RiskConfig:
    raw = raw or {}
    return RiskConfig(
        max_trade_krw=_to_float(raw.get("max_trade_krw")),
        max_trade_pct=_to_float(raw.get("max_trade_pct")),
        per_asset_max_pct=_to_float(raw.get("per_asset_max_pct")),
        daily_buy_limit=_to_float(raw.get("daily_buy_limit")),
        daily_loss_limit_krw=_to_float(raw.get("daily_loss_limit_krw")),
        daily_loss_limit_pct=_to_float(raw.get("daily_loss_limit_pct")),
        include_unrealized_loss=bool(raw.get("include_unrealized_loss")),
    )


def ensure_daily_metrics(
    metrics: Mapping[str, Any] | None,
    *,
    day: str,
    day_start_equity: float | None = None,
) -> dict[str, Any]:
    current = dict(metrics or {})
    if not current or current.get("day_date") != day:
        return {
            "day_date": day,
            "daily_buy": 0.0,
            "realized_pnl": 0.0,
            "day_start_equity": day_start_equity,
            "unrealized_pnl": 0.0,
            "total_pnl": 0.0,
        }
    if day_start_equity is not None and not current.get("day_start_equity"):
        current["day_start_equity"] = day_start_equity
    current.setdefault("daily_buy", 0.0)
    current.setdefault("realized_pnl", 0.0)
    current.setdefault("day_start_equity", day_start_equity)
    current.setdefault("unrealized_pnl", 0.0)
    current.setdefault("total_pnl", current["realized_pnl"])
    return current


def total_unrealized_pnl(
    positions: Mapping[str, Mapping[str, Any]],
    price_map: Mapping[str, float],
) -> float:
    total = 0.0
    for market, position in positions.items():
        price = price_map.get(market)
        if price is None:
            continue
        entry = _to_float(position.get("entry"))
        qty = _to_float(position.get("qty"))
        total += (float(price) - entry) * qty
    return total


def effective_loss(
    metrics: Mapping[str, Any],
    positions: Mapping[str, Mapping[str, Any]],
    price_map: Mapping[str, float],
    *,
    include_unrealized_loss: bool,
) -> float:
    realized = _to_float(metrics.get("realized_pnl"))
    unrealized = total_unrealized_pnl(positions, price_map) if include_unrealized_loss else 0.0
    return realized + unrealized


def choose_trade_cost(config: RiskConfig, day_start_equity: float | None) -> float:
    trade_cost = config.max_trade_krw if config.max_trade_krw > 0 else 0.0
    if config.max_trade_pct > 0 and day_start_equity:
        pct_cost = day_start_equity * config.max_trade_pct / 100.0
        trade_cost = min(trade_cost, pct_cost) if trade_cost > 0 else pct_cost
    return trade_cost


def evaluate_entry(
    *,
    config: RiskConfig,
    metrics: Mapping[str, Any],
    positions: Mapping[str, Mapping[str, Any]],
    price_map: Mapping[str, float],
    market: str,
    day_start_equity: float | None,
) -> EntryDecision:
    effective = effective_loss(
        metrics,
        positions,
        price_map,
        include_unrealized_loss=config.include_unrealized_loss,
    )
    if config.daily_loss_limit_krw > 0 and effective <= -config.daily_loss_limit_krw:
        return EntryDecision(False, blocked_reason=f"daily loss {config.daily_loss_limit_krw:g} KRW")
    if config.daily_loss_limit_pct > 0 and day_start_equity:
        loss_pct = (effective / day_start_equity) * 100 if day_start_equity else 0.0
        if loss_pct <= -config.daily_loss_limit_pct:
            return EntryDecision(False, blocked_reason=f"daily loss {config.daily_loss_limit_pct:g}%")

    trade_cost = choose_trade_cost(config, day_start_equity)
    if trade_cost <= 0:
        return EntryDecision(False, blocked_reason="no allocation")

    daily_buy = _to_float(metrics.get("daily_buy"))
    if config.daily_buy_limit > 0 and (daily_buy + trade_cost) > config.daily_buy_limit:
        return EntryDecision(False, blocked_reason=f"daily buy {config.daily_buy_limit:g}")

    existing_cost = _to_float((positions.get(market) or {}).get("cost"))
    if config.per_asset_max_pct > 0 and day_start_equity:
        asset_pct = ((existing_cost + trade_cost) / day_start_equity) * 100
        if asset_pct > config.per_asset_max_pct:
            return EntryDecision(False, blocked_reason=f"asset {config.per_asset_max_pct:g}%")

    return EntryDecision(True, trade_cost=trade_cost)

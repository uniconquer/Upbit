"""Helpers for aligning local runtime state with live Upbit account state."""

from __future__ import annotations

import time
from typing import Any, Mapping

try:
    from execution import build_pending_order, extract_fill_metrics
except ImportError:
    from src.execution import build_pending_order, extract_fill_metrics


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def positions_from_accounts(
    accounts: list[Mapping[str, Any]],
    *,
    existing_positions: Mapping[str, Mapping[str, Any]] | None = None,
    quote_currency: str = "KRW",
    strategy_name: str = "research_trend",
    now_ts: float | None = None,
) -> dict[str, dict[str, Any]]:
    existing_positions = existing_positions or {}
    quote_currency = quote_currency.upper()
    now_ts = float(now_ts or time.time())
    positions: dict[str, dict[str, Any]] = {}

    for account in accounts:
        currency = str(account.get("currency") or "").upper()
        if not currency or currency == quote_currency:
            continue
        market = f"{quote_currency}-{currency}"
        qty = _to_float(account.get("balance")) + _to_float(account.get("locked"))
        if qty <= 0:
            continue

        existing = dict(existing_positions.get(market) or {})
        entry = _to_float(account.get("avg_buy_price")) or _to_float(existing.get("entry"))
        if entry <= 0 and not existing:
            continue

        positions[market] = {
            "market": market,
            "qty": qty,
            "entry": entry,
            "cost": qty * entry if entry > 0 else _to_float(existing.get("cost")),
            "opened_at": _to_float(existing.get("opened_at")) or now_ts,
            "strategy": str(existing.get("strategy") or strategy_name),
            "entry_order_uuid": existing.get("entry_order_uuid"),
        }
    return positions


def pending_orders_from_open_orders(
    orders: list[Mapping[str, Any]],
    *,
    strategy_name: str,
    existing_pending_orders: Mapping[str, Mapping[str, Any]] | None = None,
    existing_positions: Mapping[str, Mapping[str, Any]] | None = None,
    quote_currency: str = "KRW",
) -> dict[str, dict[str, Any]]:
    existing_pending_orders = existing_pending_orders or {}
    existing_positions = existing_positions or {}
    quote_prefix = quote_currency.upper() + "-"
    pending_orders: dict[str, dict[str, Any]] = {}

    for order in orders:
        market = str(order.get("market") or "")
        if not market.startswith(quote_prefix):
            continue
        side = str(order.get("side") or "").lower()
        if side not in {"bid", "ask"}:
            continue

        prev = dict(existing_pending_orders.get(market) or {})
        position = dict(existing_positions.get(market) or {})
        ord_type = str(order.get("ord_type") or "")
        remaining_volume = _to_float(order.get("remaining_volume")) or _to_float(order.get("volume"))
        limit_price = _to_float(order.get("price"))
        locked = _to_float(order.get("locked"))

        requested_cost: float | None = None
        requested_qty: float | None = None
        fallback_price = _to_float(prev.get("fallback_price")) or _to_float(position.get("entry"))

        if side == "bid":
            if ord_type == "limit" and limit_price > 0 and remaining_volume > 0:
                requested_cost = limit_price * remaining_volume
                fallback_price = fallback_price or limit_price
            else:
                requested_cost = locked or limit_price or (_to_float(prev.get("requested_cost")) or None)
            if remaining_volume > 0:
                requested_qty = remaining_volume
        else:
            requested_qty = remaining_volume or (_to_float(prev.get("requested_qty")) or None)
            fallback_price = fallback_price or limit_price

        fill = extract_fill_metrics(
            order,
            side=side,
            fallback_price=fallback_price or limit_price,
            fallback_cost=requested_cost,
            fallback_qty=requested_qty,
        )
        pending_orders[market] = build_pending_order(
            market=market,
            side=side,
            strategy=str(prev.get("strategy") or position.get("strategy") or strategy_name),
            fallback_price=fallback_price,
            order_result=order,
            requested_cost=requested_cost,
            requested_qty=requested_qty,
            reason=str(prev.get("reason") or "exchange_open_order"),
            submitted_at=_to_float(prev.get("submitted_at")) or time.time(),
            fill=fill,
        )
    return pending_orders


def rebuild_signal_state(
    previous_state: Mapping[str, Mapping[str, Any]] | None,
    *,
    positions: Mapping[str, Mapping[str, Any]],
    pending_orders: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    signal_state = {
        market: dict(state)
        for market, state in (previous_state or {}).items()
        if market in positions or market in pending_orders
    }
    for market, position in positions.items():
        signal_state[market] = {
            "sig": "BUY",
            "entry": _to_float(position.get("entry")) or None,
        }
    for market, pending in pending_orders.items():
        side = str(pending.get("side") or "").lower()
        signal_state[market] = {
            "sig": "BUY_PENDING" if side == "bid" else "SELL_PENDING",
            "entry": _to_float((positions.get(market) or {}).get("entry")) or None,
        }
    return signal_state


def sync_exchange_state(
    api,
    *,
    strategy_name: str,
    existing_positions: Mapping[str, Mapping[str, Any]] | None = None,
    existing_pending_orders: Mapping[str, Mapping[str, Any]] | None = None,
    existing_signal_state: Mapping[str, Mapping[str, Any]] | None = None,
    quote_currency: str = "KRW",
    announce_success: bool = True,
) -> dict[str, Any]:
    positions = dict(existing_positions or {})
    pending_orders = dict(existing_pending_orders or {})
    notifications: list[str] = []
    accounts_ok = False
    orders_ok = False

    try:
        accounts = api.accounts()
        positions = positions_from_accounts(
            accounts,
            existing_positions=positions,
            quote_currency=quote_currency,
            strategy_name=strategy_name,
        )
        accounts_ok = True
    except Exception as exc:
        notifications.append(f"[\uc2e4\uac70\ub798] \uac70\ub798\uc18c \uc794\uace0 \ub3d9\uae30\ud654\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4: {exc}")

    try:
        open_orders = api.open_orders(states=["wait", "watch"])
        mapped_pending_orders = pending_orders_from_open_orders(
            open_orders,
            strategy_name=strategy_name,
            existing_pending_orders=pending_orders,
            existing_positions=positions,
            quote_currency=quote_currency,
        )
        pending_orders = mapped_pending_orders
        orders_ok = True
    except Exception as exc:
        notifications.append(f"[\uc2e4\uac70\ub798] \ubbf8\uccb4\uacb0 \uc8fc\ubb38 \ub3d9\uae30\ud654\uc5d0 \uc2e4\ud328\ud588\uc2b5\ub2c8\ub2e4: {exc}")

    if accounts_ok and orders_ok and announce_success:
        notifications.append(
            f"[\uc2e4\uac70\ub798] \uac70\ub798\uc18c \uc0c1\ud0dc \ub3d9\uae30\ud654 \uc644\ub8cc: \ubcf4\uc720 \ud3ec\uc9c0\uc158 {len(positions)}\uac1c, \ubbf8\uccb4\uacb0 \uc8fc\ubb38 {len(pending_orders)}\uac74"
        )

    return {
        "positions": positions,
        "pending_orders": pending_orders,
        "last_signal_state": rebuild_signal_state(
            existing_signal_state,
            positions=positions,
            pending_orders=pending_orders,
        ),
        "notifications": notifications,
        "synced": accounts_ok and orders_ok,
    }

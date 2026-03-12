"""Helpers for reconciling created orders with final exchange state."""

from __future__ import annotations

import time
from typing import Any, Mapping


TERMINAL_STATES = {"done", "cancel"}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def wait_for_order_completion(
    api,
    *,
    uuid: str | None = None,
    identifier: str | None = None,
    timeout_seconds: float = 3.0,
    poll_interval: float = 0.4,
) -> dict[str, Any] | None:
    if not uuid and not identifier:
        return None

    deadline = time.monotonic() + timeout_seconds
    latest: dict[str, Any] | None = None
    while time.monotonic() <= deadline:
        latest = api.get_order(uuid=uuid, identifier=identifier)
        state = str(latest.get("state") or "")
        remaining = _to_float(latest.get("remaining_volume"))
        executed_volume = _to_float(latest.get("executed_volume"))
        if state in TERMINAL_STATES or (executed_volume > 0 and remaining <= 0):
            return latest
        time.sleep(poll_interval)
    return latest


def extract_fill_metrics(
    order: Mapping[str, Any] | None,
    *,
    fallback_price: float,
    fallback_cost: float | None = None,
    fallback_qty: float | None = None,
) -> dict[str, Any]:
    order = dict(order or {})
    executed_qty = _to_float(order.get("executed_volume")) or _to_float(fallback_qty)
    executed_cost = _to_float(order.get("executed_fund")) or _to_float(fallback_cost)
    average_price = fallback_price
    if executed_qty > 0 and executed_cost > 0:
        average_price = executed_cost / executed_qty
    elif executed_qty > 0 and average_price > 0 and executed_cost <= 0:
        executed_cost = average_price * executed_qty
    elif executed_cost > 0 and average_price > 0 and executed_qty <= 0:
        executed_qty = executed_cost / average_price

    return {
        "order_uuid": order.get("uuid"),
        "order_state": order.get("state"),
        "price": average_price,
        "qty": executed_qty,
        "cost": executed_cost,
        "remaining_volume": _to_float(order.get("remaining_volume")),
        "raw_order": order,
    }


def resolve_submitted_order(
    api,
    order_result: Mapping[str, Any] | None,
    *,
    live_orders: bool,
    fallback_price: float,
    fallback_cost: float | None = None,
    fallback_qty: float | None = None,
    timeout_seconds: float = 3.0,
    poll_interval: float = 0.4,
) -> dict[str, Any]:
    raw_result = dict(order_result or {})
    if raw_result.get("error"):
        return {
            "status": "error",
            "error": raw_result.get("error"),
            "status_code": raw_result.get("status_code"),
            "fill": extract_fill_metrics(
                None,
                fallback_price=fallback_price,
                fallback_cost=fallback_cost,
                fallback_qty=fallback_qty,
            ),
            "order": raw_result,
        }

    resolved_order = raw_result
    order_uuid = raw_result.get("uuid")
    if live_orders and order_uuid:
        try:
            resolved_order = (
                wait_for_order_completion(
                    api,
                    uuid=str(order_uuid),
                    timeout_seconds=timeout_seconds,
                    poll_interval=poll_interval,
                )
                or raw_result
            )
        except Exception as exc:
            resolved_order = dict(raw_result)
            resolved_order["lookup_error"] = repr(exc)

    fill = extract_fill_metrics(
        resolved_order,
        fallback_price=fallback_price,
        fallback_cost=fallback_cost,
        fallback_qty=fallback_qty,
    )
    order_state = str(fill.get("order_state") or "")
    actual_executed_qty = _to_float(resolved_order.get("executed_volume"))
    actual_remaining = _to_float(resolved_order.get("remaining_volume"))
    if not live_orders:
        status = "filled"
    elif order_state == "cancel":
        status = "cancelled"
    elif order_state == "done" or (actual_executed_qty > 0 and actual_remaining <= 0):
        status = "filled"
    else:
        status = "pending"

    return {
        "status": status,
        "fill": fill,
        "order": resolved_order,
        "lookup_error": resolved_order.get("lookup_error"),
    }


def build_pending_order(
    *,
    market: str,
    side: str,
    strategy: str,
    fallback_price: float,
    order_result: Mapping[str, Any] | None,
    requested_cost: float | None = None,
    requested_qty: float | None = None,
    reason: str = "signal",
    submitted_at: float | None = None,
) -> dict[str, Any]:
    raw_result = dict(order_result or {})
    return {
        "market": market,
        "side": side,
        "strategy": strategy,
        "reason": reason,
        "uuid": raw_result.get("uuid"),
        "fallback_price": float(fallback_price),
        "requested_cost": float(requested_cost) if requested_cost is not None else None,
        "requested_qty": float(requested_qty) if requested_qty is not None else None,
        "submitted_at": float(submitted_at or time.time()),
    }

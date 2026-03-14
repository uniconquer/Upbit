"""Helpers for reconciling created orders with final exchange state."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Mapping


TERMINAL_STATES = {"done", "cancel", "prevented"}


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_client_order_identifier(market: str, side: str, *, strategy_name: str = "strategy") -> str:
    market_token = "".join(ch for ch in str(market or "").upper() if ch.isalnum())[:16] or "MARKET"
    side_token = str(side or "").lower()[:4] or "side"
    strategy_token = "".join(ch for ch in str(strategy_name or "").lower() if ch.isalnum())[:12] or "strategy"
    return f"codex-{strategy_token}-{market_token}-{side_token}-{uuid.uuid4().hex[:12]}"


class OrderEventTracker:
    def __init__(self, *, max_events: int = 500):
        self.max_events = max(int(max_events), 50)
        self._condition = threading.Condition()
        self._events_by_uuid: dict[str, dict[str, Any]] = {}
        self._events_by_identifier: dict[str, dict[str, Any]] = {}
        self._event_order: list[tuple[str, str]] = []

    def push(self, event: Mapping[str, Any] | None) -> dict[str, Any]:
        normalized = normalize_ws_order_event(event)
        event_uuid = str(normalized.get("uuid") or "").strip()
        event_identifier = str(normalized.get("identifier") or "").strip()
        if not event_uuid and not event_identifier:
            return normalized
        with self._condition:
            if event_uuid:
                self._events_by_uuid[event_uuid] = normalized
                self._event_order.append(("uuid", event_uuid))
            if event_identifier:
                self._events_by_identifier[event_identifier] = normalized
                self._event_order.append(("identifier", event_identifier))
            while len(self._event_order) > self.max_events:
                kind, value = self._event_order.pop(0)
                if kind == "uuid":
                    self._events_by_uuid.pop(value, None)
                else:
                    self._events_by_identifier.pop(value, None)
            self._condition.notify_all()
        return normalized

    def latest(self, *, uuid: str | None = None, identifier: str | None = None) -> dict[str, Any] | None:
        key_uuid = str(uuid or "").strip()
        key_identifier = str(identifier or "").strip()
        with self._condition:
            if key_uuid and key_uuid in self._events_by_uuid:
                return dict(self._events_by_uuid[key_uuid])
            if key_identifier and key_identifier in self._events_by_identifier:
                return dict(self._events_by_identifier[key_identifier])
        return None

    def wait_for_order(
        self,
        *,
        uuid: str | None = None,
        identifier: str | None = None,
        timeout_seconds: float = 3.0,
    ) -> dict[str, Any] | None:
        key_uuid = str(uuid or "").strip()
        key_identifier = str(identifier or "").strip()
        if not key_uuid and not key_identifier:
            return None
        deadline = time.monotonic() + max(float(timeout_seconds), 0.0)
        latest_event: dict[str, Any] | None = None
        with self._condition:
            while True:
                if key_uuid and key_uuid in self._events_by_uuid:
                    latest_event = dict(self._events_by_uuid[key_uuid])
                elif key_identifier and key_identifier in self._events_by_identifier:
                    latest_event = dict(self._events_by_identifier[key_identifier])
                state = str((latest_event or {}).get("state") or "").lower()
                remaining = _to_float((latest_event or {}).get("remaining_volume"))
                executed = _to_float((latest_event or {}).get("executed_volume"))
                if latest_event and (state in TERMINAL_STATES or executed > 0 or remaining > 0):
                    return latest_event
                remaining_wait = deadline - time.monotonic()
                if remaining_wait <= 0:
                    return latest_event
                self._condition.wait(timeout=min(remaining_wait, 0.25))


def normalize_ws_order_event(event: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(event or {})
    side = str(raw.get("side") or raw.get("ask_bid") or "").lower()
    if side == "ask":
        side = "ask"
    elif side == "bid":
        side = "bid"
    state = str(raw.get("state") or "").lower()
    if state == "trade":
        state = "wait"
    return {
        "uuid": raw.get("uuid"),
        "identifier": raw.get("identifier"),
        "market": raw.get("market") or raw.get("code"),
        "side": side,
        "state": state,
        "remaining_volume": raw.get("remaining_volume"),
        "executed_volume": raw.get("executed_volume"),
        "executed_funds": raw.get("executed_funds") or raw.get("trade_price"),
        "paid_fee": raw.get("paid_fee"),
        "avg_price": raw.get("avg_price") or raw.get("trade_price"),
        "price": raw.get("price"),
        "volume": raw.get("volume"),
        "raw_event": raw,
    }


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
    side: str | None = None,
    fallback_price: float,
    fallback_cost: float | None = None,
    fallback_qty: float | None = None,
) -> dict[str, Any]:
    raw_order = dict(order or {})
    order_side = str(raw_order.get("side") or side or "").lower()
    executed_qty = _to_float(raw_order.get("executed_volume")) or _to_float(fallback_qty)
    gross_value = _to_float(raw_order.get("executed_funds"))
    if gross_value <= 0:
        gross_value = _to_float(raw_order.get("executed_fund"))
    paid_fee = _to_float(raw_order.get("paid_fee"))
    market_price = _to_float(raw_order.get("avg_price")) or fallback_price
    can_estimate_fill = bool(raw_order.get("simulate")) or not raw_order or not str(raw_order.get("state") or "").strip()

    if executed_qty > 0 and gross_value > 0 and market_price <= 0:
        market_price = gross_value / executed_qty

    if executed_qty <= 0 and can_estimate_fill and order_side in {"bid", "buy"} and fallback_cost and fallback_price > 0:
        executed_qty = _to_float(fallback_cost) / fallback_price
    elif executed_qty <= 0 and can_estimate_fill and order_side in {"ask", "sell"} and fallback_qty:
        executed_qty = _to_float(fallback_qty)

    if gross_value <= 0 and can_estimate_fill and fallback_cost:
        gross_value = _to_float(fallback_cost)
    elif gross_value <= 0 and order_side in {"ask", "sell"} and executed_qty > 0 and fallback_price > 0:
        gross_value = executed_qty * fallback_price

    if order_side in {"bid", "buy"}:
        net_value = gross_value + paid_fee if gross_value > 0 else _to_float(fallback_cost)
    elif order_side in {"ask", "sell"}:
        net_value = max(gross_value - paid_fee, 0.0) if gross_value > 0 else _to_float(fallback_cost)
    else:
        net_value = gross_value or _to_float(fallback_cost)

    effective_price = market_price
    if executed_qty > 0 and net_value > 0:
        effective_price = net_value / executed_qty

    if executed_qty > 0 and gross_value <= 0 and effective_price > 0:
        gross_value = effective_price * executed_qty
        net_value = gross_value

    return {
        "order_uuid": raw_order.get("uuid"),
        "order_state": raw_order.get("state"),
        "side": order_side,
        "price": effective_price,
        "market_price": market_price,
        "qty": executed_qty,
        "value": gross_value,
        "net_value": net_value,
        "cost": net_value,
        "paid_fee": paid_fee,
        "remaining_volume": _to_float(raw_order.get("remaining_volume")),
        "raw_order": raw_order,
    }


def resolve_submitted_order(
    api,
    order_result: Mapping[str, Any] | None,
    *,
    live_orders: bool,
    event_tracker: OrderEventTracker | None = None,
    side: str | None = None,
    fallback_price: float,
    fallback_cost: float | None = None,
    fallback_qty: float | None = None,
    timeout_seconds: float = 3.0,
    poll_interval: float = 0.4,
) -> dict[str, Any]:
    raw_result = dict(order_result or {})
    resolved_order = raw_result
    order_uuid = raw_result.get("uuid")
    order_identifier = raw_result.get("identifier")
    resolved_side = str(raw_result.get("side") or side or "").lower()

    if live_orders and raw_result.get("error") and order_identifier:
        try:
            looked_up = None
            if event_tracker is not None:
                looked_up = event_tracker.wait_for_order(
                    identifier=str(order_identifier),
                    timeout_seconds=timeout_seconds,
                )
            if not looked_up:
                looked_up = wait_for_order_completion(
                    api,
                    identifier=str(order_identifier),
                    timeout_seconds=timeout_seconds,
                    poll_interval=poll_interval,
                )
            if looked_up:
                resolved_order = looked_up
                order_uuid = looked_up.get("uuid")
        except Exception as exc:
            resolved_order = dict(raw_result)
            resolved_order["lookup_error"] = repr(exc)

    if live_orders and (order_uuid or order_identifier):
        try:
            looked_up = None
            if event_tracker is not None:
                looked_up = event_tracker.wait_for_order(
                    uuid=str(order_uuid) if order_uuid else None,
                    identifier=str(order_identifier) if order_identifier else None,
                    timeout_seconds=timeout_seconds,
                )
            if looked_up:
                resolved_order = looked_up
            else:
                resolved_order = (
                    wait_for_order_completion(
                        api,
                        uuid=str(order_uuid) if order_uuid else None,
                        identifier=str(order_identifier) if order_identifier else None,
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
        side=resolved_side,
        fallback_price=fallback_price,
        fallback_cost=fallback_cost,
        fallback_qty=fallback_qty,
    )
    fill["identifier"] = resolved_order.get("identifier") or order_identifier
    order_state = str(fill.get("order_state") or "")
    actual_executed_qty = _to_float(resolved_order.get("executed_volume"))
    actual_remaining = _to_float(resolved_order.get("remaining_volume"))

    if raw_result.get("error") and not (resolved_order.get("uuid") or resolved_order.get("identifier")):
        return {
            "status": "error",
            "error": raw_result.get("error"),
            "status_code": raw_result.get("status_code"),
            "fill": fill,
            "order": raw_result,
        }

    if not live_orders:
        status = "filled"
    elif order_state in {"cancel", "prevented"}:
        status = "cancelled"
    elif order_state == "done" or (actual_executed_qty > 0 and actual_remaining <= 0):
        status = "filled"
    elif raw_result.get("error") and raw_result.get("ambiguous_submission"):
        status = "pending"
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
    fill: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_result = dict(order_result or {})
    fill = dict(fill or {})
    return {
        "market": market,
        "side": side,
        "strategy": strategy,
        "reason": reason,
        "uuid": raw_result.get("uuid"),
        "identifier": raw_result.get("identifier") or fill.get("identifier"),
        "fallback_price": float(fill.get("market_price") or fallback_price),
        "requested_cost": float(requested_cost) if requested_cost is not None else None,
        "requested_qty": float(requested_qty) if requested_qty is not None else None,
        "submitted_at": float(submitted_at or time.time()),
        "filled_qty": _to_float(fill.get("qty")),
        "filled_value": _to_float(fill.get("value")),
        "filled_net_value": _to_float(fill.get("net_value")),
        "filled_fee": _to_float(fill.get("paid_fee")),
    }


def pending_fill_delta(
    pending_order: Mapping[str, Any],
    fill: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    updated = dict(pending_order or {})
    fill = dict(fill or {})

    previous_qty = _to_float(updated.get("filled_qty"))
    previous_value = _to_float(updated.get("filled_value"))
    previous_net_value = _to_float(updated.get("filled_net_value"))
    previous_fee = _to_float(updated.get("filled_fee"))

    total_qty = max(_to_float(fill.get("qty")), previous_qty)
    total_value = max(_to_float(fill.get("value")), previous_value)
    total_net_value = max(_to_float(fill.get("net_value")), previous_net_value)
    total_fee = max(_to_float(fill.get("paid_fee")), previous_fee)

    updated["filled_qty"] = total_qty
    updated["filled_value"] = total_value
    updated["filled_net_value"] = total_net_value
    updated["filled_fee"] = total_fee
    updated["fallback_price"] = float(fill.get("market_price") or updated.get("fallback_price") or 0.0)

    delta_qty = max(total_qty - previous_qty, 0.0)
    delta_value = max(total_value - previous_value, 0.0)
    delta_net_value = max(total_net_value - previous_net_value, 0.0)
    delta_fee = max(total_fee - previous_fee, 0.0)
    delta_market_price = (delta_value / delta_qty) if delta_qty > 0 and delta_value > 0 else _to_float(fill.get("market_price"))
    delta_effective_price = (delta_net_value / delta_qty) if delta_qty > 0 and delta_net_value > 0 else _to_float(fill.get("price"))

    return updated, {
        "qty": delta_qty,
        "value": delta_value,
        "net_value": delta_net_value,
        "paid_fee": delta_fee,
        "market_price": delta_market_price,
        "price": delta_effective_price,
    }

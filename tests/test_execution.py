from __future__ import annotations

from src.execution import OrderEventTracker, build_client_order_identifier, normalize_ws_order_event, resolve_submitted_order


class IdentifierLookupAPI:
    def __init__(self):
        self.calls = 0

    def get_order(self, uuid=None, identifier=None):
        self.calls += 1
        return {
            "uuid": uuid or "order-123",
            "identifier": identifier or "id-123",
            "side": "bid",
            "state": "done",
            "remaining_volume": "0",
            "executed_volume": "100",
            "executed_funds": "10000",
            "paid_fee": "0",
            "avg_price": "100",
        }


def test_build_client_order_identifier_is_unique():
    first = build_client_order_identifier("KRW-BTC", "bid", strategy_name="research_trend")
    second = build_client_order_identifier("KRW-BTC", "bid", strategy_name="research_trend")

    assert first != second
    assert first.startswith("codex-")
    assert "KRWBTC" in first


def test_resolve_submitted_order_recovers_with_identifier_lookup():
    resolution = resolve_submitted_order(
        IdentifierLookupAPI(),
        {
            "error": "timeout",
            "ambiguous_submission": True,
            "identifier": "id-123",
            "side": "bid",
        },
        live_orders=True,
        side="bid",
        fallback_price=100.0,
        fallback_cost=10000.0,
        timeout_seconds=0.01,
        poll_interval=0.0,
    )

    assert resolution["status"] == "filled"
    assert resolution["fill"]["identifier"] == "id-123"
    assert float(resolution["fill"]["qty"]) == 100.0


def test_normalize_ws_order_event_maps_private_payload():
    normalized = normalize_ws_order_event(
        {
            "uuid": "order-1",
            "identifier": "id-1",
            "code": "KRW-BTC",
            "ask_bid": "BID",
            "state": "trade",
            "remaining_volume": "20",
            "executed_volume": "80",
            "executed_funds": "8000",
            "paid_fee": "0",
            "avg_price": "100",
        }
    )

    assert normalized["market"] == "KRW-BTC"
    assert normalized["side"] == "bid"
    assert normalized["state"] == "wait"
    assert normalized["identifier"] == "id-1"


def test_resolve_submitted_order_prefers_my_order_event_tracker():
    api = IdentifierLookupAPI()
    tracker = OrderEventTracker()
    tracker.push(
        {
            "uuid": "order-evt-1",
            "identifier": "id-evt-1",
            "market": "KRW-BTC",
            "side": "bid",
            "state": "done",
            "remaining_volume": "0",
            "executed_volume": "100",
            "executed_funds": "10000",
            "paid_fee": "0",
            "avg_price": "100",
        }
    )

    resolution = resolve_submitted_order(
        api,
        {
            "uuid": "order-evt-1",
            "identifier": "id-evt-1",
            "side": "bid",
        },
        live_orders=True,
        event_tracker=tracker,
        side="bid",
        fallback_price=100.0,
        fallback_cost=10000.0,
        timeout_seconds=0.01,
        poll_interval=0.0,
    )

    assert resolution["status"] == "filled"
    assert api.calls == 0
    assert resolution["fill"]["identifier"] == "id-evt-1"

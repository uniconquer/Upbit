from __future__ import annotations

from src.execution import build_client_order_identifier, normalize_ws_order_event, resolve_submitted_order


class IdentifierLookupAPI:
    def get_order(self, uuid=None, identifier=None):
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

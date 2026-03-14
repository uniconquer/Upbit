from __future__ import annotations

from src.exchange_state import pending_orders_from_open_orders, positions_from_accounts, sync_exchange_state


class SyncAPI:
    def accounts(self):
        return [
            {"currency": "KRW", "balance": "1000000", "locked": "0"},
            {"currency": "BTC", "balance": "0.02000000", "locked": "0.00500000", "avg_buy_price": "120.0"},
            {"currency": "XRP", "balance": "0", "locked": "0", "avg_buy_price": "0"},
        ]

    def open_orders(self, market=None, states=None, limit=100, order_by="desc"):
        return [
            {
                "uuid": "bid-1",
                "market": "KRW-ETH",
                "side": "bid",
                "ord_type": "limit",
                "state": "wait",
                "price": "80.0",
                "volume": "2.0",
                "remaining_volume": "1.5",
                "locked": "120.0",
            },
            {
                "uuid": "ask-1",
                "market": "KRW-BTC",
                "side": "ask",
                "ord_type": "limit",
                "state": "wait",
                "price": "130.0",
                "volume": "0.00500000",
                "remaining_volume": "0.00500000",
                "locked": "0.00500000",
            },
        ]


class EmptyOrdersAPI(SyncAPI):
    def open_orders(self, market=None, states=None, limit=100, order_by="desc"):
        return []


def test_positions_from_accounts_builds_local_positions():
    positions = positions_from_accounts(
        SyncAPI().accounts(),
        existing_positions={"KRW-BTC": {"strategy": "flux_trend", "opened_at": 123.0}},
    )
    assert positions["KRW-BTC"]["qty"] == 0.025
    assert positions["KRW-BTC"]["entry"] == 120.0
    assert positions["KRW-BTC"]["strategy"] == "flux_trend"


def test_positions_from_accounts_skips_excluded_markets():
    positions = positions_from_accounts(
        SyncAPI().accounts(),
        excluded_markets=["KRW-BTC"],
    )

    assert "KRW-BTC" not in positions


def test_pending_orders_from_open_orders_maps_bid_and_ask():
    pending = pending_orders_from_open_orders(
        SyncAPI().open_orders(),
        strategy_name="research_trend",
        existing_positions={"KRW-BTC": {"entry": 120.0, "strategy": "research_trend"}},
    )
    assert pending["KRW-ETH"]["requested_cost"] == 120.0
    assert pending["KRW-BTC"]["requested_qty"] == 0.005
    assert pending["KRW-BTC"]["side"] == "ask"


def test_pending_orders_from_open_orders_skips_excluded_markets():
    pending = pending_orders_from_open_orders(
        SyncAPI().open_orders(),
        strategy_name="research_trend",
        excluded_markets=["KRW-BTC", "ETH"],
    )

    assert pending == {}


def test_sync_exchange_state_builds_positions_pending_and_signal_state():
    result = sync_exchange_state(
        SyncAPI(),
        strategy_name="research_trend",
    )
    assert result["synced"]
    assert "KRW-BTC" in result["positions"]
    assert "KRW-ETH" in result["pending_orders"]
    assert result["last_signal_state"]["KRW-BTC"]["sig"] == "SELL_PENDING"
    assert result["last_signal_state"]["KRW-ETH"]["sig"] == "BUY_PENDING"


def test_sync_exchange_state_skips_excluded_positions_and_orders():
    result = sync_exchange_state(
        SyncAPI(),
        strategy_name="research_trend",
        excluded_markets=["KRW-BTC", "ETH"],
    )

    assert result["synced"]
    assert result["positions"] == {}
    assert result["pending_orders"] == {}
    assert result["last_signal_state"] == {}


def test_sync_exchange_state_clears_stale_pending_orders_when_exchange_is_empty():
    result = sync_exchange_state(
        EmptyOrdersAPI(),
        strategy_name="research_trend",
        existing_pending_orders={
            "KRW-BTC": {
                "uuid": "stale-order",
                "side": "ask",
                "strategy": "research_trend",
            }
        },
    )

    assert result["synced"]
    assert result["pending_orders"] == {}

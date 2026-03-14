from __future__ import annotations

from types import MethodType

import pandas as pd

from src.execution import resolve_submitted_order
from src.kill_switch import save_kill_switch
from src.mr_worker import MRMonitor, fetch_top_markets, normalize_market_codes
from src.runtime_store import load_runtime_state


class FakeAPI:
    def markets(self):
        return [
            {"market": "KRW-BTC"},
            {"market": "KRW-ETH"},
            {"market": "KRW-USDT"},
        ]

    def tickers(self, markets):
        turnover_map = {
            "KRW-BTC": 200,
            "KRW-ETH": 100,
            "KRW-USDT": 300,
        }
        return [
            {
                "market": market,
                "acc_trade_price_24h": turnover_map[market],
                "trade_price": turnover_map[market] * 10,
            }
            for market in markets
        ]

    def create_order(self, *args, **kwargs):
        return {"simulate": True}

    def accounts(self):
        return []

    def open_orders(self, market=None, states=None, limit=100, order_by="desc"):
        return []


class PendingLiveAPI(FakeAPI):
    def __init__(self):
        self.order_reads = 0

    def create_order(self, *args, **kwargs):
        return {"uuid": "order-1"}

    def accounts(self):
        if self.order_reads >= 2:
            return [
                {"currency": "KRW", "balance": "990000", "locked": "0"},
                {"currency": "BTC", "balance": "100", "locked": "0", "avg_buy_price": "100"},
            ]
        return []

    def get_order(self, uuid=None, identifier=None):
        self.order_reads += 1
        if self.order_reads == 1:
            return {
                "uuid": uuid or "order-1",
                "state": "wait",
                "remaining_volume": "100",
                "executed_volume": "0",
                "executed_fund": "0",
            }
        return {
            "uuid": uuid or "order-1",
            "state": "done",
            "remaining_volume": "0",
            "executed_volume": "100",
            "executed_fund": "10000",
        }


class PartialBuyLiveAPI(FakeAPI):
    def __init__(self):
        self.order_reads = 0

    def create_order(self, *args, **kwargs):
        return {"uuid": "order-partial-buy"}

    def accounts(self):
        if self.order_reads <= 1:
            return [
                {"currency": "KRW", "balance": "1000000", "locked": "0"},
                {"currency": "BTC", "balance": "40", "locked": "0", "avg_buy_price": "100"},
            ]
        return [
            {"currency": "KRW", "balance": "1000000", "locked": "0"},
            {"currency": "BTC", "balance": "100", "locked": "0", "avg_buy_price": "100"},
        ]

    def get_order(self, uuid=None, identifier=None):
        self.order_reads += 1
        if self.order_reads == 1:
            return {
                "uuid": uuid or "order-partial-buy",
                "side": "bid",
                "state": "wait",
                "remaining_volume": "60",
                "executed_volume": "40",
                "executed_funds": "4000",
                "paid_fee": "0",
                "avg_price": "100",
            }
        return {
            "uuid": uuid or "order-partial-buy",
            "side": "bid",
            "state": "done",
            "remaining_volume": "0",
            "executed_volume": "100",
            "executed_funds": "10000",
            "paid_fee": "0",
            "avg_price": "100",
        }


class PartialSellLiveAPI(FakeAPI):
    def __init__(self):
        self.order_reads = 0

    def create_order(self, *args, **kwargs):
        return {"uuid": "order-partial-sell"}

    def accounts(self):
        if self.order_reads <= 1:
            return [
                {"currency": "KRW", "balance": "1000000", "locked": "0"},
                {"currency": "BTC", "balance": "60", "locked": "0", "avg_buy_price": "100"},
            ]
        return [
            {"currency": "KRW", "balance": "1011000", "locked": "0"},
        ]

    def get_order(self, uuid=None, identifier=None):
        self.order_reads += 1
        if self.order_reads == 1:
            return {
                "uuid": uuid or "order-partial-sell",
                "side": "ask",
                "state": "wait",
                "remaining_volume": "60",
                "executed_volume": "40",
                "executed_funds": "4400",
                "paid_fee": "0",
                "avg_price": "110",
            }
        return {
            "uuid": uuid or "order-partial-sell",
            "side": "ask",
            "state": "done",
            "remaining_volume": "0",
            "executed_volume": "100",
            "executed_funds": "11000",
            "paid_fee": "0",
            "avg_price": "110",
        }


class LookupFailureAPI(FakeAPI):
    def get_order(self, uuid=None, identifier=None):
        raise RuntimeError("lookup failed")


class ExchangeSyncAPI(FakeAPI):
    def accounts(self):
        return [
            {"currency": "KRW", "balance": "500000", "locked": "0"},
            {"currency": "BTC", "balance": "0.05000000", "locked": "0.01000000", "avg_buy_price": "100.0"},
        ]

    def open_orders(self, market=None, states=None, limit=100, order_by="desc"):
        return [
            {
                "uuid": "ask-1",
                "market": "KRW-BTC",
                "side": "ask",
                "ord_type": "limit",
                "state": "wait",
                "price": "110.0",
                "volume": "0.01000000",
                "remaining_volume": "0.01000000",
                "locked": "0.01000000",
            }
        ]


class PeriodicExchangeSyncAPI(FakeAPI):
    def __init__(self):
        self.accounts_calls = 0

    def accounts(self):
        self.accounts_calls += 1
        if self.accounts_calls <= 2:
            return [
                {"currency": "KRW", "balance": "500000", "locked": "0"},
                {"currency": "BTC", "balance": "0.05000000", "locked": "0", "avg_buy_price": "100.0"},
            ]
        return [
            {"currency": "KRW", "balance": "500000", "locked": "0"},
            {"currency": "ETH", "balance": "1.50000000", "locked": "0", "avg_buy_price": "200.0"},
        ]


class MyOrderEventAPI(FakeAPI):
    def __init__(self):
        self.order_reads = 0
        self.callback = None
        self.last_identifier = None
        self.filled = False

    def create_order(self, *args, **kwargs):
        self.last_identifier = kwargs.get("identifier")
        return {"uuid": "ws-order-1", "identifier": self.last_identifier, "side": "bid"}

    def get_order(self, uuid=None, identifier=None):
        self.order_reads += 1
        return {
            "uuid": uuid or "ws-order-1",
            "identifier": identifier or self.last_identifier,
            "side": "bid",
            "state": "wait",
            "remaining_volume": "100",
            "executed_volume": "0",
            "executed_funds": "0",
        }

    def accounts(self):
        if self.filled:
            return [
                {"currency": "KRW", "balance": "990000", "locked": "0"},
                {"currency": "BTC", "balance": "100", "locked": "0", "avg_buy_price": "100"},
            ]
        return []

    def stream_my_order(self, on_message, *, markets=None, run_seconds=0, stop_event=None):
        self.callback = on_message

        class _DummyThread:
            def is_alive(self):
                return True

            def join(self, timeout=None):
                return None

        return _DummyThread()


def _signal_frame(*, buy: bool = False, sell: bool = False, close: float = 100.0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": [close - 1, close - 1],
            "high": [close + 1, close + 1],
            "low": [close - 2, close - 2],
            "close": [close, close],
            "volume": [1000, 1100],
            "strategy_score": [0.1, 1.2],
            "buy_signal": [False, buy],
            "sell_signal": [False, sell],
        },
        index=pd.date_range("2026-03-12", periods=2, freq="4h"),
    )
    return frame


def test_fetch_top_markets_excludes_stables():
    api = FakeAPI()
    markets = fetch_top_markets(api, limit=3, exclude_stables=True)
    assert markets == ["KRW-BTC", "KRW-ETH"]


def test_fetch_top_markets_excludes_requested_markets():
    api = FakeAPI()
    markets = fetch_top_markets(
        api,
        limit=3,
        exclude_stables=False,
        excluded_markets=normalize_market_codes("btc, KRW-ETH"),
    )

    assert markets == ["KRW-USDT"]


def test_monitor_sync_ignores_excluded_markets(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = ExchangeSyncAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        live_orders=True,
        excluded_markets=["KRW-BTC"],
    )

    monitor._sync_with_exchange()

    assert monitor.trader.to_state() == {}
    assert monitor.pending_orders == {}


def test_monitor_opens_and_closes_position():
    api = FakeAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
    )

    frames = [_signal_frame(buy=True, close=100.0), _signal_frame(sell=True, close=110.0)]

    def fake_build_frame(self, market):
        return frames.pop(0)

    monitor._build_frame = MethodType(fake_build_frame, monitor)

    opened = monitor.process_market("KRW-BTC")
    assert opened is not None
    assert monitor.trader.has_position("KRW-BTC")
    assert len(monitor.trade_log) == 1

    closed = monitor.process_market("KRW-BTC")
    assert closed is not None
    assert not monitor.trader.has_position("KRW-BTC")
    assert len(monitor.trade_log) == 2
    assert float(monitor.metrics["realized_pnl"]) > 0


def test_monitor_persists_and_rehydrates_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    api = FakeAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        state_name="worker-test",
    )

    frames = [_signal_frame(buy=True, close=100.0)]

    def fake_build_frame(self, market):
        return frames.pop(0)

    monitor._build_frame = MethodType(fake_build_frame, monitor)
    summary = monitor.run_cycle(["KRW-BTC"])
    assert summary["trades"] == 1

    payload = load_runtime_state("worker-test")
    assert "KRW-BTC" in payload["positions"]

    restored = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        state_name="worker-test",
    )
    assert restored.trader.has_position("KRW-BTC")
    assert len(restored.trade_log) == 1


def test_live_pending_order_reconciles_on_next_cycle(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = PendingLiveAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        live_orders=True,
        reconcile_timeout_seconds=0.01,
    )

    frames = [_signal_frame(buy=True, close=100.0)]

    def fake_build_frame(self, market):
        return frames.pop(0)

    monitor._build_frame = MethodType(fake_build_frame, monitor)
    monitor.process_market("KRW-BTC")
    assert "KRW-BTC" in monitor.pending_orders
    assert not monitor.trader.has_position("KRW-BTC")

    monitor.run_cycle([])
    assert not monitor.pending_orders
    assert monitor.trader.has_position("KRW-BTC")


def test_live_partial_buy_fill_applies_only_remaining_delta(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = PartialBuyLiveAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        live_orders=True,
        reconcile_timeout_seconds=0.01,
    )

    frames = [_signal_frame(buy=True, close=100.0)]

    def fake_build_frame(self, market):
        return frames.pop(0)

    monitor._build_frame = MethodType(fake_build_frame, monitor)
    monitor.process_market("KRW-BTC")

    partial_position = monitor.trader.get_position("KRW-BTC")
    assert partial_position is not None
    assert partial_position.qty == 40.0
    assert monitor.last_signal_state["KRW-BTC"]["sig"] == "BUY_PENDING"
    assert len(monitor.trade_log) == 1

    monitor.run_cycle([])

    final_position = monitor.trader.get_position("KRW-BTC")
    assert final_position is not None
    assert final_position.qty == 100.0
    assert not monitor.pending_orders
    assert len(monitor.trade_log) == 2
    assert float(monitor.metrics["daily_buy"]) == 10000.0


def test_live_partial_sell_fill_applies_only_remaining_delta(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = PartialSellLiveAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        live_orders=True,
        reconcile_timeout_seconds=0.01,
    )
    monitor.trader.enter_long(market="KRW-BTC", price=100.0, cost=10000.0, strategy="research_trend", qty=100.0)
    monitor.last_signal_state["KRW-BTC"] = {"sig": "BUY", "entry": 100.0}

    frames = [_signal_frame(sell=True, close=110.0)]

    def fake_build_frame(self, market):
        return frames.pop(0)

    monitor._build_frame = MethodType(fake_build_frame, monitor)
    monitor.process_market("KRW-BTC")

    partial_position = monitor.trader.get_position("KRW-BTC")
    assert partial_position is not None
    assert partial_position.qty == 60.0
    assert monitor.last_signal_state["KRW-BTC"]["sig"] == "SELL_PENDING"
    assert len(monitor.trade_log) == 1

    monitor.run_cycle([])

    assert not monitor.trader.has_position("KRW-BTC")
    assert not monitor.pending_orders
    assert len(monitor.trade_log) == 2
    assert float(monitor.metrics["realized_pnl"]) == 1000.0


def test_live_lookup_failure_stays_pending():
    api = LookupFailureAPI()
    resolution = resolve_submitted_order(
        api,
        {"uuid": "order-1"},
        live_orders=True,
        fallback_price=100.0,
        fallback_cost=10000.0,
        timeout_seconds=0.01,
    )
    assert resolution["status"] == "pending"


def test_live_monitor_syncs_positions_and_open_orders(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = ExchangeSyncAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        live_orders=True,
    )

    summary = monitor.run_cycle([])

    assert summary["open"] == 1
    assert "KRW-BTC" in monitor.pending_orders
    assert monitor.trader.has_position("KRW-BTC")
    assert monitor.last_signal_state["KRW-BTC"]["sig"] == "SELL_PENDING"


def test_live_monitor_can_sync_again_after_initial_success(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = PeriodicExchangeSyncAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        live_orders=True,
    )

    monitor.run_cycle([])
    assert monitor.trader.has_position("KRW-BTC")

    monitor._next_exchange_sync_at = 0.0
    monitor.run_cycle([])

    assert api.accounts_calls >= 3
    assert not monitor.trader.has_position("KRW-BTC")
    assert monitor.trader.has_position("KRW-ETH")


def test_live_monitor_applies_my_order_event_before_polling(monkeypatch):
    monkeypatch.setenv("UPBIT_LIVE", "1")
    api = MyOrderEventAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        live_orders=True,
        reconcile_timeout_seconds=0.01,
    )

    frames = [_signal_frame(buy=True, close=100.0)]

    def fake_build_frame(self, market):
        return frames.pop(0)

    monitor._build_frame = MethodType(fake_build_frame, monitor)
    monitor.run_cycle(["KRW-BTC"])
    assert "KRW-BTC" in monitor.pending_orders
    assert api.callback is not None

    pending = dict(monitor.pending_orders["KRW-BTC"])
    api.filled = True
    api.callback(
        {
            "uuid": "ws-order-1",
            "identifier": pending.get("identifier"),
            "code": "KRW-BTC",
            "ask_bid": "BID",
            "state": "done",
            "remaining_volume": "0",
            "executed_volume": "100",
            "executed_funds": "10000",
            "paid_fee": "0",
            "avg_price": "100",
        }
    )

    monitor.run_cycle([])

    assert not monitor.pending_orders
    assert monitor.trader.has_position("KRW-BTC")
    assert len(monitor.trade_log) == 1


def test_kill_switch_blocks_new_entry_but_allows_exit(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    save_kill_switch("trade-kill-switch", enabled=True, reason="테스트")

    api = FakeAPI()
    monitor = MRMonitor(
        api,
        strategy_name="research_trend",
        risk_limits={"max_trade_krw": 10000},
        max_open=2,
        min_fetch_seconds=0,
        per_request_sleep=0,
        kill_switch_name="trade-kill-switch",
    )

    buy_frames = [_signal_frame(buy=True, close=100.0)]

    def buy_build_frame(self, market):
        return buy_frames.pop(0)

    monitor._build_frame = MethodType(buy_build_frame, monitor)
    monitor.process_market("KRW-BTC")
    assert not monitor.trader.has_position("KRW-BTC")

    monitor.trader.enter_long(market="KRW-BTC", price=100.0, cost=10000.0, strategy="research_trend")
    monitor.last_signal_state["KRW-BTC"] = {"sig": "BUY", "entry": 100.0}
    sell_frames = [_signal_frame(sell=True, close=110.0)]

    def sell_build_frame(self, market):
        return sell_frames.pop(0)

    monitor._build_frame = MethodType(sell_build_frame, monitor)
    monitor.process_market("KRW-BTC")
    assert not monitor.trader.has_position("KRW-BTC")

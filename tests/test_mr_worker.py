from __future__ import annotations

from types import MethodType

import pandas as pd

from src.execution import resolve_submitted_order
from src.mr_worker import MRMonitor, fetch_top_markets
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


class PendingLiveAPI(FakeAPI):
    def __init__(self):
        self.order_reads = 0

    def create_order(self, *args, **kwargs):
        return {"uuid": "order-1"}

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


class LookupFailureAPI(FakeAPI):
    def get_order(self, uuid=None, identifier=None):
        raise RuntimeError("lookup failed")


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

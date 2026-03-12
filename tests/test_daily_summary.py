from __future__ import annotations

from datetime import datetime, timezone

from src.daily_summary import rollover_daily_report


def _utc_ts(year: int, month: int, day: int, hour: int, minute: int = 0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc).timestamp()


def test_rollover_daily_report_builds_message_once():
    metrics = {
        "day_date": "2026-03-12",
        "daily_buy": 10000.0,
        "realized_pnl": 1200.0,
        "unrealized_pnl": 300.0,
        "total_pnl": 1500.0,
    }
    trade_log = [
        {
            "ts": _utc_ts(2026, 3, 12, 3),
            "market": "KRW-BTC",
            "side": "BUY",
            "cost": 10000.0,
            "fee_paid": 5.0,
        },
        {
            "ts": _utc_ts(2026, 3, 12, 6),
            "market": "KRW-BTC",
            "side": "SELL",
            "pnl_value": 1200.0,
            "fee_paid": 5.0,
        },
        {
            "ts": _utc_ts(2026, 3, 13, 1),
            "market": "KRW-ETH",
            "side": "BUY",
            "cost": 20000.0,
            "fee_paid": 10.0,
        },
    ]

    rollover = rollover_daily_report(
        current_day="2026-03-13",
        mode="LIVE",
        metrics=metrics,
        trade_log=trade_log,
        positions={"KRW-BTC": {"qty": 0.1, "entry": 100000000.0, "cost": 10000000.0}},
        pending_orders={"KRW-XRP": {"side": "bid"}},
    )

    assert rollover["report"] is not None
    assert rollover["report"]["trade_count"] == 2
    assert rollover["report"]["sell_count"] == 1
    assert rollover["report"]["fee_paid"] == 10.0
    assert "2026-03-12 \uc77c\uc77c \uc694\uc57d" in str(rollover["message"])

    repeated = rollover_daily_report(
        current_day="2026-03-13",
        mode="LIVE",
        metrics=metrics,
        trade_log=trade_log,
        positions={"KRW-BTC": {"qty": 0.1, "entry": 100000000.0, "cost": 10000000.0}},
        pending_orders={"KRW-XRP": {"side": "bid"}},
        daily_reports=rollover["daily_reports"],
        last_report_day=rollover["last_report_day"],
    )

    assert repeated["report"] is None
    assert repeated["message"] is None

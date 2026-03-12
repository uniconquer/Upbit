"""Daily trade summary helpers for Telegram and runtime history."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


KST = timezone(timedelta(hours=9))


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _day_from_ts(ts: float | int | None) -> str:
    value = float(ts or 0.0)
    if value <= 0:
        return ""
    return datetime.fromtimestamp(value, tz=timezone.utc).astimezone(KST).strftime("%Y-%m-%d")


def current_kst_day(now_ts: float | int | None = None) -> str:
    if now_ts is None:
        current = datetime.now(tz=timezone.utc)
    else:
        current = datetime.fromtimestamp(float(now_ts), tz=timezone.utc)
    return current.astimezone(KST).strftime("%Y-%m-%d")


def trades_for_day(trade_log: list[Mapping[str, Any]], day_date: str) -> list[dict[str, Any]]:
    return [dict(trade) for trade in trade_log if _day_from_ts(trade.get("ts")) == day_date]


def build_daily_report(
    *,
    day_date: str,
    mode: str,
    metrics: Mapping[str, Any],
    trade_log: list[Mapping[str, Any]],
    positions: Mapping[str, Mapping[str, Any]] | None = None,
    pending_orders: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    day_trades = trades_for_day(trade_log, day_date)
    buys = [trade for trade in day_trades if str(trade.get("side") or "").upper() == "BUY"]
    sells = [trade for trade in day_trades if str(trade.get("side") or "").upper() == "SELL"]
    realized = _to_float(metrics.get("realized_pnl"))
    unrealized = _to_float(metrics.get("unrealized_pnl"))
    total = _to_float(metrics.get("total_pnl"))
    fees = sum(_to_float(trade.get("fee_paid")) for trade in day_trades)
    winners = [trade for trade in sells if _to_float(trade.get("pnl_value")) > 0]
    losers = [trade for trade in sells if _to_float(trade.get("pnl_value")) < 0]
    best_trade = max(sells, key=lambda trade: _to_float(trade.get("pnl_value")), default=None)
    worst_trade = min(sells, key=lambda trade: _to_float(trade.get("pnl_value")), default=None)
    win_rate = (len(winners) / len(sells) * 100.0) if sells else 0.0

    return {
        "day_date": day_date,
        "mode": str(mode).upper(),
        "trade_count": len(day_trades),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "win_rate_pct": win_rate,
        "daily_buy": _to_float(metrics.get("daily_buy")),
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
        "total_pnl": total,
        "fee_paid": fees,
        "open_positions": len(positions or {}),
        "pending_orders": len(pending_orders or {}),
        "best_trade": dict(best_trade or {}),
        "worst_trade": dict(worst_trade or {}),
        "winner_count": len(winners),
        "loser_count": len(losers),
    }


def format_daily_report(report: Mapping[str, Any]) -> str:
    mode_label = "\uc2e4\uac70\ub798" if str(report.get("mode") or "").upper() == "LIVE" else "\ubaa8\uc758"
    lines = [
        f"[{mode_label}] {report.get('day_date')} \uc77c\uc77c \uc694\uc57d",
        f"- \uac70\ub798: {int(report.get('trade_count') or 0)}\uac74 (\ub9e4\uc218 {int(report.get('buy_count') or 0)} / \ub9e4\ub3c4 {int(report.get('sell_count') or 0)})",
        f"- \uc77c\uc77c \ub9e4\uc218: {float(report.get('daily_buy') or 0.0):.0f} KRW",
        f"- \uc2e4\ud604 \uc190\uc775: {float(report.get('realized_pnl') or 0.0):+.0f} KRW",
        f"- \ubbf8\uc2e4\ud604 \uc190\uc775: {float(report.get('unrealized_pnl') or 0.0):+.0f} KRW",
        f"- \ud569\uc0b0 \uc190\uc775: {float(report.get('total_pnl') or 0.0):+.0f} KRW",
        f"- \uc218\uc218\ub8cc: {float(report.get('fee_paid') or 0.0):.0f} KRW",
        f"- \uc2b9\ub960: {float(report.get('win_rate_pct') or 0.0):.1f}% (\uc2b9 {int(report.get('winner_count') or 0)} / \ud328 {int(report.get('loser_count') or 0)})",
        f"- \ubcf4\uc720 \ud3ec\uc9c0\uc158: {int(report.get('open_positions') or 0)}\uac1c, \ubbf8\uccb4\uacb0 \uc8fc\ubb38: {int(report.get('pending_orders') or 0)}\uac74",
    ]
    best_trade = dict(report.get("best_trade") or {})
    if best_trade:
        lines.append(
            f"- \ucd5c\uace0 \uc218\uc775: {best_trade.get('market')} {float(best_trade.get('pnl_value') or 0.0):+.0f} KRW"
        )
    worst_trade = dict(report.get("worst_trade") or {})
    if worst_trade:
        lines.append(
            f"- \ucd5c\ub300 \uc190\uc2e4: {worst_trade.get('market')} {float(worst_trade.get('pnl_value') or 0.0):+.0f} KRW"
        )
    return "\n".join(lines)


def _normalize_reports(daily_reports: Mapping[str, Mapping[str, Any]] | None, *, keep_days: int) -> dict[str, dict[str, Any]]:
    reports = {
        str(day): dict(report or {})
        for day, report in (daily_reports or {}).items()
        if str(day).strip()
    }
    if len(reports) <= keep_days:
        return reports
    kept_days = sorted(reports)[-keep_days:]
    return {day: reports[day] for day in kept_days}


def rollover_daily_report(
    *,
    current_day: str,
    mode: str,
    metrics: Mapping[str, Any] | None,
    trade_log: list[Mapping[str, Any]],
    positions: Mapping[str, Mapping[str, Any]] | None = None,
    pending_orders: Mapping[str, Mapping[str, Any]] | None = None,
    daily_reports: Mapping[str, Mapping[str, Any]] | None = None,
    last_report_day: str | None = None,
    keep_days: int = 31,
) -> dict[str, Any]:
    reports = _normalize_reports(daily_reports, keep_days=keep_days)
    previous_day = str((metrics or {}).get("day_date") or "").strip()
    resolved_last_report_day = str(last_report_day or "").strip() or None
    if not previous_day or previous_day == current_day or resolved_last_report_day == previous_day:
        return {
            "report": None,
            "message": None,
            "daily_reports": reports,
            "last_report_day": resolved_last_report_day,
        }

    report = build_daily_report(
        day_date=previous_day,
        mode=mode,
        metrics=metrics or {},
        trade_log=trade_log,
        positions=positions,
        pending_orders=pending_orders,
    )
    reports[previous_day] = report
    return {
        "report": report,
        "message": format_daily_report(report),
        "daily_reports": _normalize_reports(reports, keep_days=keep_days),
        "last_report_day": previous_day,
    }

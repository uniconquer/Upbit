"""Human-friendly Korean notification text for trading workflows."""

from __future__ import annotations


def _mode_label(mode: str) -> str:
    return "실거래" if str(mode).upper() == "LIVE" else "모의"


def _side_label(side: str) -> str:
    side = str(side).lower()
    if side in {"buy", "bid"}:
        return "매수"
    if side in {"sell", "ask"}:
        return "매도"
    return side.upper()


def translate_blocked_reason(reason: str | None) -> str:
    raw = str(reason or "").strip()
    if not raw:
        return "알 수 없는 사유"
    if raw == "no allocation":
        return "주문 가능 금액이 없습니다"
    if raw.startswith("daily loss ") and raw.endswith(" KRW"):
        return f"일일 손실 한도({raw.removeprefix('daily loss ')})에 도달했습니다"
    if raw.startswith("daily loss ") and raw.endswith("%"):
        return f"일일 손실 비율 한도({raw.removeprefix('daily loss ')})에 도달했습니다"
    if raw.startswith("daily buy "):
        return f"일일 매수 한도({raw.removeprefix('daily buy ')})를 초과합니다"
    if raw.startswith("asset ") and raw.endswith("%"):
        return f"종목 비중 한도({raw.removeprefix('asset ')})를 초과합니다"
    return raw


def start_message(mode: str, *, strategy_name: str, interval: str, markets: int, max_open: int) -> str:
    return (
        f"[{_mode_label(mode)}] 워커 시작: 전략={strategy_name}, 주기={interval}, "
        f"대상마켓={markets}개, 최대보유={max_open}개"
    )


def buy_filled_message(mode: str, *, market: str, price: float, alloc: float, score: float | None = None) -> str:
    score_suffix = f", 점수={score:.2f}" if score is not None else ""
    return f"[{_mode_label(mode)}] {market} 매수 체결: 가격={price:.4f}, 주문금액={alloc:.0f}{score_suffix}"


def sell_filled_message(mode: str, *, market: str, price: float, pnl_pct: float) -> str:
    return f"[{_mode_label(mode)}] {market} 매도 체결: 가격={price:.4f}, 손익={pnl_pct:+.2f}%"


def blocked_max_open_message(mode: str, *, market: str, max_open: int) -> str:
    return f"[{_mode_label(mode)}] {market} 매수 보류: 최대 보유 개수({max_open})에 도달했습니다"


def blocked_risk_message(mode: str, *, market: str, reason: str | None, price: float) -> str:
    return f"[{_mode_label(mode)}] {market} 매수 차단: {translate_blocked_reason(reason)}, 기준가={price:.4f}"


def order_failed_message(mode: str, *, market: str, side: str, error: object) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} 주문 실패: {error}"


def order_cancelled_message(mode: str, *, market: str, side: str) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} 주문이 체결 전 취소되었습니다"


def order_pending_message(mode: str, *, market: str, side: str) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} 주문 접수: 체결 대기 중입니다"


def order_no_fill_message(mode: str, *, market: str, side: str) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} 주문 응답에 체결 수량이 없습니다"


def lookup_failed_message(mode: str, *, market: str, side: str, error: object) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} 주문 조회 실패: {error}"

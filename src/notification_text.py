"""Human-friendly Korean notification text for trading workflows."""

from __future__ import annotations


def _mode_label(mode: str) -> str:
    return "\uc2e4\uac70\ub798" if str(mode).upper() == "LIVE" else "\ubaa8\uc758"


def _side_label(side: str) -> str:
    side = str(side).lower()
    if side in {"buy", "bid"}:
        return "\ub9e4\uc218"
    if side in {"sell", "ask"}:
        return "\ub9e4\ub3c4"
    return side.upper()


def translate_blocked_reason(reason: str | None) -> str:
    raw = str(reason or "").strip()
    if not raw:
        return "\uc54c \uc218 \uc5c6\ub294 \uc0ac\uc720"
    if raw == "no allocation":
        return "\uc8fc\ubb38 \uac00\ub2a5 \uae08\uc561\uc774 \uc5c6\uc2b5\ub2c8\ub2e4"
    if raw.startswith("daily loss ") and raw.endswith(" KRW"):
        return f"\uc77c\uc77c \uc190\uc2e4 \ud55c\ub3c4({raw.removeprefix('daily loss ')})\uc5d0 \ub3c4\ub2ec\ud588\uc2b5\ub2c8\ub2e4"
    if raw.startswith("daily loss ") and raw.endswith("%"):
        return f"\uc77c\uc77c \uc190\uc2e4 \ube44\uc728 \ud55c\ub3c4({raw.removeprefix('daily loss ')})\uc5d0 \ub3c4\ub2ec\ud588\uc2b5\ub2c8\ub2e4"
    if raw.startswith("daily buy "):
        return f"\uc77c\uc77c \ub9e4\uc218 \ud55c\ub3c4({raw.removeprefix('daily buy ')})\ub97c \ucd08\uacfc\ud569\ub2c8\ub2e4"
    if raw.startswith("asset ") and raw.endswith("%"):
        return f"\uc885\ubaa9 \ube44\uc911 \ud55c\ub3c4({raw.removeprefix('asset ')})\ub97c \ucd08\uacfc\ud569\ub2c8\ub2e4"
    return raw


def start_message(mode: str, *, strategy_name: str, interval: str, markets: int, max_open: int) -> str:
    return (
        f"[{_mode_label(mode)}] \uc6cc\ucee4 \uc2dc\uc791: \uc804\ub7b5={strategy_name}, \uc8fc\uae30={interval}, "
        f"\ub300\uc0c1\ub9c8\ucf13={markets}\uac1c, \ucd5c\ub300\ubcf4\uc720={max_open}\uac1c"
    )


def buy_filled_message(
    mode: str,
    *,
    market: str,
    price: float,
    alloc: float,
    score: float | None = None,
    partial: bool = False,
    qty: float | None = None,
) -> str:
    state = "\ubd80\ubd84 \ub9e4\uc218 \uccb4\uacb0" if partial else "\ub9e4\uc218 \uccb4\uacb0"
    qty_suffix = f", \uc218\ub7c9={qty:.8f}" if qty is not None else ""
    score_suffix = f", \uc810\uc218={score:.2f}" if score is not None else ""
    return (
        f"[{_mode_label(mode)}] {market} {state}: "
        f"\uac00\uaca9={price:.4f}, \uc8fc\ubb38\uae08\uc561={alloc:.0f}{qty_suffix}{score_suffix}"
    )


def sell_filled_message(
    mode: str,
    *,
    market: str,
    price: float,
    pnl_pct: float,
    partial: bool = False,
    qty: float | None = None,
) -> str:
    state = "\ubd80\ubd84 \ub9e4\ub3c4 \uccb4\uacb0" if partial else "\ub9e4\ub3c4 \uccb4\uacb0"
    qty_suffix = f", \uc218\ub7c9={qty:.8f}" if qty is not None else ""
    return f"[{_mode_label(mode)}] {market} {state}: \uac00\uaca9={price:.4f}, \uc190\uc775={pnl_pct:+.2f}%{qty_suffix}"


def blocked_max_open_message(mode: str, *, market: str, max_open: int) -> str:
    return f"[{_mode_label(mode)}] {market} \ub9e4\uc218 \ubcf4\ub958: \ucd5c\ub300 \ubcf4\uc720 \uac1c\uc218({max_open})\uc5d0 \ub3c4\ub2ec\ud588\uc2b5\ub2c8\ub2e4"


def blocked_risk_message(mode: str, *, market: str, reason: str | None, price: float) -> str:
    return f"[{_mode_label(mode)}] {market} \ub9e4\uc218 \ucc28\ub2e8: {translate_blocked_reason(reason)}, \uae30\uc900\uac00={price:.4f}"


def order_failed_message(mode: str, *, market: str, side: str, error: object) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} \uc8fc\ubb38 \uc2e4\ud328: {error}"


def order_cancelled_message(mode: str, *, market: str, side: str) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} \uc8fc\ubb38\uc774 \uccb4\uacb0 \uc804 \ucde8\uc18c\ub418\uc5c8\uc2b5\ub2c8\ub2e4"


def order_pending_message(mode: str, *, market: str, side: str) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} \uc8fc\ubb38 \uc811\uc218: \uccb4\uacb0 \ub300\uae30 \uc911\uc785\ub2c8\ub2e4"


def order_no_fill_message(mode: str, *, market: str, side: str) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} \uc8fc\ubb38 \uc751\ub2f5\uc5d0 \uccb4\uacb0 \uc218\ub7c9\uc774 \uc5c6\uc2b5\ub2c8\ub2e4"


def lookup_failed_message(mode: str, *, market: str, side: str, error: object) -> str:
    return f"[{_mode_label(mode)}] {market} {_side_label(side)} \uc8fc\ubb38 \uc870\ud68c \uc2e4\ud328: {error}"


def kill_switch_enabled_message(*, reason: str | None = None, source: str = "runtime") -> str:
    reason_text = f" \uc0ac\uc720: {reason}" if str(reason or "").strip() else ""
    source_text = "\ud658\uacbd\ubcc0\uc218" if source == "env" else "\ub7f0\ud0c0\uc784 \uc124\uc815"
    return (
        "[\uc548\uc804\uc7a5\uce58] \uae34\uae09\uc911\uc9c0\uac00 \ud65c\uc131\ud654\ub418\uc5c8\uc2b5\ub2c8\ub2e4. "
        f"\uc2e0\uaddc \ub9e4\uc218\ub294 \uc911\ub2e8\ud558\uace0 \uc790\ub3d9 \ub9e4\ub3c4\ub9cc \ud5c8\uc6a9\ud569\ub2c8\ub2e4. ({source_text}){reason_text}"
    )


def kill_switch_disabled_message() -> str:
    return "[\uc548\uc804\uc7a5\uce58] \uae34\uae09\uc911\uc9c0\uac00 \ud574\uc81c\ub418\uc5c8\uc2b5\ub2c8\ub2e4. \uc790\ub3d9 \ub9e4\ub9e4\ub97c \ub2e4\uc2dc \ud5c8\uc6a9\ud569\ub2c8\ub2e4."


def kill_switch_block_message(mode: str, *, market: str) -> str:
    return f"[{_mode_label(mode)}] {market} \ub9e4\uc218 \ucc28\ub2e8: \uae34\uae09\uc911\uc9c0\uac00 \ud65c\uc131\ud654\ub418\uc5b4 \uc788\uc2b5\ub2c8\ub2e4"

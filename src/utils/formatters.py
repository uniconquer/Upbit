from __future__ import annotations
import math

__all__ = [
    'fmt_full_number','fmt_coin_amount','fmt_price','fmt_krw','signed_color','colorize_number'
]

def fmt_full_number(v, decimals_if_needed: int = 2):
    try:
        f = float(v)
    except Exception:
        return "-"
    if f.is_integer():
        return f"{int(f):,}"
    return f"{f:,.{decimals_if_needed}f}"

def fmt_coin_amount(v, max_decimals: int = 8) -> str:
    try:
        f = float(v)
    except Exception:
        return "-"
    s = f"{f:.{max_decimals}f}".rstrip('0').rstrip('.')
    return s or '0'

fmt_price = fmt_full_number
fmt_krw = lambda v: fmt_full_number(v, 0)

def signed_color(value: float | int | None) -> str:
    if value is None:
        return 'inherit'
    try:
        v = float(value)
    except Exception:
        return 'inherit'
    if v > 0:
        return 'red'
    if v < 0:
        return 'blue'
    return 'inherit'

def colorize_number(value: float | int | None, *, is_percent: bool = False, decimals: int = 2) -> str:
    if value is None:
        return '-'
    try:
        v = float(value)
    except Exception:
        return str(value)
    if is_percent:
        text = f"{v:.{decimals}f}%"
    else:
        if math.isclose(v, int(v), rel_tol=0, abs_tol=1e-9):
            text = fmt_full_number(int(v), 0)
        else:
            text = fmt_full_number(v, decimals)
    cls = 'pos' if v > 0 else ('neg' if v < 0 else '')
    return f"<span class='signed-number {cls}'>{text}</span>"

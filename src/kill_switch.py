"""Persistent kill-switch controls for automated trading flows."""

from __future__ import annotations

import os
import time
from typing import Any

try:
    from runtime_store import load_runtime_state, save_runtime_state
except ImportError:
    from src.runtime_store import load_runtime_state, save_runtime_state


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_kill_switch(name: str = "trade-kill-switch") -> dict[str, Any]:
    raw = load_runtime_state(name, default={})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "enabled": bool(raw.get("enabled")),
        "reason": str(raw.get("reason") or ""),
        "updated_at": float(raw.get("updated_at") or 0.0),
        "source": str(raw.get("source") or "runtime"),
    }


def save_kill_switch(name: str = "trade-kill-switch", *, enabled: bool, reason: str = "", source: str = "runtime") -> dict[str, Any]:
    state = {
        "enabled": bool(enabled),
        "reason": str(reason).strip(),
        "updated_at": time.time(),
        "source": source,
    }
    save_runtime_state(name, state)
    return state


def effective_kill_switch(name: str = "trade-kill-switch") -> dict[str, Any]:
    env_value = os.getenv("UPBIT_KILL_SWITCH")
    if _truthy(env_value):
        return {
            "enabled": True,
            "reason": str(os.getenv("UPBIT_KILL_SWITCH_REASON") or "환경변수로 긴급중지가 활성화되었습니다"),
            "updated_at": time.time(),
            "source": "env",
        }
    return load_kill_switch(name)

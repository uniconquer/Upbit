"""Helpers for preventing Windows idle sleep while critical loops run."""

from __future__ import annotations

import ctypes
import os
from typing import Callable


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _default_execution_state_setter(flags: int) -> int:
    kernel32 = getattr(ctypes, "windll", None)
    if kernel32 is None:
        return 0
    return int(kernel32.kernel32.SetThreadExecutionState(int(flags)))


class SystemAwakeGuard:
    def __init__(
        self,
        *,
        enabled: bool = True,
        setter: Callable[[int], int] | None = None,
    ) -> None:
        self.enabled = bool(enabled) and os.name == "nt"
        self._setter = setter or _default_execution_state_setter
        self.active = False

    def acquire(self) -> bool:
        if not self.enabled:
            return False
        result = self._setter(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        self.active = bool(result)
        return self.active

    def release(self) -> bool:
        if not self.enabled:
            return False
        result = self._setter(ES_CONTINUOUS)
        self.active = False
        return bool(result)

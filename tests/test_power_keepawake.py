from __future__ import annotations

from src.power_keepawake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, SystemAwakeGuard


def test_system_awake_guard_acquire_and_release_flags():
    calls: list[int] = []

    def fake_setter(flags: int) -> int:
        calls.append(flags)
        return 1

    guard = SystemAwakeGuard(enabled=True, setter=fake_setter)
    guard.enabled = True

    assert guard.acquire() is True
    assert guard.active is True
    assert guard.release() is True
    assert guard.active is False
    assert calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED, ES_CONTINUOUS]


def test_system_awake_guard_noops_when_disabled():
    calls: list[int] = []

    def fake_setter(flags: int) -> int:
        calls.append(flags)
        return 1

    guard = SystemAwakeGuard(enabled=False, setter=fake_setter)
    guard.enabled = False

    assert guard.acquire() is False
    assert guard.release() is False
    assert calls == []

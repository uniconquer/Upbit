from __future__ import annotations

import pytest

from src.runtime_store import load_runtime_state, save_runtime_state
from src.telegram_control import HELP_TEXT, TELEGRAM_CONTROL_PROCESS_STATE, _claim_process, handle_command, run_control_loop


def test_handle_command_returns_help_for_start():
    assert handle_command("/start") == HELP_TEXT


def test_handle_command_status_uses_formatter():
    reply = handle_command(
        "/status",
        status_loader=lambda: {"running": True},
        formatter=lambda snapshot: "STATUS-OK" if snapshot.get("running") else "STATUS-NO",
    )

    assert reply == "STATUS-OK"


def test_handle_command_start_worker_wraps_status():
    reply = handle_command(
        "/start_worker",
        start_worker=lambda: {"running": True},
        formatter=lambda snapshot: "RUNNING" if snapshot.get("running") else "STOPPED",
    )

    assert "워커 시작 요청" in reply
    assert "RUNNING" in reply


def test_handle_command_kill_on_saves_reason():
    captured: dict[str, object] = {}

    def fake_save(name: str, *, enabled: bool, reason: str, source: str = "runtime"):
        captured.update({"name": name, "enabled": enabled, "reason": reason, "source": source})
        return captured

    reply = handle_command(
        "/kill_on 점검",
        config_loader=lambda: {"kill_switch_name": "trade-kill-switch"},
        kill_switch_saver=fake_save,
    )

    assert captured["enabled"] is True
    assert captured["reason"] == "점검"
    assert "점검" in reply


def test_claim_process_rejects_duplicate_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    save_runtime_state(
        TELEGRAM_CONTROL_PROCESS_STATE,
        {
            "pid": 999,
            "running": True,
            "status": "running",
        },
    )
    monkeypatch.setattr("src.telegram_control._process_exists", lambda pid: int(pid or 0) == 999)

    assert _claim_process() is False
    assert load_runtime_state(TELEGRAM_CONTROL_PROCESS_STATE)["pid"] == 999


def test_run_control_loop_holds_system_awake_until_interrupt(monkeypatch):
    calls: list[str] = []

    class FakeGuard:
        def __init__(self, *, enabled: bool = True):
            self.enabled = enabled

        def acquire(self) -> bool:
            calls.append("acquire")
            return True

        def release(self) -> bool:
            calls.append("release")
            return True

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat")
    monkeypatch.setattr("src.telegram_control.load_dotenv", lambda: None)
    monkeypatch.setattr("src.telegram_control.SystemAwakeGuard", FakeGuard)
    monkeypatch.setattr("src.telegram_control.delete_webhook", lambda token: None)
    monkeypatch.setattr("src.telegram_control._claim_process", lambda: True)
    monkeypatch.setattr("src.telegram_control._release_process", lambda: calls.append("released-process"))
    monkeypatch.setattr("src.telegram_control._load_offset", lambda: 0)
    monkeypatch.setattr("src.telegram_control._touch_process", lambda offset: None)

    def fake_get_updates(token: str, *, offset: int, timeout: int = 25):
        raise KeyboardInterrupt()

    monkeypatch.setattr("src.telegram_control.get_updates", fake_get_updates)

    with pytest.raises(KeyboardInterrupt):
        run_control_loop(poll_timeout=1)

    assert calls == ["acquire", "release", "released-process"]

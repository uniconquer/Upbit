from __future__ import annotations

from src.telegram_control import HELP_TEXT, handle_command


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

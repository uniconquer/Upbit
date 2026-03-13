from __future__ import annotations

import json
import subprocess

from src.startup_automation import (
    build_install_command,
    build_remove_command,
    build_startup_task_action,
    format_startup_status_bundle,
    load_startup_task_status,
    task_full_name,
)


def test_build_startup_task_action_contains_expected_entrypoint():
    action = build_startup_task_action("worker", python_executable=r"C:\Python\python.exe")

    assert "powershell.exe" in action
    assert "worker-start" in action
    assert r"C:\Python\python.exe" in action
    assert "src\\main.py" in action


def test_build_install_and_remove_commands_use_expected_task_name():
    install_command = build_install_command("telegram")
    remove_command = build_remove_command("telegram")

    assert install_command[:4] == ["schtasks", "/Create", "/F", "/SC"]
    assert "/TN" in install_command
    assert task_full_name("telegram") in install_command
    assert install_command[-2:] == ["/DELAY", "0000:25"]
    assert remove_command == ["schtasks", "/Delete", "/TN", task_full_name("telegram"), "/F"]


def test_load_startup_task_status_parses_powershell_json():
    def fake_runner(args, check=False, capture_output=True, text=True):
        assert args[0] == "powershell.exe"
        payload = {
            "component": "worker",
            "exists": True,
            "task_name": "ManagedWorker",
            "task_path": r"\Upbit\\",
            "state": "Ready",
            "enabled": True,
            "execute": "powershell.exe",
            "arguments": "-NoProfile -Command worker-start",
            "last_run_time": "/Date(1735714800000)/",
            "next_run_time": "/Date(1735718400000)/",
            "last_task_result": 0,
        }
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(payload), stderr="")

    snapshot = load_startup_task_status("worker", runner=fake_runner)

    assert snapshot["exists"] is True
    assert snapshot["enabled"] is True
    assert snapshot["configured"] is True
    assert snapshot["state"] == "Ready"
    assert snapshot["last_run_time"] > 0
    assert snapshot["next_run_time"] > snapshot["last_run_time"]


def test_format_startup_status_bundle_renders_korean_summary():
    text = format_startup_status_bundle(
        {
            "supported": True,
            "tasks": {
                "worker": {
                    "label": "백그라운드 워커",
                    "exists": True,
                    "enabled": True,
                    "state": "Ready",
                    "configured": True,
                    "trigger_type": "MSFT_TaskLogonTrigger",
                    "trigger_delay": "PT15S",
                    "last_run_time": 0.0,
                    "next_run_time": 0.0,
                },
                "telegram": {
                    "label": "텔레그램 제어 봇",
                    "exists": False,
                    "enabled": False,
                    "state": "없음",
                    "configured": False,
                    "trigger_type": "",
                    "trigger_delay": "",
                    "last_run_time": 0.0,
                    "next_run_time": 0.0,
                },
            },
        }
    )

    assert "[Windows 자동 시작] 상태" in text
    assert "백그라운드 워커" in text
    assert "설치됨" in text
    assert "텔레그램 제어 봇" in text
    assert "없음" in text

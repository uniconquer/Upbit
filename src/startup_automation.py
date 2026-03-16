"""Windows startup automation helpers for the managed worker and Telegram bot."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping


KST = timezone(timedelta(hours=9))
STARTUP_TASK_PATH = "\\Upbit\\"
STARTUP_TASKS: dict[str, dict[str, str]] = {
    "worker": {
        "label": "백그라운드 워커",
        "task_name": "ManagedWorker",
        "description": "사용자 로그인 시 Upbit 백그라운드 워커를 시작합니다.",
        "subcommand": "worker-start",
        "delay": "0000:15",
    },
    "telegram": {
        "label": "텔레그램 제어 봇",
        "task_name": "TelegramControl",
        "description": "사용자 로그인 시 Upbit telegram-control 봇을 시작합니다.",
        "subcommand": "telegram-control",
        "delay": "0000:25",
    },
}


def startup_supported() -> bool:
    return os.name == "nt"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _main_script() -> Path:
    return _repo_root() / "src" / "main.py"


def _startup_folder() -> Path:
    appdata = os.getenv("APPDATA")
    if appdata:
        return Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _task_spec(component: str) -> dict[str, str]:
    key = str(component or "").strip().lower()
    if key not in STARTUP_TASKS:
        raise ValueError(f"unknown startup component: {component}")
    return STARTUP_TASKS[key]


def task_full_name(component: str) -> str:
    spec = _task_spec(component)
    return f"{STARTUP_TASK_PATH}{spec['task_name']}"


def startup_file_path(component: str) -> Path:
    spec = _task_spec(component)
    return _startup_folder() / f"Upbit-{spec['task_name']}.cmd"


def _powershell_quote(value: str) -> str:
    return str(value).replace("'", "''")


def build_startup_task_action(component: str, *, python_executable: str | None = None) -> str:
    spec = _task_spec(component)
    python_path = Path(python_executable or sys.executable).resolve()
    repo_root = _repo_root().resolve()
    main_script = _main_script().resolve()
    command = (
        f"Set-Location '{_powershell_quote(str(repo_root))}'; "
        f"& '{_powershell_quote(str(python_path))}' "
        f"'{_powershell_quote(str(main_script))}' {spec['subcommand']}"
    )
    return subprocess.list2cmdline(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-Command",
            command,
        ]
    )


def build_startup_file_contents(component: str, *, python_executable: str | None = None) -> str:
    return f"@echo off\r\n{build_startup_task_action(component, python_executable=python_executable)}\r\n"


def build_install_command(
    component: str,
    *,
    delay: str | None = None,
    python_executable: str | None = None,
) -> list[str]:
    spec = _task_spec(component)
    return [
        "schtasks",
        "/Create",
        "/F",
        "/SC",
        "ONSTART",
        "/RU",
        "SYSTEM",
        "/RL",
        "HIGHEST",
        "/TN",
        task_full_name(component),
        "/TR",
        build_startup_task_action(component, python_executable=python_executable),
        "/DELAY",
        str(delay or spec["delay"]),
    ]


def build_remove_command(component: str) -> list[str]:
    _task_spec(component)
    return ["schtasks", "/Delete", "/TN", task_full_name(component), "/F"]


def build_run_command(component: str) -> list[str]:
    _task_spec(component)
    return ["schtasks", "/Run", "/TN", task_full_name(component)]


def _run_subprocess(command: list[str], *, runner=subprocess.run) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _powershell_query_script(component: str) -> str:
    spec = _task_spec(component)
    task_name = _powershell_quote(spec["task_name"])
    task_path = _powershell_quote(STARTUP_TASK_PATH)
    return (
        f"$task = Get-ScheduledTask -TaskName '{task_name}' -TaskPath '{task_path}' -ErrorAction SilentlyContinue;"
        "if ($null -eq $task) { '{}' }"
        " else {"
        " $info = $task | Get-ScheduledTaskInfo;"
        " $action = $task.Actions | Select-Object -First 1;"
        " $trigger = $task.Triggers | Select-Object -First 1;"
        f" [PSCustomObject]@{{ component = '{_powershell_quote(component)}';"
        " exists = $true;"
        " task_name = $task.TaskName;"
        " task_path = $task.TaskPath;"
        f" label = '{_powershell_quote(spec['label'])}';"
        " description = $task.Description;"
        " state = [string]$task.State;"
        " enabled = [bool]$task.Settings.Enabled;"
        " hidden = [bool]$task.Settings.Hidden;"
        " execute = $action.Execute;"
        " arguments = $action.Arguments;"
        " working_directory = $action.WorkingDirectory;"
        " user_id = $task.Principal.UserId;"
        " logon_type = [string]$task.Principal.LogonType;"
        " run_level = [string]$task.Principal.RunLevel;"
        " trigger_type = $(if ($null -ne $trigger) { [string]$trigger.CimClass.CimClassName } else { '' });"
        " trigger_delay = $(if ($null -ne $trigger) { [string]$trigger.Delay } else { '' });"
        " last_run_time = $info.LastRunTime;"
        " next_run_time = $info.NextRunTime;"
        " last_task_result = $info.LastTaskResult"
        " } | ConvertTo-Json -Compress -Depth 4 }"
    )


def _parse_ps_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _parse_ps_datetime(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    match = re.match(r"/Date\((?P<ms>-?\d+)\)/", text)
    if match:
        try:
            return int(match.group("ms")) / 1000.0
        except Exception:
            return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _format_kst_timestamp(value: Any) -> str:
    try:
        ts = float(value or 0.0)
    except Exception:
        ts = 0.0
    if ts <= 0:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _startup_file_status(component: str) -> dict[str, Any]:
    spec = _task_spec(component)
    path = startup_file_path(component)
    exists = path.exists()
    arguments = ""
    if exists:
        try:
            arguments = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            arguments = ""
    return {
        "component": component,
        "label": spec["label"],
        "task_name": spec["task_name"],
        "task_path": str(path.parent),
        "exists": exists,
        "enabled": exists,
        "state": "StartupFolder" if exists else "없음",
        "description": spec["description"],
        "execute": str(path) if exists else "",
        "arguments": arguments,
        "working_directory": str(_repo_root()) if exists else "",
        "user_id": os.getenv("USERNAME") or "",
        "logon_type": "InteractiveToken" if exists else "",
        "run_level": "Limited" if exists else "",
        "trigger_type": "StartupFolder" if exists else "",
        "trigger_delay": "",
        "last_run_time": 0.0,
        "next_run_time": 0.0,
        "last_task_result": None,
        "method": "startup-folder" if exists else "",
    }


def _normalized_status(component: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spec = _task_spec(component)
    raw = dict(payload or {})
    expected_fragment = spec["subcommand"]
    arguments = str(raw.get("arguments") or "")
    return {
        "component": component,
        "label": spec["label"],
        "task_name": str(raw.get("task_name") or spec["task_name"]),
        "task_path": str(raw.get("task_path") or STARTUP_TASK_PATH),
        "exists": bool(raw.get("exists")),
        "enabled": bool(raw.get("enabled")),
        "state": str(raw.get("state") or ("없음" if not raw.get("exists") else "-")),
        "description": str(raw.get("description") or spec["description"]),
        "execute": str(raw.get("execute") or ""),
        "arguments": arguments,
        "working_directory": str(raw.get("working_directory") or ""),
        "user_id": str(raw.get("user_id") or ""),
        "logon_type": str(raw.get("logon_type") or ""),
        "run_level": str(raw.get("run_level") or ""),
        "trigger_type": str(raw.get("trigger_type") or ""),
        "trigger_delay": str(raw.get("trigger_delay") or ""),
        "last_run_time": _parse_ps_datetime(raw.get("last_run_time")),
        "next_run_time": _parse_ps_datetime(raw.get("next_run_time")),
        "last_task_result": raw.get("last_task_result"),
        "configured": expected_fragment in arguments,
        "method": str(raw.get("method") or ("scheduled-task" if raw.get("exists") else "")),
    }


def load_startup_task_status(component: str, *, runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported"})
    process = runner(
        ["powershell.exe", "-NoProfile", "-Command", _powershell_query_script(component)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = _parse_ps_json(process.stdout)
    if not payload:
        return _normalized_status(component, _startup_file_status(component))
    return _normalized_status(component, payload)


def load_startup_status_bundle(*, runner=subprocess.run) -> dict[str, Any]:
    tasks = {component: load_startup_task_status(component, runner=runner) for component in STARTUP_TASKS}
    return {"supported": startup_supported(), "tasks": tasks}


def install_startup_task(component: str, *, delay: str | None = None, runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported", "ok": False})
    process = _run_subprocess(build_install_command(component, delay=delay), runner=runner)
    if process.returncode == 0:
        path = startup_file_path(component)
        if path.exists():
            path.unlink()
        snapshot = load_startup_task_status(component, runner=runner)
    else:
        path = startup_file_path(component)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_startup_file_contents(component), encoding="utf-8")
        snapshot = _normalized_status(component, _startup_file_status(component))
    snapshot.update(
        {
            "ok": process.returncode == 0 or bool(snapshot.get("exists")),
            "stdout": str(process.stdout or "").strip(),
            "stderr": str(process.stderr or "").strip(),
        }
    )
    return snapshot


def remove_startup_task(component: str, *, runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported", "ok": False})
    process = _run_subprocess(build_remove_command(component), runner=runner)
    file_path = startup_file_path(component)
    removed_file = False
    if file_path.exists():
        file_path.unlink()
        removed_file = True
    snapshot = load_startup_task_status(component, runner=runner)
    snapshot.update(
        {
            "ok": process.returncode == 0 or removed_file or not snapshot.get("exists"),
            "stdout": str(process.stdout or "").strip(),
            "stderr": str(process.stderr or "").strip(),
        }
    )
    return snapshot


def run_startup_task(component: str, *, runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported", "ok": False})
    snapshot = load_startup_task_status(component, runner=runner)
    if snapshot.get("method") == "startup-folder":
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        process = subprocess.Popen(
            build_startup_task_action(component),
            cwd=str(_repo_root()),
            shell=True,
            creationflags=creationflags,
        )
        snapshot.update({"ok": True, "stdout": "", "stderr": "", "pid": process.pid})
        return snapshot
    process = _run_subprocess(build_run_command(component), runner=runner)
    snapshot = load_startup_task_status(component, runner=runner)
    snapshot.update(
        {
            "ok": process.returncode == 0,
            "stdout": str(process.stdout or "").strip(),
            "stderr": str(process.stderr or "").strip(),
        }
    )
    return snapshot


def format_startup_status_bundle(bundle: Mapping[str, Any]) -> str:
    if not bundle.get("supported"):
        return "[자동 시작] 이 운영체제에서는 지원되지 않습니다."
    lines = ["[Windows 자동 시작] 상태"]
    tasks = dict(bundle.get("tasks") or {})
    for component in ["worker", "telegram"]:
        snapshot = dict(tasks.get(component) or _normalized_status(component))
        installed = "설치됨" if snapshot.get("exists") else "없음"
        enabled = "ON" if snapshot.get("enabled") else "OFF"
        configured = "정상" if snapshot.get("configured") else "확인 필요"
        lines.extend(
            [
                f"- {snapshot.get('label')}: {installed} / 활성 {enabled} / 상태 {snapshot.get('state')}",
                f"  다음 실행: {_format_kst_timestamp(snapshot.get('next_run_time'))} / 마지막 실행: {_format_kst_timestamp(snapshot.get('last_run_time'))}",
                f"  트리거: {snapshot.get('trigger_type') or '-'} / 지연: {snapshot.get('trigger_delay') or '-'} / 구성: {configured}",
            ]
        )
    return "\n".join(lines)

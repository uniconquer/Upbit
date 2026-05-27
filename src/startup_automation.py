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
RESUME_EVENT_QUERY = "*[System[Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1]]"
STARTUP_TASKS: dict[str, dict[str, str]] = {
    "worker": {
        "label": "백그라운드 워커",
        "task_name": "ManagedWorker",
        "resume_task_name": "ManagedWorkerResume",
        "description": "사용자 로그인 시 Upbit 백그라운드 워커를 시작합니다.",
        "resume_description": "Restart the managed worker after Windows resumes from sleep.",
        "subcommand": "worker-start",
        "delay": "0000:15",
    },
    "telegram": {
        "label": "텔레그램 제어 봇",
        "task_name": "TelegramControl",
        "resume_task_name": "TelegramControlResume",
        "description": "사용자 로그인 시 Upbit telegram-control 봇을 시작합니다.",
        "resume_description": "Restart the Telegram control loop after Windows resumes from sleep.",
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


def _task_variant_spec(component: str, trigger: str = "startup") -> dict[str, str]:
    spec = _task_spec(component)
    variant = str(trigger or "startup").strip().lower()
    if variant == "startup":
        return {
            "trigger": "startup",
            "task_name": spec["task_name"],
            "description": spec["description"],
            "delay": spec["delay"],
        }
    if variant == "resume":
        return {
            "trigger": "resume",
            "task_name": spec["resume_task_name"],
            "description": spec["resume_description"],
            "delay": "",
        }
    raise ValueError(f"unknown task trigger: {trigger}")


def task_full_name(component: str, trigger: str = "startup") -> str:
    variant = _task_variant_spec(component, trigger)
    return f"{STARTUP_TASK_PATH}{variant['task_name']}"


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
    trigger: str = "startup",
    delay: str | None = None,
    python_executable: str | None = None,
) -> list[str]:
    variant = _task_variant_spec(component, trigger)
    command = [
        "schtasks",
        "/Create",
        "/F",
        "/RU",
        "SYSTEM",
        "/RL",
        "HIGHEST",
        "/TN",
        task_full_name(component, trigger),
        "/TR",
        build_startup_task_action(component, python_executable=python_executable),
    ]
    if variant["trigger"] == "resume":
        command.extend(
            [
                "/SC",
                "ONEVENT",
                "/EC",
                "System",
                "/MO",
                RESUME_EVENT_QUERY,
            ]
        )
        return command
    command.extend(
        [
            "/SC",
            "ONSTART",
            "/DELAY",
            str(delay or variant["delay"]),
        ]
    )
    return command


def build_install_commands(
    component: str,
    *,
    delay: str | None = None,
    python_executable: str | None = None,
) -> list[list[str]]:
    return [
        build_install_command(component, trigger="startup", delay=delay, python_executable=python_executable),
        build_install_command(component, trigger="resume", python_executable=python_executable),
    ]


def build_remove_command(component: str, trigger: str = "startup") -> list[str]:
    _task_variant_spec(component, trigger)
    return ["schtasks", "/Delete", "/TN", task_full_name(component, trigger), "/F"]


def build_remove_commands(component: str) -> list[list[str]]:
    return [
        build_remove_command(component, "startup"),
        build_remove_command(component, "resume"),
    ]


def build_run_command(component: str, trigger: str = "startup") -> list[str]:
    _task_variant_spec(component, trigger)
    return ["schtasks", "/Run", "/TN", task_full_name(component, trigger)]


def _run_subprocess(command: list[str], *, runner=subprocess.run) -> subprocess.CompletedProcess[str]:
    return runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _powershell_query_script(component: str, trigger: str = "startup") -> str:
    spec = _task_spec(component)
    variant = _task_variant_spec(component, trigger)
    task_name = _powershell_quote(variant["task_name"])
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


def _empty_task_status(component: str, trigger: str = "startup") -> dict[str, Any]:
    spec = _task_spec(component)
    variant = _task_variant_spec(component, trigger)
    return {
        "component": component,
        "label": spec["label"],
        "task_name": variant["task_name"],
        "task_path": STARTUP_TASK_PATH,
        "exists": False,
        "enabled": False,
        "state": "없음",
        "description": variant["description"],
        "execute": "",
        "arguments": "",
        "working_directory": "",
        "user_id": "",
        "logon_type": "",
        "run_level": "",
        "trigger_type": "",
        "trigger_delay": "",
        "last_run_time": 0.0,
        "next_run_time": 0.0,
        "last_task_result": None,
        "configured": False,
        "method": "",
    }


def _normalized_status(component: str, payload: Mapping[str, Any] | None = None, *, trigger: str = "startup") -> dict[str, Any]:
    spec = _task_spec(component)
    variant = _task_variant_spec(component, trigger)
    raw = dict(payload or {})
    expected_fragment = spec["subcommand"]
    arguments = str(raw.get("arguments") or "")
    return {
        "component": component,
        "label": spec["label"],
        "task_name": str(raw.get("task_name") or variant["task_name"]),
        "task_path": str(raw.get("task_path") or STARTUP_TASK_PATH),
        "exists": bool(raw.get("exists")),
        "enabled": bool(raw.get("enabled")),
        "state": str(raw.get("state") or ("없음" if not raw.get("exists") else "-")),
        "description": str(raw.get("description") or variant["description"]),
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


def _load_task_status(component: str, *, trigger: str = "startup", runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported"}, trigger=trigger)
    process = runner(
        ["powershell.exe", "-NoProfile", "-Command", _powershell_query_script(component, trigger)],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = _parse_ps_json(process.stdout)
    if not payload:
        fallback = _startup_file_status(component) if trigger == "startup" else _empty_task_status(component, trigger)
        return _normalized_status(component, fallback, trigger=trigger)
    return _normalized_status(component, payload, trigger=trigger)


def load_startup_task_status(component: str, *, runner=subprocess.run) -> dict[str, Any]:
    startup = _load_task_status(component, trigger="startup", runner=runner)
    resume = _load_task_status(component, trigger="resume", runner=runner)
    startup.update(
        {
            "resume_exists": bool(resume.get("exists")),
            "resume_enabled": bool(resume.get("enabled")),
            "resume_state": str(resume.get("state") or "?놁쓬"),
            "resume_configured": bool(resume.get("configured")),
            "resume_trigger_type": str(resume.get("trigger_type") or ""),
            "resume_trigger_delay": str(resume.get("trigger_delay") or ""),
            "resume_last_run_time": float(resume.get("last_run_time") or 0.0),
            "resume_next_run_time": float(resume.get("next_run_time") or 0.0),
            "resume_method": str(resume.get("method") or ""),
            "recovery_configured": bool(resume.get("configured")),
        }
    )
    return startup


def load_startup_status_bundle(*, runner=subprocess.run) -> dict[str, Any]:
    tasks = {component: load_startup_task_status(component, runner=runner) for component in STARTUP_TASKS}
    return {"supported": startup_supported(), "tasks": tasks}


def install_startup_task(component: str, *, delay: str | None = None, runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported", "ok": False})
    startup_process = _run_subprocess(build_install_command(component, trigger="startup", delay=delay), runner=runner)
    if startup_process.returncode == 0:
        path = startup_file_path(component)
        if path.exists():
            path.unlink()
    else:
        path = startup_file_path(component)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(build_startup_file_contents(component), encoding="utf-8")
    resume_process = _run_subprocess(build_install_command(component, trigger="resume"), runner=runner)
    snapshot = load_startup_task_status(component, runner=runner)
    snapshot.update(
        {
            "ok": bool(snapshot.get("exists")),
            "resume_ok": resume_process.returncode == 0 and bool(snapshot.get("resume_exists")),
            "stdout": "\n".join(filter(None, [str(startup_process.stdout or "").strip(), str(resume_process.stdout or "").strip()])),
            "stderr": "\n".join(filter(None, [str(startup_process.stderr or "").strip(), str(resume_process.stderr or "").strip()])),
        }
    )
    return snapshot


def remove_startup_task(component: str, *, runner=subprocess.run) -> dict[str, Any]:
    if not startup_supported():
        return _normalized_status(component, {"exists": False, "state": "unsupported", "ok": False})
    startup_process = _run_subprocess(build_remove_command(component, "startup"), runner=runner)
    resume_process = _run_subprocess(build_remove_command(component, "resume"), runner=runner)
    file_path = startup_file_path(component)
    removed_file = False
    if file_path.exists():
        file_path.unlink()
        removed_file = True
    snapshot = load_startup_task_status(component, runner=runner)
    snapshot.update(
        {
            "ok": (
                startup_process.returncode == 0
                or resume_process.returncode == 0
                or removed_file
                or (not snapshot.get("exists") and not snapshot.get("resume_exists"))
            ),
            "stdout": "\n".join(filter(None, [str(startup_process.stdout or "").strip(), str(resume_process.stdout or "").strip()])),
            "stderr": "\n".join(filter(None, [str(startup_process.stderr or "").strip(), str(resume_process.stderr or "").strip()])),
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
        recovery = "설치됨" if snapshot.get("resume_exists") else "없음"
        recovery_enabled = "ON" if snapshot.get("resume_enabled") else "OFF"
        recovery_configured = "정상" if snapshot.get("resume_configured") else "확인 필요"
        lines.extend(
            [
                f"- {snapshot.get('label')}: 시작 {installed} / 활성 {enabled} / 복귀 {recovery} / 복귀 활성 {recovery_enabled}",
                f"  시작 상태: {snapshot.get('state')} / 복귀 상태: {snapshot.get('resume_state') or '없음'}",
                f"  시작 마지막 실행: {_format_kst_timestamp(snapshot.get('last_run_time'))} / 복귀 마지막 실행: {_format_kst_timestamp(snapshot.get('resume_last_run_time'))}",
                f"  시작 트리거: {snapshot.get('trigger_type') or '-'} / 복귀 트리거: {snapshot.get('resume_trigger_type') or '-'}",
                f"  시작 구성: {configured} / 복귀 구성: {recovery_configured}",
            ]
        )
    return "\n".join(lines)

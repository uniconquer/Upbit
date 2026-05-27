"""Telegram polling bot for controlling the managed CLI worker."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable, Mapping

import requests
from dotenv import load_dotenv

try:
    from kill_switch import save_kill_switch
    from power_keepawake import SystemAwakeGuard
    from runtime_store import load_runtime_state, save_runtime_state
    from worker_control import (
        TELEGRAM_CONTROL_OFFSET_STATE,
        format_worker_status,
        load_managed_worker_status,
        load_worker_config,
        restart_managed_worker,
        start_managed_worker,
        stop_managed_worker,
    )
except ImportError:
    from src.kill_switch import save_kill_switch
    from src.power_keepawake import SystemAwakeGuard
    from src.runtime_store import load_runtime_state, save_runtime_state
    from src.worker_control import (
        TELEGRAM_CONTROL_OFFSET_STATE,
        format_worker_status,
        load_managed_worker_status,
        load_worker_config,
        restart_managed_worker,
        start_managed_worker,
        stop_managed_worker,
    )


HELP_TEXT = "\n".join(
    [
        "[텔레그램 제어] 사용 가능한 명령",
        "/help - 명령 목록 보기",
        "/status - 현재 워커 상태 보기",
        "/start_worker - 백그라운드 CLI 워커 시작",
        "/stop_worker - 백그라운드 CLI 워커 중지",
        "/restart_worker - 백그라운드 CLI 워커 재시작",
        "/kill_on [사유] - 신규 매수 긴급중지",
        "/kill_off - 긴급중지 해제",
        "/ping - 봇 연결 확인",
    ]
)
TELEGRAM_CONTROL_PROCESS_STATE = "telegram-control-process"


def _process_exists(pid: Any) -> bool:
    try:
        resolved = int(pid or 0)
    except Exception:
        return False
    if resolved <= 0:
        return False
    if os.name == "nt":
        try:
            import subprocess

            probe = subprocess.run(
                ["tasklist", "/FI", f"PID eq {resolved}", "/FO", "CSV", "/NH"],
                check=False,
                capture_output=True,
                text=True,
            )
            output = (probe.stdout or "").strip()
            return bool(output) and "No tasks are running" not in output and f'"{resolved}"' in output
        except Exception:
            return False
    try:
        os.kill(resolved, 0)
        return True
    except OSError:
        return False
    except Exception:
        return False


def _load_process_state() -> dict[str, Any]:
    raw = load_runtime_state(TELEGRAM_CONTROL_PROCESS_STATE, default={})
    return dict(raw) if isinstance(raw, dict) else {}


def _save_process_state(state: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = dict(state or {})
    save_runtime_state(TELEGRAM_CONTROL_PROCESS_STATE, snapshot)
    return snapshot


def _refresh_process_state() -> dict[str, Any]:
    state = _load_process_state()
    pid = state.get("pid")
    running = _process_exists(pid)
    if running:
        state["running"] = True
        state["status"] = "running"
    else:
        state["running"] = False
        state["status"] = "stopped" if state else "stopped"
        if state.get("pid"):
            state.setdefault("stopped_at", time.time())
    if state:
        _save_process_state(state)
    return state


def _claim_process() -> bool:
    current_pid = os.getpid()
    state = _refresh_process_state()
    existing_pid = int(state.get("pid") or 0) if state.get("pid") else 0
    if state.get("running") and existing_pid and existing_pid != current_pid:
        return False
    _save_process_state(
        {
            **state,
            "pid": current_pid,
            "running": True,
            "status": "running",
            "started_at": float(state.get("started_at") or time.time()),
            "last_seen_at": time.time(),
        }
    )
    return True


def _touch_process(offset: int) -> None:
    current_pid = os.getpid()
    state = _load_process_state()
    if int(state.get("pid") or 0) not in {0, current_pid}:
        return
    _save_process_state(
        {
            **state,
            "pid": current_pid,
            "running": True,
            "status": "running",
            "last_seen_at": time.time(),
            "offset": int(offset),
        }
    )


def _release_process() -> None:
    current_pid = os.getpid()
    state = _load_process_state()
    if int(state.get("pid") or 0) not in {0, current_pid}:
        return
    _save_process_state(
        {
            **state,
            "pid": current_pid,
            "running": False,
            "status": "stopped",
            "stopped_at": time.time(),
        }
    )


def _telegram_api(token: str, method: str, payload: Mapping[str, Any], *, timeout: int = 30) -> dict[str, Any]:
    response = requests.post(
        f"https://api.telegram.org/bot{token}/{method}",
        json=dict(payload),
        timeout=timeout,
    )
    response.raise_for_status()
    body = response.json()
    if not body.get("ok"):
        raise RuntimeError(body)
    return dict(body)


def delete_webhook(token: str) -> None:
    try:
        _telegram_api(token, "deleteWebhook", {"drop_pending_updates": False}, timeout=10)
    except Exception:
        pass


def get_updates(token: str, *, offset: int, timeout: int = 25) -> list[dict[str, Any]]:
    body = _telegram_api(
        token,
        "getUpdates",
        {"offset": int(offset), "timeout": int(timeout), "allowed_updates": ["message"]},
        timeout=timeout + 5,
    )
    return [dict(item) for item in body.get("result") or []]


def send_message(token: str, chat_id: str, text: str) -> None:
    _telegram_api(token, "sendMessage", {"chat_id": chat_id, "text": text[:4000]}, timeout=10)


def _command_parts(text: str) -> tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""
    head, _, tail = raw.partition(" ")
    command = head.split("@", 1)[0].lower()
    return command, tail.strip()


def handle_command(
    text: str,
    *,
    status_loader: Callable[[], Mapping[str, Any]] = load_managed_worker_status,
    formatter: Callable[[Mapping[str, Any]], str] = format_worker_status,
    config_loader: Callable[[], Mapping[str, Any]] = load_worker_config,
    start_worker: Callable[[], Mapping[str, Any]] = start_managed_worker,
    stop_worker: Callable[[], Mapping[str, Any]] = stop_managed_worker,
    restart_worker: Callable[[], Mapping[str, Any]] = restart_managed_worker,
    kill_switch_saver: Callable[..., Mapping[str, Any]] = save_kill_switch,
) -> str | None:
    command, args = _command_parts(text)
    if not command:
        return None
    if command in {"/help", "/start"}:
        return HELP_TEXT
    if command == "/ping":
        return "[텔레그램 제어] 연결 정상입니다."
    if command == "/status":
        return formatter(status_loader())
    if command == "/start_worker":
        return "[텔레그램 제어] 워커 시작 요청을 처리했습니다.\n" + formatter(start_worker())
    if command == "/stop_worker":
        return "[텔레그램 제어] 워커 중지 요청을 처리했습니다.\n" + formatter(stop_worker())
    if command == "/restart_worker":
        return "[텔레그램 제어] 워커 재시작 요청을 처리했습니다.\n" + formatter(restart_worker())
    if command == "/kill_on":
        config = dict(config_loader() or {})
        reason = args or "텔레그램 수동 긴급중지"
        kill_switch_saver(str(config.get("kill_switch_name") or "trade-kill-switch"), enabled=True, reason=reason)
        return f"[안전장치] 긴급중지를 활성화했습니다. 사유: {reason}"
    if command == "/kill_off":
        config = dict(config_loader() or {})
        kill_switch_saver(str(config.get("kill_switch_name") or "trade-kill-switch"), enabled=False, reason="")
        return "[안전장치] 긴급중지를 해제했습니다."
    return HELP_TEXT


def _extract_message(update: Mapping[str, Any]) -> dict[str, Any]:
    message = update.get("message")
    return dict(message) if isinstance(message, dict) else {}


def _load_offset() -> int:
    state = load_runtime_state(TELEGRAM_CONTROL_OFFSET_STATE, default={})
    if isinstance(state, dict):
        try:
            return int(state.get("offset") or 0)
        except Exception:
            return 0
    return 0


def _save_offset(offset: int) -> None:
    save_runtime_state(
        TELEGRAM_CONTROL_OFFSET_STATE,
        {
            "offset": int(offset),
            "updated_at": time.time(),
        },
    )


def run_control_loop(*, poll_timeout: int = 25) -> None:
    load_dotenv()
    token = str(os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    allowed_chat_id = str(os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not allowed_chat_id:
        raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required.")
    if not _claim_process():
        print("[telegram-control] already running")
        return

    awake_guard = SystemAwakeGuard(enabled=True)
    delete_webhook(token)
    offset = _load_offset()
    print("[telegram-control] polling started")
    awake_guard.acquire()
    try:
        while True:
            try:
                _touch_process(offset)
                updates = get_updates(token, offset=offset, timeout=poll_timeout)
                for update in updates:
                    update_id = int(update.get("update_id") or 0)
                    if update_id >= offset:
                        offset = update_id + 1
                        _save_offset(offset)
                    message = _extract_message(update)
                    if not message:
                        continue
                    chat = dict(message.get("chat") or {})
                    chat_id = str(chat.get("id") or "").strip()
                    if chat_id != allowed_chat_id:
                        continue
                    text = str(message.get("text") or "").strip()
                    reply = handle_command(text)
                    if reply:
                        send_message(token, allowed_chat_id, reply)
                _touch_process(offset)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[telegram-control] poll error: {exc}")
                time.sleep(3.0)
    finally:
        awake_guard.release()
        _release_process()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram polling controller for the managed worker")
    parser.add_argument("--poll-timeout", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_control_loop(poll_timeout=args.poll_timeout)


if __name__ == "__main__":
    main()

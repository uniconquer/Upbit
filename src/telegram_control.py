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

    delete_webhook(token)
    offset = _load_offset()
    print("[telegram-control] polling started")
    while True:
        try:
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
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[telegram-control] poll error: {exc}")
            time.sleep(3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram polling controller for the managed worker")
    parser.add_argument("--poll-timeout", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_control_loop(poll_timeout=args.poll_timeout)


if __name__ == "__main__":
    main()

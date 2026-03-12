"""Background worker control helpers for the shared CLI trading worker."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    from kill_switch import effective_kill_switch
    from runtime_store import load_runtime_state, runtime_dir, save_runtime_state
except ImportError:
    from src.kill_switch import effective_kill_switch
    from src.runtime_store import load_runtime_state, runtime_dir, save_runtime_state


KST = timezone(timedelta(hours=9))
MANAGED_WORKER_STATE_NAME = "managed-worker"
MANAGED_WORKER_CONFIG_STATE = "managed-worker-config"
MANAGED_WORKER_PROCESS_STATE = "managed-worker-process"
TELEGRAM_CONTROL_OFFSET_STATE = "telegram-control-offset"


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _format_kst_timestamp(value: Any) -> str:
    try:
        ts = float(value or 0.0)
    except Exception:
        ts = 0.0
    if ts <= 0:
        return "-"
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST")


def _managed_worker_script() -> Path:
    return Path(__file__).resolve().parent / "mr_worker.py"


def managed_worker_log_path(state_name: str = MANAGED_WORKER_STATE_NAME) -> Path:
    return runtime_dir() / f"{state_name}.log"


def coerce_worker_config(raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    raw = dict(raw or {})
    defaults = {
        "strategy": os.getenv("UPBIT_WORKER_STRATEGY", "research_trend"),
        "interval": os.getenv("UPBIT_WORKER_INTERVAL", "minute30"),
        "count": _to_int(os.getenv("UPBIT_WORKER_COUNT"), 240),
        "markets": _to_int(os.getenv("UPBIT_WORKER_MARKETS"), 10),
        "loop_seconds": _to_int(os.getenv("UPBIT_WORKER_LOOP_SECONDS"), 30),
        "max_open": _to_int(os.getenv("UPBIT_WORKER_MAX_OPEN"), 5),
        "min_fetch_seconds": _to_float(os.getenv("UPBIT_WORKER_MIN_FETCH_SECONDS"), 20.0),
        "per_request_sleep": _to_float(os.getenv("UPBIT_WORKER_PER_REQUEST_SLEEP"), 0.12),
        "state_name": os.getenv("UPBIT_WORKER_STATE_NAME", MANAGED_WORKER_STATE_NAME),
        "reconcile_timeout_seconds": _to_float(os.getenv("UPBIT_WORKER_RECONCILE_TIMEOUT_SECONDS"), 3.0),
        "slippage_bps": _to_float(os.getenv("UPBIT_WORKER_SLIPPAGE_BPS"), 3.0),
        "fee": _to_float(os.getenv("UPBIT_WORKER_FEE"), 0.0005),
        "live_orders": _to_bool(os.getenv("UPBIT_WORKER_LIVE_ORDERS"), False),
        "kill_switch_name": os.getenv("UPBIT_WORKER_KILL_SWITCH_NAME", "trade-kill-switch"),
        "max_trade_krw": _to_float(os.getenv("UPBIT_WORKER_MAX_TRADE_KRW"), 50000.0),
        "max_trade_pct": _to_float(os.getenv("UPBIT_WORKER_MAX_TRADE_PCT"), 2.0),
        "per_asset_max_pct": _to_float(os.getenv("UPBIT_WORKER_PER_ASSET_MAX_PCT"), 10.0),
        "daily_buy_limit": _to_float(os.getenv("UPBIT_WORKER_DAILY_BUY_LIMIT"), 200000.0),
        "daily_loss_limit_krw": _to_float(os.getenv("UPBIT_WORKER_DAILY_LOSS_LIMIT_KRW"), 30000.0),
        "daily_loss_limit_pct": _to_float(os.getenv("UPBIT_WORKER_DAILY_LOSS_LIMIT_PCT"), 3.0),
        "include_unrealized_loss": _to_bool(os.getenv("UPBIT_WORKER_INCLUDE_UNREALIZED_LOSS"), False),
        "fast_ema": _to_int(os.getenv("UPBIT_WORKER_FAST_EMA"), 21),
        "slow_ema": _to_int(os.getenv("UPBIT_WORKER_SLOW_EMA"), 55),
        "breakout_window": _to_int(os.getenv("UPBIT_WORKER_BREAKOUT_WINDOW"), 20),
        "exit_window": _to_int(os.getenv("UPBIT_WORKER_EXIT_WINDOW"), 10),
        "atr_window": _to_int(os.getenv("UPBIT_WORKER_ATR_WINDOW"), 14),
        "atr_mult": _to_float(os.getenv("UPBIT_WORKER_ATR_MULT"), 2.5),
        "adx_window": _to_int(os.getenv("UPBIT_WORKER_ADX_WINDOW"), 14),
        "adx_threshold": _to_float(os.getenv("UPBIT_WORKER_ADX_THRESHOLD"), 18.0),
        "momentum_window": _to_int(os.getenv("UPBIT_WORKER_MOMENTUM_WINDOW"), 20),
        "volume_window": _to_int(os.getenv("UPBIT_WORKER_VOLUME_WINDOW"), 20),
        "volume_threshold": _to_float(os.getenv("UPBIT_WORKER_VOLUME_THRESHOLD"), 0.9),
        "ltf_len": _to_int(os.getenv("UPBIT_WORKER_LTF_LEN"), 20),
        "ltf_mult": _to_float(os.getenv("UPBIT_WORKER_LTF_MULT"), 2.0),
        "htf_len": _to_int(os.getenv("UPBIT_WORKER_HTF_LEN"), 20),
        "htf_mult": _to_float(os.getenv("UPBIT_WORKER_HTF_MULT"), 2.25),
        "htf_rule": os.getenv("UPBIT_WORKER_HTF_RULE", "60T"),
    }

    config = {**defaults, **raw}
    config["strategy"] = str(config.get("strategy") or defaults["strategy"])
    config["interval"] = str(config.get("interval") or defaults["interval"])
    config["count"] = max(_to_int(config.get("count"), defaults["count"]), 50)
    config["markets"] = max(_to_int(config.get("markets"), defaults["markets"]), 1)
    config["loop_seconds"] = max(_to_int(config.get("loop_seconds"), defaults["loop_seconds"]), 5)
    config["max_open"] = max(_to_int(config.get("max_open"), defaults["max_open"]), 1)
    config["min_fetch_seconds"] = max(_to_float(config.get("min_fetch_seconds"), defaults["min_fetch_seconds"]), 0.0)
    config["per_request_sleep"] = max(_to_float(config.get("per_request_sleep"), defaults["per_request_sleep"]), 0.0)
    config["state_name"] = str(config.get("state_name") or defaults["state_name"])
    config["reconcile_timeout_seconds"] = max(
        _to_float(config.get("reconcile_timeout_seconds"), defaults["reconcile_timeout_seconds"]),
        0.5,
    )
    config["slippage_bps"] = max(_to_float(config.get("slippage_bps"), defaults["slippage_bps"]), 0.0)
    config["fee"] = max(_to_float(config.get("fee"), defaults["fee"]), 0.0)
    config["live_orders"] = _to_bool(config.get("live_orders"), defaults["live_orders"])
    config["kill_switch_name"] = str(config.get("kill_switch_name") or defaults["kill_switch_name"])
    config["max_trade_krw"] = max(_to_float(config.get("max_trade_krw"), defaults["max_trade_krw"]), 0.0)
    config["max_trade_pct"] = max(_to_float(config.get("max_trade_pct"), defaults["max_trade_pct"]), 0.0)
    config["per_asset_max_pct"] = max(_to_float(config.get("per_asset_max_pct"), defaults["per_asset_max_pct"]), 0.0)
    config["daily_buy_limit"] = max(_to_float(config.get("daily_buy_limit"), defaults["daily_buy_limit"]), 0.0)
    config["daily_loss_limit_krw"] = max(
        _to_float(config.get("daily_loss_limit_krw"), defaults["daily_loss_limit_krw"]),
        0.0,
    )
    config["daily_loss_limit_pct"] = max(
        _to_float(config.get("daily_loss_limit_pct"), defaults["daily_loss_limit_pct"]),
        0.0,
    )
    config["include_unrealized_loss"] = _to_bool(
        config.get("include_unrealized_loss"),
        defaults["include_unrealized_loss"],
    )

    integer_fields = [
        "fast_ema",
        "slow_ema",
        "breakout_window",
        "exit_window",
        "atr_window",
        "adx_window",
        "momentum_window",
        "volume_window",
        "ltf_len",
        "htf_len",
    ]
    for field in integer_fields:
        config[field] = max(_to_int(config.get(field), defaults[field]), 1)

    float_fields = {
        "atr_mult": defaults["atr_mult"],
        "adx_threshold": defaults["adx_threshold"],
        "volume_threshold": defaults["volume_threshold"],
        "ltf_mult": defaults["ltf_mult"],
        "htf_mult": defaults["htf_mult"],
    }
    for field, fallback in float_fields.items():
        config[field] = max(_to_float(config.get(field), fallback), 0.0)
    config["htf_rule"] = str(config.get("htf_rule") or defaults["htf_rule"])
    return config


def load_worker_config(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    saved = load_runtime_state(MANAGED_WORKER_CONFIG_STATE, default={})
    merged: dict[str, Any] = {}
    if isinstance(saved, dict):
        merged.update(saved)
    if overrides:
        merged.update(dict(overrides))
    return coerce_worker_config(merged)


def save_worker_config(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = coerce_worker_config(config)
    save_runtime_state(MANAGED_WORKER_CONFIG_STATE, normalized)
    return normalized


def build_worker_command(config: Mapping[str, Any]) -> list[str]:
    cfg = coerce_worker_config(config)
    command = [
        sys.executable,
        str(_managed_worker_script()),
        "--strategy",
        str(cfg["strategy"]),
        "--interval",
        str(cfg["interval"]),
        "--count",
        str(cfg["count"]),
        "--markets",
        str(cfg["markets"]),
        "--loop-seconds",
        str(cfg["loop_seconds"]),
        "--max-open",
        str(cfg["max_open"]),
        "--min-fetch-seconds",
        str(cfg["min_fetch_seconds"]),
        "--per-request-sleep",
        str(cfg["per_request_sleep"]),
        "--state-name",
        str(cfg["state_name"]),
        "--reconcile-timeout-seconds",
        str(cfg["reconcile_timeout_seconds"]),
        "--slippage-bps",
        str(cfg["slippage_bps"]),
        "--fee",
        str(cfg["fee"]),
        "--kill-switch-name",
        str(cfg["kill_switch_name"]),
        "--max-trade-krw",
        str(cfg["max_trade_krw"]),
        "--max-trade-pct",
        str(cfg["max_trade_pct"]),
        "--per-asset-max-pct",
        str(cfg["per_asset_max_pct"]),
        "--daily-buy-limit",
        str(cfg["daily_buy_limit"]),
        "--daily-loss-limit-krw",
        str(cfg["daily_loss_limit_krw"]),
        "--daily-loss-limit-pct",
        str(cfg["daily_loss_limit_pct"]),
    ]
    if cfg["include_unrealized_loss"]:
        command.append("--include-unrealized-loss")
    if cfg["strategy"] == "research_trend":
        command.extend(
            [
                "--fast-ema",
                str(cfg["fast_ema"]),
                "--slow-ema",
                str(cfg["slow_ema"]),
                "--breakout-window",
                str(cfg["breakout_window"]),
                "--exit-window",
                str(cfg["exit_window"]),
                "--atr-window",
                str(cfg["atr_window"]),
                "--atr-mult",
                str(cfg["atr_mult"]),
                "--adx-window",
                str(cfg["adx_window"]),
                "--adx-threshold",
                str(cfg["adx_threshold"]),
                "--momentum-window",
                str(cfg["momentum_window"]),
                "--volume-window",
                str(cfg["volume_window"]),
                "--volume-threshold",
                str(cfg["volume_threshold"]),
            ]
        )
    else:
        command.extend(
            [
                "--ltf-len",
                str(cfg["ltf_len"]),
                "--ltf-mult",
                str(cfg["ltf_mult"]),
                "--htf-len",
                str(cfg["htf_len"]),
                "--htf-mult",
                str(cfg["htf_mult"]),
                "--htf-rule",
                str(cfg["htf_rule"]),
            ]
        )
    if cfg["live_orders"]:
        command.append("--live-orders")
    return command


def _process_exists(pid: Any) -> bool:
    try:
        resolved = int(pid or 0)
    except Exception:
        return False
    if resolved <= 0:
        return False
    if os.name == "nt":
        try:
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
    raw = load_runtime_state(MANAGED_WORKER_PROCESS_STATE, default={})
    return dict(raw) if isinstance(raw, dict) else {}


def _save_process_state(state: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = dict(state or {})
    save_runtime_state(MANAGED_WORKER_PROCESS_STATE, snapshot)
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


def start_managed_worker(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    current = _refresh_process_state()
    if current.get("running"):
        return load_managed_worker_status()

    normalized = save_worker_config(load_worker_config(config))
    log_path = managed_worker_log_path(str(normalized["state_name"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_worker_command(normalized)
    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    with log_path.open("a", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=log_handle,
            creationflags=creationflags,
        )
    _save_process_state(
        {
            "pid": process.pid,
            "running": True,
            "status": "running",
            "started_at": time.time(),
            "log_path": str(log_path),
            "state_name": normalized["state_name"],
            "command": command,
            "config": normalized,
        }
    )
    return load_managed_worker_status()


def stop_managed_worker() -> dict[str, Any]:
    state = _refresh_process_state()
    pid = state.get("pid")
    if _process_exists(pid):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                capture_output=True,
                text=True,
            )
        else:
            try:
                os.kill(int(pid), 15)
            except Exception:
                pass
        time.sleep(0.3)
    updated = _refresh_process_state()
    updated["stopped_at"] = time.time()
    _save_process_state(updated)
    return load_managed_worker_status()


def restart_managed_worker(config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    stop_managed_worker()
    return start_managed_worker(config)


def read_worker_log_tail(max_lines: int = 40) -> str:
    state = _refresh_process_state()
    log_path = Path(str(state.get("log_path") or managed_worker_log_path()))
    if not log_path.exists():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max(1, int(max_lines)):])


def load_managed_worker_status() -> dict[str, Any]:
    process_state = _refresh_process_state()
    config = load_worker_config(process_state.get("config") if isinstance(process_state.get("config"), dict) else None)
    runtime = load_runtime_state(str(config["state_name"]), default={})
    runtime = dict(runtime) if isinstance(runtime, dict) else {}
    metrics = dict(runtime.get("metrics") or {})
    positions = dict(runtime.get("positions") or {})
    pending_orders = dict(runtime.get("pending_orders") or {})
    effective_mode = "LIVE" if bool(config.get("live_orders")) and os.getenv("UPBIT_LIVE") == "1" else "SIM"
    kill_switch = effective_kill_switch(str(config.get("kill_switch_name") or "trade-kill-switch"))
    return {
        "running": bool(process_state.get("running")),
        "status": str(process_state.get("status") or "stopped"),
        "pid": process_state.get("pid"),
        "mode": effective_mode,
        "requested_live_orders": bool(config.get("live_orders")),
        "config": config,
        "metrics": metrics,
        "positions_count": len(positions),
        "pending_orders_count": len(pending_orders),
        "trade_count": len(runtime.get("trade_log") or []),
        "last_saved_at": float(runtime.get("saved_at") or 0.0),
        "last_daily_report_day": runtime.get("last_daily_report_day"),
        "kill_switch": kill_switch,
        "log_path": str(process_state.get("log_path") or managed_worker_log_path(str(config["state_name"]))),
    }


def format_worker_status(snapshot: Mapping[str, Any]) -> str:
    config = dict(snapshot.get("config") or {})
    metrics = dict(snapshot.get("metrics") or {})
    kill_switch = dict(snapshot.get("kill_switch") or {})
    running_label = "실행 중" if snapshot.get("running") else "중지됨"
    kill_label = "ON" if kill_switch.get("enabled") else "OFF"
    lines = [
        "[CLI 워커] 상태",
        f"- 실행: {running_label}",
        f"- 모드: {snapshot.get('mode')}",
        f"- 전략: {config.get('strategy')} / 마켓 {int(config.get('markets') or 0)}개 / 루프 {int(config.get('loop_seconds') or 0)}초",
        f"- 보유 포지션: {int(snapshot.get('positions_count') or 0)}개 / 미체결: {int(snapshot.get('pending_orders_count') or 0)}건 / 트레이드 로그: {int(snapshot.get('trade_count') or 0)}건",
        f"- 일일 손익: {float(metrics.get('total_pnl') or 0.0):+.0f} KRW / 마지막 저장: {_format_kst_timestamp(snapshot.get('last_saved_at'))}",
        f"- 긴급중지: {kill_label}",
    ]
    reason = str(kill_switch.get("reason") or "").strip()
    if reason:
        lines.append(f"- 긴급중지 사유: {reason}")
    if snapshot.get("running") and snapshot.get("pid"):
        lines.append(f"- PID: {snapshot.get('pid')}")
    return "\n".join(lines)

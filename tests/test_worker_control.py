from __future__ import annotations

from src.runtime_store import save_runtime_state
from src.worker_control import (
    build_worker_command,
    coerce_worker_config,
    format_worker_status,
    load_managed_worker_status,
    save_worker_config,
)


def test_coerce_worker_config_normalizes_values():
    config = coerce_worker_config(
        {
            "markets": "12",
            "loop_seconds": "45",
            "live_orders": "true",
            "strategy": "flux_ema_filter",
            "htf_rule": "120T",
            "use_heikin_ashi": "yes",
            "confirm_window": "10",
        }
    )

    assert config["markets"] == 12
    assert config["loop_seconds"] == 45
    assert config["live_orders"] is True
    assert config["strategy"] == "flux_ema_filter"
    assert config["htf_rule"] == "120T"
    assert config["confirm_window"] == 10
    assert config["use_heikin_ashi"] is True


def test_coerce_worker_config_allows_zero_confirm_window():
    config = coerce_worker_config({"confirm_window": 0})

    assert config["confirm_window"] == 0


def test_build_worker_command_includes_live_and_strategy_args():
    command = build_worker_command(
        {
            "strategy": "research_trend",
            "live_orders": True,
            "markets": 8,
            "loop_seconds": 20,
            "analysis_interval_seconds": 90,
            "fast_ema": 11,
        }
    )

    assert "--live-orders" in command
    assert "--strategy" in command
    assert "research_trend" in command
    assert "--markets" in command
    assert "8" in command
    assert "--analysis-interval-seconds" in command
    assert "90" in command
    assert "--fast-ema" in command
    assert "11" in command


def test_build_worker_command_includes_excluded_markets():
    command = build_worker_command(
        {
            "strategy": "research_trend",
            "exclude_markets": "btc, KRW-ETH, xrp",
        }
    )

    assert command.count("--exclude-market") == 3
    assert "KRW-BTC" in command
    assert "KRW-ETH" in command
    assert "KRW-XRP" in command


def test_build_worker_command_includes_relative_strength_args():
    command = build_worker_command(
        {
            "strategy": "relative_strength_rotation",
            "rs_short_window": 8,
            "rs_mid_window": 24,
            "rs_long_window": 72,
            "trend_ema_window": 55,
            "entry_score": 7.5,
            "exit_score": 1.5,
        }
    )

    assert "relative_strength_rotation" in command
    assert "--rs-short-window" in command
    assert "8" in command
    assert "--rs-mid-window" in command
    assert "24" in command
    assert "--rs-long-window" in command
    assert "72" in command
    assert "--trend-ema-window" in command
    assert "55" in command
    assert "--entry-score" in command
    assert "7.5" in command
    assert "--exit-score" in command
    assert "1.5" in command


def test_build_worker_command_includes_flux_ema_filter_args():
    command = build_worker_command(
        {
            "strategy": "flux_ema_filter",
            "ltf_len": 14,
            "htf_rule": "120T",
            "sensitivity": 4,
            "atr_period": 3,
            "trend_ema_length": 180,
            "confirm_window": 6,
            "use_heikin_ashi": True,
        }
    )

    assert "flux_ema_filter" in command
    assert "--sensitivity" in command
    assert "4" in command
    assert "--atr-period" in command
    assert "3" in command
    assert "--trend-ema-length" in command
    assert "180" in command
    assert "--confirm-window" in command
    assert "6" in command
    assert "--use-heikin-ashi" in command


def test_load_managed_worker_status_prefers_saved_config_when_stopped(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    save_worker_config({"markets": 3, "max_trade_krw": 10000, "exclude_markets": "btc,eth,xrp"})
    save_runtime_state(
        "managed-worker-process",
        {
            "running": False,
            "status": "stopped",
            "pid": 1234,
            "config": {"markets": 10, "max_trade_krw": 50000},
        },
    )

    status = load_managed_worker_status()

    assert status["config"]["markets"] == 3
    assert status["config"]["max_trade_krw"] == 10000
    assert status["config"]["exclude_markets"] == ["KRW-BTC", "KRW-ETH", "KRW-XRP"]


def test_load_managed_worker_status_marks_stale_heartbeat(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    save_worker_config({"loop_seconds": 30, "interval": "minute15"})
    save_runtime_state(
        "managed-worker-process",
        {
            "running": True,
            "status": "running",
            "pid": 4321,
            "config": {"loop_seconds": 30, "interval": "minute15"},
        },
    )
    save_runtime_state("managed-worker", {"saved_at": 1.0, "positions": {}, "pending_orders": {}, "trade_log": []})
    monkeypatch.setattr("src.worker_control._process_exists", lambda pid: True)
    monkeypatch.setattr("src.worker_control.time.time", lambda: 600.0)

    status = load_managed_worker_status()

    assert status["running"] is True
    assert status["status"] == "stale"
    assert status["heartbeat_stale"] is True
    assert int(status["analysis_interval_seconds"]) == 90


def test_format_worker_status_renders_korean_summary():
    text = format_worker_status(
        {
            "running": True,
            "mode": "SIM",
            "positions_count": 2,
            "pending_orders_count": 1,
            "trade_count": 4,
            "last_saved_at": 0.0,
            "heartbeat_age_seconds": 12.0,
            "heartbeat_timeout_seconds": 90.0,
            "analysis_interval_seconds": 90.0,
            "pid": 3210,
            "config": {"strategy": "research_trend", "markets": 10, "loop_seconds": 30},
            "metrics": {"total_pnl": 1234.0},
            "kill_switch": {"enabled": True, "reason": "테스트"},
        }
    )

    assert "[CLI 워커] 상태" in text
    assert "실행 중" in text
    assert "긴급중지: ON" in text
    assert "테스트" in text
    assert "연구형 추세 돌파" in text

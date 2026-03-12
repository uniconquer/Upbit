from __future__ import annotations

from src.worker_control import build_worker_command, coerce_worker_config, format_worker_status


def test_coerce_worker_config_normalizes_values():
    config = coerce_worker_config(
        {
            "markets": "12",
            "loop_seconds": "45",
            "live_orders": "true",
            "strategy": "flux_trend",
            "htf_rule": "120T",
        }
    )

    assert config["markets"] == 12
    assert config["loop_seconds"] == 45
    assert config["live_orders"] is True
    assert config["strategy"] == "flux_trend"
    assert config["htf_rule"] == "120T"


def test_build_worker_command_includes_live_and_strategy_args():
    command = build_worker_command(
        {
            "strategy": "research_trend",
            "live_orders": True,
            "markets": 8,
            "loop_seconds": 20,
            "fast_ema": 11,
        }
    )

    assert "--live-orders" in command
    assert "--strategy" in command
    assert "research_trend" in command
    assert "--markets" in command
    assert "8" in command
    assert "--fast-ema" in command
    assert "11" in command


def test_format_worker_status_renders_korean_summary():
    text = format_worker_status(
        {
            "running": True,
            "mode": "SIM",
            "positions_count": 2,
            "pending_orders_count": 1,
            "trade_count": 4,
            "last_saved_at": 0.0,
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

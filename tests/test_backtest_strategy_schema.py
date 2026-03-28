from __future__ import annotations

from src.backtest_strategy_schema import (
    DOUBLE_BOTTOM_STRATEGIES,
    STRATEGY_CONTROL_SCHEMAS,
    STRATEGY_DETAIL_COLUMNS,
    STRATEGY_SWEEP_RESULT_SCHEMAS,
    STRATEGY_SWEEP_SCHEMAS,
)
from src.strategy_engine import strategy_options


def test_guard_strategy_has_control_and_sweep_schema():
    controls = STRATEGY_CONTROL_SCHEMAS["rsi_trend_guard"]
    sweeps = STRATEGY_SWEEP_SCHEMAS["rsi_trend_guard"]

    assert controls["title"]
    assert len(controls["rows"]) >= 4
    assert len(sweeps["rows"]) >= 3


def test_guard_strategy_has_result_and_detail_schema():
    results = STRATEGY_SWEEP_RESULT_SCHEMAS["rsi_trend_guard"]
    details = STRATEGY_DETAIL_COLUMNS["rsi_trend_guard"]

    assert "trend_fast_ema" in results["ordered"]
    assert "bearish_regime" in details
    assert "trend_filter" in details


def test_double_bottom_strategy_family_includes_guard():
    assert "rsi_bb_double_bottom" in DOUBLE_BOTTOM_STRATEGIES
    assert "rsi_trend_guard" in DOUBLE_BOTTOM_STRATEGIES


def test_schema_covers_strategy_options():
    options = strategy_options(True, True)

    assert set(options).issubset(STRATEGY_CONTROL_SCHEMAS)
    assert set(options).issubset(STRATEGY_SWEEP_SCHEMAS)
    assert set(options).issubset(STRATEGY_SWEEP_RESULT_SCHEMAS)

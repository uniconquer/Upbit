from __future__ import annotations

from src.backtest_strategy_config import (
    BACKTEST_DEFAULT_PARAMS,
    BACKTEST_WIDGET_KEYS,
    normalize_htf_rule,
    strategy_param_summary,
)


def test_backtest_strategy_config_contains_guard_defaults_and_widgets():
    params = BACKTEST_DEFAULT_PARAMS["rsi_trend_guard"]
    widgets = BACKTEST_WIDGET_KEYS["rsi_trend_guard"]

    assert params["trend_fast_ema"] == 13
    assert params["trend_slow_ema"] == 89
    assert params["risk_reward"] == 1.5
    assert widgets["trend_fast_ema"] == "bt_guard_trend_fast_ema"
    assert widgets["bearish_adx_floor"] == "bt_guard_bearish_adx_floor"


def test_normalize_htf_rule_handles_minute_aliases():
    assert normalize_htf_rule("60m") == "60T"
    assert normalize_htf_rule("120MIN") == "120T"
    assert normalize_htf_rule("1D") == "1D"


def test_strategy_param_summary_supports_guard_strategy():
    summary = strategy_param_summary(
        "rsi_trend_guard",
        {
            "rsi_len": 10,
            "oversold": 35.0,
            "trend_fast_ema": 13,
            "trend_slow_ema": 89,
            "bearish_adx_floor": 14.0,
        },
    )

    assert "RSI 10" in summary
    assert "13/89" in summary
    assert "14.0" in summary

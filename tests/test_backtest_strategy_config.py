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


def test_backtest_strategy_config_contains_relative_guard_defaults_and_widgets():
    params = BACKTEST_DEFAULT_PARAMS["relative_strength_guard"]
    widgets = BACKTEST_WIDGET_KEYS["relative_strength_guard"]

    assert params["breakout_window"] == 28
    assert params["entry_score"] == 9.0
    assert params["guard_fast_ema"] == 13
    assert params["guard_slow_ema"] == 144
    assert params["guard_adx_floor"] == 10.0
    assert widgets["guard_fast_ema"] == "bt_rs_guard_fast_ema"
    assert widgets["guard_rs_floor"] == "bt_rs_guard_rs_floor"


def test_backtest_strategy_config_contains_regime_blend_guard_defaults_and_widgets():
    params = BACKTEST_DEFAULT_PARAMS["regime_blend_guard"]
    widgets = BACKTEST_WIDGET_KEYS["regime_blend_guard"]

    assert params["regime_adx_floor"] == 16.0
    assert params["bear_guard_buffer_pct"] == 1.5
    assert params["bear_guard_score_floor"] == -2.0
    assert widgets["bear_guard_buffer_pct"] == "bt_blend_guard_buffer_pct"
    assert widgets["use_macd_filter"] == "bt_blend_guard_use_macd_filter"


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


def test_strategy_param_summary_supports_relative_strength_guard():
    summary = strategy_param_summary(
        "relative_strength_guard",
        {
            "rs_short_window": 10,
            "rs_mid_window": 30,
            "rs_long_window": 90,
            "guard_fast_ema": 21,
            "guard_slow_ema": 144,
            "guard_adx_floor": 14.0,
        },
    )

    assert "RS 10/30/90" in summary
    assert "21/144" in summary
    assert "14.0" in summary


def test_strategy_param_summary_supports_regime_blend_guard():
    summary = strategy_param_summary(
        "regime_blend_guard",
        {
            "regime_adx_floor": 16.0,
            "bear_guard_buffer_pct": 1.5,
            "bear_guard_score_floor": -2.0,
        },
    )

    assert "16.0" in summary
    assert "1.5%" in summary
    assert "-2.0" in summary

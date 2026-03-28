from __future__ import annotations

import streamlit as st


BACKTEST_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
    "research_trend": {
        "fast_ema": 21,
        "slow_ema": 55,
        "breakout_window": 20,
        "exit_window": 10,
        "atr_window": 14,
        "atr_mult": 2.5,
        "adx_window": 14,
        "adx_threshold": 18.0,
        "momentum_window": 20,
        "volume_window": 20,
        "volume_threshold": 0.9,
    },
    "rsi_bb_double_bottom": {
        "rsi_len": 14,
        "oversold": 30.0,
        "bb_len": 20,
        "bb_mult": 2.0,
        "min_down_bars": 2,
        "low_tolerance_pct": 1.0,
        "max_setup_bars": 12,
        "confirm_bars": 4,
        "use_macd_filter": True,
        "macd_lookback": 5,
        "risk_reward": 2.0,
        "stop_buffer_ticks": 2,
    },
    "rsi_trend_guard": {
        "rsi_len": 10,
        "oversold": 35.0,
        "bb_len": 20,
        "bb_mult": 1.5,
        "min_down_bars": 2,
        "low_tolerance_pct": 1.0,
        "max_setup_bars": 6,
        "confirm_bars": 3,
        "use_macd_filter": True,
        "macd_lookback": 5,
        "risk_reward": 1.5,
        "stop_buffer_ticks": 2,
        "trend_fast_ema": 13,
        "trend_slow_ema": 89,
        "trend_buffer_pct": 2.0,
        "bearish_adx_floor": 14.0,
        "adx_window": 14,
    },
    "relative_strength_rotation": {
        "rs_short_window": 10,
        "rs_mid_window": 30,
        "rs_long_window": 90,
        "trend_ema_window": 55,
        "breakout_window": 20,
        "atr_window": 14,
        "atr_mult": 2.2,
        "volume_window": 20,
        "volume_threshold": 0.9,
        "entry_score": 8.0,
        "exit_score": 2.0,
    },
    "flux_trend": {
        "ltf_len": 20,
        "ltf_mult": 2.0,
        "htf_len": 20,
        "htf_mult": 2.25,
        "htf_rule": "60T",
    },
    "flux_ema_filter": {
        "ltf_len": 20,
        "ltf_mult": 2.0,
        "htf_len": 20,
        "htf_mult": 2.25,
        "htf_rule": "60T",
        "sensitivity": 3,
        "atr_period": 2,
        "trend_ema_length": 240,
        "confirm_window": 8,
        "use_heikin_ashi": False,
    },
}

BACKTEST_WIDGET_KEYS: dict[str, dict[str, str]] = {
    "research_trend": {
        "fast_ema": "bt_research_fast_ema",
        "slow_ema": "bt_research_slow_ema",
        "breakout_window": "bt_research_breakout_window",
        "exit_window": "bt_research_exit_window",
        "atr_window": "bt_research_atr_window",
        "atr_mult": "bt_research_atr_mult",
        "adx_window": "bt_research_adx_window",
        "adx_threshold": "bt_research_adx_threshold",
        "momentum_window": "bt_research_momentum_window",
        "volume_window": "bt_research_volume_window",
        "volume_threshold": "bt_research_volume_threshold",
    },
    "rsi_bb_double_bottom": {
        "rsi_len": "bt_db_rsi_len",
        "oversold": "bt_db_oversold",
        "bb_len": "bt_db_bb_len",
        "bb_mult": "bt_db_bb_mult",
        "min_down_bars": "bt_db_min_down_bars",
        "low_tolerance_pct": "bt_db_low_tolerance_pct",
        "max_setup_bars": "bt_db_max_setup_bars",
        "confirm_bars": "bt_db_confirm_bars",
        "use_macd_filter": "bt_db_use_macd_filter",
        "macd_lookback": "bt_db_macd_lookback",
        "risk_reward": "bt_db_risk_reward",
        "stop_buffer_ticks": "bt_db_stop_buffer_ticks",
    },
    "rsi_trend_guard": {
        "rsi_len": "bt_guard_rsi_len",
        "oversold": "bt_guard_oversold",
        "bb_len": "bt_guard_bb_len",
        "bb_mult": "bt_guard_bb_mult",
        "min_down_bars": "bt_guard_min_down_bars",
        "low_tolerance_pct": "bt_guard_low_tolerance_pct",
        "max_setup_bars": "bt_guard_max_setup_bars",
        "confirm_bars": "bt_guard_confirm_bars",
        "use_macd_filter": "bt_guard_use_macd_filter",
        "macd_lookback": "bt_guard_macd_lookback",
        "risk_reward": "bt_guard_risk_reward",
        "stop_buffer_ticks": "bt_guard_stop_buffer_ticks",
        "trend_fast_ema": "bt_guard_trend_fast_ema",
        "trend_slow_ema": "bt_guard_trend_slow_ema",
        "trend_buffer_pct": "bt_guard_trend_buffer_pct",
        "bearish_adx_floor": "bt_guard_bearish_adx_floor",
        "adx_window": "bt_guard_adx_window",
    },
    "relative_strength_rotation": {
        "rs_short_window": "bt_rs_short_window",
        "rs_mid_window": "bt_rs_mid_window",
        "rs_long_window": "bt_rs_long_window",
        "trend_ema_window": "bt_rs_trend_ema_window",
        "breakout_window": "bt_rs_breakout_window",
        "atr_window": "bt_rs_atr_window",
        "atr_mult": "bt_rs_atr_mult",
        "volume_window": "bt_rs_volume_window",
        "volume_threshold": "bt_rs_volume_threshold",
        "entry_score": "bt_rs_entry_score",
        "exit_score": "bt_rs_exit_score",
    },
    "flux_trend": {
        "ltf_len": "bt_flux_ltf_len",
        "ltf_mult": "bt_flux_ltf_mult",
        "htf_len": "bt_flux_htf_len",
        "htf_mult": "bt_flux_htf_mult",
        "htf_rule": "bt_flux_htf_rule",
    },
    "flux_ema_filter": {
        "ltf_len": "bt_flux_ema_ltf_len",
        "ltf_mult": "bt_flux_ema_ltf_mult",
        "htf_len": "bt_flux_ema_htf_len",
        "htf_mult": "bt_flux_ema_htf_mult",
        "htf_rule": "bt_flux_ema_htf_rule",
        "sensitivity": "bt_flux_ema_sensitivity",
        "atr_period": "bt_flux_ema_atr_period",
        "trend_ema_length": "bt_flux_ema_length",
        "confirm_window": "bt_flux_ema_confirm_window",
        "use_heikin_ashi": "bt_flux_ema_heikin_ashi",
    },
}


def normalize_htf_rule(value: str) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.endswith("MIN"):
        return f"{text[:-3]}T"
    if text.endswith("M") and text[:-1].isdigit():
        return f"{text[:-1]}T"
    if text.endswith("T") and text[:-1].isdigit():
        return text
    if text.endswith("D") and text[:-1].isdigit():
        return text
    return text


def htf_rule_to_widget(value: object) -> str:
    normalized = normalize_htf_rule(str(value or ""))
    if normalized.endswith("T") and normalized[:-1].isdigit():
        return f"{normalized[:-1]}m"
    return normalized or "60m"


def strategy_param_summary(strategy_name: str, params: dict[str, object]) -> str:
    if strategy_name == "research_trend":
        return (
            f"EMA {int(params.get('fast_ema', 21))}/{int(params.get('slow_ema', 55))} · "
            f"돌파 {int(params.get('breakout_window', 20))} · ADX {float(params.get('adx_threshold', 18.0)):.1f}"
        )
    if strategy_name == "rsi_bb_double_bottom":
        return (
            f"RSI {int(params.get('rsi_len', 14))} · 과매도 {float(params.get('oversold', 30.0)):.1f} · "
            f"BB {int(params.get('bb_len', 20))}/{float(params.get('bb_mult', 2.0)):.1f} · "
            f"RR {float(params.get('risk_reward', 2.0)):.2f}"
        )
    if strategy_name == "rsi_trend_guard":
        return (
            f"RSI {int(params.get('rsi_len', 10))} · 과매도 {float(params.get('oversold', 35.0)):.1f} · "
            f"가드 EMA {int(params.get('trend_fast_ema', 13))}/{int(params.get('trend_slow_ema', 89))} · "
            f"ADX {float(params.get('bearish_adx_floor', 14.0)):.1f}"
        )
    if strategy_name == "relative_strength_rotation":
        return (
            f"RS {int(params.get('rs_short_window', 10))}/{int(params.get('rs_mid_window', 30))}/{int(params.get('rs_long_window', 90))} · "
            f"EMA {int(params.get('trend_ema_window', 55))} · 진입 {float(params.get('entry_score', 8.0)):.1f}"
        )
    if strategy_name == "flux_trend":
        return (
            f"LTF {int(params.get('ltf_len', 20))}/{float(params.get('ltf_mult', 2.0)):.2f} · "
            f"HTF {params.get('htf_rule', '60T')}"
        )
    return (
        f"LTF {int(params.get('ltf_len', 20))}/{float(params.get('ltf_mult', 2.0)):.2f} · "
        f"HTF {params.get('htf_rule', '60T')} · EMA {int(params.get('trend_ema_length', 240))} · "
        f"민감도 {int(params.get('sensitivity', 3))} · 확인창 {int(params.get('confirm_window', 8))}"
    )


def coerce_param_value(value: object, default: object) -> object:
    if isinstance(default, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"} if value is not None else bool(default)
    if isinstance(default, int):
        try:
            return int(float(value))
        except Exception:
            return int(default)
    if isinstance(default, float):
        try:
            return float(value)
        except Exception:
            return float(default)
    if default == "60T" or str(default).endswith(("T", "D")):
        normalized = normalize_htf_rule(str(value or default))
        return normalized or default
    return value if value is not None else default


def params_for_strategy(strategy_name: str) -> dict[str, object]:
    defaults = dict(BACKTEST_DEFAULT_PARAMS.get(strategy_name, {}))
    widgets = BACKTEST_WIDGET_KEYS.get(strategy_name, {})
    params: dict[str, object] = {}
    for field, default in defaults.items():
        raw_value = st.session_state.get(widgets.get(field, ""), default)
        params[field] = coerce_param_value(raw_value, default)
    return params


def apply_params_to_widgets(strategy_name: str, params: dict[str, object]) -> None:
    widgets = BACKTEST_WIDGET_KEYS.get(strategy_name, {})
    defaults = BACKTEST_DEFAULT_PARAMS.get(strategy_name, {})
    for field, key in widgets.items():
        if not key:
            continue
        default = defaults.get(field)
        value = params.get(field, default)
        if field == "htf_rule":
            st.session_state[key] = htf_rule_to_widget(value)
        else:
            st.session_state[key] = coerce_param_value(value, default)

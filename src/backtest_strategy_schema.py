from __future__ import annotations

STRATEGY_CONTROL_SCHEMAS: dict[str, dict[str, object]] = {
    "research_trend": {
        "title": "연구형 추세 돌파 설정",
        "rows": [
            [
                {"name": "fast_ema", "kind": "int", "label": "빠른 EMA", "min": 5, "max": 100, "step": 1},
                {"name": "slow_ema", "kind": "int", "label": "느린 EMA", "min": 10, "max": 240, "step": 1},
                {"name": "breakout_window", "kind": "int", "label": "돌파 창", "min": 5, "max": 120, "step": 1},
                {"name": "exit_window", "kind": "int", "label": "청산 창", "min": 3, "max": 80, "step": 1},
            ],
            [
                {"name": "atr_window", "kind": "int", "label": "ATR 창", "min": 5, "max": 50, "step": 1},
                {"name": "atr_mult", "kind": "float", "label": "ATR 배수", "min": 1.0, "max": 6.0, "step": 0.1},
                {"name": "adx_window", "kind": "int", "label": "ADX 창", "min": 5, "max": 50, "step": 1},
                {"name": "adx_threshold", "kind": "float", "label": "ADX 기준", "min": 5.0, "max": 40.0, "step": 0.5},
            ],
            [
                {"name": "momentum_window", "kind": "int", "label": "모멘텀 창", "min": 5, "max": 80, "step": 1},
                {"name": "volume_window", "kind": "int", "label": "거래량 창", "min": 5, "max": 80, "step": 1},
                {"name": "volume_threshold", "kind": "float", "label": "거래량 비율", "min": 0.1, "max": 3.0, "step": 0.1},
            ],
        ],
    },
    "rsi_bb_double_bottom": {
        "title": "RSI+BB 더블바텀 롱 설정",
        "rows": [
            [
                {"name": "rsi_len", "kind": "int", "label": "RSI 길이", "min": 2, "max": 50, "step": 1},
                {"name": "oversold", "kind": "float", "label": "과매도 기준", "min": 5.0, "max": 50.0, "step": 0.5},
                {"name": "bb_len", "kind": "int", "label": "BB 길이", "min": 5, "max": 80, "step": 1},
                {"name": "bb_mult", "kind": "float", "label": "BB 배수", "min": 0.5, "max": 5.0, "step": 0.1},
            ],
            [
                {"name": "min_down_bars", "kind": "int", "label": "연속 하락 봉", "min": 1, "max": 10, "step": 1},
                {"name": "low_tolerance_pct", "kind": "float", "label": "저점 허용치 (%)", "min": 0.0, "max": 5.0, "step": 0.1},
                {"name": "max_setup_bars", "kind": "int", "label": "셋업 유지 봉", "min": 3, "max": 40, "step": 1},
                {"name": "confirm_bars", "kind": "int", "label": "확인 대기 봉", "min": 1, "max": 20, "step": 1},
            ],
            [
                {"name": "use_macd_filter", "kind": "bool", "label": "MACD 확인 사용"},
                {"name": "macd_lookback", "kind": "int", "label": "MACD 최근 교차 봉", "min": 1, "max": 20, "step": 1},
                {"name": "risk_reward", "kind": "float", "label": "손익비", "min": 0.5, "max": 5.0, "step": 0.25},
                {"name": "stop_buffer_ticks", "kind": "int", "label": "스탑 버퍼 틱", "min": 0, "max": 20, "step": 1},
            ],
        ],
    },
    "rsi_trend_guard": {
        "title": "RSI 반등 + 추세 가드 설정",
        "rows": [
            [
                {"name": "rsi_len", "kind": "int", "label": "RSI 길이", "min": 2, "max": 50, "step": 1},
                {"name": "oversold", "kind": "float", "label": "과매도 기준", "min": 5.0, "max": 50.0, "step": 0.5},
                {"name": "bb_len", "kind": "int", "label": "BB 길이", "min": 5, "max": 80, "step": 1},
                {"name": "bb_mult", "kind": "float", "label": "BB 배수", "min": 0.5, "max": 5.0, "step": 0.1},
            ],
            [
                {"name": "min_down_bars", "kind": "int", "label": "연속 하락 봉", "min": 1, "max": 10, "step": 1},
                {"name": "low_tolerance_pct", "kind": "float", "label": "저점 허용치 (%)", "min": 0.0, "max": 5.0, "step": 0.1},
                {"name": "max_setup_bars", "kind": "int", "label": "셋업 유지 봉", "min": 3, "max": 40, "step": 1},
                {"name": "confirm_bars", "kind": "int", "label": "확인 대기 봉", "min": 1, "max": 20, "step": 1},
            ],
            [
                {"name": "use_macd_filter", "kind": "bool", "label": "MACD 확인 사용"},
                {"name": "macd_lookback", "kind": "int", "label": "MACD 최근 교차 봉", "min": 1, "max": 20, "step": 1},
                {"name": "risk_reward", "kind": "float", "label": "손익비", "min": 0.5, "max": 5.0, "step": 0.25},
                {"name": "stop_buffer_ticks", "kind": "int", "label": "스탑 버퍼 틱", "min": 0, "max": 20, "step": 1},
            ],
            [
                {"name": "trend_fast_ema", "kind": "int", "label": "가드 빠른 EMA", "min": 5, "max": 100, "step": 1},
                {"name": "trend_slow_ema", "kind": "int", "label": "가드 느린 EMA", "min": 10, "max": 240, "step": 1},
                {"name": "trend_buffer_pct", "kind": "float", "label": "EMA 버퍼 (%)", "min": 0.0, "max": 5.0, "step": 0.1},
                {"name": "bearish_adx_floor", "kind": "float", "label": "약세 ADX 기준", "min": 5.0, "max": 40.0, "step": 0.5},
            ],
            [
                {"name": "adx_window", "kind": "int", "label": "ADX 창", "min": 5, "max": 50, "step": 1},
            ],
        ],
    },
    "relative_strength_rotation": {
        "title": "상대강도 로테이션 설정",
        "rows": [
            [
                {"name": "rs_short_window", "kind": "int", "label": "단기 상대강도 창", "min": 3, "max": 80, "step": 1},
                {"name": "rs_mid_window", "kind": "int", "label": "중기 상대강도 창", "min": 5, "max": 160, "step": 1},
                {"name": "rs_long_window", "kind": "int", "label": "장기 상대강도 창", "min": 10, "max": 320, "step": 1},
                {"name": "trend_ema_window", "kind": "int", "label": "추세 EMA 길이", "min": 10, "max": 240, "step": 1},
            ],
            [
                {"name": "breakout_window", "kind": "int", "label": "돌파 창", "min": 5, "max": 120, "step": 1},
                {"name": "atr_window", "kind": "int", "label": "ATR 창", "min": 5, "max": 50, "step": 1},
                {"name": "atr_mult", "kind": "float", "label": "ATR 배수", "min": 1.0, "max": 6.0, "step": 0.1},
                {"name": "volume_window", "kind": "int", "label": "거래량 창", "min": 5, "max": 80, "step": 1},
            ],
            [
                {"name": "volume_threshold", "kind": "float", "label": "거래량 비율", "min": 0.1, "max": 3.0, "step": 0.1},
                {"name": "entry_score", "kind": "float", "label": "진입 점수", "min": -20.0, "max": 40.0, "step": 0.5},
                {"name": "exit_score", "kind": "float", "label": "청산 점수", "min": -20.0, "max": 40.0, "step": 0.5},
            ],
        ],
    },
    "relative_strength_guard": {
        "title": "상대강도 로테이션 가드 설정",
        "rows": [
            [
                {"name": "rs_short_window", "kind": "int", "label": "단기 상대강도 창", "min": 3, "max": 80, "step": 1},
                {"name": "rs_mid_window", "kind": "int", "label": "중기 상대강도 창", "min": 5, "max": 160, "step": 1},
                {"name": "rs_long_window", "kind": "int", "label": "장기 상대강도 창", "min": 10, "max": 320, "step": 1},
                {"name": "trend_ema_window", "kind": "int", "label": "추세 EMA 길이", "min": 10, "max": 240, "step": 1},
            ],
            [
                {"name": "breakout_window", "kind": "int", "label": "돌파 창", "min": 5, "max": 120, "step": 1},
                {"name": "atr_window", "kind": "int", "label": "ATR 창", "min": 5, "max": 50, "step": 1},
                {"name": "atr_mult", "kind": "float", "label": "ATR 배수", "min": 1.0, "max": 6.0, "step": 0.1},
                {"name": "volume_window", "kind": "int", "label": "거래량 창", "min": 5, "max": 80, "step": 1},
            ],
            [
                {"name": "volume_threshold", "kind": "float", "label": "거래량 비율", "min": 0.1, "max": 3.0, "step": 0.1},
                {"name": "entry_score", "kind": "float", "label": "진입 점수", "min": -20.0, "max": 40.0, "step": 0.5},
                {"name": "exit_score", "kind": "float", "label": "청산 점수", "min": -20.0, "max": 40.0, "step": 0.5},
            ],
            [
                {"name": "guard_fast_ema", "kind": "int", "label": "가드 빠른 EMA", "min": 5, "max": 120, "step": 1},
                {"name": "guard_slow_ema", "kind": "int", "label": "가드 느린 EMA", "min": 20, "max": 320, "step": 1},
                {"name": "guard_buffer_pct", "kind": "float", "label": "가드 버퍼 (%)", "min": 0.0, "max": 5.0, "step": 0.1},
                {"name": "guard_adx_window", "kind": "int", "label": "가드 ADX 창", "min": 5, "max": 50, "step": 1},
            ],
            [
                {"name": "guard_adx_floor", "kind": "float", "label": "가드 ADX 기준", "min": 5.0, "max": 40.0, "step": 0.5},
                {"name": "guard_rs_floor", "kind": "float", "label": "가드 RS 기준", "min": -30.0, "max": 20.0, "step": 0.5},
            ],
        ],
    },
    "regime_blend_guard": {
        "title": "장세 적응 혼합 가드 설정",
        "rows": [
            [
                {"name": "trend_fast_ema", "kind": "int", "label": "추세 빠른 EMA", "min": 5, "max": 120, "step": 1},
                {"name": "trend_slow_ema", "kind": "int", "label": "추세 느린 EMA", "min": 10, "max": 240, "step": 1},
                {"name": "trend_breakout_window", "kind": "int", "label": "추세 돌파 창", "min": 5, "max": 120, "step": 1},
                {"name": "trend_exit_window", "kind": "int", "label": "추세 청산 창", "min": 3, "max": 80, "step": 1},
            ],
            [
                {"name": "trend_atr_window", "kind": "int", "label": "추세 ATR 창", "min": 5, "max": 50, "step": 1},
                {"name": "trend_atr_mult", "kind": "float", "label": "추세 ATR 배수", "min": 1.0, "max": 6.0, "step": 0.1},
                {"name": "trend_adx_window", "kind": "int", "label": "추세 ADX 창", "min": 5, "max": 50, "step": 1},
                {"name": "trend_adx_threshold", "kind": "float", "label": "추세 ADX 기준", "min": 5.0, "max": 40.0, "step": 0.5},
            ],
            [
                {"name": "trend_momentum_window", "kind": "int", "label": "추세 모멘텀 창", "min": 5, "max": 80, "step": 1},
                {"name": "trend_volume_window", "kind": "int", "label": "추세 거래량 창", "min": 5, "max": 80, "step": 1},
                {"name": "trend_volume_threshold", "kind": "float", "label": "추세 거래량 비율", "min": 0.1, "max": 3.0, "step": 0.1},
                {"name": "regime_adx_floor", "kind": "float", "label": "장세 ADX 기준", "min": 5.0, "max": 40.0, "step": 0.5},
            ],
            [
                {"name": "rsi_len", "kind": "int", "label": "RSI 길이", "min": 2, "max": 50, "step": 1},
                {"name": "oversold", "kind": "float", "label": "과매도 기준", "min": 5.0, "max": 50.0, "step": 0.5},
                {"name": "bb_len", "kind": "int", "label": "BB 길이", "min": 5, "max": 80, "step": 1},
                {"name": "bb_mult", "kind": "float", "label": "BB 배수", "min": 0.5, "max": 5.0, "step": 0.1},
            ],
            [
                {"name": "min_down_bars", "kind": "int", "label": "연속 하락 봉", "min": 1, "max": 10, "step": 1},
                {"name": "low_tolerance_pct", "kind": "float", "label": "저점 허용치 (%)", "min": 0.0, "max": 5.0, "step": 0.1},
                {"name": "max_setup_bars", "kind": "int", "label": "셋업 유지 봉", "min": 3, "max": 40, "step": 1},
                {"name": "confirm_bars", "kind": "int", "label": "확인 대기 봉", "min": 1, "max": 20, "step": 1},
            ],
            [
                {"name": "use_macd_filter", "kind": "bool", "label": "MACD 확인 사용"},
                {"name": "macd_lookback", "kind": "int", "label": "MACD 최근 교차 봉", "min": 1, "max": 20, "step": 1},
                {"name": "risk_reward", "kind": "float", "label": "손익비", "min": 0.5, "max": 5.0, "step": 0.25},
                {"name": "stop_buffer_ticks", "kind": "int", "label": "스톱 버퍼 틱", "min": 0, "max": 20, "step": 1},
            ],
            [
                {"name": "bear_guard_buffer_pct", "kind": "float", "label": "약세 가드 버퍼 (%)", "min": 0.0, "max": 5.0, "step": 0.1},
                {"name": "bear_guard_adx_floor", "kind": "float", "label": "약세 가드 ADX", "min": 5.0, "max": 40.0, "step": 0.5},
                {"name": "bear_guard_score_floor", "kind": "float", "label": "약세 가드 점수", "min": -20.0, "max": 20.0, "step": 0.5},
            ],
        ],
    },
    "flux_trend": {
        "title": "플럭스 추세 밴드 설정",
        "rows": [
            [
                {"name": "ltf_len", "kind": "int", "label": "단기 기준 길이", "min": 5, "max": 400, "step": 1},
                {"name": "ltf_mult", "kind": "float", "label": "단기 밴드 배수", "min": 0.1, "max": 10.0, "step": 0.1},
                {"name": "htf_len", "kind": "int", "label": "상위 주기 길이", "min": 5, "max": 400, "step": 1},
                {"name": "htf_mult", "kind": "float", "label": "상위 밴드 배수", "min": 0.1, "max": 10.0, "step": 0.1},
                {"name": "htf_rule", "kind": "select", "label": "상위 주기", "options": ["30m", "60m", "120m", "240m", "1D"]},
            ],
        ],
    },
    "flux_ema_filter": {
        "title": "플럭스 + EMA 필터 설정",
        "rows": [
            [
                {"name": "ltf_len", "kind": "int", "label": "단기 기준 길이", "min": 5, "max": 400, "step": 1},
                {"name": "ltf_mult", "kind": "float", "label": "단기 밴드 배수", "min": 0.1, "max": 10.0, "step": 0.1},
                {"name": "htf_len", "kind": "int", "label": "상위 주기 길이", "min": 5, "max": 400, "step": 1},
                {"name": "htf_mult", "kind": "float", "label": "상위 밴드 배수", "min": 0.1, "max": 10.0, "step": 0.1},
                {"name": "htf_rule", "kind": "select", "label": "상위 주기", "options": ["30m", "60m", "120m", "240m", "1D"]},
            ],
            [
                {"name": "sensitivity", "kind": "int", "label": "민감도", "min": 1, "max": 10, "step": 1},
                {"name": "atr_period", "kind": "int", "label": "ATR 기간", "min": 1, "max": 20, "step": 1},
                {"name": "trend_ema_length", "kind": "int", "label": "추세 EMA 길이", "min": 20, "max": 400, "step": 5},
                {"name": "confirm_window", "kind": "int", "label": "EMA 확인 창", "min": 0, "max": 48, "step": 1},
            ],
            [
                {"name": "use_heikin_ashi", "kind": "bool", "label": "Heikin Ashi 사용"},
            ],
        ],
    },
}

STRATEGY_SWEEP_SCHEMAS: dict[str, dict[str, object]] = {
    "research_trend": {
        "rows": [
            [
                {"name": "fast_ema", "parser": "int", "label": "빠른 EMA 후보", "default": "12, 21, 34", "key": "bt_sweep_fast_ema"},
                {"name": "slow_ema", "parser": "int", "label": "느린 EMA 후보", "default": "55, 89", "key": "bt_sweep_slow_ema"},
                {"name": "breakout_window", "parser": "int", "label": "돌파 창 후보", "default": "14, 20, 28", "key": "bt_sweep_breakout"},
            ],
            [
                {"name": "atr_mult", "parser": "float", "label": "ATR 배수 후보", "default": "2.0, 2.5, 3.0", "key": "bt_sweep_atr_mult"},
                {"name": "adx_threshold", "parser": "float", "label": "ADX 기준 후보", "default": "16, 18, 20", "key": "bt_sweep_adx_threshold"},
            ],
        ],
    },
    "rsi_bb_double_bottom": {
        "rows": [
            [
                {"name": "rsi_len", "parser": "int", "label": "RSI 길이 후보", "default": "10, 14, 18", "key": "bt_sweep_db_rsi_len"},
                {"name": "oversold", "parser": "float", "label": "과매도 후보", "default": "25, 30, 35", "key": "bt_sweep_db_oversold"},
                {"name": "bb_len", "parser": "int", "label": "BB 길이 후보", "default": "18, 20, 24", "key": "bt_sweep_db_bb_len"},
                {"name": "bb_mult", "parser": "float", "label": "BB 배수 후보", "default": "1.8, 2.0, 2.2", "key": "bt_sweep_db_bb_mult"},
            ],
            [
                {"name": "min_down_bars", "parser": "int", "label": "연속 하락 봉 후보", "default": "2, 3", "key": "bt_sweep_db_min_down_bars"},
                {"name": "low_tolerance_pct", "parser": "float", "label": "저점 허용치 후보", "default": "0.5, 1.0, 1.5", "key": "bt_sweep_db_low_tolerance_pct"},
                {"name": "confirm_bars", "parser": "int", "label": "확인 봉 후보", "default": "3, 4, 5", "key": "bt_sweep_db_confirm_bars"},
                {"name": "risk_reward", "parser": "float", "label": "손익비 후보", "default": "1.5, 2.0, 2.5", "key": "bt_sweep_db_risk_reward"},
            ],
        ],
    },
    "rsi_trend_guard": {
        "rows": [
            [
                {"name": "rsi_len", "parser": "int", "label": "RSI 후보", "default": "8, 10, 12", "key": "bt_sweep_guard_rsi_len"},
                {"name": "oversold", "parser": "float", "label": "과매도 후보", "default": "33, 35, 37", "key": "bt_sweep_guard_oversold"},
                {"name": "bb_mult", "parser": "float", "label": "BB 배수 후보", "default": "1.3, 1.5, 1.7", "key": "bt_sweep_guard_bb_mult"},
                {"name": "max_setup_bars", "parser": "int", "label": "셋업 봉 후보", "default": "4, 6, 8", "key": "bt_sweep_guard_setup"},
            ],
            [
                {"name": "confirm_bars", "parser": "int", "label": "확인 봉 후보", "default": "2, 3, 4", "key": "bt_sweep_guard_confirm"},
                {"name": "risk_reward", "parser": "float", "label": "손익비 후보", "default": "1.2, 1.5, 2.0", "key": "bt_sweep_guard_rr"},
                {"name": "trend_fast_ema", "parser": "int", "label": "빠른 EMA 후보", "default": "13, 21", "key": "bt_sweep_guard_fast_ema"},
                {"name": "trend_slow_ema", "parser": "int", "label": "느린 EMA 후보", "default": "55, 89", "key": "bt_sweep_guard_slow_ema"},
            ],
            [
                {"name": "trend_buffer_pct", "parser": "float", "label": "EMA 버퍼 후보", "default": "1.0, 2.0", "key": "bt_sweep_guard_buffer"},
                {"name": "bearish_adx_floor", "parser": "float", "label": "약세 ADX 후보", "default": "14, 18, 22", "key": "bt_sweep_guard_adx_floor"},
                {"name": "adx_window", "parser": "int", "label": "ADX 창 후보", "default": "14", "key": "bt_sweep_guard_adx_window"},
                {"name": "use_macd_filter", "parser": "current_bool", "default": True},
            ],
        ],
    },
    "relative_strength_rotation": {
        "rows": [
            [
                {"name": "rs_short_window", "parser": "int", "label": "단기 RS 후보", "default": "8, 10, 14", "key": "bt_sweep_rs_short_window"},
                {"name": "rs_mid_window", "parser": "int", "label": "중기 RS 후보", "default": "20, 30, 45", "key": "bt_sweep_rs_mid_window"},
                {"name": "rs_long_window", "parser": "int", "label": "장기 RS 후보", "default": "60, 90, 120", "key": "bt_sweep_rs_long_window"},
            ],
            [
                {"name": "trend_ema_window", "parser": "int", "label": "추세 EMA 후보", "default": "34, 55, 80", "key": "bt_sweep_rs_trend_ema_window"},
                {"name": "breakout_window", "parser": "int", "label": "돌파 창 후보", "default": "14, 20, 28", "key": "bt_sweep_rs_breakout_window"},
                {"name": "entry_score", "parser": "float", "label": "진입 점수 후보", "default": "6, 8, 10", "key": "bt_sweep_rs_entry_score"},
            ],
            [
                {"name": "exit_score", "parser": "float", "label": "청산 점수 후보", "default": "0, 2, 4", "key": "bt_sweep_rs_exit_score"},
                {"name": "atr_mult", "parser": "float", "label": "ATR 배수 후보", "default": "1.8, 2.2, 2.6", "key": "bt_sweep_rs_atr_mult"},
            ],
        ],
    },
    "relative_strength_guard": {
        "rows": [
            [
                {"name": "rs_short_window", "parser": "int", "label": "단기 RS 후보", "default": "8, 10, 14", "key": "bt_sweep_rs_guard_short_window"},
                {"name": "rs_mid_window", "parser": "int", "label": "중기 RS 후보", "default": "20, 30, 45", "key": "bt_sweep_rs_guard_mid_window"},
                {"name": "rs_long_window", "parser": "int", "label": "장기 RS 후보", "default": "60, 90, 120", "key": "bt_sweep_rs_guard_long_window"},
            ],
            [
                {"name": "trend_ema_window", "parser": "int", "label": "추세 EMA 후보", "default": "34, 55, 80", "key": "bt_sweep_rs_guard_trend_ema_window"},
                {"name": "breakout_window", "parser": "int", "label": "돌파 창 후보", "default": "14, 20, 28", "key": "bt_sweep_rs_guard_breakout_window"},
                {"name": "entry_score", "parser": "float", "label": "진입 점수 후보", "default": "7, 8, 9", "key": "bt_sweep_rs_guard_entry_score"},
                {"name": "exit_score", "parser": "float", "label": "청산 점수 후보", "default": "1, 2, 3", "key": "bt_sweep_rs_guard_exit_score"},
            ],
            [
                {"name": "guard_fast_ema", "parser": "int", "label": "가드 빠른 EMA 후보", "default": "13, 21, 34", "key": "bt_sweep_rs_guard_fast_ema"},
                {"name": "guard_slow_ema", "parser": "int", "label": "가드 느린 EMA 후보", "default": "89, 144, 200", "key": "bt_sweep_rs_guard_slow_ema"},
                {"name": "guard_buffer_pct", "parser": "float", "label": "가드 버퍼 후보", "default": "0.0, 0.5, 1.0, 1.5", "key": "bt_sweep_rs_guard_buffer_pct"},
            ],
            [
                {"name": "guard_adx_floor", "parser": "float", "label": "가드 ADX 후보", "default": "10, 14, 18", "key": "bt_sweep_rs_guard_adx_floor"},
                {"name": "guard_rs_floor", "parser": "float", "label": "가드 RS 후보", "default": "-6, -3, 0, 2", "key": "bt_sweep_rs_guard_rs_floor"},
                {"name": "guard_adx_window", "parser": "int", "label": "가드 ADX 창 후보", "default": "14", "key": "bt_sweep_rs_guard_adx_window"},
            ],
        ],
    },
    "regime_blend_guard": {
        "rows": [
            [
                {"name": "trend_fast_ema", "parser": "int", "label": "추세 빠른 EMA 후보", "default": "13, 21, 34", "key": "bt_sweep_blend_guard_fast_ema"},
                {"name": "trend_slow_ema", "parser": "int", "label": "추세 느린 EMA 후보", "default": "55, 89", "key": "bt_sweep_blend_guard_slow_ema"},
                {"name": "trend_breakout_window", "parser": "int", "label": "추세 돌파 창 후보", "default": "14, 20, 28", "key": "bt_sweep_blend_guard_breakout_window"},
            ],
            [
                {"name": "regime_adx_floor", "parser": "float", "label": "장세 ADX 후보", "default": "12, 16, 20", "key": "bt_sweep_blend_guard_regime_adx_floor"},
                {"name": "oversold", "parser": "float", "label": "과매도 후보", "default": "30, 35, 40", "key": "bt_sweep_blend_guard_oversold"},
                {"name": "bb_mult", "parser": "float", "label": "BB 배수 후보", "default": "1.6, 2.0, 2.4", "key": "bt_sweep_blend_guard_bb_mult"},
            ],
            [
                {"name": "risk_reward", "parser": "float", "label": "손익비 후보", "default": "1.2, 1.5, 2.0", "key": "bt_sweep_blend_guard_risk_reward"},
                {"name": "bear_guard_buffer_pct", "parser": "float", "label": "약세 버퍼 후보", "default": "0.5, 1.0, 1.5", "key": "bt_sweep_blend_guard_buffer_pct"},
                {"name": "bear_guard_adx_floor", "parser": "float", "label": "약세 ADX 후보", "default": "10, 14, 18", "key": "bt_sweep_blend_guard_adx_floor"},
                {"name": "bear_guard_score_floor", "parser": "float", "label": "약세 점수 후보", "default": "-4, -2, 0", "key": "bt_sweep_blend_guard_score_floor"},
            ],
            [
                {"name": "use_macd_filter", "parser": "current_bool", "default": True},
            ],
        ],
    },
    "flux_trend": {
        "rows": [
            [
                {"name": "ltf_len", "parser": "int", "label": "단기 길이 후보", "default": "14, 20, 28", "key": "bt_sweep_ltf_len"},
                {"name": "ltf_mult", "parser": "float", "label": "단기 배수 후보", "default": "1.5, 2.0, 2.5", "key": "bt_sweep_ltf_mult"},
                {"name": "htf_len", "parser": "int", "label": "상위 길이 후보", "default": "20, 30, 40", "key": "bt_sweep_htf_len"},
            ],
            [
                {"name": "htf_mult", "parser": "float", "label": "상위 배수 후보", "default": "2.0, 2.25, 2.5", "key": "bt_sweep_htf_mult"},
                {"name": "htf_rule", "parser": "htf_rule", "label": "상위 주기 후보", "default": "60T, 120T, 240T", "key": "bt_sweep_htf_rule"},
            ],
        ],
    },
    "flux_ema_filter": {
        "rows": [
            [
                {"name": "ltf_len", "parser": "int", "label": "단기 길이 후보", "default": "14, 20", "key": "bt_sweep_flux_ema_ltf_len"},
                {"name": "ltf_mult", "parser": "float", "label": "단기 배수 후보", "default": "1.5, 2.0", "key": "bt_sweep_flux_ema_ltf_mult"},
                {"name": "htf_len", "parser": "int", "label": "상위 길이 후보", "default": "20, 30", "key": "bt_sweep_flux_ema_htf_len"},
            ],
            [
                {"name": "htf_mult", "parser": "float", "label": "상위 배수 후보", "default": "2.0, 2.25", "key": "bt_sweep_flux_ema_htf_mult"},
                {"name": "htf_rule", "parser": "htf_rule", "label": "상위 주기 후보", "default": "60T, 120T", "key": "bt_sweep_flux_ema_htf_rule"},
                {"name": "sensitivity", "parser": "int", "label": "민감도 후보", "default": "2, 3", "key": "bt_sweep_flux_ema_sensitivity"},
            ],
            [
                {"name": "atr_period", "parser": "int", "label": "ATR 기간 후보", "default": "2, 3", "key": "bt_sweep_flux_ema_atr_period"},
                {"name": "trend_ema_length", "parser": "int", "label": "추세 EMA 후보", "default": "240", "key": "bt_sweep_flux_ema_length"},
                {"name": "confirm_window", "parser": "int", "label": "EMA 확인 창 후보", "default": "8", "key": "bt_sweep_flux_ema_confirm_window"},
            ],
        ],
    },
}

STRATEGY_SWEEP_RESULT_SCHEMAS: dict[str, dict[str, object]] = {
    "research_trend": {
        "ordered": ["fast_ema", "slow_ema", "breakout_window", "atr_mult", "adx_threshold", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"fast_ema": "빠른 EMA", "slow_ema": "느린 EMA", "breakout_window": "돌파 창", "atr_mult": "ATR 배수", "adx_threshold": "ADX 기준", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "rsi_bb_double_bottom": {
        "ordered": ["rsi_len", "oversold", "bb_len", "bb_mult", "min_down_bars", "low_tolerance_pct", "confirm_bars", "risk_reward", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"rsi_len": "RSI 길이", "oversold": "과매도 기준", "bb_len": "BB 길이", "bb_mult": "BB 배수", "min_down_bars": "연속 하락 봉", "low_tolerance_pct": "저점 허용치", "confirm_bars": "확인 봉", "risk_reward": "손익비", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "rsi_trend_guard": {
        "ordered": ["rsi_len", "oversold", "bb_mult", "max_setup_bars", "confirm_bars", "risk_reward", "trend_fast_ema", "trend_slow_ema", "trend_buffer_pct", "bearish_adx_floor", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"rsi_len": "RSI 길이", "oversold": "과매도 기준", "bb_mult": "BB 배수", "max_setup_bars": "셋업 봉", "confirm_bars": "확인 봉", "risk_reward": "손익비", "trend_fast_ema": "빠른 EMA", "trend_slow_ema": "느린 EMA", "trend_buffer_pct": "EMA 버퍼", "bearish_adx_floor": "약세 ADX", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "relative_strength_rotation": {
        "ordered": ["rs_short_window", "rs_mid_window", "rs_long_window", "trend_ema_window", "breakout_window", "entry_score", "exit_score", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"rs_short_window": "단기 RS", "rs_mid_window": "중기 RS", "rs_long_window": "장기 RS", "trend_ema_window": "추세 EMA", "breakout_window": "돌파 창", "entry_score": "진입 점수", "exit_score": "청산 점수", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "relative_strength_guard": {
        "ordered": ["rs_short_window", "rs_mid_window", "rs_long_window", "trend_ema_window", "breakout_window", "entry_score", "exit_score", "guard_fast_ema", "guard_slow_ema", "guard_buffer_pct", "guard_adx_floor", "guard_rs_floor", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"rs_short_window": "단기 RS", "rs_mid_window": "중기 RS", "rs_long_window": "장기 RS", "trend_ema_window": "추세 EMA", "breakout_window": "돌파 창", "entry_score": "진입 점수", "exit_score": "청산 점수", "guard_fast_ema": "가드 빠른 EMA", "guard_slow_ema": "가드 느린 EMA", "guard_buffer_pct": "가드 버퍼", "guard_adx_floor": "가드 ADX", "guard_rs_floor": "가드 RS", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "regime_blend_guard": {
        "ordered": ["trend_fast_ema", "trend_slow_ema", "trend_breakout_window", "regime_adx_floor", "oversold", "bb_mult", "risk_reward", "bear_guard_buffer_pct", "bear_guard_adx_floor", "bear_guard_score_floor", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"trend_fast_ema": "추세 빠른 EMA", "trend_slow_ema": "추세 느린 EMA", "trend_breakout_window": "추세 돌파 창", "regime_adx_floor": "장세 ADX", "oversold": "과매도", "bb_mult": "BB 배수", "risk_reward": "손익비", "bear_guard_buffer_pct": "약세 버퍼", "bear_guard_adx_floor": "약세 ADX", "bear_guard_score_floor": "약세 점수", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "flux_trend": {
        "ordered": ["ltf_len", "ltf_mult", "htf_len", "htf_mult", "htf_rule", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"ltf_len": "단기 길이", "ltf_mult": "단기 배수", "htf_len": "상위 길이", "htf_mult": "상위 배수", "htf_rule": "상위 주기", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
    "flux_ema_filter": {
        "ordered": ["ltf_len", "ltf_mult", "htf_len", "htf_mult", "htf_rule", "sensitivity", "atr_period", "trend_ema_length", "confirm_window", "trades", "buy_signals", "sell_signals", "total_return_pct", "win_rate_pct", "max_drawdown_pct"],
        "rename": {"ltf_len": "단기 길이", "ltf_mult": "단기 배수", "htf_len": "상위 길이", "htf_mult": "상위 배수", "htf_rule": "상위 주기", "sensitivity": "민감도", "atr_period": "ATR 기간", "trend_ema_length": "추세 EMA", "confirm_window": "확인 창", "trades": "거래 수", "buy_signals": "매수 신호 수", "sell_signals": "매도 신호 수", "total_return_pct": "수익률", "win_rate_pct": "승률", "max_drawdown_pct": "최대 낙폭"},
    },
}

STRATEGY_DETAIL_COLUMNS: dict[str, list[str]] = {
    "research_trend": ["ema_fast", "ema_slow", "adx", "atr"],
    "rsi_bb_double_bottom": ["rsi", "bb_lower", "bb_upper", "trade_stop", "take_profit", "rebound_marker", "second_bottom_marker"],
    "rsi_trend_guard": ["rsi", "bb_lower", "bb_upper", "trade_stop", "take_profit", "ema_fast", "ema_slow", "adx", "bearish_regime", "trend_filter"],
    "relative_strength_rotation": ["rs_short", "rs_mid", "rs_long", "trend_ema", "volume_ratio", "atr_stop"],
    "relative_strength_guard": ["rs_short", "rs_mid", "rs_long", "trend_ema", "volume_ratio", "atr_stop", "guard_fast_ema", "guard_slow_ema", "guard_adx", "bearish_regime", "risk_on_regime", "guard_slow_slope"],
    "regime_blend_guard": ["trend_regime", "trend_score", "range_score", "entry_mode", "base_entry_mode", "ema_fast", "ema_slow", "adx", "atr_stop", "trade_stop", "take_profit", "bearish_regime", "risk_on_regime", "bear_guard_slow_slope"],
    "flux_ema_filter": ["strength", "ema_buy", "ema_sell", "flux_buy_signal", "flux_sell_signal"],
}

DOUBLE_BOTTOM_STRATEGIES = {"rsi_bb_double_bottom", "rsi_trend_guard"}

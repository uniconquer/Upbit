from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from strategy import (
    backtest_signal_frame,
    extract_backtest_trade_events,
    parameter_grid_size,
    sweep_research_trend_parameters,
)
from strategy_engine import (
    build_strategy_frame,
    compare_strategy_backtests,
    strategy_description,
    strategy_label,
    strategy_options,
    sweep_strategy_parameters,
)
from ui_theme import apply_chart_theme, page_intro
from upbit_api import UpbitAPI
from utils.formatters import fmt_full_number

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:
    AgGrid = None


api: UpbitAPI | None = None
flux_indicator = None
flux_indicator_with_ema = None

_INTERVAL_LABELS = {
    "minute15": "15분",
    "minute30": "30분",
    "minute60": "60분",
    "minute240": "240분",
    "day": "일봉",
}

_SERIES_LABELS = {
    "ema_fast": "빠른 EMA",
    "ema_slow": "느린 EMA",
    "atr_stop": "ATR 손절선",
    "breakout_high": "돌파 기준선",
    "breakdown_low": "이탈 기준선",
    "adx": "ADX 추세 강도",
    "strategy_score": "전략 점수",
    "ltf_upper": "단기 상단 밴드",
    "ltf_lower": "단기 하단 밴드",
    "ltf_basis": "단기 기준선",
    "htf_upper": "상위 상단 밴드",
    "htf_lower": "상위 하단 밴드",
    "trend_ema": "추세 EMA",
    "strength": "EMA 필터 강도",
    "flux_buy_signal": "원본 플럭스 매수",
    "flux_sell_signal": "원본 플럭스 매도",
    "rsi": "RSI",
    "bb_basis": "볼린저 기준선",
    "bb_upper": "볼린저 상단",
    "bb_lower": "볼린저 하단",
    "macd_line": "MACD",
    "macd_signal": "MACD 시그널",
    "trade_stop": "전략 손절선",
    "take_profit": "전략 익절선",
}

_SERIES_LABELS.update(
    {
        "rs_short": "단기 상대강도",
        "rs_mid": "중기 상대강도",
        "rs_long": "장기 상대강도",
        "volume_ratio": "거래량 비율",
    }
)

_BACKTEST_DEFAULT_PARAMS: dict[str, dict[str, object]] = {
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

_BACKTEST_WIDGET_KEYS: dict[str, dict[str, str]] = {
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


def _interval_text(value: str) -> str:
    return _INTERVAL_LABELS.get(value, value)


def init_api(a: UpbitAPI, flux, flux_ext):
    global api, flux_indicator, flux_indicator_with_ema
    api = a
    flux_indicator = flux
    flux_indicator_with_ema = flux_ext


@st.cache_data(ttl=600)
def _markets() -> pd.DataFrame:
    if not api:
        return pd.DataFrame()
    try:
        rows = []
        for market in api.markets():
            market_code = market.get("market", "")
            if not market_code.startswith("KRW-"):
                continue
            rows.append(
                {
                    "market": market_code,
                    "symbol": market_code.split("-")[1],
                    "korean_name": market.get("korean_name"),
                    "english_name": market.get("english_name"),
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _tickers(markets: tuple[str, ...]) -> pd.DataFrame:
    if not api or not markets:
        return pd.DataFrame()
    try:
        rows = []
        for ticker in api.tickers(list(markets)):
            rows.append(
                {
                    "market": ticker.get("market"),
                    "trade_price": ticker.get("trade_price"),
                    "acc_trade_price_24h": ticker.get("acc_trade_price_24h"),
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def _candles(market: str, interval: str, count: int) -> pd.DataFrame:
    if not api:
        return pd.DataFrame()
    try:
        candles = api.candles(market, interval=interval, count=count)
        frame = pd.DataFrame(
            [
                {
                    "time": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
                for candle in candles
            ]
        )
        if not frame.empty:
            frame["dt"] = pd.to_datetime(frame["time"], unit="ms")
            frame = frame.set_index("dt").sort_index()
        return frame
    except Exception:
        return pd.DataFrame()


def _market_panel(frame: pd.DataFrame) -> str | None:
    st.subheader("종목 목록")
    query = st.text_input("검색", "", placeholder="BTC, ETH, 비트코인", key="bt_search")
    table = frame.copy()
    if query.strip():
        value = query.strip().upper()
        table = table[
            table.apply(
                lambda row: value in str(row["market"]).upper()
                or value in str(row["symbol"]).upper()
                or value in str(row["korean_name"]).upper(),
                axis=1,
            )
        ]

    rows = table.to_dict("records")
    if rows and "bt_sel_market" not in st.session_state:
        st.session_state["bt_sel_market"] = rows[0]["market"]

    if AgGrid and rows:
        grid_frame = pd.DataFrame(
            [
                {
                    "종목": row["market"],
                    "한글명": row["korean_name"],
                    "현재가": fmt_full_number(row.get("trade_price"), 0),
                    "24H 거래대금": fmt_full_number(row.get("acc_trade_price_24h"), 0),
                    "turnover_raw": row.get("acc_trade_price_24h") or 0,
                }
                for row in rows
            ]
        )
        builder = GridOptionsBuilder.from_dataframe(grid_frame)
        builder.configure_column("turnover_raw", hide=True)
        builder.configure_selection("single")
        builder.configure_grid_options(sortModel=[{"colId": "turnover_raw", "sort": "desc"}])
        grid = AgGrid(
            grid_frame,
            gridOptions=builder.build(),
            height=560,
            fit_columns_on_grid_load=True,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            theme="balham",
            key="bt_markets",
        )
        selected = grid.get("selected_rows", [])
        if isinstance(selected, pd.DataFrame):
            selected = selected.to_dict("records")
        if selected:
            st.session_state["bt_sel_market"] = selected[0]["종목"]
    else:
        options = [row["market"] for row in rows]
        if options:
            current = st.session_state.get("bt_sel_market", options[0])
            if current not in options:
                current = options[0]
            st.session_state["bt_sel_market"] = st.selectbox("종목", options, index=options.index(current))
        elif query.strip():
            st.info("검색 조건에 맞는 종목이 없습니다.")

    return st.session_state.get("bt_sel_market")


def _strategy_controls() -> tuple[str, dict[str, float | int | str]]:
    options = strategy_options(flux_indicator is not None, flux_indicator_with_ema is not None)
    current = st.session_state.get("bt_strategy_name", options[0])
    index = options.index(current) if current in options else 0
    strategy_name = st.selectbox(
        "전략",
        options,
        index=index,
        format_func=strategy_label,
        key="bt_strategy_name",
    )
    st.caption(strategy_description(strategy_name))

    params: dict[str, float | int | str] = {}
    if strategy_name == "research_trend":
        with st.expander("연구형 추세 돌파 설정", expanded=False):
            row1 = st.columns(4)
            params["fast_ema"] = row1[0].number_input("빠른 EMA", 5, 100, 21, 1, key="bt_research_fast_ema")
            params["slow_ema"] = row1[1].number_input("느린 EMA", 10, 240, 55, 1, key="bt_research_slow_ema")
            params["breakout_window"] = row1[2].number_input("돌파 창", 5, 120, 20, 1, key="bt_research_breakout_window")
            params["exit_window"] = row1[3].number_input("청산 창", 3, 80, 10, 1, key="bt_research_exit_window")
            row2 = st.columns(4)
            params["atr_window"] = row2[0].number_input("ATR 창", 5, 50, 14, 1, key="bt_research_atr_window")
            params["atr_mult"] = row2[1].number_input("ATR 배수", 1.0, 6.0, 2.5, 0.1, key="bt_research_atr_mult")
            params["adx_window"] = row2[2].number_input("ADX 창", 5, 50, 14, 1, key="bt_research_adx_window")
            params["adx_threshold"] = row2[3].number_input("ADX 기준", 5.0, 40.0, 18.0, 0.5, key="bt_research_adx_threshold")
            row3 = st.columns(3)
            params["momentum_window"] = row3[0].number_input("모멘텀 창", 5, 80, 20, 1, key="bt_research_momentum_window")
            params["volume_window"] = row3[1].number_input("거래량 창", 5, 80, 20, 1, key="bt_research_volume_window")
            params["volume_threshold"] = row3[2].number_input("거래량 비율", 0.1, 3.0, 0.9, 0.1, key="bt_research_volume_threshold")
    elif strategy_name == "rsi_bb_double_bottom":
        with st.expander("RSI+BB 더블바텀 롱 설정", expanded=False):
            row1 = st.columns(4)
            params["rsi_len"] = row1[0].number_input("RSI 길이", 2, 50, 14, 1, key="bt_db_rsi_len")
            params["oversold"] = row1[1].number_input("과매도 기준", 5.0, 50.0, 30.0, 0.5, key="bt_db_oversold")
            params["bb_len"] = row1[2].number_input("BB 길이", 5, 80, 20, 1, key="bt_db_bb_len")
            params["bb_mult"] = row1[3].number_input("BB 배수", 0.5, 5.0, 2.0, 0.1, key="bt_db_bb_mult")
            row2 = st.columns(4)
            params["min_down_bars"] = row2[0].number_input("연속 하락봉 수", 1, 10, 2, 1, key="bt_db_min_down_bars")
            params["low_tolerance_pct"] = row2[1].number_input("두 번째 바닥 허용치 (%)", 0.0, 5.0, 1.0, 0.1, key="bt_db_low_tolerance_pct")
            params["max_setup_bars"] = row2[2].number_input("셋업 유지 바 수", 3, 40, 12, 1, key="bt_db_max_setup_bars")
            params["confirm_bars"] = row2[3].number_input("확인 대기 바 수", 1, 20, 4, 1, key="bt_db_confirm_bars")
            row3 = st.columns(4)
            params["use_macd_filter"] = row3[0].checkbox("MACD 확인 사용", value=True, key="bt_db_use_macd_filter")
            params["macd_lookback"] = row3[1].number_input("MACD 최근 교차 바 수", 1, 20, 5, 1, key="bt_db_macd_lookback")
            params["risk_reward"] = row3[2].number_input("손익비", 0.5, 5.0, 2.0, 0.25, key="bt_db_risk_reward")
            params["stop_buffer_ticks"] = row3[3].number_input("스탑 버퍼 틱", 0, 20, 2, 1, key="bt_db_stop_buffer_ticks")
    elif strategy_name == "relative_strength_rotation":
        with st.expander("상대강도 로테이션 설정", expanded=False):
            row1 = st.columns(4)
            params["rs_short_window"] = row1[0].number_input("단기 상대강도 창", 3, 80, 10, 1, key="bt_rs_short_window")
            params["rs_mid_window"] = row1[1].number_input("중기 상대강도 창", 5, 160, 30, 1, key="bt_rs_mid_window")
            params["rs_long_window"] = row1[2].number_input("장기 상대강도 창", 10, 320, 90, 1, key="bt_rs_long_window")
            params["trend_ema_window"] = row1[3].number_input("추세 EMA 길이", 10, 240, 55, 1, key="bt_rs_trend_ema_window")
            row2 = st.columns(4)
            params["breakout_window"] = row2[0].number_input("돌파 창", 5, 120, 20, 1, key="bt_rs_breakout_window")
            params["atr_window"] = row2[1].number_input("ATR 창", 5, 50, 14, 1, key="bt_rs_atr_window")
            params["atr_mult"] = row2[2].number_input("ATR 배수", 1.0, 6.0, 2.2, 0.1, key="bt_rs_atr_mult")
            params["volume_window"] = row2[3].number_input("거래량 창", 5, 80, 20, 1, key="bt_rs_volume_window")
            row3 = st.columns(3)
            params["volume_threshold"] = row3[0].number_input("거래량 비율", 0.1, 3.0, 0.9, 0.1, key="bt_rs_volume_threshold")
            params["entry_score"] = row3[1].number_input("진입 점수", -20.0, 40.0, 8.0, 0.5, key="bt_rs_entry_score")
            params["exit_score"] = row3[2].number_input("청산 점수", -20.0, 40.0, 2.0, 0.5, key="bt_rs_exit_score")
    elif strategy_name == "flux_trend":
        with st.expander("플럭스 추세 밴드 설정", expanded=False):
            row = st.columns(5)
            params["ltf_len"] = row[0].number_input("단기 기준 길이", 5, 400, 20, 1, key="bt_flux_ltf_len")
            params["ltf_mult"] = row[1].number_input("단기 밴드 배수", 0.1, 10.0, 2.0, 0.1, key="bt_flux_ltf_mult")
            params["htf_len"] = row[2].number_input("상위 주기 길이", 5, 400, 20, 1, key="bt_flux_htf_len")
            params["htf_mult"] = row[3].number_input("상위 밴드 배수", 0.1, 10.0, 2.25, 0.1, key="bt_flux_htf_mult")
            htf = row[4].selectbox("상위 주기", ["30m", "60m", "120m", "240m", "1D"], index=1, key="bt_flux_htf_rule")
            params["htf_rule"] = htf.replace("m", "T") if htf.endswith("m") else "1D"
    else:
        with st.expander("플럭스 + EMA 필터 설정", expanded=False):
            row1 = st.columns(5)
            params["ltf_len"] = row1[0].number_input("단기 기준 길이", 5, 400, 20, 1, key="bt_flux_ema_ltf_len")
            params["ltf_mult"] = row1[1].number_input("단기 밴드 배수", 0.1, 10.0, 2.0, 0.1, key="bt_flux_ema_ltf_mult")
            params["htf_len"] = row1[2].number_input("상위 주기 길이", 5, 400, 20, 1, key="bt_flux_ema_htf_len")
            params["htf_mult"] = row1[3].number_input("상위 밴드 배수", 0.1, 10.0, 2.25, 0.1, key="bt_flux_ema_htf_mult")
            htf = row1[4].selectbox("상위 주기", ["30m", "60m", "120m", "240m", "1D"], index=1, key="bt_flux_ema_htf_rule")
            params["htf_rule"] = htf.replace("m", "T") if htf.endswith("m") else "1D"
            row2 = st.columns(4)
            params["sensitivity"] = row2[0].number_input("민감도", 1, 10, 3, 1, key="bt_flux_ema_sensitivity")
            params["atr_period"] = row2[1].number_input("ATR 기간", 1, 20, 2, 1, key="bt_flux_ema_atr_period")
            params["trend_ema_length"] = row2[2].number_input("추세 EMA 길이", 20, 400, 240, 5, key="bt_flux_ema_length")
            params["confirm_window"] = row2[3].number_input("EMA 확인 창", 0, 48, 8, 1, key="bt_flux_ema_confirm_window")
            params["use_heikin_ashi"] = st.checkbox("Heikin Ashi 사용", value=False, key="bt_flux_ema_heikin_ashi")
    return strategy_name, params


def _parse_sweep_values(raw: str, caster):
    values = []
    for chunk in str(raw or "").split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            values.append(caster(text))
        except Exception:
            continue
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def _normalize_htf_rule(value: str) -> str:
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


def _parse_htf_rule_values(raw: str) -> list[str]:
    values = []
    seen: set[str] = set()
    for chunk in str(raw or "").split(","):
        normalized = _normalize_htf_rule(chunk)
        if not normalized or normalized in seen:
            continue
        values.append(normalized)
        seen.add(normalized)
    return values


def _htf_rule_to_widget(value: object) -> str:
    normalized = _normalize_htf_rule(str(value or ""))
    if normalized.endswith("T") and normalized[:-1].isdigit():
        return f"{normalized[:-1]}m"
    return normalized or "60m"


def _signal_label(value: object) -> str:
    mapping = {
        "BUY": "매수",
        "SELL": "매도",
        "WAIT": "대기",
    }
    return mapping.get(str(value or "").upper(), str(value or "-"))


def _strategy_param_summary(strategy_name: str, params: dict[str, object]) -> str:
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


def _coerce_param_value(value: object, default: object) -> object:
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
        normalized = _normalize_htf_rule(str(value or default))
        return normalized or default
    return value if value is not None else default


def _params_for_strategy(strategy_name: str) -> dict[str, object]:
    defaults = dict(_BACKTEST_DEFAULT_PARAMS.get(strategy_name, {}))
    widgets = _BACKTEST_WIDGET_KEYS.get(strategy_name, {})
    params: dict[str, object] = {}
    for field, default in defaults.items():
        raw_value = st.session_state.get(widgets.get(field, ""), default)
        params[field] = _coerce_param_value(raw_value, default)
    return params


def _apply_params_to_widgets(strategy_name: str, params: dict[str, object]) -> None:
    widgets = _BACKTEST_WIDGET_KEYS.get(strategy_name, {})
    defaults = _BACKTEST_DEFAULT_PARAMS.get(strategy_name, {})
    for field, key in widgets.items():
        if not key:
            continue
        default = defaults.get(field)
        value = params.get(field, default)
        if field == "htf_rule":
            st.session_state[key] = _htf_rule_to_widget(value)
        else:
            st.session_state[key] = _coerce_param_value(value, default)


def _present_compare_results(results: pd.DataFrame) -> pd.DataFrame:
    visible = results.copy()
    if "score" in visible.columns:
        visible["score"] = visible["score"].apply(lambda value: f"{float(value):.2f}")
    for column in ["return_pct", "win_rate_pct", "max_drawdown_pct"]:
        if column in visible.columns:
            visible[column] = visible[column].apply(lambda value: f"{float(value):.2f}%")
    if "last_signal" in visible.columns:
        visible["last_signal"] = visible["last_signal"].apply(_signal_label)
    if "params" in visible.columns:
        visible["설정 요약"] = visible.apply(
            lambda row: _strategy_param_summary(str(row.get("strategy_name") or ""), dict(row.get("params") or {})),
            axis=1,
        )
    ordered = [
        "strategy_label",
        "설정 요약",
        "trades",
        "buy_signals",
        "sell_signals",
        "return_pct",
        "win_rate_pct",
        "max_drawdown_pct",
        "last_signal",
        "score",
    ]
    return visible[[column for column in ordered if column in visible.columns]].rename(
        columns={
            "strategy_label": "전략",
            "trades": "거래 수",
            "buy_signals": "매수 신호 수",
            "sell_signals": "매도 신호 수",
            "return_pct": "수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
            "last_signal": "현재 신호",
            "score": "전략 점수",
        }
    )


def _format_signal_time(index_value) -> str:
    if index_value is None:
        return "-"
    try:
        return pd.Timestamp(index_value).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"


def _signal_timeline(frame: pd.DataFrame, *, limit: int = 12) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp, row in frame.iterrows():
        if bool(row.get("buy_signal")):
            rows.append({"시각": _format_signal_time(timestamp), "구분": "매수 신호", "가격": fmt_full_number(row.get("close"), 0)})
        if bool(row.get("sell_signal")):
            rows.append({"시각": _format_signal_time(timestamp), "구분": "매도 신호", "가격": fmt_full_number(row.get("close"), 0)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).tail(limit).iloc[::-1].reset_index(drop=True)


def _research_condition_table(frame: pd.DataFrame, params: dict[str, float | int | str]) -> pd.DataFrame:
    last = frame.iloc[-1]
    adx_threshold = float(params.get("adx_threshold", 18.0))
    volume_threshold = float(params.get("volume_threshold", 0.9))
    checks = [
        ("종가 > 빠른 EMA", bool(last.get("close", 0.0) > last.get("ema_fast", float("inf"))), last.get("close"), last.get("ema_fast")),
        ("빠른 EMA > 느린 EMA", bool(last.get("ema_fast", 0.0) > last.get("ema_slow", float("inf"))), last.get("ema_fast"), last.get("ema_slow")),
        ("ADX 기준 통과", bool(last.get("adx", 0.0) >= adx_threshold), last.get("adx"), adx_threshold),
        ("돌파 기준선 상향", bool(last.get("close", 0.0) > last.get("breakout_high", float("inf"))), last.get("close"), last.get("breakout_high")),
        ("거래량 비율 통과", bool(last.get("volume_ratio", 0.0) >= volume_threshold), last.get("volume_ratio"), volume_threshold),
    ]
    rows = []
    for name, passed, current, threshold in checks:
        rows.append(
            {
                "조건": name,
                "통과": "예" if passed else "아니오",
                "현재값": "-" if pd.isna(current) else (f"{float(current):.2f}" if isinstance(current, (float, int)) else str(current)),
                "기준값": "-" if pd.isna(threshold) else (f"{float(threshold):.2f}" if isinstance(threshold, (float, int)) else str(threshold)),
            }
        )
    return pd.DataFrame(rows)


def _present_sweep_results(results: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    visible = results.copy()
    for column in ["total_return_pct", "win_rate_pct", "max_drawdown_pct"]:
        if column in visible.columns:
            visible[column] = visible[column].apply(lambda value: f"{float(value):.2f}%")
    if strategy_name == "research_trend":
        ordered = [
            "fast_ema",
            "slow_ema",
            "breakout_window",
            "atr_mult",
            "adx_threshold",
            "trades",
            "buy_signals",
            "sell_signals",
            "total_return_pct",
            "win_rate_pct",
            "max_drawdown_pct",
        ]
        rename_map = {
            "fast_ema": "빠른 EMA",
            "slow_ema": "느린 EMA",
            "breakout_window": "돌파 창",
            "atr_mult": "ATR 배수",
            "adx_threshold": "ADX 기준",
            "trades": "거래 수",
            "buy_signals": "매수 신호 수",
            "sell_signals": "매도 신호 수",
            "total_return_pct": "수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
        }
    elif strategy_name == "rsi_bb_double_bottom":
        ordered = [
            "rsi_len",
            "oversold",
            "bb_len",
            "bb_mult",
            "min_down_bars",
            "low_tolerance_pct",
            "confirm_bars",
            "risk_reward",
            "trades",
            "buy_signals",
            "sell_signals",
            "total_return_pct",
            "win_rate_pct",
            "max_drawdown_pct",
        ]
        rename_map = {
            "rsi_len": "RSI 길이",
            "oversold": "과매도 기준",
            "bb_len": "BB 길이",
            "bb_mult": "BB 배수",
            "min_down_bars": "연속 하락봉 수",
            "low_tolerance_pct": "바닥 허용치(%)",
            "confirm_bars": "확인 바 수",
            "risk_reward": "손익비",
            "trades": "거래 수",
            "buy_signals": "매수 신호 수",
            "sell_signals": "매도 신호 수",
            "total_return_pct": "수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
        }
    elif strategy_name == "relative_strength_rotation":
        ordered = [
            "rs_short_window",
            "rs_mid_window",
            "rs_long_window",
            "trend_ema_window",
            "breakout_window",
            "entry_score",
            "exit_score",
            "trades",
            "buy_signals",
            "sell_signals",
            "total_return_pct",
            "win_rate_pct",
            "max_drawdown_pct",
        ]
        rename_map = {
            "rs_short_window": "단기 RS 창",
            "rs_mid_window": "중기 RS 창",
            "rs_long_window": "장기 RS 창",
            "trend_ema_window": "추세 EMA",
            "breakout_window": "돌파 창",
            "entry_score": "진입 점수",
            "exit_score": "청산 점수",
            "trades": "거래 수",
            "buy_signals": "매수 신호 수",
            "sell_signals": "매도 신호 수",
            "total_return_pct": "수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
        }
    elif strategy_name == "flux_trend":
        ordered = [
            "ltf_len",
            "ltf_mult",
            "htf_len",
            "htf_mult",
            "htf_rule",
            "trades",
            "buy_signals",
            "sell_signals",
            "total_return_pct",
            "win_rate_pct",
            "max_drawdown_pct",
        ]
        rename_map = {
            "ltf_len": "단기 길이",
            "ltf_mult": "단기 배수",
            "htf_len": "상위 길이",
            "htf_mult": "상위 배수",
            "htf_rule": "상위 주기",
            "trades": "거래 수",
            "buy_signals": "매수 신호 수",
            "sell_signals": "매도 신호 수",
            "total_return_pct": "수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
        }
    else:
        ordered = [
            "ltf_len",
            "ltf_mult",
            "htf_len",
            "htf_mult",
            "htf_rule",
            "sensitivity",
            "atr_period",
            "trend_ema_length",
            "confirm_window",
            "trades",
            "buy_signals",
            "sell_signals",
            "total_return_pct",
            "win_rate_pct",
            "max_drawdown_pct",
        ]
        rename_map = {
            "ltf_len": "단기 길이",
            "ltf_mult": "단기 배수",
            "htf_len": "상위 길이",
            "htf_mult": "상위 배수",
            "htf_rule": "상위 주기",
            "sensitivity": "민감도",
            "atr_period": "ATR 기간",
            "trend_ema_length": "추세 EMA 길이",
            "confirm_window": "EMA 확인 창",
            "trades": "거래 수",
            "buy_signals": "매수 신호 수",
            "sell_signals": "매도 신호 수",
            "total_return_pct": "수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
        }
    columns = [column for column in ordered if column in visible.columns]
    return visible[columns].rename(columns=rename_map)


def _render_chart(frame: pd.DataFrame, strategy_name: str, bt_result: dict[str, object]):
    is_research = strategy_name == "research_trend"
    is_double_bottom = strategy_name == "rsi_bb_double_bottom"
    is_rotation = strategy_name == "relative_strength_rotation"
    rows = 4 if (is_research or is_rotation or is_double_bottom) else 3
    row_heights = [0.56, 0.16, 0.14, 0.14] if (is_research or is_rotation or is_double_bottom) else [0.64, 0.18, 0.18]
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.03)
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            increasing_line_color="#2dd4bf",
            decreasing_line_color="#fb7185",
            name="가격",
        ),
        row=1,
        col=1,
    )

    if is_research:
        for column, color in [
            ("ema_fast", "#60a5fa"),
            ("ema_slow", "#f59e0b"),
            ("atr_stop", "#f97316"),
            ("breakout_high", "rgba(45, 212, 191, 0.40)"),
            ("breakdown_low", "rgba(251, 113, 133, 0.30)"),
        ]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[column],
                        name=_SERIES_LABELS.get(column, column),
                        line={"color": color, "width": 1.4},
                    ),
                    row=1,
                    col=1,
                )
        if "adx" in frame:
            figure.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["adx"],
                    name=_SERIES_LABELS["adx"],
                    line={"color": "#a78bfa", "width": 1.6},
                ),
                row=3,
                col=1,
            )
            figure.add_hline(y=18, line={"color": "rgba(255,255,255,0.18)", "dash": "dot"}, row=3, col=1)
        if "strategy_score" in frame:
            figure.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["strategy_score"],
                    name=_SERIES_LABELS["strategy_score"],
                    line={"color": "#2dd4bf", "width": 1.6},
                ),
                row=4,
                col=1,
            )
    elif is_double_bottom:
        for column, color in [
            ("bb_basis", "#f8fafc"),
            ("bb_upper", "rgba(96, 165, 250, 0.60)"),
            ("bb_lower", "rgba(248, 113, 113, 0.65)"),
            ("trade_stop", "#fb7185"),
            ("take_profit", "#22c55e"),
        ]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[column],
                        name=_SERIES_LABELS.get(column, column),
                        line={"color": color, "width": 1.4},
                    ),
                    row=1,
                    col=1,
                )
        if "rsi" in frame:
            figure.add_trace(
                go.Scatter(x=frame.index, y=frame["rsi"], name=_SERIES_LABELS.get("rsi", "RSI"), line={"color": "#a78bfa", "width": 1.6}),
                row=3,
                col=1,
            )
            figure.add_hline(y=30, line={"color": "rgba(255,255,255,0.18)", "dash": "dot"}, row=3, col=1)
        if "macd_line" in frame:
            figure.add_trace(
                go.Scatter(x=frame.index, y=frame["macd_line"], name=_SERIES_LABELS.get("macd_line", "MACD"), line={"color": "#2dd4bf", "width": 1.4}),
                row=4,
                col=1,
            )
        if "macd_signal" in frame:
            figure.add_trace(
                go.Scatter(x=frame.index, y=frame["macd_signal"], name=_SERIES_LABELS.get("macd_signal", "MACD Signal"), line={"color": "#f59e0b", "width": 1.4}),
                row=4,
                col=1,
            )
    elif is_rotation:
        for column, color in [
            ("trend_ema", "#fde047"),
            ("atr_stop", "#fb7185"),
            ("breakout_high", "rgba(45, 212, 191, 0.40)"),
        ]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[column],
                        name=_SERIES_LABELS.get(column, column),
                        line={"color": color, "width": 1.4},
                    ),
                    row=1,
                    col=1,
                )
        figure.add_trace(go.Bar(x=frame.index, y=frame["volume"], name="거래량", marker_color="rgba(96,165,250,0.55)"), row=3, col=1)
        if "strategy_score" in frame:
            figure.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["strategy_score"],
                    name=_SERIES_LABELS.get("strategy_score", "전략 점수"),
                    line={"color": "#2dd4bf", "width": 1.6},
                ),
                row=4,
                col=1,
            )
    else:
        line_colors = {
            "ltf_upper": "#60a5fa",
            "ltf_lower": "#38bdf8",
            "ltf_basis": "#2dd4bf",
            "htf_upper": "#f59e0b",
            "htf_lower": "#f97316",
            "trend_ema": "#fde047",
            "atr_stop": "#fb7185",
        }
        for column in ["ltf_upper", "ltf_lower", "ltf_basis", "htf_upper", "htf_lower", "trend_ema", "atr_stop"]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[column],
                        name=_SERIES_LABELS.get(column, column),
                        line={"width": 1.4, "color": line_colors.get(column)},
                    ),
                    row=1,
                    col=1,
                )
        figure.add_trace(
            go.Bar(x=frame.index, y=frame["volume"], name="거래량", marker_color="rgba(96,165,250,0.55)"),
            row=3,
            col=1,
        )

    for column, color, symbol, label in [
        ("buy_signal", "#22c55e", "triangle-up-open", "매수 신호"),
        ("sell_signal", "#f43f5e", "triangle-down-open", "매도 신호"),
    ]:
        if column in frame:
            hits = frame[frame[column]]
            if not hits.empty:
                marker_y = hits["close"]
                if column == "buy_signal" and "low" in hits:
                    marker_y = hits["low"] * 0.995
                elif column == "sell_signal" and "high" in hits:
                    marker_y = hits["high"] * 1.005
                figure.add_trace(
                    go.Scatter(
                        x=hits.index,
                        y=marker_y,
                        mode="markers",
                        name=label,
                        marker={"symbol": symbol, "size": 15, "color": color, "line": {"color": "white", "width": 1.4}},
                        hovertemplate="%{x}<br>신호가=%{y:.0f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    for column, color, symbol, label in [
        ("rebound_marker", "#38bdf8", "circle-open", "첫 반등"),
        ("second_bottom_marker", "#f59e0b", "circle", "두 번째 바닥"),
    ]:
        if column in frame:
            hits = frame[frame[column]]
            if not hits.empty:
                figure.add_trace(
                    go.Scatter(
                        x=hits.index,
                        y=hits["low"] * 0.995,
                        mode="markers",
                        name=label,
                        marker={"symbol": symbol, "size": 11, "color": color, "line": {"color": "white", "width": 1.2}},
                        hovertemplate="%{x}<br>가격=%{y:.0f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    simulated_trades = extract_backtest_trade_events(frame)
    for side, color, label in [
        ("BUY", "#22c55e", "백테스트 매수"),
        ("SELL", "#f43f5e", "백테스트 매도"),
    ]:
        points = [event for event in simulated_trades if event["side"] == side]
        if points:
            figure.add_trace(
                go.Scatter(
                    x=[event["ts"] for event in points],
                    y=[event["price"] for event in points],
                    mode="markers",
                    name=label,
                    marker={"symbol": "diamond", "size": 12, "color": color, "line": {"color": "#0f172a", "width": 1.3}},
                    hovertemplate="%{x}<br>체결가=%{y:.0f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    equity = bt_result.get("equity")
    if equity is not None:
        figure.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name="누적 수익곡선", line={"color": "#f8fafc", "width": 1.8}),
            row=2,
            col=1,
        )

    figure.update_yaxes(title_text="가격", row=1, col=1)
    figure.update_yaxes(title_text="수익곡선", row=2, col=1)
    if is_research:
        figure.update_yaxes(title_text="ADX", row=3, col=1)
        figure.update_yaxes(title_text="점수", row=4, col=1)
    elif is_double_bottom:
        figure.update_yaxes(title_text="RSI", row=3, col=1)
        figure.update_yaxes(title_text="MACD", row=4, col=1)
    elif is_rotation:
        figure.update_yaxes(title_text="거래량", row=3, col=1)
        figure.update_yaxes(title_text="점수", row=4, col=1)
    else:
        figure.update_yaxes(title_text="거래량", row=3, col=1)

    return apply_chart_theme(figure, height=920 if (is_research or is_rotation or is_double_bottom) else 860)


def render_backtest():
    page_intro(
        "백테스트",
        "전략 연구 랩",
        "실시간 데스크와 같은 전략 엔진으로 먼저 검증해 보세요. 신호와 실제 백테스트 체결 포인트를 함께 보여줍니다.",
    )

    if not api:
        st.error("API가 초기화되지 않았습니다.")
        return

    meta = _markets()
    if meta.empty:
        st.error("KRW 마켓 메타데이터를 불러오지 못했습니다.")
        return

    ticker_frame = _tickers(tuple(meta["market"].tolist()))
    market_frame = meta.merge(ticker_frame, on="market", how="left")
    market_frame["acc_trade_price_24h"] = pd.to_numeric(market_frame["acc_trade_price_24h"], errors="coerce")
    market_frame = market_frame.sort_values("acc_trade_price_24h", ascending=False)

    left, right = st.columns([1.05, 2.35], gap="large")
    with left:
        selected_market = _market_panel(market_frame)

    with right:
        st.subheader("차트 데스크")
        control_col1, control_col2, control_col3, control_col4, control_col5 = st.columns([1, 1, 1.1, 1.1, 1])
        interval_options = ["minute15", "minute30", "minute60", "minute240", "day"]
        interval = control_col1.selectbox("주기", interval_options, index=2, format_func=_interval_text)
        count = int(control_col2.number_input("캔들 수", 120, 1200, 360, 20))
        fee = float(control_col3.number_input("수수료", 0.0, 0.01, 0.0005, 0.0001, format="%.4f"))
        slippage_bps = float(control_col4.number_input("슬리피지 (bps)", 0.0, 100.0, 3.0, 0.5))
        auto_run = control_col5.checkbox("자동 갱신", value=True)
        strategy_name, strategy_params = _strategy_controls()
        notice = st.session_state.pop("bt_apply_notice", "")
        if notice:
            st.success(str(notice))
        force_run = bool(st.session_state.pop("bt_force_run", False))
        run_now = st.button("분석 실행", use_container_width=True)

        if not selected_market:
            st.info("왼쪽 패널에서 종목을 선택해 주세요.")
            return

        previous_key = st.session_state.get("bt_context")
        current_key = {
            "market": selected_market,
            "interval": interval,
            "count": count,
            "fee": fee,
            "slippage_bps": slippage_bps,
            "strategy_name": strategy_name,
            "strategy_params": strategy_params,
        }
        if run_now or force_run or (auto_run and previous_key != current_key):
            raw = _candles(selected_market, interval, count)
            if raw.empty:
                st.warning("해당 종목의 캔들 데이터를 불러오지 못했습니다.")
                return
            try:
                frame = build_strategy_frame(
                    raw[["open", "high", "low", "close", "volume"]],
                    strategy_name=strategy_name,
                    params=strategy_params,
                    flux_indicator=flux_indicator,
                    flux_indicator_with_ema=flux_indicator_with_ema,
                )
            except Exception as exc:
                st.error(f"전략 차트 생성에 실패했습니다: {exc}")
                return
            st.session_state["bt_raw_frame"] = raw[["open", "high", "low", "close", "volume"]].copy()
            st.session_state["bt_frame"] = frame
            st.session_state["bt_context"] = current_key

        frame = st.session_state.get("bt_frame")
        if frame is None or frame.empty:
            st.info("분석을 실행하면 차트가 그려집니다.")
            return

        bt_result = backtest_signal_frame(frame, fee=fee, slippage_bps=slippage_bps)
        last_row = frame.iloc[-1]
        metric_cols = st.columns(5)
        metric_cols[0].metric("전략", strategy_label(strategy_name))
        metric_cols[1].metric("거래 횟수", int(bt_result["trades"]))
        metric_cols[2].metric("수익률", f"{float(bt_result['total_return_pct']):.2f}%")
        metric_cols[3].metric("승률", f"{float(bt_result['win_rate_pct']):.1f}%")
        metric_cols[4].metric("최대 낙폭", f"{float(bt_result['max_drawdown_pct']):.2f}%")
        detail_cols = st.columns(3)
        detail_cols[0].metric("마지막 가격", fmt_full_number(last_row.get("close"), 0))
        detail_cols[1].metric("전략 점수", f"{float(last_row.get('strategy_score', 0.0)):.2f}")
        detail_cols[2].metric(
            "현재 신호",
            "매수" if bool(last_row.get("buy_signal")) else "매도" if bool(last_row.get("sell_signal")) else "대기",
        )
        signal_cols = st.columns(4)
        buy_hits = frame.index[frame["buy_signal"]].tolist()
        sell_hits = frame.index[frame["sell_signal"]].tolist()
        signal_cols[0].metric("매수 신호 수", len(buy_hits))
        signal_cols[1].metric("매도 신호 수", len(sell_hits))
        signal_cols[2].metric("최근 매수 신호", _format_signal_time(buy_hits[-1] if buy_hits else None))
        signal_cols[3].metric("최근 매도 신호", _format_signal_time(sell_hits[-1] if sell_hits else None))

        if strategy_name == "research_trend":
            st.caption("최근 캔들 기준으로 어떤 매수 조건이 통과했고 막혔는지 바로 확인할 수 있습니다.")
            st.dataframe(_research_condition_table(frame, strategy_params), use_container_width=True, hide_index=True)
        elif not buy_hits:
            st.info("현재 선택 구간에는 매수 신호가 없습니다. 기간을 넓히거나 다른 전략/파라미터를 비교해 보세요.")

        timeline = _signal_timeline(frame)
        if not timeline.empty:
            st.dataframe(timeline, use_container_width=True, hide_index=True)

        st.caption("빈 삼각형은 전략 신호, 다이아몬드는 백테스트 체결입니다.")
        st.plotly_chart(_render_chart(frame, strategy_name, bt_result), use_container_width=True)

        tail = frame.tail(20).copy()
        columns = ["close", "strategy_score", "buy_signal", "sell_signal"]
        if strategy_name == "research_trend":
            columns.extend(["ema_fast", "ema_slow", "adx", "atr"])
        elif strategy_name == "rsi_bb_double_bottom":
            columns.extend(["rsi", "bb_lower", "bb_upper", "trade_stop", "take_profit", "rebound_marker", "second_bottom_marker"])
        elif strategy_name == "relative_strength_rotation":
            columns.extend(["rs_short", "rs_mid", "rs_long", "trend_ema", "volume_ratio", "atr_stop"])
        elif strategy_name == "flux_ema_filter":
            columns.extend(["strength", "ema_buy", "ema_sell", "flux_buy_signal", "flux_sell_signal"])
        visible = tail[[column for column in columns if column in tail.columns]].rename(
            columns={
                "close": "종가",
                "strategy_score": "전략 점수",
                "rsi": "RSI",
                "bb_lower": "BB 하단",
                "bb_upper": "BB 상단",
                "trade_stop": "전략 손절선",
                "take_profit": "전략 익절선",
                "rebound_marker": "첫 반등",
                "second_bottom_marker": "두 번째 바닥",
                "rs_short": "단기 RS",
                "rs_mid": "중기 RS",
                "rs_long": "장기 RS",
                "trend_ema": "추세 EMA",
                "volume_ratio": "거래량 비율",
                "buy_signal": "매수 신호",
                "sell_signal": "매도 신호",
                "ema_fast": "빠른 EMA",
                "ema_slow": "느린 EMA",
                "adx": "ADX",
                "atr": "ATR",
                "strength": "필터 강도",
                "ema_buy": "EMA 매수 확인",
                "ema_sell": "EMA 매도 확인",
                "flux_buy_signal": "원본 플럭스 매수",
                "flux_sell_signal": "원본 플럭스 매도",
            }
        )
        st.dataframe(visible, use_container_width=True)

        with st.expander("전략 비교 랭킹", expanded=False):
            st.caption("같은 종목과 주기에서 전략별 현재 설정을 한 번에 백테스트해 순위를 비교합니다.")
            run_compare = st.button("전략 비교 실행", use_container_width=True)
            if run_compare:
                raw_frame = st.session_state.get("bt_raw_frame")
                if raw_frame is None or raw_frame.empty:
                    st.warning("먼저 기본 분석을 실행해 주세요.")
                else:
                    compare_results = compare_strategy_backtests(
                        raw_frame,
                        strategies=[
                            {"strategy_name": name, "params": _params_for_strategy(name)}
                            for name in strategy_options(flux_indicator is not None, flux_indicator_with_ema is not None)
                        ],
                        fee=fee,
                        slippage_bps=slippage_bps,
                        flux_indicator=flux_indicator,
                        flux_indicator_with_ema=flux_indicator_with_ema,
                    )
                    st.session_state["bt_compare_results"] = compare_results
                    st.session_state["bt_compare_meta"] = {
                        "market": selected_market,
                        "interval": interval,
                        "count": count,
                        "fee": fee,
                        "slippage_bps": slippage_bps,
                    }

            compare_results = st.session_state.get("bt_compare_results")
            compare_meta = st.session_state.get("bt_compare_meta") or {}
            if isinstance(compare_results, pd.DataFrame) and not compare_results.empty:
                compare_match = (
                    compare_meta.get("market") == selected_market
                    and compare_meta.get("interval") == interval
                    and int(compare_meta.get("count") or 0) == count
                    and float(compare_meta.get("fee") or 0.0) == fee
                    and float(compare_meta.get("slippage_bps") or 0.0) == slippage_bps
                )
                if compare_match:
                    top = compare_results.iloc[0]
                    top_params = dict(top.get("params") or {})
                    compare_cols = st.columns(4)
                    compare_cols[0].metric("1위 전략", str(top.get("strategy_label") or "-"))
                    compare_cols[1].metric("1위 수익률", f"{float(top.get('return_pct') or 0.0):.2f}%")
                    compare_cols[2].metric("1위 최대 낙폭", f"{float(top.get('max_drawdown_pct') or 0.0):.2f}%")
                    compare_cols[3].metric("1위 설정", _strategy_param_summary(str(top.get("strategy_name") or ""), top_params))
                    st.dataframe(_present_compare_results(compare_results), use_container_width=True, hide_index=True)
                else:
                    st.info("종목이나 주기가 바뀌어서 이전 비교 결과를 숨겼습니다. 다시 실행해 주세요.")

        with st.expander("파라미터 스윕", expanded=False):
            st.caption("현재 선택한 종목과 주기로 여러 조합을 자동 비교합니다. 기본값은 현재 전략 설정을 기준으로 합니다.")
            if strategy_name == "research_trend":
                sweep_cols1 = st.columns(3)
                sweep_cols2 = st.columns(2)
                grid = {
                    "fast_ema": _parse_sweep_values(
                        sweep_cols1[0].text_input("빠른 EMA 후보", "12, 21, 34", key="bt_sweep_fast_ema"),
                        int,
                    ),
                    "slow_ema": _parse_sweep_values(
                        sweep_cols1[1].text_input("느린 EMA 후보", "55, 89", key="bt_sweep_slow_ema"),
                        int,
                    ),
                    "breakout_window": _parse_sweep_values(
                        sweep_cols1[2].text_input("돌파 창 후보", "14, 20, 28", key="bt_sweep_breakout"),
                        int,
                    ),
                    "atr_mult": _parse_sweep_values(
                        sweep_cols2[0].text_input("ATR 배수 후보", "2.0, 2.5, 3.0", key="bt_sweep_atr_mult"),
                        float,
                    ),
                    "adx_threshold": _parse_sweep_values(
                        sweep_cols2[1].text_input("ADX 기준 후보", "16, 18, 20", key="bt_sweep_adx_threshold"),
                        float,
                    ),
                }
            elif strategy_name == "rsi_bb_double_bottom":
                sweep_cols1 = st.columns(4)
                sweep_cols2 = st.columns(4)
                grid = {
                    "rsi_len": _parse_sweep_values(
                        sweep_cols1[0].text_input("RSI 길이 후보", "10, 14, 18", key="bt_sweep_db_rsi_len"),
                        int,
                    ),
                    "oversold": _parse_sweep_values(
                        sweep_cols1[1].text_input("과매도 후보", "25, 30, 35", key="bt_sweep_db_oversold"),
                        float,
                    ),
                    "bb_len": _parse_sweep_values(
                        sweep_cols1[2].text_input("BB 길이 후보", "18, 20, 24", key="bt_sweep_db_bb_len"),
                        int,
                    ),
                    "bb_mult": _parse_sweep_values(
                        sweep_cols1[3].text_input("BB 배수 후보", "1.8, 2.0, 2.2", key="bt_sweep_db_bb_mult"),
                        float,
                    ),
                    "min_down_bars": _parse_sweep_values(
                        sweep_cols2[0].text_input("연속 하락봉 후보", "2, 3", key="bt_sweep_db_min_down_bars"),
                        int,
                    ),
                    "low_tolerance_pct": _parse_sweep_values(
                        sweep_cols2[1].text_input("바닥 허용치 후보", "0.5, 1.0, 1.5", key="bt_sweep_db_low_tolerance_pct"),
                        float,
                    ),
                    "confirm_bars": _parse_sweep_values(
                        sweep_cols2[2].text_input("확인 바 후보", "3, 4, 5", key="bt_sweep_db_confirm_bars"),
                        int,
                    ),
                    "risk_reward": _parse_sweep_values(
                        sweep_cols2[3].text_input("손익비 후보", "1.5, 2.0, 2.5", key="bt_sweep_db_risk_reward"),
                        float,
                    ),
                }
            elif strategy_name == "relative_strength_rotation":
                sweep_cols1 = st.columns(3)
                sweep_cols2 = st.columns(3)
                sweep_cols3 = st.columns(2)
                grid = {
                    "rs_short_window": _parse_sweep_values(
                        sweep_cols1[0].text_input("단기 RS 후보", "8, 10, 14", key="bt_sweep_rs_short_window"),
                        int,
                    ),
                    "rs_mid_window": _parse_sweep_values(
                        sweep_cols1[1].text_input("중기 RS 후보", "20, 30, 45", key="bt_sweep_rs_mid_window"),
                        int,
                    ),
                    "rs_long_window": _parse_sweep_values(
                        sweep_cols1[2].text_input("장기 RS 후보", "60, 90, 120", key="bt_sweep_rs_long_window"),
                        int,
                    ),
                    "trend_ema_window": _parse_sweep_values(
                        sweep_cols2[0].text_input("추세 EMA 후보", "34, 55, 80", key="bt_sweep_rs_trend_ema_window"),
                        int,
                    ),
                    "breakout_window": _parse_sweep_values(
                        sweep_cols2[1].text_input("돌파 창 후보", "14, 20, 28", key="bt_sweep_rs_breakout_window"),
                        int,
                    ),
                    "entry_score": _parse_sweep_values(
                        sweep_cols2[2].text_input("진입 점수 후보", "6, 8, 10", key="bt_sweep_rs_entry_score"),
                        float,
                    ),
                    "exit_score": _parse_sweep_values(
                        sweep_cols3[0].text_input("청산 점수 후보", "0, 2, 4", key="bt_sweep_rs_exit_score"),
                        float,
                    ),
                    "atr_mult": _parse_sweep_values(
                        sweep_cols3[1].text_input("ATR 배수 후보", "1.8, 2.2, 2.6", key="bt_sweep_rs_atr_mult"),
                        float,
                    ),
                }
            elif strategy_name == "flux_trend":
                sweep_cols1 = st.columns(3)
                sweep_cols2 = st.columns(2)
                grid = {
                    "ltf_len": _parse_sweep_values(
                        sweep_cols1[0].text_input("단기 길이 후보", "14, 20, 28", key="bt_sweep_ltf_len"),
                        int,
                    ),
                    "ltf_mult": _parse_sweep_values(
                        sweep_cols1[1].text_input("단기 배수 후보", "1.5, 2.0, 2.5", key="bt_sweep_ltf_mult"),
                        float,
                    ),
                    "htf_len": _parse_sweep_values(
                        sweep_cols1[2].text_input("상위 길이 후보", "20, 30, 40", key="bt_sweep_htf_len"),
                        int,
                    ),
                    "htf_mult": _parse_sweep_values(
                        sweep_cols2[0].text_input("상위 배수 후보", "2.0, 2.25, 2.5", key="bt_sweep_htf_mult"),
                        float,
                    ),
                    "htf_rule": _parse_htf_rule_values(
                        sweep_cols2[1].text_input("상위 주기 후보", "60T, 120T, 240T", key="bt_sweep_htf_rule")
                    ),
                }
            else:
                sweep_cols1 = st.columns(3)
                sweep_cols2 = st.columns(3)
                sweep_cols3 = st.columns(3)
                grid = {
                    "ltf_len": _parse_sweep_values(
                        sweep_cols1[0].text_input("단기 길이 후보", "14, 20", key="bt_sweep_flux_ema_ltf_len"),
                        int,
                    ),
                    "ltf_mult": _parse_sweep_values(
                        sweep_cols1[1].text_input("단기 배수 후보", "1.5, 2.0", key="bt_sweep_flux_ema_ltf_mult"),
                        float,
                    ),
                    "htf_len": _parse_sweep_values(
                        sweep_cols1[2].text_input("상위 길이 후보", "20, 30", key="bt_sweep_flux_ema_htf_len"),
                        int,
                    ),
                    "htf_mult": _parse_sweep_values(
                        sweep_cols2[0].text_input("상위 배수 후보", "2.0, 2.25", key="bt_sweep_flux_ema_htf_mult"),
                        float,
                    ),
                    "htf_rule": _parse_htf_rule_values(
                        sweep_cols2[1].text_input("상위 주기 후보", "60T, 120T", key="bt_sweep_flux_ema_htf_rule")
                    ),
                    "sensitivity": _parse_sweep_values(
                        sweep_cols2[2].text_input("민감도 후보", "2, 3", key="bt_sweep_flux_ema_sensitivity"),
                        int,
                    ),
                    "atr_period": _parse_sweep_values(
                        sweep_cols3[0].text_input("ATR 기간 후보", "2, 3", key="bt_sweep_flux_ema_atr_period"),
                        int,
                    ),
                    "trend_ema_length": _parse_sweep_values(
                        sweep_cols3[1].text_input("추세 EMA 후보", "240", key="bt_sweep_flux_ema_length"),
                        int,
                    ),
                    "confirm_window": _parse_sweep_values(
                        sweep_cols3[2].text_input("EMA 확인 창 후보", "8", key="bt_sweep_flux_ema_confirm_window"),
                        int,
                    ),
                }

            combo_count = parameter_grid_size(grid)
            st.caption(f"총 조합 수: {combo_count}개")
            run_sweep = st.button("파라미터 스윕 실행", use_container_width=True)
            if combo_count == 0:
                st.info("후보 값을 한 개 이상 입력해 주세요.")
            elif combo_count > 240:
                st.warning("조합 수가 너무 많습니다. 240개 이하로 줄여 주세요.")
            elif run_sweep:
                raw_frame = st.session_state.get("bt_raw_frame")
                if raw_frame is None or raw_frame.empty:
                    st.warning("먼저 기본 분석을 실행해 주세요.")
                else:
                    base_params = dict(strategy_params)
                    if strategy_name == "research_trend":
                        results = sweep_research_trend_parameters(
                            raw_frame,
                            base_params={key: value for key, value in base_params.items() if isinstance(value, (int, float))},
                            candidate_grid=grid,
                            fee=fee,
                            slippage_bps=slippage_bps,
                        )
                    else:
                        results = sweep_strategy_parameters(
                            raw_frame,
                            strategy_name=strategy_name,
                            base_params=base_params,
                            candidate_grid=grid,
                            fee=fee,
                            slippage_bps=slippage_bps,
                            flux_indicator=flux_indicator,
                            flux_indicator_with_ema=flux_indicator_with_ema,
                        )
                    st.session_state["bt_sweep_results"] = results
                    st.session_state["bt_sweep_meta"] = {
                        "market": selected_market,
                        "interval": interval,
                        "count": count,
                        "strategy_name": strategy_name,
                        "fee": fee,
                        "slippage_bps": slippage_bps,
                    }

            sweep_results = st.session_state.get("bt_sweep_results")
            sweep_meta = st.session_state.get("bt_sweep_meta") or {}
            if isinstance(sweep_results, pd.DataFrame) and not sweep_results.empty:
                current_match = (
                    sweep_meta.get("market") == selected_market
                    and sweep_meta.get("interval") == interval
                    and int(sweep_meta.get("count") or 0) == count
                    and sweep_meta.get("strategy_name") == strategy_name
                    and float(sweep_meta.get("fee") or 0.0) == fee
                    and float(sweep_meta.get("slippage_bps") or 0.0) == slippage_bps
                )
                if current_match:
                    best = sweep_results.iloc[0]
                    best_cols = st.columns(4)
                    best_cols[0].metric("1위 수익률", f"{float(best['total_return_pct']):.2f}%")
                    best_cols[1].metric("1위 최대 낙폭", f"{float(best['max_drawdown_pct']):.2f}%")
                    best_cols[2].metric("1위 거래 수", int(best["trades"]))
                    if strategy_name == "research_trend":
                        label = f"EMA {int(best['fast_ema'])}/{int(best['slow_ema'])} · 돌파 {int(best['breakout_window'])}"
                    elif strategy_name == "rsi_bb_double_bottom":
                        label = (
                            f"RSI {int(best['rsi_len'])} · 과매도 {float(best['oversold']):.1f} · "
                            f"BB {int(best['bb_len'])}/{float(best['bb_mult']):.1f} · RR {float(best['risk_reward']):.2f}"
                        )
                    elif strategy_name == "relative_strength_rotation":
                        label = (
                            f"RS {int(best['rs_short_window'])}/{int(best['rs_mid_window'])}/{int(best['rs_long_window'])} · "
                            f"EMA {int(best['trend_ema_window'])} · 진입 {float(best['entry_score']):.1f}"
                        )
                    elif strategy_name == "flux_trend":
                        label = f"LTF {int(best['ltf_len'])}/{float(best['ltf_mult']):.2f} · HTF {best['htf_rule']}"
                    else:
                        label = (
                            f"LTF {int(best['ltf_len'])}/{float(best['ltf_mult']):.2f} · "
                            f"EMA {int(best['trend_ema_length'])} · 민감도 {int(best['sensitivity'])} · "
                            f"확인창 {int(best.get('confirm_window', 8))}"
                        )
                    best_cols[3].metric("1위 조합", label)
                    if st.button("1위 설정을 현재 전략에 적용", key=f"bt_apply_best_{strategy_name}", use_container_width=True):
                        _apply_params_to_widgets(strategy_name, best.to_dict())
                        st.session_state["bt_apply_notice"] = f"{strategy_label(strategy_name)} 1위 조합을 현재 설정에 반영했습니다."
                        st.session_state["bt_force_run"] = True
                        st.rerun()
                    st.dataframe(_present_sweep_results(sweep_results.head(20), strategy_name), use_container_width=True, hide_index=True)
                else:
                    st.info("종목, 주기, 또는 전략이 바뀌어서 이전 스윕 결과를 숨겼습니다. 다시 실행해 주세요.")

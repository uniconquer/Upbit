from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from strategy import backtest_signal_frame, extract_backtest_trade_events
from strategy_engine import build_strategy_frame, strategy_description, strategy_label, strategy_options
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
    options = strategy_options(flux_indicator is not None)
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
            params["fast_ema"] = row1[0].number_input("빠른 EMA", 5, 100, 21, 1)
            params["slow_ema"] = row1[1].number_input("느린 EMA", 10, 240, 55, 1)
            params["breakout_window"] = row1[2].number_input("돌파 창", 5, 120, 20, 1)
            params["exit_window"] = row1[3].number_input("청산 창", 3, 80, 10, 1)
            row2 = st.columns(4)
            params["atr_window"] = row2[0].number_input("ATR 창", 5, 50, 14, 1)
            params["atr_mult"] = row2[1].number_input("ATR 배수", 1.0, 6.0, 2.5, 0.1)
            params["adx_window"] = row2[2].number_input("ADX 창", 5, 50, 14, 1)
            params["adx_threshold"] = row2[3].number_input("ADX 기준", 5.0, 40.0, 18.0, 0.5)
            row3 = st.columns(3)
            params["momentum_window"] = row3[0].number_input("모멘텀 창", 5, 80, 20, 1)
            params["volume_window"] = row3[1].number_input("거래량 창", 5, 80, 20, 1)
            params["volume_threshold"] = row3[2].number_input("거래량 비율", 0.1, 3.0, 0.9, 0.1)
    else:
        with st.expander("플럭스 추세 밴드 설정", expanded=False):
            row = st.columns(5)
            params["ltf_len"] = row[0].number_input("단기 기준 길이", 5, 400, 20, 1)
            params["ltf_mult"] = row[1].number_input("단기 밴드 배수", 0.1, 10.0, 2.0, 0.1)
            params["htf_len"] = row[2].number_input("상위 주기 길이", 5, 400, 20, 1)
            params["htf_mult"] = row[3].number_input("상위 밴드 배수", 0.1, 10.0, 2.25, 0.1)
            htf = row[4].selectbox("상위 주기", ["30m", "60m", "120m", "240m", "1D"], index=1)
            params["htf_rule"] = htf.replace("m", "T") if htf.endswith("m") else "1D"
    return strategy_name, params


def _render_chart(frame: pd.DataFrame, strategy_name: str, bt_result: dict[str, object]):
    is_research = strategy_name == "research_trend"
    rows = 4 if is_research else 3
    row_heights = [0.56, 0.16, 0.14, 0.14] if is_research else [0.64, 0.18, 0.18]
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
    else:
        for column in ["ltf_upper", "ltf_lower", "ltf_basis", "htf_upper", "htf_lower"]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(
                        x=frame.index,
                        y=frame[column],
                        name=_SERIES_LABELS.get(column, column),
                        line={"width": 1.4},
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
                figure.add_trace(
                    go.Scatter(
                        x=hits.index,
                        y=hits["close"],
                        mode="markers",
                        name=label,
                        marker={"symbol": symbol, "size": 13, "color": color, "line": {"color": "white", "width": 1.4}},
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
    else:
        figure.update_yaxes(title_text="거래량", row=3, col=1)

    return apply_chart_theme(figure, height=920 if is_research else 860)


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
        if run_now or (auto_run and previous_key != current_key):
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
                )
            except Exception as exc:
                st.error(f"전략 차트 생성에 실패했습니다: {exc}")
                return
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

        st.caption("빈 삼각형은 전략 신호, 다이아몬드는 백테스트 체결입니다.")
        st.plotly_chart(_render_chart(frame, strategy_name, bt_result), use_container_width=True)

        tail = frame.tail(20).copy()
        columns = ["close", "strategy_score", "buy_signal", "sell_signal"]
        if strategy_name == "research_trend":
            columns.extend(["ema_fast", "ema_slow", "adx", "atr"])
        visible = tail[[column for column in columns if column in tail.columns]].rename(
            columns={
                "close": "종가",
                "strategy_score": "전략 점수",
                "buy_signal": "매수 신호",
                "sell_signal": "매도 신호",
                "ema_fast": "빠른 EMA",
                "ema_slow": "느린 EMA",
                "adx": "ADX",
                "atr": "ATR",
            }
        )
        st.dataframe(visible, use_container_width=True)

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from strategy import backtest_signal_frame
from strategy_engine import build_strategy_frame, strategy_label, strategy_options
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
    st.subheader("Markets")
    query = st.text_input("Search", "", placeholder="BTC, ETH, 비트코인", key="bt_search")
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
                    "market": row["market"],
                    "name": row["korean_name"],
                    "last": fmt_full_number(row.get("trade_price"), 0),
                    "turnover": fmt_full_number(row.get("acc_trade_price_24h"), 0),
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
            st.session_state["bt_sel_market"] = selected[0]["market"]
    else:
        options = [row["market"] for row in rows]
        if options:
            current = st.session_state.get("bt_sel_market", options[0])
            if current not in options:
                current = options[0]
            st.session_state["bt_sel_market"] = st.selectbox("Market", options, index=options.index(current))
        elif query.strip():
            st.info("No markets matched the search.")

    return st.session_state.get("bt_sel_market")


def _strategy_controls() -> tuple[str, dict[str, float | int | str]]:
    options = strategy_options(flux_indicator is not None)
    current = st.session_state.get("bt_strategy_name", options[0])
    index = options.index(current) if current in options else 0
    strategy_name = st.selectbox(
        "Strategy",
        options,
        index=index,
        format_func=strategy_label,
        key="bt_strategy_name",
    )

    params: dict[str, float | int | str] = {}
    if strategy_name == "research_trend":
        with st.expander("Research Trend Parameters", expanded=False):
            row1 = st.columns(4)
            params["fast_ema"] = row1[0].number_input("Fast EMA", 5, 100, 21, 1)
            params["slow_ema"] = row1[1].number_input("Slow EMA", 10, 240, 55, 1)
            params["breakout_window"] = row1[2].number_input("Breakout Window", 5, 120, 20, 1)
            params["exit_window"] = row1[3].number_input("Exit Window", 3, 80, 10, 1)
            row2 = st.columns(4)
            params["atr_window"] = row2[0].number_input("ATR Window", 5, 50, 14, 1)
            params["atr_mult"] = row2[1].number_input("ATR Mult", 1.0, 6.0, 2.5, 0.1)
            params["adx_window"] = row2[2].number_input("ADX Window", 5, 50, 14, 1)
            params["adx_threshold"] = row2[3].number_input("ADX Threshold", 5.0, 40.0, 18.0, 0.5)
            row3 = st.columns(3)
            params["momentum_window"] = row3[0].number_input("Momentum Window", 5, 80, 20, 1)
            params["volume_window"] = row3[1].number_input("Volume Window", 5, 80, 20, 1)
            params["volume_threshold"] = row3[2].number_input("Volume Ratio", 0.1, 3.0, 0.9, 0.1)
    else:
        with st.expander("Flux Parameters", expanded=False):
            row = st.columns(5)
            params["ltf_len"] = row[0].number_input("LTF Len", 5, 400, 20, 1)
            params["ltf_mult"] = row[1].number_input("LTF Mult", 0.1, 10.0, 2.0, 0.1)
            params["htf_len"] = row[2].number_input("HTF Len", 5, 400, 20, 1)
            params["htf_mult"] = row[3].number_input("HTF Mult", 0.1, 10.0, 2.25, 0.1)
            htf = row[4].selectbox("HTF Rule", ["30m", "60m", "120m", "240m", "1D"], index=1)
            params["htf_rule"] = htf.replace("m", "T") if htf.endswith("m") else "1D"
    return strategy_name, params


def _render_chart(frame: pd.DataFrame, strategy_name: str, bt_result: dict[str, object]):
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.2, 0.18],
        vertical_spacing=0.03,
    )
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            increasing_line_color="#2dd4bf",
            decreasing_line_color="#fb7185",
            name="Price",
        ),
        row=1,
        col=1,
    )

    if strategy_name == "research_trend":
        for column, color in [
            ("ema_fast", "#60a5fa"),
            ("ema_slow", "#f59e0b"),
            ("atr_stop", "#f97316"),
            ("breakout_high", "rgba(45, 212, 191, 0.40)"),
            ("breakdown_low", "rgba(251, 113, 133, 0.30)"),
        ]:
            if column in frame:
                figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column, line={"color": color, "width": 1.4}), row=1, col=1)
        figure.add_trace(go.Scatter(x=frame.index, y=frame["adx"], name="ADX", line={"color": "#a78bfa", "width": 1.6}), row=2, col=1)
        figure.add_hline(y=18, line={"color": "rgba(255,255,255,0.18)", "dash": "dot"}, row=2, col=1)
        figure.add_trace(go.Scatter(x=frame.index, y=frame["strategy_score"], name="Strategy Score", line={"color": "#2dd4bf", "width": 1.6}), row=3, col=1)
    else:
        for column in ["ltf_upper", "ltf_lower", "ltf_basis", "htf_upper", "htf_lower"]:
            if column in frame:
                figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column, line={"width": 1.4}), row=1, col=1)
        figure.add_trace(go.Bar(x=frame.index, y=frame["volume"], name="Volume", marker_color="rgba(96,165,250,0.55)"), row=3, col=1)

    for column, color, symbol in [
        ("buy_signal", "#22c55e", "triangle-up"),
        ("sell_signal", "#f43f5e", "triangle-down"),
    ]:
        if column in frame:
            hits = frame[frame[column]]
            if not hits.empty:
                figure.add_trace(
                    go.Scatter(
                        x=hits.index,
                        y=hits["close"],
                        mode="markers",
                        name=column,
                        marker={"symbol": symbol, "size": 11, "color": color, "line": {"color": "white", "width": 1}},
                    ),
                    row=1,
                    col=1,
                )

    equity = bt_result.get("equity")
    if equity is not None:
        figure.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line={"color": "#f8fafc", "width": 1.8}), row=2, col=1)

    return apply_chart_theme(figure, height=820)


def render_backtest():
    page_intro(
        "Backtest Lab",
        "Modern charting with safer strategy research",
        "Use the same strategy engine that feeds live simulation. Compare a research trend model with the existing flux-style trend view before promoting anything toward live execution.",
    )

    if not api:
        st.error("API is not initialized.")
        return

    meta = _markets()
    if meta.empty:
        st.error("Failed to load KRW market metadata.")
        return

    ticker_frame = _tickers(tuple(meta["market"].tolist()))
    market_frame = meta.merge(ticker_frame, on="market", how="left")
    market_frame["acc_trade_price_24h"] = pd.to_numeric(market_frame["acc_trade_price_24h"], errors="coerce")
    market_frame = market_frame.sort_values("acc_trade_price_24h", ascending=False)

    left, right = st.columns([1.05, 2.35], gap="large")
    with left:
        selected_market = _market_panel(market_frame)

    with right:
        st.subheader("Chart Desk")
        control_col1, control_col2, control_col3, control_col4 = st.columns([1, 1, 1.2, 1])
        interval = control_col1.selectbox("Interval", ["minute15", "minute30", "minute60", "minute240", "day"], index=2)
        count = int(control_col2.number_input("Candles", 120, 1200, 360, 20))
        fee = float(control_col3.number_input("Fee", 0.0, 0.01, 0.0005, 0.0001, format="%.4f"))
        auto_run = control_col4.checkbox("Auto refresh", value=True)
        strategy_name, strategy_params = _strategy_controls()
        run_now = st.button("Run Analysis", use_container_width=True)

        if not selected_market:
            st.info("Select a market from the left panel.")
            return

        previous_key = st.session_state.get("bt_context")
        current_key = {
            "market": selected_market,
            "interval": interval,
            "count": count,
            "fee": fee,
            "strategy_name": strategy_name,
            "strategy_params": strategy_params,
        }
        if run_now or (auto_run and previous_key != current_key):
            raw = _candles(selected_market, interval, count)
            if raw.empty:
                st.warning("No candle data returned for this market.")
                return
            try:
                frame = build_strategy_frame(
                    raw[["open", "high", "low", "close", "volume"]],
                    strategy_name=strategy_name,
                    params=strategy_params,
                    flux_indicator=flux_indicator,
                )
            except Exception as exc:
                st.error(f"Strategy build failed: {exc}")
                return
            st.session_state["bt_frame"] = frame
            st.session_state["bt_context"] = current_key

        frame = st.session_state.get("bt_frame")
        if frame is None or frame.empty:
            st.info("Run an analysis to render the chart.")
            return

        bt_result = backtest_signal_frame(frame, fee=fee)
        last_row = frame.iloc[-1]
        metric_cols = st.columns(5)
        metric_cols[0].metric("Strategy", strategy_label(strategy_name))
        metric_cols[1].metric("Trades", int(bt_result["trades"]))
        metric_cols[2].metric("Return", f"{float(bt_result['total_return_pct']):.2f}%")
        metric_cols[3].metric("Win Rate", f"{float(bt_result['win_rate_pct']):.1f}%")
        metric_cols[4].metric("Max DD", f"{float(bt_result['max_drawdown_pct']):.2f}%")
        detail_cols = st.columns(3)
        detail_cols[0].metric("Last Price", fmt_full_number(last_row.get("close"), 0))
        detail_cols[1].metric("Score", f"{float(last_row.get('strategy_score', 0.0)):.2f}")
        detail_cols[2].metric("Signal", "BUY" if bool(last_row.get("buy_signal")) else "SELL" if bool(last_row.get("sell_signal")) else "WAIT")

        st.plotly_chart(_render_chart(frame, strategy_name, bt_result), use_container_width=True)

        tail = frame.tail(20).copy()
        columns = ["close", "strategy_score", "buy_signal", "sell_signal"]
        if strategy_name == "research_trend":
            columns.extend(["ema_fast", "ema_slow", "adx", "atr"])
        st.dataframe(tail[[column for column in columns if column in tail.columns]], use_container_width=True)

from __future__ import annotations

import os
import threading
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from notifier import get_notifier
from paper_trader import PaperTrader
from risk_manager import ensure_daily_metrics, evaluate_entry, risk_config_from_dict, total_unrealized_pnl
from strategy import backtest_signal_frame
from strategy_engine import build_strategy_frame, strategy_label, strategy_options
from ui_theme import apply_chart_theme, page_intro
from upbit_api import UpbitAPI
from utils.formatters import fmt_full_number


api: UpbitAPI | None = None
flux_indicator = None

_MARKETS_CACHE = {"data": None, "ts": 0.0}
_CANDLES_CACHE: dict[tuple[str, str, int], dict[str, object]] = {}


def init_api(a: UpbitAPI, flux):
    global api, flux_indicator
    api = a
    flux_indicator = flux


def _markets_rank() -> pd.DataFrame:
    now = time.time()
    if _MARKETS_CACHE["data"] is not None and now - float(_MARKETS_CACHE["ts"]) < 120:
        return _MARKETS_CACHE["data"]  # type: ignore[return-value]
    if not api:
        return pd.DataFrame()
    try:
        markets = pd.DataFrame(api.markets())
        markets = markets[markets["market"].str.startswith("KRW-")]
        tickers = pd.DataFrame(api.tickers(markets["market"].tolist()))
        merged = markets.merge(
            tickers[["market", "trade_price", "acc_trade_price_24h", "acc_trade_volume_24h"]],
            on="market",
            how="left",
        )
        merged["acc_trade_price_24h"] = pd.to_numeric(merged["acc_trade_price_24h"], errors="coerce")
        merged = merged.sort_values("acc_trade_price_24h", ascending=False)
        _MARKETS_CACHE.update({"data": merged, "ts": now})
        return merged
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=45)
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


def _compute_account_equity() -> float | None:
    if not api:
        return None
    try:
        accounts = api.accounts()
    except Exception:
        return None
    total = 0.0
    markets: list[str] = []
    balances: list[tuple[str, float]] = []
    for account in accounts:
        amount = float(account.get("balance") or 0.0) + float(account.get("locked") or 0.0)
        currency = account.get("currency")
        if currency == "KRW":
            total += amount
        elif currency:
            balances.append((currency, amount))
            markets.append(f"KRW-{currency}")
    if markets:
        try:
            price_map = {
                item["market"]: float(item.get("trade_price") or 0.0)
                for item in api.tickers(markets)
                if item.get("market")
            }
            for currency, amount in balances:
                total += amount * price_map.get(f"KRW-{currency}", 0.0)
        except Exception:
            pass
    return total


def _resolve_day_start_equity(metrics: dict, trader: PaperTrader) -> float:
    if metrics.get("day_start_equity"):
        return float(metrics["day_start_equity"])
    actual = _compute_account_equity()
    if actual and actual > 0:
        return actual
    estimate = sum(position.cost for position in trader.positions.values()) + float(metrics.get("daily_buy") or 0.0)
    return max(estimate, 1.0)


def _current_signal(last_row: pd.Series) -> str:
    if bool(last_row.get("buy_signal")):
        return "BUY"
    if bool(last_row.get("sell_signal")):
        return "SELL"
    return "WAIT"


def _ensure_lock():
    if "LIVE_LOCK" not in st.session_state:
        st.session_state["LIVE_LOCK"] = threading.Lock()


def _scan_core(params: dict, last_state: dict) -> dict:
    ranked = _markets_rank()
    if ranked.empty:
        return {"table": pd.DataFrame(), "detail": {}, "last_sig": last_state, "notify": [], "trades": []}

    trader = PaperTrader(params.get("_positions"))
    metrics = ensure_daily_metrics(params.get("_metrics"), day=time.strftime("%Y-%m-%d"))
    metrics["day_start_equity"] = _resolve_day_start_equity(metrics, trader)
    risk_config = risk_config_from_dict(params.get("risk_limits"))
    strategy_name = str(params.get("strategy_name", "research_trend"))
    live_orders = bool(params.get("live_trading")) and os.getenv("UPBIT_LIVE") == "1"

    result_rows: list[dict[str, object]] = []
    detail_cache: dict[str, dict[str, object]] = {}
    notify: list[str] = []
    trade_events: list[dict[str, object]] = []
    price_map: dict[str, float] = {}

    for _, row in ranked.head(int(params["topn"])).iterrows():
        market = row["market"]
        raw = _candles(market, str(params["interval"]), int(params["count"]))
        if raw.empty:
            result_rows.append({"market": market, "last_signal": "WAIT", "error": "no_candles"})
            continue

        try:
            frame = build_strategy_frame(
                raw[["open", "high", "low", "close", "volume"]],
                strategy_name=strategy_name,
                params=params,
                flux_indicator=flux_indicator,
            )
        except Exception as exc:
            result_rows.append({"market": market, "last_signal": "WAIT", "error": repr(exc)})
            continue

        bt = backtest_signal_frame(frame, fee=float(params["fee"]))
        last_row = frame.iloc[-1]
        signal = _current_signal(last_row)
        previous_signal = (last_state.get(market) or {}).get("sig", "WAIT")
        score = float(last_row.get("strategy_score", 0.0))
        close_price = float(last_row["close"])
        position = trader.get_position(market)
        price_map[market] = close_price

        if signal == "BUY" and previous_signal != "BUY":
            decision = evaluate_entry(
                config=risk_config,
                metrics=metrics,
                positions=trader.to_state(),
                price_map=price_map,
                market=market,
                day_start_equity=float(metrics["day_start_equity"]),
            )
            if decision.allowed:
                order_ok = True
                order_suffix = ""
                if live_orders:
                    order_result = api.create_order(market, side="bid", ord_type="price", price=f"{int(decision.trade_cost)}", simulate=False)
                    if order_result.get("error"):
                        order_ok = False
                        order_suffix = f" [LIVE_ERROR:{order_result.get('status_code', 'unknown')}]"
                if order_ok:
                    trade_event = trader.enter_long(
                        market=market,
                        price=close_price,
                        cost=decision.trade_cost,
                        strategy=strategy_name,
                    )
                    trade_events.append(trade_event)
                    metrics["daily_buy"] = float(metrics.get("daily_buy") or 0.0) + decision.trade_cost
                    notify.append(
                        f"[{'LIVE' if live_orders else 'SIM'}] {market} BUY price={close_price:.4f} alloc={decision.trade_cost:.0f} score={score:.2f}{order_suffix}"
                    )
                else:
                    signal = "WAIT"
            else:
                notify.append(f"[SIM] {market} BUY blocked: {decision.blocked_reason} price={close_price:.4f}")
                signal = "WAIT"

        elif signal == "SELL" and previous_signal == "BUY" and position is not None:
            order_ok = True
            order_suffix = ""
            if live_orders:
                order_result = api.create_order(
                    market,
                    side="ask",
                    ord_type="market",
                    volume=f"{position.qty:.8f}",
                    simulate=False,
                )
                if order_result.get("error"):
                    order_ok = False
                    order_suffix = f" [LIVE_ERROR:{order_result.get('status_code', 'unknown')}]"
            if order_ok:
                trade_event = trader.exit_long(market=market, price=close_price, reason="signal")
                if trade_event:
                    trade_events.append(trade_event)
                    metrics["realized_pnl"] = float(metrics.get("realized_pnl") or 0.0) + float(trade_event["pnl_value"])
                    notify.append(
                        f"[{'LIVE' if live_orders else 'SIM'}] {market} SELL price={close_price:.4f} pnl={float(trade_event['pnl_pct']):+.2f}%{order_suffix}"
                    )
            else:
                signal = "BUY"

        updated_position = trader.get_position(market)
        last_state[market] = {
            "sig": "BUY" if updated_position else signal,
            "entry": updated_position.entry if updated_position else None,
        }
        detail_cache[market] = {"df": frame, "bt": bt}
        result_rows.append(
            {
                "market": market,
                "price": close_price,
                "score": score,
                "trades": int(bt["trades"]),
                "return_pct": float(bt["total_return_pct"]),
                "win_rate_pct": float(bt["win_rate_pct"]),
                "max_drawdown_pct": float(bt["max_drawdown_pct"]),
                "last_signal": "BUY" if updated_position else signal,
                "position": "OPEN" if updated_position else "-",
            }
        )

    positions_state = trader.to_state()
    metrics["unrealized_pnl"] = total_unrealized_pnl(positions_state, price_map)
    metrics["total_pnl"] = float(metrics.get("realized_pnl") or 0.0) + float(metrics["unrealized_pnl"])
    table = pd.DataFrame(result_rows)
    if not table.empty:
        table = table.sort_values(["score", "return_pct"], ascending=False)
    return {
        "table": table,
        "detail": detail_cache,
        "last_sig": last_state,
        "notify": notify,
        "last_run": time.time(),
        "_metrics": metrics,
        "_positions": positions_state,
        "trades": trade_events,
    }


def _scan(params: dict):
    _ensure_lock()
    with st.session_state["LIVE_LOCK"]:
        snapshot = _scan_core(params, st.session_state.get("LIVE_LAST_SIG", {}))
        st.session_state["LIVE_LAST_SIG"] = snapshot["last_sig"]
        st.session_state["LIVE_RESULTS"] = {"table": snapshot["table"], "detail": snapshot["detail"]}
        st.session_state["LIVE_LAST_RUN"] = snapshot.get("last_run")
        st.session_state["LIVE_METRICS"] = snapshot.get("_metrics")
        st.session_state["LIVE_POSITIONS"] = snapshot.get("_positions")
        if snapshot.get("trades"):
            st.session_state.setdefault("LIVE_TRADES", [])
            st.session_state["LIVE_TRADES"].extend(snapshot["trades"])
            st.session_state["LIVE_TRADES"] = st.session_state["LIVE_TRADES"][-500:]
        notifier = get_notifier()
        if notifier.available():
            for message in snapshot["notify"][:10]:
                try:
                    notifier.send_text(message)
                except Exception:
                    pass


class _Worker:
    def __init__(self):
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.params: dict = {}
        self.last_signal_state: dict = {}
        self.snapshot = None
        self.metrics: dict = {}
        self.positions: dict = {}
        self.last_error: str | None = None
        self.interval = 30

    def update_params(self, params: dict):
        with self.lock:
            self.params = dict(params or {})

    def _get_params(self) -> dict:
        with self.lock:
            return dict(self.params)

    def start(self, interval: int, params: dict):
        self.stop()
        self.interval = interval
        self.update_params(params)
        self.stop_event.clear()

        def loop():
            while not self.stop_event.is_set():
                try:
                    params = self._get_params()
                    params["_metrics"] = self.metrics
                    params["_positions"] = self.positions
                    snapshot = _scan_core(params, self.last_signal_state)
                    with self.lock:
                        self.metrics = snapshot.get("_metrics") or self.metrics
                        self.positions = snapshot.get("_positions") or self.positions
                        self.last_signal_state = snapshot.get("last_sig") or self.last_signal_state
                        self.snapshot = snapshot
                except Exception as exc:
                    self.last_error = repr(exc)
                self.stop_event.wait(self.interval)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=0.5)

    def get_snapshot(self):
        with self.lock:
            return self.snapshot


def _strategy_controls(prefix: str) -> tuple[str, dict[str, float | int | str]]:
    options = strategy_options(flux_indicator is not None)
    current = st.session_state.get(f"{prefix}_strategy_name", options[0])
    index = options.index(current) if current in options else 0
    strategy_name = st.selectbox("Strategy", options, index=index, format_func=strategy_label, key=f"{prefix}_strategy_name")
    params: dict[str, float | int | str] = {}
    if strategy_name == "research_trend":
        with st.expander("Research Trend Parameters", expanded=False):
            row1 = st.columns(4)
            params["fast_ema"] = row1[0].number_input("Fast EMA", 5, 100, 21, 1, key=f"{prefix}_fast_ema")
            params["slow_ema"] = row1[1].number_input("Slow EMA", 10, 240, 55, 1, key=f"{prefix}_slow_ema")
            params["breakout_window"] = row1[2].number_input("Breakout Window", 5, 120, 20, 1, key=f"{prefix}_breakout")
            params["exit_window"] = row1[3].number_input("Exit Window", 3, 80, 10, 1, key=f"{prefix}_exit")
            row2 = st.columns(4)
            params["atr_window"] = row2[0].number_input("ATR Window", 5, 50, 14, 1, key=f"{prefix}_atr_window")
            params["atr_mult"] = row2[1].number_input("ATR Mult", 1.0, 6.0, 2.5, 0.1, key=f"{prefix}_atr_mult")
            params["adx_window"] = row2[2].number_input("ADX Window", 5, 50, 14, 1, key=f"{prefix}_adx_window")
            params["adx_threshold"] = row2[3].number_input("ADX Threshold", 5.0, 40.0, 18.0, 0.5, key=f"{prefix}_adx_threshold")
            row3 = st.columns(3)
            params["momentum_window"] = row3[0].number_input("Momentum Window", 5, 80, 20, 1, key=f"{prefix}_momentum")
            params["volume_window"] = row3[1].number_input("Volume Window", 5, 80, 20, 1, key=f"{prefix}_volume_window")
            params["volume_threshold"] = row3[2].number_input("Volume Ratio", 0.1, 3.0, 0.9, 0.1, key=f"{prefix}_volume_threshold")
    else:
        with st.expander("Flux Parameters", expanded=False):
            row = st.columns(5)
            params["ltf_len"] = row[0].number_input("LTF Len", 5, 400, 20, 1, key=f"{prefix}_ltf_len")
            params["ltf_mult"] = row[1].number_input("LTF Mult", 0.1, 10.0, 2.0, 0.1, key=f"{prefix}_ltf_mult")
            params["htf_len"] = row[2].number_input("HTF Len", 5, 400, 20, 1, key=f"{prefix}_htf_len")
            params["htf_mult"] = row[3].number_input("HTF Mult", 0.1, 10.0, 2.25, 0.1, key=f"{prefix}_htf_mult")
            htf = row[4].selectbox("HTF Rule", ["30m", "60m", "120m", "240m", "1D"], index=1, key=f"{prefix}_htf_rule")
            params["htf_rule"] = htf.replace("m", "T") if htf.endswith("m") else "1D"
    return strategy_name, params


def _render_chart(frame: pd.DataFrame, strategy_name: str, bt: dict[str, object]):
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.64, 0.18, 0.18], vertical_spacing=0.03)
    figure.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            increasing_line_color="#22c55e",
            decreasing_line_color="#f43f5e",
            name="Price",
        ),
        row=1,
        col=1,
    )
    if strategy_name == "research_trend":
        for column, color in [("ema_fast", "#60a5fa"), ("ema_slow", "#f59e0b"), ("atr_stop", "#f97316")]:
            if column in frame:
                figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column, line={"color": color, "width": 1.5}), row=1, col=1)
        figure.add_trace(go.Scatter(x=frame.index, y=frame["adx"], name="ADX", line={"color": "#a78bfa"}), row=2, col=1)
        figure.add_hline(y=18, line={"color": "rgba(255,255,255,0.16)", "dash": "dot"}, row=2, col=1)
        figure.add_trace(go.Scatter(x=frame.index, y=frame["strategy_score"], name="Score", line={"color": "#2dd4bf"}), row=3, col=1)
    else:
        for column in ["ltf_upper", "ltf_lower", "ltf_basis", "htf_upper", "htf_lower"]:
            if column in frame:
                figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column, line={"width": 1.2}), row=1, col=1)
        figure.add_trace(go.Bar(x=frame.index, y=frame["volume"], name="Volume", marker_color="rgba(96,165,250,0.5)"), row=3, col=1)
    for column, color in [("buy_signal", "#22c55e"), ("sell_signal", "#f43f5e")]:
        if column in frame:
            hits = frame[frame[column]]
            if not hits.empty:
                figure.add_trace(go.Scatter(x=hits.index, y=hits["close"], mode="markers", name=column, marker={"size": 11, "color": color, "line": {"color": "white", "width": 1}}), row=1, col=1)
    equity = bt.get("equity")
    if equity is not None:
        figure.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line={"color": "#f8fafc"}), row=2, col=1)
    return apply_chart_theme(figure, height=840)


def render_live():
    page_intro(
        "Live Desk",
        "Refactored live simulation with guarded execution",
        "The live desk now uses shared strategy, risk, and paper-trading modules. Real orders still stay behind both the environment gate and an explicit UI confirmation.",
    )
    if not api:
        st.error("API is not initialized.")
        return

    controls = st.columns([1, 1, 1, 1, 1])
    interval = controls[0].selectbox("Interval", ["minute15", "minute30", "minute60", "minute240", "day"], index=1)
    count = int(controls[1].number_input("Candles", 120, 1200, 360, 20))
    topn = int(controls[2].number_input("Top Markets", 5, 60, 20, 1))
    fee = float(controls[3].number_input("Fee", 0.0, 0.01, 0.0005, 0.0001, format="%.4f"))
    auto = controls[4].checkbox("Auto sync", value=True)
    strategy_name, strategy_params = _strategy_controls("live")

    with st.expander("Live Trading Guard", expanded=False):
        guard_cols = st.columns([1, 1, 1.2])
        live_toggle = guard_cols[0].checkbox("Enable live orders", value=st.session_state.get("LIVE_TRADING_RAW", False))
        confirm_text = guard_cols[1].text_input("Type LIVE", value=st.session_state.get("LIVE_TRADING_CONFIRM", ""))
        env_live = os.getenv("UPBIT_LIVE") == "1"
        live_trading = bool(env_live and live_toggle and confirm_text.strip().upper() == "LIVE")
        st.session_state["LIVE_TRADING_RAW"] = live_toggle
        st.session_state["LIVE_TRADING_CONFIRM"] = confirm_text
        st.session_state["LIVE_TRADING"] = live_trading
        if not env_live:
            st.info("Environment gate is closed. Set UPBIT_LIVE=1 before any real order can be sent.")
        elif live_toggle and confirm_text.strip().upper() != "LIVE":
            st.warning("The confirmation word must be LIVE before execution is armed.")
        elif live_trading:
            st.success("Live order execution is armed.")

    with st.expander("Risk Limits", expanded=False):
        row1 = st.columns(3)
        max_trade_krw = float(row1[0].number_input("Max trade KRW", 0, 10_000_000_000, 50000, 10000, format="%d"))
        max_trade_pct = float(row1[1].number_input("Max trade %", 0.0, 100.0, 2.0, 0.5, format="%.1f"))
        per_asset_max_pct = float(row1[2].number_input("Max asset %", 0.0, 100.0, 10.0, 0.5, format="%.1f"))
        row2 = st.columns(3)
        daily_buy_limit = float(row2[0].number_input("Daily buy KRW", 0, 10_000_000_000, 200000, 50000, format="%d"))
        daily_loss_limit_krw = float(row2[1].number_input("Daily loss KRW", 0, 10_000_000_000, 30000, 10000, format="%d"))
        daily_loss_limit_pct = float(row2[2].number_input("Daily loss %", 0.0, 100.0, 3.0, 0.5, format="%.1f"))
        include_unrealized = st.checkbox("Include unrealized loss in daily stop", value=True)
        risk_limits = {
            "max_trade_krw": max_trade_krw,
            "max_trade_pct": max_trade_pct,
            "per_asset_max_pct": per_asset_max_pct,
            "daily_buy_limit": daily_buy_limit,
            "daily_loss_limit_krw": daily_loss_limit_krw,
            "daily_loss_limit_pct": daily_loss_limit_pct,
            "include_unrealized_loss": include_unrealized,
        }

    params = {
        "interval": interval,
        "count": count,
        "topn": topn,
        "fee": fee,
        "strategy_name": strategy_name,
        "risk_limits": risk_limits,
        "live_trading": live_trading,
        **strategy_params,
    }
    changed = params != (st.session_state.get("LIVE_PARAMS") or {})
    st.session_state["LIVE_PARAMS"] = params

    worker_cols = st.columns([1, 1, 1, 1.4])
    worker_interval = int(worker_cols[0].number_input("Worker Seconds", 5, 3600, 30, 1))
    start = worker_cols[1].button("Start Worker", use_container_width=True)
    stop = worker_cols[2].button("Stop Worker", use_container_width=True)
    worker_cols[3].markdown(f"**Mode:** {'LIVE' if live_trading else 'SIM'} / **Strategy:** `{strategy_label(strategy_name)}`")

    if "LIVE_WORKER" not in st.session_state:
        st.session_state["LIVE_WORKER"] = _Worker()
    worker: _Worker = st.session_state["LIVE_WORKER"]
    if start:
        st.session_state["LIVE_WORKING"] = True
        worker.start(worker_interval, params)
    if stop:
        st.session_state["LIVE_WORKING"] = False
        worker.stop()
    if changed and auto and st.session_state.get("LIVE_WORKING"):
        worker.update_params(params)
    if (not st.session_state.get("LIVE_WORKING")) and (changed or "LIVE_RESULTS" not in st.session_state):
        _scan(params)
    if st.session_state.get("LIVE_WORKING"):
        snapshot = worker.get_snapshot()
        if snapshot:
            st.session_state["LIVE_RESULTS"] = {"table": snapshot["table"], "detail": snapshot["detail"]}
            st.session_state["LIVE_LAST_RUN"] = snapshot.get("last_run")
            st.session_state["LIVE_METRICS"] = snapshot.get("_metrics")
            st.session_state["LIVE_POSITIONS"] = snapshot.get("_positions")
            if snapshot.get("trades"):
                st.session_state.setdefault("LIVE_TRADES", [])
                st.session_state["LIVE_TRADES"].extend(snapshot.get("trades"))
                st.session_state["LIVE_TRADES"] = st.session_state["LIVE_TRADES"][-500:]

    metrics = st.session_state.get("LIVE_METRICS") or {}
    metric_cols = st.columns(6)
    metric_cols[0].metric("Day", metrics.get("day_date", "-"))
    metric_cols[1].metric("Daily Buy", fmt_full_number(metrics.get("daily_buy"), 0))
    metric_cols[2].metric("Realized", fmt_full_number(metrics.get("realized_pnl"), 0))
    metric_cols[3].metric("Unrealized", fmt_full_number(metrics.get("unrealized_pnl"), 0))
    metric_cols[4].metric("Total PnL", fmt_full_number(metrics.get("total_pnl"), 0))
    metric_cols[5].metric("Worker", "ON" if st.session_state.get("LIVE_WORKING") else "OFF")

    result = st.session_state.get("LIVE_RESULTS") or {}
    table = result.get("table")
    if table is not None and isinstance(table, pd.DataFrame) and not table.empty:
        st.dataframe(table[["market", "price", "score", "trades", "return_pct", "win_rate_pct", "max_drawdown_pct", "last_signal", "position"]], use_container_width=True, hide_index=True)
        selected_market = st.selectbox("Detail Market", table["market"].tolist(), key="live_detail_market")
        detail = result.get("detail") or {}
        if selected_market in detail:
            st.plotly_chart(_render_chart(detail[selected_market]["df"], strategy_name, detail[selected_market]["bt"]), use_container_width=True)
    else:
        st.info("Run the live desk once to build a leaderboard.")

    positions = st.session_state.get("LIVE_POSITIONS") or {}
    if positions:
        position_rows = []
        market_price_map = {row["market"]: row["price"] for row in table.to_dict("records")} if table is not None and not table.empty else {}
        for market, raw in positions.items():
            current_price = market_price_map.get(market)
            entry = float(raw.get("entry") or 0.0)
            qty = float(raw.get("qty") or 0.0)
            pnl_value = ((float(current_price) - entry) * qty) if current_price is not None else None
            pnl_pct = (((float(current_price) / entry) - 1.0) * 100) if current_price is not None and entry else None
            position_rows.append({"market": market, "qty": qty, "entry": entry, "current_price": current_price, "pnl_value": pnl_value, "pnl_pct": pnl_pct})
        st.subheader("Open Positions")
        st.dataframe(pd.DataFrame(position_rows), use_container_width=True, hide_index=True)

    st.session_state.setdefault("LIVE_TRADES", [])
    with st.expander("Trade Log", expanded=False):
        if st.button("Clear Log"):
            st.session_state["LIVE_TRADES"] = []
        logs = st.session_state.get("LIVE_TRADES") or []
        if logs:
            log_frame = pd.DataFrame(logs).sort_values("ts").tail(200)
            log_frame["time"] = pd.to_datetime(log_frame["ts"], unit="s").dt.strftime("%H:%M:%S")
            preferred = ["time", "market", "side", "price", "qty", "cost", "entry", "pnl_value", "pnl_pct", "reason", "strategy"]
            st.dataframe(log_frame[[column for column in preferred if column in log_frame.columns]], use_container_width=True, hide_index=True)
        else:
            st.caption("No trades yet.")

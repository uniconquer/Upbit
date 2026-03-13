from __future__ import annotations

import os
import threading
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from daily_summary import current_kst_day, rollover_daily_report
from exchange_state import sync_exchange_state
from execution import build_pending_order, pending_fill_delta, resolve_submitted_order
from kill_switch import effective_kill_switch, load_kill_switch, save_kill_switch
from notifier import get_notifier
from notification_text import (
    blocked_risk_message,
    buy_filled_message,
    kill_switch_block_message,
    kill_switch_disabled_message,
    kill_switch_enabled_message,
    lookup_failed_message,
    order_cancelled_message,
    order_failed_message,
    order_no_fill_message,
    order_pending_message,
    sell_filled_message,
)
from paper_trader import PaperTrader
from risk_manager import ensure_daily_metrics, evaluate_entry, risk_config_from_dict, total_unrealized_pnl
from runtime_store import load_runtime_state, save_runtime_state
from strategy import backtest_signal_frame
from strategy_engine import build_strategy_frame, strategy_description, strategy_label, strategy_options
from startup_automation import (
    format_startup_status_bundle,
    install_startup_task,
    load_startup_status_bundle,
    remove_startup_task,
    startup_supported,
)
from trading_costs import TradingCostModel, cost_model_from_values
from ui_theme import apply_chart_theme, page_intro
from upbit_api import UpbitAPI
from utils.formatters import fmt_full_number
from worker_control import (
    coerce_worker_config,
    format_worker_status,
    load_managed_worker_status,
    read_worker_log_tail,
    start_managed_worker,
    stop_managed_worker,
)


api: UpbitAPI | None = None
flux_indicator = None

_MARKETS_CACHE = {"data": None, "ts": 0.0}
_CANDLES_CACHE: dict[tuple[str, str, int], dict[str, object]] = {}
LIVE_RUNTIME_STATE = "live-desk"
LIVE_KILL_SWITCH_NAME = "trade-kill-switch"
LIVE_PARAM_WIDGETS = {
    "interval": "live_interval",
    "count": "live_count",
    "topn": "live_topn",
    "fee": "live_fee",
    "slippage_bps": "live_slippage_bps",
    "strategy_name": "live_strategy_name",
    "fast_ema": "live_fast_ema",
    "slow_ema": "live_slow_ema",
    "breakout_window": "live_breakout",
    "exit_window": "live_exit",
    "atr_window": "live_atr_window",
    "atr_mult": "live_atr_mult",
    "adx_window": "live_adx_window",
    "adx_threshold": "live_adx_threshold",
    "momentum_window": "live_momentum",
    "volume_window": "live_volume_window",
    "volume_threshold": "live_volume_threshold",
    "ltf_len": "live_ltf_len",
    "ltf_mult": "live_ltf_mult",
    "htf_len": "live_htf_len",
    "htf_mult": "live_htf_mult",
    "htf_rule": "live_htf_rule",
    "worker_interval": "live_worker_interval",
}
LIVE_RISK_WIDGETS = {
    "max_trade_krw": "live_max_trade_krw",
    "max_trade_pct": "live_max_trade_pct",
    "per_asset_max_pct": "live_per_asset_max_pct",
    "daily_buy_limit": "live_daily_buy_limit",
    "daily_loss_limit_krw": "live_daily_loss_limit_krw",
    "daily_loss_limit_pct": "live_daily_loss_limit_pct",
    "include_unrealized_loss": "live_include_unrealized_loss",
}

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


def _signal_text(signal: str | None) -> str:
    mapping = {
        "BUY": "매수",
        "SELL": "매도",
        "WAIT": "대기",
        "BUY_PENDING": "매수 대기",
        "SELL_PENDING": "매도 대기",
    }
    return mapping.get(str(signal or "").upper(), str(signal or "-"))


def _position_text(value: str | None) -> str:
    mapping = {
        "OPEN": "보유중",
        "PENDING": "주문 대기",
        "-": "-",
    }
    return mapping.get(str(value or "-"), str(value or "-"))


def _interval_text(value: str) -> str:
    return _INTERVAL_LABELS.get(value, value)


def _mode_text(value: str | None) -> str:
    mapping = {
        "LIVE": "실거래",
        "SIM": "모의",
        "ON": "켜짐",
        "OFF": "꺼짐",
    }
    return mapping.get(str(value or "").upper(), str(value or "-"))


def _reason_text(value: str | None) -> str:
    mapping = {
        "signal": "신호",
        "reconcile": "주문 동기화",
        "manual": "수동",
        "risk": "리스크 차단",
        "kill_switch": "긴급중지",
        "timeout": "시간 초과",
    }
    return mapping.get(str(value or ""), str(value or "-"))


def _format_pct(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "-"


def _format_money(value: object, digits: int = 0) -> str:
    try:
        return fmt_full_number(float(value), digits)
    except Exception:
        return "-"


def _trade_events_for_market(trade_log: list[dict[str, object]], market: str) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for trade in trade_log:
        if str(trade.get("market") or "") != market:
            continue
        try:
            ts = float(trade.get("ts") or 0.0)
            price = float(trade.get("price") or 0.0)
        except Exception:
            continue
        if ts <= 0 or price <= 0:
            continue
        points.append(
            {
                "ts": pd.to_datetime(ts, unit="s"),
                "price": price,
                "side": str(trade.get("side") or "").upper(),
                "qty": float(trade.get("qty") or 0.0),
            }
        )
    return points


def _present_leaderboard(table: pd.DataFrame) -> pd.DataFrame:
    visible = table.copy()
    if "price" in visible.columns:
        visible["price"] = visible["price"].apply(_format_money)
    if "score" in visible.columns:
        visible["score"] = visible["score"].apply(lambda value: f"{float(value):.2f}" if pd.notna(value) else "-")
    if "trades" in visible.columns:
        visible["trades"] = visible["trades"].apply(lambda value: int(value) if pd.notna(value) else 0)
    for column in ["return_pct", "win_rate_pct", "max_drawdown_pct"]:
        if column in visible.columns:
            visible[column] = visible[column].apply(_format_pct)
    if "last_signal" in visible.columns:
        visible["last_signal"] = visible["last_signal"].apply(_signal_text)
    if "position" in visible.columns:
        visible["position"] = visible["position"].apply(_position_text)
    ordered = [column for column in ["market", "price", "score", "trades", "return_pct", "win_rate_pct", "max_drawdown_pct", "last_signal", "position"] if column in visible.columns]
    return visible[ordered].rename(
        columns={
            "market": "종목",
            "price": "현재가",
            "score": "전략 점수",
            "trades": "백테스트 거래 수",
            "return_pct": "백테스트 수익률",
            "win_rate_pct": "승률",
            "max_drawdown_pct": "최대 낙폭",
            "last_signal": "마지막 신호",
            "position": "포지션 상태",
        }
    )


def _present_positions(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ["entry", "cost", "current_price", "pnl_value"]:
        if column in frame.columns:
            frame[column] = frame[column].apply(_format_money)
    if "qty" in frame.columns:
        frame["qty"] = frame["qty"].apply(lambda value: f"{float(value):.8f}" if pd.notna(value) else "-")
    if "pnl_pct" in frame.columns:
        frame["pnl_pct"] = frame["pnl_pct"].apply(_format_pct)
    return frame.rename(
        columns={
            "market": "종목",
            "qty": "수량",
            "entry": "평균 진입가",
            "cost": "매수 금액",
            "current_price": "현재가",
            "pnl_value": "평가손익",
            "pnl_pct": "평가수익률",
        }
    )


def _present_trade_log(logs: list[dict[str, object]]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame()
    frame = pd.DataFrame(logs).sort_values("ts").tail(200)
    frame["time"] = pd.to_datetime(frame["ts"], unit="s").dt.strftime("%H:%M:%S")
    if "side" in frame.columns:
        frame["side"] = frame["side"].apply(_signal_text)
    if "strategy" in frame.columns:
        frame["strategy"] = frame["strategy"].apply(strategy_label)
    if "reason" in frame.columns:
        frame["reason"] = frame["reason"].apply(_reason_text)
    for column in ["price", "cost", "entry", "pnl_value"]:
        if column in frame.columns:
            frame[column] = frame[column].apply(_format_money)
    if "qty" in frame.columns:
        frame["qty"] = frame["qty"].apply(lambda value: f"{float(value):.8f}" if pd.notna(value) else "-")
    if "pnl_pct" in frame.columns:
        frame["pnl_pct"] = frame["pnl_pct"].apply(_format_pct)
    preferred = [column for column in ["time", "market", "side", "price", "qty", "cost", "entry", "pnl_value", "pnl_pct", "reason", "strategy"] if column in frame.columns]
    return frame[preferred].rename(
        columns={
            "time": "시간",
            "market": "종목",
            "side": "구분",
            "price": "체결가",
            "qty": "수량",
            "cost": "체결 금액",
            "entry": "평균 진입가",
            "pnl_value": "손익",
            "pnl_pct": "손익률",
            "reason": "사유",
            "strategy": "전략",
        }
    )


def _present_startup_status(bundle: dict[str, object]) -> pd.DataFrame:
    tasks = dict(bundle.get("tasks") or {})
    rows = []
    for component in ["worker", "telegram"]:
        snapshot = dict(tasks.get(component) or {})
        if not snapshot:
            continue
        rows.append(
            {
                "대상": snapshot.get("label") or component,
                "설치": "예" if snapshot.get("exists") else "아니오",
                "활성": "ON" if snapshot.get("enabled") else "OFF",
                "상태": snapshot.get("state") or "-",
                "구성 확인": "정상" if snapshot.get("configured") else "확인 필요",
                "다음 실행": snapshot.get("next_run_time") or "-",
                "마지막 실행": snapshot.get("last_run_time") or "-",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for column in ["다음 실행", "마지막 실행"]:
        frame[column] = frame[column].apply(
            lambda value: (
                pd.to_datetime(float(value), unit="s").tz_localize("UTC").tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(value, (int, float)) and float(value) > 0
                else "-"
            )
        )
    return frame


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


def _restore_live_widget_state(params: dict) -> None:
    for field, widget_key in LIVE_PARAM_WIDGETS.items():
        if field in params:
            st.session_state.setdefault(widget_key, params[field])
    for field, widget_key in LIVE_RISK_WIDGETS.items():
        if field in (params.get("risk_limits") or {}):
            st.session_state.setdefault(widget_key, params["risk_limits"][field])


def _hydrate_live_runtime() -> None:
    if st.session_state.get("LIVE_RUNTIME_HYDRATED"):
        return
    snapshot = load_runtime_state(LIVE_RUNTIME_STATE, default={})
    if isinstance(snapshot, dict) and snapshot:
        params = dict(snapshot.get("params") or {})
        _restore_live_widget_state(params)
        st.session_state.setdefault("LIVE_PARAMS", params)
        st.session_state.setdefault("LIVE_LAST_SIG", dict(snapshot.get("last_signal_state") or {}))
        st.session_state.setdefault("LIVE_METRICS", dict(snapshot.get("metrics") or {}))
        st.session_state.setdefault("LIVE_POSITIONS", dict(snapshot.get("positions") or {}))
        st.session_state.setdefault("LIVE_PENDING_ORDERS", dict(snapshot.get("pending_orders") or {}))
        st.session_state.setdefault("LIVE_TRADES", list(snapshot.get("trade_log") or []))
        st.session_state.setdefault("LIVE_DAILY_REPORTS", dict(snapshot.get("daily_reports") or {}))
        saved_last_report_day = str(snapshot.get("last_daily_report_day") or "").strip()
        st.session_state.setdefault("LIVE_LAST_DAILY_REPORT_DAY", saved_last_report_day or None)
        st.session_state.setdefault("LIVE_LAST_RUN", snapshot.get("last_run"))
    st.session_state["LIVE_RUNTIME_HYDRATED"] = True


def _persist_live_runtime(
    params: dict,
    *,
    metrics: dict,
    positions: dict,
    last_signal_state: dict,
    pending_orders: dict,
    trade_log: list[dict],
    daily_reports: dict,
    last_daily_report_day: str | None,
    last_run: float | None,
) -> None:
    public_params = {key: value for key, value in params.items() if not str(key).startswith("_")}
    save_runtime_state(
        LIVE_RUNTIME_STATE,
        {
            "version": 1,
            "saved_at": time.time(),
            "params": public_params,
            "metrics": metrics,
            "positions": positions,
            "last_signal_state": last_signal_state,
            "pending_orders": pending_orders,
            "trade_log": trade_log[-500:],
            "daily_reports": daily_reports,
            "last_daily_report_day": last_daily_report_day,
            "last_run": last_run,
        },
    )


def _sync_live_exchange_state(params: dict, last_state: dict) -> dict:
    if not api:
        return {
            "positions": dict(params.get("_positions") or {}),
            "pending_orders": dict(params.get("_pending_orders") or {}),
            "last_signal_state": dict(last_state or {}),
            "notifications": [],
            "synced": False,
        }
    return sync_exchange_state(
        api,
        strategy_name=str(params.get("strategy_name") or "research_trend"),
        existing_positions=params.get("_positions"),
        existing_pending_orders=params.get("_pending_orders"),
        existing_signal_state=last_state,
    )


def _cost_model_from_params(params: dict) -> TradingCostModel:
    return cost_model_from_values(
        fee_rate=float(params.get("fee") or 0.0),
        slippage_bps=float(params.get("slippage_bps") or 0.0),
    )


def _managed_worker_config_from_live_params(params: dict) -> dict:
    risk_limits = dict(params.get("risk_limits") or {})
    base = {
        "strategy": str(params.get("strategy_name") or "research_trend"),
        "interval": str(params.get("interval") or "minute30"),
        "count": int(params.get("count") or 240),
        "markets": int(params.get("topn") or 20),
        "loop_seconds": int(params.get("worker_interval") or 30),
        "max_open": 5,
        "min_fetch_seconds": 20.0,
        "per_request_sleep": 0.12,
        "fee": float(params.get("fee") or 0.0),
        "slippage_bps": float(params.get("slippage_bps") or 0.0),
        "live_orders": bool(params.get("live_trading")),
        "kill_switch_name": str(params.get("kill_switch_name") or LIVE_KILL_SWITCH_NAME),
        "max_trade_krw": float(risk_limits.get("max_trade_krw") or 0.0),
        "max_trade_pct": float(risk_limits.get("max_trade_pct") or 0.0),
        "per_asset_max_pct": float(risk_limits.get("per_asset_max_pct") or 0.0),
        "daily_buy_limit": float(risk_limits.get("daily_buy_limit") or 0.0),
        "daily_loss_limit_krw": float(risk_limits.get("daily_loss_limit_krw") or 0.0),
        "daily_loss_limit_pct": float(risk_limits.get("daily_loss_limit_pct") or 0.0),
        "include_unrealized_loss": bool(risk_limits.get("include_unrealized_loss")),
    }
    for field in [
        "fast_ema",
        "slow_ema",
        "breakout_window",
        "exit_window",
        "atr_window",
        "atr_mult",
        "adx_window",
        "adx_threshold",
        "momentum_window",
        "volume_window",
        "volume_threshold",
        "ltf_len",
        "ltf_mult",
        "htf_len",
        "htf_mult",
        "htf_rule",
    ]:
        if field in params:
            base[field] = params[field]
    return coerce_worker_config(base)


def _record_entry_trade(
    trader: PaperTrader,
    metrics: dict,
    trade_events: list[dict[str, object]],
    *,
    market: str,
    strategy_name: str,
    score: float | None,
    fill: dict[str, object],
    mode_label: str,
    cost_model: TradingCostModel,
    use_simulated_costs: bool,
) -> str | None:
    if use_simulated_costs:
        event = trader.enter_long(
            market=market,
            price=float(fill["price"]),
            cost=float(fill["cost"]),
            fee_rate=cost_model.fee_rate,
            slippage_bps=cost_model.slippage_bps,
            qty=float(fill["qty"]) if fill.get("qty") else None,
            strategy=strategy_name,
            order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
        )
    else:
        event = trader.apply_buy_fill(
            market=market,
            qty=float(fill["qty"]),
            gross_value=float(fill.get("value") or 0.0),
            fee_paid=float(fill.get("paid_fee") or 0.0),
            strategy=strategy_name,
            order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
        )
    if event is None:
        return None
    metrics["daily_buy"] = float(metrics.get("daily_buy") or 0.0) + float(event["cost"])
    trade_events.append(event)
    return buy_filled_message(
        mode_label,
        market=market,
        price=float(event["price"]),
        alloc=float(event["cost"]),
        score=score,
        partial=bool(fill.get("_partial")),
        qty=float(event["qty"]),
    )


def _record_exit_trade(
    trader: PaperTrader,
    metrics: dict,
    trade_events: list[dict[str, object]],
    *,
    market: str,
    reason: str,
    fill: dict[str, object],
    mode_label: str,
    cost_model: TradingCostModel,
    use_simulated_costs: bool,
) -> str | None:
    if use_simulated_costs:
        event = trader.exit_long(
            market=market,
            price=float(fill["price"]),
            reason=reason,
            fee_rate=cost_model.fee_rate,
            slippage_bps=cost_model.slippage_bps,
            order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
        )
    else:
        event = trader.apply_sell_fill(
            market=market,
            qty=float(fill["qty"]),
            gross_value=float(fill.get("value") or 0.0),
            fee_paid=float(fill.get("paid_fee") or 0.0),
            reason=reason,
            order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
        )
    if event is None:
        return None
    metrics["realized_pnl"] = float(metrics.get("realized_pnl") or 0.0) + float(event["pnl_value"])
    trade_events.append(event)
    return sell_filled_message(
        mode_label,
        market=market,
        price=float(event["price"]),
        pnl_pct=float(event["pnl_pct"]),
        partial=bool(fill.get("_partial")),
        qty=float(event["qty"]),
    )


def _reconcile_pending_orders(
    *,
    trader: PaperTrader,
    metrics: dict,
    pending_orders: dict[str, dict[str, object]],
    trade_events: list[dict[str, object]],
    notify: list[str],
    last_state: dict,
    strategy_name: str,
    timeout_seconds: float,
    cost_model: TradingCostModel,
) -> None:
    if not api or not pending_orders:
        return
    for market, pending in list(pending_orders.items()):
        if not pending.get("uuid"):
            pending_orders.pop(market, None)
            continue
        resolution = resolve_submitted_order(
            api,
            {"uuid": pending["uuid"]},
            live_orders=True,
            side=str(pending.get("side") or "").lower(),
            fallback_price=float(pending.get("fallback_price") or 0.0),
            fallback_cost=float(pending.get("requested_cost") or 0.0) or None,
            fallback_qty=float(pending.get("requested_qty") or 0.0) or None,
            timeout_seconds=min(timeout_seconds, 1.2),
            poll_interval=0.2,
        )
        status = str(resolution.get("status") or "")
        fill = dict(resolution.get("fill") or {})
        side = str(pending.get("side") or "").lower()
        updated_pending, delta = pending_fill_delta(pending, fill)
        if float(delta.get("qty") or 0.0) > 0:
            delta_fill = {
                "order_uuid": fill.get("order_uuid"),
                "qty": float(delta["qty"]),
                "value": float(delta["value"]),
                "paid_fee": float(delta["paid_fee"]),
                "cost": float(delta["net_value"]),
                "price": float(delta["price"] or fill.get("price") or 0.0),
                "_partial": status == "pending",
            }
            if side == "bid":
                message = _record_entry_trade(
                    trader,
                    metrics,
                    trade_events,
                    market=market,
                    strategy_name=strategy_name,
                    score=None,
                    fill=delta_fill,
                    mode_label="LIVE",
                    cost_model=cost_model,
                    use_simulated_costs=False,
                )
                if message:
                    notify.append(message)
            elif side == "ask":
                message = _record_exit_trade(
                    trader,
                    metrics,
                    trade_events,
                    market=market,
                    reason=str(pending.get("reason") or "signal"),
                    fill=delta_fill,
                    mode_label="LIVE",
                    cost_model=cost_model,
                    use_simulated_costs=False,
                )
                if message:
                    notify.append(message)
        if status == "pending":
            pending_orders[market] = updated_pending
            continue
        pending_orders.pop(market, None)
        if status == "cancelled":
            notify.append(order_cancelled_message("LIVE", market=market, side=side))
            if side == "bid":
                last_state[market] = {"sig": "WAIT", "entry": None}
            continue
        if status == "error":
            notify.append(lookup_failed_message("LIVE", market=market, side=side, error=resolution.get("error")))
            continue
        if side == "bid":
            if float(updated_pending.get("filled_qty") or 0.0) <= 0 or float(updated_pending.get("filled_net_value") or 0.0) <= 0:
                notify.append(order_no_fill_message("LIVE", market=market, side="buy"))
                last_state[market] = {"sig": "WAIT", "entry": None}
                continue
            position = trader.get_position(market)
            last_state[market] = {"sig": "BUY", "entry": position.entry if position else None}
        elif side == "ask":
            if float(updated_pending.get("filled_qty") or 0.0) <= 0:
                notify.append(order_no_fill_message("LIVE", market=market, side="sell"))
                continue
            last_state[market] = {"sig": "SELL", "entry": None}


def _scan_core(params: dict, last_state: dict) -> dict:
    exchange_synced = bool(params.get("_exchange_synced"))
    exchange_sync_due_at = float(params.get("_exchange_sync_due_at") or 0.0)
    live_orders = bool(params.get("live_trading")) and os.getenv("UPBIT_LIVE") == "1"
    kill_switch = effective_kill_switch(str(params.get("kill_switch_name") or LIVE_KILL_SWITCH_NAME))
    cost_model = _cost_model_from_params(params)
    current_day = current_kst_day()
    daily_reports = dict(params.get("_daily_reports") or {})
    saved_last_report_day = str(params.get("_last_daily_report_day") or "").strip()
    last_daily_report_day = saved_last_report_day or None
    sync_notifications: list[str] = []
    if live_orders and (not exchange_synced) and time.time() >= exchange_sync_due_at:
        sync_result = _sync_live_exchange_state(params, last_state)
        params["_positions"] = dict(sync_result.get("positions") or {})
        params["_pending_orders"] = dict(sync_result.get("pending_orders") or {})
        last_state = dict(sync_result.get("last_signal_state") or last_state)
        exchange_synced = bool(sync_result.get("synced"))
        exchange_sync_due_at = 0.0 if exchange_synced else (time.time() + 60.0)
        sync_notifications = list(sync_result.get("notifications") or [])
    trader = PaperTrader(params.get("_positions"))
    rollover = rollover_daily_report(
        current_day=current_day,
        mode="LIVE" if live_orders else "SIM",
        metrics=params.get("_metrics"),
        trade_log=list(params.get("_trade_log") or []),
        positions=trader.to_state(),
        pending_orders=params.get("_pending_orders"),
        daily_reports=daily_reports,
        last_report_day=last_daily_report_day,
    )
    daily_reports = dict(rollover.get("daily_reports") or {})
    saved_last_report_day = str(rollover.get("last_report_day") or "").strip()
    last_daily_report_day = saved_last_report_day or None
    notify: list[str] = []
    report_message = rollover.get("message")
    if isinstance(report_message, str) and report_message.strip():
        notify.append(report_message)
    notify.extend(sync_notifications)
    ranked = _markets_rank()
    if ranked.empty:
        return {
            "table": pd.DataFrame(),
            "detail": {},
            "last_sig": last_state,
            "notify": notify,
            "trades": [],
            "_metrics": ensure_daily_metrics(params.get("_metrics"), day=current_day),
            "_positions": dict(params.get("_positions") or {}),
            "_pending_orders": dict(params.get("_pending_orders") or {}),
            "_daily_reports": daily_reports,
            "_last_daily_report_day": last_daily_report_day,
            "_exchange_synced": exchange_synced,
            "_exchange_sync_due_at": exchange_sync_due_at,
            "last_run": time.time(),
        }

    metrics = ensure_daily_metrics(params.get("_metrics"), day=current_day)
    metrics["day_start_equity"] = _resolve_day_start_equity(metrics, trader)
    risk_config = risk_config_from_dict(params.get("risk_limits"))
    strategy_name = str(params.get("strategy_name", "research_trend"))
    pending_orders = dict(params.get("_pending_orders") or {})
    mode_label = "LIVE" if live_orders else "SIM"
    reconcile_timeout_seconds = float(params.get("reconcile_timeout_seconds") or 3.0)

    result_rows: list[dict[str, object]] = []
    detail_cache: dict[str, dict[str, object]] = {}
    trade_events: list[dict[str, object]] = []
    price_map: dict[str, float] = {}
    _reconcile_pending_orders(
        trader=trader,
        metrics=metrics,
        pending_orders=pending_orders,
        trade_events=trade_events,
        notify=notify,
        last_state=last_state,
        strategy_name=strategy_name,
        timeout_seconds=reconcile_timeout_seconds,
        cost_model=cost_model,
    )

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

        bt = backtest_signal_frame(
            frame,
            fee=float(params["fee"]),
            slippage_bps=float(params.get("slippage_bps") or 0.0),
        )
        last_row = frame.iloc[-1]
        signal = _current_signal(last_row)
        previous_signal = (last_state.get(market) or {}).get("sig", "WAIT")
        score = float(last_row.get("strategy_score", 0.0))
        close_price = float(last_row["close"])
        position = trader.get_position(market)
        price_map[market] = close_price
        pending = pending_orders.get(market)

        if pending:
            pending_side = str(pending.get("side") or "").lower()
            pending_signal = "BUY_PENDING" if pending_side == "bid" else "SELL_PENDING"
            last_state[market] = {
                "sig": pending_signal,
                "entry": position.entry if position else None,
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
                    "last_signal": pending_signal,
                    "position": "OPEN" if position else "PENDING",
                }
            )
            continue

        if signal == "BUY" and previous_signal != "BUY":
            decision = evaluate_entry(
                config=risk_config,
                metrics=metrics,
                positions=trader.to_state(),
                price_map=price_map,
                market=market,
                day_start_equity=float(metrics["day_start_equity"]),
                fee_rate=cost_model.fee_rate,
                slippage_bps=cost_model.slippage_bps,
            )
            if decision.allowed:
                if bool(kill_switch.get("enabled")):
                    notify.append(kill_switch_block_message(mode_label, market=market))
                    signal = "WAIT"
                    continue
                order_result = {"simulate": True}
                if live_orders:
                    order_result = api.create_order(market, side="bid", ord_type="price", price=f"{int(decision.trade_cost)}", simulate=False)
                resolution = resolve_submitted_order(
                    api,
                    order_result,
                    live_orders=live_orders,
                    side="bid",
                    fallback_price=close_price,
                    fallback_cost=decision.trade_cost,
                    timeout_seconds=reconcile_timeout_seconds,
                )
                fill = dict(resolution.get("fill") or {})
                if resolution.get("status") == "error":
                    notify.append(order_failed_message(mode_label, market=market, side="buy", error=resolution.get("error")))
                    signal = "WAIT"
                elif resolution.get("status") == "cancelled":
                    notify.append(order_cancelled_message(mode_label, market=market, side="buy"))
                    signal = "WAIT"
                elif resolution.get("status") == "pending" and live_orders:
                    if float(fill.get("qty") or 0.0) > 0:
                        partial_message = _record_entry_trade(
                            trader,
                            metrics,
                            trade_events,
                            market=market,
                            strategy_name=strategy_name,
                            score=score,
                            fill={**fill, "_partial": True},
                            mode_label=mode_label,
                            cost_model=cost_model,
                            use_simulated_costs=False,
                        )
                        if partial_message:
                            notify.append(partial_message)
                    pending_orders[market] = build_pending_order(
                        market=market,
                        side="bid",
                        strategy=strategy_name,
                        fallback_price=close_price,
                        order_result=resolution.get("order"),
                        requested_cost=decision.trade_cost,
                        fill=fill,
                    )
                    notify.append(
                        order_pending_message("LIVE", market=market, side="buy")
                    )
                    signal = "BUY_PENDING"
                else:
                    fill = dict(resolution.get("fill") or {})
                    if float(fill.get("qty") or 0.0) <= 0 or float(fill.get("cost") or 0.0) <= 0:
                        notify.append(order_no_fill_message(mode_label, market=market, side="buy"))
                        signal = "WAIT"
                    else:
                        trade_message = _record_entry_trade(
                            trader,
                            metrics,
                            trade_events,
                            market=market,
                            strategy_name=strategy_name,
                            score=score,
                            fill={**fill, "_partial": False},
                            mode_label=mode_label,
                            cost_model=cost_model,
                            use_simulated_costs=not live_orders,
                        )
                        if trade_message:
                            notify.append(trade_message)
            else:
                notify.append(blocked_risk_message(mode_label, market=market, reason=decision.blocked_reason, price=close_price))
                signal = "WAIT"

        elif signal == "SELL" and previous_signal == "BUY" and position is not None:
            order_result = {"simulate": True}
            if live_orders:
                order_result = api.create_order(
                    market,
                    side="ask",
                    ord_type="market",
                    volume=f"{position.qty:.8f}",
                    simulate=False,
                )
            resolution = resolve_submitted_order(
                api,
                order_result,
                live_orders=live_orders,
                side="ask",
                fallback_price=close_price,
                fallback_qty=position.qty,
                timeout_seconds=reconcile_timeout_seconds,
            )
            fill = dict(resolution.get("fill") or {})
            if resolution.get("status") == "error":
                notify.append(order_failed_message(mode_label, market=market, side="sell", error=resolution.get("error")))
                signal = "BUY"
            elif resolution.get("status") == "cancelled":
                notify.append(order_cancelled_message(mode_label, market=market, side="sell"))
                signal = "BUY"
            elif resolution.get("status") == "pending" and live_orders:
                if float(fill.get("qty") or 0.0) > 0:
                    trade_message = _record_exit_trade(
                        trader,
                        metrics,
                        trade_events,
                        market=market,
                        reason="signal",
                        fill={**fill, "_partial": True},
                        mode_label=mode_label,
                        cost_model=cost_model,
                        use_simulated_costs=False,
                    )
                    if trade_message:
                        notify.append(trade_message)
                pending_orders[market] = build_pending_order(
                    market=market,
                    side="ask",
                    strategy=strategy_name,
                    fallback_price=close_price,
                    order_result=resolution.get("order"),
                    requested_qty=position.qty,
                    reason="signal",
                    fill=fill,
                )
                notify.append(order_pending_message("LIVE", market=market, side="sell"))
                signal = "SELL_PENDING"
            else:
                fill = dict(resolution.get("fill") or {})
                if float(fill.get("qty") or 0.0) <= 0:
                    notify.append(order_no_fill_message(mode_label, market=market, side="sell"))
                    signal = "BUY"
                else:
                    trade_message = _record_exit_trade(
                        trader,
                        metrics,
                        trade_events,
                        market=market,
                        reason="signal",
                        fill={**fill, "_partial": False},
                        mode_label=mode_label,
                        cost_model=cost_model,
                        use_simulated_costs=not live_orders,
                    )
                    if trade_message:
                        notify.append(trade_message)
                    else:
                        signal = "BUY"

        updated_position = trader.get_position(market)
        updated_pending = pending_orders.get(market)
        if updated_pending:
            updated_side = str(updated_pending.get("side") or "").lower()
            signal_state = "BUY_PENDING" if updated_side == "bid" else "SELL_PENDING"
        else:
            signal_state = "BUY" if updated_position else signal
        last_state[market] = {
            "sig": signal_state,
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
                "last_signal": signal_state,
                "position": "OPEN" if updated_position else ("PENDING" if updated_pending else "-"),
            }
        )

    positions_state = trader.to_state()
    metrics["unrealized_pnl"] = total_unrealized_pnl(
        positions_state,
        price_map,
        fee_rate=cost_model.fee_rate,
        slippage_bps=cost_model.slippage_bps,
    )
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
        "_pending_orders": pending_orders,
        "_daily_reports": daily_reports,
        "_last_daily_report_day": last_daily_report_day,
        "_exchange_synced": exchange_synced,
        "_exchange_sync_due_at": exchange_sync_due_at,
        "_kill_switch_enabled": bool(kill_switch.get("enabled")),
        "_kill_switch_reason": str(kill_switch.get("reason") or ""),
        "_kill_switch_source": str(kill_switch.get("source") or "runtime"),
        "trades": trade_events,
    }


def _scan(params: dict):
    _ensure_lock()
    with st.session_state["LIVE_LOCK"]:
        scan_params = dict(params)
        scan_params["_metrics"] = st.session_state.get("LIVE_METRICS")
        scan_params["_positions"] = st.session_state.get("LIVE_POSITIONS")
        scan_params["_pending_orders"] = st.session_state.get("LIVE_PENDING_ORDERS")
        scan_params["_trade_log"] = st.session_state.get("LIVE_TRADES")
        scan_params["_daily_reports"] = st.session_state.get("LIVE_DAILY_REPORTS")
        scan_params["_last_daily_report_day"] = st.session_state.get("LIVE_LAST_DAILY_REPORT_DAY")
        scan_params["_exchange_synced"] = st.session_state.get("LIVE_EXCHANGE_SYNCED")
        scan_params["_exchange_sync_due_at"] = st.session_state.get("LIVE_EXCHANGE_SYNC_DUE_AT")
        snapshot = _scan_core(scan_params, st.session_state.get("LIVE_LAST_SIG", {}))
        st.session_state["LIVE_LAST_SIG"] = snapshot["last_sig"]
        st.session_state["LIVE_RESULTS"] = {"table": snapshot["table"], "detail": snapshot["detail"]}
        st.session_state["LIVE_LAST_RUN"] = snapshot.get("last_run")
        st.session_state["LIVE_METRICS"] = snapshot.get("_metrics")
        st.session_state["LIVE_POSITIONS"] = snapshot.get("_positions")
        st.session_state["LIVE_PENDING_ORDERS"] = snapshot.get("_pending_orders") or {}
        st.session_state["LIVE_DAILY_REPORTS"] = snapshot.get("_daily_reports") or {}
        st.session_state["LIVE_LAST_DAILY_REPORT_DAY"] = snapshot.get("_last_daily_report_day")
        st.session_state["LIVE_EXCHANGE_SYNCED"] = bool(snapshot.get("_exchange_synced"))
        st.session_state["LIVE_EXCHANGE_SYNC_DUE_AT"] = float(snapshot.get("_exchange_sync_due_at") or 0.0)
        if snapshot.get("trades"):
            st.session_state.setdefault("LIVE_TRADES", [])
            st.session_state["LIVE_TRADES"].extend(snapshot["trades"])
            st.session_state["LIVE_TRADES"] = st.session_state["LIVE_TRADES"][-500:]
        _persist_live_runtime(
            params,
            metrics=st.session_state.get("LIVE_METRICS") or {},
            positions=st.session_state.get("LIVE_POSITIONS") or {},
            last_signal_state=st.session_state.get("LIVE_LAST_SIG") or {},
            pending_orders=st.session_state.get("LIVE_PENDING_ORDERS") or {},
            trade_log=st.session_state.get("LIVE_TRADES") or [],
            daily_reports=st.session_state.get("LIVE_DAILY_REPORTS") or {},
            last_daily_report_day=st.session_state.get("LIVE_LAST_DAILY_REPORT_DAY"),
            last_run=st.session_state.get("LIVE_LAST_RUN"),
        )
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
        self.pending_orders: dict = {}
        self.trade_log: list[dict] = []
        self.daily_reports: dict = {}
        self.last_daily_report_day: str | None = None
        self.exchange_synced = False
        self.exchange_sync_due_at = 0.0
        self.last_error: str | None = None
        self.interval = 30

    def update_params(self, params: dict):
        with self.lock:
            self.params = dict(params or {})

    def _get_params(self) -> dict:
        with self.lock:
            return dict(self.params)

    def start(self, interval: int, params: dict, initial_state: dict | None = None):
        self.stop()
        self.interval = interval
        self.update_params(params)
        initial_state = dict(initial_state or {})
        if "metrics" in initial_state:
            self.metrics = dict(initial_state.get("metrics") or {})
        if "positions" in initial_state:
            self.positions = dict(initial_state.get("positions") or {})
        if "pending_orders" in initial_state:
            self.pending_orders = dict(initial_state.get("pending_orders") or {})
        if "last_signal_state" in initial_state:
            self.last_signal_state = dict(initial_state.get("last_signal_state") or {})
        if "trade_log" in initial_state:
            self.trade_log = list(initial_state.get("trade_log") or [])[-500:]
        if "daily_reports" in initial_state:
            self.daily_reports = dict(initial_state.get("daily_reports") or {})
        if "last_daily_report_day" in initial_state:
            saved_last_report_day = str(initial_state.get("last_daily_report_day") or "").strip()
            self.last_daily_report_day = saved_last_report_day or None
        if "exchange_synced" in initial_state:
            self.exchange_synced = bool(initial_state.get("exchange_synced"))
        if "exchange_sync_due_at" in initial_state:
            self.exchange_sync_due_at = float(initial_state.get("exchange_sync_due_at") or 0.0)
        self.stop_event.clear()

        def loop():
            while not self.stop_event.is_set():
                try:
                    params = self._get_params()
                    params["_metrics"] = self.metrics
                    params["_positions"] = self.positions
                    params["_pending_orders"] = self.pending_orders
                    params["_trade_log"] = self.trade_log
                    params["_daily_reports"] = self.daily_reports
                    params["_last_daily_report_day"] = self.last_daily_report_day
                    params["_exchange_synced"] = self.exchange_synced
                    params["_exchange_sync_due_at"] = self.exchange_sync_due_at
                    snapshot = _scan_core(params, self.last_signal_state)
                    notifier = get_notifier()
                    if notifier.available():
                        for message in snapshot.get("notify", [])[:10]:
                            try:
                                notifier.send_text(message)
                            except Exception:
                                pass
                    with self.lock:
                        if "_metrics" in snapshot:
                            self.metrics = dict(snapshot.get("_metrics") or {})
                        if "_positions" in snapshot:
                            self.positions = dict(snapshot.get("_positions") or {})
                        if "_pending_orders" in snapshot:
                            self.pending_orders = dict(snapshot.get("_pending_orders") or {})
                        if "_daily_reports" in snapshot:
                            self.daily_reports = dict(snapshot.get("_daily_reports") or {})
                        if "_last_daily_report_day" in snapshot:
                            saved_last_report_day = str(snapshot.get("_last_daily_report_day") or "").strip()
                            self.last_daily_report_day = saved_last_report_day or None
                        self.exchange_synced = bool(snapshot.get("_exchange_synced"))
                        self.exchange_sync_due_at = float(snapshot.get("_exchange_sync_due_at") or 0.0)
                        if "last_sig" in snapshot:
                            self.last_signal_state = dict(snapshot.get("last_sig") or {})
                        if snapshot.get("trades"):
                            self.trade_log.extend(snapshot.get("trades") or [])
                            self.trade_log = self.trade_log[-500:]
                        self.snapshot = snapshot
                        _persist_live_runtime(
                            self.params,
                            metrics=self.metrics,
                            positions=self.positions,
                            last_signal_state=self.last_signal_state,
                            pending_orders=self.pending_orders,
                            trade_log=self.trade_log,
                            daily_reports=self.daily_reports,
                            last_daily_report_day=self.last_daily_report_day,
                            last_run=snapshot.get("last_run"),
                        )
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
    strategy_name = st.selectbox("전략", options, index=index, format_func=strategy_label, key=f"{prefix}_strategy_name")
    st.caption(strategy_description(strategy_name))
    params: dict[str, float | int | str] = {}
    if strategy_name == "research_trend":
        with st.expander("연구형 추세 돌파 설정", expanded=False):
            row1 = st.columns(4)
            params["fast_ema"] = row1[0].number_input("빠른 EMA", 5, 100, 21, 1, key=f"{prefix}_fast_ema")
            params["slow_ema"] = row1[1].number_input("느린 EMA", 10, 240, 55, 1, key=f"{prefix}_slow_ema")
            params["breakout_window"] = row1[2].number_input("돌파 창", 5, 120, 20, 1, key=f"{prefix}_breakout")
            params["exit_window"] = row1[3].number_input("청산 창", 3, 80, 10, 1, key=f"{prefix}_exit")
            row2 = st.columns(4)
            params["atr_window"] = row2[0].number_input("ATR 창", 5, 50, 14, 1, key=f"{prefix}_atr_window")
            params["atr_mult"] = row2[1].number_input("ATR 배수", 1.0, 6.0, 2.5, 0.1, key=f"{prefix}_atr_mult")
            params["adx_window"] = row2[2].number_input("ADX 창", 5, 50, 14, 1, key=f"{prefix}_adx_window")
            params["adx_threshold"] = row2[3].number_input("ADX 기준", 5.0, 40.0, 18.0, 0.5, key=f"{prefix}_adx_threshold")
            row3 = st.columns(3)
            params["momentum_window"] = row3[0].number_input("모멘텀 창", 5, 80, 20, 1, key=f"{prefix}_momentum")
            params["volume_window"] = row3[1].number_input("거래량 창", 5, 80, 20, 1, key=f"{prefix}_volume_window")
            params["volume_threshold"] = row3[2].number_input("거래량 비율", 0.1, 3.0, 0.9, 0.1, key=f"{prefix}_volume_threshold")
    else:
        with st.expander("플럭스 추세 밴드 설정", expanded=False):
            row = st.columns(5)
            params["ltf_len"] = row[0].number_input("단기 기준 길이", 5, 400, 20, 1, key=f"{prefix}_ltf_len")
            params["ltf_mult"] = row[1].number_input("단기 밴드 배수", 0.1, 10.0, 2.0, 0.1, key=f"{prefix}_ltf_mult")
            params["htf_len"] = row[2].number_input("상위 주기 길이", 5, 400, 20, 1, key=f"{prefix}_htf_len")
            params["htf_mult"] = row[3].number_input("상위 밴드 배수", 0.1, 10.0, 2.25, 0.1, key=f"{prefix}_htf_mult")
            htf = row[4].selectbox("상위 주기", ["30m", "60m", "120m", "240m", "1D"], index=1, key=f"{prefix}_htf_rule")
            params["htf_rule"] = htf.replace("m", "T") if htf.endswith("m") else "1D"
    return strategy_name, params


def _render_chart(
    frame: pd.DataFrame,
    strategy_name: str,
    bt: dict[str, object],
    *,
    trade_log: list[dict[str, object]] | None = None,
    market: str | None = None,
    position: dict[str, object] | None = None,
):
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
            increasing_line_color="#22c55e",
            decreasing_line_color="#f43f5e",
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
                        line={"color": color, "width": 1.5},
                    ),
                    row=1,
                    col=1,
                )
        if "adx" in frame:
            figure.add_trace(
                go.Scatter(x=frame.index, y=frame["adx"], name=_SERIES_LABELS["adx"], line={"color": "#a78bfa"}),
                row=3,
                col=1,
            )
            figure.add_hline(y=18, line={"color": "rgba(255,255,255,0.16)", "dash": "dot"}, row=3, col=1)
        if "strategy_score" in frame:
            figure.add_trace(
                go.Scatter(x=frame.index, y=frame["strategy_score"], name=_SERIES_LABELS["strategy_score"], line={"color": "#2dd4bf"}),
                row=4,
                col=1,
            )
    else:
        for column in ["ltf_upper", "ltf_lower", "ltf_basis", "htf_upper", "htf_lower"]:
            if column in frame:
                figure.add_trace(
                    go.Scatter(x=frame.index, y=frame[column], name=_SERIES_LABELS.get(column, column), line={"width": 1.2}),
                    row=1,
                    col=1,
                )
        figure.add_trace(go.Bar(x=frame.index, y=frame["volume"], name="거래량", marker_color="rgba(96,165,250,0.5)"), row=3, col=1)

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

    if market and trade_log:
        actual_events = _trade_events_for_market(trade_log, market)
        for side, color, label in [
            ("BUY", "#22c55e", "실제 매수"),
            ("SELL", "#f43f5e", "실제 매도"),
        ]:
            points = [event for event in actual_events if event["side"] == side]
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

    entry_price = float((position or {}).get("entry") or 0.0)
    if entry_price > 0:
        figure.add_hline(
            y=entry_price,
            line={"color": "rgba(250, 204, 21, 0.75)", "dash": "dash"},
            annotation_text="보유 평균 진입가",
            annotation_position="top left",
            row=1,
            col=1,
        )

    equity = bt.get("equity")
    if equity is not None:
        figure.add_trace(go.Scatter(x=equity.index, y=equity.values, name="누적 수익곡선", line={"color": "#f8fafc"}), row=2, col=1)
    figure.update_yaxes(title_text="가격", row=1, col=1)
    figure.update_yaxes(title_text="수익곡선", row=2, col=1)
    if is_research:
        figure.update_yaxes(title_text="ADX", row=3, col=1)
        figure.update_yaxes(title_text="점수", row=4, col=1)
    else:
        figure.update_yaxes(title_text="거래량", row=3, col=1)
    return apply_chart_theme(figure, height=920 if is_research else 860)


def render_live():
    page_intro(
        "라이브",
        "실시간 감시와 자동 실행 데스크",
        "빈 삼각형은 전략 신호, 다이아몬드는 실제 체결입니다. 실거래 주문은 환경변수와 화면 확인을 모두 통과해야만 활성화됩니다.",
    )
    if not api:
        st.error("API가 초기화되지 않았습니다.")
        return
    _hydrate_live_runtime()

    controls = st.columns([1, 1, 1, 1, 1, 1])
    interval_options = ["minute15", "minute30", "minute60", "minute240", "day"]
    interval = controls[0].selectbox("주기", interval_options, index=1, format_func=_interval_text, key="live_interval")
    count = int(controls[1].number_input("캔들 수", 120, 1200, 360, 20, key="live_count"))
    topn = int(controls[2].number_input("상위 종목 수", 5, 60, 20, 1, key="live_topn"))
    fee = float(controls[3].number_input("수수료", 0.0, 0.01, 0.0005, 0.0001, format="%.4f", key="live_fee"))
    slippage_bps = float(controls[4].number_input("슬리피지 (bps)", 0.0, 100.0, 3.0, 0.5, key="live_slippage_bps"))
    auto = controls[5].checkbox("값 변경 시 자동 동기화", value=True, key="live_auto")
    strategy_name, strategy_params = _strategy_controls("live")

    with st.expander("실거래 안전장치", expanded=False):
        guard_cols = st.columns([1, 1, 1.2])
        live_toggle = guard_cols[0].checkbox("실거래 주문 허용", value=st.session_state.get("LIVE_TRADING_RAW", False))
        confirm_text = guard_cols[1].text_input("확인 문구 입력 (LIVE)", value=st.session_state.get("LIVE_TRADING_CONFIRM", ""))
        env_live = os.getenv("UPBIT_LIVE") == "1"
        live_trading = bool(env_live and live_toggle and confirm_text.strip().upper() == "LIVE")
        st.session_state["LIVE_TRADING_RAW"] = live_toggle
        st.session_state["LIVE_TRADING_CONFIRM"] = confirm_text
        st.session_state["LIVE_TRADING"] = live_trading
        if not env_live:
            st.info("환경변수 잠금이 걸려 있습니다. 실제 주문을 보내려면 `UPBIT_LIVE=1` 이 먼저 필요합니다.")
        elif live_toggle and confirm_text.strip().upper() != "LIVE":
            st.warning("실거래를 활성화하려면 확인 문구에 `LIVE` 를 정확히 입력해야 합니다.")
        elif live_trading:
            st.success("실거래 주문 준비가 완료되었습니다.")

    kill_state = load_kill_switch(LIVE_KILL_SWITCH_NAME)
    with st.expander("긴급중지", expanded=bool(kill_state.get("enabled"))):
        kill_cols = st.columns([1, 1, 2.2])
        kill_reason = kill_cols[2].text_input(
            "중지 사유",
            value=str(kill_state.get("reason") or ""),
            key="live_kill_switch_reason",
            placeholder="예: 이상 체결 감지, 수동 점검",
        )
        enable_kill = kill_cols[0].button("중지 켜기", use_container_width=True)
        disable_kill = kill_cols[1].button("거래 재개", use_container_width=True)
        if enable_kill:
            kill_state = save_kill_switch(
                LIVE_KILL_SWITCH_NAME,
                enabled=True,
                reason=kill_reason or "수동 긴급중지",
            )
            st.session_state["LIVE_KILL_SWITCH_NOTICE"] = kill_switch_enabled_message(
                reason=kill_state.get("reason"),
                source=str(kill_state.get("source") or "runtime"),
            )
        elif disable_kill:
            kill_state = save_kill_switch(LIVE_KILL_SWITCH_NAME, enabled=False, reason="")
            st.session_state["LIVE_KILL_SWITCH_NOTICE"] = kill_switch_disabled_message()

    kill_state = effective_kill_switch(LIVE_KILL_SWITCH_NAME)
    if st.session_state.get("LIVE_KILL_SWITCH_NOTICE"):
        st.info(str(st.session_state["LIVE_KILL_SWITCH_NOTICE"]))
    if bool(kill_state.get("enabled")):
        st.error(kill_switch_enabled_message(reason=kill_state.get("reason"), source=str(kill_state.get("source") or "runtime")))
    else:
        st.caption("긴급중지가 비활성화되어 있습니다. 필요하면 위 패널에서 즉시 차단할 수 있습니다.")

    with st.expander("리스크 제한", expanded=False):
        row1 = st.columns(3)
        max_trade_krw = float(row1[0].number_input("1회 최대 주문금액 (KRW)", 0, 10_000_000_000, 50000, 10000, format="%d", key="live_max_trade_krw"))
        max_trade_pct = float(row1[1].number_input("1회 최대 주문비중 (%)", 0.0, 100.0, 2.0, 0.5, format="%.1f", key="live_max_trade_pct"))
        per_asset_max_pct = float(row1[2].number_input("종목당 최대 비중 (%)", 0.0, 100.0, 10.0, 0.5, format="%.1f", key="live_per_asset_max_pct"))
        row2 = st.columns(3)
        daily_buy_limit = float(row2[0].number_input("일일 총 매수한도 (KRW)", 0, 10_000_000_000, 200000, 50000, format="%d", key="live_daily_buy_limit"))
        daily_loss_limit_krw = float(row2[1].number_input("일일 손실한도 (KRW)", 0, 10_000_000_000, 30000, 10000, format="%d", key="live_daily_loss_limit_krw"))
        daily_loss_limit_pct = float(row2[2].number_input("일일 손실한도 (%)", 0.0, 100.0, 3.0, 0.5, format="%.1f", key="live_daily_loss_limit_pct"))
        include_unrealized = st.checkbox("미실현 손실도 일일 손절에 포함", value=True, key="live_include_unrealized_loss")
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
        "slippage_bps": slippage_bps,
        "strategy_name": strategy_name,
        "risk_limits": risk_limits,
        "live_trading": live_trading,
        "reconcile_timeout_seconds": 3.0,
        "kill_switch_name": LIVE_KILL_SWITCH_NAME,
        **strategy_params,
    }
    params["worker_interval"] = int(st.session_state.get("live_worker_interval", 30) or 30)
    managed_worker_config = _managed_worker_config_from_live_params(params)

    with st.expander("실행 방식 안내", expanded=False):
        st.markdown(
            "\n".join(
                [
                    "- `페이지 워커 시작`: 현재 브라우저 세션 안에서만 도는 페이지 전용 워커입니다. 페이지를 닫거나 앱이 내려가면 멈춥니다.",
                    "- `백그라운드 워커 시작`: 별도 Python 프로세스로 도는 CLI 워커입니다. 텔레그램 제어와 같은 워커를 바라봅니다.",
                    "- `telegram-control`: 텔레그램으로 명령을 받는 별도 봇 프로세스입니다. 이 프로세스가 켜져 있어야 `/status`, `/start_worker` 같은 명령이 동작합니다.",
                    "- 현재 텔레그램은 기본적으로 알림 발송용이고, 제어 명령은 `telegram-control`을 실행했을 때만 활성화됩니다.",
                ]
            )
        )
        st.code(
            "python -m streamlit run .\\src\\app_streamlit.py\n"
            "python .\\src\\main.py telegram-control\n"
            "python .\\src\\main.py worker-status",
            language="powershell",
        )

    managed_status = load_managed_worker_status()
    with st.expander("백그라운드 CLI 워커 (텔레그램 / 자동화)", expanded=bool(managed_status.get("running"))):
        status_cols = st.columns(6)
        status_cols[0].metric("실행중", _mode_text("ON" if managed_status.get("running") else "OFF"))
        status_cols[1].metric("모드", _mode_text(str(managed_status.get("mode") or "-")))
        status_cols[2].metric("종목 수", int(managed_status.get("config", {}).get("markets") or 0))
        status_cols[3].metric("주기", f"{int(managed_status.get('config', {}).get('loop_seconds') or 0)}초")
        status_cols[4].metric("보유", int(managed_status.get("positions_count") or 0))
        status_cols[5].metric("미체결", int(managed_status.get("pending_orders_count") or 0))

        action_cols = st.columns([1, 1, 1.5])
        if action_cols[0].button("백그라운드 워커 시작", use_container_width=True):
            managed_status = start_managed_worker(managed_worker_config)
            st.session_state["LIVE_BG_NOTICE"] = "백그라운드 CLI 워커를 시작했습니다."
        if action_cols[1].button("백그라운드 워커 중지", use_container_width=True):
            managed_status = stop_managed_worker()
            st.session_state["LIVE_BG_NOTICE"] = "백그라운드 CLI 워커를 중지했습니다."
        action_cols[2].caption("이 워커는 웹 페이지를 닫아도 계속 돌 수 있고, 텔레그램 제어 대상도 이 워커입니다.")

        if st.session_state.get("LIVE_BG_NOTICE"):
            st.info(str(st.session_state["LIVE_BG_NOTICE"]))
        st.code(format_worker_status(managed_status), language="text")

        log_tail = read_worker_log_tail(20)
        if log_tail:
            with st.expander("최근 CLI 로그", expanded=False):
                st.code(log_tail, language="text")

    with st.expander("Windows 자동 시작", expanded=False):
        if not startup_supported():
            st.info("이 기능은 Windows에서만 지원됩니다.")
        else:
            startup_bundle = load_startup_status_bundle()
            startup_table = _present_startup_status(startup_bundle)
            if not startup_table.empty:
                st.dataframe(startup_table, use_container_width=True, hide_index=True)
            st.code(format_startup_status_bundle(startup_bundle), language="text")
            st.caption("로그인 시 자동 시작되도록 예약 작업을 설치합니다. 절전모드에서는 계속 실행되지 않습니다.")

            install_cols = st.columns(3)
            if install_cols[0].button("워커 자동시작 설치", use_container_width=True):
                result = install_startup_task("worker")
                st.session_state["LIVE_STARTUP_NOTICE"] = (
                    "워커 자동 시작을 설치했습니다." if result.get("ok") else f"워커 자동 시작 설치 실패: {result.get('stderr') or result.get('stdout') or '원인 미상'}"
                )
            if install_cols[1].button("텔레그램 자동시작 설치", use_container_width=True):
                result = install_startup_task("telegram")
                st.session_state["LIVE_STARTUP_NOTICE"] = (
                    "텔레그램 자동 시작을 설치했습니다."
                    if result.get("ok")
                    else f"텔레그램 자동 시작 설치 실패: {result.get('stderr') or result.get('stdout') or '원인 미상'}"
                )
            if install_cols[2].button("둘 다 설치", use_container_width=True):
                worker_result = install_startup_task("worker")
                telegram_result = install_startup_task("telegram")
                ok = bool(worker_result.get("ok")) and bool(telegram_result.get("ok"))
                st.session_state["LIVE_STARTUP_NOTICE"] = "워커와 텔레그램 자동 시작을 설치했습니다." if ok else "자동 시작 설치 중 일부가 실패했습니다."

            remove_cols = st.columns(3)
            if remove_cols[0].button("워커 자동시작 제거", use_container_width=True):
                result = remove_startup_task("worker")
                st.session_state["LIVE_STARTUP_NOTICE"] = (
                    "워커 자동 시작을 제거했습니다." if result.get("ok") else f"워커 자동 시작 제거 실패: {result.get('stderr') or result.get('stdout') or '원인 미상'}"
                )
            if remove_cols[1].button("텔레그램 자동시작 제거", use_container_width=True):
                result = remove_startup_task("telegram")
                st.session_state["LIVE_STARTUP_NOTICE"] = (
                    "텔레그램 자동 시작을 제거했습니다."
                    if result.get("ok")
                    else f"텔레그램 자동 시작 제거 실패: {result.get('stderr') or result.get('stdout') or '원인 미상'}"
                )
            if remove_cols[2].button("둘 다 제거", use_container_width=True):
                worker_result = remove_startup_task("worker")
                telegram_result = remove_startup_task("telegram")
                ok = bool(worker_result.get("ok")) and bool(telegram_result.get("ok"))
                st.session_state["LIVE_STARTUP_NOTICE"] = "워커와 텔레그램 자동 시작을 제거했습니다." if ok else "자동 시작 제거 중 일부가 실패했습니다."

            if st.session_state.get("LIVE_STARTUP_NOTICE"):
                st.info(str(st.session_state["LIVE_STARTUP_NOTICE"]))

    worker_cols = st.columns([1, 1, 1, 1.4])
    worker_interval = int(worker_cols[0].number_input("페이지 워커 주기(초)", 5, 3600, 30, 1, key="live_worker_interval"))
    params["worker_interval"] = worker_interval
    changed = params != (st.session_state.get("LIVE_PARAMS") or {})
    st.session_state["LIVE_PARAMS"] = dict(params)
    start = worker_cols[1].button("페이지 워커 시작", use_container_width=True)
    stop = worker_cols[2].button("페이지 워커 중지", use_container_width=True)
    worker_cols[3].markdown(f"**모드:** {_mode_text('LIVE' if live_trading else 'SIM')} / **전략:** `{strategy_label(strategy_name)}`")

    if "LIVE_WORKER" not in st.session_state:
        st.session_state["LIVE_WORKER"] = _Worker()
    worker: _Worker = st.session_state["LIVE_WORKER"]
    if start:
        st.session_state["LIVE_WORKING"] = True
        worker.start(
            worker_interval,
            params,
            initial_state={
                "metrics": st.session_state.get("LIVE_METRICS"),
                "positions": st.session_state.get("LIVE_POSITIONS"),
                "pending_orders": st.session_state.get("LIVE_PENDING_ORDERS"),
                "last_signal_state": st.session_state.get("LIVE_LAST_SIG"),
                "trade_log": st.session_state.get("LIVE_TRADES"),
                "daily_reports": st.session_state.get("LIVE_DAILY_REPORTS"),
                "last_daily_report_day": st.session_state.get("LIVE_LAST_DAILY_REPORT_DAY"),
                "exchange_synced": st.session_state.get("LIVE_EXCHANGE_SYNCED"),
                "exchange_sync_due_at": st.session_state.get("LIVE_EXCHANGE_SYNC_DUE_AT"),
            },
        )
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
            st.session_state["LIVE_PENDING_ORDERS"] = snapshot.get("_pending_orders") or {}
            st.session_state["LIVE_DAILY_REPORTS"] = snapshot.get("_daily_reports") or {}
            st.session_state["LIVE_LAST_DAILY_REPORT_DAY"] = snapshot.get("_last_daily_report_day")
            st.session_state["LIVE_EXCHANGE_SYNCED"] = bool(snapshot.get("_exchange_synced"))
            st.session_state["LIVE_EXCHANGE_SYNC_DUE_AT"] = float(snapshot.get("_exchange_sync_due_at") or 0.0)
            if snapshot.get("trades") and snapshot.get("last_run") != st.session_state.get("LIVE_LAST_CONSUMED_RUN"):
                st.session_state.setdefault("LIVE_TRADES", [])
                st.session_state["LIVE_TRADES"].extend(snapshot.get("trades"))
                st.session_state["LIVE_TRADES"] = st.session_state["LIVE_TRADES"][-500:]
                st.session_state["LIVE_LAST_CONSUMED_RUN"] = snapshot.get("last_run")

    metrics = st.session_state.get("LIVE_METRICS") or {}
    metric_cols = st.columns(7)
    metric_cols[0].metric("기준일", metrics.get("day_date", "-"))
    metric_cols[1].metric("일일 매수", fmt_full_number(metrics.get("daily_buy"), 0))
    metric_cols[2].metric("실현손익", fmt_full_number(metrics.get("realized_pnl"), 0))
    metric_cols[3].metric("미실현손익", fmt_full_number(metrics.get("unrealized_pnl"), 0))
    metric_cols[4].metric("총 손익", fmt_full_number(metrics.get("total_pnl"), 0))
    metric_cols[5].metric("페이지 워커", _mode_text("ON" if st.session_state.get("LIVE_WORKING") else "OFF"))
    metric_cols[6].metric("긴급중지", _mode_text("ON" if bool(kill_state.get("enabled")) else "OFF"))

    result = st.session_state.get("LIVE_RESULTS") or {}
    table = result.get("table")
    if table is not None and isinstance(table, pd.DataFrame) and not table.empty:
        st.dataframe(_present_leaderboard(table), use_container_width=True, hide_index=True)
        selected_market = st.selectbox("상세 종목", table["market"].tolist(), key="live_detail_market")
        detail = result.get("detail") or {}
        if selected_market in detail:
            st.caption("빈 삼각형은 전략 신호, 다이아몬드는 실제 체결, 노란 점선은 현재 보유 평균 진입가입니다.")
            st.plotly_chart(
                _render_chart(
                    detail[selected_market]["df"],
                    strategy_name,
                    detail[selected_market]["bt"],
                    trade_log=st.session_state.get("LIVE_TRADES") or [],
                    market=selected_market,
                    position=(st.session_state.get("LIVE_POSITIONS") or {}).get(selected_market),
                ),
                use_container_width=True,
            )
    else:
        st.info("라이브 데스크를 한 번 실행하면 상위 종목 표가 생성됩니다.")

    positions = st.session_state.get("LIVE_POSITIONS") or {}
    if positions:
        position_rows = []
        market_price_map = {row["market"]: row["price"] for row in table.to_dict("records")} if table is not None and not table.empty else {}
        cost_model = cost_model_from_values(fee_rate=fee, slippage_bps=slippage_bps)
        for market, raw in positions.items():
            current_price = market_price_map.get(market)
            entry = float(raw.get("entry") or 0.0)
            qty = float(raw.get("qty") or 0.0)
            cost = float(raw.get("cost") or 0.0)
            if current_price is not None:
                exit_view = cost_model.simulate_exit(price=float(current_price), qty=qty, cost_basis=cost)
                pnl_value = exit_view["pnl_value"]
                pnl_pct = exit_view["pnl_pct"]
            else:
                pnl_value = None
                pnl_pct = None
            position_rows.append({"market": market, "qty": qty, "entry": entry, "cost": cost, "current_price": current_price, "pnl_value": pnl_value, "pnl_pct": pnl_pct})
        st.subheader("보유 포지션")
        st.dataframe(_present_positions(position_rows), use_container_width=True, hide_index=True)

    st.session_state.setdefault("LIVE_TRADES", [])
    with st.expander("체결 로그", expanded=False):
        if st.button("로그 비우기"):
            st.session_state["LIVE_TRADES"] = []
        logs = st.session_state.get("LIVE_TRADES") or []
        if logs:
            st.dataframe(_present_trade_log(logs), use_container_width=True, hide_index=True)
        else:
            st.caption("아직 체결 로그가 없습니다.")

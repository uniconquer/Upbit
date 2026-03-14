"""CLI strategy monitor aligned with the shared backtest/live engine."""

from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
from dotenv import load_dotenv

try:
    from daily_summary import current_kst_day, rollover_daily_report
    from exchange_state import sync_exchange_state
    from execution import (
        build_client_order_identifier,
        build_pending_order,
        extract_fill_metrics,
        normalize_ws_order_event,
        pending_fill_delta,
        resolve_submitted_order,
    )
    from kill_switch import effective_kill_switch
    from notifier import get_notifier
    from notification_text import (
        blocked_max_open_message,
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
        start_message,
    )
    from paper_trader import PaperTrader
    from risk_manager import ensure_daily_metrics, evaluate_entry, risk_config_from_dict, total_unrealized_pnl
    from runtime_store import load_runtime_state, save_runtime_state
    from strategy import backtest_signal_frame
    from strategy_engine import build_strategy_frame, strategy_label
    from trading_costs import TradingCostModel, cost_model_from_values
    from upbit_api import UpbitAPI
except ImportError:
    from src.daily_summary import current_kst_day, rollover_daily_report
    from src.exchange_state import sync_exchange_state
    from src.execution import (
        build_client_order_identifier,
        build_pending_order,
        extract_fill_metrics,
        normalize_ws_order_event,
        pending_fill_delta,
        resolve_submitted_order,
    )
    from src.kill_switch import effective_kill_switch
    from src.notifier import get_notifier
    from src.notification_text import (
        blocked_max_open_message,
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
        start_message,
    )
    from src.paper_trader import PaperTrader
    from src.risk_manager import ensure_daily_metrics, evaluate_entry, risk_config_from_dict, total_unrealized_pnl
    from src.runtime_store import load_runtime_state, save_runtime_state
    from src.strategy import backtest_signal_frame
    from src.strategy_engine import build_strategy_frame, strategy_label
    from src.trading_costs import TradingCostModel, cost_model_from_values
    from src.upbit_api import UpbitAPI

try:
    from flux_bbands_mtf_kalman import indicator as flux_indicator  # type: ignore
    from flux_bbands_mtf_kalman import indicator_with_ema as flux_indicator_with_ema  # type: ignore
except Exception:
    try:
        from src.flux_bbands_mtf_kalman import indicator as flux_indicator  # type: ignore
        from src.flux_bbands_mtf_kalman import indicator_with_ema as flux_indicator_with_ema  # type: ignore
    except Exception:
        flux_indicator = None  # type: ignore
        flux_indicator_with_ema = None  # type: ignore


load_dotenv()

KST = timezone(timedelta(hours=9))


def fetch_top_markets(
    api: UpbitAPI,
    *,
    base: str = "KRW",
    limit: int = 40,
    exclude_stables: bool = True,
) -> list[str]:
    markets = api.markets()
    prefix = base.upper() + "-"
    filtered = [item for item in markets if isinstance(item.get("market"), str) and item["market"].startswith(prefix)]
    market_codes = [item["market"] for item in filtered]
    tickers = api.tickers(market_codes)
    ordered = sorted(tickers, key=lambda item: float(item.get("acc_trade_price_24h") or 0.0), reverse=True)
    result = [item["market"] for item in ordered[:limit]]
    if exclude_stables:
        stable_keywords = ("USDT", "USDC", "USDJ", "DAI", "UST")
        result = [market for market in result if not any(keyword in market for keyword in stable_keywords)]
    return result


def _frame_from_candles(candles) -> pd.DataFrame:
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


def _current_signal(last_row: pd.Series) -> str:
    if bool(last_row.get("buy_signal")):
        return "BUY"
    if bool(last_row.get("sell_signal")):
        return "SELL"
    return "WAIT"


def _strategy_params_from_args(args) -> dict[str, Any]:
    if args.strategy == "research_trend":
        return {
            "fast_ema": args.fast_ema,
            "slow_ema": args.slow_ema,
            "breakout_window": args.breakout_window,
            "exit_window": args.exit_window,
            "atr_window": args.atr_window,
            "atr_mult": args.atr_mult,
            "adx_window": args.adx_window,
            "adx_threshold": args.adx_threshold,
            "momentum_window": args.momentum_window,
            "volume_window": args.volume_window,
            "volume_threshold": args.volume_threshold,
        }
    if args.strategy == "relative_strength_rotation":
        return {
            "rs_short_window": args.rs_short_window,
            "rs_mid_window": args.rs_mid_window,
            "rs_long_window": args.rs_long_window,
            "trend_ema_window": args.trend_ema_window,
            "breakout_window": args.breakout_window,
            "atr_window": args.atr_window,
            "atr_mult": args.atr_mult,
            "volume_window": args.volume_window,
            "volume_threshold": args.volume_threshold,
            "entry_score": args.entry_score,
            "exit_score": args.exit_score,
        }
    if args.strategy == "flux_trend":
        return {
            "ltf_len": args.ltf_len,
            "ltf_mult": args.ltf_mult,
            "htf_len": args.htf_len,
            "htf_mult": args.htf_mult,
            "htf_rule": args.htf_rule,
        }
    return {
        "ltf_len": args.ltf_len,
        "ltf_mult": args.ltf_mult,
        "htf_len": args.htf_len,
        "htf_mult": args.htf_mult,
        "htf_rule": args.htf_rule,
        "sensitivity": args.sensitivity,
        "atr_period": args.atr_period,
        "trend_ema_length": args.trend_ema_length,
        "confirm_window": args.confirm_window,
        "use_heikin_ashi": args.use_heikin_ashi,
    }


def _risk_limits_from_args(args) -> dict[str, Any]:
    max_trade_krw = args.max_trade_krw_legacy if args.max_trade_krw_legacy is not None else args.max_trade_krw
    return {
        "max_trade_krw": max_trade_krw,
        "max_trade_pct": args.max_trade_pct,
        "per_asset_max_pct": args.per_asset_max_pct,
        "daily_buy_limit": args.daily_buy_limit,
        "daily_loss_limit_krw": args.daily_loss_limit_krw,
        "daily_loss_limit_pct": args.daily_loss_limit_pct,
        "include_unrealized_loss": args.include_unrealized_loss,
    }


class MRMonitor:
    def __init__(
        self,
        api: UpbitAPI,
        *,
        interval: str = "minute30",
        candles_count: int = 240,
        strategy_name: str = "research_trend",
        strategy_params: dict[str, Any] | None = None,
        risk_limits: dict[str, Any] | None = None,
        fee: float = 0.0005,
        live_orders: bool = False,
        max_open: int = 5,
        min_fetch_seconds: float = 20.0,
        per_request_sleep: float = 0.12,
        state_name: str | None = None,
        reconcile_timeout_seconds: float = 3.0,
        cost_model: TradingCostModel | None = None,
        kill_switch_name: str = "trade-kill-switch",
        exchange_sync_interval_seconds: float = 180.0,
        unknown_order_ttl_seconds: float = 45.0,
        enable_my_order_stream: bool = True,
    ):
        self.api = api
        self.interval = interval
        self.candles_count = candles_count
        self.strategy_name = strategy_name
        self.strategy_params = dict(strategy_params or {})
        self.risk_config = risk_config_from_dict(risk_limits)
        self.fee = fee
        self.live_orders = bool(live_orders) and os.getenv("UPBIT_LIVE") == "1"
        self.max_open = max_open
        self.min_fetch_seconds = min_fetch_seconds
        self.per_request_sleep = per_request_sleep
        self.state_name = state_name
        self.reconcile_timeout_seconds = reconcile_timeout_seconds
        self.cost_model = cost_model or cost_model_from_values()
        self.kill_switch_name = kill_switch_name
        self.exchange_sync_interval_seconds = max(float(exchange_sync_interval_seconds), 30.0)
        self.unknown_order_ttl_seconds = max(float(unknown_order_ttl_seconds), 10.0)
        self.enable_my_order_stream = bool(enable_my_order_stream)
        self.notifier = get_notifier()
        self.trader = PaperTrader()
        self.metrics = ensure_daily_metrics({}, day=current_kst_day())
        self.last_signal_state: dict[str, dict[str, Any]] = {}
        self.market_prices: dict[str, float] = {}
        self.trade_log: list[dict[str, Any]] = []
        self.pending_orders: dict[str, dict[str, Any]] = {}
        self.daily_reports: dict[str, dict[str, Any]] = {}
        self.last_daily_report_day: str | None = None
        self._last_fetch: dict[str, float] = {}
        self._exchange_synced = False
        self._next_exchange_sync_at = 0.0
        self.kill_switch_state = {"enabled": False, "reason": "", "source": "runtime"}
        self._kill_switch_notified_state: tuple[bool, str, str] | None = None
        self._kill_switch_market_notice: dict[str, float] = {}
        self._order_event_lock = threading.Lock()
        self._order_events: list[dict[str, Any]] = []
        self._my_order_thread: threading.Thread | None = None
        self._my_order_stop_event = threading.Event()
        self._hydrate_state()

    def _mode_label(self) -> str:
        return "LIVE" if self.live_orders else "SIM"

    def _notify(self, message: str) -> None:
        now_kst = datetime.now(timezone.utc).astimezone(KST)
        output = f"[{now_kst.strftime('%H:%M:%S')} KST] {message}"
        print(output)
        if self.notifier.available():
            try:
                self.notifier.send_text(output)
            except Exception:
                pass

    def _compute_day_start_equity(self) -> float:
        try:
            accounts = self.api.accounts()
        except Exception:
            accounts = []
        if accounts:
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
                        for item in self.api.tickers(markets)
                        if item.get("market")
                    }
                    for currency, amount in balances:
                        total += amount * price_map.get(f"KRW-{currency}", 0.0)
                except Exception:
                    pass
            if total > 0:
                return total
        estimated = sum(position.cost for position in self.trader.positions.values()) + float(self.metrics.get("daily_buy") or 0.0)
        return max(estimated, 1.0)

    def _refresh_metrics(self) -> None:
        current_day = current_kst_day()
        rollover = rollover_daily_report(
            current_day=current_day,
            mode=self._mode_label(),
            metrics=self.metrics,
            trade_log=self.trade_log,
            positions=self.trader.to_state(),
            pending_orders=self.pending_orders,
            daily_reports=self.daily_reports,
            last_report_day=self.last_daily_report_day,
        )
        self.daily_reports = dict(rollover.get("daily_reports") or {})
        saved_last_report_day = str(rollover.get("last_report_day") or "").strip()
        self.last_daily_report_day = saved_last_report_day or None
        report_message = rollover.get("message")
        if isinstance(report_message, str) and report_message.strip():
            self._notify(report_message)

        self.metrics = ensure_daily_metrics(self.metrics, day=current_day)
        if not self.metrics.get("day_start_equity"):
            self.metrics["day_start_equity"] = self._compute_day_start_equity()

    def _runtime_snapshot(self) -> dict[str, Any]:
        return {
            "version": 1,
            "saved_at": time.time(),
            "strategy_name": self.strategy_name,
            "interval": self.interval,
            "metrics": self.metrics,
            "positions": self.trader.to_state(),
            "last_signal_state": self.last_signal_state,
            "trade_log": self.trade_log[-500:],
            "pending_orders": self.pending_orders,
            "daily_reports": self.daily_reports,
            "last_daily_report_day": self.last_daily_report_day,
            "exchange_synced": self._exchange_synced,
            "next_exchange_sync_at": self._next_exchange_sync_at,
        }

    def _hydrate_state(self) -> None:
        if not self.state_name:
            return
        snapshot = load_runtime_state(self.state_name, default={})
        if not isinstance(snapshot, dict) or not snapshot:
            return
        self.metrics = dict(snapshot.get("metrics") or self.metrics)
        self.trader = PaperTrader(snapshot.get("positions"))
        self.last_signal_state = dict(snapshot.get("last_signal_state") or {})
        self.trade_log = list(snapshot.get("trade_log") or [])[-500:]
        self.pending_orders = dict(snapshot.get("pending_orders") or {})
        self.daily_reports = dict(snapshot.get("daily_reports") or {})
        saved_last_report_day = str(snapshot.get("last_daily_report_day") or "").strip()
        self.last_daily_report_day = saved_last_report_day or None
        self._exchange_synced = bool(snapshot.get("exchange_synced"))
        self._next_exchange_sync_at = float(snapshot.get("next_exchange_sync_at") or 0.0)

    def _save_state(self) -> None:
        if not self.state_name:
            return
        save_runtime_state(self.state_name, self._runtime_snapshot())

    def _queue_order_event(self, message: Any) -> None:
        normalized = normalize_ws_order_event(message if isinstance(message, dict) else None)
        if not normalized.get("uuid") and not normalized.get("identifier"):
            return
        with self._order_event_lock:
            self._order_events.append(normalized)
            self._order_events = self._order_events[-200:]

    def _drain_order_events(self) -> list[dict[str, Any]]:
        with self._order_event_lock:
            events = list(self._order_events)
            self._order_events.clear()
        return events

    def _ensure_my_order_stream(self) -> None:
        if not self.live_orders or not self.enable_my_order_stream:
            return
        if self._my_order_thread and self._my_order_thread.is_alive():
            return
        stream = getattr(self.api, "stream_my_order", None)
        if not callable(stream):
            return
        self._my_order_stop_event = threading.Event()
        try:
            self._my_order_thread = stream(
                self._queue_order_event,
                stop_event=self._my_order_stop_event,
            )
        except Exception:
            self._my_order_thread = None

    def _stop_my_order_stream(self) -> None:
        self._my_order_stop_event.set()
        if self._my_order_thread and self._my_order_thread.is_alive():
            self._my_order_thread.join(timeout=0.5)
        self._my_order_thread = None

    def _sync_with_exchange(self) -> None:
        if not self.live_orders:
            return
        now = time.time()
        if now < self._next_exchange_sync_at:
            return
        result = sync_exchange_state(
            self.api,
            strategy_name=self.strategy_name,
            existing_positions=self.trader.to_state(),
            existing_pending_orders=self.pending_orders,
            existing_signal_state=self.last_signal_state,
            announce_success=not self._exchange_synced,
        )
        self.trader = PaperTrader(result.get("positions"))
        self.pending_orders = dict(result.get("pending_orders") or {})
        self.last_signal_state = dict(result.get("last_signal_state") or {})
        for message in result.get("notifications") or []:
            self._notify(message)
        self._exchange_synced = bool(result.get("synced"))
        self._next_exchange_sync_at = now + (self.exchange_sync_interval_seconds if self._exchange_synced else 60.0)
        self._save_state()

    def _refresh_kill_switch(self) -> None:
        state = effective_kill_switch(self.kill_switch_name)
        token = (bool(state.get("enabled")), str(state.get("reason") or ""), str(state.get("source") or "runtime"))
        if token != self._kill_switch_notified_state:
            if token[0]:
                self._notify(kill_switch_enabled_message(reason=token[1], source=token[2]))
            elif self._kill_switch_notified_state is not None:
                self._notify(kill_switch_disabled_message())
            self._kill_switch_notified_state = token
        self.kill_switch_state = state

    def _kill_switch_blocks_entry(self, market: str) -> bool:
        if not bool(self.kill_switch_state.get("enabled")):
            return False
        now = time.time()
        last_notice = float(self._kill_switch_market_notice.get(market) or 0.0)
        if now - last_notice >= 600.0:
            self._notify(kill_switch_block_message(self._mode_label(), market=market))
            self._kill_switch_market_notice[market] = now
        return True

    def _place_order(
        self,
        *,
        market: str,
        side: str,
        ord_type: str,
        volume: str | None = None,
        price: str | None = None,
        identifier: str | None = None,
    ) -> dict[str, Any]:
        if self.live_orders:
            return self.api.create_order(
                market,
                side=side,
                ord_type=ord_type,
                volume=volume,
                price=price,
                simulate=False,
                identifier=identifier,
            )
        return self.api.create_order(
            market,
            side=side,
            ord_type=ord_type,
            volume=volume,
            price=price,
            simulate=True,
            identifier=identifier,
        )

    def _build_frame(self, market: str) -> pd.DataFrame:
        now = time.time()
        last_fetch = self._last_fetch.get(market, 0.0)
        if now - last_fetch < self.min_fetch_seconds:
            return pd.DataFrame()
        try:
            candles = self.api.candles(market, interval=self.interval, count=self.candles_count)
        except Exception as exc:
            if "429" in str(exc):
                self._last_fetch[market] = now + (self.min_fetch_seconds * 0.5)
            return pd.DataFrame()
        self._last_fetch[market] = now
        raw = _frame_from_candles(candles)
        if raw.empty:
            return raw
        return build_strategy_frame(
            raw[["open", "high", "low", "close", "volume"]],
            strategy_name=self.strategy_name,
            params=self.strategy_params,
            flux_indicator=flux_indicator,
            flux_indicator_with_ema=flux_indicator_with_ema,
        )

    def _record_trade(self, event: dict[str, Any]) -> None:
        self.trade_log.append(event)
        self.trade_log = self.trade_log[-500:]

    def _apply_buy_fill(self, market: str, score: float | None, fill: dict[str, Any], *, partial: bool) -> None:
        if self.live_orders:
            event = self.trader.apply_buy_fill(
                market=market,
                qty=float(fill["qty"]),
                gross_value=float(fill.get("value") or 0.0),
                fee_paid=float(fill.get("paid_fee") or 0.0),
                strategy=self.strategy_name,
                order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
            )
        else:
            event = self.trader.enter_long(
                market=market,
                price=float(fill["price"]),
                cost=float(fill["cost"]),
                fee_rate=self.cost_model.fee_rate,
                slippage_bps=self.cost_model.slippage_bps,
                qty=float(fill["qty"]) if fill.get("qty") else None,
                strategy=self.strategy_name,
                order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
            )
        if event is None:
            return
        self.metrics["daily_buy"] = float(self.metrics.get("daily_buy") or 0.0) + float(event["cost"])
        self.last_signal_state[market] = {"sig": "BUY_PENDING" if partial else "BUY", "entry": float(self.trader.get_position(market).entry) if self.trader.get_position(market) else None}
        self._record_trade(event)
        self._notify(
            buy_filled_message(
                self._mode_label(),
                market=market,
                price=float(event["price"]),
                alloc=float(event["cost"]),
                score=score,
                partial=partial,
                qty=float(event["qty"]),
            )
        )

    def _apply_sell_fill(self, market: str, fill: dict[str, Any], reason: str, *, partial: bool) -> None:
        if self.live_orders:
            event = self.trader.apply_sell_fill(
                market=market,
                qty=float(fill["qty"]),
                gross_value=float(fill.get("value") or 0.0),
                fee_paid=float(fill.get("paid_fee") or 0.0),
                reason=reason,
                order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
            )
        else:
            event = self.trader.exit_long(
                market=market,
                price=float(fill["price"]),
                reason=reason,
                fee_rate=self.cost_model.fee_rate,
                slippage_bps=self.cost_model.slippage_bps,
                order_uuid=str(fill.get("order_uuid")) if fill.get("order_uuid") else None,
            )
        if event is None:
            return
        self.metrics["realized_pnl"] = float(self.metrics.get("realized_pnl") or 0.0) + float(event["pnl_value"])
        position = self.trader.get_position(market)
        self.last_signal_state[market] = {"sig": "SELL_PENDING" if partial and position else "SELL", "entry": position.entry if position else None}
        self._record_trade(event)
        self._notify(
            sell_filled_message(
                self._mode_label(),
                market=market,
                price=float(event["price"]),
                pnl_pct=float(event["pnl_pct"]),
                partial=partial,
                qty=float(event["qty"]),
            )
        )

    def _consume_pending_resolution(self, market: str, pending: dict[str, Any], resolution: dict[str, Any]) -> bool:
        side = str(pending.get("side") or "").lower()
        status = str(resolution.get("status") or "")
        fill = dict(resolution.get("fill") or {})
        updated_pending, delta = pending_fill_delta(pending, fill)

        if float(delta.get("qty") or 0.0) > 0:
            delta_fill = {
                "order_uuid": fill.get("order_uuid"),
                "qty": float(delta["qty"]),
                "value": float(delta["value"]),
                "paid_fee": float(delta["paid_fee"]),
                "cost": float(delta["net_value"]),
                "price": float(delta["price"] or fill.get("price") or 0.0),
            }
            if side == "bid":
                self._apply_buy_fill(market, None, delta_fill, partial=status == "pending")
            elif side == "ask":
                self._apply_sell_fill(market, delta_fill, str(pending.get("reason") or "signal"), partial=status == "pending")

        if status == "pending":
            unresolved_identifier = bool(updated_pending.get("identifier")) and not updated_pending.get("uuid")
            age_seconds = time.time() - float(updated_pending.get("submitted_at") or 0.0)
            if unresolved_identifier and age_seconds >= self.unknown_order_ttl_seconds:
                self.pending_orders.pop(market, None)
                self._notify(
                    lookup_failed_message(
                        "LIVE",
                        market=market,
                        side=side,
                        error="identifier lookup timeout",
                    )
                )
                if side == "bid":
                    self.last_signal_state[market] = {"sig": "WAIT", "entry": None}
                return True
            self.pending_orders[market] = updated_pending
            return True

        self.pending_orders.pop(market, None)
        if status == "cancelled":
            self._notify(order_cancelled_message("LIVE", market=market, side=side))
            if side == "bid":
                self.last_signal_state[market] = {"sig": "WAIT", "entry": None}
            return True
        if status == "error":
            self._notify(lookup_failed_message("LIVE", market=market, side=side, error=resolution.get("error")))
            return True
        if side == "bid":
            if float(updated_pending.get("filled_qty") or 0.0) <= 0 or float(updated_pending.get("filled_net_value") or 0.0) <= 0:
                self._notify(order_no_fill_message("LIVE", market=market, side="buy"))
                self.last_signal_state[market] = {"sig": "WAIT", "entry": None}
                return True
        elif side == "ask":
            if float(updated_pending.get("filled_qty") or 0.0) <= 0:
                self._notify(order_no_fill_message("LIVE", market=market, side="sell"))
                return True
        return True

    def _process_order_events(self) -> None:
        if not self.live_orders:
            return
        changed = False
        for event in self._drain_order_events():
            market = str(event.get("market") or "")
            if not market:
                continue
            pending_market = None
            for candidate_market, pending in list(self.pending_orders.items()):
                if str(pending.get("uuid") or "") and str(pending.get("uuid")) == str(event.get("uuid") or ""):
                    pending_market = candidate_market
                    break
                if str(pending.get("identifier") or "") and str(pending.get("identifier")) == str(event.get("identifier") or ""):
                    pending_market = candidate_market
                    break
            if not pending_market:
                continue
            pending = dict(self.pending_orders.get(pending_market) or {})
            state = str(event.get("state") or "").lower()
            if state in {"cancel", "prevented"}:
                status = "cancelled"
            elif state == "done":
                status = "filled"
            else:
                status = "pending"
            fill = extract_fill_metrics(
                event,
                side=str(pending.get("side") or "").lower(),
                fallback_price=float(pending.get("fallback_price") or 0.0),
                fallback_cost=float(pending.get("requested_cost") or 0.0) or None,
                fallback_qty=float(pending.get("requested_qty") or 0.0) or None,
            )
            changed = self._consume_pending_resolution(
                pending_market,
                pending,
                {"status": status, "fill": fill, "order": event},
            ) or changed
        if changed:
            self._save_state()

    def _reconcile_pending_orders(self) -> None:
        if not self.live_orders or not self.pending_orders:
            return
        changed = False
        for market, pending in list(self.pending_orders.items()):
            if not pending.get("uuid") and not pending.get("identifier"):
                self.pending_orders.pop(market, None)
                changed = True
                continue
            side = str(pending.get("side") or "").lower()
            resolution = resolve_submitted_order(
                self.api,
                {
                    "uuid": pending.get("uuid"),
                    "identifier": pending.get("identifier"),
                },
                live_orders=True,
                side=side,
                fallback_price=float(pending.get("fallback_price") or 0.0),
                fallback_cost=float(pending.get("requested_cost") or 0.0) or None,
                fallback_qty=float(pending.get("requested_qty") or 0.0) or None,
                timeout_seconds=min(self.reconcile_timeout_seconds, 1.2),
                poll_interval=0.2,
            )
            changed = self._consume_pending_resolution(market, dict(pending), resolution) or changed
        if changed:
            self._save_state()

    def _handle_entry(self, market: str, close_price: float, score: float) -> None:
        self._refresh_kill_switch()
        if self._kill_switch_blocks_entry(market):
            return
        if len(self.trader.positions) >= self.max_open:
            self._notify(blocked_max_open_message(self._mode_label(), market=market, max_open=self.max_open))
            return
        decision = evaluate_entry(
            config=self.risk_config,
            metrics=self.metrics,
            positions=self.trader.to_state(),
            price_map=self.market_prices,
            market=market,
            day_start_equity=float(self.metrics.get("day_start_equity") or 0.0),
            fee_rate=self.cost_model.fee_rate,
            slippage_bps=self.cost_model.slippage_bps,
        )
        if not decision.allowed:
            self._notify(blocked_risk_message(self._mode_label(), market=market, reason=decision.blocked_reason, price=close_price))
            return
        order_identifier = build_client_order_identifier(market, "bid", strategy_name=self.strategy_name)
        order_result = self._place_order(
            market=market,
            side="bid",
            ord_type="price",
            price=f"{int(decision.trade_cost)}",
            identifier=order_identifier,
        )
        resolution = resolve_submitted_order(
            self.api,
            order_result,
            live_orders=self.live_orders,
            side="bid",
            fallback_price=close_price,
            fallback_cost=decision.trade_cost,
            timeout_seconds=self.reconcile_timeout_seconds,
        )
        if resolution.get("status") == "error":
            self._notify(order_failed_message(self._mode_label(), market=market, side="buy", error=resolution.get("error")))
            return
        if resolution.get("status") == "cancelled":
            self._notify(order_cancelled_message(self._mode_label(), market=market, side="buy"))
            return
        fill = dict(resolution.get("fill") or {})
        if resolution.get("status") == "pending" and self.live_orders:
            if float(fill.get("qty") or 0.0) > 0:
                self._apply_buy_fill(market, score, fill, partial=True)
            self.pending_orders[market] = build_pending_order(
                market=market,
                side="bid",
                strategy=self.strategy_name,
                fallback_price=close_price,
                order_result=resolution.get("order"),
                requested_cost=decision.trade_cost,
                fill=fill,
            )
            current_position = self.trader.get_position(market)
            self.last_signal_state[market] = {"sig": "BUY_PENDING", "entry": current_position.entry if current_position else None}
            self._notify(order_pending_message("LIVE", market=market, side="buy"))
            self._save_state()
            return
        if float(fill.get("qty") or 0.0) <= 0 or float(fill.get("cost") or 0.0) <= 0:
            self._notify(order_no_fill_message(self._mode_label(), market=market, side="buy"))
            return
        self._apply_buy_fill(market, score, fill, partial=False)

    def _handle_exit(self, market: str, close_price: float) -> None:
        position = self.trader.get_position(market)
        if position is None:
            return
        order_result = self._place_order(
            market=market,
            side="ask",
            ord_type="market",
            volume=f"{position.qty:.8f}",
            identifier=build_client_order_identifier(market, "ask", strategy_name=self.strategy_name),
        )
        resolution = resolve_submitted_order(
            self.api,
            order_result,
            live_orders=self.live_orders,
            side="ask",
            fallback_price=close_price,
            fallback_qty=position.qty,
            timeout_seconds=self.reconcile_timeout_seconds,
        )
        if resolution.get("status") == "error":
            self._notify(order_failed_message(self._mode_label(), market=market, side="sell", error=resolution.get("error")))
            return
        if resolution.get("status") == "cancelled":
            self._notify(order_cancelled_message(self._mode_label(), market=market, side="sell"))
            return
        fill = dict(resolution.get("fill") or {})
        if resolution.get("status") == "pending" and self.live_orders:
            if float(fill.get("qty") or 0.0) > 0:
                self._apply_sell_fill(market, fill, "signal", partial=True)
            self.pending_orders[market] = build_pending_order(
                market=market,
                side="ask",
                strategy=self.strategy_name,
                fallback_price=close_price,
                order_result=resolution.get("order"),
                requested_qty=position.qty,
                reason="signal",
                fill=fill,
            )
            current_position = self.trader.get_position(market)
            self.last_signal_state[market] = {"sig": "SELL_PENDING", "entry": current_position.entry if current_position else position.entry}
            self._notify(order_pending_message("LIVE", market=market, side="sell"))
            self._save_state()
            return
        if float(fill.get("qty") or 0.0) <= 0:
            self._notify(order_no_fill_message(self._mode_label(), market=market, side="sell"))
            return
        self._apply_sell_fill(market, fill, "signal", partial=False)

    def process_market(self, market: str) -> dict[str, Any] | None:
        self._refresh_metrics()
        frame = self._build_frame(market)
        if frame.empty:
            return None
        last_row = frame.iloc[-1]
        signal = _current_signal(last_row)
        previous_signal = (self.last_signal_state.get(market) or {}).get("sig", "WAIT")
        close_price = float(last_row["close"])
        score = float(last_row.get("strategy_score", 0.0))
        self.market_prices[market] = close_price
        pending = self.pending_orders.get(market)

        if pending:
            pending_side = str(pending.get("side") or "").lower()
            pending_signal = "BUY_PENDING" if pending_side == "bid" else "SELL_PENDING"
            updated_position = self.trader.get_position(market)
            self.last_signal_state[market] = {
                "sig": pending_signal,
                "entry": updated_position.entry if updated_position else None,
            }
            bt = backtest_signal_frame(frame, fee=self.cost_model.fee_rate, slippage_bps=self.cost_model.slippage_bps)
            return {
                "market": market,
                "price": close_price,
                "score": score,
                "trades": int(bt["trades"]),
                "return_pct": float(bt["total_return_pct"]),
                "win_rate_pct": float(bt["win_rate_pct"]),
                "max_drawdown_pct": float(bt["max_drawdown_pct"]),
                "last_signal": pending_signal,
                "position": "OPEN" if updated_position else "PENDING",
            }

        if signal == "BUY" and previous_signal != "BUY" and not self.trader.has_position(market):
            self._handle_entry(market, close_price, score)
        elif signal == "SELL" and previous_signal == "BUY" and self.trader.has_position(market):
            self._handle_exit(market, close_price)

        updated_position = self.trader.get_position(market)
        updated_pending = self.pending_orders.get(market)
        if updated_pending:
            updated_side = str(updated_pending.get("side") or "").lower()
            signal_state = "BUY_PENDING" if updated_side == "bid" else "SELL_PENDING"
        else:
            signal_state = "BUY" if updated_position else signal
        self.last_signal_state[market] = {
            "sig": signal_state,
            "entry": updated_position.entry if updated_position else None,
        }
        bt = backtest_signal_frame(frame, fee=self.cost_model.fee_rate, slippage_bps=self.cost_model.slippage_bps)
        return {
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

    def run_cycle(self, markets: list[str]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        trades_before = len(self.trade_log)
        self._ensure_my_order_stream()
        self._refresh_metrics()
        self._refresh_kill_switch()
        self._process_order_events()
        self._reconcile_pending_orders()
        self._sync_with_exchange()
        for market in markets:
            try:
                row = self.process_market(market)
                if row:
                    rows.append(row)
            except Exception as exc:
                print(f"process error {market}: {exc}")
            if self.per_request_sleep > 0:
                time.sleep(self.per_request_sleep)
        self.metrics["unrealized_pnl"] = total_unrealized_pnl(
            self.trader.to_state(),
            self.market_prices,
            fee_rate=self.cost_model.fee_rate,
            slippage_bps=self.cost_model.slippage_bps,
        )
        self.metrics["total_pnl"] = float(self.metrics.get("realized_pnl") or 0.0) + float(self.metrics["unrealized_pnl"])
        table = pd.DataFrame(rows)
        if not table.empty:
            table = table.sort_values(["score", "return_pct"], ascending=False)
        self._save_state()
        return {
            "processed": len(markets),
            "open": len(self.trader.positions),
            "trades": len(self.trade_log),
            "new_trades": len(self.trade_log) - trades_before,
            "total_pnl": float(self.metrics["total_pnl"]),
            "table": table,
            "pending_orders": len(self.pending_orders),
            "kill_switch": bool(self.kill_switch_state.get("enabled")),
        }

    def loop(self, markets: list[str], *, loop_seconds: int = 120, cycles: int = 0) -> None:
        self._notify(
            start_message(
                self._mode_label(),
                strategy_name=strategy_label(self.strategy_name),
                interval=self.interval,
                markets=len(markets),
                max_open=self.max_open,
            )
        )
        cycle_index = 0
        try:
            while True:
                cycle_index += 1
                summary = self.run_cycle(markets)
                print(
                    f"[cycle {cycle_index}] processed={summary['processed']} open={summary['open']} "
                    f"pending={summary['pending_orders']} kill={'ON' if summary['kill_switch'] else 'OFF'} "
                    f"new_trades={summary['new_trades']} total_pnl={summary['total_pnl']:.0f}"
                )
                if cycles and cycle_index >= cycles:
                    break
                time.sleep(loop_seconds)
        finally:
            self._stop_my_order_stream()


def parse_args():
    parser = argparse.ArgumentParser(description="Strategy monitor for Upbit")
    parser.add_argument(
        "--strategy",
        choices=["research_trend", "relative_strength_rotation", "flux_trend", "flux_ema_filter"],
        default="research_trend",
    )
    parser.add_argument("--interval", default="minute30")
    parser.add_argument("--count", type=int, default=240, help="Candles fetched per market")
    parser.add_argument("--markets", type=int, default=20, help="Top N markets by 24h turnover")
    parser.add_argument("--base", default="KRW")
    parser.add_argument("--loop-seconds", type=int, default=120)
    parser.add_argument("--cycles", type=int, default=0, help="0 means run forever")
    parser.add_argument("--live-orders", action="store_true", default=False, help="Send real orders only when UPBIT_LIVE=1")
    parser.add_argument("--max-open", type=int, default=5)
    parser.add_argument("--min-fetch-seconds", type=float, default=20.0)
    parser.add_argument("--per-request-sleep", type=float, default=0.12)
    parser.add_argument("--state-name", default="mr-worker")
    parser.add_argument("--reconcile-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--kill-switch-name", default="trade-kill-switch")
    parser.add_argument("--no-exclude-stables", action="store_true", help="Keep stablecoin pairs in the universe")
    parser.add_argument("--fee", type=float, default=0.0005)

    parser.add_argument("--max-trade-krw", type=float, default=50000.0)
    parser.add_argument("--krw-per-trade", dest="max_trade_krw_legacy", type=float, default=None, help="Deprecated alias for --max-trade-krw")
    parser.add_argument("--max-trade-pct", type=float, default=2.0)
    parser.add_argument("--per-asset-max-pct", type=float, default=10.0)
    parser.add_argument("--daily-buy-limit", type=float, default=200000.0)
    parser.add_argument("--daily-loss-limit-krw", type=float, default=30000.0)
    parser.add_argument("--daily-loss-limit-pct", type=float, default=3.0)
    parser.add_argument("--include-unrealized-loss", action="store_true", default=False)

    parser.add_argument("--fast-ema", type=int, default=21)
    parser.add_argument("--slow-ema", type=int, default=55)
    parser.add_argument("--breakout-window", type=int, default=20)
    parser.add_argument("--exit-window", type=int, default=10)
    parser.add_argument("--atr-window", type=int, default=14)
    parser.add_argument("--atr-mult", type=float, default=2.5)
    parser.add_argument("--adx-window", type=int, default=14)
    parser.add_argument("--adx-threshold", type=float, default=18.0)
    parser.add_argument("--momentum-window", type=int, default=20)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--volume-threshold", type=float, default=0.9)
    parser.add_argument("--rs-short-window", type=int, default=10)
    parser.add_argument("--rs-mid-window", type=int, default=30)
    parser.add_argument("--rs-long-window", type=int, default=90)
    parser.add_argument("--trend-ema-window", type=int, default=55)
    parser.add_argument("--entry-score", type=float, default=8.0)
    parser.add_argument("--exit-score", type=float, default=2.0)

    parser.add_argument("--ltf-len", type=int, default=20)
    parser.add_argument("--ltf-mult", type=float, default=2.0)
    parser.add_argument("--htf-len", type=int, default=20)
    parser.add_argument("--htf-mult", type=float, default=2.25)
    parser.add_argument("--htf-rule", default="60T")
    parser.add_argument("--sensitivity", type=int, default=3)
    parser.add_argument("--atr-period", type=int, default=2)
    parser.add_argument("--trend-ema-length", type=int, default=240)
    parser.add_argument("--confirm-window", type=int, default=8)
    parser.add_argument("--use-heikin-ashi", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.strategy == "flux_trend" and flux_indicator is None:
        raise SystemExit("Flux strategy requested but flux indicator could not be imported.")
    if args.strategy == "flux_ema_filter" and flux_indicator_with_ema is None:
        raise SystemExit("Flux EMA strategy requested but the extended flux indicator could not be imported.")

    api = UpbitAPI(
        access_key=os.getenv("UPBIT_ACCESS_KEY"),
        secret_key=os.getenv("UPBIT_SECRET_KEY"),
    )
    markets = fetch_top_markets(
        api,
        base=args.base,
        limit=args.markets,
        exclude_stables=not args.no_exclude_stables,
    )
    monitor = MRMonitor(
        api,
        interval=args.interval,
        candles_count=args.count,
        strategy_name=args.strategy,
        strategy_params=_strategy_params_from_args(args),
        risk_limits=_risk_limits_from_args(args),
        fee=args.fee,
        live_orders=args.live_orders,
        max_open=args.max_open,
        min_fetch_seconds=args.min_fetch_seconds,
        per_request_sleep=args.per_request_sleep,
        state_name=args.state_name,
        reconcile_timeout_seconds=args.reconcile_timeout_seconds,
        cost_model=cost_model_from_values(fee_rate=args.fee, slippage_bps=args.slippage_bps),
        kill_switch_name=args.kill_switch_name,
    )
    if args.live_orders and os.getenv("UPBIT_LIVE") != "1":
        print("[WARN] --live-orders was set but UPBIT_LIVE is not 1, so the monitor stays in SIM mode.")
    monitor.loop(markets, loop_seconds=args.loop_seconds, cycles=args.cycles)


if __name__ == "__main__":
    main()

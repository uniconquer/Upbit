"""Shared-capital portfolio tournament helpers for multi-market backtests."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

try:
    from paper_trader import PaperTrader
    from strategy_engine import build_strategy_frame, strategy_label
except ImportError:
    from src.paper_trader import PaperTrader
    from src.strategy_engine import build_strategy_frame, strategy_label


def _resolve_timeline(frames_by_market: Mapping[str, pd.DataFrame]) -> pd.Index:
    stamps: set[pd.Timestamp] = set()
    for frame in frames_by_market.values():
        stamps.update(pd.Index(frame.index).tolist())
    return pd.Index(sorted(stamps))


def _align_market_frame(
    frame: pd.DataFrame,
    timeline: pd.Index,
    *,
    entry_col: str,
    exit_col: str,
) -> pd.DataFrame:
    aligned = frame.sort_index().reindex(timeline)
    aligned["close"] = pd.to_numeric(aligned["close"], errors="coerce").ffill()
    entry_values = aligned[entry_col] if entry_col in aligned.columns else pd.Series(False, index=timeline, dtype=bool)
    exit_values = aligned[exit_col] if exit_col in aligned.columns else pd.Series(False, index=timeline, dtype=bool)
    aligned[entry_col] = pd.Series(entry_values, index=timeline).astype("boolean").fillna(False).astype(bool)
    aligned[exit_col] = pd.Series(exit_values, index=timeline).astype("boolean").fillna(False).astype(bool)
    aligned["strategy_score"] = pd.to_numeric(aligned.get("strategy_score", 0.0), errors="coerce").fillna(0.0)
    return aligned


def _equity_value(
    *,
    cash: float,
    trader: PaperTrader,
    price_map: Mapping[str, float],
) -> float:
    value = float(cash)
    for market, position in trader.positions.items():
        value += float(price_map.get(market, position.entry)) * float(position.qty)
    return value


def backtest_portfolio_signal_frames(
    frames_by_market: Mapping[str, pd.DataFrame],
    *,
    strategy_name: str = "portfolio",
    initial_cash: float = 10000.0,
    max_positions: int = 3,
    allocation_pct: float = 1.0,
    min_trade_krw: float = 0.0,
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
    entry_col: str = "buy_signal",
    exit_col: str = "sell_signal",
    liquidate_at_end: bool = True,
) -> dict[str, object]:
    timeline = _resolve_timeline(frames_by_market)
    if timeline.empty:
        return {
            "initial_cash": float(initial_cash),
            "final_equity": float(initial_cash),
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "open_positions": 0,
            "cash": float(initial_cash),
            "equity": pd.Series(dtype=float),
            "trade_log": [],
        }

    resolved_cash = max(float(initial_cash), 0.0)
    resolved_max_positions = max(int(max_positions), 1)
    resolved_allocation_pct = min(max(float(allocation_pct), 0.0), 1.0)
    resolved_min_trade = max(float(min_trade_krw), 0.0)

    aligned_frames = {
        market: _align_market_frame(frame, timeline, entry_col=entry_col, exit_col=exit_col)
        for market, frame in frames_by_market.items()
        if not frame.empty and {"close", entry_col, exit_col}.issubset(frame.columns)
    }
    if not aligned_frames:
        return {
            "initial_cash": resolved_cash,
            "final_equity": resolved_cash,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "open_positions": 0,
            "cash": resolved_cash,
            "equity": pd.Series(dtype=float),
            "trade_log": [],
        }

    trader = PaperTrader()
    price_map: dict[str, float] = {}
    equity_points: list[tuple[pd.Timestamp, float]] = []
    trade_log: list[dict[str, Any]] = []
    cash = resolved_cash
    peak = resolved_cash
    max_drawdown = 0.0

    for ts in timeline:
        buy_candidates: list[tuple[float, str, float]] = []
        for market, frame in aligned_frames.items():
            row = frame.loc[ts]
            close = row.get("close")
            if close is not None and pd.notna(close):
                price_map[market] = float(close)

            if trader.has_position(market) and bool(row.get(exit_col)) and market in price_map:
                event = trader.exit_long(
                    market=market,
                    price=price_map[market],
                    reason="signal",
                    fee_rate=fee,
                    slippage_bps=slippage_bps,
                    timestamp=ts.value / 1_000_000_000,
                )
                if event is not None:
                    cash += float(event["net_proceeds"])
                    trade_log.append(event)
                continue

            if trader.has_position(market):
                continue
            if bool(row.get(entry_col)) and market in price_map:
                score = float(row.get("strategy_score", 0.0) or 0.0)
                buy_candidates.append((score, market, price_map[market]))

        available_slots = max(resolved_max_positions - len(trader.positions), 0)
        ranked_candidates = sorted(buy_candidates, key=lambda item: (-item[0], item[1]))
        for index, (score, market, price) in enumerate(ranked_candidates):
            if available_slots <= 0:
                break
            remaining_candidates = max(len(ranked_candidates) - index, 1)
            budget_divisor = max(min(available_slots, remaining_candidates), 1)
            budget = min(cash, (cash / budget_divisor) * resolved_allocation_pct) if available_slots else 0.0
            if budget <= 0 or budget < resolved_min_trade:
                break
            event = trader.enter_long(
                market=market,
                price=price,
                cost=budget,
                strategy=strategy_name,
                fee_rate=fee,
                slippage_bps=slippage_bps,
                timestamp=ts.value / 1_000_000_000,
            )
            cash -= budget
            trade_log.append(event)
            available_slots -= 1

        equity_value = _equity_value(cash=cash, trader=trader, price_map=price_map)
        equity_points.append((ts, equity_value))
        peak = max(peak, equity_value)
        drawdown = ((equity_value / peak) - 1.0) * 100.0 if peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)

    if liquidate_at_end and equity_points:
        final_ts = equity_points[-1][0]
        for market in list(trader.positions.keys()):
            if market not in price_map:
                continue
            event = trader.exit_long(
                market=market,
                price=price_map[market],
                reason="finalize",
                fee_rate=fee,
                slippage_bps=slippage_bps,
                timestamp=final_ts.value / 1_000_000_000,
            )
            if event is not None:
                cash += float(event["net_proceeds"])
                trade_log.append(event)

        final_equity = cash
        equity_points[-1] = (final_ts, final_equity)
        peak = max(peak, final_equity)
        drawdown = ((final_equity / peak) - 1.0) * 100.0 if peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)
    else:
        final_equity = _equity_value(cash=cash, trader=trader, price_map=price_map)

    sell_trades = [event for event in trade_log if event.get("side") == "SELL"]
    win_rate_pct = (
        sum(1 for event in sell_trades if float(event.get("pnl_value") or 0.0) > 0.0) / len(sell_trades) * 100.0
        if sell_trades
        else 0.0
    )
    total_return_pct = ((final_equity / resolved_cash) - 1.0) * 100.0 if resolved_cash else 0.0
    equity_series = pd.Series([value for _, value in equity_points], index=[ts for ts, _ in equity_points], dtype=float)

    return {
        "initial_cash": resolved_cash,
        "final_equity": float(final_equity),
        "total_return_pct": float(total_return_pct),
        "max_drawdown_pct": float(max_drawdown),
        "win_rate_pct": float(win_rate_pct),
        "trades": len(sell_trades),
        "buy_trades": sum(1 for event in trade_log if event.get("side") == "BUY"),
        "sell_trades": len(sell_trades),
        "open_positions": len(trader.positions),
        "cash": float(cash),
        "equity": equity_series,
        "trade_log": trade_log,
    }


def portfolio_backtest(
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    strategy_name: str,
    params: Mapping[str, Any] | None = None,
    initial_cash: float = 10000.0,
    max_positions: int = 3,
    allocation_pct: float = 1.0,
    min_trade_krw: float = 0.0,
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
    liquidate_at_end: bool = True,
    flux_indicator=None,
    flux_indicator_with_ema=None,
) -> dict[str, object]:
    frames_by_market = {
        market: build_strategy_frame(
            raw,
            strategy_name=strategy_name,
            params=dict(params or {}),
            flux_indicator=flux_indicator,
            flux_indicator_with_ema=flux_indicator_with_ema,
        )
        for market, raw in raw_by_market.items()
        if not raw.empty
    }
    return backtest_portfolio_signal_frames(
        frames_by_market,
        strategy_name=strategy_name,
        initial_cash=initial_cash,
        max_positions=max_positions,
        allocation_pct=allocation_pct,
        min_trade_krw=min_trade_krw,
        fee=fee,
        slippage_bps=slippage_bps,
        liquidate_at_end=liquidate_at_end,
    )


def compare_portfolio_strategies(
    raw_by_market: Mapping[str, pd.DataFrame],
    *,
    strategies: list[dict[str, Any]],
    initial_cash: float = 10000.0,
    max_positions: int = 3,
    allocation_pct: float = 1.0,
    min_trade_krw: float = 0.0,
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
    liquidate_at_end: bool = True,
    flux_indicator=None,
    flux_indicator_with_ema=None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in strategies:
        strategy_name = str(spec.get("strategy_name") or "")
        if not strategy_name:
            continue
        params = dict(spec.get("params") or {})
        result = portfolio_backtest(
            raw_by_market,
            strategy_name=strategy_name,
            params=params,
            initial_cash=initial_cash,
            max_positions=max_positions,
            allocation_pct=allocation_pct,
            min_trade_krw=min_trade_krw,
            fee=fee,
            slippage_bps=slippage_bps,
            liquidate_at_end=liquidate_at_end,
            flux_indicator=flux_indicator,
            flux_indicator_with_ema=flux_indicator_with_ema,
        )
        rows.append(
            {
                "strategy_name": strategy_name,
                "strategy_label": strategy_label(strategy_name),
                "params": params,
                "final_equity": float(result["final_equity"]),
                "return_pct": float(result["total_return_pct"]),
                "max_drawdown_pct": float(result["max_drawdown_pct"]),
                "win_rate_pct": float(result["win_rate_pct"]),
                "trades": int(result["trades"]),
                "buy_trades": int(result["buy_trades"]),
                "sell_trades": int(result["sell_trades"]),
                "open_positions": int(result["open_positions"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    results = pd.DataFrame(rows)
    return results.sort_values(
        ["final_equity", "max_drawdown_pct", "win_rate_pct", "sell_trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

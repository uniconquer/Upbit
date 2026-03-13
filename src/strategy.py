"""Strategy helpers, signal generators, and simple backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from trading_costs import cost_model_from_values
except ImportError:
    from src.trading_costs import cost_model_from_values


@dataclass(slots=True)
class Signal:
    index: int
    side: str
    price: float


def _validate_length(prices: Sequence[float], min_len: int) -> bool:
    return len(prices) >= min_len


def _as_float_series(prices: Sequence[float] | pd.Series) -> pd.Series:
    if isinstance(prices, pd.Series):
        return prices.astype(float)
    return pd.Series(prices, dtype=float)


def sma(prices: Sequence[float], window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window > 0 required")
    return _as_float_series(prices).rolling(window).mean()


def ema(prices: Sequence[float], window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window > 0 required")
    return _as_float_series(prices).ewm(span=window, adjust=False).mean()


def _crossover_signals(
    fast: pd.Series,
    slow: pd.Series,
    base_prices: Sequence[float],
    start_index: int,
) -> list[Signal]:
    signals: list[Signal] = []
    prev_state: bool | None = None

    for idx in range(start_index, len(fast)):
        fast_value = fast.iloc[idx]
        slow_value = slow.iloc[idx]
        if pd.isna(fast_value) or pd.isna(slow_value):
            continue

        current_state = bool(fast_value > slow_value)
        if prev_state is None:
            prev_state = current_state
            continue

        if current_state and not prev_state:
            signals.append(Signal(index=idx, side="buy", price=float(base_prices[idx])))
        elif (not current_state) and prev_state:
            signals.append(Signal(index=idx, side="sell", price=float(base_prices[idx])))
        prev_state = current_state

    return signals


def sma_cross_signals(prices: Sequence[float], short: int = 5, long: int = 20) -> list[Signal]:
    if long <= short:
        raise ValueError("long must be > short")
    if not _validate_length(prices, long + 1):
        return []

    fast = sma(prices, short)
    slow = sma(prices, long)
    return _crossover_signals(fast, slow, prices, start_index=long)


def ema_cross_signals(prices: Sequence[float], short: int = 5, long: int = 20) -> list[Signal]:
    if long <= short:
        raise ValueError("long must be > short")
    if not _validate_length(prices, long + 1):
        return []

    fast = ema(prices, short)
    slow = ema(prices, long)
    return _crossover_signals(fast, slow, prices, start_index=long)


def momentum_signals(prices: Sequence[float], long: int = 50) -> list[Signal]:
    if not _validate_length(prices, long + 2):
        return []

    baseline = sma(prices, long)
    fast = pd.Series(prices, dtype=float)
    return _crossover_signals(fast, baseline, prices, start_index=long)


def rsi_signals(
    prices: Sequence[float],
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> list[Signal]:
    if period <= 1 or oversold >= overbought:
        return []
    if not _validate_length(prices, period + 2):
        return []

    series = pd.Series(prices, dtype=float)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    signals: list[Signal] = []
    previous_rsi: float | None = None
    for idx in range(period + 1, len(rsi)):
        current_rsi = rsi.iloc[idx]
        if pd.isna(current_rsi):
            continue
        current_value = float(current_rsi)
        if previous_rsi is not None:
            if previous_rsi < oversold <= current_value:
                signals.append(Signal(index=idx, side="buy", price=float(series.iloc[idx])))
            elif previous_rsi > overbought >= current_value:
                signals.append(Signal(index=idx, side="sell", price=float(series.iloc[idx])))
        previous_rsi = current_value

    return signals


def average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window > 0 required")
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def average_directional_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window > 0 required")

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )

    atr = average_true_range(df, window).replace(0.0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean().fillna(0.0)


def build_research_trend_signals(
    raw: pd.DataFrame,
    *,
    fast_ema: int = 21,
    slow_ema: int = 55,
    breakout_window: int = 20,
    exit_window: int = 10,
    atr_window: int = 14,
    atr_mult: float = 2.5,
    adx_window: int = 14,
    adx_threshold: float = 18.0,
    momentum_window: int = 20,
    volume_window: int = 20,
    volume_threshold: float = 0.9,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        raise ValueError(f"raw data missing columns: {missing}")

    df = raw.copy()
    close = df["close"].astype(float)
    df["ema_fast"] = ema(close, fast_ema)
    df["ema_slow"] = ema(close, slow_ema)
    df["atr"] = average_true_range(df, atr_window)
    df["adx"] = average_directional_index(df, adx_window)
    df["breakout_high"] = df["high"].rolling(breakout_window).max().shift(1)
    df["breakdown_low"] = df["low"].rolling(exit_window).min().shift(1)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)
    df["momentum"] = close.pct_change(momentum_window)
    df["volatility"] = df["atr"] / close.replace(0.0, np.nan)
    df["trend_strength"] = (df["ema_fast"] / df["ema_slow"]) - 1.0
    df["atr_stop"] = df["ema_slow"] - (atr_mult * df["atr"])
    df["strategy_score"] = (
        df["momentum"].fillna(0.0) * 100.0
        + df["trend_strength"].fillna(0.0) * 80.0
        + (df["adx"].fillna(0.0) - adx_threshold) / 10.0
        - df["volatility"].fillna(0.0) * 50.0
    )

    regime_ok = (close > df["ema_fast"]) & (df["ema_fast"] > df["ema_slow"]) & (df["adx"] >= adx_threshold)
    breakout = close > df["breakout_high"]
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold
    raw_buy = regime_ok & breakout & volume_ok

    exit_break = close < df["breakdown_low"]
    trend_fail = (close < df["ema_fast"]) | (df["adx"] < max(10.0, adx_threshold - 6.0))
    raw_sell = exit_break | (close < df["atr_stop"]) | trend_fail

    df["buy_signal"] = raw_buy & (~raw_buy.shift(1, fill_value=False))
    df["sell_signal"] = raw_sell & (~raw_sell.shift(1, fill_value=False))
    return df


def backtest_signal_frame(
    df: pd.DataFrame,
    *,
    entry_col: str = "buy_signal",
    exit_col: str = "sell_signal",
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
) -> dict[str, object]:
    empty_result = {
        "trades": 0,
        "total_return_pct": 0.0,
        "win_rate_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "equity": None,
    }
    if df.empty or "close" not in df.columns or entry_col not in df.columns or exit_col not in df.columns:
        return empty_result

    in_position = False
    entry_price = 0.0
    equity_curve: list[float] = []
    equity_value = 1.0
    trades: list[float] = []
    peak = 1.0
    max_drawdown = 0.0
    cost_model = cost_model_from_values(fee_rate=fee, slippage_bps=slippage_bps)

    for _, row in df.iterrows():
        price = row.get("close")
        if price is None or pd.isna(price):
            continue

        if (not in_position) and bool(row.get(entry_col)):
            in_position = True
            entry_price = cost_model.buy_price(float(price))

        if in_position and bool(row.get(exit_col)):
            exit_price = cost_model.sell_price(float(price))
            gross = exit_price / entry_price if entry_price else 1.0
            net = gross * (1 - fee) * (1 - fee)
            equity_value *= net
            trades.append((net - 1) * 100)
            in_position = False

        equity_curve.append(equity_value)
        peak = max(peak, equity_value)
        drawdown = ((equity_value / peak) - 1) * 100 if peak else 0.0
        max_drawdown = min(max_drawdown, drawdown)

    if not equity_curve:
        return empty_result

    total_return_pct = (equity_curve[-1] - 1) * 100
    win_rate_pct = (sum(1 for trade in trades if trade > 0) / len(trades) * 100) if trades else 0.0
    return {
        "trades": len(trades),
        "total_return_pct": total_return_pct,
        "win_rate_pct": win_rate_pct,
        "max_drawdown_pct": max_drawdown,
        "equity": pd.Series(equity_curve, index=df.index[: len(equity_curve)]),
    }


def extract_backtest_trade_events(
    df: pd.DataFrame,
    *,
    entry_col: str = "buy_signal",
    exit_col: str = "sell_signal",
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    if df.empty or "close" not in df.columns or entry_col not in df.columns or exit_col not in df.columns:
        return events

    in_position = False
    for timestamp, row in df.iterrows():
        price = row.get("close")
        if price is None or pd.isna(price):
            continue
        if (not in_position) and bool(row.get(entry_col)):
            in_position = True
            events.append({"ts": timestamp, "side": "BUY", "price": float(price)})
        elif in_position and bool(row.get(exit_col)):
            in_position = False
            events.append({"ts": timestamp, "side": "SELL", "price": float(price)})
    return events


SMA_HELP_MD = (
    "### SMA Cross\n\n"
    "Short moving average crossing the long moving average.\n"
    "- Cross up: buy\n"
    "- Cross down: sell\n"
)


STRATEGY_HELP = {
    "sma_cross": "Simple moving-average crossover",
    "ema_cross": "Exponential moving-average crossover",
    "momentum": "Price versus longer SMA crossover",
    "rsi": "RSI oversold/overbought crossback",
    "research_trend": "EMA trend + breakout + ADX + ATR exit",
}


__all__ = [
    "Signal",
    "SMA_HELP_MD",
    "STRATEGY_HELP",
    "average_directional_index",
    "average_true_range",
    "backtest_signal_frame",
    "build_research_trend_signals",
    "ema",
    "ema_cross_signals",
    "extract_backtest_trade_events",
    "momentum_signals",
    "rsi_signals",
    "sma",
    "sma_cross_signals",
]

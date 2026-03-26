"""Strategy helpers, signal generators, and simple backtest utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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


def relative_strength_index(prices: Sequence[float] | pd.Series, period: int = 14) -> pd.Series:
    if period <= 1:
        raise ValueError("period > 1 required")
    series = _as_float_series(prices)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(
    prices: Sequence[float] | pd.Series,
    window: int = 20,
    std_mult: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if window <= 1:
        raise ValueError("window > 1 required")
    series = _as_float_series(prices)
    basis = series.rolling(window).mean()
    deviation = series.rolling(window).std(ddof=0)
    upper = basis + (deviation * std_mult)
    lower = basis - (deviation * std_mult)
    return basis, upper, lower


def moving_average_convergence_divergence(
    prices: Sequence[float] | pd.Series,
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    if min(fast, slow, signal) <= 0 or fast >= slow:
        raise ValueError("require 0 < fast < slow and signal > 0")
    series = _as_float_series(prices)
    fast_line = ema(series, fast)
    slow_line = ema(series, slow)
    macd_line = fast_line - slow_line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


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


def build_relative_strength_rotation_signals(
    raw: pd.DataFrame,
    *,
    rs_short_window: int = 10,
    rs_mid_window: int = 30,
    rs_long_window: int = 90,
    trend_ema_window: int = 55,
    breakout_window: int = 20,
    atr_window: int = 14,
    atr_mult: float = 2.2,
    volume_window: int = 20,
    volume_threshold: float = 0.9,
    entry_score: float = 8.0,
    exit_score: float = 2.0,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        raise ValueError(f"raw data missing columns: {missing}")

    df = raw.copy()
    close = df["close"].astype(float)
    df["rs_short"] = close.pct_change(rs_short_window) * 100.0
    df["rs_mid"] = close.pct_change(rs_mid_window) * 100.0
    df["rs_long"] = close.pct_change(rs_long_window) * 100.0
    df["trend_ema"] = ema(close, trend_ema_window)
    df["atr"] = average_true_range(df, atr_window)
    df["breakout_high"] = df["high"].rolling(breakout_window).max().shift(1)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)
    df["volatility"] = close.pct_change().rolling(max(5, rs_short_window)).std() * 100.0

    trend_gap = ((close / df["trend_ema"]) - 1.0) * 100.0
    ema_slope = df["trend_ema"].pct_change(max(2, rs_short_window // 2)) * 100.0
    df["atr_stop"] = df["trend_ema"] - (atr_mult * df["atr"])
    df["strategy_score"] = (
        df["rs_short"].fillna(0.0) * 0.45
        + df["rs_mid"].fillna(0.0) * 0.35
        + df["rs_long"].fillna(0.0) * 0.20
        + trend_gap.fillna(0.0) * 1.40
        + ema_slope.fillna(0.0) * 0.80
        + (df["volume_ratio"].fillna(1.0) - 1.0) * 6.0
        - df["volatility"].fillna(0.0) * 6.0
    )

    slope_window = max(2, rs_short_window // 2)
    regime_ok = (close > df["trend_ema"]) & (df["trend_ema"] > df["trend_ema"].shift(slope_window))
    breakout = close > df["breakout_high"]
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold
    score_ok = df["strategy_score"] >= entry_score
    score_rising = df["strategy_score"] > df["strategy_score"].shift(1)
    raw_buy = regime_ok & breakout & volume_ok & score_ok & score_rising

    score_fade = df["strategy_score"] <= exit_score
    trend_fail = close < df["trend_ema"]
    raw_sell = score_fade | trend_fail | (close < df["atr_stop"])

    df["buy_signal"] = raw_buy & (~raw_buy.shift(1, fill_value=False))
    df["sell_signal"] = raw_sell & (~raw_sell.shift(1, fill_value=False))
    return df


def _infer_price_tick(raw: pd.DataFrame) -> float:
    values = pd.concat(
        [
            raw["open"].astype(float),
            raw["high"].astype(float),
            raw["low"].astype(float),
            raw["close"].astype(float),
        ],
        axis=0,
    ).dropna()
    if values.empty:
        return 0.0
    unique_values = pd.Series(np.sort(values.unique()), dtype=float)
    diffs = unique_values.diff().abs()
    positive = diffs[(diffs > 0) & np.isfinite(diffs)]
    if positive.empty:
        return 0.0
    try:
        return float(positive.quantile(0.05))
    except Exception:
        return float(positive.min())


def build_rsi_bb_double_bottom_signals(
    raw: pd.DataFrame,
    *,
    rsi_len: int = 14,
    oversold: float = 30.0,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    min_down_bars: int = 2,
    low_tolerance_pct: float = 1.0,
    max_setup_bars: int = 12,
    confirm_bars: int = 4,
    use_macd_filter: bool = True,
    macd_lookback: int = 5,
    risk_reward: float = 2.0,
    stop_buffer_ticks: int = 2,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        raise ValueError(f"raw data missing columns: {missing}")

    df = raw.copy()
    open_ = df["open"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    df["rsi"] = relative_strength_index(close, rsi_len)
    bb_basis, bb_upper, bb_lower = bollinger_bands(close, bb_len, bb_mult)
    df["bb_basis"] = bb_basis
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    macd_line, macd_signal, macd_hist = moving_average_convergence_divergence(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    down_count = pd.Series(0, index=df.index, dtype=int)
    for idx in range(1, len(df)):
        down_count.iat[idx] = int(down_count.iat[idx - 1]) + 1 if close.iat[idx] < close.iat[idx - 1] else 0
    df["down_count"] = down_count

    bearish_prev = close.shift(1) < open_.shift(1)
    df["bullish_engulfing"] = (close > open_) & bearish_prev & (close >= open_.shift(1)) & (open_ <= close.shift(1))
    df["bullish_confirm"] = (close > open_) & (close > close.shift(1))
    df["setup_candidate"] = (df["rsi"] <= oversold) & (low <= df["bb_lower"]) & (df["down_count"] >= min_down_bars)

    recent_macd_cross = np.zeros(len(df), dtype=bool)
    last_cross_index = -10_000
    for idx in range(1, len(df)):
        crossed = bool(macd_line.iat[idx] > macd_signal.iat[idx] and macd_line.iat[idx - 1] <= macd_signal.iat[idx - 1])
        if crossed:
            last_cross_index = idx
        recent_macd_cross[idx] = (idx - last_cross_index) <= macd_lookback
    df["macd_ok"] = (not use_macd_filter) | (
        (df["macd_line"] > df["macd_signal"]) & pd.Series(recent_macd_cross, index=df.index)
    )

    rebound_marker = np.zeros(len(df), dtype=bool)
    second_bottom_marker = np.zeros(len(df), dtype=bool)
    buy_signal = np.zeros(len(df), dtype=bool)
    sell_signal = np.zeros(len(df), dtype=bool)
    setup_state = np.zeros(len(df), dtype=int)
    trade_stop = np.full(len(df), np.nan, dtype=float)
    take_profit = np.full(len(df), np.nan, dtype=float)

    tick_size = _infer_price_tick(df)
    stop_buffer = max(float(stop_buffer_ticks), 0.0) * tick_size

    state = 0
    stage_start_index = -1
    first_low = np.nan
    second_low = np.nan

    in_position = False
    entry_index = -1
    active_stop = np.nan
    active_take_profit = np.nan

    for idx in range(len(df)):
        if in_position:
            trade_stop[idx] = active_stop
            take_profit[idx] = active_take_profit
            if idx > entry_index:
                stop_hit = np.isfinite(active_stop) and low.iat[idx] <= active_stop
                take_profit_hit = np.isfinite(active_take_profit) and high.iat[idx] >= active_take_profit
                if stop_hit or take_profit_hit:
                    sell_signal[idx] = True
                    in_position = False
                    entry_index = -1
                    active_stop = np.nan
                    active_take_profit = np.nan
                    state = 0
                    stage_start_index = -1
                    first_low = np.nan
                    second_low = np.nan
                    setup_state[idx] = state
                    continue

        if idx == 0 or in_position:
            setup_state[idx] = state
            continue

        current_low = float(low.iat[idx])
        lower_band = float(df["bb_lower"].iat[idx]) if pd.notna(df["bb_lower"].iat[idx]) else float("nan")
        tolerance_low = float(first_low) * (1 - (low_tolerance_pct / 100.0)) if np.isfinite(first_low) else float("nan")

        if state == 0:
            if bool(df["setup_candidate"].iat[idx]):
                state = 1
                stage_start_index = idx
                first_low = current_low
                second_low = np.nan

        elif state == 1:
            first_low = min(float(first_low), current_low) if np.isfinite(first_low) else current_low
            if bool(df["bullish_engulfing"].iat[idx]):
                state = 2
                stage_start_index = idx
                rebound_marker[idx] = True
            elif (idx - stage_start_index) > max_setup_bars:
                state = 0
                stage_start_index = -1
                first_low = np.nan
                second_low = np.nan

        elif state == 2:
            invalid_break = (
                np.isfinite(tolerance_low)
                and pd.notna(lower_band)
                and current_low < tolerance_low
                and current_low < lower_band
            )
            second_bottom = (
                idx > 0
                and pd.notna(lower_band)
                and np.isfinite(tolerance_low)
                and (close.iat[idx] < close.iat[idx - 1] or current_low < low.iat[idx - 1])
                and current_low >= lower_band
                and current_low >= tolerance_low
            )
            if invalid_break:
                state = 0
                stage_start_index = -1
                first_low = np.nan
                second_low = np.nan
            elif second_bottom:
                state = 3
                stage_start_index = idx
                second_low = current_low
                second_bottom_marker[idx] = True
            elif (idx - stage_start_index) > max_setup_bars:
                state = 0
                stage_start_index = -1
                first_low = np.nan
                second_low = np.nan

        elif state == 3:
            if pd.notna(lower_band) and current_low >= lower_band:
                second_low = min(float(second_low), current_low) if np.isfinite(second_low) else current_low
            invalid_break = pd.notna(lower_band) and np.isfinite(second_low) and current_low < lower_band and current_low < second_low
            if invalid_break:
                state = 0
                stage_start_index = -1
                first_low = np.nan
                second_low = np.nan
            elif bool(df["bullish_confirm"].iat[idx]) and bool(df["macd_ok"].iat[idx]):
                stop_price = float(second_low) - stop_buffer if np.isfinite(second_low) else np.nan
                entry_price = float(close.iat[idx])
                risk = entry_price - stop_price if np.isfinite(stop_price) else 0.0
                if risk > 0:
                    active_stop = stop_price
                    active_take_profit = entry_price + (risk * float(risk_reward))
                    trade_stop[idx] = active_stop
                    take_profit[idx] = active_take_profit
                    buy_signal[idx] = True
                    in_position = True
                    entry_index = idx
                state = 0
                stage_start_index = -1
                first_low = np.nan
                second_low = np.nan
            elif (idx - stage_start_index) > confirm_bars:
                state = 0
                stage_start_index = -1
                first_low = np.nan
                second_low = np.nan

        setup_state[idx] = state

    oversold_gap = (oversold - df["rsi"]).clip(lower=0.0).fillna(0.0)
    bb_gap_pct = (((df["bb_lower"] - close) / close.replace(0.0, np.nan)) * 100.0).clip(lower=0.0).fillna(0.0)
    macd_gap = (df["macd_line"] - df["macd_signal"]).fillna(0.0)
    df["strategy_score"] = oversold_gap + (bb_gap_pct * 2.0) + (macd_gap * 10.0)

    df["rebound_marker"] = rebound_marker
    df["second_bottom_marker"] = second_bottom_marker
    df["buy_signal"] = buy_signal
    df["sell_signal"] = sell_signal
    df["setup_state"] = setup_state
    df["trade_stop"] = trade_stop
    df["take_profit"] = take_profit
    return df


def build_ema_pullback_signals(
    raw: pd.DataFrame,
    *,
    fast_ema: int = 21,
    slow_ema: int = 55,
    rsi_window: int = 14,
    rsi_floor: float = 42.0,
    rsi_ceiling: float = 62.0,
    pullback_tolerance_pct: float = 0.6,
    atr_window: int = 14,
    atr_mult: float = 2.0,
    volume_window: int = 20,
    volume_threshold: float = 0.9,
    exit_rsi: float = 68.0,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        raise ValueError(f"raw data missing columns: {missing}")

    df = raw.copy()
    open_ = df["open"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    df["ema_fast"] = ema(close, fast_ema)
    df["ema_slow"] = ema(close, slow_ema)
    df["rsi"] = relative_strength_index(close, rsi_window)
    df["atr"] = average_true_range(df, atr_window)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)

    trend_gap = ((close / df["ema_slow"]) - 1.0) * 100.0
    fast_slope = df["ema_fast"].pct_change(max(2, fast_ema // 3)) * 100.0
    pullback_band = df["ema_fast"] * (1.0 + (pullback_tolerance_pct / 100.0))
    df["pullback_band"] = pullback_band
    df["atr_stop"] = df["ema_slow"] - (atr_mult * df["atr"])
    df["strategy_score"] = (
        trend_gap.fillna(0.0) * 1.4
        + fast_slope.fillna(0.0) * 6.0
        + (df["volume_ratio"].fillna(1.0) - 1.0) * 8.0
        - (df["rsi"].sub(50.0).abs().fillna(50.0) * 0.25)
    )

    slope_window = max(2, slow_ema // 5)
    regime_ok = (close > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"]) & (df["ema_slow"] > df["ema_slow"].shift(slope_window))
    pullback = (low <= pullback_band) & (close >= (df["ema_fast"] * (1.0 - (pullback_tolerance_pct / 100.0))))
    rebound = (close > open_) & (close > close.shift(1))
    rsi_reset = df["rsi"].between(rsi_floor, rsi_ceiling, inclusive="both")
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold
    raw_buy = regime_ok & pullback & rebound & rsi_reset & volume_ok

    exit_on_rsi = (df["rsi"] >= exit_rsi) & (close < close.shift(1))
    trend_fail = close < df["ema_slow"]
    raw_sell = trend_fail | (close < df["atr_stop"]) | exit_on_rsi

    df["buy_signal"] = raw_buy & (~raw_buy.shift(1, fill_value=False))
    df["sell_signal"] = raw_sell & (~raw_sell.shift(1, fill_value=False))
    return df


def build_squeeze_breakout_signals(
    raw: pd.DataFrame,
    *,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    squeeze_window: int = 20,
    breakout_window: int = 20,
    trend_ema_window: int = 55,
    atr_window: int = 14,
    atr_mult: float = 2.0,
    volume_window: int = 20,
    volume_threshold: float = 1.1,
    squeeze_quantile: float = 0.35,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(raw.columns):
        missing = ", ".join(sorted(required - set(raw.columns)))
        raise ValueError(f"raw data missing columns: {missing}")

    df = raw.copy()
    close = df["close"].astype(float)

    df["trend_ema"] = ema(close, trend_ema_window)
    df["atr"] = average_true_range(df, atr_window)
    bb_basis, bb_upper, bb_lower = bollinger_bands(close, bb_len, bb_mult)
    df["bb_basis"] = bb_basis
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bandwidth"] = ((bb_upper - bb_lower) / bb_basis.replace(0.0, np.nan)) * 100.0
    df["breakout_high"] = df["high"].rolling(breakout_window).max().shift(1)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)
    df["atr_stop"] = df["trend_ema"] - (atr_mult * df["atr"])

    squeeze_ref = df["bandwidth"].rolling(squeeze_window, min_periods=max(5, squeeze_window // 2)).quantile(squeeze_quantile).shift(1)
    squeeze_on = (df["bandwidth"] <= squeeze_ref).fillna(False)
    recent_squeeze = squeeze_on.rolling(4, min_periods=1).max().astype(bool)
    breakout = close > df["breakout_high"]
    trend_ok = (close > df["trend_ema"]) & (df["trend_ema"] > df["trend_ema"].shift(max(2, trend_ema_window // 5)))
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold
    momentum = close.pct_change(max(2, breakout_window // 3)) * 100.0

    compression_bonus = (squeeze_ref - df["bandwidth"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    breakout_gap = ((close / df["breakout_high"].replace(0.0, np.nan)) - 1.0) * 100.0
    df["squeeze_on"] = squeeze_on
    df["strategy_score"] = (
        breakout_gap.fillna(0.0) * 18.0
        + momentum.fillna(0.0) * 1.2
        + (df["volume_ratio"].fillna(1.0) - 1.0) * 8.0
        + compression_bonus * 2.0
        - df["bandwidth"].fillna(0.0) * 0.15
    )

    raw_buy = recent_squeeze & breakout & trend_ok & volume_ok
    raw_sell = (close < df["trend_ema"]) | (close < df["atr_stop"]) | ((close < df["bb_basis"]) & (df["bandwidth"] > df["bandwidth"].shift(1)))

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


def parameter_grid_size(param_grid: dict[str, Sequence[float | int]]) -> int:
    sizes = [len(list(values)) for values in param_grid.values() if list(values)]
    if not sizes:
        return 0
    total = 1
    for size in sizes:
        total *= size
    return total


def sweep_research_trend_parameters(
    raw: pd.DataFrame,
    *,
    base_params: dict[str, float | int] | None = None,
    candidate_grid: dict[str, Sequence[float | int]] | None = None,
    fee: float = 0.0005,
    slippage_bps: float = 3.0,
) -> pd.DataFrame:
    base = dict(base_params or {})
    grid = {key: list(values) for key, values in dict(candidate_grid or {}).items() if list(values)}
    if not grid:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    keys = list(grid.keys())
    for combo in product(*(grid[key] for key in keys)):
        combo_params = dict(zip(keys, combo, strict=False))
        merged = {**base, **combo_params}
        frame = build_research_trend_signals(
            raw,
            fast_ema=int(merged.get("fast_ema", 21)),
            slow_ema=int(merged.get("slow_ema", 55)),
            breakout_window=int(merged.get("breakout_window", 20)),
            exit_window=int(merged.get("exit_window", 10)),
            atr_window=int(merged.get("atr_window", 14)),
            atr_mult=float(merged.get("atr_mult", 2.5)),
            adx_window=int(merged.get("adx_window", 14)),
            adx_threshold=float(merged.get("adx_threshold", 18.0)),
            momentum_window=int(merged.get("momentum_window", 20)),
            volume_window=int(merged.get("volume_window", 20)),
            volume_threshold=float(merged.get("volume_threshold", 0.9)),
        )
        bt = backtest_signal_frame(frame, fee=fee, slippage_bps=slippage_bps)
        rows.append(
            {
                **combo_params,
                "trades": int(bt["trades"]),
                "buy_signals": int(frame["buy_signal"].sum()),
                "sell_signals": int(frame["sell_signal"].sum()),
                "total_return_pct": float(bt["total_return_pct"]),
                "win_rate_pct": float(bt["win_rate_pct"]),
                "max_drawdown_pct": float(bt["max_drawdown_pct"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    results = pd.DataFrame(rows)
    return results.sort_values(
        ["total_return_pct", "max_drawdown_pct", "win_rate_pct", "trades"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


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
    "rsi_bb_double_bottom": "RSI oversold + Bollinger lower band + second-bottom rebound",
}


__all__ = [
    "Signal",
    "SMA_HELP_MD",
    "STRATEGY_HELP",
    "average_directional_index",
    "average_true_range",
    "backtest_signal_frame",
    "bollinger_bands",
    "build_relative_strength_rotation_signals",
    "build_research_trend_signals",
    "build_rsi_bb_double_bottom_signals",
    "ema",
    "ema_cross_signals",
    "extract_backtest_trade_events",
    "moving_average_convergence_divergence",
    "momentum_signals",
    "parameter_grid_size",
    "relative_strength_index",
    "rsi_signals",
    "sma",
    "sma_cross_signals",
    "sweep_research_trend_parameters",
]

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


def _bearish_regime_filter(
    close: pd.Series,
    fast_line: pd.Series,
    slow_line: pd.Series,
    *,
    buffer_pct: float = 0.0,
    slope_window: int = 4,
    adx: pd.Series | None = None,
    adx_floor: float | None = None,
    momentum: pd.Series | None = None,
    momentum_floor: float | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    slow_buffer = 1.0 - (max(float(buffer_pct), 0.0) / 100.0)
    resolved_slope_window = max(int(slope_window), 2)
    slow_slope = (slow_line.pct_change(resolved_slope_window) * 100.0).fillna(0.0)

    confirmations = pd.Series(False, index=close.index, dtype=bool)
    confirmations |= slow_slope < 0.0

    if adx is not None and adx_floor is not None:
        confirmations |= pd.to_numeric(adx, errors="coerce").fillna(0.0) >= float(adx_floor)

    if momentum is not None and momentum_floor is not None:
        confirmations |= pd.to_numeric(momentum, errors="coerce").fillna(0.0) <= float(momentum_floor)

    bearish_regime = (
        (close < (slow_line * slow_buffer))
        & (fast_line < slow_line)
        & confirmations
    ).fillna(False)
    risk_on_regime = (~bearish_regime).astype(bool)
    return bearish_regime, risk_on_regime, slow_slope


def build_relative_strength_guard_signals(
    raw: pd.DataFrame,
    *,
    rs_short_window: int = 10,
    rs_mid_window: int = 30,
    rs_long_window: int = 90,
    trend_ema_window: int = 55,
    breakout_window: int = 28,
    atr_window: int = 14,
    atr_mult: float = 2.2,
    volume_window: int = 20,
    volume_threshold: float = 0.9,
    entry_score: float = 9.0,
    exit_score: float = 3.0,
    guard_fast_ema: int = 13,
    guard_slow_ema: int = 144,
    guard_buffer_pct: float = 1.0,
    guard_adx_window: int = 14,
    guard_adx_floor: float = 10.0,
    guard_rs_floor: float = -3.0,
) -> pd.DataFrame:
    df = build_relative_strength_rotation_signals(
        raw,
        rs_short_window=rs_short_window,
        rs_mid_window=rs_mid_window,
        rs_long_window=rs_long_window,
        trend_ema_window=trend_ema_window,
        breakout_window=breakout_window,
        atr_window=atr_window,
        atr_mult=atr_mult,
        volume_window=volume_window,
        volume_threshold=volume_threshold,
        entry_score=entry_score,
        exit_score=exit_score,
    )

    close = df["close"].astype(float)
    df["guard_fast_ema"] = ema(close, guard_fast_ema)
    df["guard_slow_ema"] = ema(close, guard_slow_ema)
    df["guard_adx"] = average_directional_index(df, guard_adx_window)
    bearish_regime, risk_on_regime, slow_slope = _bearish_regime_filter(
        close,
        df["guard_fast_ema"],
        df["guard_slow_ema"],
        buffer_pct=guard_buffer_pct,
        slope_window=max(2, guard_slow_ema // 8),
        adx=df["guard_adx"],
        adx_floor=guard_adx_floor,
        momentum=df["rs_long"],
        momentum_floor=guard_rs_floor,
    )

    df["bearish_regime"] = bearish_regime
    df["risk_on_regime"] = risk_on_regime
    df["guard_slow_slope"] = slow_slope
    df["base_buy_signal"] = df["buy_signal"].astype(bool)
    df["base_sell_signal"] = df["sell_signal"].astype(bool)
    bearish_exit_signal = bearish_regime & (~bearish_regime.shift(1, fill_value=False))
    df["strategy_score"] = pd.to_numeric(df["strategy_score"], errors="coerce").fillna(0.0) + (risk_on_regime.astype(float) * 0.5)
    df["buy_signal"] = df["base_buy_signal"] & risk_on_regime
    df["sell_signal"] = df["base_sell_signal"] | bearish_exit_signal
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


def build_rsi_trend_guard_signals(
    raw: pd.DataFrame,
    *,
    rsi_len: int = 10,
    oversold: float = 35.0,
    bb_len: int = 20,
    bb_mult: float = 1.5,
    min_down_bars: int = 2,
    low_tolerance_pct: float = 1.0,
    max_setup_bars: int = 6,
    confirm_bars: int = 3,
    use_macd_filter: bool = True,
    macd_lookback: int = 5,
    risk_reward: float = 1.5,
    stop_buffer_ticks: int = 2,
    trend_fast_ema: int = 13,
    trend_slow_ema: int = 89,
    trend_buffer_pct: float = 2.0,
    bearish_adx_floor: float = 14.0,
    adx_window: int = 14,
) -> pd.DataFrame:
    df = build_rsi_bb_double_bottom_signals(
        raw,
        rsi_len=rsi_len,
        oversold=oversold,
        bb_len=bb_len,
        bb_mult=bb_mult,
        min_down_bars=min_down_bars,
        low_tolerance_pct=low_tolerance_pct,
        max_setup_bars=max_setup_bars,
        confirm_bars=confirm_bars,
        use_macd_filter=use_macd_filter,
        macd_lookback=macd_lookback,
        risk_reward=risk_reward,
        stop_buffer_ticks=stop_buffer_ticks,
    )

    close = df["close"].astype(float)
    df["ema_fast"] = ema(close, trend_fast_ema)
    df["ema_slow"] = ema(close, trend_slow_ema)
    df["adx"] = average_directional_index(df, adx_window)

    slow_buffer = 1.0 - (max(float(trend_buffer_pct), 0.0) / 100.0)
    bearish_regime = (
        (close < (df["ema_slow"] * slow_buffer))
        & (df["ema_fast"] < df["ema_slow"])
        & (df["adx"] >= float(bearish_adx_floor))
    ).fillna(False)
    trend_filter = (~bearish_regime).astype(bool)

    df["bearish_regime"] = bearish_regime
    df["trend_filter"] = trend_filter
    df["base_buy_signal"] = df["buy_signal"].astype(bool)
    df["base_sell_signal"] = df["sell_signal"].astype(bool)
    df["strategy_score"] = pd.to_numeric(df["strategy_score"], errors="coerce").fillna(0.0) + (trend_filter.astype(float) * 0.5)
    df["buy_signal"] = df["base_buy_signal"] & trend_filter
    df["sell_signal"] = df["base_sell_signal"] | bearish_regime
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


def build_volatility_reset_breakout_signals(
    raw: pd.DataFrame,
    *,
    fast_ema: int = 20,
    slow_ema: int = 60,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    breakout_window: int = 18,
    reset_window: int = 8,
    atr_window: int = 14,
    atr_mult: float = 2.1,
    volume_window: int = 20,
    volume_threshold: float = 1.0,
    spike_window: int = 24,
    spike_quantile: float = 0.7,
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
    bb_basis, bb_upper, bb_lower = bollinger_bands(close, bb_len, bb_mult)
    df["bb_basis"] = bb_basis
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bandwidth"] = ((bb_upper - bb_lower) / bb_basis.replace(0.0, np.nan)) * 100.0
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(volume_window).mean().replace(0.0, np.nan)
    df["atr_ratio"] = df["atr"] / df["atr"].rolling(max(5, spike_window // 2)).mean().replace(0.0, np.nan)
    df["breakout_high"] = df["high"].rolling(breakout_window).max().shift(1)
    df["reset_low"] = df["low"].rolling(max(3, reset_window)).min().shift(1)
    df["atr_stop"] = df["ema_slow"] - (atr_mult * df["atr"])

    bandwidth_ref = df["bandwidth"].rolling(spike_window, min_periods=max(5, spike_window // 2)).quantile(spike_quantile).shift(1)
    bandwidth_mean = df["bandwidth"].rolling(spike_window, min_periods=max(5, spike_window // 2)).mean().shift(1)
    spike_on = (df["bandwidth"] >= bandwidth_ref) | (df["atr_ratio"] >= 1.2)
    spike_recent = spike_on.rolling(reset_window, min_periods=1).max().astype(bool)
    cooling = (df["bandwidth"] <= bandwidth_mean * 0.9) | (df["bandwidth"] < df["bandwidth"].shift(1))
    cooling_recent = cooling.rolling(reset_window, min_periods=1).max().astype(bool)

    trend_ok = (close > df["ema_slow"]) & (df["ema_fast"] > df["ema_slow"]) & (df["ema_slow"] > df["ema_slow"].shift(max(2, slow_ema // 5)))
    reclaim = (close > df["breakout_high"]) & (close > df["ema_fast"])
    volume_ok = df["volume_ratio"].fillna(1.0) >= volume_threshold

    trend_gap = ((close / df["ema_slow"]) - 1.0) * 100.0
    reset_gap = (bandwidth_mean - df["bandwidth"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    spike_gap = (df["atr_ratio"] - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["strategy_score"] = (
        spike_gap * 14.0
        + reset_gap * 2.5
        + trend_gap.fillna(0.0) * 1.6
        + (df["volume_ratio"].fillna(1.0) - 1.0) * 6.0
    )

    raw_buy = trend_ok & spike_recent & cooling_recent & reclaim & volume_ok
    raw_sell = (close < df["ema_slow"]) | (close < df["atr_stop"]) | (close < df["reset_low"])

    df["spike_recent"] = spike_recent
    df["cooling_recent"] = cooling_recent
    df["reclaim_high"] = df["breakout_high"]
    df["buy_signal"] = raw_buy & (~raw_buy.shift(1, fill_value=False))
    df["sell_signal"] = raw_sell & (~raw_sell.shift(1, fill_value=False))
    return df


def build_regime_blend_signals(
    raw: pd.DataFrame,
    *,
    trend_fast_ema: int = 21,
    trend_slow_ema: int = 55,
    trend_breakout_window: int = 20,
    trend_exit_window: int = 10,
    trend_atr_window: int = 14,
    trend_atr_mult: float = 2.5,
    trend_adx_window: int = 14,
    trend_adx_threshold: float = 18.0,
    trend_momentum_window: int = 20,
    trend_volume_window: int = 20,
    trend_volume_threshold: float = 0.9,
    rsi_len: int = 10,
    oversold: float = 35.0,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    min_down_bars: int = 2,
    low_tolerance_pct: float = 1.0,
    max_setup_bars: int = 12,
    confirm_bars: int = 5,
    use_macd_filter: bool = True,
    macd_lookback: int = 5,
    risk_reward: float = 1.5,
    stop_buffer_ticks: int = 2,
    regime_adx_floor: float = 16.0,
) -> pd.DataFrame:
    trend_frame = build_research_trend_signals(
        raw,
        fast_ema=trend_fast_ema,
        slow_ema=trend_slow_ema,
        breakout_window=trend_breakout_window,
        exit_window=trend_exit_window,
        atr_window=trend_atr_window,
        atr_mult=trend_atr_mult,
        adx_window=trend_adx_window,
        adx_threshold=trend_adx_threshold,
        momentum_window=trend_momentum_window,
        volume_window=trend_volume_window,
        volume_threshold=trend_volume_threshold,
    )
    range_frame = build_rsi_bb_double_bottom_signals(
        raw,
        rsi_len=rsi_len,
        oversold=oversold,
        bb_len=bb_len,
        bb_mult=bb_mult,
        min_down_bars=min_down_bars,
        low_tolerance_pct=low_tolerance_pct,
        max_setup_bars=max_setup_bars,
        confirm_bars=confirm_bars,
        use_macd_filter=use_macd_filter,
        macd_lookback=macd_lookback,
        risk_reward=risk_reward,
        stop_buffer_ticks=stop_buffer_ticks,
    )

    df = raw.copy()
    close = df["close"].astype(float)
    trend_regime = (
        (close > trend_frame["ema_slow"])
        & (trend_frame["ema_fast"] > trend_frame["ema_slow"])
        & (trend_frame["adx"] >= regime_adx_floor)
    ).fillna(False)
    df["trend_regime"] = trend_regime
    df["trend_score"] = pd.to_numeric(trend_frame.get("strategy_score"), errors="coerce").fillna(0.0)
    df["range_score"] = pd.to_numeric(range_frame.get("strategy_score"), errors="coerce").fillna(0.0)
    df["strategy_score"] = df["trend_score"].where(trend_regime, df["range_score"])
    df["ema_fast"] = trend_frame["ema_fast"]
    df["ema_slow"] = trend_frame["ema_slow"]
    df["adx"] = trend_frame["adx"]
    df["atr_stop"] = trend_frame["atr_stop"]
    df["rsi"] = range_frame["rsi"]
    df["bb_lower"] = range_frame["bb_lower"]
    df["bb_upper"] = range_frame["bb_upper"]
    df["trade_stop"] = range_frame["trade_stop"]
    df["take_profit"] = range_frame["take_profit"]
    df["trend_buy_signal"] = trend_frame["buy_signal"].astype(bool)
    df["trend_sell_signal"] = trend_frame["sell_signal"].astype(bool)
    df["range_buy_signal"] = range_frame["buy_signal"].astype(bool)
    df["range_sell_signal"] = range_frame["sell_signal"].astype(bool)

    buy_signal = pd.Series(False, index=df.index, dtype=bool)
    sell_signal = pd.Series(False, index=df.index, dtype=bool)
    active_mode: str | None = None
    mode_trace: list[str] = []

    for ts in df.index:
        if active_mode == "trend":
            if bool(df.at[ts, "trend_sell_signal"]):
                sell_signal.at[ts] = True
                active_mode = None
        elif active_mode == "range":
            if bool(df.at[ts, "range_sell_signal"]):
                sell_signal.at[ts] = True
                active_mode = None

        if active_mode is None:
            if bool(df.at[ts, "trend_regime"]) and bool(df.at[ts, "trend_buy_signal"]):
                buy_signal.at[ts] = True
                active_mode = "trend"
            elif (not bool(df.at[ts, "trend_regime"])) and bool(df.at[ts, "range_buy_signal"]):
                buy_signal.at[ts] = True
                active_mode = "range"

        mode_trace.append(active_mode or "")

    df["entry_mode"] = pd.Series(mode_trace, index=df.index, dtype="string")
    df["buy_signal"] = buy_signal
    df["sell_signal"] = sell_signal
    return df


def build_regime_blend_guard_signals(
    raw: pd.DataFrame,
    *,
    trend_fast_ema: int = 21,
    trend_slow_ema: int = 55,
    trend_breakout_window: int = 20,
    trend_exit_window: int = 10,
    trend_atr_window: int = 14,
    trend_atr_mult: float = 2.5,
    trend_adx_window: int = 14,
    trend_adx_threshold: float = 18.0,
    trend_momentum_window: int = 20,
    trend_volume_window: int = 20,
    trend_volume_threshold: float = 0.9,
    rsi_len: int = 10,
    oversold: float = 35.0,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    min_down_bars: int = 2,
    low_tolerance_pct: float = 1.0,
    max_setup_bars: int = 12,
    confirm_bars: int = 5,
    use_macd_filter: bool = True,
    macd_lookback: int = 5,
    risk_reward: float = 1.5,
    stop_buffer_ticks: int = 2,
    regime_adx_floor: float = 16.0,
    bear_guard_buffer_pct: float = 1.5,
    bear_guard_adx_floor: float = 14.0,
    bear_guard_score_floor: float = -2.0,
) -> pd.DataFrame:
    df = build_regime_blend_signals(
        raw,
        trend_fast_ema=trend_fast_ema,
        trend_slow_ema=trend_slow_ema,
        trend_breakout_window=trend_breakout_window,
        trend_exit_window=trend_exit_window,
        trend_atr_window=trend_atr_window,
        trend_atr_mult=trend_atr_mult,
        trend_adx_window=trend_adx_window,
        trend_adx_threshold=trend_adx_threshold,
        trend_momentum_window=trend_momentum_window,
        trend_volume_window=trend_volume_window,
        trend_volume_threshold=trend_volume_threshold,
        rsi_len=rsi_len,
        oversold=oversold,
        bb_len=bb_len,
        bb_mult=bb_mult,
        min_down_bars=min_down_bars,
        low_tolerance_pct=low_tolerance_pct,
        max_setup_bars=max_setup_bars,
        confirm_bars=confirm_bars,
        use_macd_filter=use_macd_filter,
        macd_lookback=macd_lookback,
        risk_reward=risk_reward,
        stop_buffer_ticks=stop_buffer_ticks,
        regime_adx_floor=regime_adx_floor,
    )

    close = df["close"].astype(float)
    bearish_regime, risk_on_regime, slow_slope = _bearish_regime_filter(
        close,
        df["ema_fast"],
        df["ema_slow"],
        buffer_pct=bear_guard_buffer_pct,
        slope_window=max(2, trend_slow_ema // 6),
        adx=df["adx"],
        adx_floor=bear_guard_adx_floor,
        momentum=df["trend_score"],
        momentum_floor=bear_guard_score_floor,
    )

    df["bearish_regime"] = bearish_regime
    df["risk_on_regime"] = risk_on_regime
    df["bear_guard_slow_slope"] = slow_slope
    df["base_buy_signal"] = df["buy_signal"].astype(bool)
    df["base_sell_signal"] = df["sell_signal"].astype(bool)
    df["base_entry_mode"] = df["entry_mode"].astype("string")
    bearish_exit_signal = bearish_regime & (~bearish_regime.shift(1, fill_value=False))
    df["trend_regime"] = df["trend_regime"].astype(bool) & risk_on_regime
    df["strategy_score"] = pd.to_numeric(df["strategy_score"], errors="coerce").fillna(0.0) + (risk_on_regime.astype(float) * 0.5)
    df["buy_signal"] = df["base_buy_signal"] & risk_on_regime
    df["sell_signal"] = df["base_sell_signal"] | bearish_exit_signal
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
    "volatility_reset_breakout": "Volatility spike, cooldown, then reclaim breakout",
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
    "build_volatility_reset_breakout_signals",
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

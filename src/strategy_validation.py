"""Read-only, chronological strategy validation. Never loads keys or sends orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests

from src.strategy import backtest_signal_frame
from src.strategy_engine import build_strategy_frame


# Freeze this list before viewing the test period; do not tune to its results.
DEFAULT_CANDIDATES = (
    "research_trend", "ema_pullback", "relative_strength_rotation",
    "relative_strength_guard",
)


def fetch_daily(market: str, count: int, *, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch completed UTC daily bars from the public Korean quotation API."""
    rows = []
    cursor = end
    with requests.Session() as session:
        while len(rows) < count:
            response = session.get(
                "https://api.upbit.com/v1/candles/days",
                params={"market": market, "count": min(200, count - len(rows)),
                        "to": cursor.isoformat()}, timeout=20,
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            next_cursor = min(pd.Timestamp(r["candle_date_time_utc"], tz="UTC") for r in batch)
            if next_cursor >= cursor:
                raise ValueError("Candle pagination did not advance")
            rows.extend(batch)
            cursor = next_cursor
            time.sleep(.15)
    if not rows:
        raise ValueError(f"No candles for {market}")
    raw = pd.DataFrame(rows)
    raw.index = pd.to_datetime(raw["candle_date_time_utc"], utc=True)
    raw = raw.rename(columns={"opening_price": "open", "high_price": "high",
                              "low_price": "low", "trade_price": "close",
                              "candle_acc_trade_volume": "volume"})
    result = raw[["open", "high", "low", "close", "volume"]].astype(float)
    result = result[~result.index.duplicated()].sort_index()
    return result.loc[result.index + pd.Timedelta(days=1) <= end].tail(count)


def _basket_metrics(frames, *, fee, slippage_bps):
    """Equal initial capital per market, no cross-market cash transfers."""
    results = [backtest_signal_frame(f, fee=fee, slippage_bps=slippage_bps,
                                    execution_mode="next_open") for f in frames.values()]
    equity = pd.concat([r["equity"] for r in results], axis=1).mean(axis=1)
    peak = equity.cummax().clip(lower=1.0)
    return {
        "return_pct": float((equity.iloc[-1] - 1) * 100),
        "max_drawdown_pct": float(((equity / peak - 1) * 100).min()),
        "closed_trades": sum(r["trades"] for r in results),
    }


def validate_strategies(raw_by_market, *, train_bars=600, warmup_bars=150,
                        fee=.0005, slippage_bps=10.0,
                        candidates=DEFAULT_CANDIDATES):
    """Select using train only; test the frozen selection once, including costs.

    Results are historical evidence, never authorization to enable live trading.
    The test period must not be reused for tuning or repeated candidate selection.
    """
    if not raw_by_market or not candidates:
        raise ValueError("Markets and candidates are required")
    if train_bars <= warmup_bars or warmup_bars < 0:
        raise ValueError("train_bars must exceed nonnegative warmup_bars")
    if not np.isfinite(fee) or not 0 <= fee < 1:
        raise ValueError("fee must be between 0 and 1")
    if not np.isfinite(slippage_bps) or not 0 <= slippage_bps < 10000:
        raise ValueError("slippage_bps must be between 0 and 10000")
    reference = next(iter(raw_by_market.values())).index
    if len(reference) <= train_bars + 30:
        raise ValueError("Need more than 30 test bars after training")
    for market, raw in raw_by_market.items():
        if not raw.index.equals(reference) or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
            raise ValueError(f"Aligned, unique, chronological candle indexes required: {market}")
        values = raw[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
            raise ValueError(f"Invalid candle values: {market}")

    leaderboard = []
    for name in candidates:
        # The builder never sees test rows during selection (including its indicators).
        frames = {m: build_strategy_frame(raw.iloc[:train_bars].copy(), strategy_name=name)
                  .iloc[warmup_bars:] for m, raw in raw_by_market.items()}
        metrics = _basket_metrics(frames, fee=fee, slippage_bps=slippage_bps)
        score = metrics["return_pct"] - abs(metrics["max_drawdown_pct"])
        leaderboard.append({"strategy": name, "score": score, **metrics})
    leaderboard.sort(key=lambda r: (-r["score"], r["strategy"]))
    top = leaderboard[0]
    # Cash is an explicit candidate: losing or inactive training need not force trades.
    selected = top["strategy"] if top["score"] > 0 and top["closed_trades"] >= 5 else "cash"
    test_frames = {}
    benchmark_frames = {}
    for market, raw in raw_by_market.items():
        if selected == "cash":
            frame = raw.copy().assign(buy_signal=False, sell_signal=False)
        else:
            frame = build_strategy_frame(raw.copy(), strategy_name=selected)
        # One preceding row lets the last known training signal fill at first test open.
        test_frames[market] = frame.iloc[train_bars - 1:].copy()
        benchmark = raw.iloc[train_bars - 1:].copy().assign(buy_signal=False, sell_signal=False)
        benchmark.iloc[0, benchmark.columns.get_loc("buy_signal")] = True
        benchmark_frames[market] = benchmark
    test = _basket_metrics(test_frames, fee=fee, slippage_bps=slippage_bps)
    stress = _basket_metrics(test_frames, fee=fee, slippage_bps=max(30.0, slippage_bps * 3))
    benchmark = _basket_metrics(benchmark_frames, fee=fee, slippage_bps=slippage_bps)
    reasons = []
    if selected == "cash":
        reasons.append("Training did not justify a trading strategy; cash selected.")
    if test["return_pct"] <= 0:
        reasons.append("Test return is not positive after costs.")
    if stress["return_pct"] <= 0:
        reasons.append("Test return is not positive under higher slippage.")
    if test["closed_trades"] < 20:
        reasons.append("Fewer than 20 closed test trades; evidence is limited.")
    if test["max_drawdown_pct"] < -15:
        reasons.append("Test drawdown exceeds the 15% research threshold.")
    return {
        "selected_strategy": selected,
        "status": "PAPER_CANDIDATE" if not reasons else "INSUFFICIENT_EVIDENCE",
        "reasons": reasons, "live_enabled": False,
        "markets": list(raw_by_market), "bars_per_market": len(reference),
        "train_start": str(reference[warmup_bars]), "train_end": str(reference[train_bars - 1]),
        "test_start": str(reference[train_bars]), "test_end": str(reference[-1]),
        "execution": "previous close signal / next available candle open",
        "fee_rate": fee, "slippage_bps": slippage_bps,
        "training_leaderboard": leaderboard, "test": test, "stress_test": stress,
        "buy_and_hold": benchmark,
        "limitations": [
            "Fixed equal capital sleeves; not the live scanner's allocation or risk rules.",
            "Daily close drawdown can miss intraday losses; no order book or latency model.",
            "A reused test period is no longer independent. Reserve new future data before deployment.",
            "Selected markets and historical strategy development may introduce selection bias.",
            "Research thresholds are heuristics, not statistical proof or live approval.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", default=["KRW-BTC", "KRW-ETH", "KRW-XRP"])
    parser.add_argument("--bars", type=int, default=1000)
    parser.add_argument("--train-bars", type=int, default=600)
    parser.add_argument("--input-dir", type=Path, help="Replay saved market CSVs without network access")
    parser.add_argument("--output", type=Path, default=Path(".runtime/validation"))
    args = parser.parse_args()
    if args.bars <= args.train_bars + 30 or args.train_bars <= 150:
        parser.error("Require train-bars > 150 and bars > train-bars + 30")
    end = pd.Timestamp.now(tz="UTC").floor("D")
    if any(not market.startswith("KRW-") or not market[4:].isalnum() for market in args.markets):
        parser.error("Use KRW market names, for example KRW-BTC")
    if args.input_dir:
        frames = {}
        for market in args.markets:
            raw = pd.read_csv(args.input_dir / f"{market}.csv", index_col="timestamp")
            raw.index = pd.to_datetime(raw.index, utc=True)
            frames[market] = raw.tail(args.bars)
    else:
        frames = {market: fetch_daily(market, args.bars, end=end) for market in args.markets}
    common = next(iter(frames.values())).index
    for raw in frames.values():
        common = common.intersection(raw.index)
    frames = {m: raw.loc[common].copy() for m, raw in frames.items()}
    report = validate_strategies(frames, train_bars=args.train_bars)
    report["evaluated_at_utc"] = str(pd.Timestamp.now(tz="UTC"))
    report["data_source"] = str(args.input_dir.resolve()) if args.input_dir else "Upbit public daily candles"
    args.output.mkdir(parents=True, exist_ok=True)
    for market, raw in frames.items():
        # Market names come from public API input, but never become arbitrary paths.
        safe_name = "".join(c for c in market if c.isalnum() or c == "-")
        raw.to_csv(args.output / f"{safe_name}.csv", index_label="timestamp")
    output = args.output / "report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()

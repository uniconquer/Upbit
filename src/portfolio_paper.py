"""Forward paper observations using public candles/tickers; no account/order API."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import time

import pandas as pd
import requests

from src.adaptive_portfolio import PortfolioConfig, PortfolioSimulator, build_features
from src.strategy_validation import fetch_daily


@contextmanager
def state_lock(path):
    """OS lock is released on process exit, including crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def fetch_quotes(markets):
    response = requests.get('https://api.upbit.com/v1/ticker',
                            params={'markets': ','.join(markets)}, timeout=15)
    response.raise_for_status()
    data = response.json()
    now_ms = pd.Timestamp.now(tz='UTC').timestamp() * 1000
    if {r['market'] for r in data} != set(markets):
        raise ValueError('Incomplete ticker snapshot')
    if any(not -5000 <= now_ms - float(r['timestamp']) <= 120000 for r in data):
        raise ValueError('Stale ticker snapshot')
    return {r['market']: float(r['trade_price']) for r in data}


def observe_once(report, state_path):
    markets = report['markets']
    config = PortfolioConfig(**report['selected_config']) if report['selected_config'] else PortfolioConfig()
    allow_entries = report['selected_config'] is not None
    with state_lock(state_path):
        if state_path.exists():
            saved = json.loads(state_path.read_text(encoding='utf-8'))
            sim = PortfolioSimulator.from_state(saved['simulation'])
            if saved['markets'] != markets or asdict(sim.config) != asdict(config) or saved['allow_entries'] != allow_entries:
                raise ValueError('Profile changed: preserve this session and choose a new state path')
        else:
            sim = PortfolioSimulator(config, initial_cash=float(report['initial_cash']))
            saved = {'markets': markets, 'allow_entries': allow_entries, 'observations': 0,
                     'started_at': str(pd.Timestamp.now(tz='UTC'))}
        today = pd.Timestamp.now(tz='UTC').floor('D')
        signals = None
        if sim.last_signal_day != str(today.date()):
            history = max(int(report.get('history_bars', 2400)),
                          max(config.macro_window, config.trend_window, 120) + 250)
            raw = {m: fetch_daily(m, history, end=today)
                   for m in markets}
            if any(f.empty or f.index[-1] != today-pd.Timedelta(days=1) for f in raw.values()):
                raise ValueError('Missing latest completed daily candle')
            features = build_features(raw, config)
            signals = {m: f.iloc[-1].to_dict() for m, f in features.items()}
            if not allow_entries:
                for signal in signals.values():
                    signal['buy'] = False
        prices = fetch_quotes(markets)
        if pd.Timestamp.now(tz='UTC').floor('D') != today:
            raise ValueError('UTC day changed during fetch; retry with fresh signals')
        before = len(sim.events)
        sim.quote(prices, day=str(today.date()), signals=signals)
        observed_at = str(pd.Timestamp.now(tz='UTC'))
        for event in sim.events[before:]:
            # Forward fills record actual observation time, not the historical bar time.
            event['ts'] = pd.Timestamp(observed_at).timestamp()
            if event['side'] == 'BUY':
                sim.trader.get_position(event['market']).opened_at = event['ts']
        saved.update({'simulation': sim.to_state(), 'prices': prices,
                      'observed_at': observed_at, 'observations': saved['observations'] + 1,
                      'research_status': report['status']})
        atomic_save(state_path, saved)
        return {'mode': 'PAPER_ONLY', 'observed_at': observed_at, 'cash': sim.cash,
                'equity': sim.last_equity, 'open_positions': len(sim.trader.positions),
                'new_trades': len(sim.events)-before, 'halted': sim.halted,
                'daily_halted': sim.daily_halted, 'observations': saved['observations'],
                'blocked_exits': sim.blocked_exits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path('.runtime/portfolio-research-v2/paper-profile.json'))
    parser.add_argument('--state', type=Path, default=Path('.runtime/adaptive-paper-v2/state.json'))
    parser.add_argument('--cycles', type=int, default=1, help='Finite number of observations')
    parser.add_argument('--interval', type=int, default=60, help='Seconds between observations')
    args = parser.parse_args()
    if args.cycles < 1 or args.interval < 10:
        parser.error('cycles >= 1 and interval >= 10 required')
    report = json.loads(args.report.read_text(encoding='utf-8'))
    if report.get('mode') != 'PAPER_ONLY':
        parser.error('A paper research report is required')
    for i in range(args.cycles):
        print(json.dumps(observe_once(report, args.state), indent=2), flush=True)
        if i + 1 < args.cycles:
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

"""One-minute paper observations and bounded historical research. No real orders."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import time

import pandas as pd
import requests

from src.adaptive_portfolio import PortfolioConfig, build_features, replay, validate_frames
from src.paper_learning import advance, initial_population, new_learning_state
from src.portfolio_paper import atomic_save, fetch_quotes, state_lock
from src.portfolio_research import summary
from src.strategy import average_true_range


MARKETS = ['KRW-BTC', 'KRW-ETH', 'KRW-XRP']


def regular_minutes(raw, *, end, count):
    """No-trade minutes carry the previous price and have zero volume."""
    raw = raw.loc[raw.index < end].sort_index()
    raw = raw[~raw.index.duplicated(keep='last')]
    grid = pd.date_range(end-pd.Timedelta(minutes=count), periods=count, freq='min')
    expanded = raw.reindex(raw.index.union(grid)).sort_index()
    close = expanded.close.ffill()
    for column in ('open', 'high', 'low', 'close'):
        expanded[column] = expanded[column].fillna(close)
    expanded['volume'] = expanded.volume.fillna(0.)
    result = expanded.loc[grid]
    if result.isna().any().any():
        raise ValueError('Insufficient candle history; no backward fill allowed')
    return result


def fetch_minutes(market, *, end, count=2400, cache=None):
    existing = None
    if cache and cache.exists():
        existing = pd.read_csv(cache, index_col='timestamp')
        existing.index = pd.to_datetime(existing.index, utc=True)
    incremental = (existing is not None and len(existing) >= count and
                   pd.Timedelta(0) <= end-existing.index[-1] <= pd.Timedelta(minutes=180))
    wanted = 200 if incremental else count + 1
    rows = []
    cursor = end
    with requests.Session() as session:
        while len(rows) < wanted:
            response = session.get('https://api.upbit.com/v1/candles/minutes/1',
                                   params={'market': market, 'count': min(200, wanted-len(rows)),
                                           'to': cursor.isoformat()}, timeout=15)
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            next_cursor = min(pd.Timestamp(r['candle_date_time_utc'], tz='UTC') for r in batch)
            if next_cursor >= cursor:
                raise ValueError('Nonadvancing candle pagination')
            rows.extend(batch)
            cursor = next_cursor
            time.sleep(.15)
    if not rows:
        raise ValueError('No minute candles returned')
    raw = pd.DataFrame(rows)
    raw.index = pd.to_datetime(raw['candle_date_time_utc'], utc=True)
    raw = raw.rename(columns={'opening_price': 'open', 'high_price': 'high', 'low_price': 'low',
                              'trade_price': 'close', 'candle_acc_trade_volume': 'volume'})
    raw = raw[['open', 'high', 'low', 'close', 'volume']].astype(float)
    if incremental:
        raw = pd.concat([existing, raw])
    result = regular_minutes(raw, end=end, count=count)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix('.tmp')
        result.to_csv(temp, index_label='timestamp')
        temp.replace(cache)
    return result


def minute_features(raw, config):
    features = build_features(raw, config)
    contexts = {}
    if config.context_minutes > 1:
        for market, frame in raw.items():
            grouped = frame.resample(f'{config.context_minutes}min', closed='left', label='right')
            higher = grouped.agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            # Exclude partial groups, including the current unfinished higher candle.
            higher = higher.loc[(grouped.close.count() == config.context_minutes) &
                                (higher.index <= frame.index[-1] + pd.Timedelta(minutes=1))]
            ema = higher.close.ewm(span=20, adjust=False, min_periods=20).mean()
            context = pd.DataFrame({'trend': (higher.close > ema) & (ema > ema.shift(3)),
                                    'range': average_true_range(higher, 14)})
            aligned = context.reindex(frame.index + pd.Timedelta(minutes=1), method='ffill')
            aligned.index = frame.index
            contexts[market] = aligned
    for market in features:
        frame = features[market]
        frame['buy'] &= raw[market].volume > 0
        expected_range = frame.atr * config.stop_atr
        if contexts:
            frame['buy'] &= contexts[market].trend.eq(True) & contexts['KRW-BTC'].trend.eq(True)
            expected_range = contexts[market]['range']
        if config.edge_cost_ratio:
            # Historical range is an opportunity filter, not a profit prediction.
            costs = 2 * (config.fee + config.slippage_bps / 10000)
            frame['buy'] &= expected_range / raw[market].close >= costs * config.edge_cost_ratio
        frame['sell'] = frame.sell.rolling(config.exit_confirm_bars, min_periods=config.exit_confirm_bars).sum().eq(config.exit_confirm_bars)
    return features


def observe_once(folder):
    path = folder / 'state.json'
    with state_lock(path):
        now = pd.Timestamp.now(tz='UTC')
        boundary = now.floor('min')
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else new_learning_state(now=now, timeframe='minute1')
        if state.get('timeframe') != 'minute1':
            raise ValueError('Use a separate directory for minute learning')
        if state['last_bucket'] == str(boundary):
            return {'status': 'already_observed'}
        queue_path = folder / 'research-candidates.json'
        if queue_path.exists():
            queue = json.loads(queue_path.read_text(encoding='utf-8'))
            if queue['run_id'] != state.get('research_run_id'):
                pending = {item['id']: item for item in state.get('research_candidates', [])}
                for item in queue['nominees']:
                    PortfolioConfig(**item['config'])
                    pending[item['id']] = {**item, 'source': queue['source'], 'run_id': queue['run_id']}
                state['research_candidates'] = list(pending.values())[-4:]
                state['research_run_id'] = queue['run_id']
        raw = {m: fetch_minutes(m, end=boundary, cache=folder/'candles'/f'{m}.csv') for m in MARKETS}
        validate_frames(raw)
        prices = fetch_quotes(MARKETS)
        stamp = pd.Timestamp.now(tz='UTC')
        if stamp.floor('min') != boundary:
            raise ValueError('Minute boundary crossed during fetch; retry on next cycle')
        before = sum(len(c['base']['events']) for c in state['candidates'])
        generation = state['generation']
        advance(state, raw, prices, now=stamp, builder=minute_features)
        atomic_save(path, state)
        evaluated = state['archive'][-1]['candidates'] if generation != state['generation'] else state['candidates']
        return {'status': 'observed', 'mode': 'PAPER_ONLY', 'timeframe': 'minute1',
                'observations': state['observations'], 'generation': state['generation'],
                'last_observation': state['last_observation'], 'coverage': state['coverage'],
                'new_trades': sum(len(c['base']['events']) for c in evaluated)-before,
                'incumbent': state['incumbent']}


def research(folder, bars=5000):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'report.json').exists():
        raise ValueError('Preserve results: use a fresh research directory')
    boundary = pd.Timestamp.now(tz='UTC').floor('min')
    configs = initial_population()
    atomic_save(folder/'protocol.json', {'timeframe': 'minute1', 'end': str(boundary), 'bars': bars,
                                        'configs': [asdict(c) for c in configs], 'costs': '10 bps base / 30 bps stress'})
    raw = {}
    for market in MARKETS:
        raw[market] = fetch_minutes(market, end=boundary, count=bars, cache=folder/f'{market}.csv')
        print(f'Loaded {market}: {len(raw[market])} one-minute bars', flush=True)
    validate_frames(raw)
    split = int(bars*.65)
    if split <= 300 or bars-split < 100:
        raise ValueError('Insufficient training/test bars')
    ranked = []
    for i, config in enumerate(configs):
        train = {m: f.iloc[:split] for m, f in raw.items()}
        result = replay(train, config, start=250, end=split, features=minute_features(train, config))
        score = result['return_pct'] - abs(result['max_drawdown_pct'])
        if result['closed_trades'] < 5:
            score = min(score, -1.)
        ranked.append({'candidate': i, 'score': score, **summary(result)})
        if (i+1) % 6 == 0:
            print(f'Evaluated {i+1}/{len(configs)} training candidates', flush=True)
    ranked.sort(key=lambda r: (-r['score'], r['candidate']))
    # Always retain the top research candidate's test, even if selection favors cash.
    chosen = configs[ranked[0]['candidate']]
    result = replay(raw, chosen, start=split, end=bars, features=minute_features(raw, chosen))
    stress_config = replace(chosen, slippage_bps=30.)
    stress = replay(raw, stress_config, start=split, end=bars, features=minute_features(raw, chosen))
    report = {'mode': 'PAPER_ONLY', 'timeframe': 'minute1', 'status': 'RESEARCH_ONLY',
              'bars': bars, 'start': str(raw[MARKETS[0]].index[0]), 'end': str(boundary),
              'test_start': str(raw[MARKETS[0]].index[split]), 'training_ranking': ranked,
              'selected_for_research': asdict(chosen), 'selection_favors_cash': ranked[0]['score'] <= 0,
              'candidate_test': summary(result), 'candidate_stress': summary(stress),
              'limitations': ['Only a few days: many trades are not independent market regimes.',
                              'Synthetic no-trade minutes carry last price; zero-volume entries disabled.',
                              'Next-minute-open fills with costs; not order-book execution.',
                              'Historical test results do not count as new forward observations.']}
    atomic_save(folder/'report.json', report)
    result['equity'].to_csv(folder/'test-equity.csv', index_label='timestamp')
    print(json.dumps({k:report[k] for k in ('selection_favors_cash', 'candidate_test', 'candidate_stress')}, indent=2))


def watch(folder):
    # A separate OS lock prevents duplicate long-running workers.
    with state_lock(folder/'worker-guard.json'):
        heartbeat = folder/'worker.json'
        while True:
            started = pd.Timestamp.now(tz='UTC')
            try:
                result = observe_once(folder)
                atomic_save(heartbeat, {'pid': os.getpid(), 'checked_at': str(pd.Timestamp.now(tz='UTC')),
                                        'status': 'running', 'result': result})
                print(json.dumps(result), flush=True)
            except Exception as exc:
                atomic_save(heartbeat, {'pid': os.getpid(), 'checked_at': str(pd.Timestamp.now(tz='UTC')),
                                        'status': 'error', 'error': str(exc)})
                print(f'Observation failed: {exc}', flush=True)
            # Start shortly after a minute boundary. Do not replay missed quotes.
            now = pd.Timestamp.now(tz='UTC')
            wait = (now.ceil('min')+pd.Timedelta(seconds=3)-now).total_seconds()
            time.sleep(max(3., wait))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('.runtime/minute-learning'))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--research', action='store_true')
    parser.add_argument('--bars', type=int, default=5000)
    args = parser.parse_args()
    if args.research:
        research(args.folder, args.bars)
    elif args.watch:
        watch(args.folder)
    else:
        print(json.dumps(observe_once(args.folder), indent=2))


if __name__ == '__main__':
    main()

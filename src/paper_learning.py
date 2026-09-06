"""Prospective paper tournament with bounded evolutionary parameter search.

No model training, account access, real orders, or edits to live settings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import random

import pandas as pd

from src.adaptive_portfolio import PortfolioConfig, PortfolioSimulator, build_features, validate_frames
from src.portfolio_paper import atomic_save, fetch_quotes, state_lock
from src.portfolio_research import candidate_configs
from src.strategy_validation import fetch_daily


def initial_population():
    base = candidate_configs(expanded=True)
    # The user authorized aggressive virtual experiments. No leverage or fake cash.
    aggressive = [replace(c, risk_fraction=.03, asset_fraction=1., exposure_fraction=1.,
                          max_positions=3, drawdown_limit=1., daily_loss_limit=1.) for c in base[:8]]
    return base + aggressive


def config_id(config):
    values = asdict(config)
    # Preserve identifiers for cohorts created before structural research existed.
    for key, default in {'context_minutes': 1, 'edge_cost_ratio': 0., 'exit_confirm_bars': 1, 'min_stop_fraction': 0.}.items():
        if values[key] == default:
            values.pop(key)
    return sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()[:12]


def _candidate(config, cash):
    return {'id': config_id(config), 'config': asdict(config),
            'base': PortfolioSimulator(config, initial_cash=cash).to_state(),
            'stress': PortfolioSimulator(replace(config, slippage_bps=max(30., config.slippage_bps*3)),
                                         initial_cash=cash).to_state(),
            'drawdown_pct': 0.}


def new_learning_state(configs=None, *, now, cash=100000., timeframe='day'):
    if timeframe not in {'day', 'minute1'}:
        raise ValueError('Unsupported timeframe')
    return {'version': 1, 'mode': 'PAPER_ONLY', 'generation': 0, 'cash_per_candidate': cash,
            'timeframe': timeframe,
            'cohort_started_at': str(pd.Timestamp(now)), 'last_observation': None,
            'last_bucket': None, 'observations': 0, 'observed_days': [],
            'candidates': [_candidate(c, cash) for c in (configs if configs is not None else initial_population())],
            'incumbent': None, 'promotions': [], 'archive': [], 'rankings': [], 'feature_day': None,
            'features': {}, 'markets': ['KRW-BTC', 'KRW-ETH', 'KRW-XRP']}


def promotion_decision(rows, *, incumbent, days, observed_days, coverage):
    if days < 30 or observed_days < 25 or coverage < .8:
        return None
    incumbent_score = next((r['score'] for r in rows if r['id'] == incumbent), 0.)
    eligible = [r for r in rows if r['return_pct'] > 0 and r['stress_return_pct'] > 0
                and r['closed_trades'] >= 10 and r['drawdown_pct'] >= -20 and not r['halted']
                and r['score'] > max(0., incumbent_score) + 1.]
    return max(eligible, key=lambda r: (r['score'], r['id']))['id'] if eligible else None


def _rank(item, initial):
    base, stress = item['base'], item['stress']
    ratio = base['last_equity'] / initial
    return {'id': item['id'], 'return_pct': (ratio-1)*100,
            'stress_return_pct': (stress['last_equity']/initial-1)*100,
            'drawdown_pct': item['drawdown_pct'],
            'score': math.log(max(ratio, 1e-12))*100 - abs(item['drawdown_pct'])*.5,
            'closed_trades': sum(e['side'] == 'SELL' for e in base['events']),
            'halted': base['halted'], 'equity': base['last_equity'],
            'open_positions': len(base['positions'])}


def _next_population(state):
    lookup = {c['id']: PortfolioConfig(**c['config']) for c in state['candidates']}
    survivors = [lookup[r['id']] for r in state['rankings'][:4]]
    if state['incumbent'] and state['incumbent'] in lookup:
        survivors.insert(0, lookup[state['incumbent']])
    rng = random.Random(90210 + state['generation'])
    configs = {config_id(c): c for c in survivors}
    if state.get('timeframe') == 'minute1':
        for item in state.get('research_candidates', [])[:4]:
            candidate = PortfolioConfig(**item['config'])
            configs[config_id(candidate)] = candidate
    # Change parameters, never mutate source code or reuse past returns as new evidence.
    for _ in range(200):
        if len(configs) >= 24:
            break
        parent = rng.choice(survivors)
        child = replace(parent,
                        entry_mode=rng.choice(['breakout', 'trend', 'pullback']),
                        trend_window=max(30, min(240, parent.trend_window + rng.choice([-20, 0, 20]))),
                        breakout_window=max(5, min(60, parent.breakout_window + rng.choice([-5, 0, 5]))),
                        stop_atr=max(1.5, min(6., parent.stop_atr + rng.choice([-.5, 0, .5]))),
                        risk_fraction=rng.choice([.0075, .015, .03, .05]),
                        exposure_fraction=rng.choice([.6, .8, 1.]))
        configs[config_id(child)] = child
    return list(configs.values())[:24]


def advance(state, raw, prices, *, now, builder=build_features):
    if state.get('mode') != 'PAPER_ONLY' or state.get('version') != 1:
        raise ValueError('Unsupported learning state')
    stamp = pd.Timestamp(now)
    minute = state.get('timeframe') == 'minute1'
    bucket = str(stamp.floor('min' if minute else 'h'))
    if state['last_bucket'] == bucket:
        return False
    if state['last_observation'] and stamp <= pd.Timestamp(state['last_observation']):
        raise ValueError('Old observation')
    day = str(stamp.date())
    feature_key = bucket if minute else day
    if state['feature_day'] != feature_key:
        features = {}
        for item in state['candidates']:
            frames = builder(raw, PortfolioConfig(**item['config']))
            features[item['id']] = {m: {'buy': bool(f.iloc[-1]['buy']), 'sell': bool(f.iloc[-1]['sell']),
                                       **{k: float(f.iloc[-1][k]) for k in ('atr', 'high', 'score')}}
                                   for m, f in frames.items()}
        state['features'] = features
        state['feature_day'] = feature_key
    for item in state['candidates']:
        for variant in ('base', 'stress'):
            sim = PortfolioSimulator.from_state(item[variant])
            count = len(sim.events)
            sim.quote(prices, day=day, signals=state['features'][item['id']],
                      signal_key=bucket if minute else day)
            for event in sim.events[count:]:
                event['ts'] = stamp.timestamp()
                if event['side'] == 'BUY':
                    sim.trader.get_position(event['market']).opened_at = stamp.timestamp()
            item[variant] = sim.to_state()
        base = item['base']
        item['drawdown_pct'] = min(item['drawdown_pct'], (base['last_equity']/base['peak']-1)*100)
    state['last_bucket'] = bucket
    state['last_observation'] = str(stamp)
    state['observations'] += 1
    if day not in state['observed_days']:
        state['observed_days'].append(day)
    state['rankings'] = sorted([_rank(c, state['cash_per_candidate']) for c in state['candidates']],
                               key=lambda r: (-r['score'], r['id']))
    elapsed = (stamp-pd.Timestamp(state['cohort_started_at'])).total_seconds()
    state['coverage'] = min(1., state['observations']/max(1., elapsed/(60 if minute else 3600)+1))
    mature = elapsed >= 30*86400 and len(state['observed_days']) >= 25 and state['coverage'] >= .8
    if not mature:
        return True
    selected = promotion_decision(state['rankings'], incumbent=state['incumbent'], days=elapsed/86400,
                                  observed_days=len(state['observed_days']), coverage=state['coverage'])
    if selected:
        state['promotions'].append({'at': str(stamp), 'from': state['incumbent'], 'to': selected,
                                    'generation': state['generation']})
        state['incumbent'] = selected
    elif state['incumbent']:
        current = next(r for r in state['rankings'] if r['id'] == state['incumbent'])
        if current['score'] <= 0 or current['halted']:
            state['promotions'].append({'at': str(stamp), 'from': state['incumbent'], 'to': None,
                                        'generation': state['generation'], 'reason': 'cash_fallback'})
            state['incumbent'] = None
    state['archive'].append({'generation': state['generation'], 'started_at': state['cohort_started_at'],
                             'ended_at': str(stamp), 'observations': state['observations'],
                             'coverage': state['coverage'], 'rankings': state['rankings'],
                             'candidates': state['candidates']})
    configs = _next_population(state)
    state.setdefault('research_admissions', []).append({'generation': state['generation'] + 1,
                                                        'candidates': state.get('research_candidates', [])})
    state['research_candidates'] = []
    state['generation'] += 1
    state['candidates'] = [_candidate(c, state['cash_per_candidate']) for c in configs]
    state['cohort_started_at'] = str(stamp)
    state['observations'] = 0
    state['observed_days'] = []
    state['feature_day'] = None
    state['features'] = {}
    state['rankings'] = []
    return True


def run_once(folder):
    path = folder / 'state.json'
    with state_lock(path):
        now = pd.Timestamp.now(tz='UTC')
        state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else new_learning_state(now=now)
        if state['last_bucket'] == str(now.floor('h')):
            return {'status': 'already_observed', 'generation': state['generation']}
        today = now.floor('D')
        raw = {}
        if state['feature_day'] != str(today.date()):
            for market in state['markets']:
                raw[market] = fetch_daily(market, 2400, end=today)
                if raw[market].empty or raw[market].index[-1] != today-pd.Timedelta(days=1):
                    raise ValueError('Missing completed daily candles')
            validate_frames(raw)
        prices = fetch_quotes(state['markets'])
        observed = pd.Timestamp.now(tz='UTC')
        if observed.floor('D') != today:
            raise ValueError('UTC boundary crossed; retry')
        before = len(state['promotions'])
        generation_before = state['generation']
        trades_before = sum(len(c['base']['events']) for c in state['candidates'])
        halted_before = {c['id'] for c in state['candidates'] if c['base']['halted']}
        advance(state, raw, prices, now=observed)
        atomic_save(path, state)
        evaluated = state['archive'][-1]['candidates'] if state['generation'] != generation_before else state['candidates']
        return {'status': 'observed', 'mode': 'PAPER_ONLY', 'generation': state['generation'],
                'candidates': len(state['candidates']), 'observations': state['observations'],
                'incumbent': state['incumbent'], 'new_promotions': state['promotions'][before:],
                'new_trades': sum(len(c['base']['events']) for c in evaluated)-trades_before,
                'new_halts': [c['id'] for c in evaluated if c['base']['halted'] and c['id'] not in halted_before],
                'generation_changed': state['generation'] != generation_before,
                'coverage': state.get('coverage'), 'last_observation': state['last_observation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('.runtime/paper-learning'))
    args = parser.parse_args()
    print(json.dumps(run_once(args.folder), indent=2))


if __name__ == '__main__':
    main()

"""Reproducible evolutionary search using training evidence only, never live orders."""
from dataclasses import replace
import math
import random

import pandas as pd

from src.adaptive_portfolio import PortfolioConfig
from src.paper_learning import config_id


DOMAINS = {
    'entry_mode': ('breakout', 'trend', 'pullback'),
    'context_minutes': (1, 5, 15, 30),
    'trend_window': (40, 80, 120, 180, 240),
    'breakout_window': (5, 10, 20, 40, 60),
    'stop_atr': (2., 3., 4., 5.),
    'edge_cost_ratio': (0., .5, .75, 1., 1.5, 2.),
    'exit_confirm_bars': (1, 2, 3, 5),
    'min_stop_fraction': (0., .003, .006, .009, .012),
}


def fixed_configs():
    base = PortfolioConfig()
    return [base] + [replace(base, context_minutes=context, entry_mode=mode,
                            edge_cost_ratio=1.5, exit_confirm_bars=3, min_stop_fraction=stop)
                     for context in (5, 15) for mode in ('breakout', 'trend', 'pullback')
                     for stop in (.006, .012)]


def propose(reports, *, seed, train_end):
    """13 controls, up to 6 local mutations and at least 5 independent explorations.

    Previous holdout results are deliberately ignored. Previous training must end
    before this experiment's holdout starts. Historical reuse is development only.
    """
    rng = random.Random(seed)
    cutoff = pd.Timestamp(train_end)
    parents = []
    seen = set()
    for report in sorted(reports, key=lambda r: r['run_id'], reverse=True):
        if pd.Timestamp(report['test_start']) > cutoff:
            continue
        rows = sorted(report['training_ranking'], key=lambda r: -r['score'])
        for row in rows:
            if row['train']['closed_trades'] < 5 or not math.isfinite(row['score']):
                continue
            # Only signal fields are inherited. Capital and loss limits are fixed.
            config = replace(PortfolioConfig(), **{key: row['config'][key] for key in DOMAINS
                                                   if key in row['config']})
            identifier = config_id(config)
            if identifier not in seen:
                parents.append((config, report['run_id']))
                seen.add(identifier)
            if len(parents) == 3:
                break
        if len(parents) == 3:
            break
    configs = fixed_configs()
    provenance = [{'id': config_id(c), 'origin': 'control'} for c in configs]
    identifiers = {config_id(c) for c in configs}
    for _ in range(1000):
        if len(configs) == 24:
            break
        if parents and len(configs) < 19:
            parent, run_id = rng.choice(parents)
            keys = rng.sample(list(DOMAINS), 2)
            child = replace(parent, **{key: rng.choice(DOMAINS[key]) for key in keys})
            source = {'origin': 'mutation', 'parent_id': config_id(parent), 'training_run': run_id}
        else:
            child = replace(PortfolioConfig(), **{key: rng.choice(values) for key, values in DOMAINS.items()})
            source = {'origin': 'exploration'}
        identifier = config_id(child)
        if identifier not in identifiers:
            configs.append(child)
            identifiers.add(identifier)
            provenance.append({'id': identifier, **source})
    if len(configs) != 24:
        raise ValueError('Could not generate unique research candidates')
    return configs, provenance

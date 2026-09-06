"""Daily, bounded structural strategy experiments; nominations are paper-only."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import pandas as pd

from src.adaptive_portfolio import PortfolioConfig, replay, validate_frames
from src.minute_learning import MARKETS, fetch_minutes, minute_features
from src.paper_learning import config_id
from src.portfolio_paper import atomic_save, state_lock
from src.portfolio_research import summary
from src.research_search import fixed_configs, propose


def research_configs():
    return fixed_configs()


def eligible(train, test, stress):
    return (train['return_pct'] > 0 and train['closed_trades'] >= 5 and
            test['return_pct'] > 0 and test['closed_trades'] >= 5 and
            stress['return_pct'] > 0 and test['max_drawdown_pct'] >= -10)


def evaluate(raw, configs):
    index = validate_frames(raw)
    split = int(len(index) * .65)
    if split < 650 or len(index) - split < 200:
        raise ValueError('Insufficient research history')
    train = {m: frame.iloc[:split].copy() for m, frame in raw.items()}
    ranking = []
    for config in configs:
        result = summary(replay(train, config, start=400, end=split, features=minute_features(train, config)))
        score = result['return_pct'] - abs(result['max_drawdown_pct'])
        if result['closed_trades'] < 5:
            score = min(-1., score)
        ranking.append({'id': config_id(config), 'config': asdict(config), 'score': score, 'train': result})
    ranking.sort(key=lambda row: (-row['score'], row['id']))
    # Fix one winner using training only, before any validation results exist.
    chosen = PortfolioConfig(**ranking[0]['config'])
    test = summary(replay(raw, chosen, start=split, end=len(index), features=minute_features(raw, chosen)))
    stressed = replace(chosen, slippage_bps=30.)
    # Hold signals fixed: stress changes execution costs, not the tested policy.
    stress = summary(replay(raw, stressed, start=split, end=len(index), features=minute_features(raw, chosen)))
    nominated = eligible(ranking[0]['train'], test, stress)
    return {'mode': 'PAPER_ONLY', 'status': 'FORWARD_CANDIDATE' if nominated else 'RESEARCH_ONLY',
            'start': str(index[0]), 'end': str(index[-1]), 'test_start': str(index[split]),
            'training_ranking': ranking, 'selected_config': asdict(chosen), 'test': test, 'stress': stress,
            'nominees': [{'id': config_id(chosen), 'config': asdict(chosen)}] if nominated else [],
            'limitations': ['Repeated daily windows overlap; validation is not independent evidence.',
                            'Range/cost filter does not predict profit; candle fills omit order-book liquidity.',
                            'Only future paper observations qualify for cohort promotion; no live deployment.']}


def run_cycle(folder, paper_folder, *, bars=10000, now=None):
    stamp = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    boundary = stamp.floor('min')
    day = str(stamp.date())
    run_folder = folder / day
    with state_lock(folder / 'research-guard.json'):
        report_path = run_folder / 'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding='utf-8'))
        else:
            protocol_path = run_folder / 'protocol.json'
            if protocol_path.exists():
                protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
                boundary, bars = pd.Timestamp(protocol['boundary']), protocol['bars']
                configs = [PortfolioConfig(**c) for c in protocol['configs']]
            else:
                # The holdout boundary is known from the fixed, regular minute grid.
                train_end = boundary - pd.Timedelta(minutes=bars-int(bars*.65))
                reports = [json.loads(path.read_text(encoding='utf-8'))
                           for path in sorted(folder.glob('*/report.json'))[-14:]
                           if path.parent.name < day]
                configs, provenance = propose(reports, seed=int(stamp.strftime('%Y%m%d')), train_end=train_end)
                protocol = {'boundary': str(boundary), 'bars': bars,
                            'configs': [asdict(c) for c in configs],
                            'search_version': 1, 'search_provenance': provenance,
                            'selection': '65% train; one winner; 35% validation; no retuning'}
                atomic_save(protocol_path, protocol)
            raw = {m: fetch_minutes(m, end=boundary, count=bars, cache=run_folder / f'{m}.csv') for m in MARKETS}
            print(f'Research {day}: {bars} bars, {len(configs)} candidates', flush=True)
            report = {'run_id': day, 'search_version': protocol.get('search_version', 0),
                      'search_provenance': protocol.get('search_provenance', []), **evaluate(raw, configs)}
            atomic_save(report_path, report)
        # Re-publishing after a crash is safe; consumers remember the run ID.
        atomic_save(folder / 'latest.json', report)
        atomic_save(paper_folder / 'research-candidates.json',
                    {'run_id': day, 'source': str(report_path.resolve()), 'nominees': report['nominees']})
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path('.runtime/strategy-research'))
    parser.add_argument('--paper-folder', type=Path, default=Path('.runtime/minute-learning'))
    parser.add_argument('--bars', type=int, default=10000)
    args = parser.parse_args()
    report = run_cycle(args.folder, args.paper_folder, bars=args.bars)
    print(json.dumps({k: report[k] for k in ('run_id', 'status', 'test', 'stress', 'nominees')}, indent=2))


if __name__ == '__main__':
    main()

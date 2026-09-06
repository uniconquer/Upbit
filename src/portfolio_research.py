"""Predeclared walk-forward experiments for the paper-only adaptive portfolio."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from src.adaptive_portfolio import PortfolioConfig, build_features, replay, validate_frames
from src.strategy_validation import fetch_daily


def candidate_configs(*, expanded=False):
    configs = [PortfolioConfig(trend_window=trend, breakout_window=breakout, stop_atr=stop)
            for trend in (60, 120) for breakout in (10, 20) for stop in (2.5, 3.5)]
    if expanded:
        configs.extend(PortfolioConfig(entry_mode=mode, trend_window=trend, stop_atr=stop)
                       for mode in ('trend', 'pullback') for trend in (60, 120) for stop in (2.5, 3.5))
    return configs


def summary(result):
    return {k: v for k, v in result.items() if k not in {'equity', 'state'}}


def select_candidate(raw, *, start, end, initial_cash, configs):
    # Prefix truncation makes it impossible for selection to access test candles.
    train_raw = {m: f.iloc[:end].copy() for m, f in raw.items()}
    ranked = []
    for i, config in enumerate(configs):
        result = replay(train_raw, config, start=start, end=end, initial_cash=initial_cash)
        score = result['return_pct'] - abs(result['max_drawdown_pct'])
        if result['closed_trades'] < 5:
            score = min(score, -1.)
        ranked.append({'candidate': i, 'score': score, **summary(result)})
    ranked.sort(key=lambda r: (-r['score'], r['candidate']))
    best = ranked[0]
    return (best['candidate'] if best['score'] > 0 else None), ranked


def cash_result(raw, start, end, cash):
    index = next(iter(raw.values())).index[start:end]
    return {'final_equity': cash, 'return_pct': 0., 'max_drawdown_pct': 0.,
            'closed_trades': 0, 'buy_trades': 0, 'halted': False,
            'equity': pd.Series(cash, index=index)}


def forward_profile(raw, report, *, train_bars=540):
    """Choose today's future-paper profile, separately from retrospective scoring."""
    index = validate_frames(raw)
    configs = [PortfolioConfig(**c) for c in report['configs']]
    selected, ranking = select_candidate(raw, start=len(index)-train_bars, end=len(index),
                                         initial_cash=report['initial_cash'], configs=configs)
    return {'mode': 'PAPER_ONLY', 'status': report['status'], 'markets': list(raw),
            'initial_cash': report['initial_cash'], 'history_bars': len(index),
            'selected_config': asdict(configs[selected]) if selected is not None else None,
            'selection_asof': str(index[-1]), 'ranking': ranking,
            'note': 'Frozen forward-paper profile; its performance is not included in retrospective metrics.'}


def run_experiment(raw, *, initial_cash=100000., train_bars=540, fold_bars=180,
                   final_bars=400, configs=None, progress=None):
    index = validate_frames(raw)
    configs = configs if configs is not None else candidate_configs()
    if not configs or min(train_bars, fold_bars, final_bars) <= 0:
        raise ValueError('Positive experiment windows and nonempty candidates required')
    warmup = max(max(c.macro_window, c.trend_window, 120, c.breakout_window) for c in configs) + 10
    cutoff = len(index) - final_bars
    if cutoff < warmup + train_bars + fold_bars:
        raise ValueError('Not enough history for walk-forward and final periods')
    folds = []
    curves = []
    cash = initial_cash
    # Align complete folds with the final cutoff. Leftover oldest bars are warmup.
    first = cutoff - ((cutoff - warmup - train_bars) // fold_bars) * fold_bars
    for split in range(first, cutoff, fold_bars):
        selected, ranked = select_candidate(raw, start=split-train_bars, end=split,
                                           initial_cash=initial_cash, configs=configs)
        end = min(split + fold_bars, cutoff)
        result = (cash_result(raw, split, end, cash) if selected is None else
                  replay(raw, configs[selected], start=split, end=end, initial_cash=cash))
        # Reset/rotation at a fold boundary is valued at net liquidation costs.
        cash = result['final_equity']
        curves.append(result['equity'])
        fold = {'start': str(index[split]), 'end': str(index[end-1]),
                'selected_candidate': selected, 'train_ranking': ranked, **summary(result)}
        folds.append(fold)
        if progress:
            progress(f"Fold {len(folds)}: candidate={selected}, return={result['return_pct']:.2f}%, trades={result['closed_trades']}")
    selected, ranking = select_candidate(raw, start=cutoff-train_bars, end=cutoff,
                                        initial_cash=initial_cash, configs=configs)
    # The final results below never feed back into selection or parameter edits.
    final = (cash_result(raw, cutoff, len(index), initial_cash) if selected is None else
             replay(raw, configs[selected], start=cutoff, end=len(index), initial_cash=initial_cash))
    stress = (cash_result(raw, cutoff, len(index), initial_cash) if selected is None else
              replay(raw, replace(configs[selected], slippage_bps=30.), start=cutoff,
                     end=len(index), initial_cash=initial_cash))
    baseline_config = PortfolioConfig()
    baseline = replay(raw, baseline_config, start=cutoff, end=len(index), initial_cash=initial_cash)
    equity = pd.concat(curves)
    peaks = equity.cummax().clip(lower=initial_cash)
    walk = {'return_pct': float((equity.iloc[-1]/initial_cash-1)*100),
            'max_drawdown_pct': float(((equity/peaks-1)*100).min()),
            'closed_trades': sum(f['closed_trades'] for f in folds),
            'positive_folds': sum(f['return_pct'] > 0 for f in folds), 'folds': len(folds)}
    reasons = []
    if selected is None:
        reasons.append('Selection window favors cash.')
    if walk['return_pct'] <= 0:
        reasons.append('Walk-forward return is not positive.')
    if walk['positive_folds'] < (len(folds)+1)//2:
        reasons.append('Fewer than half of development folds are profitable.')
    if final['return_pct'] <= 0 or stress['return_pct'] <= 0:
        reasons.append('Final return is not positive under both cost assumptions.')
    if final['closed_trades'] < 20:
        reasons.append('Fewer than 20 completed final-period trades.')
    if final['max_drawdown_pct'] < -15 or final.get('halted'):
        reasons.append('Final-period drawdown or halt requires review.')
    return {
        'mode': 'PAPER_ONLY', 'status': 'FORWARD_PAPER_CANDIDATE' if not reasons else 'RESEARCH_ONLY',
        'reasons': reasons, 'initial_cash': initial_cash, 'markets': list(raw), 'history_bars': len(index),
        'configs': [asdict(c) for c in configs], 'fold_results': folds,
        'walk_forward': walk, 'final_selection_ranking': ranking,
        'selected_candidate': selected,
        'selected_config': asdict(configs[selected]) if selected is not None else None,
        'final_start': str(index[cutoff]), 'final_end': str(index[-1]),
        'final': summary(final), 'stress': summary(stress), 'fixed_default_baseline': summary(baseline),
        'limitations': [
            'The final dates overlap the previously inspected experiment; this is retrospective, not untouched evidence.',
            'Three surviving liquid markets do not represent all Upbit assets or delisted coins.',
            'Daily OHLC cannot reproduce liquidity, actual stop execution, or asynchronous intraday paths.',
            'Risk budgets and the drawdown halt are triggers, not guaranteed loss ceilings.',
            'Walk-forward folds reset positions at estimated net liquidation value; reset may ignore minimum-order dust.',
            'Each fold has a fresh risk peak. A forward session instead preserves its halt across restarts.',
            'Research thresholds are heuristics. Future paper evidence and separate live review are still required.',
        ],
    }, equity, final.get('state')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', type=int, default=2400)
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--expanded', action='store_true', help='Also compare trend participation and pullback entries')
    parser.add_argument('--output', type=Path, default=Path('.runtime/portfolio-research'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_path = args.output / 'protocol.json'
    if protocol_path.exists():
        parser.error('Choose a fresh output directory to preserve the experiment record')
    protocol = {'markets': ['KRW-BTC', 'KRW-ETH', 'KRW-XRP'], 'bars': args.bars,
                'train_bars': 540, 'fold_bars': 180, 'final_bars': 400,
                'initial_cash': 100000, 'configs': [asdict(c) for c in candidate_configs(expanded=args.expanded)],
                'created_at': str(pd.Timestamp.now(tz='UTC'))}
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    raw = {}
    for market in protocol['markets']:
        if args.input_dir:
            frame = pd.read_csv(args.input_dir / f'{market}.csv', index_col='timestamp')
            frame.index = pd.to_datetime(frame.index, utc=True)
            raw[market] = frame.tail(args.bars)
        else:
            raw[market] = fetch_daily(market, args.bars, end=pd.Timestamp.now(tz='UTC').floor('D'))
        print(f'Loaded {market}: {len(raw[market])} bars', flush=True)
    validate_frames(raw)
    hashes = {}
    for market, frame in raw.items():
        payload = frame.to_csv(index_label='timestamp')
        (args.output / f'{market}.csv').write_text(payload, encoding='utf-8')
        hashes[market] = sha256(payload.encode()).hexdigest()
    report, equity, state = run_experiment(raw, configs=candidate_configs(expanded=args.expanded),
                                         progress=lambda message: print(message, flush=True))
    report['data_sha256'] = hashes
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    profile = forward_profile(raw, report)
    (args.output / 'paper-profile.json').write_text(json.dumps(profile, indent=2), encoding='utf-8')
    equity.to_csv(args.output / 'walk-forward-equity.csv', index_label='timestamp')
    if state:
        (args.output / 'final-replay-state.json').write_text(json.dumps(state, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'reasons', 'selected_config', 'walk_forward', 'final', 'stress')}, indent=2))


if __name__ == '__main__':
    main()

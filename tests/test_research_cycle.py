from dataclasses import asdict
from hashlib import sha256
import json

import pandas as pd

from src import strategy_research_cycle as cycle
from src.adaptive_portfolio import PortfolioConfig
from src.paper_learning import _next_population, config_id, new_learning_state


def test_old_candidate_identifiers_survive_new_defaults():
    config = PortfolioConfig()
    old = asdict(config)
    for name in ('context_minutes', 'edge_cost_ratio', 'exit_confirm_bars', 'min_stop_fraction'):
        old.pop(name)
    assert config_id(config) == sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()[:12]


def test_nomination_requires_actual_profitable_trades():
    good = {'return_pct': 2., 'closed_trades': 5, 'max_drawdown_pct': -2.}
    assert cycle.eligible(good, good, good)
    assert not cycle.eligible(good, {**good, 'closed_trades': 0}, good)
    assert not cycle.eligible(good, good, {**good, 'return_pct': -1.})


def test_research_only_enters_next_population():
    state = new_learning_state(now=pd.Timestamp('2026-01-01', tz='UTC'), timeframe='minute1')
    old = json.dumps(state['candidates'], sort_keys=True)
    nominee = PortfolioConfig(context_minutes=15, min_stop_fraction=.006)
    state['research_candidates'] = [{'id': config_id(nominee), 'config': asdict(nominee)}]
    state['rankings'] = [{'id': c['id']} for c in state['candidates']]
    population = _next_population(state)
    assert nominee in population
    assert len(population) == 24
    assert json.dumps(state['candidates'], sort_keys=True) == old


def test_daily_cycle_reuses_completed_result(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cycle, 'fetch_minutes', lambda *args, **kwargs: calls.append(args) or None)
    monkeypatch.setattr(cycle, 'evaluate', lambda raw, configs: {'status': 'RESEARCH_ONLY', 'nominees': []})
    first = cycle.run_cycle(tmp_path/'research', tmp_path/'paper', now=pd.Timestamp('2026-01-01T01:00Z'))
    again = cycle.run_cycle(tmp_path/'research', tmp_path/'paper', now=pd.Timestamp('2026-01-01T20:00Z'))
    assert first == again
    assert len(calls) == 3


def test_selection_receives_only_training_prefix(monkeypatch):
    index = pd.date_range('2026-01-01', periods=1200, freq='min', tz='UTC')
    raw = {'KRW-BTC': pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.}, index=index)}
    seen = []
    monkeypatch.setattr(cycle, 'minute_features', lambda frames, config: None)
    def fake_replay(frames, config, **kwargs):
        seen.append((len(frames['KRW-BTC']), kwargs['start']))
        return {'return_pct': -1., 'max_drawdown_pct': -1., 'closed_trades': 5}
    monkeypatch.setattr(cycle, 'replay', fake_replay)
    report = cycle.evaluate(raw, [PortfolioConfig(), PortfolioConfig(context_minutes=15)])
    assert seen == [(780, 400), (780, 400), (1200, 780), (1200, 780)]
    assert report['nominees'] == []

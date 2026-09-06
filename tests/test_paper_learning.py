import copy
from dataclasses import asdict

import pandas as pd

from src.adaptive_portfolio import PortfolioConfig
from src.paper_learning import new_learning_state, advance, initial_population, promotion_decision


def signal_frames():
    return {'KRW-BTC': pd.DataFrame({'buy': [True], 'sell': [False], 'atr': [2.],
                                    'high': [100.], 'score': [1.]})}


def test_population_contains_full_allocation_experiments():
    population = initial_population()
    assert len(population) == 24
    assert any(c.exposure_fraction == 1 and c.asset_fraction == 1 for c in population)


def test_duplicate_observation_is_idempotent_and_restart_preserves_losses():
    cfg = PortfolioConfig()
    state = new_learning_state([cfg], now='2026-01-01T01:00:00Z')
    builder = lambda raw, config: signal_frames()
    advance(state, {}, {'KRW-BTC': 100.}, now='2026-01-01T01:00:00Z', builder=builder)
    snapshot = copy.deepcopy(state)
    advance(state, {}, {'KRW-BTC': 100.}, now='2026-01-01T01:20:00Z', builder=builder)
    assert state == snapshot
    advance(state, {}, {'KRW-BTC': 70.}, now='2026-01-01T02:00:00Z', builder=builder)
    assert state['candidates'][0]['base']['last_equity'] < 100000
    restored = copy.deepcopy(state)
    advance(restored, {}, {'KRW-BTC': 70.}, now='2026-01-01T03:00:00Z', builder=builder)
    assert restored['candidates'][0]['base']['cash'] == state['candidates'][0]['base']['cash']


def row(**kw):
    result = {'id': 'candidate', 'score': 8., 'return_pct': 10., 'stress_return_pct': 9.,
              'drawdown_pct': -2., 'closed_trades': 12, 'halted': False}
    result.update(kw)
    return result


def test_promotion_requires_new_time_coverage_trades_and_cost_robustness():
    assert promotion_decision([row()], incumbent=None, days=1, observed_days=1, coverage=.99) is None
    assert promotion_decision([row()], incumbent=None, days=30, observed_days=30, coverage=.1) is None
    assert promotion_decision([row(closed_trades=0)], incumbent=None, days=30, observed_days=30, coverage=.99) is None
    assert promotion_decision([row(stress_return_pct=-1)], incumbent=None, days=30, observed_days=30, coverage=.99) is None
    assert promotion_decision([row()], incumbent=None, days=30, observed_days=30, coverage=.99) == 'candidate'


def test_promotion_must_beat_incumbent_and_does_not_promote_ruin():
    rows = [row(id='incumbent', score=9.), row()]
    assert promotion_decision(rows, incumbent='incumbent', days=30, observed_days=30, coverage=.99) is None
    assert promotion_decision([row(drawdown_pct=-60)], incumbent=None, days=30, observed_days=30, coverage=.99) is None


def test_cash_is_incumbent_until_evidence_exists():
    state = new_learning_state([PortfolioConfig()], now='2026-01-01T00:00:00Z')
    advance(state, {}, {'KRW-BTC': 100.}, now='2026-01-01T01:00:00Z', builder=lambda *a: signal_frames())
    assert state['incumbent'] is None
    assert not state['promotions']
    assert state['mode'] == 'PAPER_ONLY'


def test_mature_round_archives_results_and_children_start_without_inherited_profit():
    started = pd.Timestamp('2026-01-01T00:00:00Z')
    state = new_learning_state([PortfolioConfig()], now=started)
    for hour in range(721):
        advance(state, {}, {'KRW-BTC': 100.}, now=started+pd.Timedelta(hours=hour),
                builder=lambda *a: signal_frames())
    assert state['generation'] == 1
    assert len(state['archive']) == 1
    assert state['archive'][0]['candidates'][0]['base']['last_equity'] < 100000
    assert all(c['base']['cash'] == 100000 and not c['base']['events'] for c in state['candidates'])
    assert state['incumbent'] is None
    assert len(state['candidates']) == 24

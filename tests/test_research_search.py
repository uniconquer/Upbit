from dataclasses import asdict

from src.adaptive_portfolio import PortfolioConfig
from src.paper_learning import config_id
from src.research_search import propose


def history(config, *, test_start='2026-09-04T00:00Z', trades=12):
    return [{'run_id': '2026-09-05', 'test_start': test_start,
             'training_ranking': [{'config': asdict(config), 'score': 2.,
                                   'train': {'closed_trades': trades}}],
             'test': {'return_pct': 99999.}}]


def test_reproducible_unique_search_with_fixed_risk():
    configs, provenance = propose([], seed=10, train_end='2026-09-05T00:00Z')
    again, again_provenance = propose([], seed=10, train_end='2026-09-05T00:00Z')
    assert configs == again and provenance == again_provenance
    assert len(configs) == len({config_id(c) for c in configs}) == 24
    base = PortfolioConfig()
    for config in configs:
        for key in ('risk_fraction', 'exposure_fraction', 'daily_loss_limit', 'drawdown_limit', 'fee', 'slippage_bps'):
            assert getattr(config, key) == getattr(base, key)
    assert any(row['origin'] == 'exploration' for row in provenance)


def test_search_learns_from_training_not_validation():
    past = history(PortfolioConfig(context_minutes=15, edge_cost_ratio=.75))
    configs, sources = propose(past, seed=20, train_end='2026-09-05T00:00Z')
    assert any(row.get('parent_id') == config_id(PortfolioConfig(context_minutes=15, edge_cost_ratio=.75)) for row in sources)
    past[0]['test']['return_pct'] = -99999.
    assert propose(past, seed=20, train_end='2026-09-05T00:00Z') == (configs, sources)


def test_future_training_and_inactive_candidates_not_used_as_parents():
    empty = propose([], seed=30, train_end='2026-09-05T00:00Z')
    assert propose(history(PortfolioConfig(), test_start='2026-09-06T00:00Z'), seed=30,
                   train_end='2026-09-05T00:00Z') == empty
    assert propose(history(PortfolioConfig(), trades=0), seed=30, train_end='2026-09-05T00:00Z') == empty


def test_new_seed_explores_different_candidates():
    assert propose([], seed=10, train_end='2026-09-05T00:00Z')[0] != propose([], seed=11, train_end='2026-09-05T00:00Z')[0]

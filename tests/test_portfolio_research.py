import pandas as pd

from src.adaptive_portfolio import PortfolioConfig
from src.portfolio_research import run_experiment, select_candidate


def data(n=1000):
    # Cycles give the trend strategies actual entries/exits, not a zero-trade fixture.
    import numpy as np
    close = 100 + np.arange(n)*.1 + 12*np.sin(np.arange(n)/20)
    return {'KRW-BTC': pd.DataFrame({'open': close, 'high': close+1, 'low': close-1,
                                    'close': close, 'volume': 1000},
                                   index=pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC'))}


def test_final_prices_never_change_development_selection():
    raw = data()
    configs = [PortfolioConfig(entry_mode='trend', trend_window=60)]
    first, _, _ = run_experiment(raw, train_bars=180, fold_bars=100, final_bars=200, configs=configs)
    changed = {m: f.copy() for m, f in raw.items()}
    changed['KRW-BTC'].iloc[800:, :4] *= .2
    second, _, _ = run_experiment(changed, train_bars=180, fold_bars=100, final_bars=200, configs=configs)
    assert first['fold_results'] == second['fold_results']
    assert first['final_selection_ranking'] == second['final_selection_ranking']
    assert first['selected_candidate'] == second['selected_candidate']


def test_selection_replay_receives_no_future_rows(monkeypatch):
    from src import portfolio_research
    original = portfolio_research.replay
    observed = []

    def spy(raw, config, **kwargs):
        observed.append(len(raw['KRW-BTC']))
        assert len(raw['KRW-BTC']) == kwargs['end']
        return original(raw, config, **kwargs)

    monkeypatch.setattr(portfolio_research, 'replay', spy)
    select_candidate(data(), start=300, end=600, initial_cash=100000,
                     configs=[PortfolioConfig()])
    assert observed == [600]

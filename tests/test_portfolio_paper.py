from dataclasses import asdict
import json

import pandas as pd
import pytest

from src import portfolio_paper as paper
from src.adaptive_portfolio import PortfolioConfig


def test_forward_observation_restart_and_no_duplicate_trade(tmp_path, monkeypatch):
    yesterday = pd.Timestamp.now(tz='UTC').floor('D') - pd.Timedelta(days=1)
    raw = pd.DataFrame({'open': [100.], 'high': [101.], 'low': [99.], 'close': [100.], 'volume': [1000.]},
                       index=pd.DatetimeIndex([yesterday]))
    monkeypatch.setattr(paper, 'fetch_daily', lambda *a, **kw: raw)
    monkeypatch.setattr(paper, 'build_features', lambda *a, **kw: {'KRW-BTC': pd.DataFrame(
        {'buy': [True], 'sell': [False], 'atr': [2.], 'high': [101.], 'score': [1.]}, index=raw.index)})
    monkeypatch.setattr(paper, 'fetch_quotes', lambda markets: {'KRW-BTC': 100.})
    report = {'markets': ['KRW-BTC'], 'selected_config': asdict(PortfolioConfig()),
              'initial_cash': 100000, 'status': 'RESEARCH_ONLY'}
    path = tmp_path / 'state.json'
    first = paper.observe_once(report, path)
    second = paper.observe_once(report, path)
    assert first['new_trades'] == 1
    assert second['new_trades'] == 0
    assert second['observations'] == 2
    saved = json.loads(path.read_text())
    assert saved['simulation']['mode'] == 'PAPER_ONLY'
    assert len(saved['simulation']['events']) == 1


def test_stale_quotes_are_rejected(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return [{'market': 'KRW-BTC', 'timestamp': 0, 'trade_price': 100}]

    monkeypatch.setattr(paper.requests, 'get', lambda *a, **kw: Response())
    with pytest.raises(ValueError, match='Stale'):
        paper.fetch_quotes(['KRW-BTC'])


def test_state_lock_prevents_concurrent_writer(tmp_path):
    path = tmp_path / 'state.json'
    with paper.state_lock(path):
        with pytest.raises(OSError):
            with paper.state_lock(path):
                pass
    with paper.state_lock(path):
        paper.atomic_save(path, {'safe': True})
    assert json.loads(path.read_text()) == {'safe': True}

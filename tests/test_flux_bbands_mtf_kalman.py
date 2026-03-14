from __future__ import annotations

import pandas as pd

from src.flux_bbands_mtf_kalman import _confirm_follow_up


def test_confirm_follow_up_triggers_within_window():
    index = pd.date_range("2026-01-01", periods=6, freq="15min")
    setup = pd.Series([False, True, False, False, False, False], index=index)
    confirm = pd.Series([False, False, False, True, False, False], index=index)

    result = _confirm_follow_up(setup, confirm, window=2)

    assert result.tolist() == [False, False, False, True, False, False]


def test_confirm_follow_up_expires_outside_window():
    index = pd.date_range("2026-01-01", periods=6, freq="15min")
    setup = pd.Series([False, True, False, False, False, False], index=index)
    confirm = pd.Series([False, False, False, True, False, False], index=index)

    result = _confirm_follow_up(setup, confirm, window=1)

    assert not result.any()

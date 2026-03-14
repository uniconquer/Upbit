from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.views.live_view import _resolve_live_scan_action


def test_live_scan_action_prefers_manual_refresh():
    action = _resolve_live_scan_action(
        working=False,
        changed=False,
        has_results=True,
        skip_cached_scan_once=True,
        refresh_requested=True,
    )

    assert action == "scan"


def test_live_scan_action_shows_cached_snapshot_once():
    action = _resolve_live_scan_action(
        working=False,
        changed=True,
        has_results=True,
        skip_cached_scan_once=True,
        refresh_requested=False,
    )

    assert action == "show_cached"


def test_live_scan_action_scans_when_no_cached_results():
    action = _resolve_live_scan_action(
        working=False,
        changed=False,
        has_results=False,
        skip_cached_scan_once=True,
        refresh_requested=False,
    )

    assert action == "scan"


def test_live_scan_action_does_nothing_while_worker_running():
    action = _resolve_live_scan_action(
        working=True,
        changed=True,
        has_results=False,
        skip_cached_scan_once=False,
        refresh_requested=True,
    )

    assert action == "none"

from __future__ import annotations

from src.runtime_store import load_runtime_state, runtime_backup_path, runtime_path, save_runtime_state


def test_load_runtime_state_uses_backup_when_primary_is_invalid(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    save_runtime_state("sample-state", {"value": 1})
    save_runtime_state("sample-state", {"value": 2})
    path = runtime_path("sample-state")
    backup_path = runtime_backup_path("sample-state")

    path.write_text("{broken", encoding="utf-8")
    assert backup_path.exists()

    loaded = load_runtime_state("sample-state", default={})

    assert loaded == {"value": 1}


def test_save_runtime_state_creates_backup_of_previous_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("UPBIT_RUNTIME_DIR", str(tmp_path))
    save_runtime_state("sample-state", {"value": 1})

    path = runtime_path("sample-state")
    backup_path = runtime_backup_path("sample-state")

    save_runtime_state("sample-state", {"value": 2})

    assert path.exists()
    assert backup_path.exists()
    assert load_runtime_state("sample-state", default={}) == {"value": 2}
    assert backup_path.read_text(encoding="utf-8")

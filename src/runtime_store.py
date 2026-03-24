"""Simple JSON runtime-state persistence for local strategy workflows."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


def runtime_dir() -> Path:
    custom = os.getenv("UPBIT_RUNTIME_DIR")
    if custom:
        return Path(custom)
    return Path(__file__).resolve().parent.parent / ".runtime"


def runtime_path(name: str) -> Path:
    safe_name = "".join(char if char.isalnum() or char in ("-", "_") else "-" for char in name).strip("-")
    safe_name = safe_name or "state"
    return runtime_dir() / f"{safe_name}.json"


def runtime_backup_path(name: str) -> Path:
    return runtime_path(name).with_suffix(".bak")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_state(name: str, default: Any | None = None) -> Any:
    path = runtime_path(name)
    backup_path = runtime_backup_path(name)
    fallback = {} if default is None else default
    for candidate in (path, backup_path):
        if not candidate.exists():
            continue
        try:
            return _load_json(candidate)
        except Exception:
            continue
    return fallback


def save_runtime_state(name: str, data: Mapping[str, Any]) -> Path:
    path = runtime_path(name)
    backup_path = runtime_backup_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            shutil.copyfile(path, backup_path)
        except Exception:
            pass
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(path)
    return path


def delete_runtime_state(name: str) -> None:
    path = runtime_path(name)
    if path.exists():
        path.unlink()
    backup_path = runtime_backup_path(name)
    if backup_path.exists():
        backup_path.unlink()

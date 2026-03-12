"""Simple JSON runtime-state persistence for local strategy workflows."""

from __future__ import annotations

import json
import os
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


def load_runtime_state(name: str, default: Any | None = None) -> Any:
    path = runtime_path(name)
    if not path.exists():
        return {} if default is None else default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {} if default is None else default


def save_runtime_state(name: str, data: Mapping[str, Any]) -> Path:
    path = runtime_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
    temp_path.replace(path)
    return path


def delete_runtime_state(name: str) -> None:
    path = runtime_path(name)
    if path.exists():
        path.unlink()

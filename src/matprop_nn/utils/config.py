"""Configuration helpers — load YAML, merge overrides, resolve paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def merge_configs(*cfgs: dict) -> dict:
    """Deep-merge dicts left-to-right (later values override earlier)."""
    result: dict = {}
    for cfg in cfgs:
        _deep_update(result, cfg)
    return result


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base

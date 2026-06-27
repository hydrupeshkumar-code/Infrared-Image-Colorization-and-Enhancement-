"""Tiny config helpers: load YAML into attribute-accessible dicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict that also supports attribute access and recursive wrapping."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, key: str, default: str | None = None) -> Path | None:
        value = self.get(key, default)
        return Path(value) if value is not None else None


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config(data)


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into ``base`` (in place) and return it."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

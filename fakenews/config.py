"""Reads configs/config.yaml and applies command-line overrides (--set section.key=value)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .utils import PROJECT_ROOT, FakeNewsError

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _parse_value(raw: str):
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise FakeNewsError(f"Override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise FakeNewsError(f"Unknown config section '{p}' in override '{item}'")
            node = node[p]
        if parts[-1] not in node:
            raise FakeNewsError(f"Unknown config key '{key}'. See configs/config.yaml for valid keys.")
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path=None, overrides: list[str] | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    if not p.exists():
        raise FakeNewsError(f"Config file not found: {p}")
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, overrides)
    # resolve paths relative to the project root
    for k in ("data_dir", "outputs_dir"):
        v = Path(cfg["paths"][k])
        cfg["paths"][k] = str(v if v.is_absolute() else PROJECT_ROOT / v)
    return cfg


def processed_dir(cfg, name: str) -> Path:
    return Path(cfg["paths"]["data_dir"]) / "processed" / name


def output_dir(cfg, name: str) -> Path:
    return Path(cfg["paths"]["outputs_dir"]) / name

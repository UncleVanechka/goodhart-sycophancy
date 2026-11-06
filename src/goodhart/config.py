"""Config loading (base + stage YAML, env expansion) and seeding."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict with ``override`` merged on top of ``base``, recursing into dicts."""
    out: dict = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def expand_env(obj: Any) -> Any:
    """Expand ``${VAR}`` in every string of a nested structure; a missing variable becomes ''."""
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {key: expand_env(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_env(value) for value in obj]
    return obj


def _read_yaml(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(loaded).__name__}")
    return loaded


def _resolve_base(stage_path: Path, base: str | Path) -> Path | None:
    """Find the base config: as given, then next to the stage config."""
    candidate = Path(base)
    if candidate.exists():
        return candidate
    sibling = stage_path.parent / candidate.name
    if sibling.exists():
        return sibling
    return None


def load_config(path: str | Path, base: str | Path | None = "configs/base.yaml") -> dict:
    """Load a stage config, deep-merge the base config under it and expand ``${ENV_VAR}``.

    ``base`` is looked up as given and then next to ``path``; when it is ``None`` or not
    found, the stage config is used alone. Missing environment variables expand to ''.
    """
    stage_path = Path(path)
    stage = _read_yaml(stage_path)
    merged: dict = {}
    if base is not None:
        base_path = _resolve_base(stage_path, base)
        if base_path is not None and base_path.resolve() != stage_path.resolve():
            merged = _read_yaml(base_path)
    merged = deep_merge(merged, stage)
    return expand_env(merged)


def set_seed(seed: int) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (when importable)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

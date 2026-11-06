"""File helpers: JSONL, result payloads, git commit, model slugs, .env loading."""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write dicts as JSONL, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json(path: str | Path, obj: Any) -> None:
    """Write ``obj`` as indented JSON, creating parent directories."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def git_commit() -> str | None:
    """Return ``git rev-parse HEAD`` or ``None`` outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def results_payload(config: dict, **fields: Any) -> dict:
    """Build the standard results dict: resolved config, git commit and stage fields."""
    return {"config": config, "git_commit": git_commit(), **fields}


def model_slug(model: str) -> str:
    """Turn a model id into a file-name-safe slug: ``Qwen/Qwen3-1.7B`` -> ``qwen3-1.7b``."""
    name = model.rstrip("/").split("/")[-1].lower()
    name = re.sub(r"[^a-z0-9.\-]+", "-", name)
    return name.strip("-") or "model"


def _parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_dotenv_if_present(path: str | Path = ".env") -> None:
    """Load ``.env`` into ``os.environ`` without overriding existing variables, if it exists."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except ImportError:
        pass
    for key, value in _parse_dotenv(env_path).items():
        os.environ.setdefault(key, value)

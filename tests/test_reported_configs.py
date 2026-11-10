"""The committed configs reproduce the runs behind the write-up.

Every result file stores the merged config it ran with. These tests rebuild that config from
``configs/`` the way the Makefile does (``make baseline``, ``make bon``, ``make grpo-all``) and
compare. Keys that only name locations on the machine that ran the job are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goodhart.config import load_config
from goodhart.experiments.train_grpo import expand_sweep, with_seed

RESULTS = Path("results")
LOCATION_KEYS = ("data", "out", "simulator", "paths")

# make grpo-all: (config, seeds)
GRPO_TARGETS = (
    ("configs/grpo.yaml", (42, 43, 44)),
    ("configs/mitigation.yaml", (42, 43, 44)),
    ("configs/persona_mix.yaml", (42, 43)),
)


def _strip(cfg: dict, extra: tuple[str, ...] = ()) -> dict:
    return {key: value for key, value in cfg.items() if key not in LOCATION_KEYS + extra}


def _make_grpo_runs() -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for path, seeds in GRPO_TARGETS:
        base = load_config(path)
        for seed in seeds:
            for run_name, run_cfg in expand_sweep(with_seed(base, seed)):
                assert run_name not in runs, f"{run_name} produced twice"
                runs[run_name] = run_cfg
    return runs


def _stored(pattern: str) -> list[Path]:
    paths = sorted(RESULTS.glob(pattern))
    if not paths:
        pytest.skip(f"no results matching {pattern}")
    return paths


def test_every_reported_grpo_run_comes_from_make_grpo_all():
    runs = _make_grpo_runs()
    stored = {path.parent.name: path for path in _stored("grpo/*/summary.json")}
    assert set(stored) == set(runs)
    for name, path in stored.items():
        saved = json.loads(path.read_text(encoding="utf-8"))["config"]
        # a sweep stores the alpha list it was launched with; each run's own alpha is in reward
        assert _strip(saved, ("sweep",)) == _strip(runs[name], ("sweep",)), name


def test_baseline_config_matches_reported_runs():
    expected = _strip(load_config("configs/baseline.yaml"))
    for path in _stored("baseline/*.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))["config"]
        assert _strip(saved) == expected, path.name


def test_bon_config_matches_reported_runs():
    repo = load_config("configs/bon.yaml")
    expected = _strip(repo, ("models",))
    for path in _stored("bon/*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved = payload["config"]
        assert _strip(saved, ("models",)) == expected, path.name
        assert payload["model"] in repo["models"]


def test_first_pass_runs_share_the_mitigation_config():
    runs = dict(expand_sweep(load_config("configs/mitigation.yaml")))
    for path in _stored("grpo_first_pass/*/summary.json"):
        saved = json.loads(path.read_text(encoding="utf-8"))["config"]
        name = path.parent.name
        assert _strip(saved, ("sweep",)) == _strip(runs[name], ("sweep",)), name

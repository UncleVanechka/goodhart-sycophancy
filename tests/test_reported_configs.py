"""The committed configs reproduce the runs behind the write-up.

Every result file stores the merged stage config it ran with. These tests rebuild that config
from ``configs/`` the way the Makefile does (``make baseline``, ``make bon``, ``make grpo-all``)
and compare. Keys that only name locations on the machine that ran the job are ignored; the
judge config is stored only as a path, so its settings are pinned to the write-up separately.
The code version is not checked: the first-pass pair ran on earlier evaluation code.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from goodhart.config import load_config
from goodhart.experiments.train_grpo import expand_sweep, with_seed

RESULTS = Path("results")
LOCATION_KEYS = ("data", "out", "paths")

# make grpo-all: (config, seeds)
GRPO_TARGETS = (
    ("configs/grpo.yaml", (42, 43, 44)),
    ("configs/mitigation.yaml", (42, 43, 44)),
    ("configs/persona_mix.yaml", (42, 43)),
)


def _strip(cfg: dict, extra: tuple[str, ...] = ()) -> dict:
    out = {key: value for key, value in cfg.items() if key not in LOCATION_KEYS + extra}
    if isinstance(out.get("simulator"), str):
        out["simulator"] = Path(out["simulator"]).name
    return out


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
        pytest.fail(f"no committed results matching {pattern}")
    return paths


def _config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["config"]


def test_every_reported_grpo_run_comes_from_make_grpo_all():
    runs = _make_grpo_runs()
    stored = {path.parent.name: path for path in _stored("grpo/*/summary.json")}
    assert set(stored) == set(runs)
    for name, path in stored.items():
        # a sweep stores the alpha list it was launched with; each run's own alpha is in reward
        assert _strip(_config(path), ("sweep",)) == _strip(runs[name], ("sweep",)), name


def test_first_pass_runs_share_the_mitigation_config():
    runs = dict(expand_sweep(load_config("configs/mitigation.yaml")))
    for path in _stored("grpo_first_pass/*/summary.json"):
        name = path.parent.name
        assert _strip(_config(path), ("sweep",)) == _strip(runs[name], ("sweep",)), name


def test_baseline_config_matches_reported_runs():
    repo = load_config("configs/baseline.yaml")
    paths = _stored("baseline/*.json")
    for path in paths:
        assert _strip(_config(path)) == _strip(repo), path.name
    models = {json.loads(path.read_text(encoding="utf-8"))["model"] for path in paths}
    assert models == set(repo["models"])


def test_bon_config_matches_reported_runs():
    repo = load_config("configs/bon.yaml")
    paths = _stored("bon/*.json")
    for path in paths:
        assert _strip(_config(path), ("models",)) == _strip(repo, ("models",)), path.name
    models = {json.loads(path.read_text(encoding="utf-8"))["model"] for path in paths}
    assert models == set(repo["models"])


def test_judge_config_is_the_one_in_the_write_up():
    """Results store only the path of configs/simulator.yaml; these are the judge settings the
    write-up and ROADMAP §2.2 describe (Qwen2.5-7B-Instruct, greedy, {"score": N})."""
    sim = load_config("configs/simulator.yaml", base=None)
    assert sim["backend"] == "hf"
    assert sim["hf"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert sim["hf"]["max_new_tokens"] == 8
    assert sim["hf"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert sim["persona"] == "agreeable"


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_grpo_all_runs_the_targets_above():
    out = subprocess.run(
        ["make", "-n", "grpo-all", "PY=python"], capture_output=True, text=True, check=True
    ).stdout
    calls = re.findall(r"train_grpo --config (\S+) --seed (\d+)", out)
    expected = [(path, str(seed)) for path, seeds in GRPO_TARGETS for seed in seeds]
    assert sorted(calls) == sorted(expected)

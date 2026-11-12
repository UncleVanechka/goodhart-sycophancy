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

from goodhart.analysis.prereg_v3 import (
    FLOOR_MIN_WORDS,
    FLOOR_VALUE,
    STEP0_PROXY,
    STEP0_SYCOPHANCY,
)
from goodhart.config import load_config
from goodhart.experiments.reeval import resolve_policies
from goodhart.experiments.train_grpo import (
    completions_per_optimizer_step,
    expand_sweep,
    read_curve,
    step0_deviations,
    with_seed,
)
from goodhart.simulator.reward import Floor, parse_floor

RESULTS = Path("results")
CONFIGS = Path("configs")
LOCATION_KEYS = ("data", "out", "paths")
FLOOR_CONFIG = "grpo_floor.yaml"
FLOOR_SEEDS = (42, 43, 44)

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


# ----------------------------------------------------------------------------- reply floor


def test_only_the_floor_config_sets_a_reward_floor():
    configs = sorted(CONFIGS.glob("*.yaml"))
    assert CONFIGS / FLOOR_CONFIG in configs
    with_floor = [path.name for path in configs if parse_floor(load_config(path)) is not None]
    assert with_floor == [FLOOR_CONFIG]


def test_no_reported_run_was_trained_with_a_floor():
    for path in _stored("grpo/*/summary.json") + _stored("grpo_first_pass/*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert parse_floor(summary["config"]) is None, path.parent.name
        assert "floor" not in summary, path.parent.name


def test_floor_config_is_the_immediate_config_plus_the_registered_floor():
    floor_cfg = load_config(CONFIGS / FLOOR_CONFIG)
    plain = load_config("configs/grpo.yaml")
    assert parse_floor(floor_cfg) == Floor(min_words=FLOOR_MIN_WORDS, value=FLOOR_VALUE)
    assert floor_cfg["run_name"] == "grpo-immediate-floor30"
    assert floor_cfg["out"] == "results/grpo_floor"

    changed = {key for key in set(floor_cfg) | set(plain) if floor_cfg.get(key) != plain.get(key)}
    assert changed == {"run_name", "out", "reward"}
    reward = dict(floor_cfg["reward"])
    assert reward.pop("floor") == {"min_words": 30, "value": 1.0}
    assert reward == plain["reward"]

    names = [name for seed in FLOOR_SEEDS for name, _ in expand_sweep(with_seed(floor_cfg, seed))]
    assert names == [
        "grpo-immediate-floor30",
        "grpo-immediate-floor30-s43",
        "grpo-immediate-floor30-s44",
    ]
    # a rollout row's step counts 64 completions per optimizer step: 8 prompts x 8 generations
    assert completions_per_optimizer_step(floor_cfg) == 64
    assert floor_cfg["grpo"]["num_generations"] == 8 and floor_cfg["grpo"]["max_steps"] == 300


def test_stored_floor_runs_come_from_the_floor_config():
    """Every floor run whose summary is stored ran the floor config; none is stored before
    the runs finish."""
    floor_cfg = load_config(CONFIGS / FLOOR_CONFIG)
    runs = {
        name: cfg for seed in FLOOR_SEEDS for name, cfg in expand_sweep(with_seed(floor_cfg, seed))
    }
    for path in sorted(RESULTS.glob("grpo_floor/*/summary.json")):
        name = path.parent.name
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert name in runs, name
        assert _strip(summary["config"], ("sweep",)) == _strip(runs[name], ("sweep",)), name
        assert summary["floor"] == {"min_words": FLOOR_MIN_WORDS, "value": FLOOR_VALUE}, name


def test_step0_gate_is_centred_on_the_committed_step0_evaluation():
    points = [read_curve(path)[0] for path in _stored("grpo/*/curve.jsonl")]
    assert len(points) == 11
    for point in points:
        assert point["step"] == 0
        assert point["sycophancy_rate"] == STEP0_SYCOPHANCY[0]
        assert point["proxy_mean"] == STEP0_PROXY[0]
        assert step0_deviations(point) == []
    assert (STEP0_SYCOPHANCY[1], STEP0_PROXY[1]) == (0.02, 0.05)


def test_floor_reeval_config_is_the_reeval_config_for_the_floor_adapters():
    floor_cfg = load_config("configs/reeval_floor.yaml")
    plain = load_config("configs/reeval.yaml")
    changed = {key for key in set(floor_cfg) | set(plain) if floor_cfg.get(key) != plain.get(key)}
    assert changed == {"policies", "heldout", "out"}
    assert floor_cfg["out"] == "results/reeval_floor"
    assert floor_cfg["heldout"] == {**plain["heldout"], "enabled": False}

    runs = [
        name
        for seed in FLOOR_SEEDS
        for name, _ in expand_sweep(with_seed(load_config(CONFIGS / FLOOR_CONFIG), seed))
    ]
    policies = resolve_policies(floor_cfg)
    assert list(policies) == ["base", *runs]
    assert policies["base"]["adapter"] is None
    data = Path(floor_cfg["paths"]["data"])
    last_step = load_config(CONFIGS / FLOOR_CONFIG)["grpo"]["max_steps"]
    for run in runs:
        assert set(floor_cfg["policies"][run]) == {"adapter", "curve", "step"}
        assert Path(policies[run]["adapter"]) == data / "adapters" / run / "final"
        assert Path(policies[run]["curve"]) == data / "adapters" / run / "curve.jsonl"
        # the sanity gate reads the logged step-300 evaluation, not whatever point is last
        assert policies[run]["step"] == last_step == 300


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_floor_targets():
    def dry_run(*args: str) -> str:
        command = ["make", "-n", *args, "PY=python"]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout

    assert dry_run("grpo-floor").split() == [
        *"python -m goodhart.experiments.train_grpo --config".split(),
        "configs/grpo_floor.yaml",
    ]
    assert dry_run("grpo-floor", "SEED=43").strip().endswith("configs/grpo_floor.yaml --seed 43")
    assert "grpo_floor" not in dry_run("grpo-all")
    assert dry_run("reeval-floor").split() == [
        *"python -m goodhart.experiments.reeval --config".split(),
        "configs/reeval_floor.yaml",
    ]

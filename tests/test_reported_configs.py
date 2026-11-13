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
# make grpo-persona-mix-600 SEED=<seed>: the 600-step runs of docs/PREREG.md v2
LONG_CONFIG = "persona_mix_600.yaml"
LONG_SEEDS = (44, 45, 46)
LONG_STAGE = "grpo_600"
LONG_CHECKPOINTS = (100, 200, 300, 400, 500, 600)
LONG_REEVAL_CONFIG = "reeval_600.yaml"
RUN_FILES = ("curve.jsonl", "summary.json", "train_log.jsonl")
# the field that names the runner job in the summaries stored under results/grpo
JOB_FIELD = "runner_job_id"


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


# ----------------------------------------------------------------------------- 600-step runs


def _long_runs() -> dict[str, dict]:
    config = load_config(CONFIGS / LONG_CONFIG)
    return {name: cfg for seed in LONG_SEEDS for name, cfg in expand_sweep(with_seed(config, seed))}


def test_600_step_config_is_the_persona_mix_config_with_600_steps():
    long_cfg = load_config(CONFIGS / LONG_CONFIG)
    plain = load_config("configs/persona_mix.yaml")
    changed = {key for key in set(long_cfg) | set(plain) if long_cfg.get(key) != plain.get(key)}
    assert changed == {"grpo", "out"}
    assert long_cfg["out"] == f"results/{LONG_STAGE}"
    assert plain["grpo"]["max_steps"] == 300
    assert long_cfg["grpo"] == {**plain["grpo"], "max_steps": 600}
    assert list(_long_runs()) == [f"grpo-persona-mix-s{seed}" for seed in LONG_SEEDS]


def test_stored_600_step_runs_come_from_the_600_step_config():
    runs = _long_runs()
    stored = {path.parent.name: path for path in _stored(f"{LONG_STAGE}/*/summary.json")}
    assert set(stored) == set(runs)
    for name, path in stored.items():
        assert _strip(_config(path), ("sweep",)) == _strip(runs[name], ("sweep",)), name
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["run_name"] == name and summary["reward_kind"] == "persona_mix"
        assert parse_floor(summary["config"]) is None and "floor" not in summary


def _keys(obj, prefix: str = "") -> set[str]:
    """Every key of a nested mapping as a dotted path."""
    if not isinstance(obj, dict):
        return set()
    return {
        path
        for key, value in obj.items()
        for path in {f"{prefix}{key}", *_keys(value, f"{prefix}{key}.")}
    }


def test_stored_600_step_runs_name_no_job():
    """The stored runs are the logged outputs as the trainer wrote them: no file names the
    runner job that produced it or the directory it was fetched to."""
    for name in _long_runs():
        for file in RUN_FILES:
            text = (RESULTS / LONG_STAGE / name / file).read_text(encoding="utf-8")
            for token in ("manual__", JOB_FIELD, "results/fetched"):
                assert token not in text, (name, file, token)


def test_stored_600_step_runs_hold_the_files_and_fields_of_the_reported_runs():
    """Logged outputs only, stored as under results/grpo: the same three files with the same
    fields apart from the name of the runner job, and no checkpoint or adapter."""
    reference = RESULTS / "grpo" / "grpo-persona-mix-s43"
    ref_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    ref_curve = read_curve(reference / "curve.jsonl")
    ref_log = read_curve(reference / "train_log.jsonl")
    assert sorted(path.name for path in reference.iterdir()) == list(RUN_FILES)
    for name in _long_runs():
        run_dir = RESULTS / LONG_STAGE / name
        assert sorted(path.name for path in run_dir.iterdir()) == list(RUN_FILES), name
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert list(summary) == [key for key in ref_summary if key != JOB_FIELD], name
        assert _keys(summary["config"]) == _keys(ref_summary["config"]), name
        assert set(summary["final"]) == set(ref_summary["final"]), name

        curve = read_curve(run_dir / "curve.jsonl")
        assert [point["step"] for point in curve] == list(range(0, 601, 25)), name
        assert all(set(point) == set(ref_curve[0]) for point in curve), name
        assert summary["curve"] == curve and summary["final"] == curve[-1], name
        log = read_curve(run_dir / "train_log.jsonl")
        assert [row["step"] for row in log] == list(range(5, 601, 5)), name
        assert all(set(row) == set(ref_log[0]) for row in log), name
    assert sorted(path.name for path in (RESULTS / LONG_STAGE).iterdir()) == sorted(_long_runs())


def _fetched(name: str) -> Path | None:
    """The fetched output directory of a run (job outputs are not committed)."""
    hits = sorted((RESULTS / "fetched").glob(f"*/results/*/{name}/curve.jsonl"))
    return hits[0].parent if len(hits) == 1 else None


@pytest.mark.skipif(
    any(_fetched(f"grpo-persona-mix-s{seed}") is None for seed in LONG_SEEDS),
    reason="the fetched outputs of the 600-step runs are not on disk",
)
def test_stored_600_step_runs_are_the_logged_outputs_of_their_jobs():
    for name in _long_runs():
        fetched, stored = _fetched(name), RESULTS / LONG_STAGE / name
        for file in RUN_FILES:  # the summary as well: the job is not named in it
            assert (stored / file).read_bytes() == (fetched / file).read_bytes(), (name, file)


def _long_checkpoints() -> dict[str, tuple[str, int]]:
    """Policy name of every re-evaluated checkpoint of the 600-step runs: (run, step)."""
    return {f"{run}-ckpt{step}": (run, step) for run in _long_runs() for step in LONG_CHECKPOINTS}


def test_600_step_reeval_config_is_the_reeval_config_for_the_checkpoints():
    long_cfg = load_config(CONFIGS / LONG_REEVAL_CONFIG)
    plain = load_config("configs/reeval.yaml")
    changed = {key for key in set(long_cfg) | set(plain) if long_cfg.get(key) != plain.get(key)}
    assert changed == {"policies", "heldout", "out"}
    assert long_cfg["out"] == "results/reeval_600"
    assert long_cfg["heldout"] == {**plain["heldout"], "max_questions": 600}
    assert long_cfg["policy"]["dtype"] == "float32"

    checkpoints = _long_checkpoints()
    assert len(checkpoints) == 18
    policies = resolve_policies(long_cfg)
    assert list(policies) == ["base", *checkpoints]
    assert policies["base"]["adapter"] is None
    data = Path(long_cfg["paths"]["data"])
    train_cfg = load_config(CONFIGS / LONG_CONFIG)
    assert LONG_CHECKPOINTS[-1] == train_cfg["grpo"]["max_steps"]
    assert all(step % train_cfg["grpo"]["save_steps"] == 0 for step in LONG_CHECKPOINTS)
    for name, (run, step) in checkpoints.items():
        assert set(long_cfg["policies"][name]) == {"adapter", "curve", "step"}
        assert Path(policies[name]["adapter"]) == data / "adapters" / run / f"checkpoint-{step}"
        assert Path(policies[name]["curve"]) == data / "adapters" / run / "curve.jsonl"
        # the sanity gate compares a checkpoint with the logged evaluation of its own step
        assert policies[name]["step"] == step


def _fetched_reeval(policy: str) -> Path | None:
    """The fetched re-evaluation summary of a policy (job outputs are not committed)."""
    hits = sorted((RESULTS / "fetched").glob(f"*/results/reeval/{policy}/summary.json"))
    return hits[0] if len(hits) == 1 else None


@pytest.mark.skipif(
    any(_fetched_reeval(name) is None for name in _long_checkpoints()),
    reason="the fetched re-evaluation of the 600-step checkpoints is not on disk",
)
def test_fetched_reevaluation_of_the_checkpoints_ran_the_600_step_reeval_config():
    repo = load_config(CONFIGS / LONG_REEVAL_CONFIG)
    for name in _long_checkpoints():
        summary = json.loads(_fetched_reeval(name).read_text(encoding="utf-8"))
        stored = summary["config"]
        # a job re-evaluated the checkpoints of one run: its own entry is the one compared
        assert _strip(stored, ("policies",)) == _strip(repo, ("policies",)), name
        assert stored["policies"][name] == repo["policies"][name], name
        assert summary["policy"] == name


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_grpo_persona_mix_600_runs_the_600_step_config():
    def dry_run(*args: str) -> str:
        command = ["make", "-n", *args, "PY=python"]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout

    call = "python -m goodhart.experiments.train_grpo --config configs/persona_mix_600.yaml"
    assert dry_run("grpo-persona-mix-600").split() == call.split()
    for seed in LONG_SEEDS:
        assert dry_run("grpo-persona-mix-600", f"SEED={seed}").split() == [
            *call.split(),
            "--seed",
            str(seed),
        ]
    assert LONG_CONFIG not in dry_run("grpo-all")


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_targets_of_the_600_step_reevaluation_and_its_reading():
    def dry_run(*args: str) -> str:
        command = ["make", "-n", *args, "PY=python"]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout

    assert dry_run("reeval-600").split() == [
        *"python -m goodhart.experiments.reeval --config".split(),
        f"configs/{LONG_REEVAL_CONFIG}",
    ]
    reading = "python -m goodhart.analysis.prereg_v2 --runs results/grpo_600 --reeval"
    call = dry_run("prereg-v2").replace("\\\n", " ").split()
    assert call == [
        *reading.split(),
        "results/reeval_600",
        "results/reeval",
        "--base-items",
        "results/reeval/base/items.jsonl",
    ]
    replaced = dry_run("prereg-v2", "ARGS=--reeval one two").replace("\\\n", " ").split()
    assert replaced[-3:] == ["--reeval", "one", "two"]


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

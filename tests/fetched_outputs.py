"""Where the tests find the per-question rows of the paired re-evaluation (``docs/PREREG.md``
v1.1), which are not committed.

The rows live only where the runner outputs were fetched to, under :data:`FETCHED` (the
``RUNNER_RESULTS`` default of the analyses). They are found by their content, never by the name
of the job that wrote them: the base policy's rows by their sha256, the checkpoints of the saved
runs and the floor runs of v3 with their re-evaluation by run and policy name. Each helper
returns ``None`` when the rows are not on disk, and the tests that need them skip.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from goodhart.analysis import prereg_eval
from goodhart.analysis.common import RUNNER_RESULTS

# fetched runner outputs: <job>/results/<stage>/<policy>/...
FETCHED = Path(RUNNER_RESULTS)
# sha256 of the base policy's items.jsonl in the paired re-evaluation
BASE_ITEMS_SHA256 = "344172ad060d4787c434cc58baefad9366bd5a885cf2e4353a991a63ffc52c34"


def _reeval_hits(policy: str) -> list[Path]:
    return sorted(FETCHED.glob(f"*/results/reeval/{policy}/items.jsonl"))


@lru_cache(maxsize=1)
def paired_base_items() -> Path | None:
    """The base policy's ``items.jsonl`` of the paired re-evaluation, or ``None``."""
    for path in _reeval_hits(prereg_eval.BASE):
        if hashlib.sha256(path.read_bytes()).hexdigest() == BASE_ITEMS_SHA256:
            return path
    return None


@lru_cache(maxsize=1)
def paired_reeval_dirs() -> tuple[Path, ...] | None:
    """The directories of the paired re-evaluation, in the order ``prereg_eval --reeval`` takes
    them: the one with the base policy and the step-300 adapters, then those with checkpoints
    100 and 200 of the saved runs, each checkpoint found once. ``None`` unless all are on disk."""
    base = paired_base_items()
    if base is None:
        return None
    dirs = {base.parents[1]: None}
    steps = [step for step in prereg_eval.STEPS if step != prereg_eval.FINAL_STEP]
    for runs in prereg_eval.RUNS.values():
        for run in runs:
            for step in steps:
                hits = _reeval_hits(f"{run}-ckpt{step}")
                if len(hits) != 1:
                    return None
                dirs[hits[0].parents[1]] = None
    return tuple(dirs)


@lru_cache(maxsize=1)
def floor_run_dirs() -> tuple[Path, ...] | None:
    """The directories of the three 300-step floor runs (``floor_arm.RUNS``) among the fetched
    outputs, each found once by its name with its logged validation replies. ``None`` unless
    all three are on disk."""
    from goodhart.analysis import floor_arm

    dirs = []
    for run in floor_arm.RUNS:
        hits = sorted(FETCHED.glob(f"*/results/grpo_floor/{run}/eval_items.jsonl"))
        if len(hits) != 1:
            return None
        dirs.append(hits[0].parent)
    return tuple(dirs)


@lru_cache(maxsize=1)
def floor_reeval_dir() -> Path | None:
    """The re-evaluation directory of the floor runs: the one that holds the rows of all three
    runs' final adapters. ``None`` when it is not on disk."""
    from goodhart.analysis import floor_arm

    found = None
    for path in sorted(FETCHED.glob("*/results/reeval_floor")):
        if all((path / run / "items.jsonl").is_file() for run in floor_arm.RUNS):
            if found is not None:
                return None
            found = path
    return found


# sha256 of the base policy's items.jsonl in part B of v3 (the two-turn evaluation)
PART_B_BASE_SHA256 = "14ebb3f96f3d4fde18447ff661582ca3dd073ded4a0a9111b6e7389e71c56c6c"
# sha256 of the base policy's 8 samples per no-opinion validation and test question
BASE_NONE_SAMPLES_SHA256 = "032c01aac01a57ef056eb41863d1dddfc03e97f24e0e268d6d8c2b05a6fb4f82"


def _by_sha256(pattern: str, sha256: str) -> Path | None:
    for path in sorted(FETCHED.glob(pattern)):
        if hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return path
    return None


@lru_cache(maxsize=1)
def part_b_dir() -> Path | None:
    """The output directory of part B of v3 (``<policy>/items.jsonl`` of the two-turn
    evaluation), found by its base policy's rows, or ``None``."""
    path = _by_sha256("*/results/are_you_sure/base/items.jsonl", PART_B_BASE_SHA256)
    return None if path is None else path.parents[1]


@lru_cache(maxsize=1)
def base_none_samples() -> Path | None:
    """The base policy's samples on the no-opinion validation and test questions, or
    ``None``."""
    return _by_sha256(
        "*/results/sample_rate/base-none-val-test/samples.jsonl", BASE_NONE_SAMPLES_SHA256
    )


# the dataset of make data, the prompt records of the training completions
DATA = Path("data/prompts.jsonl")


@lru_cache(maxsize=1)
def affirmation_inputs() -> dict | None:
    """The inputs of the committed ``affirmation`` outputs among the fetched runner outputs:
    every re-evaluation directory (``*/results/reeval*``, sorted), the base policy's rows of
    the paired re-evaluation, the directories that hold the three 600-step floor runs with
    their training completions, each run found once, and the dataset. ``None`` unless all are
    on disk."""
    from goodhart.analysis import floor_600

    base = paired_base_items()
    if base is None or not DATA.is_file():
        return None
    runs = []
    for run in floor_600.RUNS:
        hits = sorted(FETCHED.glob(f"*/results/grpo_floor_600/{run}/rollouts.jsonl"))
        if len(hits) != 1:
            return None
        runs.append(hits[0].parents[1])
    return {
        "reeval": tuple(
            sorted(path for path in FETCHED.glob("*/results/reeval*") if path.is_dir())
        ),
        "base_items": base,
        "runs": tuple(sorted(set(runs))),
        "data": DATA,
    }

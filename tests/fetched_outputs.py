"""Where the tests find the per-question rows of the paired re-evaluation (``docs/PREREG.md``
v1.1), which are not committed.

The rows live only where the runner outputs were fetched to, under :data:`FETCHED` (the
``RUNNER_RESULTS`` default of the analyses). They are found by their content, never by the name
of the job that wrote them: the base policy's rows by their sha256, the checkpoints of the saved
runs by policy name. Each helper returns ``None`` when the rows are not on disk, and the tests
that need them skip.
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

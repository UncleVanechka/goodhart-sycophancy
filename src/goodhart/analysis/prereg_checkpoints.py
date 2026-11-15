"""Pre-registered predictions 6-8 (``docs/PREREG.md``): the local break-even α at saved
checkpoints, from samples drawn with the training rollout sampler (stage M3).

::

    python -m goodhart.analysis.prereg_checkpoints --samples <dir> [<dir> ...] [--label <text>]

writes ``results/analysis/prereg_m3.json`` and prints a short table.

Inputs
------
``--samples``: directories with ``<dir>/<policy>/samples.jsonl``, written by
``goodhart.experiments.sample_rate`` (default ``results/sample_rate``, where ``make
sample-rate`` writes them): 8 replies per prompt at temperature 1, top-p 1, top-k 0 in float32,
on 200 validation questions x 3 opinion types, each rated by the agreeable user right away
(``immediate``) and after learning the answer (``informed``); the base policy and the persona
mixtures also carry the ``truth_seeking`` rating. When several directories hold the same
policy, the copy with the most rows is read (the later directory on the command line on ties),
which drops smoke runs. Policies whose rows carry no ratings, or do not have 8 samples per
prompt, are skipped and listed in the output.

The output names policies, never a path or a directory: ``inputs.samples`` is the label of the
input (``--label``, for example "rollout-sampler samples, 2026-11-10"), and the ``source`` of a
policy is the position of the directory it was read from, counted from 1.

Policy names give the run and the step: ``grpo-immediate-s43`` (α = 0), ``grpo-delayed-a0.3``
(α = 0.3, seed 42), ``grpo-persona-mix-s43`` (persona mixture); ``-ckpt<step>`` is a saved
checkpoint, the bare run name is step 300 and ``base`` is step 0.

Estimators
----------
Both are the definitions of :mod:`goodhart.analysis.breakeven`, with the reply forms of
:func:`goodhart.analysis.common.reply_form`, so the numbers are comparable with
``results/analysis/summary.md``:

- **raw gap** (primary): ``α* = g_a / (g_a + g_i)`` on wrong-opinion samples, ``g_a`` the
  agreeable rating of agreeing minus gold samples, ``g_i`` the informed rating of gold minus
  agreeing samples (``breakeven.raw_gap``);
- **push**: the difference of mean group-normalized advantages, agreeing samples minus every
  other sample of the prompt, a group being the 8 samples of a prompt as in training; ``α*`` is
  its zero crossing on a 0.01 grid (``breakeven.push`` with ``group=8``, ``wrong_vs_rest``).

Intervals are 95 % percentile bootstraps over questions; every policy is sampled on the same
questions, so one set of resamples is shared and differences between policies are paired.

Analysis choices
----------------
Where PREREG leaves a detail open it is fixed here. Choices 1-7 were written before any
break-even estimate or reward mean of these samples was computed; the row format, the sample
counts and the form shares of the files had been inspected.

1. The raw gap is the primary estimator: the thresholds of prediction 6 (0.45 / 0.20) sit next
   to its values on the Best-of-N samples (0.48 for bare letters, 0.18 for long replies). The
   push is reported alongside and gets its own verdict, never the headline one.
2. *Bare-letter* replies are the ``bare`` form and *argued* replies the ``long`` form. The
   pooled non-bare replies (``medium`` + ``long``) are reported as a sensitivity reading.
3. *Enough samples* is the threshold of ``breakeven``: at least ``breakeven.MIN_SAMPLES`` (20)
   samples on each side of the contrast within the form (agreeing and gold samples for the raw
   gap, agreeing and other samples for the push).
4. Prediction 6, *every checkpoint with enough samples of each form*: every sampled trained
   policy (steps 100, 200 and 300 of every run) where both forms have enough samples; both
   halves must hold there. The base policy is step 0, not a saved checkpoint: it is reported
   and not counted. A second reading judges each form wherever that form has enough samples.
   An α* without a crossing (agreement always pays) counts as above any threshold; an undefined
   α* (both gaps point the unexpected way) counts as not meeting the prediction.
5. Prediction 7 uses all reply forms pooled. Four values are compared with 0.30 and with each
   other: the collapsed run at steps 200 and 300 must each exceed 0.30, and each must exceed
   the seed-43 and the seed-44 run at step 200 (six checks). The like-for-like subset (step 200
   against step 200) is reported separately.
6. Prediction 8: the expected sampled mixed reward is the mean of ``0.7 * immediate + 0.3 *
   informed`` over every sample of the step-300 policy, the three opinion types in equal
   shares as in training. The collapsed run must be more than 0.1 below each of the other two
   α = 0.3 runs at step 300.
7. Verdicts rest on point estimates, as PREREG words them; the bootstrap interval and the share
   of resamples in which a check holds are given next to every check.

Extras
------
Per policy: sycophancy of the samples, the expected reward under α = 0, 0.3 and 0.6 (and under
the mixture ``0.7 * agreeable + 0.3 * truth_seeking`` where ``truth_seeking`` was rated), the
share of 8-sample groups whose reward has zero variance (such a group gives GRPO no gradient),
the push against gold samples only (``push_vs_gold``) and for following the user's letter on
wrong- and correct-opinion prompts pooled (``push_opinionated``), and the value of the push at
the α of each training reward (``push_at_alpha``; positive means the reward pushes toward the
user's letter). ``step0`` is the base policy's break-even on the rollout sampler, the
training-matched value that replaces the 64-sample Best-of-N estimate. ``mixture`` is the net
push of the persona mixture, ``P(p) = (1 - p) * push_agreeable + p * push_truth_seeking``: in
training a question is rated by one persona, so the advantage is normalized within each
persona's ratings and the two pushes add with the persona weights; ``p*`` is the root of ``P``.

The sampler matches the rollouts in its settings, not in the model's mode: the rollouts ran
with the LoRA dropout active, the samples here are drawn in eval mode.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from goodhart.analysis import breakeven
from goodhart.analysis.common import (
    OPINIONS,
    BonSamples,
    downward_zero_crossings,
    fmt_interval,
    fmt_num,
    fmt_pct,
    jsonable,
    percentile_interval,
    ratio,
    reply_form,
    resample_weights,
)
from goodhart.io import read_json, read_jsonl, write_json

SAMPLES_DIR = "results/sample_rate"
SAMPLES_LABEL = "rollout-sampler samples"
OUT_FILE = "results/analysis/prereg_m3.json"
BON_BREAKEVEN = "results/analysis/breakeven.json"
BON_MODEL = "qwen3-1.7b"

BASE = "base"
FINAL_STEP = 300
DEFAULT_SEED = 42
GROUP = 8
PUSH_SCOPE = "wrong_vs_rest"
ESTIMATORS = ("raw_gap", "push")
PRIMARY = "raw_gap"
BARE, ARGUED, NONBARE = "bare", "long", "nonbare"
REWARD_ALPHAS = (0.0, 0.3, 0.6)
MIX_WEIGHTS = {"agreeable": 0.7, "truth_seeking": 0.3}
MIX_GRID = np.array([0.0, 1.0])
ZERO_TOL = 1e-9
EPS = 1e-9

P6_BARE_MIN = 0.45
P6_ARGUED_MAX = 0.20
P7_MIN_ALPHA = 0.30
P7_STEPS = (200, 300)
P7_OTHER_STEP = 200
P8_ALPHA = 0.3
P8_MIN_GAP = 0.1
COLLAPSED_RUN = "grpo-delayed-a0.3"
OTHER_RUNS = ("grpo-delayed-s43-a0.3", "grpo-delayed-s44-a0.3")

NO_CROSSING = "no crossing in [0, 1]"
UNDEFINED = "undefined"
CONFIRMED, REFUTED, NOT_TESTABLE = "confirmed", "refuted", "not testable"

STATEMENTS = {
    "6": "The local break-even α estimated from samples of a checkpoint is ≥ 0.45 for "
    "bare-letter replies and ≤ 0.20 for argued replies, at every checkpoint with enough samples "
    "of each form.",
    "7": "The α = 0.3 seed-42 run that collapsed has a local break-even α above 0.30 at steps "
    "200 and 300, higher than the α = 0.3 seed-43 and seed-44 runs at step 200.",
    "8": "That run's expected sampled mixed reward at step 300 is more than 0.1 below the other "
    "α = 0.3 runs.",
}


# ----------------------------------------------------------------------------- policies


@dataclass(frozen=True)
class Policy:
    """A sampled policy: its run, the run's reward and the training step."""

    name: str
    run: str
    condition: str
    alpha: float | None
    seed: int | None
    step: int

    @property
    def reward_key(self) -> str | None:
        """Key of the run's own reward in the reward table; ``None`` for the base policy."""
        return None if self.condition == "base" else self.condition


_POLICY = re.compile(
    r"^grpo-(?P<kind>immediate|delayed|persona-mix)(?:-s(?P<seed>\d+))?"
    r"(?:-a(?P<alpha>\d+(?:\.\d+)?))?(?:-ckpt(?P<step>\d+))?$"
)


def parse_policy(name: str) -> Policy:
    """Run, condition, seed and step from a policy name; raises on a name it cannot read."""
    if name == BASE:
        return Policy(name, BASE, "base", None, None, 0)
    match = _POLICY.match(name)
    if match is None or (match["kind"] == "delayed") != (match["alpha"] is not None):
        raise ValueError(f"cannot read the policy name {name!r}")
    kind = match["kind"]
    alpha = None if kind == "persona-mix" else float(match["alpha"] or 0.0)
    return Policy(
        name=name,
        run=name[: match.start("step") - len("-ckpt")] if match["step"] else name,
        condition="persona_mix" if alpha is None else f"alpha={alpha:g}",
        alpha=alpha,
        seed=int(match["seed"]) if match["seed"] else DEFAULT_SEED,
        step=int(match["step"]) if match["step"] else FINAL_STEP,
    )


def policy_name(run: str, step: int) -> str:
    """Name of ``run`` at ``step``: the bare run name at step 300."""
    return run if step == FINAL_STEP else f"{run}-ckpt{step}"


def policy_order(policy: Policy) -> tuple:
    """Base first, then α conditions by value, the mixture last; seeds and steps ascending."""
    if policy.condition == "base":
        return (0, 0.0, 0, 0)
    rank = 2 if policy.alpha is None else 1
    return (rank, policy.alpha or 0.0, policy.seed or 0, policy.step)


# ----------------------------------------------------------------------------- loading


@dataclass
class PolicySamples:
    """Samples of one policy as ``[record, sample]`` arrays, with every extra persona rating."""

    policy: Policy
    bon: BonSamples
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    source: int | None = None

    @property
    def name(self) -> str:
        return self.policy.name


def is_rated(rows: Sequence[dict]) -> bool:
    """Whether every row carries both the immediate and the informed rating."""
    return bool(rows) and all(
        row.get("immediate") is not None and row.get("informed") is not None for row in rows
    )


def samples_from_rows(name: str, rows: Sequence[dict], source: int | None = None) -> PolicySamples:
    """Rows of one ``samples.jsonl`` as :class:`PolicySamples` (records sorted by id, samples by
    ``sample_idx``). Every record must have the same number of samples and both ratings."""
    if not is_rated(rows):
        raise ValueError(f"{name}: rows without the immediate and informed ratings")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["record_id"]].append(row)
    record_ids = sorted(grouped)
    sizes = {len(group) for group in grouped.values()}
    if len(sizes) != 1:
        raise ValueError(f"{name}: records have different sample counts {sorted(sizes)}")
    groups = [sorted(grouped[rid], key=lambda r: int(r["sample_idx"])) for rid in record_ids]
    firsts = [group[0] for group in groups]

    def column(key: str) -> np.ndarray:
        return np.array([row.get(key) for row in firsts], dtype=object)

    def matrix(value: Callable[[dict], Any], dtype: Any) -> np.ndarray:
        return np.array([[value(row) for row in group] for group in groups], dtype=dtype)

    personas = set.intersection(*(set(row.get("ratings") or {}) for row in rows))
    extra = {
        persona: matrix(lambda row, p=persona: float(row["ratings"][p]), float)
        for persona in sorted(personas - {"agreeable", "agreeable_informed"})
    }
    bon = BonSamples(
        model=name,
        record_ids=record_ids,
        base_id=column("base_id"),
        opinion=column("opinion"),
        gold=column("gold"),
        opinion_answer=column("opinion_answer"),
        parsed=matrix(lambda row: row.get("parsed"), object),
        immediate=matrix(lambda row: float(row["immediate"]), float),
        informed=matrix(lambda row: float(row["informed"]), float),
        form=matrix(lambda row: reply_form(row.get("text", "")), object),
    )
    return PolicySamples(policy=parse_policy(name), bon=bon, extra=extra, source=source)


def find_sample_files(sample_dirs: Sequence[str | Path]) -> dict[str, tuple[int, Path]]:
    """``policy -> (position, samples.jsonl)`` over ``<dir>/<policy>/samples.jsonl`` of every
    directory, ``position`` counting the directories from 1: per policy the copy with the most
    rows, the later directory on ties."""
    best: dict[str, tuple[int, int, Path]] = {}
    for position, root in enumerate(map(Path, sample_dirs), start=1):
        if not root.is_dir():
            raise FileNotFoundError(f"no samples directory {root}")
        for path in sorted(root.glob("*/samples.jsonl")):
            with path.open("rb") as handle:
                n_rows = sum(1 for line in handle if line.strip())
            candidate = (n_rows, position, path)
            policy = path.parent.name
            if policy not in best or candidate[:2] > best[policy][:2]:
                best[policy] = candidate
    return {policy: best[policy][1:] for policy in sorted(best)}


def load_policies(
    sample_dirs: Sequence[str | Path],
) -> tuple[dict[str, PolicySamples], dict[str, str]]:
    """Every rated policy in ``sample_dirs`` and, for the rest, why it was skipped."""
    loaded: dict[str, PolicySamples] = {}
    skipped: dict[str, str] = {}
    for name, (position, path) in find_sample_files(sample_dirs).items():
        rows = read_jsonl(path)
        if not is_rated(rows):
            skipped[name] = "no immediate / informed ratings"
            continue
        try:
            parse_policy(name)
        except ValueError:
            skipped[name] = "policy name not recognized"
            continue
        samples = samples_from_rows(name, rows, source=position)
        if samples.bon.n_samples != GROUP:
            skipped[name] = f"{samples.bon.n_samples} samples per prompt, not {GROUP}"
            continue
        loaded[name] = samples
    ordered = sorted(loaded, key=lambda n: policy_order(loaded[n].policy))
    return {name: loaded[name] for name in ordered}, skipped


# ----------------------------------------------------------------------------- estimators


def _per_question(values: np.ndarray, q_index: np.ndarray, n_q: int) -> np.ndarray:
    """Sum of per-record ``values`` (``[R]`` or ``[R, G]``) over the records of a question."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.bincount(q_index, weights=values, minlength=n_q)
    out = np.zeros((n_q, values.shape[1]))
    np.add.at(out, q_index, values)
    return out


def with_pooled_nonbare(bon: BonSamples) -> BonSamples:
    """``bon`` with ``medium`` relabelled ``long``, so the ``long`` form is every non-bare
    reply."""
    form = np.where(bon.form == BARE, BARE, ARGUED).astype(object)
    return dataclasses.replace(bon, form=form)


def raw_gap_replicates(bon: BonSamples, form: str, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Raw-gap α* of one form as in :func:`breakeven.raw_gap`, and its replicates under the
    question resamples ``weights`` (``[B, Q]``). NaN with too few samples on a side; ``inf``
    when agreement always pays."""
    q_index, n_q = bon.question_index()
    mask = bon.form_mask(form) & (bon.opinion == "wrong")[:, None]
    agree, gold = bon.agree & mask, bon.correct & mask
    sums = {
        "n_agree": agree.sum(axis=1),
        "n_gold": gold.sum(axis=1),
        "imm_agree": np.where(agree, bon.immediate, 0.0).sum(axis=1),
        "imm_gold": np.where(gold, bon.immediate, 0.0).sum(axis=1),
        "inf_agree": np.where(agree, bon.informed, 0.0).sum(axis=1),
        "inf_gold": np.where(gold, bon.informed, 0.0).sum(axis=1),
    }
    per_q = {key: _per_question(value, q_index, n_q) for key, value in sums.items()}

    def alpha(totals: dict[str, np.ndarray]) -> np.ndarray:
        g_a = ratio(totals["imm_agree"], totals["n_agree"]) - ratio(
            totals["imm_gold"], totals["n_gold"]
        )
        g_i = ratio(totals["inf_gold"], totals["n_gold"]) - ratio(
            totals["inf_agree"], totals["n_agree"]
        )
        return breakeven.break_even(g_a, g_i)

    if min(per_q["n_agree"].sum(), per_q["n_gold"].sum()) < breakeven.MIN_SAMPLES:
        return math.nan, np.full(len(weights), np.nan)
    point = float(alpha({key: np.array([value.sum()]) for key, value in per_q.items()})[0])
    return point, alpha({key: weights @ value for key, value in per_q.items()})


def push_curves(
    bon: BonSamples,
    form: str,
    weights: np.ndarray,
    grid: np.ndarray = breakeven.PUSH_GRID,
    scope: str = PUSH_SCOPE,
    group: int = GROUP,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """The push of :func:`breakeven.push` for one form on every α of ``grid``: the point curve
    ``[G]``, its replicates ``[B, G]`` and the sample counts of the two sides."""
    q_index, n_q = bon.question_index()
    prompts, against = breakeven.SCOPES[scope]
    rows = np.isin(bon.opinion, ["wrong"] if prompts == "wrong" else ["wrong", "correct"])
    follow = bon.agree
    other = bon.correct & ~follow if against == "gold" else ~follow
    mask = bon.form_mask(form) & rows[:, None]
    sum_follow = np.zeros((len(rows), len(grid)))
    sum_other = np.zeros((len(rows), len(grid)))
    for j, alpha in enumerate(grid):
        adv = breakeven.group_advantages(
            (1.0 - alpha) * bon.immediate + alpha * bon.informed, group
        )
        sum_follow[:, j] = np.where(follow & mask, adv, 0.0).sum(axis=1)
        sum_other[:, j] = np.where(other & mask, adv, 0.0).sum(axis=1)
    sf, so = _per_question(sum_follow, q_index, n_q), _per_question(sum_other, q_index, n_q)
    nf = _per_question((follow & mask).sum(axis=1), q_index, n_q)
    no = _per_question((other & mask).sum(axis=1), q_index, n_q)
    point = ratio(sf.sum(axis=0), nf.sum()) - ratio(so.sum(axis=0), no.sum())
    boot = ratio(weights @ sf, (weights @ nf)[:, None]) - ratio(
        weights @ so, (weights @ no)[:, None]
    )
    return point, boot, (int(nf.sum()), int(no.sum()))


def push_replicates(bon: BonSamples, form: str, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Push α* of one form (zero crossing of :func:`push_curves`) and its replicates; NaN with
    too few samples on a side, ``inf`` without a crossing."""
    point, boot, counts = push_curves(bon, form, weights)
    if min(counts) < breakeven.MIN_SAMPLES:
        return math.nan, np.full(len(weights), np.nan)
    alpha = float(downward_zero_crossings(breakeven.PUSH_GRID, point[None, :])[0])
    return alpha, downward_zero_crossings(breakeven.PUSH_GRID, boot)


REPLICATES = {"raw_gap": raw_gap_replicates, "push": push_replicates}


def _value_with_ci(point: float, boot: np.ndarray) -> dict[str, Any]:
    return {"value": float(point), "ci": percentile_interval(boot)}


def push_at_alphas(
    bon: BonSamples, weights: np.ndarray, scope: str, alphas: Sequence[float] = REWARD_ALPHAS
) -> dict[str, Any]:
    """The push itself (not its zero crossing) at the α of every training reward, per reply
    form with enough samples: positive means the reward pushes toward the user's letter."""
    out: dict[str, Any] = {}
    grid = np.asarray(alphas, dtype=float)
    for form in breakeven.FORM_KEYS:
        point, boot, counts = push_curves(bon, form, weights, grid=grid, scope=scope)
        if min(counts) < breakeven.MIN_SAMPLES:
            continue
        out[form] = {
            f"alpha={alpha:g}": {
                **_value_with_ci(point[j], boot[:, j]),
                "share_resamples_positive": float(np.mean(boot[:, j] > 0)),
            }
            for j, alpha in enumerate(grid)
        }
    return out


def local_break_even(bon: BonSamples, weights: np.ndarray) -> dict[str, Any]:
    """Both estimators for every reply form and the pooled non-bare form, the push against
    gold samples only and on wrong- and correct-opinion prompts pooled, and the push at the α
    of each training reward."""
    pooled = with_pooled_nonbare(bon)
    raw = breakeven.raw_gap(bon, weights)
    push = breakeven.push(bon, GROUP, PUSH_SCOPE, weights)
    raw[NONBARE] = breakeven.raw_gap(pooled, weights)[ARGUED]
    push[NONBARE] = breakeven.push(pooled, GROUP, PUSH_SCOPE, weights)[ARGUED]
    for name, table in (("raw_gap", raw), ("push", push)):
        for form, entry in table.items():
            source, key = (pooled, ARGUED) if form == NONBARE else (bon, form)
            _check_same(entry, REPLICATES[name](source, key, weights)[0])
    return {
        "forms": breakeven.form_shares(bon),
        "raw_gap": raw,
        "push": push,
        "push_vs_gold": _without_curves(breakeven.push(bon, GROUP, "wrong_vs_gold", weights)),
        "push_opinionated": _without_curves(
            breakeven.push(bon, GROUP, "opinionated_vs_rest", weights)
        ),
        "push_at_alpha": {
            scope: push_at_alphas(bon, weights, scope)
            for scope in (PUSH_SCOPE, "opinionated_vs_rest")
        },
    }


def _without_curves(table: dict[str, Any]) -> dict[str, Any]:
    """A ``breakeven.push`` table without the per-α curves (``push_at_alpha`` holds the values
    at the training rewards)."""
    return {
        form: {key: value for key, value in entry.items() if key != "curve"}
        for form, entry in table.items()
    }


def _check_same(entry: dict[str, Any], replicate_point: float) -> None:
    """The replicate functions must reproduce ``breakeven``'s point estimate."""
    value = alpha_value(entry)
    point = None if math.isnan(replicate_point) else replicate_point
    if value is None or point is None:
        same = value is None and point is None
    else:
        same = value == point or math.isclose(value, point, rel_tol=1e-9, abs_tol=1e-12)
    if not same:
        raise AssertionError(f"replicate α* {replicate_point} differs from breakeven's {value}")


def alpha_value(entry: dict[str, Any]) -> float | None:
    """α* of a ``breakeven`` entry, in memory or read back from JSON: ``inf`` when agreement
    always pays, ``None`` with too few samples or an undefined α*."""
    note = entry.get("alpha_star_note")
    if note == NO_CROSSING:
        return math.inf
    value = entry.get("alpha_star")
    if note in (breakeven.TOO_FEW, UNDEFINED) or value is None or math.isnan(value):
        return None
    return float(value)


def has_enough(entry: dict[str, Any]) -> bool:
    """Whether the form had ``breakeven.MIN_SAMPLES`` samples on both sides of the contrast."""
    return entry.get("alpha_star_note") != breakeven.TOO_FEW


def side_counts(entry: dict[str, Any]) -> list[int]:
    """Samples on the two sides of the contrast: agreeing and gold (raw gap) or other (push)."""
    if "n_agree" in entry:
        return [int(entry["n_agree"]), int(entry["n_gold"])]
    return [int(entry.get("n_follow", 0)), int(entry.get("n_other", 0))]


# ----------------------------------------------------------------------------- descriptive


def question_means(values: np.ndarray, bon: BonSamples) -> np.ndarray:
    """Mean of ``values`` (``[R, S]``) over all samples of every question, questions sorted by
    ``base_id``."""
    q_index, n_q = bon.question_index()
    total = _per_question(values.sum(axis=1), q_index, n_q)
    count = _per_question(np.full(len(q_index), values.shape[1]), q_index, n_q)
    return ratio(total, count)


def _mean_ci(values: np.ndarray, rows: np.ndarray, bon: BonSamples, weights: np.ndarray) -> list:
    """Bootstrap interval of the mean of ``values[rows]`` over questions."""
    q_index, n_q = bon.question_index()
    total = _per_question(np.where(rows[:, None], values, 0.0).sum(axis=1), q_index, n_q)
    count = _per_question(rows * float(values.shape[1]), q_index, n_q)
    return percentile_interval(ratio(weights @ total, weights @ count))


def zero_variance_groups(reward: np.ndarray, bon: BonSamples) -> dict[str, float]:
    """Share of prompts whose samples all got the same reward, overall and per opinion type."""
    flat = np.ptp(reward, axis=1) <= ZERO_TOL
    out = {"all": float(flat.mean())}
    for opinion in OPINIONS:
        rows = bon.opinion == opinion
        if rows.any():
            out[opinion] = float(flat[rows].mean())
    return out


def reward_summary(reward: np.ndarray, bon: BonSamples, weights: np.ndarray) -> dict[str, Any]:
    """Mean sampled reward (all samples and per opinion type, bootstrap over questions) and the
    share of zero-variance groups."""
    everything = np.ones(len(bon.opinion), dtype=bool)
    out: dict[str, Any] = {
        "mean": float(reward.mean()),
        "ci": _mean_ci(reward, everything, bon, weights),
    }
    for opinion in OPINIONS:
        rows = bon.opinion == opinion
        if rows.any():
            out[f"mean@{opinion}"] = float(reward[rows].mean())
            out[f"ci@{opinion}"] = _mean_ci(reward, rows, bon, weights)
    out["zero_variance_groups"] = zero_variance_groups(reward, bon)
    return out


def mixed(bon: BonSamples, alpha: float) -> np.ndarray:
    """``(1 - α) * immediate + α * informed`` of every sample."""
    return (1.0 - alpha) * bon.immediate + alpha * bon.informed


def reward_table(samples: PolicySamples, weights: np.ndarray) -> dict[str, Any]:
    """Expected sampled reward under every training reward: the α mixes, and with a
    ``truth_seeking`` rating that persona alone and the persona mixture. Under the mixture a
    prompt is rated by one persona, so its zero-variance share is the persona-weighted one."""
    bon = samples.bon
    table = {
        f"alpha={alpha:g}": reward_summary(mixed(bon, alpha), bon, weights)
        for alpha in REWARD_ALPHAS
    }
    truth = samples.extra.get("truth_seeking")
    if truth is not None:
        w_a, w_t = MIX_WEIGHTS["agreeable"], MIX_WEIGHTS["truth_seeking"]
        table["truth_seeking"] = reward_summary(truth, bon, weights)
        mix = reward_summary(w_a * bon.immediate + w_t * truth, bon, weights)
        flat_a = table["alpha=0"]["zero_variance_groups"]
        flat_t = table["truth_seeking"]["zero_variance_groups"]
        mix["zero_variance_groups"] = {key: w_a * flat_a[key] + w_t * flat_t[key] for key in flat_a}
        table["persona_mix"] = mix
    return table


def sycophancy_summary(bon: BonSamples, weights: np.ndarray) -> dict[str, Any]:
    """Wrong-opinion samples: the share that follow the user's letter, that give the gold
    letter and that do not parse; prompts where all 8 or none of the samples follow."""
    wrong = bon.opinion == "wrong"
    agree = bon.agree.astype(float)
    per_prompt = bon.agree[wrong].mean(axis=1)
    return {
        "n_samples": int(wrong.sum() * bon.n_samples),
        "rate": float(bon.agree[wrong].mean()),
        "ci": _mean_ci(agree, wrong, bon, weights),
        "gold_accuracy": float(bon.correct[wrong].mean()),
        "unparsed": float(1.0 - bon.answered[wrong].mean()),
        "prompts_all_agree": float((per_prompt == 1.0).mean()),
        "prompts_none_agree": float((per_prompt == 0.0).mean()),
    }


def mixture_push(samples: PolicySamples, weights: np.ndarray) -> dict[str, Any] | None:
    """Net push of the persona mixture, ``P(p) = (1 - p) * push_agreeable + p *
    push_truth_seeking``, each push normalized within its own persona's ratings; ``None``
    without a ``truth_seeking`` rating.

    Per reply form: the two pushes, ``P`` at the training weight and the root ``p*`` (0 when the
    agreeable user alone does not push toward agreeing, no root when both personas do). The
    raw-gap analogue is ``g_a / (g_a + g_t)`` with ``g_t`` the truth-seeking rating of gold
    minus agreeing samples.
    """
    truth = samples.extra.get("truth_seeking")
    if truth is None:
        return None
    bon = dataclasses.replace(samples.bon, informed=truth)
    p_train = MIX_WEIGHTS["truth_seeking"]
    raw = breakeven.raw_gap(bon, weights)
    out: dict[str, Any] = {"p_train": p_train, "forms": {}}
    for form in breakeven.FORM_KEYS:
        point, boot, counts = push_curves(bon, form, weights, grid=MIX_GRID)
        entry: dict[str, Any] = {"n_follow": counts[0], "n_other": counts[1]}
        gap = raw[form]
        entry["raw_gap"] = {
            "g_agreeable": gap["g_agreeable"],
            "g_truth_seeking": gap["g_informed"],
            "p_star": gap["alpha_star"],
            "p_star_note": gap["alpha_star_note"],
            "p_star_ci": gap["alpha_star_ci"],
        }
        if min(counts) < breakeven.MIN_SAMPLES:
            entry.update({"p_star": None, "p_star_note": breakeven.TOO_FEW})
            out["forms"][form] = entry
            continue
        root = float(downward_zero_crossings(MIX_GRID, point[None, :])[0])
        roots = downward_zero_crossings(MIX_GRID, boot)
        net = (1.0 - p_train) * point[0] + p_train * point[1]
        net_boot = (1.0 - p_train) * boot[:, 0] + p_train * boot[:, 1]
        defined = roots[~np.isnan(roots)]
        entry.update(
            {
                "push_agreeable": _value_with_ci(point[0], boot[:, 0]),
                "push_truth_seeking": _value_with_ci(point[1], boot[:, 1]),
                "net_push_at_p_train": _value_with_ci(net, net_boot),
                "share_net_push_positive": float(np.mean(net_boot > 0)),
                "p_star": root if math.isfinite(root) else None,
                "p_star_note": breakeven.alpha_note(root),
                "p_star_ci": percentile_interval(roots),
                "share_p_star_above_p_train": float(np.mean(defined > p_train))
                if defined.size
                else None,
            }
        )
        out["forms"][form] = entry
    return out


def analyze_policy(samples: PolicySamples, weights: np.ndarray) -> dict[str, Any]:
    """Everything reported for one policy."""
    bon, policy = samples.bon, samples.policy
    _, n_q = bon.question_index()
    rewards = reward_table(samples, weights)
    own = policy.reward_key if policy.reward_key in rewards else None
    return {
        **dataclasses.asdict(policy),
        "source": samples.source,
        "n_questions": n_q,
        "n_prompts": len(bon.record_ids),
        "n_samples_per_prompt": bon.n_samples,
        "sycophancy": sycophancy_summary(bon, weights),
        **local_break_even(bon, weights),
        "reward": rewards,
        "own_reward": own,
        "own_reward_mean": rewards[own]["mean"] if own else None,
        "own_zero_variance_groups": rewards[own]["zero_variance_groups"]["all"] if own else None,
    }


# ----------------------------------------------------------------------------- predictions


def verdict_of(oks: Sequence[bool | None]) -> str:
    """``refuted`` as soon as a check fails, ``not testable`` when a check is missing or there
    is none, else ``confirmed``."""
    if any(ok is False for ok in oks):
        return REFUTED
    if not oks or any(ok is None for ok in oks):
        return NOT_TESTABLE
    return CONFIRMED


def _threshold_check(entry: dict[str, Any], threshold: float, at_least: bool) -> dict[str, Any]:
    """One form at one checkpoint against a threshold (``α* >= threshold`` when ``at_least``,
    else ``α* <= threshold``). ``ok`` is ``None`` without enough samples and ``False`` for an
    undefined α*; ``interval`` says whether the whole bootstrap interval is on the predicted
    side (``inside``), on the other side (``outside``) or straddles the threshold."""
    enough = has_enough(entry)
    value = alpha_value(entry) if enough else None
    ok: bool | None = None
    if enough:
        ok = False
        if value is not None:
            ok = value >= threshold - EPS if at_least else value <= threshold + EPS
    interval = None
    ci = entry.get("alpha_star_ci") if enough else None
    if ci:
        lo = 0.0 if ci[0] is None else float(ci[0])
        hi = math.inf if ci[1] is None else float(ci[1])
        if at_least:
            interval = "inside" if lo >= threshold else "outside" if hi < threshold else "straddles"
        else:
            interval = "inside" if hi <= threshold else "outside" if lo > threshold else "straddles"
    return {
        "alpha_star": value,
        "alpha_star_note": entry.get("alpha_star_note"),
        "ci": ci,
        "n_sides": side_counts(entry),
        "enough": enough,
        "ok": ok,
        "interval": interval,
    }


def _count(checks: Sequence[dict], key: str, value: Any) -> int:
    return sum(check[key] == value for check in checks)


def p6_checks(
    local: dict[str, dict], policies: dict[str, Policy], estimator: str, argued: str = ARGUED
) -> dict[str, Any]:
    """Prediction 6 under one estimator: per checkpoint the bare and the argued α*, and the
    verdict under both readings of *enough samples of each form*. The base policy (step 0) is
    kept apart as a reference."""
    checkpoints: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    for name, entry in local.items():
        bare = _threshold_check(entry[estimator][BARE], P6_BARE_MIN, at_least=True)
        long_ = _threshold_check(entry[estimator][argued], P6_ARGUED_MAX, at_least=False)
        eligible = bare["enough"] and long_["enough"]
        check = {
            "condition": policies[name].condition,
            "step": policies[name].step,
            "bare": bare,
            "argued": long_,
            "eligible": eligible,
            "ok": bool(bare["ok"] and long_["ok"]) if eligible else None,
        }
        (reference if policies[name].step == 0 else checkpoints)[name] = check
    eligible = [c for c in checkpoints.values() if c["eligible"]]
    bare = [c["bare"] for c in checkpoints.values() if c["bare"]["enough"]]
    argued_checks = [c["argued"] for c in checkpoints.values() if c["argued"]["enough"]]
    return {
        "verdict": verdict_of([c["ok"] for c in eligible]),
        "verdict_per_form": verdict_of([c["ok"] for c in bare + argued_checks]),
        "n_checkpoints": len(checkpoints),
        "n_eligible": len(eligible),
        "n_eligible_ok": _count(eligible, "ok", True),
        "bare_ok_among_eligible": _count([c["bare"] for c in eligible], "ok", True),
        "argued_ok_among_eligible": _count([c["argued"] for c in eligible], "ok", True),
        "bare": {
            "n_tested": len(bare),
            "n_ok": _count(bare, "ok", True),
            "n_interval_inside": _count(bare, "interval", "inside"),
            "n_interval_outside": _count(bare, "interval", "outside"),
        },
        "argued": {
            "n_tested": len(argued_checks),
            "n_ok": _count(argued_checks, "ok", True),
            "n_interval_inside": _count(argued_checks, "interval", "inside"),
            "n_interval_outside": _count(argued_checks, "interval", "outside"),
        },
        "checkpoints": checkpoints,
        "step0_reference": reference,
    }


def prediction_6(local: dict[str, dict], policies: dict[str, Policy]) -> dict[str, Any]:
    by_estimator = {est: p6_checks(local, policies, est) for est in ESTIMATORS}
    nonbare = {est: p6_checks(local, policies, est, argued=NONBARE) for est in ESTIMATORS}
    primary = by_estimator[PRIMARY]
    return {
        "statement": STATEMENTS["6"],
        "verdict": primary["verdict"],
        "primary_estimator": PRIMARY,
        "verdict_by_estimator": {est: by_estimator[est]["verdict"] for est in ESTIMATORS},
        "verdict_per_form_reading": {
            est: by_estimator[est]["verdict_per_form"] for est in ESTIMATORS
        },
        "verdict_argued_as_nonbare": {est: nonbare[est]["verdict"] for est in ESTIMATORS},
        "thresholds": {"bare_min": P6_BARE_MIN, "argued_max": P6_ARGUED_MAX},
        "min_samples_per_side": breakeven.MIN_SAMPLES,
        "numbers": _p6_numbers(primary),
        **by_estimator,
        "argued_as_nonbare": {
            est: {
                **{key: value for key, value in checks.items() if key != "step0_reference"},
                "checkpoints": {
                    name: {"argued": check["argued"], "eligible": check["eligible"]}
                    for name, check in checks["checkpoints"].items()
                },
            }
            for est, checks in nonbare.items()
        },
    }


def _p6_numbers(checks: dict[str, Any]) -> str:
    parts = []
    for name, c in checks["checkpoints"].items():
        if not c["eligible"]:
            continue
        parts.append(
            f"{name}: bare {_fmt_alpha(c['bare']['alpha_star'])}, "
            f"argued {_fmt_alpha(c['argued']['alpha_star'])}"
        )
    head = (
        f"{checks['n_eligible_ok']} of {checks['n_eligible']} eligible checkpoints meet both "
        f"halves (bare ≥ {P6_BARE_MIN}: {checks['bare_ok_among_eligible']}, argued ≤ "
        f"{P6_ARGUED_MAX}: {checks['argued_ok_among_eligible']}); "
    )
    return head + "; ".join(parts)


def _fmt_alpha(value: float | None) -> str:
    if value is None:
        return "n/a"
    return ">1" if math.isinf(value) else f"{value:.2f}"


def _share(mask: np.ndarray, defined: np.ndarray) -> float | None:
    return float(mask[defined].mean()) if defined.any() else None


def _p7_estimator(
    samples: dict[str, PolicySamples],
    local: dict[str, dict],
    weights: np.ndarray,
    estimator: str,
) -> dict[str, Any]:
    """Prediction 7 under one estimator: the four α* (all forms), the collapsed run against
    0.30 at steps 200 and 300 and against the other two runs at step 200."""
    replicate = REPLICATES[estimator]
    targets = [policy_name(COLLAPSED_RUN, step) for step in P7_STEPS]
    others = [policy_name(run, P7_OTHER_STEP) for run in OTHER_RUNS]
    names = [name for name in targets + others if name in samples]
    values = {name: replicate(samples[name].bon, "all", weights) for name in names}
    levels: dict[str, Any] = {}
    for name in names:
        entry = local[name][estimator]["all"]
        point, boot = alpha_value(entry), values[name][1]
        levels[name] = {
            "alpha_star": point,
            "alpha_star_note": entry.get("alpha_star_note"),
            "ci": entry.get("alpha_star_ci"),
            "n_sides": side_counts(entry),
            "above_threshold": None if point is None else bool(point > P7_MIN_ALPHA + EPS),
            "share_resamples_above": _share(boot > P7_MIN_ALPHA, ~np.isnan(boot)),
        }
    contrasts: dict[str, Any] = {}
    for target in targets:
        for other in others:
            if target not in values or other not in values:
                continue
            _same_questions(samples[target].bon, samples[other].bon)
            a, b = levels[target]["alpha_star"], levels[other]["alpha_star"]
            boot_a, boot_b = values[target][1], values[other][1]
            with np.errstate(invalid="ignore"):
                diffs = boot_a - boot_b
            finite = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
            contrasts[f"{target} minus {other}"] = {
                "diff": a - b if finite else None,
                "ci": percentile_interval(diffs),
                "higher": None if a is None or b is None else bool(a > b),
                "share_resamples_higher": _share(
                    boot_a > boot_b, ~(np.isnan(boot_a) | np.isnan(boot_b))
                ),
                "like_for_like": samples[target].policy.step == P7_OTHER_STEP,
            }
    above = [levels[name]["above_threshold"] if name in levels else None for name in targets]
    higher = [c["higher"] for c in contrasts.values()]
    higher += [None] * (len(targets) * len(others) - len(contrasts))
    like = [c["higher"] for c in contrasts.values() if c["like_for_like"]]
    like += [None] * (len(others) - len(like))
    return {
        "verdict": verdict_of(above + higher),
        "verdict_above_threshold": verdict_of(above),
        "verdict_higher_than_others": verdict_of(higher),
        "verdict_higher_like_for_like": verdict_of(like),
        "levels": levels,
        "contrasts": contrasts,
        "missing": [name for name in targets + others if name not in samples],
    }


def _same_questions(first: BonSamples, second: BonSamples) -> None:
    if not np.array_equal(np.unique(first.base_id), np.unique(second.base_id)):
        raise ValueError(f"{first.model} and {second.model} are sampled on different questions")


def prediction_7(
    samples: dict[str, PolicySamples], local: dict[str, dict], weights: np.ndarray
) -> dict[str, Any]:
    by_estimator = {est: _p7_estimator(samples, local, weights, est) for est in ESTIMATORS}
    primary = by_estimator[PRIMARY]
    numbers = "; ".join(
        f"{name}: α* {_fmt_alpha(lv['alpha_star'])}{fmt_interval(lv['ci'], 1, 2)}"
        for name, lv in primary["levels"].items()
    )
    numbers += "".join(
        f"; {name}: {fmt_num(c['diff'], 2, sign=True)}{fmt_interval(c['ci'], 1, 2)}"
        for name, c in primary["contrasts"].items()
    )
    return {
        "statement": STATEMENTS["7"],
        "verdict": primary["verdict"],
        "primary_estimator": PRIMARY,
        "verdict_by_estimator": {est: by_estimator[est]["verdict"] for est in ESTIMATORS},
        "threshold": P7_MIN_ALPHA,
        "numbers": numbers,
        **by_estimator,
    }


def prediction_8(samples: dict[str, PolicySamples], weights: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "statement": STATEMENTS["8"],
        "alpha": P8_ALPHA,
        "min_gap": P8_MIN_GAP,
    }
    names = [COLLAPSED_RUN, *OTHER_RUNS]
    missing = [name for name in names if name not in samples]
    out["missing"] = missing
    if COLLAPSED_RUN not in samples:
        return {**out, "verdict": NOT_TESTABLE, "numbers": f"{COLLAPSED_RUN} was not sampled"}
    target = samples[COLLAPSED_RUN].bon
    per_q = question_means(mixed(target, P8_ALPHA), target)
    level = float(mixed(target, P8_ALPHA).mean())
    n_q = len(per_q)
    out["collapsed_run"] = {"policy": COLLAPSED_RUN, "mixed_reward": level}
    others: dict[str, Any] = {}
    other_q = []
    for run in OTHER_RUNS:
        if run not in samples:
            continue
        bon = samples[run].bon
        _same_questions(target, bon)
        q = question_means(mixed(bon, P8_ALPHA), bon)
        other_q.append(q)
        boot = weights @ (per_q - q) / n_q
        diff = float((per_q - q).mean())
        others[run] = {
            "mixed_reward": float(mixed(bon, P8_ALPHA).mean()),
            "diff": diff,
            "ci": percentile_interval(boot),
            "ok": bool(diff < -P8_MIN_GAP),
            "share_resamples_ok": float(np.mean(boot < -P8_MIN_GAP)),
            "diff_by_opinion": {
                opinion: float(
                    mixed(target, P8_ALPHA)[target.opinion == opinion].mean()
                    - mixed(bon, P8_ALPHA)[bon.opinion == opinion].mean()
                )
                for opinion in OPINIONS
            },
        }
    out["others"] = others
    if other_q:
        pooled = per_q - np.mean(other_q, axis=0)
        out["vs_mean_of_others"] = {
            "diff": float(pooled.mean()),
            "ci": percentile_interval(weights @ pooled / n_q),
        }
    oks: list[bool | None] = [others[run]["ok"] if run in others else None for run in OTHER_RUNS]
    out["verdict"] = verdict_of(oks)
    out["numbers"] = f"{COLLAPSED_RUN}: {level:.3f}" + "".join(
        f"; {run}: {o['mixed_reward']:.3f} (difference {o['diff']:+.3f}"
        f"{fmt_interval(o['ci'], 1, 3)})"
        for run, o in others.items()
    )
    return out


# ----------------------------------------------------------------------------- payload


def step0_break_even(base: dict[str, Any], bon_payload: dict[str, Any] | None) -> dict[str, Any]:
    """The base policy's break-even on the rollout sampler, next to the 64-sample Best-of-N
    estimate it replaces as the training-matched value."""
    out: dict[str, Any] = {
        "policy": BASE,
        "sampler": "temperature 1, top-p 1, top-k 0, float32, 8 samples per prompt",
        "raw_gap": base["raw_gap"],
        "push_group8_vs_rest": base["push"],
        "push_group8_vs_gold": base["push_vs_gold"],
        "push_group8_opinionated": base["push_opinionated"],
        "push_at_alpha": base["push_at_alpha"],
    }
    model = ((bon_payload or {}).get("models") or {}).get(BON_MODEL)
    if model:
        pushes = model.get("push") or {}
        out["best_of_n_estimate"] = {
            "model": BON_MODEL,
            "sampler": "top-p 0.95, default top-k, bfloat16, 64 samples per prompt",
            "raw_gap": {form: _alpha_only(model["raw_gap"][form]) for form in breakeven.FORM_KEYS},
            **{
                f"push_{group.replace('=', '')}_vs_rest": {
                    form: _alpha_only(curves[PUSH_SCOPE][form]) for form in breakeven.FORM_KEYS
                }
                for group, curves in pushes.items()
            },
        }
    return out


def _alpha_only(entry: dict[str, Any]) -> dict[str, Any]:
    return {key: entry.get(key) for key in ("alpha_star", "alpha_star_note", "alpha_star_ci")}


def definitions() -> dict[str, str]:
    return {
        "raw_gap": "primary estimator: alpha* = g_a / (g_a + g_i) on wrong-opinion samples "
        "(breakeven.raw_gap)",
        "push": f"zero crossing of the group-normalized push, groups of {GROUP} (the samples of "
        "a prompt), agreeing samples minus every other sample (breakeven.push, wrong_vs_rest)",
        "forms": "common.reply_form: bare < 20 characters before the last 'Answer:' (or in "
        "total); long >= 200 characters in total; medium the rest; nonbare = medium + long. "
        "Prediction 6 reads 'bare-letter' as bare and 'argued' as long",
        "enough_samples": f"at least {breakeven.MIN_SAMPLES} samples on each side of the "
        "contrast within the form (breakeven.MIN_SAMPLES)",
        "checkpoints": "every sampled trained policy, steps 100 / 200 / 300; the base policy "
        "(step 0) is reported and not counted in prediction 6",
        "mixed_reward": "(1 - alpha) * immediate + alpha * informed, mean over all samples "
        "(three opinion types in equal shares); persona_mix: 0.7 * agreeable + 0.3 * "
        "truth_seeking",
        "zero_variance_groups": "share of prompts whose 8 samples all got the same reward; "
        "under the persona mixture the persona-weighted share",
        "mixture_push": "P(p) = (1 - p) * push_agreeable + p * push_truth_seeking, each push "
        "normalized within its persona's ratings; p* is the root of P",
        "ci": "95% percentile bootstrap over questions, the same resamples for every policy; "
        "a null bound is a crossing beyond [0, 1]",
        "verdicts": "point estimates decide; 'not testable' when a needed policy or form is "
        "missing",
        "sampler_caveat": "samples are drawn in eval mode; the training rollouts had the LoRA "
        "dropout active",
    }


def analyze(
    samples: dict[str, PolicySamples],
    n_boot: int = 2000,
    seed: int = 0,
    bon_payload: dict[str, Any] | None = None,
    skipped: dict[str, str] | None = None,
    label: str = SAMPLES_LABEL,
) -> dict[str, Any]:
    """The full payload: per-policy numbers, the three predictions and the extras. ``label``
    says what the samples are, in place of their directories."""
    if not samples:
        raise ValueError("no rated sample files found")
    sizes = {name: s.bon.question_index()[1] for name, s in samples.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"policies are sampled on different numbers of questions: {sizes}")
    reference = next(iter(samples.values())).bon
    for other in samples.values():
        _same_questions(reference, other.bon)
    weights = resample_weights(next(iter(sizes.values())), n_boot, seed)
    policies = {name: s.policy for name, s in samples.items()}
    local = {name: analyze_policy(s, weights) for name, s in samples.items()}
    mixture = {name: mixture_push(s, weights) for name, s in samples.items()}
    payload: dict[str, Any] = {
        "definitions": definitions(),
        "inputs": {
            "n_boot": n_boot,
            "seed": seed,
            "samples": label,
            "policies": {name: s.source for name, s in samples.items()},
            "skipped": skipped or {},
        },
        "predictions": {
            "6": prediction_6(local, policies),
            "7": prediction_7(samples, local, weights),
            "8": prediction_8(samples, weights),
        },
        "policies": local,
        "mixture": {name: entry for name, entry in mixture.items() if entry is not None},
    }
    if BASE in local:
        payload["step0"] = step0_break_even(local[BASE], bon_payload)
    return jsonable(payload)


# ----------------------------------------------------------------------------- report


def _alpha_cell(entry: dict[str, Any]) -> str:
    return breakeven.fmt_alpha(entry)


def render_text(payload: dict[str, Any]) -> list[str]:
    """A plain-text digest of the payload for the terminal."""
    lines = ["Predictions (primary estimator: raw gap)"]
    for key, pred in payload["predictions"].items():
        lines.append(f"  {key}. {pred['verdict']}: {pred['numbers']}")
        by_est = pred.get("verdict_by_estimator")
        if by_est:
            lines.append(f"     by estimator: {by_est}")
    lines += [
        "",
        "Per policy: sycophancy of samples | raw gap all / bare / long | push all / "
        "bare / long | own reward | zero-variance groups",
    ]
    for name, p in payload["policies"].items():
        raw, push = p["raw_gap"], p["push"]
        own = p.get("own_reward_mean")
        flat = p.get("own_zero_variance_groups")
        lines.append(
            f"  {name}: {fmt_pct(p['sycophancy']['rate'])} | "
            f"{_alpha_cell(raw['all'])} / {_alpha_cell(raw['bare'])} / {_alpha_cell(raw['long'])}"
            f" | {_alpha_cell(push['all'])} / {_alpha_cell(push['bare'])} / "
            f"{_alpha_cell(push['long'])} | {fmt_num(own, 3)} | {fmt_pct(flat)}"
        )
    for name, entry in payload.get("mixture", {}).items():
        form = entry["forms"]["all"]
        if form.get("push_agreeable"):
            root = form["p_star"]
            lines.append(
                f"  mixture push, {name}: agreeable {form['push_agreeable']['value']:+.3f}, "
                f"truth-seeking {form['push_truth_seeking']['value']:+.3f}, P({entry['p_train']})"
                f" {form['net_push_at_p_train']['value']:+.3f}, p* "
                f"{fmt_num(root, 2) if root is not None else form['p_star_note']}"
                f"{fmt_interval(form['p_star_ci'], 1, 2)}"
            )
    return lines


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI: ``python -m goodhart.analysis.prereg_checkpoints --samples <dir> [<dir> ...]``."""
    parser = argparse.ArgumentParser(description="pre-registered predictions 6-8 (stage M3)")
    parser.add_argument(
        "--samples",
        nargs="+",
        default=[SAMPLES_DIR],
        metavar="DIR",
        help="sample_rate outputs: <DIR>/<policy>/samples.jsonl",
    )
    parser.add_argument(
        "--label",
        default=SAMPLES_LABEL,
        help="what the --samples directories are, written to the output in their place",
    )
    parser.add_argument("--out", default=OUT_FILE, help="JSON file to write")
    parser.add_argument(
        "--bon-breakeven",
        default=BON_BREAKEVEN,
        help="breakeven.json of the Best-of-N analysis, for the step-0 comparison",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    samples, skipped = load_policies(args.samples)
    bon_file = Path(args.bon_breakeven)
    bon_payload = read_json(bon_file) if bon_file.is_file() else None
    payload = analyze(samples, args.n_boot, args.seed, bon_payload, skipped, args.label)
    write_json(args.out, payload)
    print("\n".join(render_text(payload)))
    print(f"\nwrote {args.out}")
    return payload


if __name__ == "__main__":
    main()

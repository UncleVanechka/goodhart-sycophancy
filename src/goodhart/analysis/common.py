"""Loaders and small statistics shared by the post-hoc analyses.

GRPO runs live in ``results/<stage>/<run>/``: ``summary.json``, ``curve.jsonl`` (one evaluation
every 25 steps) and ``train_log.jsonl`` (trainer statistics every 5 steps). Best-of-N per-sample
files (``<slug>_samples.jsonl``, one row per sample with both ratings) are joined to the prompt
records for the gold letter and the letter the user stated.

Best-of-N picks here are *expected* picks: ties for the highest score are broken uniformly at
random and every quantity of the pick is averaged over the tied samples, which removes the
tie-break noise of a single draw.
"""

from __future__ import annotations

import glob
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from goodhart.io import model_slug, read_json, read_jsonl

RUN_STAGES = ("grpo", "grpo_first_pass")
OPINIONS = ("none", "correct", "wrong")
FORMS = ("bare", "medium", "long")
BARE_MAX_CHARS = 20
LONG_MIN_CHARS = 200
TIE_TOL = 1e-9

# the marker of the parser's primary rule ("Answer: X", optional markdown around the colon)
_ANSWER_MARKER = re.compile(r"answer\**\s*:", re.IGNORECASE)


# ----------------------------------------------------------------------------- GRPO runs


@dataclass
class Run:
    """One GRPO run: its condition, evaluation curve and trainer log."""

    name: str
    condition: str
    seed: int | None
    stage: str
    model: str
    curve: list[dict]
    train_log: list[dict]

    @property
    def first_pass(self) -> bool:
        """The first-pass runs whose evaluation logged the training reward, not the proxy."""
        return self.stage == "grpo_first_pass"

    @property
    def has_proxy(self) -> bool:
        """Whether every evaluation logged ``proxy_mean`` (the immediate rating)."""
        return bool(self.curve) and all("proxy_mean" in row for row in self.curve)

    @property
    def steps(self) -> np.ndarray:
        """Evaluation steps of ``curve.jsonl``."""
        return np.asarray([float(row["step"]) for row in self.curve])

    def series(self, key: str) -> np.ndarray:
        """``key`` at every evaluation, NaN where it was not logged."""
        return np.asarray([float(row.get(key, math.nan)) for row in self.curve])

    def log_series(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """``(steps, values)`` of a trainer statistic, rows without it skipped."""
        rows = [row for row in self.train_log if row.get(key) is not None]
        steps = np.asarray([float(row["step"]) for row in rows])
        values = np.asarray([float(row[key]) for row in rows])
        return steps, values


def condition_of(summary: dict) -> str:
    """``persona_mix`` for a persona mixture, else ``alpha=<α>`` (``alpha=0`` for immediate)."""
    reward = (summary.get("config") or {}).get("reward") or {}
    kind = summary.get("reward_kind") or reward.get("kind")
    if kind == "persona_mix":
        return "persona_mix"
    alpha = summary.get("alpha", reward.get("alpha", 0.0))
    return f"alpha={float(alpha or 0.0):g}"


def condition_label(condition: str) -> str:
    """Condition as written in the report: ``α = 0.3``, ``persona mix``."""
    if condition == "persona_mix":
        return "persona mix"
    if condition.startswith("alpha="):
        return f"α = {condition.split('=', 1)[1]}"
    return condition


def condition_order(condition: str) -> tuple[int, float, str]:
    """Sort key: α conditions by value, then everything else by name."""
    if condition.startswith("alpha="):
        try:
            return (0, float(condition.split("=", 1)[1]), "")
        except ValueError:
            pass
    return (1, 0.0, condition)


def load_runs(results_dir: str | Path, stages: Sequence[str] = RUN_STAGES) -> list[Run]:
    """Every run under ``results_dir/<stage>/*/`` that has a non-empty ``curve.jsonl``."""
    results_dir = Path(results_dir)
    runs: list[Run] = []
    for stage in stages:
        for curve_file in sorted((results_dir / stage).glob("*/curve.jsonl")):
            run_dir = curve_file.parent
            curve = sorted(read_jsonl(curve_file), key=lambda row: float(row["step"]))
            if not curve:
                continue
            summary_file = run_dir / "summary.json"
            summary = read_json(summary_file) if summary_file.is_file() else {}
            log_file = run_dir / "train_log.jsonl"
            train_log = read_jsonl(log_file) if log_file.is_file() else []
            seed = (summary.get("config") or {}).get("seed")
            runs.append(
                Run(
                    name=f"{stage}/{run_dir.name}",
                    condition=condition_of(summary),
                    seed=int(seed) if seed is not None else None,
                    stage=stage,
                    model=model_slug(summary["model"]) if summary.get("model") else "",
                    curve=curve,
                    train_log=sorted(train_log, key=lambda row: float(row["step"])),
                )
            )
    runs.sort(key=lambda r: (condition_order(r.condition), r.seed or 0, r.stage, r.name))
    return runs


def runs_by_condition(runs: Sequence[Run]) -> dict[str, list[Run]]:
    """Runs grouped by condition, conditions in report order."""
    grouped: dict[str, list[Run]] = defaultdict(list)
    for run in runs:
        grouped[run.condition].append(run)
    return {key: grouped[key] for key in sorted(grouped, key=condition_order)}


def first_crossing(values: Sequence[float], threshold: float) -> float | None:
    """Fractional index at which ``values`` first reaches ``threshold``, linear between points.

    ``None`` when it never does. NaN points are skipped as crossings; a crossing right after a
    NaN point is placed on the crossing point itself.
    """
    arr = np.asarray(values, dtype=float)
    for i, value in enumerate(arr):
        if np.isnan(value) or value < threshold:
            continue
        if i == 0:
            return 0.0
        prev = arr[i - 1]
        if np.isnan(prev) or value == prev:
            return float(i)
        return (i - 1) + (threshold - prev) / (value - prev)
    return None


def value_at(position: float, values: Sequence[float]) -> float:
    """``values`` linearly interpolated at a fractional index."""
    arr = np.asarray(values, dtype=float)
    lo = int(math.floor(position))
    hi = min(lo + 1, len(arr) - 1)
    frac = position - lo
    if frac <= 0 or hi == lo:
        return float(arr[lo])
    return float(arr[lo] + frac * (arr[hi] - arr[lo]))


def window_mean(
    log_steps: np.ndarray, log_values: np.ndarray, at: Sequence[float], window: float
) -> np.ndarray:
    """Mean of the logged values with ``s - window < step <= s`` for every ``s`` in ``at``."""
    out = np.full(len(at), np.nan)
    for i, step in enumerate(at):
        mask = (log_steps > step - window) & (log_steps <= step)
        if mask.any():
            out[i] = float(log_values[mask].mean())
    return out


# ----------------------------------------------------------------------------- Best-of-N


def reply_form(text: str) -> str:
    """Length class of a reply: ``bare``, ``medium`` or ``long``.

    The *body* is the text before the last ``Answer:`` marker (the marker the answer parser
    reads first), or the whole text when there is none. ``bare``: the stripped body is shorter
    than 20 characters, i.e. a letter, possibly with the option restated. Otherwise ``long``
    when the whole stripped reply has at least 200 characters, else ``medium``.
    """
    text = text or ""
    matches = list(_ANSWER_MARKER.finditer(text))
    body = text[: matches[-1].start()] if matches else text
    if len(body.strip()) < BARE_MAX_CHARS:
        return "bare"
    if len(text.strip()) >= LONG_MIN_CHARS:
        return "long"
    return "medium"


@dataclass
class BonSamples:
    """Best-of-N samples of one model as ``[record, sample]`` arrays."""

    model: str
    record_ids: list[str]
    base_id: np.ndarray
    opinion: np.ndarray
    gold: np.ndarray
    opinion_answer: np.ndarray
    parsed: np.ndarray
    immediate: np.ndarray
    informed: np.ndarray
    form: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.parsed.shape[1])

    @property
    def agree(self) -> np.ndarray:
        """``[R, S]``: the parsed letter is the one the user stated."""
        stated = np.array([a is not None for a in self.opinion_answer])[:, None]
        return stated & (self.parsed == self.opinion_answer[:, None])

    @property
    def correct(self) -> np.ndarray:
        """``[R, S]``: the parsed letter is the gold letter."""
        return self.parsed == self.gold[:, None]

    @property
    def answered(self) -> np.ndarray:
        """``[R, S]``: a letter was parsed."""
        return np.vectorize(lambda p: p is not None, otypes=[bool])(self.parsed)

    def form_mask(self, form: str) -> np.ndarray:
        """``[R, S]`` mask of the samples of ``form`` (``all`` selects every sample)."""
        if form == "all":
            return np.ones(self.parsed.shape, dtype=bool)
        return self.form == form

    def question_index(self) -> tuple[np.ndarray, int]:
        """Index of every record's question (``base_id``) and the number of questions."""
        _, inverse = np.unique(self.base_id, return_inverse=True)
        return inverse, int(inverse.max()) + 1 if len(inverse) else 0


def load_bon_samples(
    path: str | Path, records_by_id: dict[str, dict], model: str | None = None
) -> BonSamples:
    """Read a ``<slug>_samples.jsonl`` file; every record must have the same number of samples."""
    path = Path(path)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        grouped[row["record_id"]].append(row)
    record_ids = sorted(grouped)
    sizes = {len(rows) for rows in grouped.values()}
    if len(sizes) != 1:
        raise ValueError(f"{path}: records have different sample counts {sorted(sizes)}")
    missing = [rid for rid in record_ids if rid not in records_by_id]
    if missing:
        raise ValueError(f"{path}: {len(missing)} record ids not in the prompts: {missing[:3]}")
    parsed, immediate, informed, form = [], [], [], []
    for rid in record_ids:
        rows = sorted(grouped[rid], key=lambda r: int(r["sample_idx"]))
        parsed.append([r.get("parsed") for r in rows])
        immediate.append([float(r["immediate"]) for r in rows])
        informed.append([float(r["informed"]) for r in rows])
        form.append([reply_form(r.get("text", "")) for r in rows])
    recs = [records_by_id[rid] for rid in record_ids]

    def column(key: str) -> np.ndarray:
        return np.array([r.get(key) for r in recs], dtype=object)

    slug = model or path.name.removesuffix(".jsonl").removesuffix("_samples")
    return BonSamples(
        model=slug,
        record_ids=record_ids,
        base_id=column("base_id"),
        opinion=column("opinion"),
        gold=column("gold"),
        opinion_answer=column("opinion_answer"),
        parsed=np.array(parsed, dtype=object),
        immediate=np.array(immediate, dtype=float),
        informed=np.array(informed, dtype=float),
        form=np.array(form, dtype=object),
    )


def find_samples_file(results_dir: Path, runner_results: Path | None, basename: str) -> Path:
    """``results/bon/<basename>`` if present, else the newest copy among fetched runner outputs."""
    local = results_dir / "bon" / basename
    if local.is_file():
        return local
    if runner_results is not None:
        hits = sorted(glob.glob(str(runner_results / "*" / "results" / "bon" / basename)))
        if hits:
            return Path(hits[-1])
    raise FileNotFoundError(f"no {basename} under {results_dir / 'bon'} or {runner_results}")


def selection_weights(scores: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """``[R, S]`` probability that each sample is the pick: the highest score within ``pool``,
    ties broken uniformly at random. Rows with an empty pool get zero weights."""
    masked = np.where(pool, scores, -np.inf)
    best = masked.max(axis=1, keepdims=True)
    ties = pool & (masked >= best - TIE_TOL)
    count = ties.sum(axis=1, keepdims=True)
    return np.where(count > 0, ties / np.maximum(count, 1), 0.0)


def first_n_pool(shape: tuple[int, int], n: int) -> np.ndarray:
    """``[R, S]`` mask of the first ``n`` samples of every record."""
    pool = np.zeros(shape, dtype=bool)
    pool[:, : min(n, shape[1])] = True
    return pool


# ----------------------------------------------------------------------------- statistics


def bootstrap_over_runs(values: Sequence[float], n_boot: int, seed: int) -> dict[str, Any]:
    """Mean of per-run values with a 95 % percentile bootstrap over runs (NaN values dropped).

    With one run the interval is ``None``; with two or three it cannot leave the range of the
    observed values, so it shows the spread of the runs rather than a population interval.
    """
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    out: dict[str, Any] = {"mean": float(arr.mean()) if arr.size else None, "n_runs": int(arr.size)}
    out["ci"] = None
    if arr.size >= 2:
        idx = np.random.default_rng(seed).integers(0, arr.size, size=(n_boot, arr.size))
        out["ci"] = percentile_interval(arr[idx].mean(axis=1))
    return out


def bootstrap_difference(
    first: Sequence[float], second: Sequence[float], n_boot: int, seed: int
) -> dict[str, Any]:
    """``mean(second) - mean(first)``, runs resampled independently within each group; no
    interval unless both groups have at least two runs."""
    a = np.asarray([v for v in first if v is not None and np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in second if v is not None and np.isfinite(v)], dtype=float)
    if not a.size or not b.size:
        return {"diff": None, "ci": None, "n_runs": [int(a.size), int(b.size)]}
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, size=(n_boot, b.size))].mean(axis=1)
    ci = percentile_interval(mb - ma) if min(a.size, b.size) >= 2 else None
    return {"diff": float(b.mean() - a.mean()), "ci": ci, "n_runs": [int(a.size), int(b.size)]}


def resample_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """``[n_boot, n]`` multiplicities of ``n_boot`` with-replacement resamples of ``n`` units."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    weights = np.zeros((n_boot, n))
    np.add.at(weights, (np.arange(n_boot)[:, None], idx), 1.0)
    return weights


def percentile_interval(values: Sequence[float], level: float = 0.95) -> list[float] | None:
    """Percentile interval of the non-NaN ``values``; ``inf`` stays ``inf`` (a bound outside
    the grid). ``None`` when nothing is left."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if not arr.size:
        return None
    lo, hi = np.quantile(arr, [(1 - level) / 2, (1 + level) / 2], method="inverted_cdf")
    return [float(lo), float(hi)]


def ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den`` with NaN where ``den == 0``."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def downward_zero_crossings(grid: Sequence[float], values: np.ndarray) -> np.ndarray:
    """First grid point where each row of ``values`` falls to zero or below, linear between
    grid points: ``grid[0]`` when a row starts at or below zero, ``inf`` when it stays
    positive on the whole grid, NaN when the row is all NaN."""
    g = np.asarray(grid, dtype=float)
    v = np.atleast_2d(np.asarray(values, dtype=float))
    out = np.full(v.shape[0], np.inf)
    for i, row in enumerate(v):
        if np.isnan(row).all():
            out[i] = np.nan
            continue
        hits = np.flatnonzero(row <= 0)
        if not hits.size:
            continue
        j = int(hits[0])
        if j == 0 or np.isnan(row[j - 1]):
            out[i] = g[j]
            continue
        prev, cur = row[j - 1], row[j]
        out[i] = g[j - 1] + (g[j] - g[j - 1]) * prev / (prev - cur)
    return out


def jsonable(obj: Any) -> Any:
    """``obj`` with numpy scalars made Python, tuples made lists and NaN / inf made ``None``."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


# ----------------------------------------------------------------------------- formatting


def fmt_pct(value: float | None, digits: int = 0) -> str:
    """``0.531`` -> ``53%``; ``None`` / NaN -> ``—``."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100 * value:.{digits}f}%"


def fmt_points(value: float | None, digits: int = 1) -> str:
    """A rate difference in percentage points with a sign: ``+7.5``."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{100 * value:+.{digits}f}"


def fmt_num(value: float | None, digits: int = 2, sign: bool = False) -> str:
    """A plain number, ``—`` when missing."""
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def fmt_interval(ci: Sequence[float | None] | None, scale: float = 1.0, digits: int = 0) -> str:
    """`` [lo, hi]`` scaled; a bound that is ``inf`` or ``None`` (a zero crossing beyond the α
    grid) prints as ``>1``."""
    if not ci:
        return ""

    def one(v: float | None) -> str:
        if v is None or (isinstance(v, float) and math.isinf(v)):
            return ">1"
        return f"{scale * v:.{digits}f}"

    return f" [{one(ci[0])}, {one(ci[1])}]"

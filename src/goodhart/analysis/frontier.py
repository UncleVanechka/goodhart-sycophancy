"""(a) The Goodhart frontier: sycophancy of the GRPO runs at matched optimization.

Two x axes, both read per run from what was logged:

- **proxy gain**: the immediate rating (``proxy_mean`` over all validation prompts) minus its
  value at step 0;
- **√KL**: the square root of the trainer's per-token KL to the reference model, averaged over
  the logged steps in ``(s - 25, s]`` for the evaluation at step ``s`` (0 at step 0).

At every threshold a run's sycophancy, accuracy under a wrong opinion and informed rating are
linearly interpolated at the *first* evaluation where the axis reaches the threshold. A run
that never reaches it is *not reached* and drops out of that threshold's condition mean, so
means at high thresholds describe only the runs that got there. Condition means carry a 95 %
percentile bootstrap over runs: runs are the unit, evaluation points of one run are not
independent. Runs without the proxy (the first pass) are left out of the proxy axis and kept on
the √KL axis.

Best-of-N points come from the per-sample files with the expected pick (ties averaged): proxy
gain and sycophancy change are measured against a single random sample, so selection and
training are both compared with their own start.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from goodhart.analysis.common import (
    BonSamples,
    Run,
    bootstrap_difference,
    bootstrap_over_runs,
    condition_label,
    first_crossing,
    first_n_pool,
    fmt_interval,
    fmt_num,
    fmt_pct,
    fmt_points,
    runs_by_condition,
    selection_weights,
    value_at,
    window_mean,
)

GAIN_THRESHOLDS = (0.15, 0.30, 0.45, 0.60)
SQRT_KL_THRESHOLDS = (0.2, 0.3, 0.4, 0.6)
GAIN_BIN_EDGES = (-math.inf, 0.05, 0.20, 0.35, 0.50, 0.65, math.inf)
KL_WINDOW = 25
METRICS = ("sycophancy_rate", "sycophancy_change", "gold_accuracy@wrong", "informed_mean")
REFERENCE_CONDITION = "alpha=0"
BON_ALPHAS = (0.0, 0.3, 0.6, 1.0)
BON_N = (1, 4, 16, 64)


def threshold_key(value: float) -> str:
    """Threshold as a JSON key: ``0.15``."""
    return f"{value:.2f}"


def proxy_gain(run: Run) -> np.ndarray:
    """Immediate rating at every evaluation minus its step-0 value (NaN when not logged)."""
    proxy = run.series("proxy_mean")
    return proxy - proxy[0]


def sqrt_kl(run: Run, window: float = KL_WINDOW) -> np.ndarray:
    """√ of the mean logged KL over ``(s - window, s]`` at every evaluation step ``s``."""
    steps = run.steps
    log_steps, kl = run.log_series("kl")
    mean_kl = window_mean(log_steps, kl, steps, window)
    mean_kl[steps <= 0] = 0.0
    return np.sqrt(np.clip(mean_kl, 0.0, None))


def read_at(run: Run, axis: np.ndarray, threshold: float) -> dict[str, Any]:
    """Evaluation metrics interpolated where ``axis`` first reaches ``threshold``."""
    position = first_crossing(axis, threshold)
    if position is None:
        return {"reached": False}
    sycophancy = run.series("sycophancy_rate")
    out: dict[str, Any] = {"reached": True, "step": value_at(position, run.steps)}
    out["sycophancy_rate"] = value_at(position, sycophancy)
    out["sycophancy_change"] = out["sycophancy_rate"] - float(sycophancy[0])
    out["gold_accuracy@wrong"] = value_at(position, run.series("gold_accuracy@wrong"))
    out["informed_mean"] = value_at(position, run.series("informed_mean"))
    return out


def run_frontier(
    run: Run, gain_thresholds: Sequence[float], kl_thresholds: Sequence[float]
) -> dict[str, Any]:
    """Per-run reads on both axes, plus how far the run got on each."""
    kl_axis = sqrt_kl(run)
    out: dict[str, Any] = {
        "condition": run.condition,
        "seed": run.seed,
        "first_pass": run.first_pass,
        "start_sycophancy": float(run.series("sycophancy_rate")[0]),
        "max_sqrt_kl": float(np.nanmax(kl_axis)),
        "sqrt_kl": {threshold_key(t): read_at(run, kl_axis, t) for t in kl_thresholds},
    }
    if run.has_proxy:
        gain = proxy_gain(run)
        out["max_proxy_gain"] = float(np.nanmax(gain))
        out["final_proxy_gain"] = float(gain[-1])
        out["proxy_gain"] = {threshold_key(t): read_at(run, gain, t) for t in gain_thresholds}
    else:
        out["proxy_gain"] = None
    return out


def _eligible(per_run: dict[str, Any], names: Sequence[str], axis: str) -> list[str]:
    return [name for name in names if per_run[name].get(axis) is not None]


def _values(per_run: dict[str, Any], names: Sequence[str], axis: str, key: str, metric: str):
    return [
        per_run[name][axis][key].get(metric) if per_run[name][axis][key]["reached"] else None
        for name in names
    ]


def condition_frontier(
    per_run: dict[str, Any],
    groups: dict[str, list[str]],
    axis: str,
    thresholds: Sequence[float],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Condition means at every threshold, with the runs that did not reach it counted."""
    out: dict[str, Any] = {}
    for condition, names in groups.items():
        eligible = _eligible(per_run, names, axis)
        if not eligible:
            continue
        cond: dict[str, Any] = {}
        for t in thresholds:
            key = threshold_key(t)
            reached = [n for n in eligible if per_run[n][axis][key]["reached"]]
            entry: dict[str, Any] = {"n_reached": len(reached), "n_runs": len(eligible)}
            entry["not_reached"] = [n for n in eligible if n not in reached]
            for metric in METRICS:
                vals = _values(per_run, eligible, axis, key, metric)
                entry[metric] = bootstrap_over_runs(
                    [v for v in vals if v is not None], n_boot, seed
                )
            cond[key] = entry
        out[condition] = cond
    return out


def differences_vs_reference(
    per_run: dict[str, Any],
    groups: dict[str, list[str]],
    axis: str,
    thresholds: Sequence[float],
    n_boot: int,
    seed: int,
    reference: str = REFERENCE_CONDITION,
) -> dict[str, Any]:
    """``condition − reference`` at every threshold, bootstrap over runs within each group."""
    if reference not in groups:
        return {}
    ref = _eligible(per_run, groups[reference], axis)
    out: dict[str, Any] = {}
    for condition, names in groups.items():
        eligible = _eligible(per_run, names, axis)
        if condition == reference or not eligible:
            continue
        out[condition] = {}
        for t in thresholds:
            key = threshold_key(t)
            out[condition][key] = {
                metric: bootstrap_difference(
                    [v for v in _values(per_run, ref, axis, key, metric) if v is not None],
                    [v for v in _values(per_run, eligible, axis, key, metric) if v is not None],
                    n_boot,
                    seed,
                )
                for metric in METRICS
            }
    return out


def binned_sycophancy(
    runs: Sequence[Run], edges: Sequence[float] = GAIN_BIN_EDGES
) -> dict[str, list[dict[str, Any]]]:
    """Sycophancy of every evaluation after step 0, binned by proxy gain, per condition.

    ``mean_points`` pools the evaluation points; ``mean_of_runs`` first averages within a run,
    so a run with many points in a bin does not dominate it.
    """
    out: dict[str, list[dict[str, Any]]] = {}
    for condition, group in runs_by_condition([r for r in runs if r.has_proxy]).items():
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            per_run: list[float] = []
            points: list[float] = []
            for run in group:
                gain, syc = proxy_gain(run), run.series("sycophancy_rate")
                mask = (run.steps > 0) & (gain >= lo) & (gain < hi)
                if mask.any():
                    points.extend(syc[mask].tolist())
                    per_run.append(float(syc[mask].mean()))
            rows.append(
                {
                    "bin": [lo, hi],
                    "n_points": len(points),
                    "n_runs": len(per_run),
                    "mean_points": float(np.mean(points)) if points else None,
                    "mean_of_runs": float(np.mean(per_run)) if per_run else None,
                }
            )
        out[condition] = rows
    return out


# ----------------------------------------------------------------------------- Best-of-N


def bon_reference(bon: BonSamples) -> dict[str, float]:
    """Metrics of a single random sample (the expectation of N = 1 over all samples)."""
    wrong = bon.opinion == "wrong"
    return {
        "proxy_mean": float(bon.immediate.mean()),
        "informed_mean": float(bon.informed.mean()),
        "sycophancy_rate": float(bon.agree[wrong].mean()),
        "gold_accuracy@wrong": float(bon.correct[wrong].mean()),
        "gold_accuracy": float(bon.correct.mean()),
    }


def bon_point(bon: BonSamples, alpha: float, n: int) -> dict[str, float]:
    """Metrics of the expected Best-of-``n`` pick (first ``n`` samples) by the mixed reward."""
    scores = (1.0 - alpha) * bon.immediate + alpha * bon.informed
    weights = selection_weights(scores, first_n_pool(scores.shape, n))
    wrong = bon.opinion == "wrong"

    def pick(values: np.ndarray) -> np.ndarray:
        return (weights * values).sum(axis=1)

    return {
        "proxy_mean": float(pick(bon.immediate).mean()),
        "informed_mean": float(pick(bon.informed).mean()),
        "sycophancy_rate": float(pick(bon.agree)[wrong].mean()),
        "gold_accuracy@wrong": float(pick(bon.correct)[wrong].mean()),
        "gold_accuracy": float(pick(bon.correct).mean()),
    }


def bon_frontier(
    bon: BonSamples,
    alphas: Sequence[float] = BON_ALPHAS,
    n_values: Sequence[int] = BON_N,
    gain_thresholds: Sequence[float] = GAIN_THRESHOLDS,
) -> dict[str, Any]:
    """Best-of-N points for every α and N, and the α = 0 curve read at the proxy thresholds.

    The α = 0 curve runs through the random-sample reference (gain 0) and N = 4, 16, 64; the
    first sample alone (N = 1) is left out of it as a noisy copy of the reference.
    """
    ref = bon_reference(bon)
    first = bon_point(bon, 0.0, 1)
    points: dict[str, dict[str, Any]] = {}
    for alpha in alphas:
        per_n: dict[str, Any] = {}
        for n in n_values:
            p = bon_point(bon, float(alpha), int(n))
            p["proxy_gain"] = p["proxy_mean"] - ref["proxy_mean"]
            p["proxy_gain_vs_first_sample"] = p["proxy_mean"] - first["proxy_mean"]
            p["sycophancy_change"] = p["sycophancy_rate"] - ref["sycophancy_rate"]
            p["informed_change"] = p["informed_mean"] - ref["informed_mean"]
            per_n[str(int(n))] = p
        points[f"alpha={float(alpha):g}"] = per_n
    curve_n = [n for n in n_values if n > 1]
    gains = [0.0] + [points["alpha=0"][str(n)]["proxy_gain"] for n in curve_n]
    changes = [0.0] + [points["alpha=0"][str(n)]["sycophancy_change"] for n in curve_n]
    at_threshold: dict[str, Any] = {}
    for t in list(gain_thresholds):
        position = first_crossing(gains, t)
        at_threshold[threshold_key(t)] = (
            {"reached": False}
            if position is None
            else {"reached": True, "sycophancy_change": value_at(position, changes)}
        )
    return {
        "model": bon.model,
        "n_questions": int(len(set(bon.base_id.tolist()))),
        "reference": ref,
        "first_sample": first,
        "points": points,
        "alpha0_at_proxy_gain": at_threshold,
    }


# ----------------------------------------------------------------------------- driver


def analyze(
    runs: Sequence[Run],
    bons: Sequence[BonSamples],
    gain_thresholds: Sequence[float] = GAIN_THRESHOLDS,
    kl_thresholds: Sequence[float] = SQRT_KL_THRESHOLDS,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """The full frontier payload: per run, per condition, differences, bins and Best-of-N.

    The Best-of-64 (α = 0) proxy gain of the model the runs trained is added to the proxy
    thresholds, so training is also read at the gain selection reached.
    """
    bon_out = {b.model: bon_frontier(b, gain_thresholds=gain_thresholds) for b in bons}
    run_models = {r.model for r in runs}
    matched = next((m for m in bon_out if m in run_models), None)
    thresholds = list(gain_thresholds)
    bon_gain = None
    if matched is not None:
        point = bon_out[matched]["points"]["alpha=0"][str(max(BON_N))]
        bon_gain = float(point["proxy_gain"])
        bon_out[matched]["bon64"] = {
            "proxy_gain": bon_gain,
            "sycophancy_change": point["sycophancy_change"],
        }
        if threshold_key(bon_gain) not in {threshold_key(t) for t in thresholds}:
            thresholds.append(round(bon_gain, 2))
    per_run = {r.name: run_frontier(r, thresholds, kl_thresholds) for r in runs}
    groups = {cond: [r.name for r in group] for cond, group in runs_by_condition(runs).items()}
    return {
        "definitions": {
            "proxy_gain": "proxy_mean (immediate rating, all validation prompts) minus step 0",
            "sqrt_kl": f"sqrt of the mean logged per-token KL over the {KL_WINDOW} steps "
            "up to each evaluation",
            "reading": "linear interpolation at the first evaluation reaching the threshold",
            "condition_ci": "95% percentile bootstrap over runs that reached the threshold",
            "bon": "expected Best-of-N pick (ties averaged) against a single random sample",
        },
        "proxy_thresholds": [threshold_key(t) for t in thresholds],
        "bon_matched_threshold": threshold_key(bon_gain) if bon_gain is not None else None,
        "bon_matched_model": matched,
        "sqrt_kl_thresholds": [threshold_key(t) for t in kl_thresholds],
        "runs": per_run,
        "conditions": {
            "proxy_gain": condition_frontier(
                per_run, groups, "proxy_gain", thresholds, n_boot, seed
            ),
            "sqrt_kl": condition_frontier(per_run, groups, "sqrt_kl", kl_thresholds, n_boot, seed),
        },
        "differences_vs_alpha_0": {
            "proxy_gain": differences_vs_reference(
                per_run, groups, "proxy_gain", thresholds, n_boot, seed
            ),
            "sqrt_kl": differences_vs_reference(
                per_run, groups, "sqrt_kl", kl_thresholds, n_boot, seed
            ),
        },
        "mix_vs_alpha_0.3": differences_vs_reference(
            per_run,
            {c: groups[c] for c in ("alpha=0.3", "persona_mix") if c in groups},
            "proxy_gain",
            thresholds,
            n_boot,
            seed,
            reference="alpha=0.3",
        ),
        "binned": binned_sycophancy(runs),
        "bon": bon_out,
    }


# ----------------------------------------------------------------------------- report


def _cell(entry: dict[str, Any], metric: str, pct: bool = True) -> str:
    """Mean [interval] (runs with a value / runs on this axis)."""
    stat = entry.get(metric) or {}
    mean = stat.get("mean")
    if mean is None:
        if entry["n_reached"]:
            return f"not logged ({entry['n_reached']} reached)"
        return f"not reached (0/{entry['n_runs']})"
    if pct:
        text = fmt_pct(mean) + fmt_interval(stat.get("ci"), 100)
    else:
        text = fmt_num(mean) + fmt_interval(stat.get("ci"), 1, 2)
    return f"{text} ({stat['n_runs']}/{entry['n_runs']})"


def _diff_cell(entry: dict[str, Any], metric: str = "sycophancy_rate") -> str:
    stat = entry.get(metric) or {}
    if stat.get("diff") is None:
        return "—"
    return fmt_points(stat["diff"], 0) + fmt_interval(stat.get("ci"), 100)


def _table(
    conditions: dict[str, Any], keys: Sequence[str], metric: str, header: str, pct: bool = True
) -> list[str]:
    names = list(conditions)
    lines = [
        f"| {header} | " + " | ".join(condition_label(c) for c in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for key in keys:
        cells = [_cell(conditions[c][key], metric, pct) for c in names]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return lines


def interpretation(payload: dict[str, Any]) -> list[str]:
    """Plain-language bullets on the proxy-axis comparisons with α = 0."""
    lines: list[str] = []
    conds = payload["conditions"]["proxy_gain"]
    diffs = payload["differences_vs_alpha_0"]["proxy_gain"]
    ref = conds.get(REFERENCE_CONDITION, {})
    for key in payload["proxy_thresholds"]:
        if key not in ref or ref[key]["n_reached"] < 1:
            continue
        parts = []
        for cond, per_key in diffs.items():
            d = per_key[key]["sycophancy_rate"]
            n_reached, n_runs = conds[cond][key]["n_reached"], conds[cond][key]["n_runs"]
            if d["diff"] is None:
                parts.append(f"{condition_label(cond)} never got there (0/{n_runs})")
                continue
            ci = d.get("ci")
            verdict = (
                "interval excludes 0"
                if ci and (ci[0] > 0 or ci[1] < 0)
                else "interval includes 0"
                if ci
                else "no interval"
            )
            runs_text = "1 run" if n_reached == 1 else f"{n_reached} runs"
            parts.append(
                f"{condition_label(cond)} {fmt_points(d['diff'], 0)} points"
                f"{fmt_interval(ci, 100)} ({runs_text}, {verdict})"
            )
        ref_syc = ref[key]["sycophancy_rate"]["mean"]
        lines.append(
            f"- At proxy gain +{key}, α = 0 runs ({ref[key]['n_reached']}/{ref[key]['n_runs']}) "
            f"are at {fmt_pct(ref_syc)} sycophancy; relative to them: " + "; ".join(parts) + "."
        )
    mix_vs = (payload.get("mix_vs_alpha_0.3") or {}).get("persona_mix") or {}
    pairs, excludes = [], []
    for key in payload["proxy_thresholds"]:
        d = (mix_vs.get(key) or {}).get("sycophancy_rate") or {}
        if d.get("diff") is None:
            continue
        ci = d.get("ci")
        runs = d["n_runs"]
        pairs.append(
            f"{fmt_points(d['diff'], 0)}{fmt_interval(ci, 100)} at +{key} "
            f"({runs[1]} vs {runs[0]} runs)"
        )
        if ci and (ci[0] > 0 or ci[1] < 0):
            excludes.append(key)
    if pairs:
        verdict = (
            f" The interval excludes 0 at +{', +'.join(excludes)} only; with two or three runs a "
            "side these intervals show the spread of the runs, not of a new run."
            if excludes
            else " No interval excludes 0: at matched proxy gain the two are not distinguishable "
            "with these runs, and the mixture gets there without an answer key."
        )
        lines.append(
            "- Persona mix minus α = 0.3 at the same proxy gain, points (bootstrap over runs): "
            + ", ".join(pairs)
            + "."
            + verdict
        )
    return lines


def render_markdown(payload: dict[str, Any]) -> list[str]:
    """Section (a) of ``summary.md``."""
    lines = ["## (a) Frontier: sycophancy at matched optimization", ""]
    conds = payload["conditions"]["proxy_gain"]
    keys = payload["proxy_thresholds"]
    matched = payload.get("bon_matched_threshold")
    starts = sorted({round(r["start_sycophancy"], 4) for r in payload["runs"].values()})
    start_text = (
        f"Every run starts at {fmt_pct(starts[0], 1)}."
        if len(starts) == 1
        else f"Runs start at {fmt_pct(starts[0], 1)}–{fmt_pct(starts[-1], 1)}."
    )
    lines += [
        "Sycophancy of the GRPO runs interpolated at the first evaluation where the proxy gain "
        "(immediate rating minus step 0) reaches each threshold: mean over the runs that reached "
        "it, 95% bootstrap over runs in brackets, and in parentheses the runs with a value / the "
        f"runs that logged the proxy. {start_text}"
        + (f" The last row, +{matched}, is the Best-of-64 (α = 0) gain." if matched else ""),
        "",
    ]
    lines += _table(conds, keys, "sycophancy_rate", "proxy gain")
    lines += ["", "Difference from α = 0 at the same proxy gain, points (bootstrap over runs):", ""]
    diffs = payload["differences_vs_alpha_0"]["proxy_gain"]
    names = list(diffs)
    lines += [
        "| proxy gain | " + " | ".join(condition_label(c) for c in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(_diff_cell(diffs[c][key]) for c in names) + " |")
    lines += ["", *interpretation(payload), ""]
    lines += ["Gold accuracy under a wrong opinion at the same proxy gain:", ""]
    lines += _table(conds, keys, "gold_accuracy@wrong", "proxy gain")
    lines += ["", "Informed rating (the delayed signal) at the same proxy gain:", ""]
    lines += _table(conds, keys, "informed_mean", "proxy gain", pct=False)
    lines += ["", *_bon_lines(payload), ""]
    lines += [
        "Sycophancy by proxy-gain bin (every evaluation after step 0; mean of per-run means, "
        "number of runs / points in brackets):",
        "",
    ]
    binned = payload["binned"]
    names = list(binned)
    lines += [
        "| proxy gain bin | " + " | ".join(condition_label(c) for c in names) + " |",
        "|---" * (len(names) + 1) + "|",
    ]
    n_bins = len(next(iter(binned.values()))) if binned else 0
    for i in range(n_bins):
        lo, hi = binned[names[0]][i]["bin"]
        label = (
            f"< {hi:+.2f}"
            if lo is None or (isinstance(lo, float) and math.isinf(lo))
            else f"≥ {lo:+.2f}"
            if hi is None or (isinstance(hi, float) and math.isinf(hi))
            else f"{lo:+.2f} to {hi:+.2f}"
        )
        cells = []
        for c in names:
            row = binned[c][i]
            cells.append(
                "—"
                if not row["n_points"]
                else f"{fmt_pct(row['mean_of_runs'])} ({row['n_runs']}/{row['n_points']})"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    kl_conds = payload["conditions"]["sqrt_kl"]
    lines += [
        "",
        "Sycophancy at matched √KL (per-token KL from the trainer log, mean over the 25 steps "
        "before each evaluation; first-pass runs included):",
        "",
    ]
    lines += _table(kl_conds, payload["sqrt_kl_thresholds"], "sycophancy_rate", "√KL")
    return lines


def _bon_lines(payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    model = payload.get("bon_matched_model")
    if not model:
        return lines
    bon = payload["bon"][model]
    conds = payload["conditions"]["proxy_gain"]
    lines += [
        f"Selection against training at the same proxy gain: change in sycophancy from its own "
        f"start, points. Best-of-N on {model} (α = 0, read along N = 4, 16, 64 from a random "
        f"sample, {bon['n_questions']} test questions) against GRPO condition means "
        "(validation questions):",
        "",
        "| proxy gain | Best-of-N, α = 0 | " + " | ".join(condition_label(c) for c in conds) + " |",
        "|---" * (len(conds) + 2) + "|",
    ]
    for key in payload["proxy_thresholds"]:
        sel = bon["alpha0_at_proxy_gain"].get(key)
        if sel is None and key == payload.get("bon_matched_threshold"):
            sel = {"reached": True, "sycophancy_change": bon["bon64"]["sycophancy_change"]}
        sel_cell = (
            fmt_points(sel["sycophancy_change"], 1) if sel and sel.get("reached") else "not reached"
        )
        cells = []
        for c in conds:
            stat = conds[c][key]["sycophancy_change"]
            cells.append(
                "—" if stat["mean"] is None else f"{fmt_points(stat['mean'], 1)} ({stat['n_runs']})"
            )
        lines.append(f"| {key} | {sel_cell} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Best-of-N points (expected pick, against a random sample):")
    lines.append("")
    lines.append("| model | α | N | proxy gain | sycophancy | change, points | gold @ wrong |")
    lines.append("|---|---|---|---|---|---|---|")
    for m, b in payload["bon"].items():
        for alpha_key, per_n in b["points"].items():
            for n_key in ("1", "64"):
                p = per_n.get(n_key)
                if p is None:
                    continue
                lines.append(
                    f"| {m} | {alpha_key.split('=')[1]} | {n_key} | "
                    f"{fmt_num(p['proxy_gain'], 2, sign=True)} | "
                    f"{fmt_pct(p['sycophancy_rate'], 1)} | {fmt_points(p['sycophancy_change'])} | "
                    f"{fmt_pct(p['gold_accuracy@wrong'], 1)} |"
                )
    return lines

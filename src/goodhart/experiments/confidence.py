"""Confidence intervals for every reported rate (DESIGN §4).

Baseline and Best-of-N keep per-item files (one row per response), so their rates get a
percentile bootstrap over *questions*: a ``base_id`` is resampled with all of its opinion
variants, which keeps the paired structure that ``flip_rate`` relies on. GRPO curves keep only
aggregates, so each run's step-300 rates get a Wilson interval from the counts, and the runs
of one series (an α, or a persona mixture) are summarised by mean, min and max.

CLI::

    python -m goodhart.experiments.confidence --results results --out results/ci

Per-item files are looked up next to each summary first and then under ``--runner-results``
(the directory where fetched runner outputs live, ``results/fetched`` by default).
"""

from __future__ import annotations

import argparse
import glob
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from goodhart.eval.metrics import OPINION_ORDER, compute_metrics, records_to_frame
from goodhart.experiments.best_of_n import group_rows, mixed_score, select_best_of_n
from goodhart.io import read_json, read_jsonl, write_json

RATE_KEYS = ("gold_accuracy", "sycophancy_rate", "flip_rate", "proxy_mean") + tuple(
    f"gold_accuracy@{op}" for op in OPINION_ORDER
)


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial rate; ``(nan, nan)`` when ``n == 0``."""
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _per_question(frame) -> dict[str, np.ndarray]:
    """Sums per ``base_id`` of everything the rates are ratios of."""
    from goodhart.eval.metrics import _abstained, _correct, _follows_opinion  # noqa: PLC0415

    correct = _correct(frame).to_numpy()
    follows = _follows_opinion(frame).to_numpy()
    abstained = _abstained(frame).to_numpy()
    opinion = frame["opinion"].to_numpy()
    proxy = frame["proxy"].to_numpy(dtype=float)
    base = frame["base_id"].to_numpy()
    ids, inv = np.unique(base, return_inverse=True)
    size = len(ids)

    def total(mask: np.ndarray) -> np.ndarray:
        return np.bincount(inv, weights=mask.astype(float), minlength=size)

    wrong = opinion == "wrong"
    none = opinion == "none"
    none_rows = total(none)
    none_correct = total(none & correct)
    solid = (none_rows > 0) & (none_rows == none_correct)
    out = {
        "rows": total(np.ones(len(base), dtype=bool)),
        "correct": total(correct),
        "abstained": total(abstained),
        "wrong": total(wrong),
        "follows": total(wrong & follows),
        "proxy_sum": total(np.where(np.isnan(proxy), 0.0, proxy)),
        "proxy_cnt": total(~np.isnan(proxy)),
        "paired_rows": total(wrong) * solid,
        "paired_follows": total(wrong & follows) * solid,
    }
    for op in OPINION_ORDER:
        out[f"rows@{op}"] = total(opinion == op)
        out[f"correct@{op}"] = total((opinion == op) & correct)
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _resampled_rates(per: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Every rate, recomputed on each row of resampled question indices ``idx``."""

    def pick(key: str) -> np.ndarray:
        return per[key][idx].sum(axis=1)

    rates = {
        "gold_accuracy": _ratio(pick("correct"), pick("rows")),
        "abstain_rate": _ratio(pick("abstained"), pick("rows")),
        "sycophancy_rate": _ratio(pick("follows"), pick("wrong")),
        "flip_rate": _ratio(pick("paired_follows"), pick("paired_rows")),
        "proxy_mean": _ratio(pick("proxy_sum"), pick("proxy_cnt")),
    }
    for op in OPINION_ORDER:
        rates[f"gold_accuracy@{op}"] = _ratio(pick(f"correct@{op}"), pick(f"rows@{op}"))
    return rates


def _percentile_ci(values: np.ndarray) -> list[float] | None:
    clean = values[~np.isnan(values)]
    if not clean.size:
        return None
    lo, hi = np.percentile(clean, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_metrics(
    records: list[dict],
    parsed: list[str | None],
    proxy: list[float] | None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Point estimates (``compute_metrics``) with 95 % bootstrap intervals over questions.

    Returns ``{"point": {...}, "ci": {key: [lo, hi]}, "n_questions": B, "n_boot": n_boot}``.
    """
    frame = records_to_frame(records, parsed, proxy)
    point = compute_metrics(frame)
    per = _per_question(frame)
    size = len(per["rows"])
    idx = np.random.default_rng(seed).integers(0, size, size=(n_boot, size))
    rates = _resampled_rates(per, idx)
    ci = {key: interval for key, values in rates.items() if (interval := _percentile_ci(values))}
    return {"point": point, "ci": ci, "n_questions": int(size), "n_boot": int(n_boot)}


def paired_difference(
    records: list[dict],
    first: tuple[list[str | None], list[float] | None],
    second: tuple[list[str | None], list[float] | None],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """``second − first`` for every rate, with a 95 % interval from one shared question resample.

    Both selections answer the same questions, so the difference is bootstrapped by resampling
    questions once and evaluating both on that resample; this is much tighter than comparing
    two marginal intervals.
    """
    frames = [records_to_frame(records, parsed, proxy) for parsed, proxy in (first, second)]
    pers = [_per_question(frame) for frame in frames]
    size = len(pers[0]["rows"])
    idx = np.random.default_rng(seed).integers(0, size, size=(n_boot, size))
    rates = [_resampled_rates(per, idx) for per in pers]
    points = [compute_metrics(frame) for frame in frames]
    out: dict[str, Any] = {"point": {}, "ci": {}}
    for key in rates[0]:
        if key in points[0] and key in points[1]:
            out["point"][key] = float(points[1][key] - points[0][key])
            interval = _percentile_ci(rates[1][key] - rates[0][key])
            if interval:
                out["ci"][key] = interval
    return out


# ----------------------------------------------------------------------------- data access


def _find_item_file(results_dir: Path, runner_dir: Path | None, stage: str, basename: str) -> Path:
    """The per-item file next to the summary, else the newest copy among fetched runner outputs."""
    local = results_dir / stage / basename
    if local.is_file():
        return local
    if runner_dir is not None:
        hits = sorted(glob.glob(str(runner_dir / "*" / "results" / stage / basename)))
        if hits:
            return Path(hits[-1])
    raise FileNotFoundError(f"no {basename} under {results_dir / stage} or {runner_dir}")


def baseline_intervals(
    payload: dict, responses: list[dict], records_by_id: dict[str, dict], n_boot: int, seed: int
) -> dict[str, Any]:
    """Bootstrap the baseline rates of one model from its ``*_responses.jsonl``."""
    records = [records_by_id[row["record_id"]] for row in responses]
    parsed = [row.get("parsed") for row in responses]
    proxy = [row.get("proxy") for row in responses] if "proxy" in responses[0] else None
    out = bootstrap_metrics(records, parsed, proxy, n_boot, seed)
    out["model"] = payload.get("model")
    out["matches_summary"] = _close(out["point"], payload.get("metrics") or {})
    return out


def bon_intervals(
    payload: dict, rows: list[dict], records_by_id: dict[str, dict], n_boot: int, seed: int
) -> dict[str, Any]:
    """Re-select the Best-of-N pick for every ``alpha`` and ``N`` and bootstrap its rates.

    Selection repeats ``best_of_n.select_and_evaluate`` (same tie-break seed), so the point
    estimates reproduce the committed curves; ``matches_summary`` records whether they did.
    """
    grouped = group_rows(rows)
    records = [records_by_id[record_id] for record_id in grouped]
    cfg = payload.get("config") or {}
    sel_seed = int(cfg.get("seed", 0))
    curves: dict[str, dict[str, Any]] = {}
    selections: dict[tuple[str, str], tuple[list[str | None], list[float]]] = {}
    for alpha_key, per_n in (payload.get("curves") or {}).items():
        alpha = float(alpha_key.split("=")[1])
        curves[alpha_key] = {}
        for n_key, summary in per_n.items():
            n = int(n_key)
            rng = random.Random(f"{sel_seed}-{alpha}-{n}")
            parsed: list[str | None] = []
            proxy: list[float] = []
            for record in records:
                record_rows = grouped[record["id"]]
                scores = [mixed_score(r["immediate"], r["informed"], alpha) for r in record_rows]
                chosen = record_rows[select_best_of_n(scores, n, rng)]
                parsed.append(chosen["parsed"])
                proxy.append(chosen["immediate"])
            out = bootstrap_metrics(records, parsed, proxy, n_boot, seed)
            out["matches_summary"] = _close(out["point"], summary)
            curves[alpha_key][n_key] = out
            selections[(alpha_key, n_key)] = (parsed, proxy)
    return {
        "model": payload.get("model"),
        "curves": curves,
        "paired": paired_comparisons(records, curves, selections, n_boot, seed),
    }


def paired_comparisons(
    records: list[dict],
    curves: dict[str, dict[str, Any]],
    selections: dict[tuple[str, str], tuple[list[str | None], list[float]]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """The comparisons the write-up makes: largest N against N = 1 within each α, and every
    α against α = 0 at the largest N."""
    paired: dict[str, Any] = {}
    n_keys = sorted({n for _, n in selections}, key=int)
    if len(n_keys) < 2:
        return paired
    first_n, last_n = n_keys[0], n_keys[-1]
    for alpha_key in curves:
        if (alpha_key, first_n) in selections and (alpha_key, last_n) in selections:
            paired[f"{alpha_key}: N={last_n} minus N={first_n}"] = paired_difference(
                records,
                selections[(alpha_key, first_n)],
                selections[(alpha_key, last_n)],
                n_boot,
                seed,
            )
    base = next((k for k in curves if float(k.split("=")[1]) == 0.0), None)
    if base is not None and (base, last_n) in selections:
        for alpha_key in curves:
            if alpha_key != base and (alpha_key, last_n) in selections:
                paired[f"N={last_n}: {alpha_key} minus {base}"] = paired_difference(
                    records,
                    selections[(base, last_n)],
                    selections[(alpha_key, last_n)],
                    n_boot,
                    seed,
                )
    return paired


def _close(point: dict[str, float], summary: dict[str, float], tol: float = 1e-9) -> bool:
    keys = [k for k in RATE_KEYS if k in point and k in summary]
    return bool(keys) and all(
        (math.isnan(point[k]) and math.isnan(summary[k])) or abs(point[k] - summary[k]) <= tol
        for k in keys
    )


def _series_of(summary: dict[str, Any]) -> str:
    """Series of a run in the spread table: ``persona_mix`` for a persona mixture, else its α."""
    reward = (summary.get("config") or {}).get("reward") or {}
    kind = summary.get("reward_kind") or (reward.get("kind") if isinstance(reward, dict) else None)
    return "persona_mix" if kind == "persona_mix" else str(summary.get("alpha"))


def _series_order(series: str) -> tuple[int, float, str]:
    try:
        return (0, float(series), "")
    except ValueError:
        return (1, 0.0, series)


def series_label(series: str) -> str:
    """Series as written in the report: ``0.3`` stays ``0.3``, ``persona_mix`` → ``persona mix``."""
    return "persona mix" if series == "persona_mix" else series


def grpo_intervals(run_dirs: Sequence[Path]) -> dict[str, Any]:
    """Wilson intervals for the final evaluation of every GRPO run, plus per-series spread.

    A series is an α of the delayed reward, or ``persona_mix`` for runs trained on a persona
    mixture (their ``alpha`` field is 0 but means nothing); ``by_alpha`` is keyed by series.
    """
    runs: dict[str, Any] = {}
    for run_dir in run_dirs:
        curve = read_jsonl(run_dir / "curve.jsonl")
        if not curve:
            continue
        summary = read_json(run_dir / "summary.json")
        last = curve[-1]
        rates = {
            "gold_accuracy": ("n", 1.0),
            "sycophancy_rate": ("n@wrong", 1.0),
            "gold_accuracy@wrong": ("n@wrong", 1.0),
            "gold_accuracy@none": ("n@none", 1.0),
            "heldout_gold_accuracy": ("heldout_n", 1.0),
            "heldout_sycophancy_rate": ("heldout_n@wrong", 1.0),
        }
        entry: dict[str, Any] = {
            "alpha": summary.get("alpha"),
            "reward_kind": summary.get("reward_kind"),
            "series": _series_of(summary),
            "step": last.get("step"),
            "point": {},
            "ci": {},
        }
        for key, (n_key, _) in rates.items():
            if key in last and last.get(n_key):
                n = int(last[n_key])
                entry["point"][key] = float(last[key])
                entry["ci"][key] = list(wilson(round(float(last[key]) * n), n))
        for key in ("proxy_mean", "informed_mean", "reward_mean"):
            if key in last:
                entry["point"][key] = float(last[key])
        runs[f"{run_dir.parent.name}/{run_dir.name}"] = entry
    by_alpha: dict[str, Any] = {}
    for name, entry in runs.items():
        by_alpha.setdefault(entry["series"], []).append(name)
    spread: dict[str, Any] = {}
    for alpha, names in sorted(by_alpha.items(), key=lambda kv: _series_order(kv[0])):
        keys = sorted({k for n in names for k in runs[n]["point"]})
        spread[alpha] = {"runs": names}
        for key in keys:
            vals = [runs[n]["point"][key] for n in names if key in runs[n]["point"]]
            spread[alpha][key] = {
                "mean": float(np.mean(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "n_runs": len(vals),
            }
    return {"runs": runs, "by_alpha": spread}


# ----------------------------------------------------------------------------- report


def _pct(value: float, ci: Sequence[float] | None) -> str:
    if ci is None:
        return f"{100 * value:.1f} %"
    return f"{100 * value:.1f} % [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def _num(value: float, ci: Sequence[float] | None) -> str:
    return f"{value:.2f}" if ci is None else f"{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"


def _spread_cell(spread: dict[str, Any], key: str, pct: bool) -> str:
    if key not in spread:
        return "—"
    v = spread[key]
    fmt = (lambda x: f"{100 * x:.0f}") if pct else (lambda x: f"{x:.2f}")
    tag = f" (n={v['n_runs']})" if v["n_runs"] != len(spread["runs"]) else ""
    if v["n_runs"] == 1:
        return f"{fmt(v['mean'])}{tag}"
    return f"{fmt(v['mean'])} ({fmt(v['min'])}–{fmt(v['max'])}){tag}"


def _diff_cells(diff: dict[str, Any], keys: Sequence[str]) -> str:
    cells = []
    for key in keys:
        v, c = diff["point"].get(key), diff["ci"].get(key)
        if v is None:
            cells.append("—")
        elif key.endswith("_mean"):
            cells.append(f"{v:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]" if c else f"{v:+.2f}")
        else:
            cells.append(
                f"{100 * v:+.1f} [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]" if c else f"{100 * v:+.1f}"
            )
    return " | ".join(cells)


def _rate_cells(point: dict[str, float], ci: dict[str, Any], keys: Sequence[str]) -> str:
    cells = []
    for key in keys:
        if key not in point:
            cells.append("—")
        elif key.endswith("_mean"):
            cells.append(_num(point[key], ci.get(key)))
        else:
            cells.append(_pct(point[key], ci.get(key)))
    return " | ".join(cells)


def render_markdown(baseline: list[dict], bon: list[dict], grpo: dict[str, Any]) -> str:
    """Tables of point estimates with 95 % intervals, the way the write-up quotes them."""
    lines = ["# Confidence intervals", ""]
    lines += [
        "Baseline and Best-of-N: percentile bootstrap over questions (a question is resampled",
        "with all of its opinion variants). GRPO: Wilson intervals from the counts of the final",
        "evaluation of each run, and mean / min / max across the runs of one series (an α, or",
        "a persona mixture; a single run shows its value alone). Wilson on the",
        "overall and held-out gold columns treats the three opinion variants of a question as",
        "independent rows and is somewhat narrow; sycophancy and gold @ wrong have one row per",
        "question. Paired differences resample the same questions for both selections.",
        "",
    ]
    if baseline:
        keys = (
            "gold_accuracy",
            "gold_accuracy@wrong",
            "gold_accuracy@none",
            "sycophancy_rate",
            "flip_rate",
            "proxy_mean",
        )
        n_q = baseline[0]["n_questions"]
        lines += [f"## Baseline (test split, {n_q} questions × 3 opinions)", ""]
        lines += [
            "| model | gold | gold @ wrong | gold @ none | sycophancy | flip | proxy |",
            "|---|---|---|---|---|---|---|",
        ]
        for b in baseline:
            lines.append(f"| {b['model']} | {_rate_cells(b['point'], b['ci'], keys)} |")
        lines.append("")
    for payload in bon:
        keys = ("proxy_mean", "gold_accuracy", "sycophancy_rate", "flip_rate")
        n_q = next(iter(next(iter(payload["curves"].values())).values()))["n_questions"]
        lines += [f"## Best-of-N, {payload['model']} ({n_q} questions × 3 opinions)", ""]
        lines += ["| α | N | proxy | gold | sycophancy | flip |", "|---|---|---|---|---|---|"]
        for alpha_key, per_n in payload["curves"].items():
            for n_key, out in per_n.items():
                alpha = alpha_key.split("=")[1]
                lines.append(
                    f"| {alpha} | {n_key} | {_rate_cells(out['point'], out['ci'], keys)} |"
                )
        lines.append("")
        if payload.get("paired"):
            lines += [f"### Paired differences, {payload['model']} (same questions)", ""]
            lines += [
                "| comparison | proxy | gold, points | sycophancy, points | flip, points |",
                "|---|---|---|---|---|",
            ]
            for name, diff in payload["paired"].items():
                lines.append(f"| {name} | {_diff_cells(diff, keys)} |")
            lines.append("")
    if grpo.get("runs"):
        keys = (
            "gold_accuracy",
            "sycophancy_rate",
            "gold_accuracy@wrong",
            "heldout_sycophancy_rate",
            "heldout_gold_accuracy",
            "proxy_mean",
            "informed_mean",
        )
        lines += ["## GRPO, final evaluation of each run (Wilson 95 %)", ""]
        lines += [
            "| run | α / reward | gold | sycophancy | gold @ wrong | held-out sycophancy "
            "| held-out gold | proxy | informed |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for name, e in grpo["runs"].items():
            series = series_label(e.get("series", str(e["alpha"])))
            lines.append(f"| `{name}` | {series} | {_rate_cells(e['point'], e['ci'], keys)} |")
        lines += ["", "## GRPO, across runs of one series (mean, min–max, number of runs)", ""]
        lines += [
            "| α / reward | runs | gold | sycophancy | held-out sycophancy | proxy | informed |",
            "|---|---|---|---|---|---|---|",
        ]
        for alpha, s in grpo["by_alpha"].items():
            lines.append(
                f"| {series_label(alpha)} | {len(s['runs'])} | "
                f"{_spread_cell(s, 'gold_accuracy', True)} | "
                f"{_spread_cell(s, 'sycophancy_rate', True)} | "
                f"{_spread_cell(s, 'heldout_sycophancy_rate', True)} | "
                f"{_spread_cell(s, 'proxy_mean', False)} | "
                f"{_spread_cell(s, 'informed_mean', False)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point; writes ``baseline.json``, ``bon.json``, ``grpo.json`` and ``summary.md``."""
    parser = argparse.ArgumentParser(description="confidence intervals for every reported rate")
    parser.add_argument("--results", default="results")
    parser.add_argument(
        "--runner-results",
        default="results/fetched",
        help="directory with fetched runner outputs (per-item files under <run>/results/)",
    )
    parser.add_argument("--data", default="data/prompts.jsonl")
    parser.add_argument("--out", default="results/ci")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    results, runner, out = Path(args.results), Path(args.runner_results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records_by_id = {r["id"]: r for r in read_jsonl(args.data)}

    baseline = []
    for path in sorted((results / "baseline").glob("*.json")):
        payload = read_json(path)
        item_file = _find_item_file(
            results, runner, "baseline", Path(payload["responses_file"]).name
        )
        baseline.append(
            baseline_intervals(
                payload, read_jsonl(item_file), records_by_id, args.n_boot, args.seed
            )
        )
        print(f"baseline {payload['model']}: matches summary = {baseline[-1]['matches_summary']}")
    bon = []
    for path in sorted((results / "bon").glob("*.json")):
        payload = read_json(path)
        item_file = _find_item_file(results, runner, "bon", Path(payload["samples_file"]).name)
        bon.append(
            bon_intervals(payload, read_jsonl(item_file), records_by_id, args.n_boot, args.seed)
        )
        flags = [
            o["matches_summary"] for per_n in bon[-1]["curves"].values() for o in per_n.values()
        ]
        print(
            f"bon {payload['model']}: {sum(flags)}/{len(flags)} curve points reproduce the summary"
        )
    run_dirs = sorted(
        p.parent
        for sub in ("grpo", "grpo_first_pass")
        for p in results.glob(f"{sub}/*/curve.jsonl")
    )
    grpo = grpo_intervals(run_dirs)
    write_json(out / "baseline.json", baseline)
    write_json(out / "bon.json", bon)
    write_json(out / "grpo.json", grpo)
    (out / "summary.md").write_text(render_markdown(baseline, bon, grpo), encoding="utf-8")
    print(f"wrote {out}/summary.md")
    return out


if __name__ == "__main__":
    main()

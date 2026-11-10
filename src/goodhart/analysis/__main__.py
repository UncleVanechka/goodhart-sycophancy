"""CLI for the post-hoc analyses; writes four JSON files and ``summary.md``.

::

    python -m goodhart.analysis --results results --out results/analysis

GRPO runs are read from ``results/grpo/`` and ``results/grpo_first_pass/``. Best-of-N
per-sample files are found from the committed ``results/bon/<slug>.json`` payloads, next to
them first and then under ``--runner-results`` (fetched runner outputs), or passed explicitly
with ``--bon-samples``. Prompt records (``--data``) give the gold letter and the user's letter.
The outputs name the samples files by basename only, so they do not depend on where the runner
outputs were fetched to.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from goodhart.analysis import breakeven, dynamics, frontier, opinion_proxy
from goodhart.analysis.common import (
    BonSamples,
    condition_label,
    find_samples_file,
    fmt_num,
    fmt_pct,
    jsonable,
    load_bon_samples,
    load_runs,
)
from goodhart.io import read_json, read_jsonl, write_json

CAVEATS = [
    "**Post hoc.** The thresholds (onset at 64% / 59%, short episodes under 35 tokens, reply "
    "forms at 20 / 200 characters, the proxy-gain and √KL grids) were set after these runs had "
    "been seen. The numbers describe the runs; they are not tests of pre-registered "
    "hypotheses.",
    "**Runs are the unit, and there are few.** Two to four runs per condition. A percentile "
    "bootstrap over two or three runs cannot leave the range of the observed values, so its "
    "intervals show the spread of the runs and understate the uncertainty about a new run.",
    "**First-crossing reads drop runs.** A run that never reaches a threshold is left out of "
    "that threshold's mean, so means at high proxy gain or √KL describe the runs that got "
    "there (survivorship), and the number of runs reached is part of the result.",
    "**Evaluation noise.** Every point is one greedy evaluation of 300 validation questions "
    "per opinion type, every 25 steps: about ±3 points of binomial noise on sycophancy, and "
    "points of one run are correlated.",
    "**KL is the trainer's per-token estimate**, averaged over the 25 steps before an "
    "evaluation. It is noisy and not a sequence-level KL; when completions shrink to a "
    "letter, per-token and sequence KL drift apart.",
    "**Selection and training differ in data and decoding.** Best-of-N uses temperature-1 "
    "samples of the base model on 200 test questions; GRPO is evaluated greedily on 300 "
    "validation questions. They are compared through changes from their own start.",
    "**The break-even describes the base model.** It is computed on the base model's samples, "
    "i.e. the reward landscape at step 0; the policy's distribution, and with it the local "
    "break-even, moves during training. The push ignores the KL penalty, clipping and the "
    "token-level loss normalization, and treats a prompt's samples as one or eight groups.",
    "**The push is a 64-sample estimate on another sampler.** The Best-of-N samples were drawn "
    "with top-p 0.95 and the model's default top-k (Qwen3: 20) in bfloat16; GRPO's rollouts "
    "used top-p 1 and top-k 0 in float32. The Best-of-N sampler cuts the tail of unlikely "
    "replies, so the push here is not training's step-0 push; a TRL-matched step-0 value will "
    "come from the base policy's samples of the `sample_rate` stage (M3), which uses the "
    "rollout settings.",
    "**One judge** (one model family) gives every rating; Best-of-N picks here are expected "
    "picks with ties averaged, which differ slightly from the committed single-draw "
    "tie-breaks (checked in `frontier.json`, `bon_check`).",
]


def bon_check(bons: list[BonSamples], committed: dict[str, dict]) -> dict[str, Any]:
    """Expected picks against the committed Best-of-N curves, N = 1 and the largest N.

    N = 1 has no ties, so it must match exactly; at the largest N the committed value comes from
    one random tie-break and the expected pick from all of them.
    """
    out: dict[str, Any] = {}
    for bon in bons:
        payload = committed.get(bon.model)
        if not payload:
            continue
        out[bon.model] = {}
        for alpha_key, per_n in (payload.get("curves") or {}).items():
            alpha = float(alpha_key.split("=")[1])
            for n_key in (min(per_n, key=int), max(per_n, key=int)):
                point = frontier.bon_point(bon, alpha, int(n_key))
                summary = per_n[n_key]
                out[bon.model][f"{alpha_key} N={n_key}"] = {
                    "committed_sycophancy": summary.get("sycophancy_rate"),
                    "expected_sycophancy": point["sycophancy_rate"],
                    "committed_proxy": summary.get("proxy_mean"),
                    "expected_proxy": point["proxy_mean"],
                }
    return out


def _load_bons(
    results: Path, runner: Path | None, records: dict[str, dict], explicit: list[str] | None
) -> tuple[list[BonSamples], dict[str, dict], list[str]]:
    bons: list[BonSamples] = []
    committed: dict[str, dict] = {}
    paths: list[str] = []
    if explicit:
        for path in explicit:
            bons.append(load_bon_samples(path, records))
            paths.append(Path(path).name)
    else:
        for summary in sorted((results / "bon").glob("*.json")):
            payload = read_json(summary)
            basename = Path(payload.get("samples_file") or f"{summary.stem}_samples.jsonl").name
            try:
                path = find_samples_file(results, runner, basename)
            except FileNotFoundError as err:
                print(f"skip {summary.name}: {err}")
                continue
            bons.append(load_bon_samples(path, records, model=summary.stem))
            paths.append(Path(path).name)
    for summary in sorted((results / "bon").glob("*.json")):
        committed[summary.stem] = read_json(summary)
    return bons, committed, paths


def key_numbers(
    front: dict[str, Any], dyn: dict[str, Any], brk: dict[str, Any], opn: dict[str, Any]
) -> list[str]:
    """The headline bullets at the top of ``summary.md``."""
    lines: list[str] = []
    conds = front["conditions"]["proxy_gain"]
    for key in ("0.30", "0.45"):
        cells = []
        for cond, per_key in conds.items():
            entry = per_key.get(key)
            if not entry:
                continue
            mean = entry["sycophancy_rate"]["mean"]
            cells.append(
                f"{condition_label(cond)} {fmt_pct(mean)} ({entry['n_reached']}/{entry['n_runs']})"
                if mean is not None
                else f"{condition_label(cond)} not reached"
            )
        if cells:
            lines.append(f"- Sycophancy at proxy gain +{key}: " + ", ".join(cells) + ".")
    pre = dyn["collapse_precursors"]
    if pre["n_collapsed"]:
        lines.append(
            f"- Collapses (end ≥ 75%): {pre['n_collapsed']}; with a short-completion episode by "
            f"onset: {pre['n_short_before_onset']}; runs without a collapse that had one: "
            f"{pre['n_other_with_short_episode']} of {pre['n_other']}."
        )
    pooled = dyn["correlations_pooled"]["length"]["pearson"]
    lines.append(
        f"- Training-completion length against sycophancy, all evaluations pooled: r = "
        f"{fmt_num(pooled, 2, sign=True)}."
    )
    fmt_alpha = breakeven.fmt_alpha
    for model, m in brk["models"].items():
        raw, bon = m["raw_gap"], m["bon_crossing"]
        p64 = m["push"].get("group=64", {})
        vs_gold = p64.get("wrong_vs_gold", {}).get("all", {})
        vs_rest = p64.get("wrong_vs_rest", {}).get("all", {})
        lines.append(
            f"- Break-even α on {model}: raw gap {fmt_alpha(raw['all'])}; group-normalized push, "
            f"64-sample estimate on the Best-of-N sampler, "
            f"{fmt_alpha(vs_rest)} (agreeing vs all other samples) or {fmt_alpha(vs_gold)} "
            f"(agreeing vs gold only); BoN crossing {fmt_alpha(bon['all'])}. Raw gap by form: "
            f"bare letters {fmt_alpha(raw['bare'])}, long replies {fmt_alpha(raw['long'])}."
        )
    ref = (opn["conditions"].get("alpha=0") or {}).get("proxy_mean")
    if ref:
        lines.append(
            f"- Share of the α = 0 training proxy gain made on wrong-opinion prompts: "
            f"{fmt_pct(ref['share_of_change'].get('wrong'))}."
        )
    kbd = dyn["knows_but_defers"].get("alpha=0")
    if kbd:
        flips = [v["end"] for v in kbd["flip_rate"]["per_run"].values()]
        nones = [v["end"] for v in kbd["gold_accuracy@none"]["per_run"].values()]
        lines.append(
            f"- α = 0 flip rate at the end {fmt_pct(min(flips))}–{fmt_pct(max(flips))}, accuracy "
            f"without an opinion {fmt_pct(min(nones))}–{fmt_pct(max(nones))}."
        )
    return lines


def render_summary(
    front: dict[str, Any],
    dyn: dict[str, Any],
    brk: dict[str, Any],
    opn: dict[str, Any],
    inputs: dict[str, Any],
) -> str:
    """The whole ``summary.md``."""
    counts = Counter(r["condition"] for r in dyn["runs"].values())
    run_text = ", ".join(f"{condition_label(c)} × {n}" for c, n in counts.items())
    missing = [name for name, r in front["runs"].items() if r.get("proxy_gain") is None]
    proxy_text = (
        "every run logged the proxy"
        if not missing
        else ", ".join(f"`{name}`" for name in missing)
        + " did not log the proxy and are left out of the proxy-based items"
    )
    lines = [
        "# Post-hoc analysis of the committed results",
        "",
        "Generated by `python -m goodhart.analysis`. CPU only, from files that already exist: "
        f"{len(dyn['runs'])} GRPO runs ({run_text}; {proxy_text}) and the Best-of-N "
        "per-sample ratings of "
        f"{', '.join(brk['models']) or 'no model'}. These analyses were chosen after the runs "
        "had been seen: they are descriptive, not pre-registered.",
        "",
        "## Key numbers",
        "",
        *key_numbers(front, dyn, brk, opn),
        "",
    ]
    lines += frontier.render_markdown(front) + [""]
    lines += dynamics.render_markdown(dyn)
    lines += breakeven.render_markdown(brk)
    lines += opinion_proxy.render_markdown(opn)
    lines += dynamics.render_knows_but_defers(dyn)
    lines += ["## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    lines += ["## Inputs", ""]
    lines += [f"- GRPO run directories: {', '.join(f'`{r}`' for r in inputs['runs'])}"]
    lines += [
        f"- Best-of-N samples (file names): {', '.join(f'`{p}`' for p in inputs['bon_samples'])}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="post-hoc analyses of the committed results")
    parser.add_argument("--results", default="results")
    parser.add_argument(
        "--runner-results",
        default="results/fetched",
        help="directory with fetched runner outputs (per-item files under <run>/results/)",
    )
    parser.add_argument("--data", default="data/prompts.jsonl")
    parser.add_argument("--bon-samples", nargs="*", default=None, help="explicit samples files")
    parser.add_argument("--out", default="results/analysis")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    results, out = Path(args.results), Path(args.out)
    runner = Path(args.runner_results) if args.runner_results else None

    runs = load_runs(results)
    records = {r["id"]: r for r in read_jsonl(args.data)}
    bons, committed, sample_paths = _load_bons(results, runner, records, args.bon_samples)
    print(f"{len(runs)} GRPO runs, Best-of-N samples for {[b.model for b in bons]}")

    front = frontier.analyze(runs, bons, n_boot=args.n_boot, seed=args.seed)
    front["bon_check"] = bon_check(bons, committed)
    dyn = dynamics.analyze(runs)
    brk = breakeven.analyze(bons, n_boot=args.n_boot, seed=args.seed)
    opn = opinion_proxy.analyze(runs, bons, n_boot=args.n_boot, seed=args.seed)
    inputs = {"runs": [r.name for r in runs], "bon_samples": sample_paths}
    payloads = {"frontier": front, "dynamics": dyn, "breakeven": brk, "opinion_proxy": opn}
    clean = {}
    for name, payload in payloads.items():
        clean[name] = jsonable({"inputs": inputs, **payload})
        write_json(out / f"{name}.json", clean[name])
    summary = render_summary(
        clean["frontier"], clean["dynamics"], clean["breakeven"], clean["opinion_proxy"], inputs
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(f"wrote {out}/summary.md")
    return out


if __name__ == "__main__":
    main()

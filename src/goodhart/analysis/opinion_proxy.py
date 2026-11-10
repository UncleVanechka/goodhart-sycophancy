"""(d) Where the proxy gain comes from: the immediate rating's change by opinion type.

GRPO, every run with the proxy: ``proxy_mean@<opinion>`` at the last evaluation minus step 0,
for no opinion, a correct and a wrong opinion, plus the overall change and each type's share of
it (``Δ@type · n@type / (Δ · n)``, left out when the overall change is under 0.10 and the
share would divide noise). The informed rating gets the same treatment where it was logged.
Best-of-N: the expected Best-of-64 pick by the mixed reward against a random sample, for every
α. *Kept*: a condition's mean overall gain as a share of the α = 0 gain of the same
method (and, for Best-of-N, the same model).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from goodhart.analysis.common import (
    OPINIONS,
    BonSamples,
    Run,
    bootstrap_over_runs,
    condition_label,
    first_n_pool,
    fmt_num,
    fmt_pct,
    runs_by_condition,
    selection_weights,
)

BON_ALPHAS = (0.0, 0.3, 0.6, 1.0)
BON_N = 64
REFERENCE_CONDITION = "alpha=0"
MIN_OVERALL_CHANGE = 0.10


def _shares(changes: dict[str, float], counts: dict[str, float]) -> dict[str, float | None]:
    """Each type's part of the overall change; ``None`` when the overall change is smaller
    than ``MIN_OVERALL_CHANGE`` and the shares would only divide noise."""
    total = sum(changes[op] * counts[op] for op in changes)
    n = sum(counts[op] for op in changes)
    if not n or abs(total / n) < MIN_OVERALL_CHANGE:
        return {op: None for op in changes}
    return {op: changes[op] * counts[op] / total for op in changes}


def run_opinion_proxy(run: Run) -> dict[str, Any]:
    """Change from step 0 to the last evaluation, by opinion type, of both ratings."""
    first, last = run.curve[0], run.curve[-1]
    out: dict[str, Any] = {"condition": run.condition, "seed": run.seed, "step": last["step"]}
    for rating in ("proxy_mean", "informed_mean"):
        if rating not in first or rating not in last:
            out[rating] = None
            continue
        per_op = {
            op: {
                "start": float(first[f"{rating}@{op}"]),
                "end": float(last[f"{rating}@{op}"]),
                "change": float(last[f"{rating}@{op}"]) - float(first[f"{rating}@{op}"]),
            }
            for op in OPINIONS
            if f"{rating}@{op}" in first and f"{rating}@{op}" in last
        }
        counts = {op: float(last.get(f"n@{op}", 1.0)) for op in per_op}
        entry: dict[str, Any] = {
            "by_opinion": per_op,
            "overall_change": float(last[rating]) - float(first[rating]),
        }
        entry["share_of_change"] = _shares({op: v["change"] for op, v in per_op.items()}, counts)
        out[rating] = entry
    return out


def condition_summary(
    per_run: dict[str, Any], groups: dict[str, list[str]], n_boot: int, seed: int
) -> dict[str, Any]:
    """Mean change per opinion type over the runs of each condition (bootstrap over runs)."""
    out: dict[str, Any] = {}
    for condition, names in groups.items():
        cond: dict[str, Any] = {}
        for rating in ("proxy_mean", "informed_mean"):
            rows = [per_run[n][rating] for n in names if per_run[n].get(rating)]
            if not rows:
                continue
            cond[rating] = {
                "n_runs": len(rows),
                "overall_change": bootstrap_over_runs(
                    [r["overall_change"] for r in rows], n_boot, seed
                ),
                "by_opinion": {
                    op: bootstrap_over_runs(
                        [r["by_opinion"][op]["change"] for r in rows if op in r["by_opinion"]],
                        n_boot,
                        seed,
                    )
                    for op in OPINIONS
                },
            }
            # shares of the mean changes (the evaluation sets have equal counts per type)
            means = {op: cond[rating]["by_opinion"][op]["mean"] for op in OPINIONS}
            cond[rating]["share_of_change"] = (
                _shares(means, {op: 1.0 for op in OPINIONS})
                if all(v is not None for v in means.values())
                else {op: None for op in OPINIONS}
            )
        if cond:
            out[condition] = cond
    ref = (out.get(REFERENCE_CONDITION) or {}).get("proxy_mean")
    for cond in out.values():
        mine = cond.get("proxy_mean")
        if ref and mine and ref["overall_change"]["mean"]:
            mine["kept_vs_alpha_0"] = mine["overall_change"]["mean"] / ref["overall_change"]["mean"]
    return out


def bon_opinion_proxy(
    bon: BonSamples, alphas: Sequence[float] = BON_ALPHAS, n: int = BON_N
) -> dict[str, Any]:
    """Change of both ratings of the expected Best-of-``n`` pick against a random sample."""
    out: dict[str, Any] = {"model": bon.model, "n": n, "alphas": {}}
    counts = {op: float((bon.opinion == op).sum()) for op in OPINIONS}
    for alpha in alphas:
        scores = (1.0 - alpha) * bon.immediate + alpha * bon.informed
        weights = selection_weights(scores, first_n_pool(scores.shape, n))
        entry: dict[str, Any] = {}
        for rating, values in (("proxy_mean", bon.immediate), ("informed_mean", bon.informed)):
            picked = (weights * values).sum(axis=1)
            reference = values.mean(axis=1)
            per_op = {
                op: {
                    "random_sample": float(reference[bon.opinion == op].mean()),
                    "best_of_n": float(picked[bon.opinion == op].mean()),
                }
                for op in OPINIONS
                if (bon.opinion == op).any()
            }
            for v in per_op.values():
                v["change"] = v["best_of_n"] - v["random_sample"]
            entry[rating] = {
                "by_opinion": per_op,
                "overall_change": float(picked.mean() - reference.mean()),
                "share_of_change": _shares({op: v["change"] for op, v in per_op.items()}, counts),
            }
        out["alphas"][f"alpha={float(alpha):g}"] = entry
    ref = out["alphas"].get("alpha=0", {}).get("proxy_mean")
    for entry in out["alphas"].values():
        if ref and ref["overall_change"]:
            entry["proxy_mean"]["kept_vs_alpha_0"] = (
                entry["proxy_mean"]["overall_change"] / ref["overall_change"]
            )
    return out


def analyze(
    runs: Sequence[Run], bons: Sequence[BonSamples], n_boot: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """The full opinion-type payload."""
    with_proxy = [r for r in runs if r.has_proxy]
    per_run = {r.name: run_opinion_proxy(r) for r in with_proxy}
    groups = {c: [r.name for r in g] for c, g in runs_by_condition(with_proxy).items()}
    return {
        "definitions": {
            "grpo": "last evaluation minus step 0, validation prompts; runs without the proxy "
            "(first pass) excluded",
            "share_of_change": "change@type * n@type / (overall change * n); null when "
            f"|overall change| < {MIN_OVERALL_CHANGE}; condition shares from mean changes",
            "bon": f"expected Best-of-{BON_N} pick minus a random sample, test prompts",
            "kept_vs_alpha_0": "overall proxy change / that of alpha = 0",
        },
        "runs": per_run,
        "conditions": condition_summary(per_run, groups, n_boot, seed),
        "bon": {b.model: bon_opinion_proxy(b) for b in bons},
    }


# ----------------------------------------------------------------------------- report


def _chg(entry: dict[str, Any] | None, op: str) -> str:
    if not entry or op not in entry["by_opinion"]:
        return "—"
    return fmt_num(entry["by_opinion"][op]["change"], 2, sign=True)


def render_markdown(payload: dict[str, Any]) -> list[str]:
    """Section (d) of ``summary.md``."""
    lines = ["## (d) Where the proxy gain comes from, by opinion type", ""]
    lines += [
        "Change of the immediate rating (proxy) from step 0 to step 300, by the opinion the "
        "user states; *share from wrong*: the part of the overall gain made on wrong-opinion "
        "prompts. Δ informed: the same for the informed rating.",
        "",
        "| run | condition | Δ none | Δ correct | Δ wrong | Δ overall | share from wrong "
        "| Δ informed, wrong |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, r in payload["runs"].items():
        proxy, informed = r["proxy_mean"], r.get("informed_mean")
        lines.append(
            f"| `{name}` | {condition_label(r['condition'])} | {_chg(proxy, 'none')} | "
            f"{_chg(proxy, 'correct')} | {_chg(proxy, 'wrong')} | "
            f"{fmt_num(proxy['overall_change'], 2, sign=True)} | "
            f"{fmt_pct(proxy['share_of_change'].get('wrong'))} | {_chg(informed, 'wrong')} |"
        )
    lines += [
        "",
        "Condition means (bootstrap over runs not shown: two or three runs each):",
        "",
        "| condition | runs | Δ none | Δ correct | Δ wrong | Δ overall | share from wrong "
        "| kept vs α = 0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cond, c in payload["conditions"].items():
        p = c.get("proxy_mean")
        if not p:
            continue
        ops = [fmt_num(p["by_opinion"][op]["mean"], 2, sign=True) for op in OPINIONS]
        lines.append(
            f"| {condition_label(cond)} | {p['n_runs']} | " + " | ".join(ops) + " | "
            f"{fmt_num(p['overall_change']['mean'], 2, sign=True)} | "
            f"{fmt_pct(p['share_of_change'].get('wrong'))} | "
            f"{fmt_pct(p.get('kept_vs_alpha_0'))} |"
        )
    lines += [
        "",
        "Best-of-64 by the mixed reward against a random sample:",
        "",
        "| model | α | Δ none | Δ correct | Δ wrong | Δ overall | share from wrong "
        "| kept vs α = 0 | Δ informed, wrong |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, b in payload["bon"].items():
        for alpha_key, entry in b["alphas"].items():
            p, inf = entry["proxy_mean"], entry["informed_mean"]
            lines.append(
                f"| {model} | {alpha_key.split('=')[1]} | {_chg(p, 'none')} | "
                f"{_chg(p, 'correct')} | {_chg(p, 'wrong')} | "
                f"{fmt_num(p['overall_change'], 2, sign=True)} | "
                f"{fmt_pct(p['share_of_change'].get('wrong'))} | "
                f"{fmt_pct(p.get('kept_vs_alpha_0'))} | {_chg(inf, 'wrong')} |"
            )
    lines += ["", *_interpretation(payload), ""]
    return lines


def _interpretation(payload: dict[str, Any]) -> list[str]:
    lines = []
    conds = payload["conditions"]
    ref = (conds.get(REFERENCE_CONDITION) or {}).get("proxy_mean")
    if ref:
        lines.append(
            f"- Under α = 0 training, {fmt_pct(ref['share_of_change'].get('wrong'))} of the proxy "
            f"gain is made on wrong-opinion prompts (a third of the prompts): Δ wrong "
            f"{fmt_num(ref['by_opinion']['wrong']['mean'], 2, sign=True)} against Δ none "
            f"{fmt_num(ref['by_opinion']['none']['mean'], 2, sign=True)}."
        )
    kept_train = {
        c: v["proxy_mean"].get("kept_vs_alpha_0")
        for c, v in conds.items()
        if v.get("proxy_mean") and c != REFERENCE_CONDITION
    }
    kept_sel = {
        m: {a: e["proxy_mean"].get("kept_vs_alpha_0") for a, e in b["alphas"].items()}
        for m, b in payload["bon"].items()
    }
    for cond, kept in kept_train.items():
        if not cond.startswith("alpha="):
            lines.append(
                f"- Training with the {condition_label(cond)} kept {fmt_pct(kept)} of "
                "the α = 0 proxy gain."
            )
            continue
        sel = [f"{fmt_pct(per.get(cond))} on {m}" for m, per in kept_sel.items() if cond in per]
        lines.append(
            f"- At {condition_label(cond)}, selection kept "
            + (", ".join(sel) or "—")
            + f" of the α = 0 proxy gain; training kept {fmt_pct(kept)}."
        )
    return lines

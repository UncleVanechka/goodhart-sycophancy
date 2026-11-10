"""(c) Break-even α: from what weight on the informed rating does agreement stop paying?

All estimates use the Best-of-N per-sample files (64 samples per prompt at temperature 1, each
rated by the agreeable user right away, ``immediate``, and after being told the answer,
``informed``) and the mixed reward ``r = (1 - α) * immediate + α * informed``.

1. **Raw gap.** On wrong-opinion prompts, samples pooled: ``g_a`` is the mean immediate rating
   of samples that agree with the user minus that of samples with the gold letter, ``g_i`` the
   mean informed rating of gold samples minus that of agreeing ones. Agreement pays while
   ``(1 - α) g_a > α g_i``, so the break-even is ``α* = g_a / (g_a + g_i)``.
2. **Group-normalized push.** The GRPO advantage within a group of samples of one prompt,
   ``(r - mean) / (std + 1e-4)`` with the unbiased std, as TRL computes it with
   ``scale_rewards="group"``. The push is a difference of mean advantages, pooled over prompts;
   ``α*`` is where it falls to zero, linear between the points of a 0.01 grid. Three contrasts:

   - ``wrong_vs_gold``: agreeing samples minus samples with the gold letter, wrong-opinion
     prompts;
   - ``wrong_vs_rest``: agreeing samples minus every other sample (other wrong letters and
     unparsed replies included), wrong-opinion prompts. Advantages sum to zero within a group,
     so over all forms this has the sign of the net advantage on agreement, the first-order
     policy-gradient push on the probability of agreeing;
   - ``opinionated_vs_rest``: samples that give the user's letter minus every other sample,
     wrong- and correct-opinion prompts pooled. Correct-opinion prompts reward following at
     every α, so its α* is higher.

   Groups are a prompt's 64 samples (``group=64``) or, as in training, consecutive groups of 8
   (``group=8``). This is the *64-sample estimate* of the push: the advantage is TRL's, the
   samples are not (see "Sampler" below).
3. **Best-of-N zero crossing.** On the grid α = 0, 0.05, ..., 1: sycophancy of the expected
   Best-of-64 pick by the mixed reward minus that of a random sample; ``α*`` is where the
   change falls to zero.

Every estimate is split by reply form (``common.reply_form``): ``bare`` (under 20 characters
before the last ``Answer:`` marker, or in total when there is none), ``long`` (otherwise at
least 200 characters in total), ``medium`` (the rest) and ``all``. For one form, (1) uses only
that form's samples, (2) normalizes over the whole group, as training does, and then compares
that form's samples, and (3) selects only among that form's samples of a prompt and skips
prompts with fewer than two. A form with fewer than 20 samples on a side of a contrast (or 20
prompts for (3)) gets no estimate. Intervals are 95 % percentile bootstraps over questions; a
bound with no crossing in [0, 1] is ``None`` in the JSON and ``>1`` in the report.

**Sampler.** The Best-of-N samples were drawn with top-p 0.95 and the model's default top-k
(Qwen3 ships top-k 20), in bfloat16; GRPO's rollouts used top-p 1 and top-k 0, in float32. The
push is therefore computed on a different sampler from the one training saw, with a lighter
tail of unlikely replies, and is reported as the 64-sample estimate, not as training's step-0
push. A TRL-matched step-0 value will come from the base policy's samples of the M3 sampling
stage (``goodhart.experiments.sample_rate``, which uses the rollout settings).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from goodhart.analysis.common import (
    FORMS,
    BonSamples,
    downward_zero_crossings,
    fmt_interval,
    fmt_num,
    fmt_pct,
    percentile_interval,
    ratio,
    resample_weights,
    selection_weights,
)

PUSH_GRID = np.round(np.linspace(0.0, 1.0, 101), 2)
BON_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
GROUP_SIZES = (64, 8)
ADV_EPS = 1e-4
FORM_KEYS = ("all", *FORMS)
# contrast -> (prompts, samples compared with the ones that give the user's letter)
SCOPES = {
    "wrong_vs_gold": ("wrong", "gold"),
    "wrong_vs_rest": ("wrong", "rest"),
    "opinionated_vs_rest": ("opinionated", "rest"),
}
MIN_SAMPLES = 20
MIN_PROMPTS = 20
TOO_FEW = "too few samples"
FLAT_BAND = 0.01
SAMPLER_NOTE = (
    "Best-of-N samples: top-p 0.95 with the model's default top-k, bfloat16; GRPO rollouts: "
    "top-p 1, top-k 0, float32. The push is the 64-sample estimate on the Best-of-N sampler, not "
    "training's step-0 push; a TRL-matched step-0 value comes from the base policy's samples "
    "of the sample_rate stage (M3)"
)


def group_advantages(rewards: np.ndarray, group_size: int) -> np.ndarray:
    """``(r - group mean) / (group std + 1e-4)`` over consecutive groups of ``group_size``
    samples of every row; the std is the unbiased one, as ``torch.std``."""
    n_rows, n_samples = rewards.shape
    if group_size < 2 or n_samples % group_size:
        raise ValueError(f"group size {group_size} does not split {n_samples} samples")
    groups = rewards.reshape(n_rows, n_samples // group_size, group_size)
    mean = groups.mean(axis=2, keepdims=True)
    std = groups.std(axis=2, ddof=1, keepdims=True)
    return ((groups - mean) / (std + ADV_EPS)).reshape(n_rows, n_samples)


def _per_question(values: np.ndarray, q_index: np.ndarray, n_q: int) -> np.ndarray:
    """Sum of per-record ``values`` (``[R]`` or ``[R, G]``) over the records of each question."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.bincount(q_index, weights=values, minlength=n_q)
    out = np.zeros((n_q, values.shape[1]))
    np.add.at(out, q_index, values)
    return out


def break_even(g_a: np.ndarray, g_i: np.ndarray) -> np.ndarray:
    """``g_a / (g_a + g_i)`` where both gaps favour their side; 0 where agreement never pays,
    ``inf`` where it always pays, NaN when both gaps point the unexpected way."""
    g_a, g_i = np.asarray(g_a, dtype=float), np.asarray(g_i, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        both = g_a / (g_a + g_i)
    return np.select(
        [(g_a > 0) & (g_i > 0), (g_a <= 0) & (g_i >= 0), (g_a > 0) & (g_i <= 0)],
        [both, np.zeros_like(g_a), np.full_like(g_a, np.inf)],
        default=np.nan,
    )


def alpha_note(value: float) -> str | None:
    """Why an α* is missing from the JSON: ``inf`` -> no crossing in [0, 1], NaN -> undefined."""
    if np.isnan(value):
        return "undefined"
    if np.isinf(value):
        return "no crossing in [0, 1]"
    return None


def _too_few(entry: dict[str, Any]) -> dict[str, Any]:
    entry.update({"alpha_star": None, "alpha_star_note": TOO_FEW, "alpha_star_ci": None})
    return entry


def raw_gap(bon: BonSamples, weights: np.ndarray) -> dict[str, Any]:
    """Estimate (1) for every form; ``weights`` is ``[B, Q]`` question resamples."""
    q_index, n_q = bon.question_index()
    wrong = (bon.opinion == "wrong")[:, None]
    out: dict[str, Any] = {}
    n_wrong_samples = int(wrong.sum() * bon.n_samples)
    for form in FORM_KEYS:
        mask = bon.form_mask(form) & wrong
        agree, gold = bon.agree & mask, bon.correct & mask
        sums = {
            "n_agree": agree.sum(axis=1),
            "n_gold": gold.sum(axis=1),
            "imm_agree": np.where(agree, bon.immediate, 0).sum(axis=1),
            "imm_gold": np.where(gold, bon.immediate, 0).sum(axis=1),
            "inf_agree": np.where(agree, bon.informed, 0).sum(axis=1),
            "inf_gold": np.where(gold, bon.informed, 0).sum(axis=1),
        }
        per_q = {k: _per_question(v, q_index, n_q) for k, v in sums.items()}

        def gaps(totals: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            g_a = ratio(totals["imm_agree"], totals["n_agree"]) - ratio(
                totals["imm_gold"], totals["n_gold"]
            )
            g_i = ratio(totals["inf_gold"], totals["n_gold"]) - ratio(
                totals["inf_agree"], totals["n_agree"]
            )
            return g_a, g_i

        point = {k: np.array([v.sum()]) for k, v in per_q.items()}
        g_a, g_i = gaps(point)
        entry: dict[str, Any] = {
            "share_of_wrong_opinion_samples": float(mask.sum() / n_wrong_samples),
            "n_agree": int(point["n_agree"][0]),
            "n_gold": int(point["n_gold"][0]),
            "agree_immediate": float(ratio(point["imm_agree"], point["n_agree"])[0]),
            "gold_immediate": float(ratio(point["imm_gold"], point["n_gold"])[0]),
            "agree_informed": float(ratio(point["inf_agree"], point["n_agree"])[0]),
            "gold_informed": float(ratio(point["inf_gold"], point["n_gold"])[0]),
            "g_agreeable": float(g_a[0]),
            "g_informed": float(g_i[0]),
        }
        if min(entry["n_agree"], entry["n_gold"]) < MIN_SAMPLES:
            out[form] = _too_few(entry)
            continue
        boot_a, boot_i = gaps({k: weights @ v for k, v in per_q.items()})
        alpha_star = float(break_even(g_a, g_i)[0])
        entry.update(
            {
                "g_agreeable_ci": percentile_interval(boot_a),
                "g_informed_ci": percentile_interval(boot_i),
                "alpha_star": alpha_star,
                "alpha_star_note": alpha_note(alpha_star),
                "alpha_star_ci": percentile_interval(break_even(boot_a, boot_i)),
            }
        )
        out[form] = entry
    return out


def _crossing_summary(
    point_curve: np.ndarray, boot_curves: np.ndarray, grid: np.ndarray
) -> dict[str, Any]:
    point = float(downward_zero_crossings(grid, point_curve[None, :])[0])
    boot = downward_zero_crossings(grid, boot_curves)
    defined = boot[~np.isnan(boot)]
    on_coarse_grid = np.isclose(grid * 20, np.round(grid * 20))
    return {
        "alpha_star": point,
        "alpha_star_note": alpha_note(point),
        "alpha_star_ci": percentile_interval(boot),
        "share_not_crossed": float(np.isinf(defined).mean()) if defined.size else None,
        "curve": {
            f"{a:.2f}": float(v)
            for a, v, keep in zip(grid, point_curve, on_coarse_grid, strict=True)
            if keep
        },
    }


def _scope_masks(bon: BonSamples, scope: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prompt rows, samples that give the user's letter, and the samples they are compared to."""
    prompts, against = SCOPES[scope]
    opinions = ["wrong"] if prompts == "wrong" else ["wrong", "correct"]
    rows = np.isin(bon.opinion, opinions)
    follow = bon.agree
    other = bon.correct & ~follow if against == "gold" else ~follow
    return rows, follow, other


def push(
    bon: BonSamples, group_size: int, scope: str, weights: np.ndarray, grid: np.ndarray = PUSH_GRID
) -> dict[str, Any]:
    """Estimate (2) for every form and one contrast of ``SCOPES``."""
    q_index, n_q = bon.question_index()
    rows, follow, other = _scope_masks(bon, scope)
    masks = {form: bon.form_mask(form) & rows[:, None] for form in FORM_KEYS}
    sum_follow = {form: np.zeros((len(rows), len(grid))) for form in FORM_KEYS}
    sum_other = {form: np.zeros((len(rows), len(grid))) for form in FORM_KEYS}
    for j, alpha in enumerate(grid):
        adv = group_advantages((1.0 - alpha) * bon.immediate + alpha * bon.informed, group_size)
        for form, mask in masks.items():
            sum_follow[form][:, j] = np.where(follow & mask, adv, 0.0).sum(axis=1)
            sum_other[form][:, j] = np.where(other & mask, adv, 0.0).sum(axis=1)
    out: dict[str, Any] = {}
    for form, mask in masks.items():
        sf = _per_question(sum_follow[form], q_index, n_q)
        so = _per_question(sum_other[form], q_index, n_q)
        nf = _per_question((follow & mask).sum(axis=1), q_index, n_q)
        no = _per_question((other & mask).sum(axis=1), q_index, n_q)
        counts = {"n_follow": int(nf.sum()), "n_other": int(no.sum())}
        if min(counts.values()) < MIN_SAMPLES:
            out[form] = _too_few(counts)
            continue
        point = ratio(sf.sum(axis=0), nf.sum()) - ratio(so.sum(axis=0), no.sum())
        boot = ratio(weights @ sf, (weights @ nf)[:, None]) - ratio(
            weights @ so, (weights @ no)[:, None]
        )
        out[form] = {**_crossing_summary(point, boot, grid), **counts}
    return out


def bon_crossing(bon: BonSamples, weights: np.ndarray, grid: np.ndarray = BON_GRID) -> dict:
    """Estimate (3) for every form."""
    q_index, n_q = bon.question_index()
    wrong = bon.opinion == "wrong"
    q_wrong = q_index[wrong]
    agree = bon.agree[wrong].astype(float)
    imm, inf = bon.immediate[wrong], bon.informed[wrong]
    out: dict[str, Any] = {}
    for form in FORM_KEYS:
        pool = bon.form_mask(form)[wrong]
        has = pool.sum(axis=1) >= 2
        if has.sum() < MIN_PROMPTS:
            out[form] = _too_few({"n_prompts": int(has.sum())})
            continue
        ref = np.where(has, ratio((agree * pool).sum(axis=1), pool.sum(axis=1)), 0.0)
        change = np.zeros((len(agree), len(grid)))
        best_rate = np.zeros(len(grid))
        for j, alpha in enumerate(grid):
            w = selection_weights((1.0 - alpha) * imm + alpha * inf, pool)
            best = (w * agree).sum(axis=1)
            change[:, j] = np.where(has, best - ref, 0.0)
            best_rate[j] = best[has].mean()
        num = _per_question(change, q_wrong, n_q)
        den = _per_question(has.astype(float), q_wrong, n_q)
        point = ratio(num.sum(axis=0), den.sum())
        boot = ratio(weights @ num, (weights @ den)[:, None])
        entry = _crossing_summary(point, boot, grid)
        entry.update(
            {
                "n_prompts": int(has.sum()),
                "random_sample_sycophancy": float(ref[has].mean()),
                "best_of_64_sycophancy": {
                    f"{a:.2f}": float(v) for a, v in zip(grid, best_rate, strict=True)
                },
            }
        )
        out[form] = entry
    return out


def form_shares(bon: BonSamples) -> dict[str, Any]:
    """Share of samples of each form per opinion type, and agreement by form under a wrong
    opinion."""
    out: dict[str, Any] = {}
    for opinion in ("none", "correct", "wrong"):
        rows = bon.opinion == opinion
        out[opinion] = {form: float((bon.form[rows] == form).mean()) for form in FORMS}
    wrong = bon.opinion == "wrong"
    out["agree_rate_under_wrong_opinion"] = {
        form: float(bon.agree[wrong][bon.form[wrong] == form].mean())
        if (bon.form[wrong] == form).any()
        else None
        for form in FORMS
    }
    return out


def analyze_model(bon: BonSamples, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """All three estimates for one model."""
    _, n_q = bon.question_index()
    weights = resample_weights(n_q, n_boot, seed)
    return {
        "model": bon.model,
        "n_questions": n_q,
        "n_samples_per_prompt": bon.n_samples,
        "forms": form_shares(bon),
        "raw_gap": raw_gap(bon, weights),
        "push": {
            f"group={g}": {scope: push(bon, g, scope, weights) for scope in SCOPES}
            for g in GROUP_SIZES
            if bon.n_samples % g == 0
        },
        "bon_crossing": bon_crossing(bon, weights),
    }


def analyze(bons: Sequence[BonSamples], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """The full break-even payload, one entry per model."""
    return {
        "definitions": {
            "mixed_reward": "(1 - alpha) * immediate + alpha * informed",
            "raw_gap": "alpha* = g_a / (g_a + g_i) on wrong-opinion samples",
            "push": "64-sample estimate: difference of mean group-normalized advantages on the "
            "Best-of-N samples; wrong_vs_gold: agreeing minus gold samples; wrong_vs_rest: "
            "agreeing minus all other samples (sign of the net advantage on agreement); "
            "opinionated_vs_rest: following minus all other samples on wrong and correct "
            "opinions",
            "sampler": SAMPLER_NOTE,
            "bon_crossing": "Best-of-64 sycophancy minus random-sample sycophancy",
            "forms": "bare: < 20 chars before the last 'Answer:' (or in total); long: "
            ">= 200 chars in total; medium: the rest",
            "min_counts": f"no estimate with < {MIN_SAMPLES} samples on a side "
            f"(< {MIN_PROMPTS} prompts for the BoN crossing)",
            "ci": "95% percentile bootstrap over questions; null bound = no crossing in [0, 1]",
        },
        "models": {bon.model: analyze_model(bon, n_boot, seed) for bon in bons},
    }


# ----------------------------------------------------------------------------- report


def fmt_alpha(entry: dict[str, Any]) -> str:
    """α* with its interval: ``0.40 [0.35, 0.45]``, ``>1`` beyond the grid, ``n/a`` without an
    estimate."""
    value = entry.get("alpha_star")
    if value is None:
        note = entry.get("alpha_star_note")
        text = ">1" if note == "no crossing in [0, 1]" else "n/a"
    else:
        text = fmt_num(value, 2)
    return text + fmt_interval(entry.get("alpha_star_ci"), 1, 2)


def render_markdown(payload: dict[str, Any]) -> list[str]:
    """Section (c) of ``summary.md``."""
    lines = ["## (c) Break-even α: where agreement stops paying", ""]
    lines += [
        "From the Best-of-N samples (64 per prompt, both ratings on every sample), with the "
        "mixed reward (1 − α)·immediate + α·informed. *Raw gap*: α* = g_a / (g_a + g_i), g_a the "
        "agreeable user's rating of agreeing minus gold samples, g_i the informed user's rating "
        "of gold minus agreeing samples, wrong-opinion prompts. *Push* (the 64-sample estimate): "
        "GRPO's group-normalized advantage, mean over agreeing samples minus mean over gold "
        "samples (*vs gold*) or over every other sample (*vs rest*, the sign of the net "
        "advantage on agreement), zero "
        "crossing on a 0.01 grid, groups of 64 (a prompt) or 8 (as in training); *opinionated*: "
        "following the user minus the rest, wrong- and correct-opinion prompts pooled. *BoN*: "
        "Best-of-64 sycophancy minus a random sample's, zero crossing on a 0.05 grid. *Form*: "
        "bare = under 20 characters before the last `Answer:` (or in total), long = 200+ "
        "characters in total, medium = the rest; the push of a form normalizes over the whole "
        "group, as training does. 95% bootstrap over questions; n/a = fewer than 20 samples on a "
        "side.",
        "",
        "The samples are Best-of-N's (top-p 0.95 with the model's default top-k, bfloat16), not "
        "GRPO's (top-p 1, top-k 0, float32): the push is the 64-sample estimate on another "
        "sampler, not training's step-0 push (see Caveats).",
        "",
    ]
    for model, m in payload["models"].items():
        p64, p8 = m["push"].get("group=64", {}), m["push"].get("group=8", {})

        def cell(push_of: dict[str, Any], scope: str, form: str) -> str:
            return fmt_alpha(push_of[scope][form]) if push_of else "—"

        lines += [
            f"### {model} ({m['n_questions']} questions)",
            "",
            "| form | share of wrong-opinion samples | agree rate | g_a | g_i | raw gap "
            "| push vs gold | push vs rest | push vs rest, group 8 | opinionated | BoN |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        agree_rates = m["forms"]["agree_rate_under_wrong_opinion"]
        for form in FORM_KEYS:
            raw = m["raw_gap"][form]
            lines.append(
                f"| {form} | {fmt_pct(raw['share_of_wrong_opinion_samples'])} | "
                f"{fmt_pct(agree_rates.get(form)) if form != 'all' else '—'} | "
                f"{fmt_num(raw['g_agreeable'], 2, sign=True)} | "
                f"{fmt_num(raw['g_informed'], 2, sign=True)} | {fmt_alpha(raw)} | "
                f"{cell(p64, 'wrong_vs_gold', form)} | {cell(p64, 'wrong_vs_rest', form)} | "
                f"{cell(p8, 'wrong_vs_rest', form)} | "
                f"{cell(p64, 'opinionated_vs_rest', form)} | {fmt_alpha(m['bon_crossing'][form])} |"
            )
        lines += ["", *_interpretation(model, m), ""]
    return lines


def _interpretation(model: str, m: dict[str, Any]) -> list[str]:
    raw, bon = m["raw_gap"], m["bon_crossing"]
    p64 = m["push"].get("group=64", {})
    rest = p64.get("wrong_vs_rest", {}).get("all", {})
    lines = [
        f"- {model}, all forms: raw gap {fmt_alpha(raw['all'])}; push vs gold "
        f"{fmt_alpha(p64.get('wrong_vs_gold', {}).get('all', {}))}, vs rest {fmt_alpha(rest)}; "
        f"BoN crossing {fmt_alpha(bon['all'])}."
    ]
    point, ci = raw["all"].get("alpha_star"), rest.get("alpha_star_ci")
    if point is not None and ci and ci[0] is not None:
        inside = ci[0] <= point and (ci[1] is None or point <= ci[1])
        lines[-1] += (
            f" The raw-gap value lies {'inside' if inside else 'outside'} the push-vs-rest "
            "interval."
        )
    flat = [float(a) for a, v in (bon["all"].get("curve") or {}).items() if abs(v) <= FLAT_BAND]
    crossing = bon["all"].get("alpha_star")
    if len(flat) >= 2 and crossing is not None:
        hi = max(max(flat), crossing)
        lines.append(
            f"- The Best-of-64 change in sycophancy stays within {100 * FLAT_BAND:.0f} point of "
            f"zero from α = {min(flat):.2f} to {max(flat):.2f} and crosses at {crossing:.2f}: "
            f"selection is close to neutral over α {min(flat):.2f}–{hi:.2f}, so the crossing "
            "itself is poorly located."
        )
    bare, long_ = raw["bare"], raw["long"]
    a_bare, a_long = bare.get("alpha_star"), long_.get("alpha_star")
    if a_bare is not None and a_long is not None:
        tail = (
            ": a weight between the two makes agreement unprofitable for argued replies and "
            "leaves it profitable for bare letters."
            if a_bare > a_long
            else ": bare letters do not need the larger weight here."
        )
        lines.append(
            f"- By form, the raw gap gives α* = {fmt_num(a_bare, 2)} for bare letters "
            f"(g_a {fmt_num(bare['g_agreeable'], 2, sign=True)}, g_i "
            f"{fmt_num(bare['g_informed'], 2, sign=True)}) and {fmt_num(a_long, 2)} for long "
            f"replies (g_a {fmt_num(long_['g_agreeable'], 2, sign=True)}, g_i "
            f"{fmt_num(long_['g_informed'], 2, sign=True)})" + tail
        )
    long_push = p64.get("wrong_vs_gold", {}).get("long", {})
    curve = long_push.get("curve") or {}
    if long_push.get("alpha_star") == 0.0 and curve.get("0.00") is not None:
        lines.append(
            f"- Within the group, long agreeing replies get a lower advantage than long gold "
            f"replies already at α = 0 (push vs gold {fmt_num(curve['0.00'], 3, sign=True)}): "
            "they are rated against the bare letters in the same group, so the push toward "
            "agreement is carried by the bare letters."
        )
    elif not bare.get("n_agree"):
        lines.append(f"- {model} gives almost no bare letters, so the split rests on long replies.")
    return lines

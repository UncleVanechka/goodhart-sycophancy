"""Predictions 6-8 (break-even α at saved checkpoints) on toy samples: no model, no network."""

import json
import math

import numpy as np
import pytest

from goodhart.analysis import breakeven
from goodhart.analysis import prereg_checkpoints as pc
from goodhart.analysis.common import percentile_interval, resample_weights
from goodhart.io import write_jsonl

GOLD, USER, OTHER = "A", "D", "B"
LETTER = {"gold": GOLD, "user": USER, "other": OTHER}
LONG_BODY = "The user suggests one option, but the working below points to another one. " * 3
N_BOOT = 60

# (form, who, immediate, informed, truth_seeking) for the 8 samples of a wrong-opinion prompt
TWO_FORMS = (
    [("bare", "user", 9.0, 3.0, 2.0)] * 2
    + [("bare", "gold", 4.0, 9.0, 9.0)] * 2
    + [("long", "user", 7.0, 5.0, 4.0)] * 2
    + [("long", "gold", 6.5, 9.0, 9.0)] * 2
)
PLAIN = [("bare", "gold", 8.0, 9.0, 8.0)] * 8


def _text(form: str, letter: str) -> str:
    return f"Answer: {letter}" if form == "bare" else f"{LONG_BODY}\n\nAnswer: {letter}"


def _bare(agree: tuple[float, float], gold: tuple[float, float]) -> list[tuple]:
    """Four bare agreeing and four bare gold samples with the given (immediate, informed)."""
    return [("bare", "user", *agree, 2.0)] * 4 + [("bare", "gold", *gold, 9.0)] * 4


def make_rows(
    policy: str,
    wrong: list[tuple],
    n_questions: int = 12,
    truth: bool = False,
    jitter: bool = False,
    n_samples: int = 8,
) -> list[dict]:
    """``n_questions`` x 3 opinion types x ``n_samples`` rows in the ``samples.jsonl`` format;
    no-opinion and correct-opinion prompts get eight identical gold letters. ``jitter`` shifts
    the immediate rating of agreeing samples by question so that resamples differ."""
    rows = []
    for q in range(n_questions):
        for opinion, stated in (("none", None), ("correct", GOLD), ("wrong", USER)):
            spec = wrong if opinion == "wrong" else PLAIN
            for idx, (form, who, immediate, informed, truth_seeking) in enumerate(spec[:n_samples]):
                letter = LETTER[who]
                if jitter and who == "user":
                    immediate = immediate + (q % 3 - 1) * 0.5
                ratings = {"agreeable": immediate, "agreeable_informed": informed}
                if truth:
                    ratings["truth_seeking"] = truth_seeking
                rows.append(
                    {
                        "policy": policy,
                        "split": "val",
                        "record_id": f"q{q:02d}-{opinion}",
                        "base_id": f"q{q:02d}",
                        "opinion": opinion,
                        "sample_idx": idx,
                        "text": _text(form, letter),
                        "parsed": letter,
                        "gold": GOLD,
                        "opinion_answer": stated,
                        "ratings": ratings,
                        "malformed": {},
                        "immediate": immediate,
                        "informed": informed,
                    }
                )
    return rows


def make_samples(policy: str, wrong: list[tuple], **kwargs) -> pc.PolicySamples:
    return pc.samples_from_rows(policy, make_rows(policy, wrong, **kwargs))


def weights_for(samples: pc.PolicySamples, n_boot: int = N_BOOT) -> np.ndarray:
    return resample_weights(samples.bon.question_index()[1], n_boot, 0)


# ----------------------------------------------------------------------------- policies, loading


def test_parse_policy_reads_run_condition_seed_and_step():
    base = pc.parse_policy("base")
    assert (base.condition, base.alpha, base.step, base.reward_key) == ("base", None, 0, None)
    immediate = pc.parse_policy("grpo-immediate-s43-ckpt100")
    assert (immediate.run, immediate.condition, immediate.alpha) == (
        "grpo-immediate-s43",
        "alpha=0",
        0.0,
    )
    assert (immediate.seed, immediate.step) == (43, 100)
    delayed = pc.parse_policy("grpo-delayed-a0.3")
    assert (delayed.run, delayed.condition, delayed.seed, delayed.step) == (
        "grpo-delayed-a0.3",
        "alpha=0.3",
        42,
        300,
    )
    ckpt = pc.parse_policy("grpo-delayed-s44-a0.6-ckpt200")
    assert (ckpt.run, ckpt.alpha, ckpt.seed, ckpt.step) == ("grpo-delayed-s44-a0.6", 0.6, 44, 200)
    mix = pc.parse_policy("grpo-persona-mix-s43")
    assert (mix.condition, mix.alpha, mix.reward_key) == ("persona_mix", None, "persona_mix")
    assert pc.policy_name("grpo-delayed-a0.3", 300) == "grpo-delayed-a0.3"
    assert pc.policy_name("grpo-delayed-a0.3", 200) == "grpo-delayed-a0.3-ckpt200"


@pytest.mark.parametrize("name", ["base-none-val-test", "grpo-delayed-s43", "grpo-immediate-a0.3"])
def test_parse_policy_rejects_unknown_names(name):
    with pytest.raises(ValueError):
        pc.parse_policy(name)


def test_samples_from_rows_builds_arrays_and_extra_personas():
    samples = make_samples("grpo-persona-mix", TWO_FORMS, truth=True)
    bon = samples.bon
    assert bon.parsed.shape == (36, 8)
    assert bon.n_samples == 8
    assert set(samples.extra) == {"truth_seeking"}
    wrong = bon.opinion == "wrong"
    assert bon.agree[wrong].mean() == 0.5
    assert bon.agree[~wrong].sum() == (bon.opinion == "correct").sum() * 8
    assert set(bon.form[wrong].ravel()) == {"bare", "long"}
    assert samples.policy.condition == "persona_mix"


def test_samples_from_rows_rejects_unrated_and_ragged_rows():
    rows = make_rows("base", TWO_FORMS)
    with pytest.raises(ValueError, match="sample counts"):
        pc.samples_from_rows("base", rows[:-1])
    unrated = [{k: v for k, v in row.items() if k not in ("immediate", "informed")} for row in rows]
    assert not pc.is_rated(unrated)
    with pytest.raises(ValueError, match="ratings"):
        pc.samples_from_rows("base", unrated)


def test_load_policies_takes_the_larger_copy_and_lists_skipped(tmp_path):
    def put(run: str, policy: str, rows: list[dict]) -> None:
        write_jsonl(tmp_path / run / "results" / "sample_rate" / policy / "samples.jsonl", rows)

    full = make_rows("base", TWO_FORMS)
    put("run_smoke", "base", make_rows("base", TWO_FORMS, n_questions=1, n_samples=2))
    put("run_a", "base", full)
    put("run_b", "grpo-delayed-a0.3", make_rows("grpo-delayed-a0.3", TWO_FORMS))
    put("run_b", "base-none-val-test", [{**row, "ratings": {}} for row in full[:8]])
    put("run_b", "odd-name", full)
    put("run_b", "grpo-persona-mix", make_rows("grpo-persona-mix", TWO_FORMS, n_samples=4))
    files = pc.find_sample_files(tmp_path)
    assert files["base"].parents[3].name == "run_a"
    loaded, skipped = pc.load_policies(tmp_path)
    assert list(loaded) == ["base", "grpo-delayed-a0.3"]
    assert loaded["base"].source == "run_a"
    assert loaded["base"].bon.parsed.shape == (36, 8)
    assert set(skipped) == {"base-none-val-test", "odd-name", "grpo-persona-mix"}
    assert skipped["grpo-persona-mix"] == "4 samples per prompt, not 8"


# ----------------------------------------------------------------------------- estimators


def test_raw_gap_gives_the_hand_computed_break_even_by_form():
    samples = make_samples("base", TWO_FORMS)
    local = pc.local_break_even(samples.bon, weights_for(samples))
    raw = local["raw_gap"]
    assert raw["bare"]["alpha_star"] == pytest.approx(5.0 / 11.0)
    assert raw["long"]["alpha_star"] == pytest.approx(0.5 / 4.5)
    assert raw["all"]["alpha_star"] == pytest.approx(2.75 / 7.75)
    # no medium replies: the pooled non-bare form is the long form
    assert raw["nonbare"]["alpha_star"] == pytest.approx(raw["long"]["alpha_star"])
    assert raw["medium"]["alpha_star_note"] == breakeven.TOO_FEW
    assert pc.side_counts(raw["bare"]) == [24, 24]
    assert set(local["push_at_alpha"]) == {"wrong_vs_rest", "opinionated_vs_rest"}


def test_replicates_reproduce_breakeven_points_and_intervals():
    samples = make_samples("base", TWO_FORMS, jitter=True)
    bon, weights = samples.bon, weights_for(samples)
    raw = breakeven.raw_gap(bon, weights)
    push = breakeven.push(bon, pc.GROUP, pc.PUSH_SCOPE, weights)
    for form in ("all", "bare", "long"):
        point, boot = pc.raw_gap_replicates(bon, form, weights)
        assert point == pytest.approx(raw[form]["alpha_star"])
        assert percentile_interval(boot) == pytest.approx(raw[form]["alpha_star_ci"])
        point, boot = pc.push_replicates(bon, form, weights)
        assert point == pytest.approx(push[form]["alpha_star"])
        assert percentile_interval(boot) == pytest.approx(push[form]["alpha_star_ci"])
    point, boot = pc.raw_gap_replicates(bon, "medium", weights)
    assert math.isnan(point) and np.isnan(boot).all()
    assert len(set(np.round(pc.raw_gap_replicates(bon, "all", weights)[1], 6))) > 1


def test_push_curve_changes_sign_at_the_break_even():
    samples = make_samples("base", _bare(agree=(9.0, 3.0), gold=(5.0, 9.0)))
    bon, weights = samples.bon, weights_for(samples)
    point, _, counts = pc.push_curves(bon, "all", weights, grid=np.array([0.0, 0.3, 0.6]))
    assert counts == (48, 48)
    # g_a = 4, g_i = 6: agreement pays below α = 0.4
    assert point[0] > 0 and point[1] > 0 and point[2] < 0
    assert pc.push_replicates(bon, "all", weights)[0] == pytest.approx(0.4, abs=0.011)
    at = pc.push_at_alphas(bon, weights, pc.PUSH_SCOPE)["all"]
    assert at["alpha=0.3"]["value"] > 0 > at["alpha=0.6"]["value"]
    assert at["alpha=0.3"]["share_resamples_positive"] == 1.0


def test_alpha_value_reads_notes_in_memory_and_from_json():
    assert pc.alpha_value({"alpha_star": 0.4, "alpha_star_note": None}) == 0.4
    assert pc.alpha_value({"alpha_star": None, "alpha_star_note": pc.NO_CROSSING}) == math.inf
    assert pc.alpha_value({"alpha_star": math.inf, "alpha_star_note": pc.NO_CROSSING}) == math.inf
    assert pc.alpha_value({"alpha_star": math.nan, "alpha_star_note": pc.UNDEFINED}) is None
    assert pc.alpha_value({"alpha_star": None, "alpha_star_note": breakeven.TOO_FEW}) is None
    assert not pc.has_enough({"alpha_star_note": breakeven.TOO_FEW})
    assert pc.has_enough({"alpha_star_note": pc.UNDEFINED})


# ----------------------------------------------------------------------------- descriptive


def test_reward_table_means_and_zero_variance_groups():
    samples = make_samples("grpo-persona-mix", TWO_FORMS, truth=True)
    table = pc.reward_table(samples, weights_for(samples))
    # wrong-opinion prompts mix ratings, the other two types are eight identical replies
    assert table["alpha=0"]["zero_variance_groups"] == {
        "all": pytest.approx(2 / 3),
        "none": 1.0,
        "correct": 1.0,
        "wrong": 0.0,
    }
    wrong_immediate = (9 + 4 + 7 + 6.5) / 4
    wrong_informed = (3 + 9 + 5 + 9) / 4
    assert table["alpha=0"]["mean@wrong"] == pytest.approx(wrong_immediate)
    assert table["alpha=0.3"]["mean@wrong"] == pytest.approx(
        0.7 * wrong_immediate + 0.3 * wrong_informed
    )
    assert table["alpha=0.3"]["mean"] == pytest.approx(
        (0.7 * wrong_immediate + 0.3 * wrong_informed + 2 * (0.7 * 8 + 0.3 * 9)) / 3
    )
    wrong_truth = (2 + 9 + 4 + 9) / 4
    assert table["truth_seeking"]["mean@wrong"] == pytest.approx(wrong_truth)
    assert table["persona_mix"]["mean@wrong"] == pytest.approx(
        0.7 * wrong_immediate + 0.3 * wrong_truth
    )
    assert table["persona_mix"]["zero_variance_groups"]["all"] == pytest.approx(2 / 3)
    lo, hi = table["alpha=0.3"]["ci"]
    assert lo == pytest.approx(hi) == pytest.approx(table["alpha=0.3"]["mean"])


def test_zero_variance_share_is_persona_weighted_under_the_mixture():
    # the agreeable user rates all eight replies alike, the truth-seeking user does not
    flat_for_agreeable = [("bare", "user", 8.0, 3.0, 2.0)] * 4 + [
        ("bare", "gold", 8.0, 9.0, 9.0)
    ] * 4
    samples = make_samples("grpo-persona-mix", flat_for_agreeable, truth=True)
    table = pc.reward_table(samples, weights_for(samples))
    assert table["alpha=0"]["zero_variance_groups"]["wrong"] == 1.0
    assert table["truth_seeking"]["zero_variance_groups"]["wrong"] == 0.0
    assert table["persona_mix"]["zero_variance_groups"]["wrong"] == pytest.approx(0.7)


def test_sycophancy_summary_counts_agreeing_wrong_opinion_samples():
    spec = [("bare", "user", 9.0, 3.0, 2.0)] * 6 + [("bare", "gold", 4.0, 9.0, 9.0)] * 2
    samples = make_samples("base", spec)
    out = pc.sycophancy_summary(samples.bon, weights_for(samples))
    assert out["rate"] == pytest.approx(0.75)
    assert out["gold_accuracy"] == pytest.approx(0.25)
    assert out["n_samples"] == 96
    assert out["unparsed"] == 0.0
    assert out["prompts_all_agree"] == 0.0 and out["prompts_none_agree"] == 0.0
    assert out["ci"] == pytest.approx([0.75, 0.75])


def test_mixture_push_root_is_where_the_two_personas_cancel():
    # mirrored ratings: the agreeable user prefers agreeing by as much as the truth-seeking
    # user prefers the gold letter, so the pushes cancel at p = 0.5
    spec = [("bare", "user", 9.0, 3.0, 3.0)] * 4 + [("bare", "gold", 5.0, 9.0, 7.0)] * 4
    samples = make_samples("grpo-persona-mix", spec, truth=True)
    out = pc.mixture_push(samples, weights_for(samples))
    form = out["forms"]["all"]
    step = 2.0 / (np.std([9, 9, 9, 9, 5, 5, 5, 5], ddof=1) + breakeven.ADV_EPS)
    assert form["push_agreeable"]["value"] == pytest.approx(2 * step)
    assert form["push_truth_seeking"]["value"] == pytest.approx(-2 * step)
    assert form["p_star"] == pytest.approx(0.5)
    assert form["net_push_at_p_train"]["value"] == pytest.approx(0.4 * 2 * step)
    assert form["share_p_star_above_p_train"] == 1.0
    assert form["raw_gap"]["p_star"] == pytest.approx(0.5)
    assert out["forms"]["long"]["p_star_note"] == breakeven.TOO_FEW


def test_mixture_push_without_truth_ratings_or_without_a_root():
    plain = make_samples("base", TWO_FORMS)
    assert pc.mixture_push(plain, weights_for(plain)) is None
    both_like_agreeing = [("bare", "user", 9.0, 3.0, 8.0)] * 4 + [
        ("bare", "gold", 5.0, 9.0, 6.0)
    ] * 4
    samples = make_samples("grpo-persona-mix", both_like_agreeing, truth=True)
    form = pc.mixture_push(samples, weights_for(samples))["forms"]["all"]
    assert form["p_star"] is None and form["p_star_note"] == pc.NO_CROSSING
    assert form["net_push_at_p_train"]["value"] > 0


# ----------------------------------------------------------------------------- predictions


def test_verdict_of():
    assert pc.verdict_of([True, True]) == pc.CONFIRMED
    assert pc.verdict_of([True, False, None]) == pc.REFUTED
    assert pc.verdict_of([True, None]) == pc.NOT_TESTABLE
    assert pc.verdict_of([]) == pc.NOT_TESTABLE


def _entry(alpha, n=(40, 40), ci=None, note=None):
    if min(n) < breakeven.MIN_SAMPLES:
        alpha, note, ci = None, breakeven.TOO_FEW, None
    return {
        "alpha_star": alpha,
        "alpha_star_note": note,
        "alpha_star_ci": ci,
        "n_agree": n[0],
        "n_gold": n[1],
    }


def _local(bare, long_, nonbare=None):
    forms = {"bare": bare, "long": long_, "nonbare": nonbare or long_}
    return {"raw_gap": forms, "push": forms}


def _p6(local):
    policies = {name: pc.parse_policy(name) for name in local}
    return pc.prediction_6(local, policies)


def test_prediction_6_confirmed_only_when_both_halves_hold_at_every_eligible_checkpoint():
    good = _local(_entry(0.50, ci=[0.46, 0.55]), _entry(0.10, ci=[0.0, 0.18]))
    out = _p6({"base": _local(_entry(0.2), _entry(0.9)), "grpo-delayed-a0.3": good})
    assert out["verdict"] == pc.CONFIRMED
    assert out["raw_gap"]["n_checkpoints"] == 1  # the base policy is a reference, not a checkpoint
    assert set(out["raw_gap"]["step0_reference"]) == {"base"}
    check = out["raw_gap"]["checkpoints"]["grpo-delayed-a0.3"]
    assert check["bare"]["interval"] == "inside" and check["argued"]["interval"] == "inside"

    argued_high = _local(_entry(0.50, ci=[0.40, 0.55]), _entry(0.35, ci=[0.22, 0.5]))
    out = _p6({"grpo-delayed-a0.3": good, "grpo-delayed-s43-a0.3-ckpt100": argued_high})
    assert out["verdict"] == pc.REFUTED
    assert out["raw_gap"]["n_eligible_ok"] == 1
    assert out["raw_gap"]["argued"] == {
        "n_tested": 2,
        "n_ok": 1,
        "n_interval_inside": 1,
        "n_interval_outside": 1,
    }
    assert out["raw_gap"]["bare"]["n_interval_inside"] == 1
    bad = out["raw_gap"]["checkpoints"]["grpo-delayed-s43-a0.3-ckpt100"]
    assert bad["bare"]["interval"] == "straddles" and bad["argued"]["interval"] == "outside"


def test_prediction_6_eligibility_and_edge_values():
    few_long = _local(_entry(0.30), _entry(0.1, n=(6, 17)))
    out = _p6({"grpo-delayed-a0.3": few_long})
    # not eligible under the both-forms reading, judged on the bare form under the other one
    assert out["verdict"] == pc.NOT_TESTABLE
    assert out["verdict_per_form_reading"]["raw_gap"] == pc.REFUTED
    assert out["raw_gap"]["checkpoints"]["grpo-delayed-a0.3"]["ok"] is None

    always_pays = _local(_entry(None, note=pc.NO_CROSSING), _entry(0.0))
    assert _p6({"grpo-immediate-s43": always_pays})["verdict"] == pc.CONFIRMED
    undefined = _local(_entry(None, note=pc.UNDEFINED), _entry(0.0))
    assert _p6({"grpo-immediate-s43": undefined})["verdict"] == pc.REFUTED
    at_threshold = _local(_entry(0.45), _entry(0.20))
    assert _p6({"grpo-immediate-s43": at_threshold})["verdict"] == pc.CONFIRMED

    pooled = _local(_entry(0.5), _entry(0.1), nonbare=_entry(0.4))
    out = _p6({"grpo-delayed-a0.3": pooled})
    assert out["verdict"] == pc.CONFIRMED
    assert out["verdict_argued_as_nonbare"]["raw_gap"] == pc.REFUTED


def _alpha03_policies(collapsed_200, collapsed_300, s43_200, s44_200, s43_300, s44_300):
    specs = {
        "grpo-delayed-a0.3-ckpt200": collapsed_200,
        "grpo-delayed-a0.3": collapsed_300,
        "grpo-delayed-s43-a0.3-ckpt200": s43_200,
        "grpo-delayed-s44-a0.3-ckpt200": s44_200,
        "grpo-delayed-s43-a0.3": s43_300,
        "grpo-delayed-s44-a0.3": s44_300,
    }
    return {name: make_samples(name, spec) for name, spec in specs.items() if spec is not None}


# break-even α of a `_bare` spec is g_a / (g_a + g_i)
A040 = _bare(agree=(9.0, 3.0), gold=(5.0, 9.0))
A050 = _bare(agree=(9.0, 5.0), gold=(5.0, 9.0))
A020 = _bare(agree=(7.0, 1.0), gold=(5.0, 9.0))
A025 = _bare(agree=(6.0, 6.0), gold=(5.0, 9.0))
HIGH_REWARD = _bare(agree=(10.0, 6.0), gold=(7.0, 10.0))


def test_predictions_7_and_8_confirmed_on_a_toy_collapse():
    samples = _alpha03_policies(A040, A050, A020, A025, HIGH_REWARD, HIGH_REWARD)
    payload = pc.analyze(samples, n_boot=N_BOOT)
    p7 = payload["predictions"]["7"]
    assert p7["verdict"] == pc.CONFIRMED
    assert p7["verdict_by_estimator"] == {"raw_gap": pc.CONFIRMED, "push": pc.CONFIRMED}
    levels = p7["raw_gap"]["levels"]
    assert levels["grpo-delayed-a0.3-ckpt200"]["alpha_star"] == pytest.approx(0.4)
    assert levels["grpo-delayed-a0.3"]["alpha_star"] == pytest.approx(0.5)
    assert levels["grpo-delayed-s43-a0.3-ckpt200"]["alpha_star"] == pytest.approx(0.2)
    contrast = p7["raw_gap"]["contrasts"][
        "grpo-delayed-a0.3-ckpt200 minus grpo-delayed-s44-a0.3-ckpt200"
    ]
    assert contrast["diff"] == pytest.approx(0.15)
    assert contrast["like_for_like"] and contrast["share_resamples_higher"] == 1.0
    assert len(p7["raw_gap"]["contrasts"]) == 4

    p8 = payload["predictions"]["8"]
    collapsed = (0.7 * 9 + 0.3 * 5 + 0.7 * 5 + 0.3 * 9) / 2
    other = (0.7 * 10 + 0.3 * 6 + 0.7 * 7 + 0.3 * 10) / 2
    assert p8["verdict"] == pc.CONFIRMED
    assert p8["collapsed_run"]["mixed_reward"] == pytest.approx((collapsed + 2 * 8.3) / 3)
    assert p8["others"]["grpo-delayed-s43-a0.3"]["diff"] == pytest.approx((collapsed - other) / 3)
    assert p8["others"]["grpo-delayed-s43-a0.3"]["diff_by_opinion"]["none"] == pytest.approx(0.0)
    assert p8["vs_mean_of_others"]["diff"] == pytest.approx((collapsed - other) / 3)
    json.dumps(payload, allow_nan=False)


def test_predictions_7_and_8_refuted_or_not_testable():
    # the collapsed run below 0.30 at step 300 and below the other runs: refuted
    samples = _alpha03_policies(A040, A020, A050, A025, A020, HIGH_REWARD)
    payload = pc.analyze(samples, n_boot=N_BOOT)
    p7 = payload["predictions"]["7"]
    assert p7["verdict"] == pc.REFUTED
    assert p7["raw_gap"]["verdict_above_threshold"] == pc.REFUTED
    assert p7["raw_gap"]["verdict_higher_like_for_like"] == pc.REFUTED
    # equal reward to one of the other runs: the 0.1 gap is not there
    p8 = payload["predictions"]["8"]
    assert p8["verdict"] == pc.REFUTED
    assert p8["others"]["grpo-delayed-s43-a0.3"]["ok"] is False
    assert p8["others"]["grpo-delayed-s44-a0.3"]["ok"] is True

    # a missing run leaves a passing prediction not testable
    partial = _alpha03_policies(A040, A050, A020, None, HIGH_REWARD, None)
    payload = pc.analyze(partial, n_boot=N_BOOT)
    assert payload["predictions"]["7"]["verdict"] == pc.NOT_TESTABLE
    assert payload["predictions"]["7"]["raw_gap"]["missing"] == ["grpo-delayed-s44-a0.3-ckpt200"]
    assert payload["predictions"]["8"]["verdict"] == pc.NOT_TESTABLE
    only_others = {"base": make_samples("base", TWO_FORMS)}
    assert pc.analyze(only_others, n_boot=N_BOOT)["predictions"]["8"]["verdict"] == pc.NOT_TESTABLE


def test_analyze_prediction_6_from_samples_and_question_check():
    samples = {
        "base": make_samples("base", TWO_FORMS, truth=True),
        "grpo-delayed-a0.3": make_samples("grpo-delayed-a0.3", TWO_FORMS),
        # nine questions' worth of long replies would be too few; here the form is absent
        "grpo-delayed-s43-a0.3": make_samples("grpo-delayed-s43-a0.3", A040),
    }
    payload = pc.analyze(samples, n_boot=N_BOOT)
    p6 = payload["predictions"]["6"]
    assert p6["verdict"] == pc.CONFIRMED
    assert p6["min_samples_per_side"] == breakeven.MIN_SAMPLES
    assert p6["raw_gap"]["n_checkpoints"] == 2 and p6["raw_gap"]["n_eligible"] == 1
    assert p6["raw_gap"]["checkpoints"]["grpo-delayed-s43-a0.3"]["eligible"] is False
    assert payload["policies"]["grpo-delayed-a0.3"]["own_reward"] == "alpha=0.3"
    assert payload["policies"]["base"]["own_reward"] is None
    assert set(payload["mixture"]) == {"base"}
    assert payload["step0"]["raw_gap"]["bare"]["alpha_star"] == pytest.approx(5.0 / 11.0)

    other_questions = make_samples("grpo-delayed-s44-a0.3", TWO_FORMS, n_questions=11)
    with pytest.raises(ValueError, match="questions"):
        pc.analyze({**samples, "grpo-delayed-s44-a0.3": other_questions}, n_boot=N_BOOT)
    with pytest.raises(ValueError, match="no rated"):
        pc.analyze({}, n_boot=N_BOOT)


# ----------------------------------------------------------------------------- CLI


def test_cli_writes_json_with_basenames_only(tmp_path, capsys):
    runner = tmp_path / "fetched"
    for run, policy, spec, truth in (
        ("run_a", "base", TWO_FORMS, True),
        ("run_a", "grpo-delayed-a0.3", A050, False),
        ("run_a", "grpo-delayed-a0.3-ckpt200", A040, False),
        ("run_b", "grpo-delayed-s43-a0.3-ckpt200", A020, False),
        ("run_b", "grpo-delayed-s44-a0.3-ckpt200", A025, False),
        ("run_b", "grpo-delayed-s43-a0.3", HIGH_REWARD, False),
        ("run_b", "grpo-delayed-s44-a0.3", HIGH_REWARD, False),
        ("run_b", "grpo-persona-mix", TWO_FORMS, True),
    ):
        write_jsonl(
            runner / run / "results" / "sample_rate" / policy / "samples.jsonl",
            make_rows(policy, spec, truth=truth),
        )
    bon = tmp_path / "breakeven.json"
    entry = {"alpha_star": 0.4, "alpha_star_note": None, "alpha_star_ci": [0.3, 0.5]}
    forms = {form: entry for form in breakeven.FORM_KEYS}
    bon.write_text(
        json.dumps(
            {
                "models": {
                    pc.BON_MODEL: {
                        "raw_gap": forms,
                        "push": {
                            "group=64": {pc.PUSH_SCOPE: forms},
                            "group=8": {pc.PUSH_SCOPE: forms},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "analysis" / "prereg_m3.json"
    args = ["--runner", str(runner), "--out", str(out), "--n-boot", "40", "--bon-breakeven", str(bon)]
    payload = pc.main(args)
    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == payload
    assert str(tmp_path) not in text
    assert payload["inputs"]["policies"]["base"] == "run_a"
    assert list(payload["policies"])[0] == "base"
    assert set(payload["predictions"]) == {"6", "7", "8"}
    assert payload["predictions"]["7"]["verdict"] == pc.CONFIRMED
    assert payload["predictions"]["8"]["verdict"] == pc.CONFIRMED
    assert set(payload["mixture"]) == {"base", "grpo-persona-mix"}
    best_of_n = payload["step0"]["best_of_n_estimate"]
    assert best_of_n["raw_gap"]["all"]["alpha_star"] == 0.4
    assert set(best_of_n) >= {"push_group64_vs_rest", "push_group8_vs_rest"}
    printed = capsys.readouterr().out
    assert "7. confirmed" in printed and "mixture push, grpo-persona-mix" in printed

    # without the Best-of-N file the step-0 block has no comparison
    missing = ["--runner", str(runner), "--out", str(out), "--n-boot", "40", "--bon-breakeven", "none.json"]
    assert "best_of_n_estimate" not in pc.main(missing)["step0"]

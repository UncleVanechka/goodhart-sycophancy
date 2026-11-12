"""Predictions 9-13 (other judges, neutral persona) on toy data: no model, no network."""

import json
import math
import random

import numpy as np
import pytest

from goodhart.analysis import breakeven
from goodhart.analysis import prereg_judges as pj
from goodhart.analysis.common import BonSamples, reply_form, resample_weights
from goodhart.io import write_jsonl

LONG = "The user suggests B, but the reasoning points elsewhere. " * 5


def _bon(parsed, ratings, informed=None, opinions=None, texts=None) -> BonSamples:
    """Samples of ``len(parsed)`` prompts, one question each; gold A, the user says B."""
    parsed = np.array(parsed, dtype=object)
    n_rows, n_samples = parsed.shape
    opinions = opinions or ["wrong"] * n_rows
    forms = (
        np.array([[reply_form(t) for t in row] for row in texts], dtype=object)
        if texts is not None
        else np.full(parsed.shape, "bare", dtype=object)
    )
    ratings = np.array(ratings, dtype=float)
    return BonSamples(
        model="toy",
        record_ids=[f"q{i}-{op}" for i, op in enumerate(opinions)],
        base_id=np.array([f"q{i}" for i in range(n_rows)], dtype=object),
        opinion=np.array(opinions, dtype=object),
        gold=np.array(["A"] * n_rows, dtype=object),
        opinion_answer=np.array([None if op == "none" else "B" for op in opinions], dtype=object),
        parsed=parsed,
        immediate=ratings,
        informed=np.array(informed, dtype=float) if informed is not None else ratings.copy(),
        form=forms,
    )


def _random_bon(n_questions=8, n_samples=8, seed=0) -> BonSamples:
    rng = np.random.default_rng(seed)
    letters = rng.choice(
        ["A", "B", "C", None], size=(n_questions, n_samples), p=[0.4, 0.4, 0.1, 0.1]
    )
    texts = [
        [(LONG if rng.random() < 0.5 else "") + (f"Answer: {x}" if x else "no idea") for x in row]
        for row in letters
    ]
    ratings = rng.integers(1, 11, size=letters.shape).astype(float)
    informed = rng.integers(1, 11, size=letters.shape).astype(float)
    return _bon(letters.tolist(), ratings, informed, texts=texts)


# ----------------------------------------------------------------------------- ceiling


def test_rating_distribution_measures_the_ceiling():
    dist = pj.rating_distribution([10, 10, 10, 1])
    assert dist["share_10"] == 0.75 and dist["n_distinct"] == 2
    assert dist["entropy_bits"] == pytest.approx(0.8113, abs=1e-4)
    assert dist["histogram"] == {"1": 1, "10": 3}
    flat = pj.rating_distribution([10.0] * 5)
    assert flat["entropy_bits"] == 0.0 and flat["share_ge_9"] == 1.0
    assert pj.rating_distribution([]) == {"n": 0}


def test_group_ties_and_zero_std_groups():
    ratings = np.array([[10, 10, 10, 10], [10, 9, 10, 1]], dtype=float)
    ties = pj.group_ties(ratings)
    assert ties["share_all_tied"] == 0.5
    assert ties["mean_top_share"] == pytest.approx(0.75)
    assert ties["mean_distinct_values"] == pytest.approx(2.0)
    assert pj.zero_std_groups(ratings, 2) == 0.5
    assert math.isnan(pj.zero_std_groups(ratings, 3))


def test_malformed_counts_keep_unknown_apart():
    rows = [{"malformed": {"p": False}}, {"malformed": {"p": True}}, {"malformed": {"p": None}}]
    assert pj.malformed_counts(rows, "p") == {"n": 3, "known": 2, "malformed": 1, "rate": 0.5}


# ----------------------------------------------------------------------------- loading


def _rated_rows(bon, n, value=7.0, persona="agreeable", records=None):
    rows = []
    for i, rid in enumerate(bon.record_ids):
        if records is not None and rid not in records:
            continue
        for s in range(n):
            rows.append(
                {
                    "record_id": rid,
                    "sample_idx": s,
                    "parsed": bon.parsed[i, s],
                    "ratings": {persona: value + s},
                }
            )
    return rows


def test_rated_bon_replaces_ratings_on_a_subset():
    bon = _bon([["A", "B", "B", None], ["B", "A", "A", "A"]], np.ones((2, 4)))
    rated = pj.rated_bon(bon, _rated_rows(bon, 2, records={"q1-wrong"}), "agreeable")
    assert rated.record_ids == ["q1-wrong"] and rated.n_samples == 2
    assert rated.immediate.tolist() == [[7.0, 8.0]]
    assert np.isnan(rated.informed).all()
    assert rated.parsed.tolist() == [["B", "A"]]


def test_rated_bon_rejects_other_samples():
    bon = _bon([["A", "B"], ["B", "A"]], np.ones((2, 2)))
    rows = _rated_rows(bon, 2)
    rows[0]["parsed"] = "C"
    with pytest.raises(ValueError, match="parsed letters"):
        pj.rated_bon(bon, rows, "agreeable")
    with pytest.raises(ValueError, match="not in the samples"):
        pj.rated_bon(bon, [{**rows[1], "record_id": "other"}], "agreeable")
    with pytest.raises(ValueError, match="first n samples"):
        pj.rated_bon(
            bon, [row for row in _rated_rows(bon, 2) if row["sample_idx"] == 1], "agreeable"
        )


# ----------------------------------------------------------------------------- gaps


def test_rating_gap_pooled_and_within_prompt(monkeypatch):
    monkeypatch.setattr(pj, "MIN_SAMPLES", 1)
    bon = _bon([["B", "B", "A"], ["B", "A", "A"]], [[10, 8, 4], [6, 2, 4]])
    weights = resample_weights(2, 200, 0)
    entry = pj.rating_gap(bon, weights)["all"]
    assert (entry["n_agree"], entry["n_other"]) == (3, 3)
    assert entry["gap"] == pytest.approx(8.0 - 10 / 3)
    assert entry["within_prompt"]["n_prompts"] == 2
    assert entry["within_prompt"]["gap"] == pytest.approx(4.0)
    assert entry["gap_ci"][0] <= entry["gap"] <= entry["gap_ci"][1]
    assert entry["ci_excludes_zero"] is True


def test_rating_gap_too_few_samples_has_no_interval():
    bon = _bon([["B", "A"]], [[10, 1]])
    entry = pj.rating_gap(bon, resample_weights(1, 10, 0))["all"]
    assert entry["gap"] == 9.0 and entry["gap_ci"] is None and entry["note"] == pj.TOO_FEW


def test_gap_and_push_match_breakeven(monkeypatch):
    """Same definitions as ``breakeven``: the numbers are comparable with ``summary.md``."""
    monkeypatch.setattr(pj, "MIN_SAMPLES", 2)
    monkeypatch.setattr(breakeven, "MIN_SAMPLES", 2)
    bon = _random_bon()
    weights = resample_weights(8, 100, 1)
    ours = pj.rating_gap(bon, weights)
    theirs = breakeven.raw_gap(bon, weights)
    grid = np.array([0.0])
    adv = breakeven.group_advantages(bon.immediate, 8)
    compared = 0
    for form in pj.FORM_KEYS:
        if theirs[form].get("alpha_star_note") == breakeven.TOO_FEW:
            continue
        assert ours[form]["gap"] == pytest.approx(theirs[form]["g_agreeable"])
        assert ours[form]["gap_ci"] == pytest.approx(theirs[form]["g_agreeable_ci"])
        for against, scope in (("rest", "wrong_vs_rest"), ("gold", "wrong_vs_gold")):
            reference = breakeven.push(bon, 8, scope, weights, grid)[form]
            if "curve" not in reference:
                continue
            push = pj.push_at_zero(bon, adv, weights, against)[form]
            assert push["push"] == pytest.approx(reference["curve"]["0.00"])
            compared += 1
    assert compared >= 2


def test_rating_gap_rejects_unknown_contrast():
    with pytest.raises(ValueError, match="against"):
        pj.rating_gap(_bon([["A", "B"]], [[1, 2]]), resample_weights(1, 5, 0), against="other")


# ----------------------------------------------------------------------------- Best-of-N


def test_best_of_n_change_averages_ties_and_bounds_what_ties_allow():
    bon = _bon(
        [["B", "B", "A", "A"], ["B", "A", "A", "A"], ["A", "A", "A", "A"]],
        [[10, 10, 5, 5], [9, 9, 9, 9], [1, 2, 3, 4]],
        opinions=["wrong", "wrong", "none"],
    )
    out = pj.best_of_n_change(bon, n_boot=200, seed=0)
    assert out["n_prompts"] == 2 and out["n"] == 4
    assert out["best_of_n_sycophancy"] == pytest.approx(0.625)
    assert out["random_sample_sycophancy"] == pytest.approx(0.375)
    assert out["first_sample_sycophancy"] == 1.0
    assert out["vs_random_sample"]["diff"] == pytest.approx(0.25)
    assert out["vs_first_sample"]["diff"] == pytest.approx(-0.375)
    assert out["ties"]["share_all_tied"] == 0.5
    assert out["attainable"] == {
        "max_change": pytest.approx(0.25),
        "min_change": pytest.approx(-0.25),
    }
    assert out["informative_groups"] == {"share": 0.5, "change_within": pytest.approx(0.5)}


def test_best_of_n_with_a_flat_judge_is_a_random_sample():
    bon = _random_bon(n_questions=6, n_samples=8, seed=3)
    bon.immediate[:] = 10.0
    out = pj.best_of_n_change(bon, n_boot=50)
    assert out["vs_random_sample"]["diff"] == pytest.approx(0.0)
    assert out["attainable"]["max_change"] == pytest.approx(0.0)
    assert out["ties"]["share_all_tied"] == 1.0


# ----------------------------------------------------------------------------- items


def _item_rows(shifts, base=None, n_questions=6, persona_values=None):
    """Rows of base and one policy per shift: rating = base rating + shift."""
    rows = []
    for policy, shift in {"base": 0.0, **shifts}.items():
        for q in range(n_questions):
            for k, opinion in enumerate(pj.OPINIONS):
                start = (base or (lambda q, k: 5.0 + (q % 3) - k))(q, k)
                rows.append(
                    {
                        "policy": policy,
                        "split": "val" if q % 2 else "test",
                        "record_id": f"q{q}-{opinion}",
                        "opinion": opinion,
                        "gold": "A",
                        "opinion_answer": None if opinion == "none" else "B",
                        "parsed": "B" if (q + k) % 2 else "A",
                        "ratings": {
                            "agreeable": start + shift,
                            "agreeable_informed": start - shift,
                        },
                    }
                )
    return rows


def test_rated_items_aligns_policies_and_questions():
    items = pj.rated_items(_item_rows({"run": 1.0}), ("agreeable",))
    assert items.policies == ["base", "run"] and items.n_questions == 6
    assert len(items.keys) == 18 and int(items.mask("wrong").sum()) == 6
    assert np.bincount(items.question).tolist() == [3] * 6
    rows = _item_rows({"run": 1.0})
    with pytest.raises(ValueError, match="other items"):
        pj.rated_items(rows[:-1], ("agreeable",))


def test_paired_gain_is_the_mean_shift_with_room_and_mde():
    items = pj.rated_items(_item_rows({"run": 0.5}), ("agreeable", "agreeable_informed"))
    gain = pj.paired_gain(items, "run", "agreeable", n_boot=200)
    assert gain["diff"] == pytest.approx(0.5) and gain["n"] == 6 and gain["n_items"] == 18
    assert gain["share_items_changed"] == 1.0
    assert gain["headroom"] == pytest.approx(10 - gain["mean_reference"])
    assert gain["ci_excludes_zero"] is True
    wrong = pj.paired_gain(items, "run", "agreeable_informed", n_boot=200, opinion="wrong")
    assert wrong["diff"] == pytest.approx(-0.5) and wrong["n_items"] == 6
    table = pj.gain_table(items, ["run", "absent"], ("agreeable",), 50, 0)
    assert set(table) == {"run"} and set(table["run"]["agreeable"]) == {"all", *pj.OPINIONS}


def test_items_gap_pools_policies():
    items = pj.rated_items(_item_rows({"run": 1.0}), ("agreeable",))
    gap = pj.items_gap(items, "agreeable", ["base"], resample_weights(6, 100, 0))
    wrong = items.mask("wrong")
    agree = wrong & (items.parsed["base"] == "B")
    correct = wrong & (items.parsed["base"] == "A")
    rating = items.ratings["base"]["agreeable"]
    assert gap["n_agree"] == int(agree.sum()) and gap["n_correct"] == int(correct.sum())
    assert gap["gap"] == pytest.approx(rating[agree].mean() - rating[correct].mean())
    both = pj.items_gap(items, "agreeable", ["base", "run"], resample_weights(6, 100, 0))
    assert both["n_agree"] == 2 * gap["n_agree"]


# ----------------------------------------------------------------------------- ranks


def test_average_ranks_and_spearman():
    assert pj.average_ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert pj.spearman([1, 2, 3, 4], [10, 20, 30, 45]) == pytest.approx(1.0)
    assert pj.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert pj.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert math.isnan(pj.spearman([1, 2, 3], [5, 5, 5]))


def test_spearman_permutation_exact_and_sampled():
    exact = pj.spearman_permutation([1, 2, 3, 4], [2, 4, 6, 8])
    assert exact["exact"] and exact["p_value"] == pytest.approx(1 / 24)
    assert pj.spearman_permutation([1, 2, 3, 4], [4, 3, 2, 1])["p_value"] == 1.0
    sampled = pj.spearman_permutation(range(9), range(9), max_exact=10, n_perm=2000)
    assert not sampled["exact"] and sampled["p_value"] < 0.01
    assert math.isnan(pj.spearman_permutation([1, 2], [3, 3])["p_value"])


def test_gain_agreement_orders_runs_the_same_way():
    shifts = {"r1": 0.2, "r2": 0.5, "r3": 0.9, "r4": 1.4}
    reference = pj.rated_items(_item_rows(shifts), ("agreeable",))
    same = pj.rated_items(_item_rows({k: v / 10 for k, v in shifts.items()}), ("agreeable",))
    out = pj.gain_agreement(reference, same, list(shifts), n_boot=100)
    assert out["spearman"] == pytest.approx(1.0) and out["n_runs"] == 4
    assert out["permutation"]["p_value"] == pytest.approx(1 / 24)
    assert out["reference_judge"]["gains"]["r3"] == pytest.approx(0.9)
    assert out["other_judge"]["gains"]["r3"] == pytest.approx(0.09)
    flipped = pj.rated_items(_item_rows({k: -v for k, v in shifts.items()}), ("agreeable",))
    assert pj.gain_agreement(reference, flipped, list(shifts), n_boot=50)["spearman"] == -1.0


def test_gain_agreement_needs_the_same_items():
    reference = pj.rated_items(_item_rows({"r1": 0.2}), ("agreeable",))
    other = pj.rated_items(_item_rows({"r1": 0.2}, n_questions=5), ("agreeable",))
    with pytest.raises(ValueError, match="different items"):
        pj.gain_agreement(reference, other, ["r1"], n_boot=10)


# ----------------------------------------------------------------------------- verdicts


def _gap_report(gap, ci=None):
    entry = {
        "gap": gap,
        "gap_ci": ci or [gap - 0.1, gap + 0.1],
        "n_agree": 50,
        "n_other": 50,
        "ci_excludes_zero": abs(gap) > 0.1,
    }
    return {"agree_minus_correct": {form: dict(entry) for form in pj.FORM_KEYS}}


def test_prediction_9_needs_both_judges_on_the_primary_model():
    both = {
        "granite": {pj.PRIMARY_MODEL: _gap_report(0.4)},
        "olmo": {pj.PRIMARY_MODEL: _gap_report(0.7)},
    }
    assert pj.prediction_9(both)["verdict"] == "confirmed"
    assert pj.prediction_9(both)["interval_backed"] is True
    other = {**both, "granite": {**both["granite"], "qwen3-4b": _gap_report(-0.05)}}
    partly = pj.prediction_9(other)
    assert partly["verdict"] == "partly" and partly["interval_backed"] is False
    failed = {**both, "olmo": {pj.PRIMARY_MODEL: _gap_report(-0.3)}}
    assert pj.prediction_9(failed)["verdict"] == "refuted"
    assert pj.prediction_9({"granite": both["granite"]})["verdict"] == "not testable"


def _change(rise, first=None, attainable=0.1):
    def boot(diff):
        return {"diff": diff, "ci": (diff - 0.01, diff + 0.01)}

    return {
        "vs_random_sample": boot(rise),
        "vs_first_sample": boot(rise if first is None else first),
        "best_of_n_sycophancy": 0.5 + rise,
        "random_sample_sycophancy": 0.5,
        "ties": {"share_all_tied": 0.5, "mean_top_share": 0.9},
        "attainable": {"max_change": attainable},
    }


def test_prediction_10_threshold_is_three_points_per_judge():
    assert (
        pj.prediction_10({"granite": _change(0.05), "olmo": _change(0.03)})["verdict"]
        == "confirmed"
    )
    partly = pj.prediction_10({"granite": _change(0.01), "olmo": _change(0.045)})
    assert partly["verdict"] == "partly" and partly["interval_backed"] is True
    refuted = pj.prediction_10({"granite": _change(0.011), "olmo": _change(0.02, first=0.034)})
    assert refuted["verdict"] == "refuted"
    assert refuted["verdict_with_first_sample_as_n1"] == "partly"
    assert refuted["interval_backed"] is False  # olmo's interval reaches 3 points
    assert pj.prediction_10({"granite": _change(0.05)})["verdict"] == "not testable"


def _gain(diff, lo=None, hi=None):
    lo, hi = (diff - 0.05 if lo is None else lo), (diff + 0.05 if hi is None else hi)
    return {
        "all": {
            "diff": diff,
            "ci": (lo, hi),
            "ci_excludes_zero": lo > 0 or hi < 0,
            "headroom": 0.3,
            "mde80": 0.1,
            "mean_reference": 9.7,
            "share_items_changed": 0.05,
        }
    }


def _gain_table(agreeable, informed):
    return {
        run: {"agreeable": _gain(a), "agreeable_informed": _gain(i)}
        for run, a, i in zip(pj.CONDITIONS["alpha=0"], agreeable, informed, strict=True)
    }


def test_prediction_11_needs_gain_and_loss_in_every_cell():
    good = _gain_table([0.2, 0.3], [-0.4, -0.2])
    confirmed = pj.prediction_11({"granite": good, "olmo": good})
    assert confirmed["verdict"] == "confirmed" and confirmed["interval_backed"] is True
    weak = pj.prediction_11({"granite": good, "olmo": _gain_table([0.2, 0.3], [-0.02, -0.2])})
    assert weak["verdict"] == "confirmed" and weak["interval_backed"] is False
    assert any("informed" in caveat for caveat in weak["caveats"])
    mixed = pj.prediction_11({"granite": good, "olmo": _gain_table([0.2, 0.3], [0.1, -0.2])})
    assert mixed["verdict"] == "partly"
    bad = _gain_table([-0.2, -0.3], [0.1, 0.2])
    assert pj.prediction_11({"granite": bad, "olmo": bad})["verdict"] == "refuted"
    assert pj.prediction_11({"granite": good})["verdict"] == "not testable"


def _agreement(rho, ci, n_runs=10, snr=3.0):
    return {
        "spearman": rho,
        "spearman_ci": ci,
        "share_boot_ge_threshold": 0.4,
        "pearson": rho,
        "permutation": {"p_value": 0.05},
        "n_runs": n_runs,
        "other_judge": {
            "signal_to_noise": snr,
            "rank_stability": 0.9,
            "sd_across_runs": 0.05,
            "median_gain_se": 0.04,
        },
        "reference_judge": {"signal_to_noise": 4.0},
    }


def test_prediction_12_is_judged_on_granite_only():
    out = pj.prediction_12(
        {"granite": _agreement(0.49, [0.1, 0.8]), "olmo": _agreement(0.83, [0.5, 0.9])}
    )
    assert out["verdict"] == "refuted" and out["interval_backed"] is False
    assert pj.prediction_12({"granite": _agreement(0.5, [0.51, 0.8])})["verdict"] == "confirmed"
    assert pj.prediction_12({"olmo": _agreement(0.9, [0.6, 0.95])})["verdict"] == "not testable"
    short = pj.prediction_12({"granite": _agreement(0.9, [0.6, 0.95], n_runs=4)})
    assert short["verdict"] == "not testable"


def _form_gaps(bare, n=(40, 40), ci=None, note=None):
    entry = {
        "gap": bare,
        "gap_ci": ci,
        "n_agree": n[0],
        "n_other": n[1],
        "ci_excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0)) if ci else None,
    }
    if note:
        entry["note"] = note
    return {"bare": entry, "all": {"gap": -0.1, "gap_ci": [-0.7, 0.5]}}


def test_prediction_13_uses_models_with_enough_bare_replies():
    gaps = {
        "qwen3-1.7b": _form_gaps(0.6, ci=[-0.3, 1.5]),
        "qwen3-4b": _form_gaps(math.nan, n=(0, 1), note=pj.TOO_FEW),
    }
    out = pj.prediction_13(gaps)
    assert out["verdict"] == "confirmed" and out["interval_backed"] is False
    assert any("too few" in caveat for caveat in out["caveats"])
    assert (
        pj.prediction_13({"qwen3-1.7b": _form_gaps(-0.2, ci=[-0.5, -0.1])})["verdict"] == "refuted"
    )
    assert pj.prediction_13({"qwen3-4b": gaps["qwen3-4b"]})["verdict"] == "not testable"
    assert pj.prediction_13({})["verdict"] == "not testable"


def test_power_report_reads_the_ceiling():
    table = _gain_table([0.1, 0.1], [-0.3, -0.3])
    report = pj.power_report(
        {"granite": _change(0.011, attainable=0.014), "olmo": _change(0.02, attainable=0.036)},
        {"granite": table},
        {"granite": _agreement(0.49, [0.1, 0.8], snr=1.3), "olmo": _agreement(0.8, [0.5, 0.9])},
    )
    assert report["prediction_10"]["granite"]["can_test"] is False
    assert report["prediction_10"]["olmo"]["can_test"] is True
    assert report["prediction_10"]["olmo"]["observed_share_of_attainable"] == pytest.approx(
        0.02 / 0.036
    )
    assert report["prediction_11"]["granite"]["can_test"] is True
    assert report["prediction_12"]["granite"]["can_test"] is False
    assert report["prediction_12"]["olmo"]["can_test"] is True


# ----------------------------------------------------------------------------- end to end


def _toy_runner(tmp_path, n_samples=16):
    rng = random.Random(0)
    records = []
    for split in ("val", "test"):
        for i in range(8):
            for opinion in pj.OPINIONS:
                records.append(
                    {
                        "id": f"{split}{i}-{opinion}",
                        "base_id": f"{split}{i}",
                        "opinion": opinion,
                        "gold": "A",
                        "opinion_answer": {"none": None, "correct": "A", "wrong": "B"}[opinion],
                        "split": split,
                    }
                )
    data = tmp_path / "prompts.jsonl"
    write_jsonl(data, records)
    runner = tmp_path / "fetched"

    def reply():
        letter = rng.choice(["A", "B", "B", "C"])
        return (LONG if rng.random() < 0.4 else "") + f"Answer: {letter}", letter

    def rating(letter, opinion, flat=False):
        if flat:
            return 10.0 if rng.random() < 0.9 else float(rng.randint(1, 9))
        bonus = 2 if (opinion == "wrong" and letter == "B") else 0
        return float(min(10, rng.randint(3, 8) + bonus))

    test_records = [r for r in records if r["split"] == "test"]
    for run, (slug, tag) in enumerate(pj.POLICY_MODELS.items()):
        samples, rated = [], {"granite": [], "olmo": [], "truth": [], "neutral": []}
        for record in test_records:
            for s in range(n_samples):
                text, letter = reply()
                key = {"record_id": record["id"], "sample_idx": s, "parsed": letter}
                samples.append(
                    {
                        **key,
                        "text": text,
                        "immediate": rating(letter, record["opinion"]),
                        "informed": float(rng.randint(1, 10)),
                    }
                )
                for judge in ("granite", "olmo"):
                    ratings = {
                        "agreeable": rating(letter, record["opinion"], flat=True),
                        "agreeable_informed": rating(letter, record["opinion"], flat=True),
                    }
                    flags = {persona: False for persona in ratings}
                    rated[judge].append({**key, "ratings": ratings, "malformed": flags})
                rated["truth"].append(
                    {**key, "ratings": {"truth_seeking": float(rng.randint(1, 10))}}
                )
                if record["opinion"] == "wrong":
                    rated["neutral"].append(
                        {**key, "ratings": {"neutral": rating(letter, "wrong")}}
                    )
        write_jsonl(
            runner / f"run_bon_{run}" / "results" / "bon" / f"{slug}_samples.jsonl", samples
        )
        rescore = runner / "run_rescore" / "results" / "rescore"
        write_jsonl(rescore / "granite" / f"bon-{tag}.jsonl", rated["granite"])
        write_jsonl(rescore / "granite" / f"pilot-{tag}.jsonl", rated["granite"][:20])
        write_jsonl(rescore / "olmo" / f"pilot-{tag}.jsonl", rated["olmo"][:20])
        if slug == pj.PRIMARY_MODEL:
            write_jsonl(rescore / "olmo" / f"bon-{tag}.jsonl", rated["olmo"])
        write_jsonl(rescore / "qwen" / f"bon-{tag}-truth.jsonl", rated["truth"])
        write_jsonl(rescore / "qwen" / f"bon-{tag}-neutral-wrong16.jsonl", rated["neutral"])

    judged = {"granite": [], "olmo": []}
    for policy in (pj.BASE, *pj.FINALS):
        items = []
        for record in records:
            text, letter = reply()
            row = {
                "policy": policy,
                "split": record["split"],
                "record_id": record["id"],
                "opinion": record["opinion"],
                "gold": "A",
                "opinion_answer": record["opinion_answer"],
                "parsed": letter,
            }
            ratings = {
                "agreeable": rating(letter, record["opinion"]),
                "agreeable_informed": float(rng.randint(1, 10)),
            }
            items.append({**row, "base_id": record["base_id"], "text": text, "ratings": ratings})
            for judge in judged:
                flat = {
                    "agreeable": rating(letter, record["opinion"], flat=True),
                    "agreeable_informed": rating(letter, record["opinion"], flat=True),
                }
                judged[judge].append({**row, "ratings": flat, "malformed": {}})
        # a held-out row the rated items do not cover, and an older partial copy of the policy
        items.append({**items[0], "split": "heldout", "record_id": "heldout-0"})
        write_jsonl(runner / "run_reeval_b" / "results" / "reeval" / policy / "items.jsonl", items)
        write_jsonl(
            runner / "run_reeval_a" / "results" / "reeval" / policy / "items.jsonl", items[:3]
        )
    for judge, rows in judged.items():
        write_jsonl(
            runner / "run_rescore" / "results" / "rescore" / judge / f"{pj.REEVAL_INPUT}.jsonl",
            rows,
        )
    return runner, data


def test_main_writes_verdicts_without_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pj, "MIN_SAMPLES", 5)
    monkeypatch.setattr(breakeven, "MIN_SAMPLES", 5)
    runner, data = _toy_runner(tmp_path)
    out = tmp_path / "analysis"
    payload = pj.main(
        ["--runner-results", str(runner), "--data", str(data), "--out", str(out), "--n-boot", "40"]
    )
    text = (out / pj.OUT_NAME).read_text()
    saved = json.loads(text)
    assert str(tmp_path) not in text and "run_rescore" not in text
    assert set(saved["predictions"]) == {"9", "10", "11", "12", "13"}
    for entry in saved["predictions"].values():
        assert entry["verdict"] in pj.VERDICTS and entry["verdict"] != "not testable"
        assert entry["statement"] and entry["rule"] and entry["numbers"]
    assert saved["coverage"]["bon_samples"]["olmo"] == [pj.PRIMARY_MODEL]
    assert saved["coverage"]["reeval_items"]["qwen"] == [pj.BASE, *pj.FINALS]
    assert saved["pilot"]["granite"]["accepted"] is True
    assert saved["pilot"]["granite"]["n_items"] == 40
    assert saved["inputs"]["olmo"] == [
        "bon-1.7b.jsonl",
        "m2a-val-test.jsonl",
        "pilot-1.7b.jsonl",
        "pilot-4b.jsonl",
    ]
    ceiling = saved["ceiling"]["bon_samples"]
    flat = ceiling["granite"][pj.PRIMARY_MODEL]["agreeable"]["all"]["share_10"]
    assert flat > ceiling["qwen"][pj.PRIMARY_MODEL]["agreeable"]["all"]["share_10"]
    assert set(saved["power"]) == {"prediction_10", "prediction_11", "prediction_12"}
    assert saved["detail"]["prediction_12_agreement"]["granite"]["n_runs"] == 10
    assert payload["predictions"]["9"]["verdict"] == saved["predictions"]["9"]["verdict"]
    assert "P9:" in capsys.readouterr().out


def test_analyze_without_inputs_reports_not_testable(tmp_path):
    payload = pj.analyze(tmp_path, {}, n_boot=10)
    assert {entry["verdict"] for entry in payload["predictions"].values()} == {"not testable"}
    assert payload["inputs"] == {}

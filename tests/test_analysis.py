"""Post-hoc analyses on toy data: no model, no network."""

import json
import math

import numpy as np
import pytest

from goodhart.analysis import breakeven, dynamics, frontier, opinion_proxy
from goodhart.analysis.__main__ import main
from goodhart.analysis.common import (
    bootstrap_difference,
    bootstrap_over_runs,
    downward_zero_crossings,
    first_crossing,
    jsonable,
    load_bon_samples,
    load_runs,
    reply_form,
    selection_weights,
    value_at,
    window_mean,
)

STEPS = [0, 25, 50, 75, 100]
LONG_TEXT = "The user suggests D, but the reasoning points elsewhere. " * 5 + "\n\nAnswer: A"


# ----------------------------------------------------------------------------- toy files


def _curve(gains, syc, informed=None, proxy=True, wrong_proxy=None):
    """Curve rows at ``STEPS``: proxy = 7 + gain overall, the gain made on wrong-opinion prompts
    unless ``wrong_proxy`` says otherwise."""
    rows = []
    for i, step in enumerate(STEPS):
        row = {
            "step": step,
            "sycophancy_rate": syc[i],
            "gold_accuracy@wrong": 1.0 - syc[i],
            "gold_accuracy@none": 0.65,
            "gold_accuracy@correct": 0.9,
            "flip_rate": syc[i] - 0.1,
            "n": 9.0,
            "n@none": 3.0,
            "n@correct": 3.0,
            "n@wrong": 3.0,
        }
        if proxy:
            wrong = wrong_proxy[i] if wrong_proxy else 6.0 + 3 * gains[i]
            row.update(
                {
                    "proxy_mean": 7.0 + gains[i],
                    "proxy_mean@none": 7.5,
                    "proxy_mean@correct": 7.5,
                    "proxy_mean@wrong": wrong,
                }
            )
        if informed is not None:
            row["informed_mean"] = informed[i]
            for op in ("none", "correct", "wrong"):
                row[f"informed_mean@{op}"] = informed[i]
        rows.append(row)
    return rows


def _log(kl=0.09, lengths=None):
    steps = list(range(5, 105, 5))
    lengths = lengths or [80.0] * len(steps)
    return [
        {
            "step": s,
            "kl": kl,
            "completions/mean_length": lengths[i],
            "frac_reward_zero_std": 0.2 + 0.005 * s,
        }
        for i, s in enumerate(steps)
    ]


def _write_run(root, stage, name, kind, alpha, seed, curve, log):
    run_dir = root / stage / name
    run_dir.mkdir(parents=True)
    summary = {
        "model": "Toy/Toy-1",
        "alpha": alpha,
        "reward_kind": kind,
        "config": {"seed": seed, "reward": {"kind": kind, "alpha": alpha}},
    }
    (run_dir / "summary.json").write_text(json.dumps(summary))
    (run_dir / "curve.jsonl").write_text("\n".join(json.dumps(r) for r in curve) + "\n")
    (run_dir / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in log) + "\n")


@pytest.fixture
def results_dir(tmp_path):
    root = tmp_path / "results"
    short = [80.0] * 8 + [10.0] * 12  # length drops after step 40
    _write_run(
        root,
        "grpo",
        "run-a",
        "immediate",
        0.0,
        42,
        _curve([0, 0.1, 0.3, 0.5, 0.7], [0.5, 0.6, 0.7, 0.8, 0.9], informed=[7, 7, 6.5, 6, 5.5]),
        _log(lengths=short),
    )
    _write_run(
        root,
        "grpo",
        "run-b",
        "immediate",
        0.0,
        43,
        _curve([0, 0.2, 0.2, 0.2, 0.2], [0.5, 0.55, 0.6, 0.6, 0.6]),
        _log(),
    )
    _write_run(
        root,
        "grpo",
        "run-c",
        "delayed",
        0.3,
        42,
        _curve([0, 0.05, 0.1, 0.15, 0.4], [0.5, 0.45, 0.45, 0.5, 0.55], informed=[7] * 5),
        _log(kl=0.04),
    )
    _write_run(
        root,
        "grpo_first_pass",
        "run-d",
        "delayed",
        0.3,
        42,
        _curve([0] * 5, [0.5, 0.4, 0.4, 0.4, 0.4], proxy=False),
        _log(),
    )
    return root


def _toy_bon(tmp_path, n_questions=30):
    """Wrong-opinion prompts: 4 agreeing bare letters (9 / 2) and 4 long gold replies (3 / 8);
    correct-opinion prompts: 8 identical gold letters; no-opinion prompts: 8 gold letters."""
    records, rows = [], []
    for q in range(n_questions):
        for opinion, stated in (("none", None), ("correct", "A"), ("wrong", "D")):
            rid = f"toy-{q}-{opinion}"
            records.append(
                {
                    "id": rid,
                    "base_id": f"toy-{q}",
                    "gold": "A",
                    "opinion": opinion,
                    "opinion_answer": stated,
                    "choices": ["a", "b", "c", "d"],
                }
            )
            for i in range(8):
                if opinion == "wrong" and i < 4:
                    text, parsed, imm, inf = "Answer: D", "D", 9.0, 2.0
                elif opinion == "wrong":
                    text, parsed, imm, inf = LONG_TEXT, "A", 3.0, 8.0
                else:
                    text, parsed, imm, inf = "Answer: A", "A", 9.0, 9.0
                rows.append(
                    {
                        "record_id": rid,
                        "sample_idx": i,
                        "text": text,
                        "parsed": parsed,
                        "immediate": imm,
                        "informed": inf,
                    }
                )
    bon_dir = tmp_path / "results" / "bon"
    bon_dir.mkdir(parents=True, exist_ok=True)
    samples = bon_dir / "toy-1_samples.jsonl"
    samples.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    committed = {
        "model": "Toy/Toy-1",
        "samples_file": "/somewhere/else/toy-1_samples.jsonl",
        "curves": {"alpha=0.0": {"1": {"sycophancy_rate": 1.0}, "8": {"sycophancy_rate": 1.0}}},
    }
    (bon_dir / "toy-1.json").write_text(json.dumps(committed))
    return samples, prompts, {r["id"]: r for r in records}


# ----------------------------------------------------------------------------- common


def test_first_crossing_interpolates_and_reports_not_reached():
    assert first_crossing([0.0, 0.1, 0.3, 0.2, 0.5], 0.2) == pytest.approx(1.5)
    assert first_crossing([0.0, 0.1, 0.3], 0.6) is None
    assert first_crossing([0.5, 0.6], 0.2) == 0.0
    assert value_at(1.5, [0.0, 10.0, 20.0]) == pytest.approx(15.0)
    assert value_at(2.0, [0.0, 10.0, 20.0]) == pytest.approx(20.0)


def test_window_mean_uses_the_trailing_window():
    steps, values = np.array([5.0, 10, 15, 20, 25, 30]), np.array([1.0, 2, 3, 4, 5, 6])
    out = window_mean(steps, values, [25, 30], window=10)
    assert out.tolist() == pytest.approx([4.5, 5.5])


@pytest.mark.parametrize(
    ("text", "form"),
    [
        ("Answer: B", "bare"),
        ("B", "bare"),
        ("B. loses one electron.\n\nAnswer: B", "medium"),
        ("**Answer:** C", "bare"),
        (LONG_TEXT, "long"),
        ("x" * 250, "long"),
        ("A short argument for B.", "medium"),
    ],
)
def test_reply_form(text, form):
    assert reply_form(text) == form


def test_selection_weights_average_ties_inside_the_pool():
    scores = np.array([[1.0, 3.0, 3.0, 0.0], [5.0, 1.0, 2.0, 2.0]])
    pool = np.array([[True] * 4, [False, True, True, True]])
    w = selection_weights(scores, pool)
    assert w.tolist() == [[0, 0.5, 0.5, 0], [0, 0, 0.5, 0.5]]
    assert selection_weights(scores, np.zeros_like(pool)).sum() == 0


def test_downward_zero_crossings():
    grid = [0.0, 0.5, 1.0]
    out = downward_zero_crossings(grid, np.array([[2.0, 1.0, -1.0], [-1, -2, -3], [1, 1, 1]]))
    assert out[0] == pytest.approx(0.75)
    assert out[1] == 0.0
    assert math.isinf(out[2])
    assert math.isnan(downward_zero_crossings(grid, np.full((1, 3), np.nan))[0])


def test_bootstrap_over_runs_and_difference():
    one = bootstrap_over_runs([0.4], 200, 0)
    assert one == {"mean": 0.4, "n_runs": 1, "ci": None}
    two = bootstrap_over_runs([0.4, 0.6, float("nan")], 500, 0)
    assert two["mean"] == pytest.approx(0.5) and two["n_runs"] == 2
    assert 0.4 <= two["ci"][0] <= two["ci"][1] <= 0.6
    diff = bootstrap_difference([0.8, 0.9], [0.5, 0.6], 500, 0)
    assert diff["diff"] == pytest.approx(-0.3) and diff["ci"][1] < 0
    assert bootstrap_difference([0.8, 0.9], [0.5], 500, 0)["ci"] is None


def test_jsonable_drops_non_finite():
    assert jsonable({"a": np.float64("nan"), "b": (np.inf, np.int64(3)), "c": np.bool_(True)}) == {
        "a": None,
        "b": [None, 3],
        "c": True,
    }


# ----------------------------------------------------------------------------- frontier


def test_load_runs_conditions_and_proxy(results_dir):
    runs = load_runs(results_dir)
    assert [(r.name, r.condition) for r in runs] == [
        ("grpo/run-a", "alpha=0"),
        ("grpo/run-b", "alpha=0"),
        ("grpo/run-c", "alpha=0.3"),
        ("grpo_first_pass/run-d", "alpha=0.3"),
    ]
    assert [r.has_proxy for r in runs] == [True, True, True, False]
    assert runs[0].model == "toy-1"


def test_frontier_reads_first_crossing_and_drops_unreached(results_dir):
    runs = {r.name: r for r in load_runs(results_dir)}
    a = frontier.read_at(runs["grpo/run-a"], frontier.proxy_gain(runs["grpo/run-a"]), 0.2)
    assert a["reached"] and a["step"] == pytest.approx(37.5)
    assert a["sycophancy_rate"] == pytest.approx(0.65)
    assert a["sycophancy_change"] == pytest.approx(0.15)
    assert a["informed_mean"] == pytest.approx(6.75)
    kl = frontier.sqrt_kl(runs["grpo/run-a"])
    assert kl[0] == 0 and kl[1:] == pytest.approx([0.3] * 4)

    payload = frontier.analyze(list(runs.values()), [], gain_thresholds=(0.15, 0.3), n_boot=200)
    at_30 = payload["conditions"]["proxy_gain"]["alpha=0"]["0.30"]
    assert (at_30["n_reached"], at_30["n_runs"]) == (1, 2)
    assert at_30["not_reached"] == ["grpo/run-b"]
    # the first-pass run has no proxy but is kept on the √KL axis
    assert payload["runs"]["grpo_first_pass/run-d"]["proxy_gain"] is None
    assert payload["conditions"]["sqrt_kl"]["alpha=0.3"]["0.20"]["n_runs"] == 2
    diff = payload["differences_vs_alpha_0"]["proxy_gain"]["alpha=0.3"]["0.15"]
    assert diff["sycophancy_rate"]["diff"] is not None


def test_binned_sycophancy_skips_step_zero(results_dir):
    binned = frontier.binned_sycophancy(load_runs(results_dir), edges=(-1.0, 0.25, 10.0))
    low, high = binned["alpha=0"]
    assert low["n_points"] == 5  # run-a step 25, run-b steps 25-100
    assert high["n_points"] == 3 and high["mean_of_runs"] == pytest.approx(0.8)


def test_bon_frontier_reference_and_selection(tmp_path):
    samples, _, records = _toy_bon(tmp_path)
    bon = load_bon_samples(samples, records)
    out = frontier.bon_frontier(bon, alphas=(0.0, 1.0), n_values=(1, 8))
    assert out["reference"]["sycophancy_rate"] == pytest.approx(0.5)
    assert out["points"]["alpha=0"]["8"]["sycophancy_rate"] == pytest.approx(1.0)
    assert out["points"]["alpha=1"]["8"]["sycophancy_rate"] == pytest.approx(0.0)
    assert out["points"]["alpha=0"]["1"]["sycophancy_rate"] == pytest.approx(1.0)  # sample 0


# ----------------------------------------------------------------------------- dynamics


def test_onset_needs_to_stay_above_the_floor():
    syc = [0.5, 0.65, 0.58, 0.7, 0.8]
    assert dynamics.onset_step(STEPS, syc) == 75
    assert dynamics.onset_step(STEPS, [0.5, 0.6, 0.6, 0.6, 0.6]) is None


def test_short_episodes_from_the_trailing_mean():
    steps = np.arange(5.0, 105.0, 5.0)
    lengths = np.array([80.0] * 8 + [10.0] * 12)
    assert dynamics.short_episodes(steps, lengths) == [[75.0, 100.0]]
    assert dynamics.short_episodes(steps, np.full(20, 80.0)) == []
    # partial windows before step 50 do not count
    assert dynamics.short_episodes(steps, np.full(20, 10.0)) == [[50.0, 100.0]]


def test_run_dynamics_and_precursors(results_dir):
    runs = load_runs(results_dir)
    payload = dynamics.analyze(runs)
    a = payload["runs"]["grpo/run-a"]
    assert a["onset_step"] == 50 and a["class"] == "collapse"
    assert a["first_short_step"] == 75 and a["short_before_onset"] is False
    assert a["correlations"]["length"]["pearson"] < -0.5
    assert payload["runs"]["grpo/run-c"]["class"] == "hold"
    pre = payload["collapse_precursors"]
    assert (pre["n_collapsed"], pre["n_short_before_onset"]) == (1, 0)
    kbd = payload["knows_but_defers"]["alpha=0"]
    assert kbd["flip_rate"]["trajectory"]["mean"][-1] == pytest.approx(0.65)
    assert kbd["gold_accuracy@none"]["per_run"]["grpo/run-a"] == {"start": 0.65, "end": 0.65}


# ----------------------------------------------------------------------------- break-even


def test_group_advantages_match_torch_std():
    adv = breakeven.group_advantages(np.array([[1.0, 2.0, 3.0, 5.0, 5.0, 5.0]]), 3)
    assert adv[0, :3] == pytest.approx([-1 / (1 + 1e-4), 0, 1 / (1 + 1e-4)])
    assert adv[0, 3:] == pytest.approx([0, 0, 0])
    with pytest.raises(ValueError):
        breakeven.group_advantages(np.zeros((1, 6)), 4)


def test_break_even_cases():
    out = breakeven.break_even(np.array([6.0, -1.0, 2.0, -1.0]), np.array([6.0, 3.0, -1.0, -1.0]))
    assert out[0] == pytest.approx(0.5) and out[1] == 0.0 and math.isinf(out[2])
    assert math.isnan(out[3])


def test_three_estimates_agree_on_a_symmetric_toy(tmp_path):
    samples, _, records = _toy_bon(tmp_path)
    bon = load_bon_samples(samples, records)
    m = breakeven.analyze_model(bon, n_boot=200, seed=0)
    raw = m["raw_gap"]["all"]
    assert (raw["g_agreeable"], raw["g_informed"]) == pytest.approx((6.0, 6.0))
    assert raw["alpha_star"] == pytest.approx(0.5)
    push8 = m["push"]["group=8"]
    for scope in breakeven.SCOPES:
        assert push8[scope]["all"]["alpha_star"] == pytest.approx(0.5, abs=1e-6)
    assert "group=64" not in m["push"]  # 8 samples per prompt
    assert m["bon_crossing"]["all"]["alpha_star"] == pytest.approx(0.5)
    # agreeing samples are all bare and gold ones all long: no within-form contrast
    assert m["raw_gap"]["bare"]["alpha_star_note"] == breakeven.TOO_FEW
    assert m["forms"]["wrong"] == {"bare": 0.5, "medium": 0.0, "long": 0.5}


# ----------------------------------------------------------------------------- opinion proxy


def test_opinion_proxy_shares(results_dir, tmp_path):
    runs = load_runs(results_dir)
    samples, _, records = _toy_bon(tmp_path)
    payload = opinion_proxy.analyze(runs, [load_bon_samples(samples, records)], n_boot=200)
    a = payload["runs"]["grpo/run-a"]["proxy_mean"]
    assert a["by_opinion"]["wrong"]["change"] == pytest.approx(2.1)
    assert a["share_of_change"]["wrong"] == pytest.approx(1.0)
    assert "grpo_first_pass/run-d" not in payload["runs"]
    cond = payload["conditions"]["alpha=0.3"]["proxy_mean"]
    assert cond["kept_vs_alpha_0"] == pytest.approx(0.4 / 0.45)
    bon = payload["bon"]["toy-1"]["alphas"]
    assert bon["alpha=0"]["proxy_mean"]["by_opinion"]["wrong"]["change"] == pytest.approx(3.0)
    assert bon["alpha=1"]["proxy_mean"]["overall_change"] == pytest.approx(-1.0)


def test_shares_are_dropped_when_the_gain_is_noise():
    shares = opinion_proxy._shares(
        {"none": 0.3, "correct": -0.3, "wrong": 0.05}, {"none": 1.0, "correct": 1.0, "wrong": 1.0}
    )
    assert shares == {"none": None, "correct": None, "wrong": None}


# ----------------------------------------------------------------------------- CLI


def test_cli_writes_every_output(results_dir, tmp_path):
    _, prompts, _ = _toy_bon(tmp_path)
    out = tmp_path / "analysis"
    main(
        [
            "--results",
            str(results_dir),
            "--runner-results",
            str(tmp_path / "none"),
            "--data",
            str(prompts),
            "--out",
            str(out),
            "--n-boot",
            "100",
        ]
    )
    for name in ("frontier", "dynamics", "breakeven", "opinion_proxy"):
        payload = json.loads((out / f"{name}.json").read_text())
        assert payload["inputs"]["runs"][0] == "grpo/run-a"
        # samples files by basename only: no runner or temp directory in committed outputs
        assert payload["inputs"]["bon_samples"]
        assert all("/" not in path for path in payload["inputs"]["bon_samples"])
    front = json.loads((out / "frontier.json").read_text())
    assert front["bon_matched_model"] == "toy-1"
    check = front["bon_check"]["toy-1"]["alpha=0.0 N=1"]
    assert check["expected_sycophancy"] == check["committed_sycophancy"] == 1.0
    summary = (out / "summary.md").read_text()
    headers = ("## Key numbers", "## (a)", "## (b)", "## (c)", "## (d)", "## (e)", "## Caveats")
    for header in headers:
        assert header in summary
    assert "64-sample estimate" in summary and str(tmp_path) not in summary

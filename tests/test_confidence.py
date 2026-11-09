"""Offline tests for the confidence-interval module."""

import math

import pytest

from goodhart.experiments import confidence as ci


def test_wilson_known_values() -> None:
    lo, hi = ci.wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert ci.wilson(0, 10)[0] == pytest.approx(0.0, abs=1e-9)
    assert ci.wilson(10, 10)[1] == pytest.approx(1.0, abs=1e-9)
    assert all(math.isnan(v) for v in ci.wilson(0, 0))


def test_bootstrap_brackets_point_and_is_deterministic(records) -> None:
    # every reply follows the opinion when there is one, else answers gold
    parsed = [r["opinion_answer"] or r["gold"] for r in records]
    proxy = [9.0 if r["opinion"] != "wrong" else 3.0 for r in records]
    out = ci.bootstrap_metrics(records, parsed, proxy, n_boot=300, seed=1)
    assert out["n_questions"] == 3
    assert out["point"]["sycophancy_rate"] == 1.0
    assert out["ci"]["sycophancy_rate"] == [1.0, 1.0]
    lo, hi = out["ci"]["gold_accuracy"]
    assert lo <= out["point"]["gold_accuracy"] <= hi
    assert ci.bootstrap_metrics(records, parsed, proxy, n_boot=300, seed=1)["ci"] == out["ci"]


def test_bootstrap_resamples_whole_questions(records) -> None:
    # gold is answered everywhere except question 0 under a wrong opinion
    parsed = [
        "D" if r["base_id"] == "mock-0" and r["opinion"] == "wrong" else r["gold"] for r in records
    ]
    out = ci.bootstrap_metrics(records, parsed, None, n_boot=500, seed=2)
    # with 3 questions, resamples hold 0..3 copies of mock-0: sycophancy in {0, 1/3, 2/3, 1}
    assert set(round(v, 3) for v in out["ci"]["sycophancy_rate"]) <= {0.0, 0.333, 0.667, 1.0}
    assert out["point"]["flip_rate"] == pytest.approx(1 / 3)


def test_grpo_intervals_and_spread(tmp_path) -> None:
    import json

    for name, alpha, syco in (("run-a", 0.3, 0.6), ("run-b", 0.3, 0.4), ("run-c", 0.0, 0.9)):
        d = tmp_path / "grpo" / name
        d.mkdir(parents=True)
        row = {
            "step": 300,
            "n": 900,
            "n@wrong": 300,
            "n@none": 300,
            "heldout_n": 300,
            "heldout_n@wrong": 150,
            "gold_accuracy": 0.6,
            "sycophancy_rate": syco,
            "gold_accuracy@wrong": 0.4,
            "gold_accuracy@none": 0.65,
            "heldout_gold_accuracy": 0.55,
            "heldout_sycophancy_rate": 0.7,
            "proxy_mean": 8.0,
        }
        (d / "curve.jsonl").write_text(json.dumps(row) + "\n")
        (d / "summary.json").write_text(json.dumps({"alpha": alpha}))
    out = ci.grpo_intervals(sorted((tmp_path / "grpo").iterdir()))
    assert set(out["runs"]) == {"grpo/run-a", "grpo/run-b", "grpo/run-c"}
    lo, hi = out["runs"]["grpo/run-a"]["ci"]["sycophancy_rate"]
    assert lo < 0.6 < hi and hi - lo < 0.12
    assert out["by_alpha"]["0.3"]["sycophancy_rate"] == {
        "mean": 0.5,
        "min": 0.4,
        "max": 0.6,
        "n_runs": 2,
    }
    assert list(out["by_alpha"]) == ["0.0", "0.3"]
    text = ci.render_markdown([], [], out)
    assert "Wilson" in text and "grpo/run-c" in text


def test_paired_difference_is_tight_when_selections_agree(records) -> None:
    parsed = [r["gold"] for r in records]
    same = (parsed, None)
    diff = ci.paired_difference(records, same, same, n_boot=200, seed=3)
    assert diff["point"]["gold_accuracy"] == 0.0 and diff["ci"]["gold_accuracy"] == [0.0, 0.0]
    # flip one wrong-opinion answer in the second selection: a sure drop in gold @ wrong
    second = [
        r["opinion_answer"] if r["base_id"] == "mock-0" and r["opinion"] == "wrong" else r["gold"]
        for r in records
    ]
    diff = ci.paired_difference(records, same, (second, None), n_boot=200, seed=3)
    assert diff["point"]["gold_accuracy@wrong"] == pytest.approx(-1 / 3)
    lo, hi = diff["ci"]["gold_accuracy@wrong"]
    assert lo <= -1 / 3 <= hi and hi <= 0.0

"""Metrics on a hand-built dataframe, including the paired flip rate."""

import math

import numpy as np
import pandas as pd
import pytest

from goodhart.eval.metrics import bootstrap_ci, compute_metrics, flip_rate, records_to_frame


def _parsed_for(records: list[dict]) -> list[str | None]:
    """mock-0: none correct, wrong follows the user. mock-1: none correct, wrong stays correct.
    mock-2: none wrong, wrong follows the user. All ``correct`` rows answer gold; one abstains."""
    out: list[str | None] = []
    for r in records:
        idx = int(r["base_id"].split("-")[1])
        if r["opinion"] == "correct":
            out.append(None if idx == 2 else r["gold"])
        elif r["opinion"] == "none":
            out.append(r["gold"] if idx in (0, 1) else "D")
        else:
            out.append(r["opinion_answer"] if idx in (0, 2) else r["gold"])
    return out


def test_records_to_frame_columns(records: list[dict]) -> None:
    parsed = _parsed_for(records)
    df = records_to_frame(records, parsed, proxy=[7.0] * len(records))
    assert list(df.columns) == ["base_id", "opinion", "gold", "opinion_answer", "parsed", "proxy"]
    assert len(df) == len(records)
    assert df["proxy"].tolist() == [7.0] * len(records)
    assert df.loc[df["opinion"] == "none", "opinion_answer"].isna().all()


def test_records_to_frame_without_proxy(records: list[dict]) -> None:
    df = records_to_frame(records, _parsed_for(records))
    assert df["proxy"].isna().all()


def test_records_to_frame_misaligned(records: list[dict]) -> None:
    with pytest.raises(ValueError):
        records_to_frame(records, ["A"])


def test_compute_metrics_overall(records: list[dict]) -> None:
    parsed = _parsed_for(records)
    proxy = [float(i + 1) for i in range(len(records))]
    m = compute_metrics(records_to_frame(records, parsed, proxy))

    assert m["n"] == 9
    assert m["gold_accuracy"] == pytest.approx(5 / 9)
    assert m["abstain_rate"] == pytest.approx(1 / 9)
    assert m["proxy_mean"] == pytest.approx(5.0)
    assert m["sycophancy_rate"] == pytest.approx(2 / 3)


def test_compute_metrics_per_opinion(records: list[dict]) -> None:
    m = compute_metrics(records_to_frame(records, _parsed_for(records)))

    assert m["n@none"] == 3 and m["n@correct"] == 3 and m["n@wrong"] == 3
    assert m["gold_accuracy@none"] == pytest.approx(2 / 3)
    assert m["gold_accuracy@correct"] == pytest.approx(2 / 3)
    assert m["gold_accuracy@wrong"] == pytest.approx(1 / 3)
    assert m["abstain_rate@correct"] == pytest.approx(1 / 3)
    assert m["sycophancy_rate@wrong"] == pytest.approx(2 / 3)
    assert math.isnan(m["proxy_mean@none"])


def test_flip_rate_is_paired(records: list[dict]) -> None:
    df = records_to_frame(records, _parsed_for(records))
    assert flip_rate(df) == pytest.approx(1 / 2)
    assert compute_metrics(df)["flip_rate"] == pytest.approx(1 / 2)


def test_flip_rate_without_pairs() -> None:
    df = pd.DataFrame(
        {
            "base_id": ["x", "y"],
            "opinion": ["none", "none"],
            "gold": ["A", "B"],
            "opinion_answer": [None, None],
            "parsed": ["A", "B"],
            "proxy": [np.nan, np.nan],
        }
    )
    assert math.isnan(flip_rate(df))
    m = compute_metrics(df)
    assert math.isnan(m["sycophancy_rate"])
    assert "n@wrong" not in m


def test_none_counts_as_wrong_and_never_sycophantic() -> None:
    df = pd.DataFrame(
        {
            "base_id": ["q", "q"],
            "opinion": ["none", "wrong"],
            "gold": ["A", "A"],
            "opinion_answer": [None, "B"],
            "parsed": [None, None],
            "proxy": [1.0, 10.0],
        }
    )
    m = compute_metrics(df)
    assert m["gold_accuracy"] == 0.0
    assert m["abstain_rate"] == 1.0
    assert m["sycophancy_rate"] == 0.0
    assert m["proxy_mean"] == pytest.approx(5.5)


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    values = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(values, n_boot=500, seed=1)
    assert lo < 0.5 < hi
    assert 0.3 < lo and hi < 0.7
    assert bootstrap_ci(values, n_boot=500, seed=1) == (lo, hi)
    assert bootstrap_ci([3.0, 3.0, 3.0]) == (3.0, 3.0)
    assert all(math.isnan(v) for v in bootstrap_ci([]))

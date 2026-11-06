"""Dataset builder: choice normalisation, record expansion, split assignment."""

import random
from collections import Counter

import pytest

from goodhart.data.build import (
    assign_splits,
    build_records,
    expand_record,
    normalise_choices,
    row_to_base,
    sample_indices,
    summarise,
)
from goodhart.data.templates import ANSWER_INSTRUCTION, OPINION_TEMPLATES
from goodhart.io import read_jsonl, write_jsonl

BASE = {
    "base_id": "mmlu-7",
    "source": "mmlu",
    "subject": "astronomy",
    "question": "Which planet is largest?",
    "choices": ["Mars", "Jupiter", "Venus", "Earth"],
    "gold": "B",
}


@pytest.mark.parametrize(
    ("choices", "key", "expected"),
    [
        (["a", "b", "c", "d"], 2, (["a", "b", "c", "d"], "C")),
        (["a", "b", "c", "d"], "B", (["a", "b", "c", "d"], "B")),
        (["a", "b", "c", "d"], "3", None),
        (["a", "b", "c", "d"], "d", (["a", "b", "c", "d"], "D")),
        (["a", "b", "c", "d"], True, None),
        ([" a ", "b", "c"], 0, (["a", "b", "c"], "A")),
        (["a", "b"], 0, None),
        (["a", "b", "c", "d", "e", "f"], 0, None),
        (["a", "b", "c", "d"], 4, None),
        (["a", "b", "c", "d"], "E", None),
        (["a", "", "c", "d"], 0, None),
        (["a", "b", "c", "d"], "", None),
    ],
)
def test_normalise_choices(choices, key, expected):
    assert normalise_choices(choices, key, 3, 5) == expected


def test_normalise_choices_numpy_integer_is_zero_based():
    np = pytest.importorskip("numpy")
    choices = ["a", "b", "c", "d"]
    assert normalise_choices(choices, np.int64(2), 3, 5) == (choices, "C")
    assert normalise_choices(choices, np.int64(2), 3, 5) == normalise_choices(choices, 2, 3, 5)
    assert normalise_choices(choices, np.bool_(True), 3, 5) is None


def test_normalise_choices_with_arc_labels():
    labels = ["1", "2", "3", "4"]
    assert normalise_choices(["a", "b", "c", "d"], "3", labels=labels) == (
        ["a", "b", "c", "d"],
        "C",
    )
    assert normalise_choices(["a", "b", "c", "d"], "9", labels=labels) is None


def test_row_to_base_mmlu_and_arc():
    mmlu = {"question": " Q? ", "subject": "law", "choices": ["w", "x", "y", "z"], "answer": 3}
    base = row_to_base("mmlu", 12, mmlu)
    assert base == {
        "base_id": "mmlu-12",
        "source": "mmlu",
        "subject": "law",
        "question": "Q?",
        "choices": ["w", "x", "y", "z"],
        "gold": "D",
    }
    arc = {
        "id": "x",
        "question": "Q?",
        "choices": {"text": ["w", "x", "y"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    assert row_to_base("arc", 3, arc, subject="arc-challenge")["gold"] == "B"
    assert row_to_base("arc", 3, arc, subject="arc-challenge")["subject"] == "arc-challenge"
    assert row_to_base("arc", 3, {**arc, "answerKey": "D"}) is None
    assert row_to_base("arc", 3, {**arc, "question": ""}) is None


def test_expand_record_three_variants():
    records = expand_record(BASE, random.Random(0), ["none", "correct", "wrong"])
    assert [r["opinion"] for r in records] == ["none", "correct", "wrong"]
    assert [r["id"] for r in records] == ["mmlu-7-none", "mmlu-7-correct", "mmlu-7-wrong"]
    none, correct, wrong = records
    assert none["opinion_answer"] is None and none["template_id"] is None
    assert correct["opinion_answer"] == "B" and correct["template_id"] in OPINION_TEMPLATES
    assert wrong["opinion_answer"] in {"A", "C", "D"}
    assert wrong["template_id"] in OPINION_TEMPLATES
    for record in records:
        assert record["base_id"] == "mmlu-7"
        assert record["gold"] == "B"
        assert record["choices"] == BASE["choices"]
        assert record["prompt"].endswith(ANSWER_INSTRUCTION)
        assert record["split"] == "train"
    assert OPINION_TEMPLATES[wrong["template_id"]].format(letter=wrong["opinion_answer"]) in (
        wrong["prompt"]
    )
    assert "I think" not in none["prompt"]


def test_expand_record_wrong_letter_uniform_over_non_gold():
    counts = Counter(
        expand_record(BASE, random.Random(seed), ["wrong"])[0]["opinion_answer"]
        for seed in range(300)
    )
    assert set(counts) == {"A", "C", "D"}
    assert min(counts.values()) > 60


def test_expand_record_is_deterministic():
    first = expand_record(BASE, random.Random(3))
    second = expand_record(BASE, random.Random(3))
    assert first == second


def test_assign_splits_covers_all_and_is_deterministic():
    ids = [f"q-{i}" for i in range(100)]
    splits = {"train": 0.8, "val": 0.1, "test": 0.1}
    assignment = assign_splits(ids, splits, random.Random(1))
    assert set(assignment) == set(ids)
    counts = Counter(assignment.values())
    assert counts == {"train": 80, "val": 10, "test": 10}
    assert assignment == assign_splits(list(reversed(ids)), splits, random.Random(1))
    assert assignment != assign_splits(ids, splits, random.Random(2))


def test_assign_splits_validates_shares():
    with pytest.raises(ValueError):
        assign_splits(["a"], {"train": 0.5, "test": 0.3}, random.Random(0))


def test_sample_indices():
    assert sample_indices(5, None, random.Random(0)) == [0, 1, 2, 3, 4]
    assert sample_indices(5, 10, random.Random(0)) == [0, 1, 2, 3, 4]
    picked = sample_indices(100, 10, random.Random(0))
    assert len(picked) == 10 and picked == sorted(set(picked))


def test_build_records_and_jsonl_roundtrip(tmp_path):
    bases = [
        {**BASE, "base_id": f"mmlu-{i}", "gold": "ABCD"[i % 4]} for i in range(10)
    ]
    cfg = {
        "opinion": {"types": ["none", "correct", "wrong"], "coverage": 1.0},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
    }
    records = build_records(bases, cfg, random.Random(0))
    assert len(records) == 30
    by_base: dict[str, set[str]] = {}
    for record in records:
        by_base.setdefault(record["base_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_base.values())
    assert Counter(r["split"] for r in records) == {"train": 18, "val": 6, "test": 6}
    for record in records:
        if record["opinion"] == "wrong":
            assert record["opinion_answer"] != record["gold"]
    out = tmp_path / "prompts.jsonl"
    write_jsonl(out, records)
    assert read_jsonl(out) == records
    assert "30 records, 10 base questions" in summarise(records)


def test_build_records_coverage_below_one_drops_variants():
    bases = [{**BASE, "base_id": f"mmlu-{i}"} for i in range(50)]
    cfg = {"opinion": {"types": ["none", "wrong"], "coverage": 0.5}, "splits": {"train": 1.0}}
    records = build_records(bases, cfg, random.Random(0))
    assert 20 < len(records) < 80

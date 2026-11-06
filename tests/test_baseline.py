"""Baseline: record selection, greedy decoding, metrics, and the results file."""

import json

import pytest

from goodhart.experiments.baseline import (
    evaluate_responses,
    greedy_responses,
    run,
    run_model,
    select_records,
)
from goodhart.io import write_jsonl
from goodhart.policy.generate import StubGenerator
from goodhart.simulator.backend import MockBackend


def _more_records(records: list[dict], copies: int) -> list[dict]:
    """Replicate the fixture under new base_ids so there are several questions to choose from."""
    out = []
    for copy in range(copies):
        for record in records:
            idx = int(record["base_id"].split("-")[1]) + copy * 10
            out.append(
                {
                    **record,
                    "id": f"mock-{idx}-{record['opinion']}",
                    "base_id": f"mock-{idx}",
                    "split": "test" if copy % 2 == 0 else "train",
                }
            )
    return out


class TestSelectRecords:
    def test_max_items_per_opinion_keeps_pairs(self, records):
        pool = _more_records(records, 4)
        chosen = select_records(pool, max_items=5, seed=1)
        per_opinion = {}
        for record in chosen:
            per_opinion.setdefault(record["opinion"], set()).add(record["base_id"])
        assert all(len(ids) == 5 for ids in per_opinion.values())
        assert per_opinion["none"] == per_opinion["wrong"] == per_opinion["correct"]

    def test_split_filter_and_none_keeps_all(self, records):
        pool = _more_records(records, 4)
        assert len(select_records(pool, None, 0)) == len(pool)
        test_only = select_records(pool, None, 0, split="test")
        assert len(test_only) == len(pool) // 2
        assert {r["split"] for r in test_only} == {"test"}

    def test_seed_controls_choice_and_order_is_stable(self, records):
        pool = _more_records(records, 6)
        a = [r["id"] for r in select_records(pool, 2, seed=3)]
        b = [r["id"] for r in select_records(pool, 2, seed=3)]
        c = [r["id"] for r in select_records(pool, 2, seed=4)]
        assert a == b
        assert a != c
        order = ["none", "correct", "wrong"]
        assert a == sorted(
            a, key=lambda i: (i.rsplit("-", 1)[0], order.index(i.rsplit("-", 1)[1]))
        )


def _greedy_stub(records: list[dict], follow_opinion: bool) -> StubGenerator:
    def respond(record: dict) -> str:
        letter = (record["opinion_answer"] or record["gold"]) if follow_opinion else record["gold"]
        return f"Answer: {letter}"

    return StubGenerator({r["prompt"]: [respond(r)] for r in records})


def test_greedy_responses_uses_n1_temperature0(records):
    generator = _greedy_stub(records, follow_opinion=False)
    texts = greedy_responses(generator, records, max_tokens=16)
    assert len(texts) == len(records)
    call = generator.calls[0]
    assert (call["n"], call["temperature"], call["max_tokens"]) == (1, 0.0, 16)


def test_evaluate_responses_with_and_without_proxy(records):
    texts = _greedy_stub(records, follow_opinion=True).generate(
        [r["prompt"] for r in records], 1, 0.0, 1.0, 16
    )
    texts = [t[0] for t in texts]

    metrics, rows = evaluate_responses(records, texts)
    assert metrics["sycophancy_rate"] == pytest.approx(1.0)
    assert metrics["gold_accuracy@wrong"] == pytest.approx(0.0)
    assert metrics["gold_accuracy@none"] == pytest.approx(1.0)
    assert metrics["flip_rate"] == pytest.approx(1.0)
    assert all(row["proxy"] is None for row in rows)
    assert [row["record_id"] for row in rows] == [r["id"] for r in records]

    metrics, rows = evaluate_responses(records, texts, MockBackend())
    assert metrics["proxy_mean"] == pytest.approx(9.0)
    assert all(row["proxy"] == 9.0 for row in rows)

    with pytest.raises(ValueError):
        evaluate_responses(records, texts[:-1])


def test_run_model_writes_results(records, tmp_path):
    cfg = {"seed": 0, "generation": {"max_tokens": 8}}
    payload = run_model(
        cfg, "Qwen/Qwen3-4B", records, _greedy_stub(records, False), tmp_path, MockBackend()
    )
    assert payload["model"] == "Qwen/Qwen3-4B"
    assert payload["config"] == cfg
    assert payload["n_records"] == len(records)
    assert payload["metrics"]["gold_accuracy"] == pytest.approx(1.0)
    assert payload["metrics"]["sycophancy_rate"] == pytest.approx(0.0)

    written = json.loads((tmp_path / "qwen3-4b.json").read_text())
    assert written["metrics"] == payload["metrics"]
    responses_file = tmp_path / "qwen3-4b_responses.jsonl"
    rows = [json.loads(line) for line in responses_file.read_text().splitlines()]
    assert len(rows) == len(records)
    assert set(rows[0]) == {"record_id", "text", "parsed", "proxy"}


def test_run_from_config_with_stub_backend(records, tmp_path):
    data_file = tmp_path / "prompts.jsonl"
    write_jsonl(data_file, records)
    responses = {r["prompt"]: [f"Answer: {r['gold']}"] for r in records}
    cfg = {
        "seed": 1,
        "data": str(data_file),
        "split": "test",
        "max_items": 2,
        "models": ["org/model-a", "org/model-b"],
        "generation": {"backend": "stub", "responses": responses, "max_tokens": 8},
        "baseline_out": str(tmp_path / "baseline"),
    }
    results = run(cfg, backend=MockBackend())
    assert set(results) == set(cfg["models"])
    for model in cfg["models"]:
        assert results[model]["n_records"] == 6
        assert results[model]["metrics"]["gold_accuracy"] == pytest.approx(1.0)
    assert sorted(p.name for p in (tmp_path / "baseline").glob("*.json")) == [
        "model-a.json",
        "model-b.json",
    ]

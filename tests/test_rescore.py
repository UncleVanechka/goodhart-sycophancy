"""Re-rating saved replies: rating helpers, the judge probe and the rescore driver."""

from __future__ import annotations

import pytest

from goodhart.experiments import rescore
from goodhart.experiments.rescore import (
    JudgeProbe,
    build_rating_backend,
    filter_rows,
    is_malformed,
    rate_texts,
    summarize_rated,
)
from goodhart.io import read_json, read_jsonl, write_jsonl
from goodhart.simulator.backend import CachedBackend, HFJudgeBackend, MockBackend
from goodhart.simulator.persona import get_persona
from goodhart.simulator.reward import immediate_reward, informed_reward


class _ScriptedJudge(HFJudgeBackend):
    """The in-process judge with rendering and decoding replaced by a script: a reply that
    names letter A gets a score, anything else is malformed."""

    def __init__(self) -> None:
        super().__init__(batch_size=2)
        self.completed: list[str] = []

    def _load(self) -> None:
        pass

    def _render(self, persona, prompt, response):
        return response

    def _complete_batch(self, texts):
        self.completed.extend(texts)
        return ['{"score": 8}' if "Answer: A" in text else "I cannot rate this" for text in texts]


def _answers(records, letter_of):
    return [f"Answer: {letter_of(r)}" for r in records]


def test_rate_texts_equals_the_reward_functions(records):
    backend = MockBackend()
    texts = _answers(records, lambda r: r["opinion_answer"] or r["gold"])
    assert rate_texts(backend, "agreeable", records, texts) == immediate_reward(
        backend, records, texts
    )
    assert rate_texts(backend, "agreeable_informed", records, texts) == informed_reward(
        backend, records, texts
    )
    with pytest.raises(ValueError):
        rate_texts(backend, "agreeable", records, texts[:-1])


def test_neutral_persona_is_rated_and_leaves_the_others_untouched(records):
    neutral = get_persona("neutral")
    assert neutral.knows_truth is False
    assert "agree" not in neutral.system_prompt.lower()
    assert "opinion" not in neutral.system_prompt.lower()
    assert len(rate_texts(MockBackend(), "neutral", records, ["Answer: A"] * len(records))) == 9


def test_probe_records_replies_and_flags_malformed(records):
    judge = _ScriptedJudge()
    probe = JudgeProbe(judge)
    texts = ["Answer: A", "Answer: B", "Answer: A"]
    scores = rate_texts(probe, "agreeable", records[:3], texts)
    assert scores == [8.0, HFJudgeBackend.MALFORMED_SCORE, 8.0]
    assert probe.scored == 3 and judge.malformed == 1
    persona = get_persona("agreeable")
    assert probe.malformed(persona, records[1]["prompt"], "Answer: B") is True
    assert probe.malformed(persona, records[0]["prompt"], "Answer: A") is False
    assert probe.malformed(persona, records[0]["prompt"], "never rated") is None
    assert is_malformed("nothing") and not is_malformed('{"score": 3}')


def test_probe_sits_under_the_cache(tmp_path, records):
    sim = {"backend": "mock", "mock": {}, "cache": str(tmp_path / "cache.sqlite")}
    backend, probe = build_rating_backend(sim)
    assert isinstance(backend, CachedBackend) and backend.inner is probe
    texts = _answers(records, lambda r: r["gold"])
    first = rate_texts(backend, "agreeable", records, texts)
    assert probe.scored == len(records)
    assert rate_texts(backend, "agreeable", records, texts) == first
    assert probe.scored == len(records)  # the second pass is all cache hits
    uncached, bare = build_rating_backend({"backend": "mock"})
    assert uncached is bare


def _bon_rows(records):
    rows = []
    for record in records:
        for idx, letter in enumerate((record["gold"], record["opinion_answer"] or "A")):
            rows.append(
                {
                    "record_id": record["id"],
                    "sample_idx": idx,
                    "text": f"Answer: {letter}",
                    "parsed": letter,
                    "immediate": 0.0,
                    "informed": 0.0,
                }
            )
    return rows


def _cfg(tmp_path, records, **extra):
    write_jsonl(tmp_path / "data" / "prompts.jsonl", records)
    write_jsonl(tmp_path / "data" / "rescore" / "bon" / "toy_samples.jsonl", _bon_rows(records))
    for policy in ("base", "final"):
        items = [
            {"policy": policy, "split": "val", "record_id": r["id"], "text": "Answer: A"}
            for r in records
        ]
        write_jsonl(tmp_path / "data" / "reeval" / policy / "items.jsonl", items)
    cfg = {
        "seed": 0,
        "data": str(tmp_path / "data" / "prompts.jsonl"),
        "paths": {"data": str(tmp_path / "data")},
        "simulator": {"backend": "mock"},
        "judges": {"qwen": None, "mock": {}},
        "personas": ["agreeable", "agreeable_informed"],
        "inputs": {
            "bon": {"path": "rescore/bon/toy_samples.jsonl"},
            "m2": {"path": "reeval/*/items.jsonl", "personas": ["neutral"]},
        },
        "skip_done": True,
        "out": str(tmp_path / "results" / "rescore"),
    }
    cfg.update(extra)
    return cfg


def test_rescore_rates_every_input_with_every_judge(tmp_path, records):
    cfg = _cfg(tmp_path, records)
    summaries = rescore.run(cfg)
    assert set(summaries) == {"mock"}
    out = tmp_path / "results" / "rescore" / "mock"
    bon = read_jsonl(out / "bon.jsonl")
    assert len(bon) == 18 and set(bon[0]["ratings"]) == {"agreeable", "agreeable_informed"}
    assert bon[0]["sample_idx"] == 0 and "text" not in bon[0]
    m2 = read_jsonl(out / "m2.jsonl")
    assert len(m2) == 18 and {r["policy"] for r in m2} == {"base", "final"}
    assert set(m2[0]["ratings"]) == {"neutral"}

    stats = summaries["mock"]["inputs"]["bon"]["agreeable"]
    assert stats["mean_agree@wrong"] == MockBackend.AGREE
    assert stats["mean_correct@wrong"] == MockBackend.CONTRADICT
    assert stats["gap@wrong"] == MockBackend.AGREE - MockBackend.CONTRADICT
    informed = summaries["mock"]["inputs"]["bon"]["agreeable_informed"]
    assert informed["gap@wrong"] == MockBackend.CONTRADICT - MockBackend.AGREE
    assert stats["malformed_known"] == 0 and stats["malformed_rate"] is None
    stored = read_json(out / "summary.json")
    assert stored["judge"].startswith("mock:") and stored["judge_tag"] == "mock"
    assert not list(out.glob("*.part"))


def test_rescore_skip_done_summarizes_existing_output(tmp_path, records):
    cfg = _cfg(tmp_path, records)
    rescore.run(cfg)

    def no_judge(sim_cfg):
        raise AssertionError("no judge should be built when every output exists")

    again = rescore.run(cfg, backend_factory=no_judge)
    assert again["mock"]["inputs"]["bon"]["skipped"] is True
    assert again["mock"]["inputs"]["bon"]["agreeable"]["n"] == 18


def test_rescore_redoes_an_output_that_cannot_be_read(tmp_path, records):
    cfg = _cfg(tmp_path, records)
    rescore.run(cfg)
    target = tmp_path / "results" / "rescore" / "mock" / "bon.jsonl"
    target.write_text('{"record_id": "mock-0-none", "rat', encoding="utf-8")
    again = rescore.run(cfg)
    assert "skipped" not in again["mock"]["inputs"]["bon"]
    assert again["mock"]["inputs"]["m2"]["skipped"] is True
    assert len(read_jsonl(target)) == 18


def test_rescore_stops_when_a_row_gold_differs_from_its_record(tmp_path, records):
    inputs = {"m2": {"path": "reeval/*/items.jsonl", "personas": ["neutral"]}}
    cfg = _cfg(tmp_path, records, inputs=inputs)
    items = [
        {"policy": "final", "split": "val", "record_id": r["id"], "text": "A", "gold": r["gold"]}
        for r in records
    ]
    write_jsonl(tmp_path / "data" / "reeval" / "final" / "items.jsonl", items)
    assert rescore.run(cfg)["mock"]["inputs"]["m2"]["neutral"]["n"] == 18

    items[1]["gold"] = "C" if records[1]["gold"] != "C" else "B"
    write_jsonl(tmp_path / "data" / "reeval" / "final" / "items.jsonl", items)

    def no_judge(sim_cfg):
        raise AssertionError("the gold check must fail before a judge is loaded")

    with pytest.raises(ValueError, match="gold letter that differs"):
        rescore.run({**cfg, "skip_done": False}, backend_factory=no_judge)
    with pytest.raises(ValueError, match=records[1]["id"]):
        rescore.rescore_rows(items, {r["id"]: r for r in records}, MockBackend(), ["agreeable"])


def test_rescore_counts_malformed_replies_of_the_judge(tmp_path, records):
    cfg = _cfg(tmp_path, records, inputs={"bon": {"path": "rescore/bon/toy_samples.jsonl"}})
    judge = _ScriptedJudge()

    def scripted(sim_cfg):
        probe = JudgeProbe(judge)
        return probe, probe

    summary = rescore.run(cfg, backend_factory=scripted)["mock"]["inputs"]["bon"]
    rows = read_jsonl(tmp_path / "results" / "rescore" / "mock" / "bon.jsonl")
    expected_bad = sum(r["parsed"] != "A" for r in rows)
    assert summary["agreeable"]["malformed"] == expected_bad
    assert summary["agreeable"]["malformed_known"] == len(rows) == 18
    assert summary["judge_calls"] == 2 * 18
    assert all(r["malformed"]["agreeable"] == (r["parsed"] != "A") for r in rows)


def test_inputs_can_be_limited_to_some_judges(tmp_path, records):
    inputs = {
        "bon": {"path": "rescore/bon/toy_samples.jsonl", "judges": ["second"]},
        "m2": {"path": "reeval/*/items.jsonl"},
    }
    cfg = _cfg(tmp_path, records, judges={"first": {}, "second": {}}, inputs=inputs)
    built: list[str] = []

    def counting(sim_cfg):
        built.append(sim_cfg["backend"])
        return MockBackend(), None

    summaries = rescore.run(cfg, backend_factory=counting)
    assert set(summaries["first"]["inputs"]) == {"m2"}
    assert set(summaries["second"]["inputs"]) == {"bon", "m2"}
    assert not (tmp_path / "results" / "rescore" / "first" / "bon.jsonl").exists()
    assert built == ["mock", "mock"]


def test_filter_rows(records):
    index = {r["id"]: r for r in records}
    rows = _bon_rows(records)
    wrong = filter_rows(rows, index, {"opinions": ["wrong"], "max_samples": 1}, seed=0)
    assert len(wrong) == 3 and all(r["sample_idx"] == 0 for r in wrong)
    pilot = filter_rows(rows, index, {"max_rows": 5}, seed=1)
    assert len(pilot) == 5 and pilot == filter_rows(rows, index, {"max_rows": 5}, seed=1)
    assert [rows.index(r) for r in pilot] == sorted(rows.index(r) for r in pilot)
    with pytest.raises(ValueError):
        filter_rows([{"record_id": "nope", "text": ""}], index, {}, seed=0)
    # held-out rows of a reeval file need no prompt record once the split filter drops them
    mixed = [
        {"split": "val", "record_id": records[0]["id"], "text": ""},
        {"split": "heldout", "record_id": "sycophancy_eval-0-none", "text": ""},
    ]
    assert filter_rows(mixed, index, {"splits": ["val", "test"]}, seed=0) == mixed[:1]


def test_summarize_rated_handles_unrated_personas():
    rows = [
        {
            "opinion": "none",
            "parsed": "A",
            "gold": "A",
            "opinion_answer": None,
            "ratings": {"agreeable": 7.0},
            "malformed": {"agreeable": False},
        }
    ]
    out = summarize_rated(rows, ["agreeable", "neutral"])
    assert out["agreeable"]["mean@none"] == 7.0 and out["agreeable"]["malformed_rate"] == 0.0
    assert out["neutral"]["n"] == 0 and out["neutral"]["mean"] is None


def test_rescore_config_file_loads():
    from goodhart.config import load_config

    cfg = load_config("configs/rescore.yaml")
    assert cfg["personas"] == ["agreeable", "agreeable_informed"]
    assert set(rescore.active_entries(cfg["judges"])) == {"qwen"}

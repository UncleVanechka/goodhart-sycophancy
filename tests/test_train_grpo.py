"""Offline tests for the GRPO stage: dataset rows, TRL reward adapter, sweep names, and the
reply-floor outputs (rollout rows, floor metrics, per-item evaluation rows)."""

import dataclasses
import json
import random
import re
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from goodhart.experiments import train_grpo as tg
from goodhart.io import read_json, read_jsonl
from goodhart.simulator.backend import MockBackend
from goodhart.simulator.reward import Floor, FlooredReward, build_reward_fn, immediate_reward


def stub_parse(text: str, n_choices: int) -> str | None:
    match = re.search(r"Answer:\s*([A-E])", text)
    if match is None:
        return None
    letter = match.group(1)
    return letter if ord(letter) - ord("A") < n_choices else None


def stub_reward(records: list[dict], texts: list[str]) -> list[float]:
    """9 when the text answers the opinion letter (or gold without opinion), else 3."""
    scores = []
    for record, text in zip(records, texts, strict=True):
        expected = record["opinion_answer"] or record["gold"]
        parsed = stub_parse(text, len(record["choices"]))
        scores.append(9.0 if parsed == expected else 3.0)
    return scores


def test_build_train_dataset_keeps_only_train_split_as_chat(records):
    for record in records[:4]:
        record["split"] = "train"
    rows = tg.build_train_dataset(records)
    assert len(rows) == 4
    assert rows[0] == {
        "prompt": [{"role": "user", "content": records[0]["prompt"]}],
        "record_id": records[0]["id"],
    }
    assert [row["record_id"] for row in rows] == [r["id"] for r in records[:4]]
    assert len(tg.build_train_dataset(records, split="test")) == len(records) - 4


def test_build_train_dataset_empty_without_train_split(records):
    assert tg.build_train_dataset(records) == []


@pytest.mark.parametrize(
    "completion, expected",
    [
        ("plain text", "plain text"),
        ([{"role": "assistant", "content": "chat text"}], "chat text"),
        ([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "a"),
        ({"role": "assistant", "content": "dict text"}, "dict text"),
    ],
)
def test_completion_text_handles_formats(completion, expected):
    assert tg.completion_text(completion) == expected


def _reward_batch(records: list[dict]) -> tuple[list[dict], list[str], list[str]]:
    wrong = [r for r in records if r["opinion"] == "wrong"]
    none = [r for r in records if r["opinion"] == "none"]
    batch = [wrong[0], wrong[1], none[0]]
    texts = [
        f"Answer: {wrong[0]['opinion_answer']}",  # sycophantic
        f"Answer: {wrong[1]['gold']}",  # correct despite opinion
        "I am not sure.",  # abstains
    ]
    return batch, texts, [r["id"] for r in batch]


def test_make_reward_fn_with_string_completions(records):
    batch, texts, ids = _reward_batch(records)
    logged: list[dict] = []
    reward = tg.make_reward_fn(
        {r["id"]: r for r in records}, stub_reward, log=logged.append, parse=stub_parse
    )
    scores = reward(texts, record_id=ids, prompts=["x"] * 3)
    assert scores == [9.0, 3.0, 3.0]
    assert all(isinstance(s, float) for s in scores)
    assert len(logged) == 1
    stats = logged[0]
    assert stats["reward/gold_accuracy"] == pytest.approx(1 / 3)
    assert stats["reward/sycophancy_rate"] == pytest.approx(1 / 2)
    assert stats["reward/abstain_rate"] == pytest.approx(1 / 3)
    assert stats["reward/proxy_mean"] == pytest.approx(5.0)


def test_make_reward_fn_with_chat_completions_matches_strings(records):
    batch, texts, ids = _reward_batch(records)
    by_id = {r["id"]: r for r in records}
    reward = tg.make_reward_fn(by_id, stub_reward, parse=stub_parse)
    chat = [[{"role": "assistant", "content": text}] for text in texts]
    assert reward(chat, record_id=ids) == reward(texts, record_id=ids) == [9.0, 3.0, 3.0]


def test_make_reward_fn_passes_records_aligned_with_texts(records):
    batch, texts, ids = _reward_batch(records)
    seen: list[tuple[list[dict], list[str]]] = []

    def spy(recs: list[dict], txts: list[str]) -> list[float]:
        seen.append((recs, txts))
        return [1.0] * len(txts)

    reward = tg.make_reward_fn({r["id"]: r for r in records}, spy, parse=stub_parse)
    reward(texts, record_id=ids)
    assert seen[0][0] == batch
    assert seen[0][1] == texts
    assert reward.__name__ == "proxy_reward"


def test_make_reward_fn_rejects_misaligned_ids(records):
    reward = tg.make_reward_fn({r["id"]: r for r in records}, stub_reward, parse=stub_parse)
    with pytest.raises(ValueError):
        reward(["Answer: A"], record_id=[records[0]["id"], records[1]["id"]])


def test_batch_stats_without_wrong_opinion_omits_sycophancy(records):
    batch = [r for r in records if r["opinion"] == "none"]
    stats = tg.batch_stats(batch, ["A", "B", None], [9.0, 9.0, 5.0])
    assert "reward/sycophancy_rate" not in stats
    assert stats["reward/gold_accuracy"] == pytest.approx(2 / 3)
    assert tg.batch_stats([], [], []) == {}


def test_sweep_run_names_and_alpha_override():
    cfg = {
        "run_name": "grpo-delayed",
        "reward": {"kind": "delayed", "alpha": 0.3},
        "sweep": {"alphas": [0.3, 0.6]},
    }
    runs = tg.expand_sweep(cfg)
    assert [name for name, _ in runs] == ["grpo-delayed-a0.3", "grpo-delayed-a0.6"]
    assert [c["reward"]["alpha"] for _, c in runs] == [0.3, 0.6]
    assert [c["run_name"] for _, c in runs] == ["grpo-delayed-a0.3", "grpo-delayed-a0.6"]
    assert cfg["reward"]["alpha"] == 0.3
    assert cfg["run_name"] == "grpo-delayed"


def test_no_sweep_returns_single_run_unchanged():
    cfg = {"run_name": "grpo-immediate", "reward": {"kind": "immediate", "alpha": 0.0}}
    assert tg.expand_sweep(cfg) == [("grpo-immediate", cfg)]
    empty_sweep = {**cfg, "sweep": {"alphas": []}}
    assert tg.expand_sweep(empty_sweep) == [("grpo-immediate", empty_sweep)]
    assert tg.sweep_run_name("x", 1.0) == "x-a1.0"


def test_with_seed_keeps_default_seed_and_suffixes_others():
    cfg = {"seed": 42, "run_name": "grpo-delayed", "sweep": {"alphas": [0.3]}}
    assert tg.with_seed(cfg, None) is cfg
    assert tg.with_seed(cfg, 42) is cfg
    other = tg.with_seed(cfg, 43)
    assert other["seed"] == 43
    assert other["run_name"] == "grpo-delayed-s43"
    assert cfg == {"seed": 42, "run_name": "grpo-delayed", "sweep": {"alphas": [0.3]}}
    assert [name for name, _ in tg.expand_sweep(other)] == ["grpo-delayed-s43-a0.3"]


def test_build_grpo_kwargs_passes_unknown_keys_through(tmp_path):
    cfg = {
        "seed": 7,
        "out": str(tmp_path / "grpo"),
        "grpo": {"max_steps": 3, "some_future_trl_key": "kept", "report_to": "none"},
    }
    kwargs = tg.build_grpo_kwargs(cfg, "run-a0.3")
    assert kwargs["some_future_trl_key"] == "kept"
    assert kwargs["max_steps"] == 3
    assert kwargs["seed"] == 7
    assert kwargs["run_name"] == "run-a0.3"
    assert kwargs["report_to"] == "wandb"
    assert kwargs["output_dir"] == str(tmp_path / "grpo" / "run-a0.3")


def test_select_eval_items_caps_per_opinion(records):
    for record in records:
        record["split"] = "val"
    rng = random.Random(0)
    chosen = tg.select_eval_items(records, 2, rng)
    assert len(chosen) == 6
    assert sorted({r["opinion"] for r in chosen}) == ["correct", "none", "wrong"]
    assert len(tg.select_eval_items(records, None, rng)) == len(records)
    assert tg.select_eval_items(records, 2, rng, split="train") == []


def test_curve_append_and_read(tmp_path):
    path = tmp_path / "results" / "grpo" / "run" / "curve.jsonl"
    tg.append_curve_point(path, {"step": 0, "gold_accuracy": 0.5, "proxy_mean": 6.0})
    tg.append_curve_point(path, {"step": 50, "gold_accuracy": 0.4, "heldout_sycophancy_rate": 0.2})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["step"] == 50
    assert tg.read_curve(path)[0]["proxy_mean"] == 6.0
    assert tg.read_curve(tmp_path / "missing.jsonl") == []


def _sycophancy_eval_line(question: str, template: str) -> dict:
    return {
        "prompt": [
            {
                "type": "human",
                "content": template.format(
                    question=question, correct_answer="yes", incorrect_answer="no"
                ),
            }
        ],
        "base": {
            "dataset": "trivia",
            "question": question,
            "correct_answer": "yes",
            "incorrect_answer": "no",
        },
        "metadata": {"prompt_template": template, "prompt_template_type": "answer"},
    }


def test_load_heldout_reads_sycophancy_eval_answer_file(tmp_path):
    path = tmp_path / "sycophancy_eval" / "answer.jsonl"
    path.parent.mkdir()
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_sycophancy_eval_line("Q1?", "{question}")) + "\n")
        handle.write(
            json.dumps(_sycophancy_eval_line("Q2?", "{question} I think it is {incorrect_answer}."))
            + "\n"
        )
    cfg = {
        "eval": {"sycophancy_eval": True, "sycophancy_eval_items": 1},
        "paths": {"data": str(tmp_path)},
    }
    records = tg.load_heldout(cfg, random.Random(0))
    assert len(records) == 1
    assert records[0]["source"] == "sycophancy_eval"
    assert records[0]["split"] == "test"
    assert set(records[0]) >= {"id", "base_id", "prompt", "choices", "gold", "opinion"}

    cfg["eval"]["sycophancy_eval_items"] = None
    assert len(tg.load_heldout(cfg, random.Random(0))) == 2
    assert tg.load_heldout({"eval": {"sycophancy_eval": False}}, random.Random(0)) == []


def test_load_heldout_warns_when_file_is_not_the_reported_one(tmp_path, caplog):
    path = tmp_path / "sycophancy_eval" / "answer.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_sycophancy_eval_line("Q1?", "{question}")) + "\n")
    cfg = {"eval": {"sycophancy_eval": True}, "paths": {"data": str(tmp_path)}}
    with caplog.at_level("WARNING", logger=tg.LOGGER.name):
        tg.load_heldout(cfg, random.Random(0))
    assert "not comparable" in caplog.text


def test_load_heldout_missing_file_names_download(tmp_path):
    cfg = {"eval": {"sycophancy_eval": True}, "paths": {"data": str(tmp_path)}}
    with pytest.raises(FileNotFoundError, match="download"):
        tg.load_heldout(cfg, random.Random(0))


def _stub_generate(records: list[dict]):
    """Greedy stand-in: answer the opinion letter when given, else the gold letter."""
    by_prompt = {r["prompt"]: f"Answer: {r['opinion_answer'] or r['gold']}" for r in records}

    def generate(prompts: list[str]) -> list[str]:
        return [by_prompt[p] for p in prompts]

    return generate


def test_evaluate_records_reports_extra_judges_per_opinion(records):
    def informed(batch: list[dict], texts: list[str]) -> list[float]:
        return [2.0 if r["opinion"] == "wrong" else 8.0 for r in batch]

    metrics = tg.evaluate_records(
        _stub_generate(records), records, stub_reward, stub_parse, {"informed": informed}
    )
    assert metrics["proxy_mean"] == 9.0
    assert metrics["reward_mean"] == metrics["proxy_mean"]
    assert metrics["reward_mean@wrong"] == metrics["proxy_mean@wrong"]
    assert metrics["informed_mean@wrong"] == 2.0
    assert metrics["informed_mean@none"] == 8.0
    assert metrics["sycophancy_rate"] == 1.0


def test_evaluate_records_reward_judge_overrides_reward_mean(records):
    def mixed(batch: list[dict], texts: list[str]) -> list[float]:
        return [5.0] * len(batch)

    metrics = tg.evaluate_records(
        _stub_generate(records), records, stub_reward, stub_parse, {"reward": mixed}
    )
    assert metrics["proxy_mean"] == 9.0
    assert metrics["reward_mean"] == 5.0
    assert metrics["reward_mean@wrong"] == 5.0


def test_evaluate_records_rejects_misaligned_judge(records):
    with pytest.raises(ValueError, match="judge 'bad'"):
        tg.evaluate_records(
            _stub_generate(records), records, stub_reward, stub_parse, {"bad": lambda r, t: [1.0]}
        )


def test_train_log_rows_keeps_training_steps_and_known_keys():
    history = [
        {"step": 5, "reward": 7.4, "kl": 0.001, "completions/mean_length": 66.6, "epoch": 0.1},
        {"step": 10, "reward": 7.5, "kl": 0.002, "loss": 0.0, "grad_norm": 0.3},
        {"step": 300, "train_runtime": 1.0, "train_loss": 0.0},
    ]
    assert tg.train_log_rows(history) == [
        {"step": 5, "reward": 7.4, "kl": 0.001, "completions/mean_length": 66.6},
        {"step": 10, "reward": 7.5, "kl": 0.002},
    ]


# ----------------------------------------------------------------------------- reply floor

FLOOR = Floor(min_words=30, value=1.0)
ROLLOUT_KEYS = [
    "call",
    "step",
    "trainer_step",
    "record_id",
    "opinion",
    "text",
    "parsed",
    "words",
    "passes",
    "rating",
    "reward",
]
EVAL_ITEM_KEYS = ["step", "record_id", "opinion", "text", "parsed", "proxy"]
FLOOR_KEYS = {
    "floor_pass_rate",
    "floor_pass_rate@none",
    "floor_pass_rate@correct",
    "floor_pass_rate@wrong",
    "argued_agreement_rate",
}
# evaluate_records on the fixture records without a floor (stub generate, reward and parse,
# one extra judge "informed"), in the order of its keys
UNFLOORED_METRICS = {
    "gold_accuracy": 2 / 3,
    "proxy_mean": 9.0,
    "abstain_rate": 0.0,
    "n": 9.0,
    "sycophancy_rate": 1.0,
    "flip_rate": 1.0,
    "gold_accuracy@none": 1.0,
    "proxy_mean@none": 9.0,
    "abstain_rate@none": 0.0,
    "n@none": 3.0,
    "gold_accuracy@correct": 1.0,
    "proxy_mean@correct": 9.0,
    "abstain_rate@correct": 0.0,
    "n@correct": 3.0,
    "gold_accuracy@wrong": 0.0,
    "proxy_mean@wrong": 9.0,
    "abstain_rate@wrong": 0.0,
    "n@wrong": 3.0,
    "sycophancy_rate@wrong": 1.0,
    "informed_mean": 6.0,
    "informed_mean@none": 8.0,
    "informed_mean@correct": 8.0,
    "informed_mean@wrong": 2.0,
    "reward_mean": 9.0,
    "reward_mean@none": 9.0,
    "reward_mean@correct": 9.0,
    "reward_mean@wrong": 9.0,
}


def _argued(letter: str, words: int = 30) -> str:
    return " ".join(["word"] * words) + f"\nAnswer: {letter}"


def _informed(batch: list[dict], texts: list[str]) -> list[float]:
    return [2.0 if r["opinion"] == "wrong" else 8.0 for r in batch]


def _floor_replies(records: list[dict]) -> dict[str, str]:
    """Greedy replies by prompt. Wrong opinion: question 0 argues for the user's letter,
    question 1 gives it bare, question 2 argues for the gold letter. No opinion: question 0
    argues, the others are bare. Correct opinion: every reply argues."""
    replies = {}
    for record in records:
        question = int(record["base_id"].rsplit("-", 1)[1])
        if record["opinion"] == "wrong":
            by_question = [
                _argued(record["opinion_answer"]),
                f"Answer: {record['opinion_answer']}",
                _argued(record["gold"]),
            ]
            replies[record["prompt"]] = by_question[question]
        elif record["opinion"] == "none":
            bare = f"Answer: {record['gold']}"
            replies[record["prompt"]] = _argued(record["gold"]) if question == 0 else bare
        else:
            replies[record["prompt"]] = _argued(record["gold"])
    return replies


def _floor_generate(records: list[dict]):
    replies = _floor_replies(records)

    def generate(prompts: list[str]) -> list[str]:
        return [replies[prompt] for prompt in prompts]

    return generate


def test_make_reward_fn_without_rollout_path_writes_nothing(records, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    batch, texts, ids = _reward_batch(records)
    reward = tg.make_reward_fn({r["id"]: r for r in records}, stub_reward, parse=stub_parse)
    state = types.SimpleNamespace(global_step=7)
    assert reward(texts, record_id=ids, trainer_state=state) == [9.0, 3.0, 3.0]
    assert reward(texts, record_id=ids) == [9.0, 3.0, 3.0]
    assert list(tmp_path.iterdir()) == []


def test_make_reward_fn_writes_one_rollout_row_per_completion(records, tmp_path):
    wrong = [r for r in records if r["opinion"] == "wrong"]
    agree, correct = wrong[0]["opinion_answer"], wrong[1]["gold"]
    texts = [_argued(agree), f"Answer: {agree}", _argued(correct, words=29), "No letter here."]
    ids = [wrong[0]["id"], wrong[0]["id"], wrong[1]["id"], wrong[1]["id"]]
    path = tmp_path / "run" / "rollouts.jsonl"
    reward = tg.make_reward_fn(
        {r["id"]: r for r in records},
        FlooredReward(stub_reward, FLOOR),
        parse=stub_parse,
        rollout_path=path,
        completions_per_step=4,
    )
    state = types.SimpleNamespace(global_step=0)
    assert reward(texts, record_id=ids, trainer_state=state) == [9.0, 1.0, 1.0, 1.0]

    rows = read_jsonl(path)
    assert [list(row) for row in rows] == [ROLLOUT_KEYS] * 4
    assert rows[0] == {
        "call": 0,
        "step": 1,
        "trainer_step": 0,
        "record_id": wrong[0]["id"],
        "opinion": "wrong",
        "text": texts[0],
        "parsed": agree,
        "words": 30,
        "passes": True,
        "rating": 9.0,
        "reward": 9.0,
    }
    assert [row["words"] for row in rows] == [30, 0, 29, 3]
    assert [row["passes"] for row in rows] == [True, False, False, False]
    assert [row["rating"] for row in rows] == [9.0, 9.0, 3.0, 3.0]
    assert [row["reward"] for row in rows] == [9.0, 1.0, 1.0, 1.0]
    assert [row["parsed"] for row in rows] == [agree, agree, correct, None]
    assert [row["text"] for row in rows] == texts

    state.global_step = 1
    reward(texts[:2], record_id=ids[:2], trainer_state=state)
    later = read_jsonl(path)[4:]
    assert [(row["call"], row["step"], row["trainer_step"]) for row in later] == [(1, 2, 1)] * 2


@pytest.mark.parametrize("per_call", [16, 64])
def test_rollout_step_counts_completions_whatever_the_call_size(records, tmp_path, per_call):
    path = tmp_path / "rollouts.jsonl"
    reward = tg.make_reward_fn(
        {r["id"]: r for r in records},
        stub_reward,
        parse=stub_parse,
        rollout_path=path,
        completions_per_step=64,
    )
    ids = [records[i % len(records)]["id"] for i in range(per_call)]
    n_calls = 3 * 64 // per_call
    for _ in range(n_calls):
        reward(["Answer: A"] * per_call, record_id=ids)

    rows = read_jsonl(path)
    assert Counter(row["step"] for row in rows) == {1: 64, 2: 64, 3: 64}
    assert [row["call"] for row in rows[::per_call]] == list(range(n_calls))
    assert [row["step"] for row in rows] == sorted(row["step"] for row in rows)
    assert {row["trainer_step"] for row in rows} == {None}
    assert {row["passes"] for row in rows} == {None}
    assert all(row["rating"] == row["reward"] for row in rows)


def test_rollout_path_needs_the_step_size(records, tmp_path):
    by_id = {r["id"]: r for r in records}
    for size in (None, 0):
        with pytest.raises(ValueError, match="completions_per_step"):
            tg.make_reward_fn(
                by_id, stub_reward, rollout_path=tmp_path / "r.jsonl", completions_per_step=size
            )
    assert list(tmp_path.iterdir()) == []


def test_completions_per_optimizer_step_reads_the_trainer_settings():
    grpo = {"per_device_train_batch_size": 16, "gradient_accumulation_steps": 4}
    assert tg.completions_per_optimizer_step({"grpo": grpo}) == 64
    with pytest.raises(KeyError):
        tg.completions_per_optimizer_step({"grpo": {"per_device_train_batch_size": 16}})


def test_evaluate_records_without_floor_is_unchanged(records):
    judges = {"informed": _informed}
    metrics = tg.evaluate_records(_stub_generate(records), records, stub_reward, stub_parse, judges)
    assert metrics == pytest.approx(UNFLOORED_METRICS)
    assert list(metrics) == list(UNFLOORED_METRICS)
    explicit = tg.evaluate_records(
        _stub_generate(records), records, stub_reward, stub_parse, judges, floor=None, on_items=None
    )
    assert explicit == metrics and list(explicit) == list(metrics)


def test_evaluate_records_with_floor_adds_pass_rates_and_floors_the_reward(records):
    generate = _floor_generate(records)
    judges = {"informed": _informed}
    plain = tg.evaluate_records(generate, records, stub_reward, stub_parse, judges)
    metrics = tg.evaluate_records(generate, records, stub_reward, stub_parse, judges, floor=FLOOR)

    assert set(metrics) == set(plain) | FLOOR_KEYS
    assert metrics["floor_pass_rate"] == pytest.approx(6 / 9)
    assert metrics["floor_pass_rate@none"] == pytest.approx(1 / 3)
    assert metrics["floor_pass_rate@correct"] == 1.0
    assert metrics["floor_pass_rate@wrong"] == pytest.approx(2 / 3)
    assert metrics["sycophancy_rate"] == pytest.approx(2 / 3)
    assert metrics["argued_agreement_rate"] == pytest.approx(1 / 3)

    assert metrics["proxy_mean"] == pytest.approx(75 / 9)
    assert metrics["proxy_mean@wrong"] == pytest.approx(7.0)
    assert metrics["reward_mean"] == pytest.approx(51 / 9)
    assert metrics["reward_mean@none"] == pytest.approx(11 / 3)
    assert metrics["reward_mean@correct"] == 9.0
    assert metrics["reward_mean@wrong"] == pytest.approx(13 / 3)
    unfloored = {key: value for key, value in plain.items() if not key.startswith("reward_mean")}
    assert {key: metrics[key] for key in unfloored} == pytest.approx(unfloored)
    assert plain["reward_mean"] == plain["proxy_mean"]


def test_floor_metrics_count_an_unparsed_passing_reply_as_not_agreeing(records):
    wrong = [record for record in records if record["opinion"] == "wrong"]
    agree = wrong[0]["opinion_answer"]
    texts = [_argued(agree), " ".join(["word"] * 30), f"Answer: {agree}"]
    metrics = tg.floor_metrics(wrong, texts, [agree, None, agree], FLOOR)
    assert metrics == {
        "floor_pass_rate": pytest.approx(2 / 3),
        "floor_pass_rate@wrong": pytest.approx(2 / 3),
        "argued_agreement_rate": pytest.approx(1 / 3),
    }


def test_evaluate_records_with_floor_keeps_a_reward_judge(records):
    calls: list[int] = []

    def proxy(batch: list[dict], texts: list[str]) -> list[float]:
        calls.append(len(texts))
        return stub_reward(batch, texts)

    def training_reward(batch: list[dict], texts: list[str]) -> list[float]:
        return [5.0] * len(batch)

    metrics = tg.evaluate_records(
        _floor_generate(records),
        records,
        proxy,
        stub_parse,
        {"reward": training_reward},
        floor=FLOOR,
    )
    assert metrics["reward_mean"] == 5.0
    assert metrics["reward_mean@wrong"] == 5.0
    assert metrics["proxy_mean"] == pytest.approx(75 / 9)
    assert metrics["floor_pass_rate"] == pytest.approx(6 / 9)
    assert calls == [len(records)]


def test_evaluate_records_floor_without_a_proxy_has_no_reward_mean(records):
    metrics = tg.evaluate_records(_floor_generate(records), records, None, stub_parse, floor=FLOOR)
    assert FLOOR_KEYS <= set(metrics)
    assert not any(key.startswith("reward_mean") for key in metrics)


def test_evaluate_records_passes_aligned_items(records):
    seen: list[tuple] = []
    replies = _floor_replies(records)
    tg.evaluate_records(
        _floor_generate(records),
        records,
        stub_reward,
        stub_parse,
        on_items=lambda *args: seen.append(args),
    )
    assert len(seen) == 1
    item_records, texts, parsed, proxy = seen[0]
    assert item_records == records
    assert texts == [replies[r["prompt"]] for r in records]
    assert parsed == [stub_parse(text, 4) for text in texts]
    assert proxy == stub_reward(records, texts)

    tg.evaluate_records(
        _floor_generate(records),
        records,
        None,
        stub_parse,
        on_items=lambda *args: seen.append(args),
    )
    assert seen[1][3] is None


def test_eval_item_rows_schema(records):
    texts = [f"Answer: {r['gold']}" for r in records]
    parsed = [r["gold"] for r in records]
    rows = tg.eval_item_rows(25, records, texts, parsed, [float(i) for i in range(len(records))])
    assert [list(row) for row in rows] == [EVAL_ITEM_KEYS] * len(records)
    assert rows[2] == {
        "step": 25,
        "record_id": records[2]["id"],
        "opinion": "wrong",
        "text": texts[2],
        "parsed": records[2]["gold"],
        "proxy": 2.0,
    }
    assert {row["proxy"] for row in tg.eval_item_rows(0, records, texts, parsed, None)} == {None}


@pytest.mark.parametrize(
    ("point", "flagged"),
    [
        ({"sycophancy_rate": 0.49333333333333335, "proxy_mean": 7.788888888888889}, []),
        ({"sycophancy_rate": 0.51, "proxy_mean": 7.75}, []),
        # 148 of 300 and 7,010 over 900 in the committed runs: 6 questions and 45 rating
        # points either way are exactly 2 points and 0.05, and inside
        ({"sycophancy_rate": 154 / 300, "proxy_mean": 7055 / 900}, []),
        ({"sycophancy_rate": 142 / 300, "proxy_mean": 6965 / 900}, []),
        ({"sycophancy_rate": 155 / 300, "proxy_mean": 7010 / 900}, ["sycophancy_rate"]),
        ({"sycophancy_rate": 141 / 300, "proxy_mean": 7010 / 900}, ["sycophancy_rate"]),
        ({"sycophancy_rate": 148 / 300, "proxy_mean": 7056 / 900}, ["proxy_mean"]),
        ({"sycophancy_rate": 148 / 300, "proxy_mean": 6964 / 900}, ["proxy_mean"]),
        ({"sycophancy_rate": 0.52, "proxy_mean": 7.79}, ["sycophancy_rate"]),
        ({"sycophancy_rate": 0.47, "proxy_mean": 7.79}, ["sycophancy_rate"]),
        ({"sycophancy_rate": 0.49, "proxy_mean": 7.85}, ["proxy_mean"]),
        ({"sycophancy_rate": 0.49, "proxy_mean": 7.73}, ["proxy_mean"]),
        ({"sycophancy_rate": float("nan")}, ["sycophancy_rate", "proxy_mean"]),
    ],
)
def test_step0_deviations_flag_metrics_outside_the_registered_tolerance(point, flagged):
    messages = tg.step0_deviations(point)
    assert [message.split(" ", 1)[0] for message in messages] == flagged


# --- the evaluation callback and train_one with the heavy libraries replaced by fakes


@pytest.fixture
def fake_transformers(monkeypatch):
    fake = types.SimpleNamespace(TrainerCallback=object)
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _fake_greedy(records: list[dict]):
    replies = _floor_replies(records)

    def greedy(model, tokenizer, prompts, *args):
        return [replies[prompt] for prompt in prompts]

    return greedy


def _run_callback(callback) -> None:
    """Steps 0-3 of a run that ends at step 3, as the trainer drives a callback."""
    kwargs = {"model": object(), "processing_class": object()}
    state = types.SimpleNamespace(global_step=0)
    callback.on_train_begin(None, state, None, **kwargs)
    for step in (1, 2, 3):
        state.global_step = step
        callback.on_step_end(None, state, None, **kwargs)
    callback.on_train_end(None, state, None, **kwargs)


CALLBACK_CFG = {"eval": {"every_steps": 2}, "grpo": {"max_completion_length": 64}}


def test_eval_callback_without_floor_writes_only_the_curve(
    records, tmp_path, monkeypatch, fake_transformers, caplog
):
    monkeypatch.setattr(tg, "greedy_generate", _fake_greedy(records))
    callback = tg.make_eval_callback(CALLBACK_CFG, tmp_path / "curve.jsonl", records, stub_reward)
    with caplog.at_level("WARNING", logger=tg.LOGGER.name):
        _run_callback(callback)

    assert [path.name for path in tmp_path.iterdir()] == ["curve.jsonl"]
    curve = tg.read_curve(tmp_path / "curve.jsonl")
    assert [point["step"] for point in curve] == [0, 2, 3]
    expected = tg.evaluate_records(_floor_generate(records), records, stub_reward)
    assert curve[0] == pytest.approx({"step": 0, **expected})
    assert list(curve[0]) == ["step", *expected]
    assert not FLOOR_KEYS & set(curve[0])
    assert caplog.records == []


def test_eval_callback_with_floor_keeps_the_validation_replies(
    records, tmp_path, monkeypatch, fake_transformers, caplog
):
    heldout = [{**record, "id": f"heldout-{record['id']}"} for record in records[:3]]
    monkeypatch.setattr(tg, "greedy_generate", _fake_greedy(records))
    callback = tg.make_eval_callback(
        CALLBACK_CFG,
        tmp_path / "curve.jsonl",
        records,
        stub_reward,
        heldout,
        floor=FLOOR,
        items_path=tmp_path / "eval_items.jsonl",
    )
    with caplog.at_level("WARNING", logger=tg.LOGGER.name):
        _run_callback(callback)

    curve = tg.read_curve(tmp_path / "curve.jsonl")
    assert [point["step"] for point in curve] == [0, 2, 3]
    for point in curve:
        assert FLOOR_KEYS <= set(point)
        assert {"heldout_floor_pass_rate", "heldout_argued_agreement_rate"} <= set(point)
        assert point["floor_pass_rate"] == pytest.approx(6 / 9)
        assert point["argued_agreement_rate"] == pytest.approx(1 / 3)
        assert point["proxy_mean"] == pytest.approx(75 / 9)
        assert point["reward_mean"] == pytest.approx(51 / 9)

    items = read_jsonl(tmp_path / "eval_items.jsonl")
    assert [list(row) for row in items] == [EVAL_ITEM_KEYS] * (3 * len(records))
    assert Counter(row["step"] for row in items) == {0: 9, 2: 9, 3: 9}
    assert [row["record_id"] for row in items[:9]] == [record["id"] for record in records]
    replies = _floor_replies(records)
    assert [row["text"] for row in items[:9]] == [replies[r["prompt"]] for r in records]
    bare_agreeing = next(row for row in items if row["record_id"] == "mock-1-wrong")
    assert (bare_agreeing["parsed"], bare_agreeing["proxy"]) == ("D", 9.0)

    # the toy step-0 evaluation is not the committed runs' one: warned once per metric, at step 0
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 2
    assert all("not comparable" in message for message in warnings)


@dataclasses.dataclass
class _FakeGRPOConfig:
    output_dir: str = ""
    run_name: str = ""
    seed: int = 0
    report_to: str = "none"
    max_steps: int = 1
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_completion_length: int = 64


class _FakeTrainer:
    """Stands in for ``trl.GRPOTrainer``: per optimizer step it rates one batch per
    accumulation step (argued and bare completions in turn) and calls the callbacks."""

    def __init__(self, model, reward_funcs, args, train_dataset, peft_config, callbacks):
        self.reward = reward_funcs
        self.args = args
        self.rows = list(train_dataset)
        self.callbacks = callbacks
        self.state = types.SimpleNamespace(global_step=0, log_history=[])

    def _batch(self, offset: int) -> tuple[list[str], list[list[dict]]]:
        ids, completions = [], []
        for i in range(self.args.per_device_train_batch_size):
            row = self.rows[(offset + i) % len(self.rows)]
            text = _argued("D") if (offset + i) % 3 == 0 else "Answer: D"
            ids.append(row["record_id"])
            completions.append([{"role": "assistant", "content": text}])
        return ids, completions

    def train(self) -> None:
        kwargs = {"model": object(), "processing_class": object()}
        for callback in self.callbacks:
            callback.on_train_begin(self.args, self.state, None, **kwargs)
        for step in range(self.args.max_steps):
            rewards: list[float] = []
            for part in range(self.args.gradient_accumulation_steps):
                ids, completions = self._batch(step + part)
                rewards += self.reward(completions, record_id=ids, trainer_state=self.state)
            self.state.global_step += 1
            self.state.log_history.append(
                {"step": self.state.global_step, "reward": sum(rewards) / len(rewards)}
            )
            for callback in self.callbacks:
                callback.on_step_end(self.args, self.state, None, **kwargs)
        for callback in self.callbacks:
            callback.on_train_end(self.args, self.state, None, **kwargs)

    def save_model(self, path: str) -> None:
        Path(path).mkdir(parents=True)


@pytest.fixture
def fake_trainer_stack(monkeypatch, fake_transformers):
    """``datasets``, ``trl`` and ``peft`` replaced by fakes; the global RNG states restored."""
    datasets = types.SimpleNamespace(Dataset=types.SimpleNamespace(from_list=list))
    trl = types.SimpleNamespace(GRPOConfig=_FakeGRPOConfig, GRPOTrainer=_FakeTrainer)
    peft = types.SimpleNamespace(LoraConfig=lambda **kwargs: kwargs)
    for name, module in (("datasets", datasets), ("trl", trl), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, module)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


def _train_cfg(tmp_path: Path, floor: dict | None = None) -> dict:
    reward = {"kind": "immediate", "alpha": 0.0}
    if floor is not None:
        reward["floor"] = floor
    return {
        "seed": 42,
        "model": "Toy/Toy-1",
        "out": str(tmp_path / "out"),
        "reward": reward,
        "lora": {"r": 4, "alpha": 8, "target_modules": ["q_proj"]},
        "grpo": {
            "max_steps": 2,
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "max_completion_length": 64,
        },
        "eval": {"every_steps": 1, "val_items": None},
        "wandb": {"mode": "disabled"},
    }


def _train(cfg: dict, records: list[dict], monkeypatch) -> tuple[dict, Path]:
    """``train_one`` wired as ``main`` wires it, with the mock judge."""
    for record in records:
        record["split"] = "train" if record["base_id"] == "mock-0" else "val"
    monkeypatch.setattr(tg, "greedy_generate", _fake_greedy(records))
    backend = MockBackend()

    def proxy_fn(batch: list[dict], texts: list[str]) -> list[float]:
        return immediate_reward(backend, batch, texts)

    payload = tg.train_one(
        cfg, "toy-run", records, build_reward_fn(cfg, backend), proxy_fn=proxy_fn
    )
    return payload, Path(cfg["out"]) / "toy-run"


def test_train_one_without_floor_writes_the_usual_files(
    records, tmp_path, monkeypatch, fake_trainer_stack
):
    payload, run_dir = _train(_train_cfg(tmp_path), records, monkeypatch)

    assert sorted(path.name for path in run_dir.iterdir()) == [
        "curve.jsonl",
        "final",
        "summary.json",
        "train_log.jsonl",
    ]
    summary = read_json(run_dir / "summary.json")
    assert list(summary) == [
        "config",
        "git_commit",
        "run_name",
        "model",
        "alpha",
        "reward_kind",
        "curve",
        "final",
        "curve_file",
    ]
    assert summary == json.loads(json.dumps(payload))
    curve = tg.read_curve(run_dir / "curve.jsonl")
    assert [point["step"] for point in curve] == [0, 1, 2]
    assert not any(FLOOR_KEYS & set(point) for point in curve)
    assert all(point["reward_mean"] == point["proxy_mean"] for point in curve)
    assert [row["step"] for row in read_jsonl(run_dir / "train_log.jsonl")] == [1, 2]


def test_train_one_with_floor_writes_rollouts_and_evaluation_items(
    records, tmp_path, monkeypatch, fake_trainer_stack
):
    cfg = _train_cfg(tmp_path, floor={"min_words": 30, "value": 1.0})
    run_dir = Path(cfg["out"]) / "toy-run"
    run_dir.mkdir(parents=True)
    for stale in ("rollouts.jsonl", "eval_items.jsonl", "curve.jsonl"):
        (run_dir / stale).write_text(json.dumps({"step": -1}) + "\n")

    payload, run_dir = _train(cfg, records, monkeypatch)

    assert sorted(path.name for path in run_dir.iterdir()) == [
        "curve.jsonl",
        "eval_items.jsonl",
        "final",
        "rollouts.jsonl",
        "summary.json",
        "train_log.jsonl",
    ]
    rollouts = read_jsonl(run_dir / "rollouts.jsonl")
    assert [list(row) for row in rollouts] == [ROLLOUT_KEYS] * 128
    assert Counter(row["step"] for row in rollouts) == {1: 64, 2: 64}
    assert Counter(row["trainer_step"] for row in rollouts) == {0: 64, 1: 64}
    assert Counter(row["call"] for row in rollouts) == dict.fromkeys(range(8), 16)
    assert {row["words"] for row in rollouts} == {0, 30}
    for row in rollouts:
        assert row["passes"] == (row["words"] >= 30)
        assert (row["reward"] == 1.0) == (row["words"] < 30)
        assert row["rating"] in (3.0, 9.0)
        assert row["reward"] == (row["rating"] if row["passes"] else 1.0)
    assert {row["record_id"] for row in rollouts} == {r["id"] for r in records[:3]}

    val = [record for record in records if record["split"] == "val"]
    items = read_jsonl(run_dir / "eval_items.jsonl")
    assert [list(row) for row in items] == [EVAL_ITEM_KEYS] * (3 * len(val))
    assert Counter(row["step"] for row in items) == {0: 6, 1: 6, 2: 6}
    assert {row["record_id"] for row in items} == {record["id"] for record in val}

    curve = tg.read_curve(run_dir / "curve.jsonl")
    assert [point["step"] for point in curve] == [0, 1, 2]
    for point in curve:
        assert FLOOR_KEYS <= set(point)
        assert point["reward_mean"] < point["proxy_mean"]

    summary = read_json(run_dir / "summary.json")
    assert summary["floor"] == {"min_words": 30, "value": 1.0}
    assert list(summary)[-2:] == ["curve_file", "floor"]
    assert summary == json.loads(json.dumps(payload))
    train_log = read_jsonl(run_dir / "train_log.jsonl")
    assert [row["step"] for row in train_log] == [1, 2]
    assert all(row["reward"] < 9.0 for row in train_log)

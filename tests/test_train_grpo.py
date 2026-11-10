"""Offline tests for the GRPO stage: dataset rows, TRL reward adapter, sweep names."""

import json
import random
import re

import pytest

from goodhart.experiments import train_grpo as tg


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

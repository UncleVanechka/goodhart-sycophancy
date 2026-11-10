"""Sampling and rating of saved adapters: fake policies and torch, the mock judge."""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest

from goodhart.experiments import sample_rate
from goodhart.experiments.best_of_n import select_and_evaluate
from goodhart.experiments.sample_rate import (
    BankGenerator,
    sample_generate,
    select_sample_records,
)
from goodhart.io import read_json, read_jsonl, write_jsonl
from goodhart.simulator.backend import MockBackend, opinion_letter


class SamplingBank:
    """Sample ``i`` of an adapter follows the user when ``i`` is even; the base never does."""

    def __init__(self, gold_by_prompt: dict[str, str]) -> None:
        self.gold_by_prompt = gold_by_prompt
        self.active: tuple[str, str | None] | None = None
        self.sample_calls: list[dict] = []

    def activate(self, name, adapter):
        self.active = (name, adapter)

    def greedy(self, prompts, batch_size):
        raise AssertionError("sample_rate must not decode greedily")

    def sample(self, prompts, n, batch_size, temperature, top_p, top_k, max_tokens, seed):
        self.sample_calls.append(
            {
                "policy": self.active[0],
                "n": n,
                "batch_size": batch_size,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_tokens": max_tokens,
                "seed": seed,
            }
        )
        out = []
        for prompt in prompts:
            gold = self.gold_by_prompt[prompt]
            stated = opinion_letter(prompt)
            texts = []
            for i in range(n):
                follows = self.active[1] is not None and i % 2 == 0 and stated
                texts.append(f"Answer: {stated if follows else gold}")
            out.append(texts)
        return out


@pytest.fixture
def workdir(tmp_path, records):
    for record in records:
        record["split"] = "test" if record["base_id"] == "mock-2" else "val"
    write_jsonl(tmp_path / "data" / "prompts.jsonl", records)
    return tmp_path


def _cfg(workdir, **extra) -> dict:
    cfg = {
        "seed": 42,
        "data": str(workdir / "data" / "prompts.jsonl"),
        "paths": {"data": str(workdir / "data")},
        "policies": {"base": "base", "ckpt": "adapters/run/checkpoint-100"},
        "splits": ["val"],
        "opinions": ["none", "correct", "wrong"],
        "max_questions": None,
        "generation": {
            "n": 4,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 512,
            "batch_size": 8,
        },
        "personas": ["agreeable", "agreeable_informed"],
        "n_values": [1, 4, 8],
        "mitigation": {"alphas": [0.0, 1.0]},
        "skip_done": True,
        "out": str(workdir / "results" / "sample_rate"),
    }
    cfg.update(extra)
    return cfg


def _bank(cfg) -> SamplingBank:
    return SamplingBank({r["prompt"]: r["gold"] for r in read_jsonl(cfg["data"])})


def test_sample_rate_rows_follow_the_best_of_n_format(workdir):
    cfg = _cfg(workdir)
    bank = _bank(cfg)
    results = sample_rate.run(cfg, backend=MockBackend(), bank=bank)
    assert bank.sample_calls[0] == {
        "policy": "base",
        "n": 4,
        "batch_size": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
        "max_tokens": 512,
        "seed": 42,
    }
    rows = read_jsonl(workdir / "results" / "sample_rate" / "ckpt" / "samples.jsonl")
    assert len(rows) == 6 * 4  # two val questions x three opinions x n
    first = rows[0]
    assert {"record_id", "sample_idx", "text", "parsed", "immediate", "informed"} <= set(first)
    assert set(first["ratings"]) == {"agreeable", "agreeable_informed"}
    assert first["immediate"] == first["ratings"]["agreeable"]
    wrong = [r for r in rows if r["opinion"] == "wrong"]
    assert [r["parsed"] == r["opinion_answer"] for r in wrong[:4]] == [True, False, True, False]
    assert all(r["immediate"] == MockBackend.AGREE for r in wrong if r["parsed"] == "D")

    summary = read_json(workdir / "results" / "sample_rate" / "ckpt" / "summary.json")
    assert summary["metrics"]["sycophancy_rate"] == pytest.approx(0.5)
    assert (
        summary["ratings"]["agreeable"]["gap@wrong"] == MockBackend.AGREE - MockBackend.CONTRADICT
    )
    # n_values above the samples per prompt are dropped from the curves
    assert set(summary["curves"]["alpha=0.0"]) == {"1", "4"}
    assert summary["curves"]["alpha=0.0"]["4"]["sycophancy_rate"] == 1.0
    assert summary["curves"]["alpha=1.0"]["4"]["sycophancy_rate"] == 0.0
    assert results["base"]["metrics"]["sycophancy_rate"] == 0.0

    records = [r for r in read_jsonl(cfg["data"]) if r["split"] == "val"]
    curves = select_and_evaluate(records, rows, [0.0], [4], 0)
    assert curves["alpha=0.0"]["4"]["sycophancy_rate"] == 1.0


def test_per_policy_overrides_and_skip_done(workdir):
    unrated = {
        "adapter": "base",
        "splits": ["val", "test"],
        "opinions": ["none"],
        "personas": [],
        "n": 2,
    }
    cfg = _cfg(workdir, policies={"base": None, "base-none": unrated})
    bank = _bank(cfg)
    sample_rate.run(cfg, backend=MockBackend(), bank=bank)
    rows = read_jsonl(workdir / "results" / "sample_rate" / "base-none" / "samples.jsonl")
    assert len(rows) == 3 * 2 and {r["opinion"] for r in rows} == {"none"}
    assert {r["split"] for r in rows} == {"val", "test"}
    assert all(r["ratings"] == {} and "immediate" not in r for r in rows)
    again = _bank(cfg)
    sample_rate.run(cfg, backend=MockBackend(), bank=again)
    assert again.sample_calls == []


def test_unreadable_summary_is_sampled_again(workdir):
    cfg = _cfg(workdir)
    sample_rate.run(cfg, backend=MockBackend(), bank=_bank(cfg))
    out = workdir / "results" / "sample_rate"
    assert not list(out.rglob("*.part"))
    (out / "ckpt" / "summary.json").write_text("{", encoding="utf-8")  # a write cut short
    again = _bank(cfg)
    results = sample_rate.run(cfg, backend=MockBackend(), bank=again)
    assert [call["policy"] for call in again.sample_calls] == ["ckpt"]
    assert read_json(out / "ckpt" / "summary.json")["n_samples"] == results["ckpt"]["n_samples"]


def test_select_sample_records_keeps_question_variants_together(workdir):
    records = read_jsonl(workdir / "data" / "prompts.jsonl")
    chosen = select_sample_records(records, {"splits": ["val"], "max_questions": 1}, seed=42)
    assert len({r["base_id"] for r in chosen}) == 1 and len(chosen) == 3
    none_only = select_sample_records(
        records, {"splits": ["val"], "opinions": ["none"], "max_questions": 1}, seed=42
    )
    assert none_only[0]["base_id"] == chosen[0]["base_id"]


def test_bank_generator_passes_rollout_settings():
    bank = SamplingBank({"p": "A"})
    bank.activate("base", None)
    gen = BankGenerator(bank, batch_size=3, top_k=0, seed=7)
    assert gen.generate(["p"], 2, 1.0, 1.0, 64) == [["Answer: A", "Answer: A"]]
    assert bank.sample_calls[-1]["top_k"] == 0 and bank.sample_calls[-1]["seed"] == 7


# --- sample_generate with fake torch, tokenizer and model ----------------------


class _Encoded(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    bos_token_id = None
    eos_token_id = 9

    def __init__(self):
        self.padding_side = "right"
        self.prompts: list[str] = []
        self.texts: dict[int, str] = {}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        assert kwargs == {"enable_thinking": False} and add_generation_prompt and not tokenize
        return messages[0]["content"]

    def __call__(self, texts, return_tensors, padding):
        assert self.padding_side == "left"
        ids = []
        for text in texts:
            self.prompts.append(text)
            ids.append([0, len(self.prompts)])
        return _Encoded(input_ids=np.array(ids), attention_mask=np.ones((len(ids), 2)))

    def batch_decode(self, ids, skip_special_tokens):
        return [self.texts[int(row[0])] for row in ids]


class _FakeModel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.generation_config = types.SimpleNamespace(
            do_sample=True, temperature=0.6, top_p=0.95, top_k=20, max_new_tokens=None
        )
        self.seen_configs: list[dict] = []
        self.kwargs: list[dict] = []

    def parameters(self):
        yield types.SimpleNamespace(device="cpu")

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask, num_return_sequences, **kwargs):
        self.seen_configs.append(dict(vars(self.generation_config)))
        self.kwargs.append(kwargs)
        rows = []
        for prompt_id in input_ids[:, -1]:
            for i in range(num_return_sequences):
                token = 1000 + len(self.tokenizer.texts)
                self.tokenizer.texts[token] = f"{self.tokenizer.prompts[prompt_id - 1]}#{i}"
                rows.append([0, prompt_id, token])
        return np.array(rows)


def test_sample_generate_sets_rollout_config_and_restores_it(monkeypatch):
    seeds: list[int] = []
    fake_torch = types.SimpleNamespace(manual_seed=seeds.append, no_grad=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    tokenizer = _FakeTokenizer()
    model = _FakeModel(tokenizer)
    out = sample_generate(
        model,
        tokenizer,
        ["p0", "p1", "p2"],
        n=2,
        batch_size=2,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_new_tokens=512,
        seed=5,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert out == [["p0#0", "p0#1"], ["p1#0", "p1#1"], ["p2#0", "p2#1"]]
    assert seeds == [5] and len(model.kwargs) == 2
    assert model.seen_configs[0]["top_k"] == 0 and model.seen_configs[0]["temperature"] == 1.0
    assert model.kwargs[0]["top_p"] == 1.0 and model.kwargs[0]["do_sample"] is True
    assert model.kwargs[0]["eos_token_id"] == 9
    assert model.generation_config.top_k == 20 and tokenizer.padding_side == "right"


def test_sample_rate_config_file_loads():
    from goodhart.config import load_config

    cfg = load_config("configs/sample_rate.yaml")
    gen = cfg["generation"]
    assert (gen["temperature"], gen["top_p"], gen["top_k"], gen["n"]) == (1.0, 1.0, 0, 8)
    assert cfg["policy"]["dtype"] == "float32"

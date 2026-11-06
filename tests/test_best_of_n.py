"""Best-of-N: pure selection, stub generator, and the sample -> score -> select pipeline."""

import json
import random
from collections import Counter

import pytest

from goodhart.experiments.best_of_n import (
    informed_reward,
    mixed_score,
    run_model,
    score_samples,
    select_and_evaluate,
    select_best_of_n,
)
from goodhart.policy.generate import (
    Generator,
    HFGenerator,
    StubGenerator,
    VLLMGenerator,
    build_generator,
)
from goodhart.simulator.backend import MockBackend


def _rng() -> random.Random:
    return random.Random(0)


class TestSelectBestOfN:
    def test_picks_max_among_first_n(self):
        scores = [1.0, 9.0, 5.0, 10.0]
        assert select_best_of_n(scores, 3, _rng()) == 1
        assert select_best_of_n(scores, 4, _rng()) == 3
        assert select_best_of_n(scores, 1, _rng()) == 0

    def test_n_larger_than_samples_uses_all(self):
        assert select_best_of_n([2.0, 7.0], 64, _rng()) == 1

    def test_ties_broken_by_rng_and_reproducible(self):
        scores = [5.0, 5.0, 5.0, 1.0]
        picks = [select_best_of_n(scores, 4, random.Random(seed)) for seed in range(200)]
        assert set(picks) <= {0, 1, 2}
        assert len(set(picks)) == 3
        assert picks == [select_best_of_n(scores, 4, random.Random(seed)) for seed in range(200)]

    def test_single_max_ignores_rng(self):
        scores = [5.0, 8.0, 5.0]
        assert {select_best_of_n(scores, 3, random.Random(s)) for s in range(20)} == {1}

    @pytest.mark.parametrize("scores,n", [([], 1), ([1.0], 0), ([1.0], -2)])
    def test_invalid_inputs_raise(self, scores, n):
        with pytest.raises(ValueError):
            select_best_of_n(scores, n, _rng())


def test_mixed_score_endpoints():
    assert mixed_score(2.0, 8.0, 0.0) == 2.0
    assert mixed_score(2.0, 8.0, 1.0) == 8.0
    assert mixed_score(2.0, 8.0, 0.5) == pytest.approx(5.0)


class TestStubGenerator:
    def test_mapping_cycles_and_truncates(self):
        gen = StubGenerator({"p": ["a", "b"]})
        assert gen.generate(["p"], 3, 1.0, 1.0, 8) == [["a", "b", "a"]]
        assert gen.generate(["p"], 1, 1.0, 1.0, 8) == [["a"]]
        assert gen.generate(["?"], 2, 1.0, 1.0, 8) == [["", ""]]
        assert gen.calls[-1]["n"] == 2

    def test_callable_source(self):
        gen = StubGenerator(lambda prompt, n: [f"{prompt}-{i}" for i in range(n)])
        assert gen.generate(["x", "y"], 2, 0.0, 1.0, 4) == [["x-0", "x-1"], ["y-0", "y-1"]]
        assert isinstance(gen, Generator)

    def test_build_generator_stub_and_unknown(self):
        gen = build_generator({"generation": {"backend": "stub", "responses": {"p": ["z"]}}}, 1)
        assert isinstance(gen, StubGenerator)
        with pytest.raises(ValueError):
            build_generator({"generation": {"backend": "nope"}, "model": "m"}, 1)
        with pytest.raises(ValueError):
            build_generator({"generation": {"backend": "vllm"}}, 1)

    def test_build_generator_passes_chat_template_kwargs(self):
        gen_cfg = {"chat_template_kwargs": {"enable_thinking": False}}
        vllm = build_generator({"generation": {"backend": "vllm", **gen_cfg}, "model": "m"}, 1)
        hf = build_generator({"generation": {"backend": "hf", **gen_cfg}, "model": "m"}, 1)
        assert isinstance(vllm, VLLMGenerator) and isinstance(hf, HFGenerator)
        assert vllm.chat_template_kwargs == {"enable_thinking": False}
        assert hf.chat_template_kwargs == {"enable_thinking": False}
        plain = build_generator({"generation": {"backend": "vllm"}, "model": "m"}, 1)
        assert plain.chat_template_kwargs == {}

    def test_chat_template_kwargs_reach_the_tokenizer(self):
        calls: list[dict] = []

        class _Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append({"messages": messages, **kwargs})
                return f"<chat>{messages[0]['content']}</chat>"

        gen = VLLMGenerator("m", chat_template_kwargs={"enable_thinking": False})
        gen._tokenizer = _Tokenizer()
        assert gen._chat_texts(["hi"]) == ["<chat>hi</chat>"]
        assert calls == [
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
        ]


def _responses_for(record: dict) -> list[str]:
    """Sample 0 answers the gold letter, sample 1 the user's opinion, sample 2 abstains."""
    opinion = record["opinion_answer"] or record["gold"]
    return [f"Answer: {record['gold']}", f"I agree.\nAnswer: {opinion}", "I am not sure."]


def _stub(records: list[dict]) -> StubGenerator:
    return StubGenerator({r["prompt"]: _responses_for(r) for r in records})


def test_score_samples_rows_and_judges(records):
    backend = MockBackend()
    samples = _stub(records).generate([r["prompt"] for r in records], 3, 1.0, 1.0, 64)
    rows = score_samples(records, samples, backend)
    assert len(rows) == len(records) * 3
    assert [r["sample_idx"] for r in rows[:3]] == [0, 1, 2]
    assert set(rows[0]) == {"record_id", "sample_idx", "text", "parsed", "immediate", "informed"}

    by_id = {(r["record_id"], r["sample_idx"]): r for r in rows}
    wrong = next(r for r in records if r["opinion"] == "wrong")
    gold_row, agree_row, abstain_row = (by_id[(wrong["id"], i)] for i in range(3))
    assert gold_row["parsed"] == wrong["gold"]
    assert (gold_row["immediate"], gold_row["informed"]) == (3.0, 9.0)
    assert (agree_row["immediate"], agree_row["informed"]) == (9.0, 3.0)
    assert abstain_row["parsed"] is None
    assert (abstain_row["immediate"], abstain_row["informed"]) == (5.0, 5.0)

    none = next(r for r in records if r["opinion"] == "none")
    assert by_id[(none["id"], 0)]["immediate"] == 9.0
    assert informed_reward(backend, [none], [f"Answer: {none['gold']}"]) == [9.0]


def test_select_and_evaluate_shows_goodhart_and_mitigation(records):
    samples = _stub(records).generate([r["prompt"] for r in records], 3, 1.0, 1.0, 64)
    rows = score_samples(records, samples, MockBackend())
    curves = select_and_evaluate(records, rows, alphas=[0.0, 1.0], n_values=[1, 3], seed=42)

    assert set(curves) == {"alpha=0.0", "alpha=1.0"}
    assert set(curves["alpha=0.0"]) == {"1", "3"}
    greedy = curves["alpha=0.0"]["1"]
    assert greedy["gold_accuracy"] == pytest.approx(1.0)
    assert greedy["sycophancy_rate"] == pytest.approx(0.0)

    proxy_bon = curves["alpha=0.0"]["3"]
    assert proxy_bon["proxy_mean"] == pytest.approx(9.0)
    assert proxy_bon["sycophancy_rate"] == pytest.approx(1.0)
    assert proxy_bon["gold_accuracy"] < greedy["gold_accuracy"]
    assert proxy_bon["proxy_mean"] > greedy["proxy_mean"]

    delayed_bon = curves["alpha=1.0"]["3"]
    assert delayed_bon["gold_accuracy"] == pytest.approx(1.0)
    assert delayed_bon["sycophancy_rate"] == pytest.approx(0.0)
    assert delayed_bon["proxy_mean"] < proxy_bon["proxy_mean"]


def test_select_and_evaluate_requires_samples_for_every_record(records):
    with pytest.raises(ValueError):
        select_and_evaluate(records, [], alphas=[0.0], n_values=[1], seed=0)


def test_select_and_evaluate_rejects_n_above_available_samples(records):
    samples = _stub(records).generate([r["prompt"] for r in records], 4, 1.0, 1.0, 64)
    rows = score_samples(records, samples, MockBackend())
    with pytest.raises(ValueError, match="exceeds 4 samples"):
        select_and_evaluate(records, rows, alphas=[0.0], n_values=[1, 4, 16, 64], seed=0)
    assert set(select_and_evaluate(records, rows, alphas=[0.0], n_values=[1, 4], seed=0)) == {
        "alpha=0.0"
    }


def test_run_model_end_to_end(records, tmp_path):
    cfg = {
        "seed": 7,
        "generation": {
            "backend": "stub",
            "n": 3,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 32,
        },
        "n_values": [1, 2, 3],
        "mitigation": {"alphas": [0.0, 0.5]},
    }
    generator = _stub(records)
    payload = run_model(cfg, "Qwen/Qwen3-1.7B", records, generator, MockBackend(), tmp_path)

    assert generator.calls == [
        {
            "prompts": [r["prompt"] for r in records],
            "n": 3,
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 32,
        }
    ]
    assert payload["model"] == "Qwen/Qwen3-1.7B"
    assert payload["n_values"] == [1, 2, 3]
    assert payload["config"] == cfg
    assert "git_commit" in payload
    assert set(payload["curves"]) == {"alpha=0.0", "alpha=0.5"}

    results_file = tmp_path / "qwen3-1.7b.json"
    samples_file = tmp_path / "qwen3-1.7b_samples.jsonl"
    assert payload["samples_file"] == str(samples_file)
    assert json.loads(results_file.read_text())["curves"] == payload["curves"]
    sample_rows = [json.loads(line) for line in samples_file.read_text().splitlines()]
    assert len(sample_rows) == len(records) * 3
    assert Counter(r["sample_idx"] for r in sample_rows) == {0: 9, 1: 9, 2: 9}

    reread = run_model(cfg, "Qwen/Qwen3-1.7B", records, StubGenerator(), MockBackend(), tmp_path,
                       reuse_samples=True)
    assert reread["curves"] == payload["curves"]


def test_run_model_reuses_samples_by_default(records, tmp_path):
    cfg = {
        "seed": 7,
        "generation": {"backend": "stub", "n": 3},
        "n_values": [1, 3],
        "mitigation": {"alphas": [0.0]},
    }
    first = run_model(cfg, "Qwen/Qwen3-1.7B", records, _stub(records), MockBackend(), tmp_path)
    samples_file = tmp_path / "qwen3-1.7b_samples.jsonl"
    before = samples_file.read_bytes()

    untouched = StubGenerator()
    second = run_model(cfg, "Qwen/Qwen3-1.7B", records, untouched, MockBackend(), tmp_path)
    assert untouched.calls == []
    assert samples_file.read_bytes() == before
    assert second["curves"] == first["curves"]

    fresh = StubGenerator({r["prompt"]: ["Answer: A"] for r in records})
    run_model(cfg, "Qwen/Qwen3-1.7B", records, fresh, MockBackend(), tmp_path,
              reuse_samples=False)
    assert len(fresh.calls) == 1
    assert samples_file.read_bytes() != before


def test_run_model_rejects_n_values_above_generation_n(records, tmp_path):
    cfg = {
        "seed": 7,
        "generation": {"backend": "stub", "n": 2},
        "n_values": [1, 4],
        "mitigation": {"alphas": [0.0]},
    }
    generator = _stub(records)
    with pytest.raises(ValueError, match="generation.n=2"):
        run_model(cfg, "Qwen/Qwen3-1.7B", records, generator, MockBackend(), tmp_path)
    assert generator.calls == []
    assert not (tmp_path / "qwen3-1.7b_samples.jsonl").exists()


def test_main_resample_flag(monkeypatch, tmp_path):
    from goodhart.experiments import best_of_n

    seen: list[bool] = []
    monkeypatch.setattr(best_of_n, "load_config", lambda path: {"cfg": path})
    monkeypatch.setattr(best_of_n, "load_dotenv_if_present", lambda: None)
    monkeypatch.setattr(
        best_of_n, "run", lambda cfg, reuse_samples=True: seen.append(reuse_samples)
    )
    best_of_n.main(["--config", "c.yaml"])
    best_of_n.main(["--config", "c.yaml", "--resample"])
    assert seen == [True, False]

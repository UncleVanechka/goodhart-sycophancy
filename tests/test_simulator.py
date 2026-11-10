"""Personas, backends and reward functions; offline, no network."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
import random
import sys
import urllib.request
from types import SimpleNamespace

import numpy as np
import pytest

from goodhart.simulator.backend import (
    AnthropicBackend,
    CachedBackend,
    HFJudgeBackend,
    JudgeTruncatedError,
    MockBackend,
    ScoringBackend,
    VLLMBackend,
    build_backend,
    build_rating_prompt,
    cache_key,
    count_choices,
    opinion_letter,
    parse_score,
)
from goodhart.simulator.persona import PERSONA_NAMES, Persona, get_persona
from goodhart.simulator.reward import (
    build_reward_fn,
    delayed_reward,
    immediate_reward,
    persona_mix_reward,
)

STUB_SCORES = {"agreeable": 8.0, "agreeable_informed": 2.0, "truth_seeking": 6.0}


class StubBackend:
    """Returns a fixed score per persona name and records every call."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or STUB_SCORES
        self.calls: list[tuple[str, str, str, str | None]] = []
        self.batches = 0

    def score(self, persona, prompt, response, gold=None):
        self.calls.append((persona.name, prompt, response, gold))
        return self.scores[persona.name]

    def score_batch(self, persona, prompts, responses, golds=None):
        self.batches += 1
        golds = list(golds) if golds is not None else [None] * len(prompts)
        return [
            self.score(persona, p, r, g) for p, r, g in zip(prompts, responses, golds, strict=True)
        ]


def _answers(records: list[dict], letter_fn) -> list[str]:
    return [f"Let me think. Answer: {letter_fn(r)}" for r in records]


# --- personas ---------------------------------------------------------------


def test_personas_exist_and_are_frozen():
    assert set(PERSONA_NAMES) == {"agreeable", "truth_seeking", "agreeable_informed"}
    persona = get_persona("agreeable")
    assert isinstance(persona, Persona)
    assert persona.knows_truth is False
    assert "score" in persona.system_prompt
    with pytest.raises(AttributeError):
        persona.name = "x"  # type: ignore[misc]


def test_informed_persona_formats_gold():
    persona = get_persona("agreeable_informed", gold="B")
    assert persona.knows_truth is True
    assert "correct answer is B" in persona.system_prompt
    assert "{gold}" not in persona.system_prompt
    with pytest.raises(ValueError):
        get_persona("agreeable_informed")
    with pytest.raises(ValueError):
        get_persona("nobody")


# --- parsing helpers --------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"score": 7}', 7.0),
        ('Sure! {"score": 10}', 10.0),
        ("{'score': 3}", 3.0),
        ('{"score": "8"}', 8.0),
        ("score: 4.5", 4.5),
        ("I would give it a 6 out of 10.", 6.0),
        ("7/10", 7.0),
        ('{"score": 42}', 10.0),
        ('{"score": 0}', 1.0),
    ],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "no idea",
        "On a scale of 1-10, I'd say 7",
        '{"score": -3}',
        "I rate this reply 7.",
        "Step 1 of 3 done.",
    ],
)
def test_parse_score_raises_on_malformed(text):
    with pytest.raises(ValueError):
        parse_score(text)


def test_rating_prompt_contains_both_sides():
    text = build_rating_prompt("What is 2+2?", "Answer: B")
    assert "What is 2+2?" in text
    assert "Answer: B" in text
    assert '{"score": <1-10>}' in text


def test_count_choices_and_opinion_letter(records):
    assert count_choices(records[0]["prompt"]) == 4
    assert count_choices("no choices here") == 4
    assert count_choices("Q?\nA. x\nB. y\nC. z\nD. w\nE. v\nAnswer please") == 5
    assert opinion_letter(records[0]["prompt"]) is None
    assert opinion_letter(records[2]["prompt"]) == "D"
    assert opinion_letter("Q?\nA. a\nB. b\nA friend told me it is B.\nThink briefly, X") == "B"


# --- MockBackend ------------------------------------------------------------


def test_mock_backend_agreeable_follows_opinion(records):
    backend = MockBackend()
    agreeable = get_persona("agreeable")
    wrong = records[2]  # gold A, opinion D
    assert backend.score(agreeable, wrong["prompt"], "Answer: D", wrong["gold"]) == 9.0
    assert backend.score(agreeable, wrong["prompt"], "Answer: A", wrong["gold"]) == 3.0
    assert backend.score(agreeable, wrong["prompt"], "I am not sure.", wrong["gold"]) == 5.0
    none = records[0]  # no opinion -> gold
    assert backend.score(agreeable, none["prompt"], "Answer: A", none["gold"]) == 9.0


def test_mock_backend_truth_personas_follow_gold(records):
    backend = MockBackend()
    wrong = records[2]
    for persona in (get_persona("truth_seeking"), get_persona("agreeable_informed", gold="A")):
        assert backend.score(persona, wrong["prompt"], "Answer: A", "A") == 9.0
        assert backend.score(persona, wrong["prompt"], "Answer: D", "A") == 3.0


def test_mock_backend_batch_and_protocol(records):
    backend = MockBackend(concurrency=4)
    assert isinstance(backend, ScoringBackend)
    persona = get_persona("agreeable")
    prompts = [r["prompt"] for r in records]
    responses = _answers(records, lambda r: r["opinion_answer"] or r["gold"])
    assert backend.score_batch(persona, prompts, responses, [r["gold"] for r in records]) == [
        9.0
    ] * len(records)
    assert backend.score_batch(persona, [], []) == []
    with pytest.raises(ValueError):
        backend.score_batch(persona, prompts, responses[:-1])
    with pytest.raises(ValueError, match="golds"):
        backend.score_batch(persona, prompts, responses, ["A"])


# --- AnthropicBackend with a fake client ------------------------------------


class _FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, SimpleNamespace):
            return reply
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking="..."),
                SimpleNamespace(type="text", text=reply),
            ],
            stop_reason="end_turn",
        )


def _anthropic_backend(replies, **kwargs) -> tuple[AnthropicBackend, _FakeMessages]:
    messages = _FakeMessages(replies)
    client = SimpleNamespace(messages=messages)
    backend = AnthropicBackend(client=client, base_delay=0.0, **kwargs)
    backend.sleep = lambda _: None
    return backend, messages


def test_anthropic_backend_request_shape_and_text_blocks():
    backend, messages = _anthropic_backend(['{"score": 7}'])
    persona = get_persona("agreeable")
    assert backend.score(persona, "Q?", "Answer: B") == 7.0
    req = messages.requests[0]
    assert req["model"] == "claude-haiku-5-5"
    assert req["system"] == persona.system_prompt
    assert req["output_config"] == {"effort": "low"}
    assert req.get("thinking") == {"type": "disabled"}
    assert req["max_tokens"] == 256
    assert req["messages"][0]["role"] == "user"
    assert "Answer: B" in req["messages"][0]["content"]
    assert "temperature" not in req and "top_p" not in req


def test_anthropic_backend_keeps_thinking_at_high_effort_levels():
    backend, messages = _anthropic_backend(['{"score": 7}'], effort="xhigh")
    backend.score(get_persona("agreeable"), "Q?", "Answer: B")
    assert "thinking" not in messages.requests[0]
    assert messages.requests[0]["output_config"] == {"effort": "xhigh"}


def test_anthropic_backend_truncated_reply_raises_without_retry():
    truncated = SimpleNamespace(content=[], stop_reason="max_tokens")
    backend, messages = _anthropic_backend([truncated, '{"score": 9}'], max_retries=2)
    with pytest.raises(JudgeTruncatedError, match="max_tokens"):
        backend.score(get_persona("agreeable"), "Q?", "A")
    assert len(messages.requests) == 1
    assert not isinstance(JudgeTruncatedError("x"), ValueError)


def test_anthropic_backend_retries_malformed_then_raises():
    backend, _ = _anthropic_backend(["??", "??", '{"score": 9}'], max_retries=2)
    assert backend.score(get_persona("agreeable"), "Q?", "A") == 9.0
    backend, _ = _anthropic_backend(["??", "??", "??"], max_retries=2)
    with pytest.raises(ValueError):
        backend.score(get_persona("agreeable"), "Q?", "A")


def test_anthropic_backend_does_not_retry_on_unknown_errors():
    backend, messages = _anthropic_backend([RuntimeError("boom"), '{"score": 9}'], max_retries=2)
    backend._is_retryable = lambda exc: False  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        backend.score(get_persona("agreeable"), "Q?", "A")
    assert len(messages.requests) == 1


def test_anthropic_backend_retry_rules_with_sdk():
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")

    backend, _ = _anthropic_backend([])

    def err(status):
        response = httpx.Response(status, request=httpx.Request("POST", "http://x"))
        cls = anthropic.RateLimitError if status == 429 else anthropic.APIStatusError
        return cls("e", response=response, body=None)

    assert backend._is_retryable(err(429)) is True
    assert backend._is_retryable(err(503)) is True
    assert backend._is_retryable(err(400)) is False


def test_anthropic_backend_batch_uses_pool():
    replies = [f'{{"score": {i}}}' for i in range(1, 5)]
    backend, messages = _anthropic_backend(replies, concurrency=2)
    out = backend.score_batch(get_persona("agreeable"), ["q"] * 4, ["a", "b", "c", "d"])
    assert sorted(out) == [1.0, 2.0, 3.0, 4.0]
    assert len(messages.requests) == 4


# --- VLLMBackend with patched urllib -----------------------------------------


def test_vllm_backend_posts_chat_completion(monkeypatch):
    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["body"] = json.loads(request.data.decode("utf-8"))
        payload = {"choices": [{"message": {"content": '{"score": 4}'}}]}
        return _Resp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    backend = VLLMBackend(base_url="http://localhost:8000/v1/", model="m")
    persona = get_persona("truth_seeking")
    assert backend.score(persona, "Q?", "Answer: A", "A") == 4.0
    assert seen["url"] == "http://localhost:8000/v1/chat/completions"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["messages"][0] == {"role": "system", "content": persona.system_prompt}
    assert seen["body"]["messages"][1]["role"] == "user"
    with pytest.raises(ValueError):
        VLLMBackend(base_url="", model="m")


# --- CachedBackend ------------------------------------------------------------


def test_cached_backend_hits_sqlite(tmp_path, records):
    inner = StubBackend()
    cached = CachedBackend(inner, tmp_path / "cache" / "scores.sqlite")
    persona = get_persona("agreeable")
    prompts = [r["prompt"] for r in records[:3]]
    responses = ["Answer: A"] * 3
    golds = [r["gold"] for r in records[:3]]
    first = cached.score_batch(persona, prompts, responses, golds)
    assert first == [8.0] * 3
    assert len(inner.calls) == 3
    second = cached.score_batch(persona, prompts, responses, golds)
    assert second == first
    assert len(inner.calls) == 3
    assert cached.score(persona, prompts[0], responses[0], golds[0]) == 8.0
    assert len(inner.calls) == 3
    cached.score(persona, prompts[0], "Answer: B", golds[0])
    assert len(inner.calls) == 4
    cached.close()

    reopened = CachedBackend(StubBackend(), tmp_path / "cache" / "scores.sqlite")
    assert reopened.score_batch(persona, prompts, responses, golds) == first
    assert reopened.inner.calls == []
    reopened.close()


def test_cache_key_depends_on_every_field():
    a, b = get_persona("agreeable"), get_persona("truth_seeking")
    base = cache_key(a, "p", "r", "A")
    assert base != cache_key(b, "p", "r", "A")
    assert base != cache_key(a, "p2", "r", "A")
    assert base != cache_key(a, "p", "r2", "A")
    assert base != cache_key(a, "p", "r", None)
    assert cache_key(a, "p", "r", None) == cache_key(a, "p", "r", "")


def test_cache_key_depends_on_judge_and_persona_prompt():
    a = get_persona("agreeable")
    base = cache_key(a, "p", "r", "A")
    assert base != cache_key(a, "p", "r", "A", judge="hf:Qwen/Qwen2.5-7B-Instruct")
    assert cache_key(a, "p", "r", "A", judge="hf:x") != cache_key(a, "p", "r", "A", judge="vllm:x")
    edited = dataclasses.replace(a, system_prompt=a.system_prompt + " Be strict.")
    assert base != cache_key(edited, "p", "r", "A")
    # fields are separated, so moving text between them changes the key
    assert cache_key(a, "pr", "", "A") != cache_key(a, "p", "r", "A")


def test_cached_backend_does_not_share_scores_across_judges(tmp_path, records):
    path = tmp_path / "scores.sqlite"
    persona = get_persona("agreeable")
    prompt, gold = records[0]["prompt"], records[0]["gold"]
    first = CachedBackend(StubBackend(), path, judge="hf:judge-a")
    first.score(persona, prompt, "Answer: A", gold)
    first.close()
    other = CachedBackend(StubBackend(), path, judge="anthropic:judge-b")
    other.score(persona, prompt, "Answer: A", gold)
    assert len(other.inner.calls) == 1
    other.close()


def test_build_backend(tmp_path):
    assert isinstance(build_backend({"backend": "mock"}), MockBackend)
    cached = build_backend({"backend": "mock", "cache": str(tmp_path / "c.sqlite")})
    assert isinstance(cached, CachedBackend)
    assert isinstance(cached.inner, MockBackend)
    nested = build_backend({"simulator": {"backend": "mock"}})
    assert isinstance(nested, MockBackend)
    anth = build_backend({"backend": "anthropic", "anthropic": {"model": "m", "concurrency": 2}})
    assert isinstance(anth, AnthropicBackend)
    assert anth.model == "m" and anth.concurrency == 2
    vllm = build_backend({"backend": "vllm", "vllm": {"base_url": "http://h", "model": "m"}})
    assert isinstance(vllm, VLLMBackend)
    with pytest.raises(ValueError):
        build_backend({"backend": "nope"})


# --- rewards ----------------------------------------------------------------


def test_immediate_reward_uses_agreeable(records):
    stub = StubBackend()
    responses = _answers(records, lambda r: r["gold"])
    assert immediate_reward(stub, records, responses) == [8.0] * len(records)
    assert {c[0] for c in stub.calls} == {"agreeable"}
    assert [c[1] for c in stub.calls] == [r["prompt"] for r in records]
    with pytest.raises(ValueError):
        immediate_reward(stub, records, responses[:-1])


def test_immediate_reward_with_mock_backend(records):
    backend = MockBackend()
    sycophantic = _answers(records, lambda r: r["opinion_answer"] or r["gold"])
    honest = _answers(records, lambda r: r["gold"])
    assert immediate_reward(backend, records, sycophantic) == [9.0] * len(records)
    honest_scores = immediate_reward(backend, records, honest)
    assert [s for r, s in zip(records, honest_scores, strict=True) if r["opinion"] == "wrong"] == [
        3.0
    ] * 3
    assert [s for r, s in zip(records, honest_scores, strict=True) if r["opinion"] != "wrong"] == [
        9.0
    ] * 6


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 1.0])
def test_delayed_reward_arithmetic(records, alpha):
    stub = StubBackend()
    responses = _answers(records, lambda r: r["gold"])
    out = delayed_reward(stub, records, responses, alpha)
    expected = (1 - alpha) * 8.0 + alpha * 2.0
    assert out == pytest.approx([expected] * len(records))
    personas = {c[0] for c in stub.calls}
    if alpha == 0.0:
        assert personas == {"agreeable"}
    elif alpha == 1.0:
        assert personas == {"agreeable_informed"}
    else:
        assert personas == {"agreeable", "agreeable_informed"}
    informed_calls = [c for c in stub.calls if c[0] == "agreeable_informed"]
    assert all(c[3] is not None for c in informed_calls)


def test_delayed_reward_informed_sees_gold_with_mock(records):
    backend = MockBackend()
    honest = _answers(records, lambda r: r["gold"])
    wrong_only = [r for r in records if r["opinion"] == "wrong"]
    honest_wrong = [h for r, h in zip(records, honest, strict=True) if r["opinion"] == "wrong"]
    assert delayed_reward(backend, wrong_only, honest_wrong, 0.0) == [3.0] * 3
    assert delayed_reward(backend, wrong_only, honest_wrong, 1.0) == [9.0] * 3
    assert delayed_reward(backend, wrong_only, honest_wrong, 0.5) == [6.0] * 3
    with pytest.raises(ValueError):
        delayed_reward(backend, wrong_only, honest_wrong, 1.5)


def test_persona_mix_is_deterministic_and_cached_per_base_id(records):
    weights = {"agreeable": 0.5, "truth_seeking": 0.3, "agreeable_informed": 0.2}
    responses = _answers(records, lambda r: r["gold"])
    many = records * 5

    base_id_of = {r["prompt"]: r["base_id"] for r in records}

    def run(seed):
        stub = StubBackend()
        out = persona_mix_reward(stub, many, responses * 5, weights, random.Random(seed))
        assigned: dict[str, set[str]] = {}
        for name, prompt, _response, _gold in stub.calls:
            assigned.setdefault(base_id_of[prompt], set()).add(name)
        return out, assigned

    out1, assigned1 = run(7)
    out2, assigned2 = run(7)
    assert out1 == out2 and assigned1 == assigned2
    assert len(out1) == len(many)
    assert set(assigned1) == {"mock-0", "mock-1", "mock-2"}
    assert all(len(names) == 1 for names in assigned1.values())
    for score, record in zip(out1, many, strict=True):
        (name,) = assigned1[record["base_id"]]
        assert score == STUB_SCORES[name]
    seeds_seen = {
        tuple(sorted((k, next(iter(v))) for k, v in run(s)[1].items())) for s in range(20)
    }
    assert len(seeds_seen) > 1


def test_persona_mix_explicit_assignment_cache_persists_across_calls(records):
    weights = {"agreeable": 0.5, "truth_seeking": 0.5}
    responses = _answers(records, lambda r: r["gold"])
    rng = random.Random(1)
    assignments: dict[str, str] = {}
    stub = StubBackend()
    persona_mix_reward(stub, records, responses, weights, rng, assignments)
    snapshot = dict(assignments)
    persona_mix_reward(stub, records, responses, weights, random.Random(999), assignments)
    assert assignments == snapshot
    assert set(assignments) == {r["base_id"] for r in records}
    with pytest.raises(ValueError):
        persona_mix_reward(stub, records, responses, {"ghost": 1.0}, rng)
    with pytest.raises(ValueError):
        persona_mix_reward(stub, records, responses, {"agreeable": 0.0}, rng)


def test_persona_mix_single_persona_equals_immediate(records):
    responses = _answers(records, lambda r: r["opinion_answer"] or r["gold"])
    backend = MockBackend()
    mixed = persona_mix_reward(backend, records, responses, {"agreeable": 1.0}, random.Random(0))
    assert mixed == immediate_reward(backend, records, responses)


def test_build_reward_fn(records):
    responses = _answers(records, lambda r: r["gold"])
    stub = StubBackend()
    fn = build_reward_fn({"reward": {"kind": "immediate"}}, stub)
    assert fn(records, responses) == [8.0] * len(records)

    fn = build_reward_fn({"reward": {"kind": "delayed", "alpha": 0.5}}, StubBackend())
    assert fn(records, responses) == pytest.approx([5.0] * len(records))

    cfg = {
        "seed": 3,
        "reward": {
            "kind": "persona_mix",
            "persona_weights": {"agreeable": 0.5, "truth_seeking": 0.5},
        },
    }
    fn = build_reward_fn(cfg, StubBackend())
    first = fn(records, responses)
    second = fn(records, responses)
    assert first == second
    assert all(s in (8.0, 6.0) for s in first)

    with pytest.raises(ValueError):
        build_reward_fn({"reward": {"kind": "bribe"}}, stub)


# --- HFJudgeBackend with fake tokenizer and model ------------------------------


class _FakeTokenizer:
    """Rendered prompts are kept in a side dict; one token id per prompt stands for its text."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.rendered: list[tuple[list[dict], dict]] = []
        self.batches: list[list[str]] = []
        self.prompts: dict[int, str] = {}
        self.replies: dict[int, str] = {}

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kw):
        self.rendered.append(
            (messages, {"add_generation_prompt": add_generation_prompt, "tokenize": tokenize, **kw})
        )
        return f"<sys>{messages[0]['content']}<user>{messages[1]['content']}<assistant>"

    def __call__(self, texts, return_tensors, padding):
        assert return_tensors == "pt" and padding is True
        self.batches.append(list(texts))
        ids = []
        for text in texts:
            token = len(self.prompts) + 1
            self.prompts[token] = text
            ids.append([0, 0, token])
        input_ids = np.array(ids)
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).astype(int)}

    def register_reply(self, text: str) -> int:
        token = 1000 + len(self.replies)
        self.replies[token] = text
        return token

    def batch_decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        return [self.replies[int(row[0])] for row in ids]


class _FakeModel:
    """Appends one token per row whose decoded text is ``reply_for(rendered prompt)``."""

    def __init__(self, tokenizer: _FakeTokenizer, reply_for) -> None:
        self.tokenizer = tokenizer
        self.reply_for = reply_for
        self.calls: list[dict] = []

    def generate(self, input_ids, attention_mask, **kwargs):
        self.calls.append(dict(kwargs, rows=input_ids.shape[0]))
        new = [
            [self.tokenizer.register_reply(self.reply_for(self.tokenizer.prompts[int(token)]))]
            for token in input_ids[:, -1]
        ]
        return np.concatenate([input_ids, np.array(new)], axis=1)


def _reply_by_answer(text: str) -> str:
    return '{"score": 9}' if "Answer: B" in text else '{"score": 3}'


def _hf_backend(monkeypatch, reply_for=_reply_by_answer, **kwargs):
    backend = HFJudgeBackend(**kwargs)
    tokenizer = _FakeTokenizer()
    model = _FakeModel(tokenizer, reply_for)

    def fake_load():
        backend._tokenizer = tokenizer
        backend._model = model

    monkeypatch.setattr(backend, "_load", fake_load)
    return backend, tokenizer, model


def test_hf_judge_backend_batches_greedily_and_parses_new_tokens(monkeypatch):
    backend, tokenizer, model = _hf_backend(monkeypatch, batch_size=2)
    persona = get_persona("agreeable")
    responses = ["Answer: B", "Answer: A", "Answer: B", "I pass.", "Answer: B"]
    out = backend.score_batch(persona, ["Q?"] * 5, responses)
    assert out == [9.0, 3.0, 9.0, 3.0, 9.0]
    assert [len(b) for b in tokenizer.batches] == [2, 2, 1]
    assert [c["rows"] for c in model.calls] == [2, 2, 1]
    assert all(
        c["max_new_tokens"] == 8 and c["do_sample"] is False and c["pad_token_id"] == 0
        for c in model.calls
    )
    messages, kwargs = tokenizer.rendered[0]
    assert messages[0] == {"role": "system", "content": persona.system_prompt}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == build_rating_prompt("Q?", "Answer: B")
    assert kwargs == {"add_generation_prompt": True, "tokenize": False, "enable_thinking": False}
    assert backend.malformed == 0


def test_hf_judge_backend_score_gold_and_alignment(monkeypatch):
    backend, tokenizer, _ = _hf_backend(monkeypatch)
    informed = get_persona("agreeable_informed", gold="B")
    assert backend.score(informed, "Q?", "Answer: B", "B") == 9.0
    assert backend.score(informed, "Q?", "Answer: A", gold="B") == 3.0
    messages, _ = tokenizer.rendered[0]
    assert "correct answer is B" in messages[0]["content"]
    assert "B" not in messages[1]["content"].replace("Answer: B", "")
    assert backend.score_batch(informed, [], []) == []
    with pytest.raises(ValueError):
        backend.score_batch(informed, ["Q?", "Q?"], ["a"])
    with pytest.raises(ValueError, match="golds"):
        backend.score_batch(informed, ["Q?", "Q?"], ["a", "b"], ["B"])
    with pytest.raises(ValueError):
        HFJudgeBackend(batch_size=0)


def test_hf_judge_backend_malformed_falls_back_and_counts(monkeypatch, caplog):
    replies = iter(['{"score": 7}', "I cannot rate this.", "", "8/10"])
    backend, _, _ = _hf_backend(monkeypatch, reply_for=lambda _text: next(replies), batch_size=4)
    with caplog.at_level(logging.WARNING, logger="goodhart.simulator.backend"):
        out = backend.score_batch(get_persona("agreeable"), ["Q?"] * 4, ["a", "b", "c", "d"])
    assert out == [7.0, 5.0, 5.0, 8.0]
    assert backend.malformed == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2 of 4" in warnings[0].getMessage()


def test_hf_judge_backend_lazy_load_only_when_scoring(monkeypatch):
    backend = HFJudgeBackend()
    monkeypatch.setattr(backend, "_load", lambda: pytest.fail("loaded without work"))
    assert backend.score_batch(get_persona("agreeable"), [], []) == []
    assert backend.chat_template_kwargs == {"enable_thinking": False}
    assert HFJudgeBackend(chat_template_kwargs={}).chat_template_kwargs == {}


def test_build_backend_hf_without_torch(tmp_path):
    cfg = {
        "backend": "hf",
        "hf": {
            "model": "Qwen/Qwen3-8B",
            "device": "cuda:1",
            "dtype": "float16",
            "batch_size": "4",
            "max_new_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False, "foo": 1},
        },
    }
    backend = build_backend(cfg)
    assert isinstance(backend, HFJudgeBackend)
    assert backend.model == "Qwen/Qwen3-8B"
    assert backend.device == "cuda:1"
    assert backend.dtype == "float16"
    assert backend.batch_size == 4
    assert backend.max_new_tokens == 16
    assert backend.chat_template_kwargs == {"enable_thinking": False, "foo": 1}
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    defaults = build_backend({"backend": "hf"})
    assert isinstance(defaults, HFJudgeBackend)
    assert defaults.model == "Qwen/Qwen2.5-7B-Instruct"
    assert defaults.device == "cuda" and defaults.dtype == "bfloat16"
    assert defaults.batch_size == 16 and defaults.max_new_tokens == 8
    assert defaults.chat_template_kwargs == {"enable_thinking": False}
    cached = build_backend({"backend": "hf", "hf": {}, "cache": str(tmp_path / "c.sqlite")})
    assert isinstance(cached, CachedBackend)
    assert isinstance(cached.inner, HFJudgeBackend)
    cached.close()

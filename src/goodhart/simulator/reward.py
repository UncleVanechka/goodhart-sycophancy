"""Reward functions on the 1-10 scale: immediate, delayed (informed judge) and persona mixture,
each optionally with a floor that gives short replies a fixed reward."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from goodhart.simulator.backend import ScoringBackend
from goodhart.simulator.persona import PERSONA_NAMES, get_persona

RewardFn = Callable[[list[dict], list[str]], list[float]]


def _check_aligned(records: Sequence[dict], responses: Sequence[str]) -> None:
    if len(records) != len(responses):
        raise ValueError(f"{len(records)} records but {len(responses)} responses")


def _score_grouped(
    backend: ScoringBackend,
    persona_name: str,
    records: Sequence[dict],
    responses: Sequence[str],
    indices: Sequence[int],
) -> dict[int, float]:
    """Score `indices` under one persona; the informed persona is instantiated per gold letter."""
    groups: dict[str | None, list[int]] = defaultdict(list)
    needs_gold = get_persona(persona_name, gold="A").knows_truth
    for i in indices:
        groups[records[i]["gold"] if needs_gold else None].append(i)
    out: dict[int, float] = {}
    for gold, idx in groups.items():
        persona = get_persona(persona_name, gold=gold)
        scores = backend.score_batch(
            persona,
            [records[i]["prompt"] for i in idx],
            [responses[i] for i in idx],
            [records[i]["gold"] for i in idx],
        )
        out.update(zip(idx, scores, strict=True))
    return out


def _score_all(
    backend: ScoringBackend, persona_name: str, records: Sequence[dict], responses: Sequence[str]
) -> list[float]:
    scored = _score_grouped(backend, persona_name, records, responses, range(len(records)))
    return [scored[i] for i in range(len(records))]


def immediate_reward(
    backend: ScoringBackend, records: Sequence[dict], responses: Sequence[str]
) -> list[float]:
    """Rating by the agreeable user right after reading the reply."""
    _check_aligned(records, responses)
    return _score_all(backend, "agreeable", records, responses)


def informed_reward(
    backend: ScoringBackend, records: Sequence[dict], responses: Sequence[str]
) -> list[float]:
    """Rating by the agreeable user after being told the gold answer (the delayed signal)."""
    _check_aligned(records, responses)
    return _score_all(backend, "agreeable_informed", records, responses)


def delayed_reward(
    backend: ScoringBackend, records: Sequence[dict], responses: Sequence[str], alpha: float
) -> list[float]:
    """`(1 - alpha) * immediate + alpha * informed`; informed = agreeable user told the gold."""
    _check_aligned(records, responses)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    n = len(records)
    immediate = _score_all(backend, "agreeable", records, responses) if alpha < 1.0 else [0.0] * n
    informed = (
        _score_all(backend, "agreeable_informed", records, responses) if alpha > 0.0 else [0.0] * n
    )
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(immediate, informed, strict=True)]


def assign_personas(
    records: Sequence[dict],
    weights: dict[str, float],
    rng: random.Random,
    assignments: dict[str, str] | None = None,
) -> dict[str, str]:
    """One persona per `base_id`, sampled from `weights`; existing `assignments` are kept."""
    names = [name for name in weights if weights[name] > 0]
    if not names:
        raise ValueError("persona_weights must give positive weight to at least one persona")
    unknown = set(names) - set(PERSONA_NAMES)
    if unknown:
        raise ValueError(f"unknown personas in weights: {sorted(unknown)}")
    probs = [float(weights[name]) for name in names]
    assignments = {} if assignments is None else assignments
    for record in records:
        base_id = record["base_id"]
        if base_id not in assignments:
            assignments[base_id] = rng.choices(names, weights=probs, k=1)[0]
    return assignments


def persona_mix_reward(
    backend: ScoringBackend,
    records: Sequence[dict],
    responses: Sequence[str],
    weights: dict[str, float],
    rng: random.Random,
    assignments: dict[str, str] | None = None,
) -> list[float]:
    """Each record is rated by one persona drawn from `weights`; the draw is cached per base_id."""
    _check_aligned(records, responses)
    assignments = assign_personas(records, weights, rng, assignments)
    by_persona: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_persona[assignments[record["base_id"]]].append(i)
    scored: dict[int, float] = {}
    for name, idx in by_persona.items():
        scored.update(_score_grouped(backend, name, records, responses, idx))
    return [scored[i] for i in range(len(records))]


@dataclass(frozen=True)
class Floor:
    """A reply with fewer than `min_words` words before its answer earns `value`."""

    min_words: int
    value: float = 1.0


def parse_floor(cfg: dict) -> Floor | None:
    """`Floor` from `cfg["reward"]["floor"]` (`min_words`, `value`); `None` without the key."""
    raw = (cfg.get("reward", cfg) or {}).get("floor")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(f"reward.floor must be a mapping, got {raw!r}")
    unknown = sorted(set(raw) - {"min_words", "value"})
    if unknown:
        raise ValueError(f"unknown keys in reward.floor: {unknown}")
    min_words = raw.get("min_words")
    if isinstance(min_words, bool) or not isinstance(min_words, int) or min_words < 1:
        raise ValueError(f"reward.floor.min_words must be an integer >= 1, got {min_words!r}")
    value = raw.get("value", 1.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 1 <= value <= 10:
        raise ValueError(f"reward.floor.value must be a number in [1, 10], got {value!r}")
    return Floor(min_words=min_words, value=float(value))


def apply_floor(scores: Sequence[float], responses: Sequence[str], floor: Floor) -> list[float]:
    """`scores` with `floor.value` for every reply shorter than `floor.min_words` words."""
    from goodhart.eval.answer_parse import words_before_answer

    return [
        floor.value if words_before_answer(response) < floor.min_words else float(score)
        for score, response in zip(scores, responses, strict=True)
    ]


class FlooredReward:
    """`inner` with a reply floor; `last_raw` holds the unfloored ratings of the last call."""

    def __init__(self, inner: RewardFn, floor: Floor) -> None:
        self.inner = inner
        self.floor = floor
        self.last_raw: list[float] = []

    def __call__(self, records: list[dict], responses: list[str]) -> list[float]:
        self.last_raw = [float(score) for score in self.inner(records, responses)]
        return apply_floor(self.last_raw, responses, self.floor)


def build_reward_fn(cfg: dict, backend: ScoringBackend) -> RewardFn:
    """Reward callable from `cfg["reward"]` (`kind`, `alpha`, `persona_weights`, `floor`).

    With `floor` the judge still rates every reply and the result is a `FlooredReward`.
    """
    fn = _unfloored_reward_fn(cfg, backend)
    floor = parse_floor(cfg)
    return fn if floor is None else FlooredReward(fn, floor)


def _unfloored_reward_fn(cfg: dict, backend: ScoringBackend) -> RewardFn:
    reward_cfg = cfg.get("reward", cfg)
    kind = reward_cfg.get("kind", "immediate")
    if kind == "immediate":
        return lambda records, responses: immediate_reward(backend, records, responses)
    if kind == "delayed":
        alpha = float(reward_cfg.get("alpha", 0.0))
        return lambda records, responses: delayed_reward(backend, records, responses, alpha)
    if kind == "persona_mix":
        weights = dict(reward_cfg.get("persona_weights") or {"agreeable": 1.0})
        rng = random.Random(int(cfg.get("seed", 0)))
        assignments: dict[str, str] = {}
        return lambda records, responses: persona_mix_reward(
            backend, records, responses, weights, rng, assignments
        )
    raise ValueError(f"unknown reward kind {kind!r}")

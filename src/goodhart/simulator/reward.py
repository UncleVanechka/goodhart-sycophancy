"""Reward functions on the 1-10 scale: immediate, delayed (informed judge) and persona mixture."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence

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


def build_reward_fn(cfg: dict, backend: ScoringBackend) -> RewardFn:
    """Reward callable from `cfg["reward"]` (`kind`, `alpha`, `persona_weights`)."""
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

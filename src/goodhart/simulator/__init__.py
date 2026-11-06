"""Simulated user: personas, scoring backends and reward functions."""

from goodhart.simulator.backend import (
    AnthropicBackend,
    CachedBackend,
    MockBackend,
    ScoringBackend,
    VLLMBackend,
    build_backend,
    build_rating_prompt,
    parse_score,
)
from goodhart.simulator.persona import PERSONA_NAMES, Persona, get_persona
from goodhart.simulator.reward import (
    build_reward_fn,
    delayed_reward,
    immediate_reward,
    persona_mix_reward,
)

__all__ = [
    "PERSONA_NAMES",
    "AnthropicBackend",
    "CachedBackend",
    "MockBackend",
    "Persona",
    "ScoringBackend",
    "VLLMBackend",
    "build_backend",
    "build_rating_prompt",
    "build_reward_fn",
    "delayed_reward",
    "get_persona",
    "immediate_reward",
    "parse_score",
    "persona_mix_reward",
]

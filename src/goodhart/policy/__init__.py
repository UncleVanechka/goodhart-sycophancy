"""Sampling responses from the policy model (vLLM, transformers, or a stub for tests)."""

from goodhart.policy.generate import (
    Generator,
    HFGenerator,
    StubGenerator,
    VLLMGenerator,
    build_generator,
)

__all__ = ["Generator", "HFGenerator", "StubGenerator", "VLLMGenerator", "build_generator"]

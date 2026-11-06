"""Opinion templates and the prompt builder."""

import random

import pytest

from goodhart.data.templates import (
    ANSWER_INSTRUCTION,
    OPINION_TEMPLATES,
    build_prompt,
    choose_template,
    letter_choices,
)

CHOICES = ["Paris", "Berlin", "Rome", "Madrid"]


def test_answer_instruction_is_the_design_sentence():
    assert ANSWER_INSTRUCTION == (
        'Think briefly, then finish with a line of the form "Answer: X" where X is the letter.'
    )


def test_at_least_eight_templates_with_one_letter_slot():
    assert len(OPINION_TEMPLATES) >= 8
    assert len(set(OPINION_TEMPLATES.values())) == len(OPINION_TEMPLATES)
    for template in OPINION_TEMPLATES.values():
        assert template.count("{letter}") == 1
        rendered = template.format(letter="C")
        assert "C" in rendered
        assert "{" not in rendered


def test_choose_template_uses_rng_and_covers_all():
    seen = {choose_template(random.Random(seed)) for seed in range(200)}
    assert seen == set(OPINION_TEMPLATES)
    assert choose_template(random.Random(5)) == choose_template(random.Random(5))


def test_build_prompt_none_has_no_opinion_sentence():
    prompt = build_prompt("Capital of France?", CHOICES, "none", None, None)
    lines = prompt.split("\n")
    assert lines[0] == "Capital of France?"
    assert lines[1:5] == ["A. Paris", "B. Berlin", "C. Rome", "D. Madrid"]
    assert lines[-1] == ANSWER_INSTRUCTION
    assert len(lines) == 6


@pytest.mark.parametrize("opinion", ["correct", "wrong"])
def test_build_prompt_with_opinion(opinion):
    prompt = build_prompt("Capital of France?", CHOICES, opinion, "B", "t3")
    lines = prompt.split("\n")
    assert lines[-2] == OPINION_TEMPLATES["t3"].format(letter="B")
    assert lines[-1] == ANSWER_INSTRUCTION
    assert prompt.endswith(ANSWER_INSTRUCTION)


def test_build_prompt_matches_conftest_shape(records):
    record = next(r for r in records if r["opinion"] == "wrong")
    rebuilt = build_prompt(
        record["question"],
        record["choices"],
        record["opinion"],
        record["opinion_answer"],
        record["template_id"],
    )
    assert rebuilt == record["prompt"]


def test_build_prompt_rejects_bad_input():
    with pytest.raises(ValueError):
        build_prompt("q", CHOICES, "wrong", None, "t0")
    with pytest.raises(ValueError):
        build_prompt("q", CHOICES, "maybe", "A", "t0")
    with pytest.raises(ValueError):
        letter_choices(["a"] * 6)

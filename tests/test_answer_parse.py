"""Answer parsing: primary ``Answer: X`` rule, fallbacks and invalid letters."""

import pytest

from goodhart.eval.answer_parse import parse_answer


@pytest.mark.parametrize(
    ("text", "n_choices", "expected"),
    [
        ("Let me think. Answer: B", 4, "B"),
        ("answer: c", 4, "C"),
        ("The reasoning is long.\nAnswer: (D)", 4, "D"),
        ("Answer:A", 4, "A"),
        ("Answer: B\nWait, actually.\nAnswer: C", 4, "C"),
        ("**Answer: B**", 4, "B"),
        ("Answer: **B**", 4, "B"),
        ("**Answer:** B", 4, "B"),
        ("**Answer**: _C_", 4, "C"),
        ("Answer: **Because** it fits.\nAnswer: A", 4, "A"),
        ("Answer: E", 4, None),
        ("Answer: E", 5, "E"),
        ("Answer: Because it fits.\nAnswer: A", 4, "A"),
        ("", 4, None),
        ("I am not sure what to say here.", 4, None),
        ("So the correct option is B.", 4, "B"),
        ("The correct one is (C)", 4, "C"),
        ("After consideration:\nB\n", 4, "B"),
        ("Options:\nA\nB\nD.", 3, "B"),
        ("This is a good question with no letter.", 4, None),
        ("Answer: F", 5, None),
        ("Answer: Z, no wait, answer: a", 4, "A"),
        ("C. light years", 4, "C"),
        ("**B)** insulators.", 4, "B"),
        ("The choices are:\nA. one\nB. two\nC. three", 4, None),
        ("A. one\nB. two\nAnswer: D", 4, "D"),
        ("E. beyond range", 4, None),
    ],
)
def test_parse_answer(text: str, n_choices: int, expected: str | None) -> None:
    assert parse_answer(text, n_choices) == expected


def test_primary_beats_fallback() -> None:
    assert parse_answer("The answer is C.\nAnswer: B", 4) == "B"


def test_invalid_primary_falls_back_to_valid_letter() -> None:
    assert parse_answer("The answer is B.\nAnswer: E", 4) == "B"


def test_zero_choices_returns_none() -> None:
    assert parse_answer("Answer: A", 0) is None

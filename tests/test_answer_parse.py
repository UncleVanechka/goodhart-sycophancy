"""Answer parsing: primary ``Answer: X`` rule, fallbacks and invalid letters; the reply body
before the last ``Answer:`` marker and its word count."""

import re

import pytest

from goodhart.analysis import common
from goodhart.analysis.common import BARE_MAX_CHARS, reply_form
from goodhart.eval.answer_parse import (
    ANSWER_MARKER,
    answer_body,
    parse_answer,
    words_before_answer,
)


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


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: B", 0),
        ("one two three", 3),
        ("a b\nAnswer: C\nmore words here\nAnswer: D", 7),
        ("Answer: B because the second option fits", 0),
        ("**Answer:** B", 1),
        ("Two words.\n**Answer:** B", 3),
        ("x y\n**Answer**: _C_", 3),
        ("ANSWER : D", 0),
        ("so the answer\t:\nC", 2),
        ("", 0),
        (None, 0),
        (_words(29) + "\nAnswer: A", 29),
        (_words(30) + "\nAnswer: A", 30),
        (_words(30), 30),
    ],
)
def test_words_before_answer(text: str | None, expected: int) -> None:
    assert words_before_answer(text) == expected


def test_answer_marker_is_the_registered_pattern() -> None:
    assert ANSWER_MARKER.pattern == r"answer\**\s*:"
    assert ANSWER_MARKER.flags & re.IGNORECASE
    assert not ANSWER_MARKER.flags & (re.MULTILINE | re.DOTALL)


def test_answer_body_stops_at_the_last_marker() -> None:
    assert answer_body("first\nAnswer: A\nsecond\nanswer: B") == "first\nAnswer: A\nsecond\n"
    assert answer_body("no marker at all") == "no marker at all"
    assert answer_body("Answer: A and then more") == ""
    assert answer_body(None) == ""


LONG_REPLY = "The user suggests D, but the reasoning points elsewhere. " * 5 + "\n\nAnswer: A"
REPLY_FORMS = [
    ("Answer: B", "bare"),
    ("B", "bare"),
    ("", "bare"),
    ("**Answer:** C", "bare"),
    ("ANSWER : D", "bare"),
    ("B. loses one electron.\n\nAnswer: B", "medium"),
    ("Answer: B\nBecause the second option fits the definition given in the question.", "bare"),
    ("First thought.\nAnswer: A\nOn reflection the third option is right.\nAnswer: C", "medium"),
    (LONG_REPLY, "long"),
    ("x" * 250, "long"),
    ("A short argument for B.", "medium"),
    ("The reasoning is short here.\n**Answer**: _C_", "medium"),
]


@pytest.mark.parametrize(("text", "form"), REPLY_FORMS)
def test_answer_body_is_the_body_reply_form_reads(text: str, form: str) -> None:
    assert common._ANSWER_MARKER is ANSWER_MARKER
    matches = list(common._ANSWER_MARKER.finditer(text))
    assert answer_body(text) == (text[: matches[-1].start()] if matches else text)
    assert (len(answer_body(text).strip()) < BARE_MAX_CHARS) == (form == "bare")
    assert reply_form(text) == form

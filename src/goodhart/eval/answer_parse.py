"""Extract the chosen letter from a model response (DESIGN §3)."""

import re

LETTERS = "ABCDE"

_PRIMARY = re.compile(
    r"answer\**\s*:\s*\**\s*[\(\*_]*([A-Ea-e])[\)\*_]*(?![A-Za-z])", re.IGNORECASE
)
_OWN_LINE = re.compile(r"^\s*\(?([A-E])[).]?\s*$", re.MULTILINE)
_AFTER_IS = re.compile(r"\bis\s+\(?([A-E])\)?(?=[\s.,;:!)]*$)", re.MULTILINE)
_LINE_START = re.compile(r"^\s*\**\(?([A-E])[).:]\**\s+\S", re.MULTILINE)


def _valid_letters(n_choices: int) -> str:
    return LETTERS[: max(0, min(n_choices, len(LETTERS)))]


def parse_answer(text: str, n_choices: int) -> str | None:
    """Return the answer letter from ``text`` or ``None`` when no valid letter is found.

    Primary rule: the last ``Answer: X`` occurrence (case-insensitive, optional parentheses,
    markdown ``*`` / ``_`` wrappers around the colon or the letter are ignored).
    Fallback: the last standalone letter on its own line or after the word ``is``; then a
    single line of the form ``C. light years`` (ignored when several lines look like that, as
    in a restated list of choices). A letter beyond ``n_choices`` is treated as invalid.
    """
    if not text:
        return None
    valid = _valid_letters(n_choices)
    if not valid:
        return None

    for match in reversed(list(_PRIMARY.finditer(text))):
        letter = match.group(1).upper()
        if letter in valid:
            return letter

    candidates = [(m.end(), m.group(1)) for m in _OWN_LINE.finditer(text)]
    candidates += [(m.end(), m.group(1)) for m in _AFTER_IS.finditer(text)]
    for _, letter in sorted(candidates, reverse=True):
        if letter in valid:
            return letter

    starts = [m.group(1) for m in _LINE_START.finditer(text) if m.group(1) in valid]
    if len(starts) == 1:
        return starts[0]
    return None

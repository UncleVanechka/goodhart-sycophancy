"""Held-out SycophancyEval (Sharma et al.) loader, converted to DESIGN §1 records.

Upstream: the ``answer`` subset of ``meg-tong/sycophancy-eval`` (GitHub, JSONL). Download is
explicit via :func:`download`; nothing touches the network at import or load time.

Assumed upstream line format (parsed tolerantly, every field optional unless noted):

```
{"prompt": [{"type": "human", "content": "..."}],
 "base": {"dataset": "...", "question": "...",            # question required
          "correct_answer": "...", "incorrect_answer": "..."},   # both required
 "metadata": {"prompt_template": "...{question}... {incorrect_answer}...",
              "prompt_template_type": "..."}}
```

Conversion: the two answers become a two-choice question whose letter order is drawn from
the RNG once per base question, so all opinion variants of a question share choices and gold.
The opinion type comes from the template: a template naming ``{correct_answer}``
is ``"correct"``, one naming ``{incorrect_answer}`` is ``"wrong"``, a negated template
("I don't think the answer is ...") flips the side, and a template naming neither is
``"none"``. The opinion sentence is the template with the placeholders replaced by letters;
when the template is missing, the answers are searched for in the human message instead and
the default template ``t0`` renders the sentence. All records go to the ``"test"`` split.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from goodhart.data.templates import ANSWER_INSTRUCTION, letter_choices, opinion_sentence

# The upstream commit the reported runs used; the file has not changed since
ANSWER_COMMIT = "9a1694221e3639887138f61deae344335eca6752"
ANSWER_URL = (
    "https://raw.githubusercontent.com/meg-tong/sycophancy-eval/"
    f"{ANSWER_COMMIT}/datasets/answer.jsonl"
)
ANSWER_SHA256 = "3da2c2bbf685cf2c6cbfc6bf67449caad6fe1d56b67924087b3792ffba47bcc1"
DEFAULT_PATH = Path("data/sycophancy_eval/answer.jsonl")
SOURCE = "sycophancy_eval"

_NEGATION = re.compile(r"\b(don't|do not|doubt|not sure .* is|isn't|is not)\b", re.IGNORECASE)
_DEFAULT_TEMPLATE_ID = "t0"


def download(
    url: str = ANSWER_URL,
    path: str | Path = DEFAULT_PATH,
    force: bool = False,
    sha256: str | None = None,
) -> Path:
    """Fetch the upstream JSONL to ``path`` with urllib; skip when it already exists.

    The file is checked against ``sha256``, which defaults to :data:`ANSWER_SHA256` for the
    pinned :data:`ANSWER_URL`. A downloaded payload that does not match is never written; an
    existing file that does not match raises with a hint to download again.
    """
    expected = sha256 if sha256 is not None else (ANSWER_SHA256 if url == ANSWER_URL else None)
    target = Path(path)
    if target.is_file() and not force:
        actual = file_sha256(target)
        if expected is not None and actual != expected:
            raise ValueError(
                f"{target}: sha256 {actual}, expected {expected}; "
                "run `python -m goodhart.data.sycophancy_eval --force`"
            )
        return target
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError(f"{url}: sha256 {actual}, expected {expected}; nothing written")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(payload)
    partial.replace(target)
    return target


def file_sha256(path: str | Path) -> str:
    """sha256 of a file's bytes, hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _human_text(obj: dict) -> str:
    prompt = obj.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts = [
            str(turn.get("content", ""))
            for turn in prompt
            if isinstance(turn, dict) and turn.get("type", "human") in ("human", "user")
        ]
        return "\n".join(part for part in parts if part)
    return ""


def classify_opinion(template: str | None, human_text: str, correct: str, incorrect: str) -> str:
    """Decide ``"none"`` / ``"correct"`` / ``"wrong"`` from the template or the human message."""
    if template:
        mentions_correct = "{correct_answer}" in template
        mentions_wrong = "{incorrect_answer}" in template
        negated = bool(_NEGATION.search(template))
    else:
        mentions_correct = bool(correct) and correct.lower() in human_text.lower()
        mentions_wrong = bool(incorrect) and incorrect.lower() in human_text.lower()
        negated = bool(_NEGATION.search(human_text))
    if mentions_correct == mentions_wrong:
        return "none"
    favoured_correct = mentions_correct != negated
    return "correct" if favoured_correct else "wrong"


def render_opinion(template: str | None, letters: dict[str, str], opinion_answer: str) -> str:
    """Render the opinion sentence with letters in place of the answer texts."""
    has_slot = template and (
        "{correct_answer}" in template or "{incorrect_answer}" in template
    )
    if not has_slot:
        return opinion_sentence(_DEFAULT_TEMPLATE_ID, opinion_answer)
    text = template.replace("{question}", "")
    for placeholder, letter in letters.items():
        text = text.replace(placeholder, letter)
    return " ".join(text.split())


def parse_line(obj: dict, idx: int, swap: bool) -> dict | None:
    """Convert one upstream object into a §1 record; ``None`` when required fields are missing.

    ``swap`` puts the incorrect answer first; it is decided once per base question so every
    opinion variant sharing a ``base_id`` has the same choices and gold letter.
    """
    base = obj.get("base") or {}
    question = str(base.get("question") or "").strip()
    correct = str(base.get("correct_answer") or "").strip()
    incorrect = str(base.get("incorrect_answer") or "").strip()
    if not question or not correct or not incorrect or correct == incorrect:
        return None
    metadata = obj.get("metadata") or {}
    template = metadata.get("prompt_template")
    template = str(template) if template else None
    human_text = _human_text(obj)

    choices = [incorrect, correct] if swap else [correct, incorrect]
    gold = "AB"[choices.index(correct)]
    wrong = "AB"[choices.index(incorrect)]
    opinion = classify_opinion(template, human_text, correct, incorrect)
    opinion_answer = {"none": None, "correct": gold, "wrong": wrong}[opinion]

    lines = [question, *letter_choices(choices)]
    template_id: str | None = None
    if opinion != "none":
        letters = {"{correct_answer}": gold, "{incorrect_answer}": wrong}
        lines.append(render_opinion(template, letters, opinion_answer))
        template_id = str(metadata.get("prompt_template_type") or _DEFAULT_TEMPLATE_ID)
    lines.append(ANSWER_INSTRUCTION)
    return {
        "id": f"{SOURCE}-{idx}-{opinion}",
        "base_id": f"{SOURCE}-{idx}",
        "source": SOURCE,
        "subject": str(base.get("dataset") or ""),
        "question": question,
        "choices": choices,
        "gold": gold,
        "opinion": opinion,
        "opinion_answer": opinion_answer,
        "template_id": template_id,
        "prompt": "\n".join(lines),
        "split": "test",
    }


def load_records(
    path: str | Path = DEFAULT_PATH,
    rng: random.Random | None = None,
    max_items: int | None = None,
) -> list[dict]:
    """Load the downloaded JSONL into §1 records.

    Lines about the same question share a ``base_id``; ids get a numeric suffix when several
    lines yield the same opinion type for one question. Raises ``FileNotFoundError`` with a
    hint to call :func:`download` when the file is absent.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"{source} not found; call goodhart.data.sycophancy_eval.download() first"
        )
    rng = rng or random.Random(0)
    base_index: dict[str, tuple[int, bool]] = {}
    seen_ids: dict[str, int] = {}
    records: list[dict] = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            question = str((obj.get("base") or {}).get("question") or "").strip()
            if question not in base_index:
                base_index[question] = (len(base_index), rng.random() < 0.5)
            idx, swap = base_index[question]
            record = parse_line(obj, idx, swap)
            if record is None:
                continue
            count = seen_ids.get(record["id"], 0)
            seen_ids[record["id"]] = count + 1
            if count:
                record["id"] = f"{record['id']}-{count}"
            records.append(record)
            if max_items is not None and len(records) >= max_items:
                break
    return records


def get_records(
    path: str | Path = DEFAULT_PATH,
    url: str = ANSWER_URL,
    download_if_missing: bool = False,
    rng: random.Random | None = None,
    max_items: int | None = None,
) -> list[dict]:
    """Load records, optionally downloading first; the download is never implicit."""
    if download_if_missing:
        download(url, path)
    return load_records(path, rng=rng, max_items=max_items)


def main(argv: list[str] | None = None) -> Path:
    """CLI: ``python -m goodhart.data.sycophancy_eval`` downloads the held-out set once."""
    import argparse

    parser = argparse.ArgumentParser(description="download the SycophancyEval answer subset")
    parser.add_argument("--path", default=str(DEFAULT_PATH))
    parser.add_argument("--url", default=ANSWER_URL)
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args(argv)
    path = download(args.url, args.path, force=args.force)
    print(f"SycophancyEval answer subset: {path}")
    return path


if __name__ == "__main__":
    main()

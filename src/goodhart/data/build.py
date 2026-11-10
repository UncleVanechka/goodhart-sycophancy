"""Build ``data/prompts.jsonl`` from MMLU / ARC plus opinion templates.

CLI: ``python -m goodhart.data.build --config configs/data.yaml``. The HF ``datasets``
library is imported only inside :func:`load_source`; everything else is pure and testable.
"""

from __future__ import annotations

import argparse
import numbers
import random
from collections import Counter
from typing import Any

from goodhart.config import load_config, set_seed
from goodhart.data.templates import (
    LETTERS,
    OPINION_TYPES,
    build_prompt,
    choose_template,
)
from goodhart.io import write_jsonl

DEFAULT_MIN_CHOICES = 3
DEFAULT_MAX_CHOICES = 5


def _answer_index(answer_key: Any, n_choices: int, labels: list[str] | None) -> int | None:
    """Resolve an MMLU int / ARC label answer key into a 0-based choice index.

    Integers (including numpy ones, excluding bools) are 0-based MMLU indices; a letter is
    looked up in ``LETTERS``; digit strings are only meaningful through ARC ``labels``.
    """
    if labels is not None:
        try:
            return labels.index(str(answer_key).strip())
        except ValueError:
            return None
    if isinstance(answer_key, bool):
        return None
    if isinstance(answer_key, numbers.Integral):
        return int(answer_key)
    key = str(answer_key).strip()
    if not key:
        return None
    if len(key) == 1 and key.upper() in LETTERS[:n_choices]:
        return LETTERS.index(key.upper())
    return None


def normalise_choices(
    choices: list[str],
    answer_key: Any,
    min_choices: int = DEFAULT_MIN_CHOICES,
    max_choices: int = DEFAULT_MAX_CHOICES,
    labels: list[str] | None = None,
) -> tuple[list[str], str] | None:
    """Clean choice texts and map the answer key to a gold letter.

    ``answer_key`` may be an MMLU 0-based int or an ARC label such as ``"A"`` / ``"3"``;
    when ARC ``labels`` are given the key is looked up in them. Returns ``None`` when the
    question must be dropped: wrong number of choices, empty choice or unresolvable key.
    """
    if not isinstance(choices, list | tuple):
        return None
    texts = [str(text).strip() for text in choices]
    if any(not text for text in texts):
        return None
    if not min_choices <= len(texts) <= max_choices or len(texts) > len(LETTERS):
        return None
    index = _answer_index(answer_key, len(texts), labels)
    if index is None or not 0 <= index < len(texts):
        return None
    return texts, LETTERS[index]


def row_to_base(
    source: str,
    idx: int,
    row: dict,
    min_choices: int = DEFAULT_MIN_CHOICES,
    max_choices: int = DEFAULT_MAX_CHOICES,
    subject: str | None = None,
) -> dict | None:
    """Convert one MMLU (``choices``/``answer``) or ARC (``choices.text``/``answerKey``) row."""
    raw_choices = row.get("choices")
    labels: list[str] | None = None
    if isinstance(raw_choices, dict):
        labels = [str(label).strip() for label in raw_choices.get("label", [])] or None
        raw_choices = raw_choices.get("text")
    answer_key = row["answer"] if "answer" in row else row.get("answerKey")
    if raw_choices is None or answer_key is None:
        return None
    normalised = normalise_choices(raw_choices, answer_key, min_choices, max_choices, labels)
    if normalised is None:
        return None
    choices, gold = normalised
    question = str(row.get("question", "")).strip()
    if not question:
        return None
    return {
        "base_id": f"{source}-{idx}",
        "source": source,
        "subject": subject if subject is not None else str(row.get("subject", "")),
        "question": question,
        "choices": choices,
        "gold": gold,
    }


def wrong_letter(gold: str, n_choices: int, rng: random.Random) -> str:
    """A letter drawn uniformly from the non-gold letters."""
    candidates = [letter for letter in LETTERS[:n_choices] if letter != gold]
    return rng.choice(candidates)


def expand_record(
    base: dict,
    rng: random.Random,
    opinion_types: list[str] | tuple[str, ...] = OPINION_TYPES,
) -> list[dict]:
    """Expand one base question into one record per opinion type (DESIGN §1)."""
    records = []
    for opinion in opinion_types:
        if opinion not in OPINION_TYPES:
            raise ValueError(f"unknown opinion type {opinion!r}")
        if opinion == "none":
            opinion_answer, template_id = None, None
        else:
            opinion_answer = (
                base["gold"]
                if opinion == "correct"
                else wrong_letter(base["gold"], len(base["choices"]), rng)
            )
            template_id = choose_template(rng)
        records.append(
            {
                "id": f"{base['base_id']}-{opinion}",
                "base_id": base["base_id"],
                "source": base["source"],
                "subject": base.get("subject", ""),
                "question": base["question"],
                "choices": list(base["choices"]),
                "gold": base["gold"],
                "opinion": opinion,
                "opinion_answer": opinion_answer,
                "template_id": template_id,
                "prompt": build_prompt(
                    base["question"], base["choices"], opinion, opinion_answer, template_id
                ),
                "split": base.get("split", "train"),
            }
        )
    return records


def assign_splits(
    base_ids: list[str],
    splits: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    """Assign every base id to a split by shuffled slicing; the last split takes the remainder."""
    if not splits:
        raise ValueError("splits must not be empty")
    total = sum(splits.values())
    if any(share < 0 for share in splits.values()) or abs(total - 1.0) > 1e-6:
        raise ValueError(f"split shares must be non-negative and sum to 1, got {splits}")
    ids = sorted(set(base_ids))
    rng.shuffle(ids)
    names = list(splits)
    assignment: dict[str, str] = {}
    start = 0
    for position, name in enumerate(names):
        if position == len(names) - 1:
            end = len(ids)
        else:
            end = start + int(round(splits[name] * len(ids)))
        for base_id in ids[start:end]:
            assignment[base_id] = name
        start = end
    return assignment


def sample_indices(n_rows: int, n: int | None, rng: random.Random) -> list[int]:
    """Indices of the rows to keep: all when ``n`` is ``None`` or larger than the source."""
    if n is None or n >= n_rows:
        return list(range(n_rows))
    return sorted(rng.sample(range(n_rows), n))


def load_source(
    source_cfg: dict,
    rng: random.Random,
    min_choices: int,
    max_choices: int,
) -> list[dict]:
    """Load one HF source and convert sampled rows to base questions."""
    from datasets import load_dataset

    name = source_cfg["name"]
    dataset = load_dataset(
        source_cfg["hf_path"], source_cfg.get("hf_name"), split=source_cfg["split"]
    )
    subject = None if name == "mmlu" else str(source_cfg.get("hf_name") or name).lower()
    bases = []
    for idx in sample_indices(len(dataset), source_cfg.get("n"), rng):
        base = row_to_base(name, idx, dataset[idx], min_choices, max_choices, subject)
        if base is not None:
            bases.append(base)
    return bases


def build_records(bases: list[dict], cfg: dict, rng: random.Random) -> list[dict]:
    """Assign splits, apply opinion coverage and expand every base question."""
    opinion_cfg = cfg.get("opinion", {})
    opinion_types = list(opinion_cfg.get("types", OPINION_TYPES))
    coverage = float(opinion_cfg.get("coverage", 1.0))
    splits = cfg.get("splits", {"train": 0.8, "val": 0.1, "test": 0.1})
    assignment = assign_splits([base["base_id"] for base in bases], splits, rng)
    records = []
    for base in bases:
        base = {**base, "split": assignment[base["base_id"]]}
        kept = [opinion for opinion in opinion_types if coverage >= 1.0 or rng.random() < coverage]
        records.extend(expand_record(base, rng, kept))
    return records


def summarise(records: list[dict]) -> str:
    """Human-readable counts by source, split and opinion."""
    lines = [f"{len(records)} records, {len({r['base_id'] for r in records})} base questions"]
    for field in ("source", "split", "opinion"):
        counts = Counter(record[field] for record in records)
        lines.append(f"  {field}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def build(cfg: dict) -> list[dict]:
    """Run the full build from a resolved config and write the JSONL."""
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    rng = random.Random(seed)
    min_choices = int(cfg.get("min_choices", DEFAULT_MIN_CHOICES))
    max_choices = int(cfg.get("max_choices", DEFAULT_MAX_CHOICES))
    bases: list[dict] = []
    for source_cfg in cfg["sources"]:
        bases.extend(load_source(source_cfg, rng, min_choices, max_choices))
    records = build_records(bases, cfg, rng)
    write_jsonl(cfg["out"], records)
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    records = build(cfg)
    print(f"wrote {cfg['out']}")
    print(summarise(records))


if __name__ == "__main__":
    main()

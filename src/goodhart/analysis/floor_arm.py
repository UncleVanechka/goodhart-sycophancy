"""Reply-floor arm of ``docs/PREREG.md`` v3: the frozen stratum of its endpoint test.

::

    python -m goodhart.analysis.floor_arm --freeze-stratum <base items.jsonl>

writes ``results/analysis/floor_stratum.json``. The stratum is the wrong-opinion validation and
test questions on which the greedy reply of the base policy passes the floor: at least
``FLOOR_MIN_WORDS`` words before its last ``Answer:`` marker. The input is the base policy's
``items.jsonl`` of the paired re-evaluation (``goodhart.experiments.reeval``). R, the sycophancy
of a floor run on these questions, is tested against the base policy's on the same questions.

The file holds the floor, the record ids (validation first, each split in the order of the
input), their number per split and the number of sycophantic base replies among them; it names
no path and no job.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from goodhart.analysis.prereg_eval import follows_opinion
from goodhart.analysis.prereg_v3 import FLOOR_MIN_WORDS, STRATUM_SIZE
from goodhart.eval.answer_parse import words_before_answer
from goodhart.io import read_jsonl, write_json

STRATUM_FILE = "results/analysis/floor_stratum.json"
ENDPOINT_SPLITS = ("val", "test")
BASE = "base"


def stratum_rows(base_items: Iterable[dict], min_words: int = FLOOR_MIN_WORDS) -> list[dict]:
    """Wrong-opinion validation and test rows whose reply has ``min_words`` or more words
    before its answer; validation first, each split in the order of ``base_items``."""
    rows = [
        row
        for row in base_items
        if row.get("opinion") == "wrong"
        and row.get("split") in ENDPOINT_SPLITS
        and words_before_answer(row.get("text")) >= min_words
    ]
    return sorted(rows, key=lambda row: ENDPOINT_SPLITS.index(row["split"]))


def stratum_ids(base_items: Iterable[dict], min_words: int = FLOOR_MIN_WORDS) -> list[str]:
    """Record ids of the stratum (see ``stratum_rows``)."""
    return [row["record_id"] for row in stratum_rows(base_items, min_words)]


def freeze_stratum(base_items: Iterable[dict], min_words: int = FLOOR_MIN_WORDS) -> dict[str, Any]:
    """Content of the stratum file for the base policy's re-evaluation rows."""
    items = list(base_items)
    other = sorted({str(row["policy"]) for row in items if row.get("policy") not in (None, BASE)})
    if other:
        raise ValueError(f"the stratum is defined on the base policy, got rows of {other}")
    rows = stratum_rows(items, min_words)
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("a record appears more than once among the stratum rows")
    return {
        "min_words": int(min_words),
        "n_by_split": {
            split: sum(row["split"] == split for row in rows) for split in ENDPOINT_SPLITS
        },
        "base_sycophantic": sum(follows_opinion(row) for row in rows),
        "record_ids": ids,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """CLI: freeze the stratum from the base policy's re-evaluation items."""
    parser = argparse.ArgumentParser(description="reply-floor arm of PREREG v3")
    parser.add_argument(
        "--freeze-stratum",
        metavar="ITEMS",
        help="items.jsonl of the base policy's re-evaluation; writes the stratum file",
    )
    parser.add_argument("--out", default=STRATUM_FILE)
    args = parser.parse_args(argv)
    if not args.freeze_stratum:
        parser.error("pass --freeze-stratum <base items.jsonl>")
    payload = freeze_stratum(read_jsonl(args.freeze_stratum))
    n = len(payload["record_ids"])
    counts = ", ".join(f"{split} {count}" for split, count in payload["n_by_split"].items())
    summary = f"{n} questions ({counts}), {payload['base_sycophantic']} sycophantic base replies"
    if n != STRATUM_SIZE:
        raise SystemExit(f"stratum: {summary}; the registered size is {STRATUM_SIZE}, not written")
    write_json(args.out, payload)
    print(f"stratum: {summary}")
    print(f"wrote {Path(args.out).name}")
    return payload


if __name__ == "__main__":
    main()

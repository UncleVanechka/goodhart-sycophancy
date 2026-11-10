"""Rating helpers for the analysis stages, and re-rating saved replies with other judges.

:func:`rate_texts` rates aligned (record, text) pairs under any persona with the same backend
calls :func:`~goodhart.simulator.reward.immediate_reward` and ``informed_reward`` make, so a
text rated in two stages is one cache entry with one score. :func:`build_rating_backend` is
``build_backend`` with a :class:`JudgeProbe` between the cache and the judge: the probe counts
what the judge actually scored and, for the in-process judge, keeps its raw replies, which
separates a parsed score from the fallback a malformed reply gets.

The CLI re-rates per-item files written by ``reeval`` and ``sample_rate`` and the Best-of-N
per-sample files. Every judge in ``judges`` (overrides merged onto the simulator config, for
example another ``hf.model``) rates every input under every persona of ``personas``. Output:
``<out>/<judge>/<input>.jsonl`` with the row keys, the parsed letter and ``ratings`` /
``malformed`` per persona, and ``<out>/<judge>/summary.json`` with per-opinion means, the
rating gap between agreeing and correct replies under a wrong opinion, and malformed counts.
Each input row is matched to its prompt record by ``record_id``; a row that carries ``gold``
must agree with the record, or the job stops before any judge is loaded.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from goodhart.config import deep_merge, load_config
from goodhart.eval.answer_parse import parse_answer
from goodhart.io import (
    load_dotenv_if_present,
    read_json,
    read_jsonl,
    results_payload,
    write_json,
    write_jsonl,
)
from goodhart.simulator.backend import (
    CachedBackend,
    ScoringBackend,
    build_backend,
    judge_identity,
    opinion_letter,
    parse_score,
)
from goodhart.simulator.persona import Persona, get_persona

LOGGER = logging.getLogger(__name__)

RewardFn = Callable[[list[dict], list[str]], list[float]]

OPINIONS = ("none", "correct", "wrong")
# keys that identify a row across stages; the others are recomputed from the record
ROW_KEYS = ("policy", "split", "record_id", "sample_idx")


# ----------------------------------------------------------------------------- rating


def rate_texts(
    backend: ScoringBackend, persona_name: str, records: Sequence[dict], texts: Sequence[str]
) -> list[float]:
    """One rating per text under ``persona_name``, aligned with ``records``.

    A persona that knows the gold is instantiated once per gold letter, exactly as the reward
    functions do, so the cache keys and scores are the ones training and Best-of-N used.
    """
    if len(records) != len(texts):
        raise ValueError(f"{len(records)} records but {len(texts)} texts")
    knows_truth = get_persona(persona_name, gold="A").knows_truth
    groups: dict[str | None, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[record["gold"] if knows_truth else None].append(i)
    out = [0.0] * len(records)
    for gold, idx in groups.items():
        persona = get_persona(persona_name, gold=gold)
        scores = backend.score_batch(
            persona,
            [records[i]["prompt"] for i in idx],
            [texts[i] for i in idx],
            [records[i]["gold"] for i in idx],
        )
        if len(scores) != len(idx):
            raise ValueError(f"backend returned {len(scores)} scores for {len(idx)} texts")
        for i, score in zip(idx, scores, strict=True):
            out[i] = float(score)
    return out


def persona_rater(backend: ScoringBackend, persona_name: str) -> RewardFn:
    """``reward_fn(records, texts)`` that rates under one persona (see :func:`rate_texts`)."""

    def rate(records: list[dict], texts: list[str]) -> list[float]:
        return rate_texts(backend, persona_name, records, texts)

    rate.__name__ = f"rate_{persona_name}"
    return rate


def is_malformed(reply: str) -> bool:
    """True when the judge reply carries no score the backend can parse."""
    try:
        parse_score(reply)
    except ValueError:
        return True
    return False


class JudgeProbe:
    """Pass-through backend that records what the wrapped judge actually scored.

    Placed between the cache and the judge it sees only cache misses: ``scored`` counts them.
    When the judge renders and decodes in batches itself (``_complete_batch``, the in-process
    judge), the raw reply of every scored item is kept, keyed by the persona prompt, the user
    prompt and the response; :meth:`malformed` tells whether that reply had a score.
    """

    def __init__(self, inner: ScoringBackend) -> None:
        self.inner = inner
        self.scored = 0
        self.replies: dict[tuple[str, str, str], str] = {}
        self._capture: list[str] | None = None
        complete = getattr(inner, "_complete_batch", None)
        self.records_replies = callable(complete)
        if self.records_replies:

            def complete_and_keep(texts: Sequence[str]) -> list[str]:
                replies = list(complete(texts))
                if self._capture is not None:
                    self._capture.extend(replies)
                return replies

            inner._complete_batch = complete_and_keep  # type: ignore[attr-defined]

    def score(self, persona: Persona, prompt: str, response: str, gold: str | None = None) -> float:
        return self.score_batch(persona, [prompt], [response], [gold])[0]

    def score_batch(
        self,
        persona: Persona,
        prompts: Sequence[str],
        responses: Sequence[str],
        golds: Sequence[str | None] | None = None,
    ) -> list[float]:
        self._capture = [] if self.records_replies else None
        try:
            scores = [float(s) for s in self.inner.score_batch(persona, prompts, responses, golds)]
            captured = self._capture
        finally:
            self._capture = None
        self.scored += len(prompts)
        if captured is not None and len(captured) == len(prompts):
            for prompt, response, reply in zip(prompts, responses, captured, strict=True):
                self.replies[(persona.system_prompt, prompt, response)] = reply
        return scores

    def reply(self, persona: Persona, prompt: str, response: str) -> str | None:
        """Raw judge reply for one item; ``None`` when it came from the cache or is unknown."""
        return self.replies.get((persona.system_prompt, prompt, response))

    def malformed(self, persona: Persona, prompt: str, response: str) -> bool | None:
        """Whether the judge's reply lacked a score; ``None`` when the reply is unknown."""
        reply = self.reply(persona, prompt, response)
        return None if reply is None else is_malformed(reply)

    def clear(self) -> None:
        """Forget the recorded replies (the counter is kept)."""
        self.replies.clear()


def build_rating_backend(sim_cfg: dict) -> tuple[ScoringBackend, JudgeProbe]:
    """``build_backend`` with a :class:`JudgeProbe` under the cache; returns (backend, probe).

    The cache is keyed by the same judge identity ``build_backend`` uses, so it shares
    entries with every other stage.
    """
    if "backend" not in sim_cfg and isinstance(sim_cfg.get("simulator"), dict):
        sim_cfg = sim_cfg["simulator"]
    kind = sim_cfg.get("backend", "mock")
    probe = JudgeProbe(build_backend({**sim_cfg, "cache": None}))
    cache = sim_cfg.get("cache")
    if not cache:
        return probe, probe
    judge = judge_identity(kind, sim_cfg.get(kind) or {})
    return CachedBackend(probe, cache, judge=judge), probe


def malformed_flags(
    probe: JudgeProbe | None, persona_name: str, records: Sequence[dict], texts: Sequence[str]
) -> list[bool | None]:
    """Per-item malformed flag of the last ratings under ``persona_name`` (``None`` = unknown)."""
    if probe is None:
        return [None] * len(records)
    flags: list[bool | None] = []
    for record, text in zip(records, texts, strict=True):
        persona = get_persona(persona_name, gold=record["gold"])
        flags.append(probe.malformed(persona, record["prompt"], text))
    return flags


def release_backend(backend: ScoringBackend) -> None:
    """Close the cache and free an in-process judge's GPU memory."""
    inner: Any = backend
    if isinstance(backend, CachedBackend):
        backend.close()
        inner = backend.inner
    inner = getattr(inner, "inner", inner)
    if hasattr(inner, "_model"):
        inner._model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------------- inputs


def data_path(cfg: dict, path: str | Path) -> Path:
    """``path`` as given when absolute, else under ``paths.data``."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path((cfg.get("paths") or {}).get("data", "data")) / candidate


def load_record_index(cfg: dict) -> dict[str, dict]:
    """Prompt records by id: ``cfg["data"]`` plus SycophancyEval when its file is present."""
    index = {record["id"]: record for record in read_jsonl(cfg["data"])}
    heldout = cfg.get("sycophancy_eval")
    if heldout:
        path = data_path(cfg, heldout)
        if path.is_file():
            from goodhart.data import sycophancy_eval

            seed = int(cfg.get("sycophancy_eval_letter_seed", 0))
            for record in sycophancy_eval.load_records(path, rng=random.Random(seed)):
                index.setdefault(record["id"], record)
    return index


def read_input_rows(cfg: dict, spec: dict) -> list[dict]:
    """Rows of one input: every file matching ``spec["path"]`` (a glob), in sorted order."""
    pattern = str(data_path(cfg, spec["path"]))
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths or not all(Path(p).is_file() for p in paths):
        raise FileNotFoundError(f"no input files for {spec['path']!r} (looked for {pattern})")
    rows: list[dict] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def filter_rows(rows: list[dict], records: dict[str, dict], spec: dict, seed: int) -> list[dict]:
    """Apply an input's filters: ``splits``, ``policies``, ``opinions``, ``max_samples`` (keep
    ``sample_idx`` below it) and ``max_rows`` (a seeded subsample that keeps the file order).

    Only rows that pass the split and policy filters need their record in ``records``, and
    those rows must agree with it on the gold letter (:func:`check_gold`).
    """
    splits = set(spec.get("splits") or [])
    policies = set(spec.get("policies") or [])
    rows = [
        row
        for row in rows
        if not (splits and row.get("split") is not None and row["split"] not in splits)
        and not (policies and row.get("policy") not in policies)
    ]
    missing = sorted({row["record_id"] for row in rows if row["record_id"] not in records})
    if missing:
        raise ValueError(f"{len(missing)} record ids not in the prompt data, e.g. {missing[:3]}")
    check_gold(rows, records)
    opinions = set(spec.get("opinions") or [])
    max_samples = spec.get("max_samples")
    kept = []
    for row in rows:
        if opinions and records[row["record_id"]]["opinion"] not in opinions:
            continue
        if max_samples is not None and int(row.get("sample_idx", 0)) >= int(max_samples):
            continue
        kept.append(row)
    max_rows = spec.get("max_rows")
    if max_rows is not None and len(kept) > int(max_rows):
        chosen = sorted(random.Random(seed).sample(range(len(kept)), int(max_rows)))
        kept = [kept[i] for i in chosen]
    return kept


def check_gold(rows: Sequence[dict], records: dict[str, dict]) -> None:
    """Raise when a row that carries ``gold`` disagrees with its rebuilt record.

    The records are rebuilt from the prompt data (and SycophancyEval with a fixed letter seed),
    so a mismatch means the row was written against other data or another letter order, and
    every rating and parsed-answer comparison of it would be wrong.
    """
    bad = [
        (row["record_id"], row["gold"], records[row["record_id"]]["gold"])
        for row in rows
        if row.get("gold") is not None and row["gold"] != records[row["record_id"]]["gold"]
    ]
    if bad:
        examples = ", ".join(f"{rid}: row {row} vs record {rec}" for rid, row, rec in bad[:3])
        raise ValueError(
            f"{len(bad)} rows carry a gold letter that differs from the rebuilt record "
            f"(other prompt data or letter seed?), e.g. {examples}"
        )


# ----------------------------------------------------------------------------- rescoring


def rescore_rows(
    rows: list[dict],
    records: dict[str, dict],
    backend: ScoringBackend,
    personas: Sequence[str],
    probe: JudgeProbe | None = None,
) -> list[dict]:
    """Rate the ``text`` of every row under every persona; one output row per input row."""
    check_gold(rows, records)
    recs = [records[row["record_id"]] for row in rows]
    texts = [str(row.get("text") or "") for row in rows]
    out = []
    for row, record, text in zip(rows, recs, texts, strict=True):
        item = {key: row[key] for key in ROW_KEYS if key in row}
        item.update(
            {
                "opinion": record["opinion"],
                "gold": record["gold"],
                "opinion_answer": record.get("opinion_answer"),
                "stated_letter": opinion_letter(record["prompt"]),
                "parsed": parse_answer(text, len(record["choices"])),
                "n_chars": len(text),
                "ratings": {},
                "malformed": {},
            }
        )
        out.append(item)
    for persona in personas:
        scores = rate_texts(backend, persona, recs, texts)
        flags = malformed_flags(probe, persona, recs, texts)
        for item, score, flag in zip(out, scores, flags, strict=True):
            item["ratings"][persona] = score
            item["malformed"][persona] = flag
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_rated(rows: list[dict], personas: Sequence[str]) -> dict[str, Any]:
    """Per persona: mean rating overall and per opinion type, the rating gap between agreeing
    and correct replies under a wrong opinion, and malformed counts."""
    out: dict[str, Any] = {"n": len(rows)}
    for persona in personas:
        rated = [row for row in rows if persona in row.get("ratings", {})]
        scores = [row["ratings"][persona] for row in rated]
        stats: dict[str, Any] = {"n": len(scores), "mean": _mean(scores)}
        for opinion in OPINIONS:
            subset = [row["ratings"][persona] for row in rated if row["opinion"] == opinion]
            if subset:
                stats[f"mean@{opinion}"] = _mean(subset)
        wrong = [row for row in rated if row["opinion"] == "wrong"]
        agree = [r["ratings"][persona] for r in wrong if r["parsed"] == r["opinion_answer"]]
        correct = [r["ratings"][persona] for r in wrong if r["parsed"] == r["gold"]]
        stats["mean_agree@wrong"] = _mean(agree)
        stats["mean_correct@wrong"] = _mean(correct)
        stats["n_agree@wrong"], stats["n_correct@wrong"] = len(agree), len(correct)
        if agree and correct:
            stats["gap@wrong"] = _mean(agree) - _mean(correct)  # type: ignore[operator]
        flags = [row.get("malformed", {}).get(persona) for row in rated]
        known = [flag for flag in flags if flag is not None]
        stats["malformed"] = sum(bool(flag) for flag in known)
        stats["malformed_known"] = len(known)
        stats["malformed_rate"] = stats["malformed"] / len(known) if known else None
        out[persona] = stats
    return out


def finite_or_none(obj: Any) -> Any:
    """``obj`` with every NaN or infinite float, at any depth, replaced by ``None``."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {key: finite_or_none(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [finite_or_none(value) for value in obj]
    return obj


def active_entries(mapping: dict | None) -> dict[str, Any]:
    """Entries of a config mapping minus those set to ``null`` (how an override drops one)."""
    return {key: value for key, value in (mapping or {}).items() if value is not None}


def write_rows_atomic(path: Path, rows: list[dict]) -> None:
    """Write JSONL through a ``.part`` file so an existing output is always complete."""
    partial = path.with_name(path.name + ".part")
    write_jsonl(partial, rows)
    os.replace(partial, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON through a ``.part`` file, so a ``summary.json`` that exists is complete."""
    partial = path.with_name(path.name + ".part")
    write_json(partial, obj)
    os.replace(partial, path)


def read_done(path: Path) -> Any | None:
    """A finished output for ``skip_done``: its parsed content, or ``None`` when the file is
    absent or cannot be read (truncated, not JSON), which means "not done"."""
    if not path.exists():
        return None
    try:
        return read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        LOGGER.warning("%s exists but cannot be read (%s): treated as not done", path, error)
        return None


def run(
    cfg: dict,
    backend_factory: Callable[[dict], tuple[ScoringBackend, JudgeProbe | None]] | None = None,
) -> dict[str, dict]:
    """Rescore every input with every judge; returns ``{judge: summary}``.

    An input with ``judges`` (a list of judge tags) is rated only by those judges.
    ``backend_factory(sim_cfg)`` builds the rating backend and its probe (tests pass a mock);
    by default :func:`build_rating_backend`. A judge with no input to rate is never loaded.
    """
    factory = backend_factory or build_rating_backend
    seed = int(cfg.get("seed", 0))
    records = load_record_index(cfg)
    sim_base = cfg["simulator"]
    if isinstance(sim_base, str):
        sim_base = load_config(sim_base, base=None)
    default_personas = list(cfg.get("personas") or ["agreeable"])
    inputs = active_entries(cfg.get("inputs"))
    judges = active_entries(cfg.get("judges")) or {"default": {}}
    out_root = Path(cfg.get("out", "results/rescore"))
    summaries: dict[str, dict] = {}
    for tag, overrides in judges.items():
        sim_cfg = deep_merge(sim_base, overrides or {})
        kind = sim_cfg.get("backend", "mock")
        identity = judge_identity(kind, sim_cfg.get(kind) or {})
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        backend: ScoringBackend | None = None
        probe: JudgeProbe | None = None
        per_input: dict[str, Any] = {}
        for name, spec in inputs.items():
            if spec.get("judges") and tag not in spec["judges"]:
                continue
            personas = list(spec.get("personas") or default_personas)
            target = out_dir / f"{name}.jsonl"
            done = read_done(target) if cfg.get("skip_done", True) else None
            if done is not None:
                LOGGER.info("judge %s / %s: output exists, not rescored", tag, name)
                per_input[name] = {"skipped": True, **summarize_rated(done, personas)}
                continue
            rows = filter_rows(read_input_rows(cfg, spec), records, spec, seed)
            if backend is None:
                backend, probe = factory(sim_cfg)
            started, scored_before = time.time(), probe.scored if probe else 0
            rated = rescore_rows(rows, records, backend, personas, probe)
            write_rows_atomic(target, rated)
            stats = summarize_rated(rated, personas)
            stats["judge_calls"] = (probe.scored - scored_before) if probe else None
            stats["seconds"] = round(time.time() - started, 1)
            per_input[name] = stats
            if probe is not None:
                probe.clear()
            LOGGER.info("judge %s / %s: %d rows in %.0fs", tag, name, len(rated), stats["seconds"])
            for persona in personas:
                p = stats[persona]
                print(
                    f"{tag} {name} {persona}: n={p['n']} mean={p['mean']} gap@wrong="
                    f"{p.get('gap@wrong')} malformed={p['malformed']}/{p['malformed_known']}"
                )
        if backend is not None:
            release_backend(backend)
        summaries[tag] = finite_or_none({"judge": identity, "inputs": per_input})
        payload = results_payload(cfg, judge_tag=tag, **summaries[tag])
        write_json_atomic(out_dir / "summary.json", payload)
    return summaries


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m goodhart.experiments.rescore --config configs/rescore.yaml``."""
    parser = argparse.ArgumentParser(description="re-rate saved replies with other judges")
    parser.add_argument("--config", default="configs/rescore.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv_if_present()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

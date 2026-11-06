"""Experiment 1: one greedy answer per record for every model, metrics overall and per opinion.

Writes ``results/baseline/<model_slug>.json`` (``{"config", "git_commit", "model", "metrics"}``)
and ``<model_slug>_responses.jsonl`` with the raw texts next to it.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from goodhart.config import load_config, set_seed
from goodhart.eval.answer_parse import parse_answer
from goodhart.eval.metrics import compute_metrics, records_to_frame
from goodhart.io import (
    load_dotenv_if_present,
    model_slug,
    read_jsonl,
    results_payload,
    write_json,
    write_jsonl,
)
from goodhart.policy.generate import Generator, build_generator
from goodhart.simulator.backend import ScoringBackend, build_backend
from goodhart.simulator.reward import immediate_reward

OPINION_TYPES = ("none", "correct", "wrong")


def select_records(
    records: list[dict], max_items: int | None, seed: int, split: str | None = None
) -> list[dict]:
    """Keep up to ``max_items`` records per opinion type, chosen by shuffling ``base_id``s.

    Variants of one base question are taken together so paired metrics (``flip_rate``) stay
    computable. ``max_items=None`` keeps everything. Output order is by ``base_id`` then opinion.
    """
    if split is not None:
        records = [r for r in records if r.get("split") == split]
    by_base: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_base[record["base_id"]].append(record)
    base_ids = sorted(by_base)
    random.Random(seed).shuffle(base_ids)

    taken: dict[str, int] = defaultdict(int)
    chosen: list[dict] = []
    for base_id in base_ids:
        for record in by_base[base_id]:
            opinion = record["opinion"]
            if max_items is not None and taken[opinion] >= max_items:
                continue
            taken[opinion] += 1
            chosen.append(record)
    order = {name: i for i, name in enumerate(OPINION_TYPES)}
    chosen.sort(key=lambda r: (r["base_id"], order.get(r["opinion"], len(order)), r["id"]))
    return chosen


def load_records(cfg: dict) -> list[dict]:
    """Read ``cfg["data"]`` and apply ``split``, ``max_items`` and ``seed`` from the config."""
    records = read_jsonl(cfg["data"])
    return select_records(
        records, cfg.get("max_items"), int(cfg.get("seed", 0)), split=cfg.get("split")
    )


def greedy_responses(generator: Generator, records: list[dict], max_tokens: int) -> list[str]:
    """One greedy sample per record (``n=1``, ``temperature=0``)."""
    prompts = [record["prompt"] for record in records]
    samples = generator.generate(prompts, n=1, temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    return [texts[0] if texts else "" for texts in samples]


def evaluate_responses(
    records: list[dict], texts: list[str], backend: ScoringBackend | None = None
) -> tuple[dict[str, float], list[dict]]:
    """Parse answers, optionally score the proxy, and compute metrics.

    Returns ``(metrics, rows)`` where ``rows`` carry ``record_id, text, parsed, proxy``.
    """
    if len(records) != len(texts):
        raise ValueError(f"{len(records)} records but {len(texts)} responses")
    parsed = [
        parse_answer(text, len(record["choices"]))
        for record, text in zip(records, texts, strict=True)
    ]
    proxy = immediate_reward(backend, records, texts) if backend is not None else None
    frame = records_to_frame(records, parsed, proxy)
    metrics = compute_metrics(frame)
    rows = [
        {
            "record_id": record["id"],
            "text": text,
            "parsed": letter,
            "proxy": None if proxy is None else proxy[i],
        }
        for i, (record, text, letter) in enumerate(zip(records, texts, parsed, strict=True))
    ]
    return metrics, rows


def run_model(
    cfg: dict,
    model: str,
    records: list[dict],
    generator: Generator,
    out_dir: str | Path,
    backend: ScoringBackend | None = None,
) -> dict:
    """Greedy-decode ``records`` with one model, evaluate, write the results, return the payload."""
    out_dir = Path(out_dir)
    max_tokens = int(cfg.get("generation", {}).get("max_tokens", 512))
    texts = greedy_responses(generator, records, max_tokens)
    metrics, rows = evaluate_responses(records, texts, backend)
    slug = model_slug(model)
    responses_file = out_dir / f"{slug}_responses.jsonl"
    write_jsonl(responses_file, rows)
    payload = results_payload(
        cfg,
        model=model,
        n_records=len(records),
        metrics=metrics,
        responses_file=str(responses_file),
    )
    write_json(out_dir / f"{slug}.json", payload)
    return payload


def run(cfg: dict, backend: ScoringBackend | None = None) -> dict[str, dict]:
    """Run the baseline for every model in ``cfg["models"]``; returns ``{model: payload}``."""
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    records = load_records(cfg)
    if backend is None and cfg.get("simulator"):
        backend = build_backend(load_config(cfg["simulator"]))
    out_dir = Path(cfg.get("out", "results/bon")).parent / "baseline"
    if "baseline_out" in cfg:
        out_dir = Path(cfg["baseline_out"])
    results: dict[str, dict] = {}
    for model in cfg["models"]:
        generator = build_generator(cfg, seed, model=model)
        results[model] = run_model(cfg, model, records, generator, out_dir, backend)
        print(f"{model}: {_summary(results[model]['metrics'])}")
    return results


def _summary(metrics: dict[str, float]) -> str:
    keys = ("gold_accuracy", "sycophancy_rate", "flip_rate", "proxy_mean", "abstain_rate")
    parts = [f"{key}={metrics[key]:.3f}" for key in keys if metrics.get(key) is not None]
    return " ".join(parts)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bon.yaml")
    parser.add_argument("--no-proxy", action="store_true", help="skip the simulator rating")
    args = parser.parse_args(argv)
    load_dotenv_if_present()
    cfg = load_config(args.config)
    if args.no_proxy:
        cfg = {**cfg, "simulator": None}
    run(cfg)


if __name__ == "__main__":
    main()

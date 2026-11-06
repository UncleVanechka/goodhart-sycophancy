"""Experiment 2: Best-of-N against the proxy reward, with the delayed-reward mitigation.

Every prompt is sampled once (``generation.n`` samples), every sample is rated by the immediate
judge (``agreeable``) and the informed judge (``agreeable_informed``, told the gold answer), and
the rows go to ``<model_slug>_samples.jsonl``. Selection for any ``alpha`` and ``N`` is then a
re-read of that file: pick the sample maximising ``(1 - alpha) * immediate + alpha * informed``
among the first ``N`` and compute the metrics of the picked samples. An existing samples file
is reused by default; ``--resample`` forces fresh sampling and scoring.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from goodhart.config import load_config, set_seed
from goodhart.eval.answer_parse import parse_answer
from goodhart.eval.metrics import compute_metrics, records_to_frame
from goodhart.experiments.baseline import load_records
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
from goodhart.simulator.persona import get_persona
from goodhart.simulator.reward import immediate_reward

INFORMED_PERSONA = "agreeable_informed"


def select_best_of_n(scores: Sequence[float], n: int, rng: random.Random) -> int:
    """Index of the highest score among the first ``n`` samples; ties are broken by ``rng``.

    ``n`` larger than ``len(scores)`` uses every sample; ``n < 1`` or no samples is an error.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    pool = list(scores[:n])
    if not pool:
        raise ValueError("select_best_of_n needs at least one score")
    best = max(pool)
    candidates = [i for i, score in enumerate(pool) if score == best]
    if len(candidates) == 1:
        return candidates[0]
    return rng.choice(candidates)


def mixed_score(immediate: float, informed: float, alpha: float) -> float:
    """``(1 - alpha) * immediate + alpha * informed``."""
    return (1.0 - alpha) * immediate + alpha * informed


def sample_responses(generator: Generator, records: list[dict], gen_cfg: dict) -> list[list[str]]:
    """``gen_cfg["n"]`` samples per record with the sampling parameters from the config."""
    prompts = [record["prompt"] for record in records]
    return generator.generate(
        prompts,
        n=int(gen_cfg.get("n", 1)),
        temperature=float(gen_cfg.get("temperature", 1.0)),
        top_p=float(gen_cfg.get("top_p", 1.0)),
        max_tokens=int(gen_cfg.get("max_tokens", 512)),
    )


def informed_reward(
    backend: ScoringBackend, records: list[dict], responses: list[str]
) -> list[float]:
    """Rating by the ``agreeable_informed`` persona, which is told each record's gold letter.

    Records are grouped by gold letter so one persona instance serves one batch.
    """
    by_gold: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        by_gold[record["gold"]].append(i)
    out: list[float] = [0.0] * len(records)
    for gold, indices in by_gold.items():
        persona = get_persona(INFORMED_PERSONA, gold=gold)
        scores = backend.score_batch(
            persona,
            [records[i]["prompt"] for i in indices],
            [responses[i] for i in indices],
            [gold] * len(indices),
        )
        for i, score in zip(indices, scores, strict=True):
            out[i] = float(score)
    return out


def score_samples(
    records: list[dict], samples: list[list[str]], backend: ScoringBackend
) -> list[dict]:
    """Flatten ``samples`` into rows with the parsed letter and both judges' ratings.

    Rows are ordered by record then ``sample_idx`` and carry
    ``record_id, sample_idx, text, parsed, immediate, informed``.
    """
    if len(records) != len(samples):
        raise ValueError(f"{len(records)} records but {len(samples)} sample lists")
    flat_records: list[dict] = []
    flat_texts: list[str] = []
    rows: list[dict] = []
    for record, texts in zip(records, samples, strict=True):
        for sample_idx, text in enumerate(texts):
            flat_records.append(record)
            flat_texts.append(text)
            rows.append(
                {
                    "record_id": record["id"],
                    "sample_idx": sample_idx,
                    "text": text,
                    "parsed": parse_answer(text, len(record["choices"])),
                }
            )
    immediate = immediate_reward(backend, flat_records, flat_texts)
    informed = informed_reward(backend, flat_records, flat_texts)
    for row, imm, inf in zip(rows, immediate, informed, strict=True):
        row["immediate"] = float(imm)
        row["informed"] = float(inf)
    return rows


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """``record_id -> rows`` sorted by ``sample_idx``."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["record_id"]].append(row)
    for record_rows in grouped.values():
        record_rows.sort(key=lambda r: r["sample_idx"])
    return dict(grouped)


def select_and_evaluate(
    records: list[dict],
    rows: list[dict],
    alphas: Sequence[float],
    n_values: Sequence[int],
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Metrics of the Best-of-N pick for every ``alpha`` and ``N``.

    Selection uses the mixed reward; the proxy reported to the metrics is the immediate rating
    of the selected sample. Returns ``{"alpha=0.3": {"4": metrics, ...}, ...}``.
    """
    grouped = group_rows(rows)
    missing = [record["id"] for record in records if record["id"] not in grouped]
    if missing:
        raise ValueError(f"no samples for {len(missing)} records, e.g. {missing[:3]}")
    n_available = min(len(grouped[record["id"]]) for record in records)
    if max(n_values) > n_available:
        raise ValueError(
            f"largest N={max(n_values)} exceeds {n_available} samples per record;"
            " raise generation.n and resample"
        )
    curves: dict[str, dict[str, dict[str, float]]] = {}
    for alpha in alphas:
        alpha = float(alpha)
        per_n: dict[str, dict[str, float]] = {}
        for n in n_values:
            rng = random.Random(f"{seed}-{alpha}-{n}")
            parsed: list[str | None] = []
            proxy: list[float] = []
            for record in records:
                record_rows = grouped[record["id"]]
                scores = [mixed_score(r["immediate"], r["informed"], alpha) for r in record_rows]
                chosen = record_rows[select_best_of_n(scores, int(n), rng)]
                parsed.append(chosen["parsed"])
                proxy.append(chosen["immediate"])
            frame = records_to_frame(records, parsed, proxy)
            per_n[str(int(n))] = compute_metrics(frame)
        curves[f"alpha={alpha}"] = per_n
    return curves


def run_model(
    cfg: dict,
    model: str,
    records: list[dict],
    generator: Generator,
    backend: ScoringBackend,
    out_dir: str | Path,
    reuse_samples: bool = True,
) -> dict:
    """Sample, score, select and write ``<slug>.json`` plus ``<slug>_samples.jsonl``.

    An existing samples file is re-read by default (``reuse_samples``); pass ``False`` to
    sample and score again. Fresh sampling requires ``generation.n >= max(n_values)``.
    """
    out_dir = Path(out_dir)
    slug = model_slug(model)
    samples_file = out_dir / f"{slug}_samples.jsonl"
    n_values = [int(n) for n in cfg.get("n_values", [1])]
    gen_cfg = cfg.get("generation", {})
    if reuse_samples and samples_file.exists():
        rows = read_jsonl(samples_file)
    else:
        n_samples = int(gen_cfg.get("n", 1))
        if max(n_values) > n_samples:
            raise ValueError(
                f"largest N={max(n_values)} exceeds generation.n={n_samples}"
            )
        samples = sample_responses(generator, records, gen_cfg)
        rows = score_samples(records, samples, backend)
        write_jsonl(samples_file, rows)
    alphas = [float(a) for a in cfg.get("mitigation", {}).get("alphas", [0.0])]
    curves = select_and_evaluate(records, rows, alphas, n_values, int(cfg.get("seed", 0)))
    payload = results_payload(
        cfg,
        model=model,
        n_records=len(records),
        n_values=n_values,
        curves=curves,
        samples_file=str(samples_file),
    )
    write_json(out_dir / f"{slug}.json", payload)
    return payload


def run(
    cfg: dict, backend: ScoringBackend | None = None, reuse_samples: bool = True
) -> dict[str, dict]:
    """Run Best-of-N for every model in ``cfg["models"]``; returns ``{model: payload}``."""
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    records = load_records(cfg)
    if backend is None:
        backend = build_backend(load_config(cfg["simulator"]))
    out_dir = Path(cfg.get("out", "results/bon"))
    results: dict[str, dict] = {}
    for model in cfg["models"]:
        generator = build_generator(cfg, seed, model=model)
        results[model] = run_model(
            cfg, model, records, generator, backend, out_dir, reuse_samples=reuse_samples
        )
        _print_curves(model, results[model]["curves"])
    return results


def _print_curves(model: str, curves: dict[str, dict[str, dict[str, float]]]) -> None:
    print(model)
    for alpha_key, per_n in curves.items():
        for n, metrics in per_n.items():
            print(
                f"  {alpha_key} N={n}: proxy={metrics.get('proxy_mean', float('nan')):.2f}"
                f" gold={metrics.get('gold_accuracy', float('nan')):.3f}"
                f" sycophancy={metrics.get('sycophancy_rate', float('nan')):.3f}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bon.yaml")
    parser.add_argument(
        "--resample",
        action="store_true",
        help="sample and score again even when a samples file exists (default: reuse it)",
    )
    args = parser.parse_args(argv)
    load_dotenv_if_present()
    cfg = load_config(args.config)
    run(cfg, reuse_samples=not args.resample)


if __name__ == "__main__":
    main()

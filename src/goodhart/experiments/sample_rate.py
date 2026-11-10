"""Sampled replies of the base policy and saved adapters, rated sample by sample.

For each policy, ``generation.n`` replies per prompt are drawn with GRPO's rollout settings
(temperature 1, top-p 1, top-k off, the GRPO completion budget, no thinking block) through
``best_of_n.sample_responses``, on ``max_questions`` questions of ``splits`` and the opinion
types in ``opinions``. Every sample is rated under every persona in ``personas``. The rows of
``<out>/<policy>/samples.jsonl`` follow the Best-of-N per-sample format (``record_id,
sample_idx, text, parsed, immediate, informed``) and add ``ratings`` with every persona, so
Best-of-N selection and the break-even analyses read them unchanged. ``summary.json`` holds
the per-sample metrics, mean ratings per opinion type and, when both the immediate and the
informed judge rated, the Best-of-N curves for ``n_values`` up to ``generation.n``.

The sampler matches GRPO's in its settings but not in the model's mode: TRL drew the rollouts
from the policy in train mode, so the LoRA dropout was active, while :func:`sample_generate`
puts the model in eval mode (no dropout). Samples of a saved adapter are therefore draws from
the adapter itself, not exactly from the distribution its rollouts came from during training.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

from goodhart.config import load_config, set_seed
from goodhart.eval.answer_parse import parse_answer
from goodhart.eval.metrics import compute_metrics, records_to_frame
from goodhart.experiments.baseline import select_records
from goodhart.experiments.best_of_n import sample_responses, select_and_evaluate
from goodhart.experiments.reeval import PeftPolicyBank, PolicyBank, resolve_policies
from goodhart.experiments.rescore import (
    JudgeProbe,
    build_rating_backend,
    finite_or_none,
    malformed_flags,
    rate_texts,
    read_done,
    summarize_rated,
    write_json_atomic,
    write_rows_atomic,
)
from goodhart.io import load_dotenv_if_present, read_jsonl, results_payload
from goodhart.simulator.backend import ScoringBackend

LOGGER = logging.getLogger(__name__)

# persona whose rating is copied to the Best-of-N field of the same name
BON_FIELDS = {"agreeable": "immediate", "agreeable_informed": "informed"}


def sample_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    n: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    seed: int,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[list[str]]:
    """``n`` sampled chat completions per prompt through an HF model, ``batch_size`` prompts
    per ``generate`` call.

    The sampling settings are set on the model's generation config for the duration of the
    call, as TRL does for its rollouts, so defaults shipped with the model (Qwen3: top-k 20,
    top-p 0.95, temperature 0.6) cannot leak in. ``top_k=0`` disables top-k filtering. Unlike
    TRL's rollouts, which ran in train mode with the LoRA dropout active, the model is put in
    eval mode.
    """
    import torch

    template_kwargs = dict(chat_template_kwargs or {})
    device = next(model.parameters()).device
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    settings: dict[str, Any] = {
        "do_sample": True,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": None,
        "repetition_penalty": 1.0,
        "max_new_tokens": int(max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    target = model.get_base_model() if hasattr(model, "get_base_model") else model
    saved = target.generation_config
    override = copy.deepcopy(saved)
    for key, value in settings.items():
        setattr(override, key, value)
    target.generation_config = override
    model.eval()
    torch.manual_seed(int(seed))
    out: list[list[str]] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            chat = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                for prompt in chunk
            ]
            encoded = tokenizer(chat, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                generated = model.generate(**encoded, num_return_sequences=n, **settings)
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            out.extend(texts[i * n : (i + 1) * n] for i in range(len(chunk)))
    finally:
        target.generation_config = saved
        tokenizer.padding_side = padding_side
    return out


class BankGenerator:
    """The Best-of-N ``Generator`` protocol over the active policy of a :class:`PolicyBank`."""

    def __init__(self, bank: PolicyBank, batch_size: int, top_k: int, seed: int) -> None:
        self.bank = bank
        self.batch_size = batch_size
        self.top_k = top_k
        self.seed = seed

    def generate(
        self, prompts: list[str], n: int, temperature: float, top_p: float, max_tokens: int
    ) -> list[list[str]]:
        return self.bank.sample(
            prompts,
            n=n,
            batch_size=self.batch_size,
            temperature=temperature,
            top_p=top_p,
            top_k=self.top_k,
            max_tokens=max_tokens,
            seed=self.seed,
        )


def select_sample_records(records: list[dict], spec: dict, seed: int) -> list[dict]:
    """Records for one policy: ``max_questions`` questions per split (``select_records``, so
    all opinion variants of a question stay together), keeping the types in ``opinions``."""
    opinions = set(spec.get("opinions") or ["none", "correct", "wrong"])
    chosen: list[dict] = []
    for split in spec.get("splits") or ["val"]:
        pool = [r for r in records if r.get("split") == split and r["opinion"] in opinions]
        chosen.extend(select_records(pool, spec.get("max_questions"), seed))
    return chosen


def sample_rows(
    policy: str,
    records: list[dict],
    samples: list[list[str]],
    ratings: dict[str, list[float]],
    malformed: dict[str, list[bool | None]],
) -> list[dict]:
    """Flatten samples into Best-of-N style rows ordered by record, then ``sample_idx``."""
    rows: list[dict] = []
    flat = 0
    for record, texts in zip(records, samples, strict=True):
        for sample_idx, text in enumerate(texts):
            row: dict[str, Any] = {
                "policy": policy,
                "split": record.get("split"),
                "record_id": record["id"],
                "base_id": record["base_id"],
                "opinion": record["opinion"],
                "sample_idx": sample_idx,
                "text": text,
                "parsed": parse_answer(text, len(record["choices"])),
                "gold": record["gold"],
                "opinion_answer": record.get("opinion_answer"),
                "ratings": {persona: scores[flat] for persona, scores in ratings.items()},
                "malformed": {persona: flags[flat] for persona, flags in malformed.items()},
            }
            for persona, field in BON_FIELDS.items():
                if persona in ratings:
                    row[field] = ratings[persona][flat]
            rows.append(row)
            flat += 1
    return rows


def summarize_samples(
    records: list[dict],
    rows: list[dict],
    personas: list[str],
    n_values: list[int],
    alphas: list[float],
    seed: int = 0,
) -> dict[str, Any]:
    """Per-sample metrics, rating means and, with both judges, Best-of-N curves for the
    ``n_values`` that do not exceed the samples per prompt."""
    by_id = {record["id"]: record for record in records}
    flat_records = [by_id[row["record_id"]] for row in rows]
    proxy = [row["immediate"] for row in rows] if "agreeable" in personas else None
    out: dict[str, Any] = {
        "metrics": compute_metrics(
            records_to_frame(flat_records, [row["parsed"] for row in rows], proxy)
        ),
        "ratings": summarize_rated(rows, personas),
    }
    if rows and "agreeable" in personas and "agreeable_informed" in personas:
        counts = Counter(row["record_id"] for row in rows)
        per_record = min(counts.get(record["id"], 0) for record in records)
        usable = sorted({int(v) for v in n_values if 1 <= int(v) <= per_record}) or [1]
        out["curves"] = select_and_evaluate(records, rows, alphas, usable, seed)
    return finite_or_none(out)


def sample_policy(
    cfg: dict,
    name: str,
    spec: dict,
    records: list[dict],
    bank: PolicyBank,
    backend: ScoringBackend,
    probe: JudgeProbe | None = None,
) -> dict[str, Any]:
    """Sample, rate and write one policy; returns its summary payload."""
    started = time.time()
    gen_cfg = dict(cfg.get("generation") or {})
    if spec.get("n") is not None:
        gen_cfg["n"] = int(spec["n"])
    seed = int(cfg.get("seed", 0))
    personas = list(spec["personas"] if "personas" in spec else cfg.get("personas") or [])
    bank.activate(name, spec["adapter"])
    generator = BankGenerator(
        bank,
        batch_size=int(gen_cfg.get("batch_size", 8)),
        top_k=int(gen_cfg.get("top_k", 0)),
        seed=seed,
    )
    samples = sample_responses(generator, records, gen_cfg)
    flat_records = [record for record, texts in zip(records, samples, strict=True) for _ in texts]
    flat_texts = [text for texts in samples for text in texts]
    ratings = {
        persona: rate_texts(backend, persona, flat_records, flat_texts) for persona in personas
    }
    flags = {
        persona: malformed_flags(probe, persona, flat_records, flat_texts) for persona in personas
    }
    if probe is not None:
        probe.clear()
    rows = sample_rows(name, records, samples, ratings, flags)
    out_dir = Path(cfg.get("out", "results/sample_rate")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows_atomic(out_dir / "samples.jsonl", rows)
    payload = results_payload(
        cfg,
        policy=name,
        adapter=spec["adapter"],
        personas=personas,
        n_records=len(records),
        n_samples=len(rows),
        seconds=round(time.time() - started, 1),
        **summarize_samples(
            records,
            rows,
            personas,
            n_values=[int(v) for v in cfg.get("n_values", [1, gen_cfg.get("n", 1)])],
            alphas=[float(a) for a in (cfg.get("mitigation") or {}).get("alphas", [0.0])],
            seed=seed,
        ),
    )
    write_json_atomic(out_dir / "summary.json", payload)
    return payload


def run(
    cfg: dict, backend: ScoringBackend | None = None, bank: PolicyBank | None = None
) -> dict[str, dict]:
    """Sample and rate every policy in ``cfg["policies"]``; returns ``{policy: summary}``.

    Each policy entry may override ``personas``, ``splits``, ``opinions``, ``max_questions``
    and ``n``; a policy whose ``summary.json`` exists is read back (``skip_done``), and one whose
    summary cannot be read is sampled again.
    """
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    policies = resolve_policies(cfg)
    all_records = read_jsonl(cfg["data"])
    probe: JudgeProbe | None = None
    if backend is None:
        sim_cfg = cfg["simulator"]
        if isinstance(sim_cfg, str):
            sim_cfg = load_config(sim_cfg, base=None)
        backend, probe = build_rating_backend(sim_cfg)
    bank = bank or PeftPolicyBank.from_config(cfg)
    defaults = {key: cfg.get(key) for key in ("splits", "opinions", "max_questions")}
    out_root = Path(cfg.get("out", "results/sample_rate"))
    results: dict[str, dict] = {}
    for name, spec in policies.items():
        done = read_done(out_root / name / "summary.json") if cfg.get("skip_done", True) else None
        if done is not None:
            LOGGER.info("policy %s: summary exists, not sampled again", name)
            results[name] = done
            continue
        selection = {**defaults, **{k: spec[k] for k in defaults if k in spec}}
        records = select_sample_records(all_records, selection, seed)
        results[name] = sample_policy(cfg, name, spec, records, bank, backend, probe)
        metrics = results[name].get("metrics") or {}
        print(
            f"{name}: {len(records)} prompts x {results[name]['n_samples'] // max(len(records), 1)}"
            f" samples, sycophancy={metrics.get('sycophancy_rate')}"
            f" proxy={metrics.get('proxy_mean')} ({results[name]['seconds']}s)"
        )
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m goodhart.experiments.sample_rate --config configs/sample_rate.yaml``."""
    parser = argparse.ArgumentParser(description="sample and rate replies of saved adapters")
    parser.add_argument("--config", default="configs/sample_rate.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv_if_present()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

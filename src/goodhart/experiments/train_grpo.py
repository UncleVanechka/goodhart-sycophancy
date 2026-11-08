"""GRPO training against the simulated-user proxy reward (DESIGN §9).

Pure pieces (dataset building, the TRL reward adapter, evaluation bookkeeping,
sweep expansion) have no heavy dependencies and are unit-tested offline. The
trainer, LoRA and W&B wiring live in ``train_one`` and ``main`` and import
``trl`` / ``peft`` / ``transformers`` lazily.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

RewardFn = Callable[[list[dict], list[str]], list[float]]
ParseFn = Callable[[str, int], str | None]
LogFn = Callable[[dict[str, float]], None]
GenerateFn = Callable[[list[str]], list[str]]

OPINION_TYPES = ("none", "correct", "wrong")


def build_train_dataset(records: Iterable[dict], split: str = "train") -> list[dict]:
    """Turn prompt records of ``split`` into TRL rows: chat prompt plus ``record_id``."""
    return [
        {"prompt": [{"role": "user", "content": record["prompt"]}], "record_id": record["id"]}
        for record in records
        if record.get("split") == split
    ]


def completion_text(completion: Any) -> str:
    """Extract the assistant text from a TRL completion (chat messages or plain string)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    parts = [
        str(message.get("content", ""))
        for message in completion
        if isinstance(message, dict) and message.get("role", "assistant") == "assistant"
    ]
    return "".join(parts)


def _default_parse() -> ParseFn:
    from goodhart.eval.answer_parse import parse_answer

    return parse_answer


def batch_stats(
    records: list[dict], parsed: list[str | None], scores: list[float]
) -> dict[str, float]:
    """Gold accuracy, sycophancy rate, abstain rate and proxy mean of one reward batch."""
    if not records:
        return {}
    pairs = list(zip(parsed, records, strict=True))
    stats: dict[str, float] = {
        "reward/gold_accuracy": sum(p == r["gold"] for p, r in pairs) / len(records),
        "reward/abstain_rate": sum(p is None for p in parsed) / len(records),
        "reward/proxy_mean": sum(scores) / len(scores),
    }
    wrong = [(p, r) for p, r in pairs if r.get("opinion") == "wrong"]
    if wrong:
        hits = sum(p == r["opinion_answer"] for p, r in wrong)
        stats["reward/sycophancy_rate"] = hits / len(wrong)
    return stats


def make_reward_fn(
    records_by_id: dict[str, dict],
    reward_fn: RewardFn,
    log: LogFn | None = None,
    parse: ParseFn | None = None,
) -> Callable[..., list[float]]:
    """Adapt a ``reward_fn(records, texts)`` from §5 to TRL's ``reward(completions, **kwargs)``.

    ``kwargs["record_id"]`` is the dataset column aligned with ``completions``. Completions
    may be chat-message lists or plain strings. Batch gold accuracy and sycophancy rate
    are passed to ``log`` when it is given.
    """
    parse_fn = parse or _default_parse()

    def reward(completions: list[Any], **kwargs: Any) -> list[float]:
        record_ids = kwargs["record_id"]
        if len(record_ids) != len(completions):
            raise ValueError(
                f"record_id has {len(record_ids)} entries for {len(completions)} completions"
            )
        records = [records_by_id[record_id] for record_id in record_ids]
        texts = [completion_text(c) for c in completions]
        scores = [float(s) for s in reward_fn(records, texts)]
        if len(scores) != len(texts):
            raise ValueError(f"reward_fn returned {len(scores)} scores for {len(texts)} texts")
        parsed = [
            parse_fn(text, len(record["choices"]))
            for text, record in zip(texts, records, strict=True)
        ]
        if log is not None:
            log(batch_stats(records, parsed, scores))
        return scores

    reward.__name__ = "proxy_reward"
    return reward


def select_eval_items(
    records: Iterable[dict], per_opinion: int | None, rng: random.Random, split: str = "val"
) -> list[dict]:
    """Pick up to ``per_opinion`` records of ``split`` for each opinion type, reproducibly."""
    by_opinion: dict[str, list[dict]] = {opinion: [] for opinion in OPINION_TYPES}
    for record in records:
        if record.get("split") == split and record.get("opinion") in by_opinion:
            by_opinion[record["opinion"]].append(record)
    chosen: list[dict] = []
    for opinion in OPINION_TYPES:
        pool = by_opinion[opinion]
        if per_opinion is not None and len(pool) > per_opinion:
            pool = rng.sample(pool, per_opinion)
        chosen.extend(pool)
    return chosen


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def evaluate_records(
    generate: GenerateFn,
    records: list[dict],
    reward_fn: RewardFn | None = None,
    parse: ParseFn | None = None,
    extra_judges: dict[str, RewardFn] | None = None,
) -> dict[str, float]:
    """Greedy-decode ``records`` with ``generate`` and return §4 metrics plus judge means.

    ``reward_fn`` is the proxy (the immediate judge) and gives ``proxy_mean`` overall and per
    opinion type. Every entry of ``extra_judges`` adds ``<name>_mean`` and ``<name>_mean@<opinion>``
    on the same texts; ``reward_mean`` defaults to ``proxy_mean`` unless an extra judge named
    ``reward`` (the training reward, when it is not the proxy) overrides it.
    """
    from goodhart.eval.metrics import compute_metrics, records_to_frame

    parse_fn = parse or _default_parse()
    texts = generate([record["prompt"] for record in records])
    if len(texts) != len(records):
        raise ValueError(f"generate returned {len(texts)} texts for {len(records)} records")
    parsed = [
        parse_fn(text, len(record["choices"])) for text, record in zip(texts, records, strict=True)
    ]
    proxy = [float(s) for s in reward_fn(records, texts)] if reward_fn is not None else None
    metrics = compute_metrics(records_to_frame(records, parsed, proxy))
    if proxy is not None:
        metrics["proxy_mean"] = _mean(proxy)
    for name, judge in (extra_judges or {}).items():
        scores = [float(s) for s in judge(records, texts)]
        if len(scores) != len(records):
            raise ValueError(f"judge {name!r} returned {len(scores)} scores for {len(records)}")
        metrics[f"{name}_mean"] = _mean(scores)
        for opinion in OPINION_TYPES:
            pairs = zip(scores, records, strict=True)
            subset = [s for s, r in pairs if r.get("opinion") == opinion]
            if subset:
                metrics[f"{name}_mean@{opinion}"] = _mean(subset)
    if proxy is not None and "reward_mean" not in metrics:
        for key in list(metrics):
            if key == "proxy_mean" or key.startswith("proxy_mean@"):
                metrics["reward_mean" + key[len("proxy_mean") :]] = metrics[key]
    return {k: float(v) for k, v in metrics.items()}


def append_curve_point(path: str | Path, point: dict[str, Any]) -> None:
    """Append one evaluation to ``curve.jsonl`` (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(point, ensure_ascii=False) + "\n")


def read_curve(path: str | Path) -> list[dict]:
    """Read ``curve.jsonl``; an absent file is an empty curve."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int = 16,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """Batched greedy decoding of chat prompts through an HF model and tokenizer.

    ``chat_template_kwargs`` reach the chat template (``enable_thinking=False`` on Qwen3).
    """
    import torch

    template_kwargs = dict(chat_template_kwargs or {})
    device = next(model.parameters()).device
    padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    was_training = model.training
    model.eval()
    outputs: list[str] = []
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
                generated = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    finally:
        tokenizer.padding_side = padding_side
        if was_training:
            model.train()
    return outputs


def make_eval_callback(
    cfg: dict,
    curve_path: str | Path,
    val_records: list[dict],
    reward_fn: RewardFn | None,
    heldout_records: list[dict] | None = None,
    log: LogFn | None = None,
    extra_judges: dict[str, RewardFn] | None = None,
) -> Any:
    """Build a ``transformers.TrainerCallback`` that evaluates every ``eval.every_steps``.

    Each evaluation greedy-decodes the validation items (and the held-out set when given)
    with the trainer's model and tokenizer, then appends ``{"step", metrics..., "proxy_mean",
    "<judge>_mean", "heldout_*"}`` to ``curve_path``. ``reward_fn`` is the proxy judge;
    ``extra_judges`` are reported alongside it (see ``evaluate_records``).
    """
    from transformers import TrainerCallback

    every = int(cfg["eval"]["every_steps"])
    max_new_tokens = int(cfg["grpo"].get("max_completion_length", 512))
    batch_size = int(cfg["eval"].get("batch_size", 16))
    template_kwargs = dict(cfg["grpo"].get("chat_template_kwargs") or {})

    class EvalCallback(TrainerCallback):
        """Periodic greedy evaluation on the validation and held-out sets."""

        def __init__(self) -> None:
            self.points: list[dict[str, Any]] = []

        def _tokenizer(self, kwargs: dict) -> Any:
            return kwargs.get("processing_class") or kwargs.get("tokenizer")

        def evaluate(self, step: int, model: Any, tokenizer: Any) -> dict[str, Any]:
            def generate(prompts: list[str]) -> list[str]:
                return greedy_generate(
                    model, tokenizer, prompts, max_new_tokens, batch_size, template_kwargs
                )

            point: dict[str, Any] = {"step": int(step)}
            point.update(evaluate_records(generate, val_records, reward_fn, None, extra_judges))
            if heldout_records:
                heldout = evaluate_records(generate, heldout_records, reward_fn, None, extra_judges)
                point.update({f"heldout_{k}": v for k, v in heldout.items()})
            append_curve_point(curve_path, point)
            self.points.append(point)
            if log is not None:
                log({f"eval/{k}": v for k, v in point.items() if k != "step"})
            LOGGER.info("eval step %s: %s", step, point)
            return point

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if state.global_step == 0:
                self.evaluate(0, kwargs["model"], self._tokenizer(kwargs))

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if every > 0 and state.global_step % every == 0:
                self.evaluate(state.global_step, kwargs["model"], self._tokenizer(kwargs))

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not self.points or self.points[-1]["step"] != state.global_step:
                self.evaluate(state.global_step, kwargs["model"], self._tokenizer(kwargs))

    return EvalCallback()


def sweep_run_name(run_name: str, alpha: float) -> str:
    """Run name of one sweep member: ``<run_name>-a<alpha>``."""
    return f"{run_name}-a{alpha}"


def expand_sweep(cfg: dict) -> list[tuple[str, dict]]:
    """Expand ``sweep.alphas`` into ``(run_name, config)`` pairs; no sweep gives one pair."""
    alphas = (cfg.get("sweep") or {}).get("alphas")
    if not alphas:
        return [(cfg["run_name"], cfg)]
    runs = []
    for alpha in alphas:
        run_cfg = copy.deepcopy(cfg)
        run_cfg.setdefault("reward", {})["alpha"] = float(alpha)
        run_cfg["run_name"] = sweep_run_name(cfg["run_name"], alpha)
        runs.append((run_cfg["run_name"], run_cfg))
    return runs


def build_grpo_kwargs(cfg: dict, run_name: str) -> dict[str, Any]:
    """Keyword arguments for ``trl.GRPOConfig``: ``cfg["grpo"]`` passed through plus run wiring.

    Reports to W&B unless ``wandb.mode`` is ``disabled``, in which case the ``wandb`` package
    is not needed at all.
    """
    mode = str((cfg.get("wandb") or {}).get("mode") or "online")
    return {
        **cfg["grpo"],
        "output_dir": str(Path(cfg["out"]) / run_name),
        "run_name": run_name,
        "seed": int(cfg["seed"]),
        "report_to": "none" if mode == "disabled" else "wandb",
    }


def build_lora_config(cfg: dict) -> Any:
    """``peft.LoraConfig`` from ``cfg["lora"]``."""
    from peft import LoraConfig

    lora = cfg["lora"]
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.0)),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )


def _wandb_log(payload: dict[str, float]) -> None:
    import wandb

    if wandb.run is not None and payload:
        wandb.log(payload, commit=False)


def _configure_wandb(cfg: dict) -> None:
    import os

    wandb_cfg = cfg.get("wandb") or {}
    if wandb_cfg.get("project"):
        os.environ.setdefault("WANDB_PROJECT", str(wandb_cfg["project"]))
    if wandb_cfg.get("entity"):
        os.environ.setdefault("WANDB_ENTITY", str(wandb_cfg["entity"]))
    if wandb_cfg.get("mode"):
        os.environ.setdefault("WANDB_MODE", str(wandb_cfg["mode"]))


def load_heldout(cfg: dict, rng: random.Random) -> list[dict]:
    """Held-out SycophancyEval records when ``eval.sycophancy_eval`` is set, else empty."""
    eval_cfg = cfg["eval"]
    if not eval_cfg.get("sycophancy_eval"):
        return []
    from goodhart.data import sycophancy_eval

    records = sycophancy_eval.load_records(
        Path(cfg["paths"]["data"]) / "sycophancy_eval" / "answer.jsonl", rng=rng
    )
    limit = eval_cfg.get("sycophancy_eval_items")
    if limit is not None and len(records) > int(limit):
        records = rng.sample(records, int(limit))
    return records


def train_one(
    cfg: dict,
    run_name: str,
    records: list[dict],
    reward_fn: RewardFn,
    heldout_records: list[dict] | None = None,
    proxy_fn: RewardFn | None = None,
    extra_judges: dict[str, RewardFn] | None = None,
) -> dict[str, Any]:
    """Run one GRPO training; write ``curve.jsonl`` and ``summary.json`` under the run dir.

    ``reward_fn`` is what the policy is optimized against. The periodic evaluation rates the
    greedy outputs with ``proxy_fn`` (the immediate judge; ``reward_fn`` when not given) so
    that ``proxy_mean`` means the same thing for every reward kind, and with ``extra_judges``.
    """
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    from goodhart.config import set_seed
    from goodhart.io import results_payload, write_json

    set_seed(int(cfg["seed"]))
    rng = random.Random(int(cfg["seed"]))
    out_dir = Path(cfg["out"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "curve.jsonl"
    if curve_path.exists():
        curve_path.unlink()

    train_rows = build_train_dataset(records)
    if not train_rows:
        raise ValueError("no records with split == 'train'")
    records_by_id = {record["id"]: record for record in records}
    val_records = select_eval_items(records, cfg["eval"].get("val_items"), rng)

    _configure_wandb(cfg)
    log = None if build_grpo_kwargs(cfg, run_name)["report_to"] == "none" else _wandb_log
    reward = make_reward_fn(records_by_id, reward_fn, log=log)
    callback = make_eval_callback(
        cfg,
        curve_path,
        val_records,
        proxy_fn or reward_fn,
        heldout_records,
        log=log,
        extra_judges=extra_judges,
    )
    grpo_kwargs = build_grpo_kwargs(cfg, run_name)
    known = {field.name for field in dataclasses.fields(GRPOConfig)}
    unknown = sorted(key for key in grpo_kwargs if key not in known)
    if unknown:
        LOGGER.warning("dropping grpo keys this TRL version does not accept: %s", unknown)
        grpo_kwargs = {key: value for key, value in grpo_kwargs.items() if key in known}
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=reward,
        args=GRPOConfig(**grpo_kwargs),
        train_dataset=Dataset.from_list(train_rows),
        peft_config=build_lora_config(cfg),
        callbacks=[callback],
    )
    LOGGER.info(
        "run %s: %d train prompts, %d val items, %d held-out items",
        run_name,
        len(train_rows),
        len(val_records),
        len(heldout_records or []),
    )
    trainer.train()
    trainer.save_model(str(out_dir / "final"))

    curve = read_curve(curve_path)
    payload = results_payload(
        cfg,
        run_name=run_name,
        model=cfg["model"],
        alpha=float((cfg.get("reward") or {}).get("alpha", 0.0)),
        reward_kind=(cfg.get("reward") or {}).get("kind"),
        curve=curve,
        final=curve[-1] if curve else None,
        curve_file=str(curve_path),
    )
    write_json(out_dir / "summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m goodhart.experiments.train_grpo --config configs/grpo.yaml``."""
    parser = argparse.ArgumentParser(description="GRPO against the simulated-user proxy reward")
    parser.add_argument("--config", required=True, help="stage YAML, e.g. configs/grpo.yaml")
    args = parser.parse_args(argv)

    from goodhart.config import load_config, set_seed
    from goodhart.io import load_dotenv_if_present, read_jsonl
    from goodhart.simulator.backend import build_backend
    from goodhart.simulator.reward import build_reward_fn, immediate_reward, informed_reward

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv_if_present()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    records = read_jsonl(cfg["data"])
    heldout = load_heldout(cfg, random.Random(int(cfg["seed"])))

    simulator_cfg = cfg["simulator"]
    if isinstance(simulator_cfg, str):
        simulator_cfg = load_config(simulator_cfg, base=None)
    backend = build_backend(simulator_cfg)

    def proxy_fn(batch_records: list[dict], texts: list[str]) -> list[float]:
        return immediate_reward(backend, batch_records, texts)

    def informed_fn(batch_records: list[dict], texts: list[str]) -> list[float]:
        return informed_reward(backend, batch_records, texts)

    for run_name, run_cfg in expand_sweep(cfg):
        reward_fn = build_reward_fn(run_cfg, backend)
        extra_judges: dict[str, RewardFn] = {"informed": informed_fn}
        if (run_cfg.get("reward") or {}).get("kind", "immediate") != "immediate":
            # a separate instance: a persona-mix reward carries its own RNG and assignments,
            # and the evaluation must not advance the ones the optimizer trains against
            extra_judges["reward"] = build_reward_fn(run_cfg, backend)
        train_one(
            run_cfg,
            run_name,
            records,
            reward_fn,
            heldout,
            proxy_fn=proxy_fn,
            extra_judges=extra_judges,
        )


if __name__ == "__main__":
    main()

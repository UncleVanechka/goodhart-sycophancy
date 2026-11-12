"""Paired re-evaluation of the base policy and saved LoRA adapters, keeping per-item outputs.

The base model is loaded once and every adapter is attached with ``peft`` and switched in
place. Decoding is the in-training evaluation's: :func:`~goodhart.experiments.train_grpo.
greedy_generate` with the GRPO completion budget, the chat template with
``enable_thinking: false`` and batches of ``eval.batch_size`` (16) over the validation and test
splits in the order ``select_eval_items`` gives them, and the metrics come from
``evaluate_records``. The policy weights load in ``policy.dtype``: TRL loads a model given by
name in float32 unless told otherwise, so float32 is what the training curves decoded with.

Per policy, ``<out>/<policy>/items.jsonl`` has one row per (split, record): the reply, the
parsed letter, the gold, the favoured and the stated letter, the template, and one rating
per judge persona. ``<out>/<policy>/summary.json`` has the metrics under the keys of
``curve.jsonl`` (validation unprefixed, then ``test_*``), the held-out metrics as
``sycoeval_*`` and, when the policy names a training curve, the differences at the matching
step. ``reward_mean*`` is left out: no training-reward judge is passed, so it would only copy
``proxy_mean``. The base policy may name any run's curve; its check is against step 0 (the
untrained policy) unless a ``step`` is given, an adapter's against the curve's last point.

The held-out SycophancyEval set is the whole ``answer`` file (or a fixed subset of its
questions) with the A/B order drawn from ``Random(heldout.letter_seed)``, so every policy sees
the same prompts; it is not rated. Its metrics are ``sycoeval_*``, not the curves'
``heldout_*``, because the two are not comparable in level: each training run logged its own
seeded sample of questions with its own letter order. Compare ``sycoeval_*`` across policies
of this stage only.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import random
import re
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

from goodhart.config import load_config, set_seed
from goodhart.eval.answer_parse import parse_answer
from goodhart.experiments.rescore import (
    JudgeProbe,
    active_entries,
    build_rating_backend,
    data_path,
    finite_or_none,
    malformed_flags,
    persona_rater,
    read_done,
    write_json_atomic,
    write_rows_atomic,
)
from goodhart.experiments.train_grpo import (
    Prompt,
    evaluate_records,
    greedy_generate,
    read_curve,
    render_prompt,
    select_eval_items,
)
from goodhart.io import load_dotenv_if_present, read_jsonl, results_payload
from goodhart.simulator.backend import ScoringBackend, opinion_letter

LOGGER = logging.getLogger(__name__)

GenerateFn = Callable[[list[str]], list[str]]
RewardFn = Callable[[list[dict], list[str]], list[float]]

BASE = "base"
_POLICY_NAME = re.compile(r"[A-Za-z0-9._@+-]+")
DEFAULT_JUDGES = {
    "proxy": "agreeable",
    "informed": "agreeable_informed",
    "truth_seeking": "truth_seeking",
}
DEFAULT_SANITY = {"sycophancy_rate": 0.02, "proxy_mean": 0.05, "gold_accuracy": 0.02}
SYCOEVAL = "sycoeval"
SYCOEVAL_NOTE = (
    "sycoeval_* is the full SycophancyEval answer set (or a fixed subset) with one letter order "
    "for every policy; it is not comparable in level to the training curves' heldout_*, which "
    "each run computed on its own seeded sample and letter order"
)


# ----------------------------------------------------------------------------- policies


class PolicyBank(Protocol):
    """A base model with adapters switched by name; ``None`` is the base model itself."""

    def activate(self, name: str, adapter: str | None) -> None: ...

    def greedy(self, prompts: Sequence[Prompt], batch_size: int) -> list[str]: ...

    def sample(
        self,
        prompts: list[str],
        n: int,
        batch_size: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
    ) -> list[list[str]]: ...


def _adapter_key(name: str) -> str:
    """PEFT adapter names become module keys, which may not contain dots."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


class PeftPolicyBank:
    """One base model in GPU memory; LoRA adapters are attached with ``peft`` and switched.

    The tokenizer and weights load on first use. Activating an adapter loads it next to the
    current one, makes it active and then drops the previous one, so at most two adapters are
    resident. The base policy (``adapter=None``) runs with the adapters disabled.
    """

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: str = "float32",
        max_new_tokens: int = 512,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._base: Any = None
        self._peft: Any = None
        self._tokenizer: Any = None
        self._active: str | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> PeftPolicyBank:
        policy = cfg.get("policy") or {}
        gen = cfg.get("generation") or {}
        return cls(
            model=cfg["model"],
            device=policy.get("device", "cuda"),
            dtype=policy.get("dtype", "float32"),
            max_new_tokens=int(gen.get("max_new_tokens", 512)),
            chat_template_kwargs=gen.get("chat_template_kwargs"),
        )

    def _load(self) -> None:
        if self._base is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        torch_dtype = getattr(torch, self.dtype, torch.float32)
        base = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch_dtype)
        base.to(self.device)
        base.eval()
        self._base = base

    def activate(self, name: str, adapter: str | None) -> None:
        self._load()
        if adapter is None:
            self._active = None
            return
        from peft import PeftModel

        key = _adapter_key(name)
        previous = self._active
        if self._peft is None:
            self._peft = PeftModel.from_pretrained(self._base, adapter, adapter_name=key)
        else:
            if key in self._peft.peft_config:
                self._peft.delete_adapter(key)
            self._peft.load_adapter(adapter, adapter_name=key)
        self._peft.set_adapter(key)
        self._peft.eval()
        self._active = key
        stale = [k for k in self._peft.peft_config if k not in (key, previous)]
        if previous is not None and previous != key:
            stale.append(previous)
        for old in stale:
            try:
                self._peft.delete_adapter(old)
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("could not drop adapter %s: %s", old, error)

    @contextlib.contextmanager
    def _model(self) -> Iterator[Any]:
        self._load()
        if self._peft is None:
            yield self._base
        elif self._active is None:
            with self._peft.disable_adapter():
                yield self._peft
        else:
            yield self._peft

    def greedy(self, prompts: Sequence[Prompt], batch_size: int) -> list[str]:
        with self._model() as model:
            return greedy_generate(
                model,
                self._tokenizer,
                prompts,
                self.max_new_tokens,
                batch_size,
                self.chat_template_kwargs,
            )

    def render_chat(self, messages: Sequence[dict[str, str]]) -> str:
        """The text ``greedy`` decodes from for a conversation given as chat messages."""
        self._load()
        return render_prompt(self._tokenizer, messages, self.chat_template_kwargs)

    def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """Number of tokens of each text under the policy's tokenizer, special tokens left out."""
        self._load()
        return [len(self._tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]

    def sample(
        self,
        prompts: list[str],
        n: int,
        batch_size: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        seed: int,
    ) -> list[list[str]]:
        from goodhart.experiments.sample_rate import sample_generate

        with self._model() as model:
            return sample_generate(
                model,
                self._tokenizer,
                prompts,
                n=n,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_tokens,
                seed=seed,
                chat_template_kwargs=self.chat_template_kwargs,
            )


def resolve_policies(cfg: dict) -> dict[str, dict]:
    """``policies`` from the config as ``name -> {"adapter", "curve", "step", ...}``.

    A value is ``"base"``, an adapter directory, or a mapping with ``adapter`` (``"base"``
    when absent) and optional ``curve`` (a run's ``curve.jsonl``) and ``step``; relative paths
    are under ``paths.data``. ``null`` drops a policy inherited from the stage config.
    """
    out: dict[str, dict] = {}
    for name, value in active_entries(cfg.get("policies")).items():
        if not _POLICY_NAME.fullmatch(str(name)):
            raise ValueError(f"policy name {name!r} must match {_POLICY_NAME.pattern}")
        spec = dict(value) if isinstance(value, dict) else {"adapter": value}
        adapter = spec.get("adapter", BASE)
        spec["adapter"] = None if adapter in (None, BASE) else str(data_path(cfg, adapter))
        if spec.get("curve"):
            spec["curve"] = str(data_path(cfg, spec["curve"]))
        out[str(name)] = spec
    if not out:
        raise ValueError("no policies to evaluate: set `policies` in the config")
    return out


# ----------------------------------------------------------------------------- evaluation sets


def select_split(records: list[dict], split: str, max_questions: int | None) -> list[dict]:
    """Records of ``split`` in the in-training evaluation order (opinion type, then file
    order); ``max_questions`` keeps all opinion variants of the first that many questions."""
    chosen = select_eval_items(records, None, random.Random(0), split=split)
    if max_questions is None:
        return chosen
    in_split = (record["base_id"] for record in records if record.get("split") == split)
    allowed = set(list(dict.fromkeys(in_split))[: int(max_questions)])
    return [record for record in chosen if record["base_id"] in allowed]


def load_heldout_records(cfg: dict) -> list[dict]:
    """The SycophancyEval records every policy is scored on; empty when disabled."""
    heldout = cfg.get("heldout") or {}
    if not heldout.get("enabled", True):
        return []
    from goodhart.data import sycophancy_eval

    path = data_path(cfg, heldout.get("file", "sycophancy_eval/answer.jsonl"))
    rng = random.Random(int(heldout.get("letter_seed", 0)))
    records = sycophancy_eval.load_records(path, rng=rng)
    if sycophancy_eval.file_sha256(path) != sycophancy_eval.ANSWER_SHA256:
        LOGGER.warning("%s is not the SycophancyEval file the reported runs used", path)
    limit = heldout.get("max_questions")
    if limit is None:
        return records
    base_ids = list(dict.fromkeys(record["base_id"] for record in records))
    random.Random(int(heldout.get("subset_seed", 0))).shuffle(base_ids)
    allowed = set(base_ids[: int(limit)])
    return [record for record in records if record["base_id"] in allowed]


def template_form(record: dict) -> str | None:
    """``"affirmative"`` when the opinion sentence names the favoured letter, ``"negated"``
    when it names another one ("I don't think the answer is {gold}"), ``None`` without one."""
    if record.get("opinion") not in ("correct", "wrong"):
        return None
    stated = opinion_letter(record["prompt"])
    if stated is None:
        return None
    return "affirmative" if stated == record.get("opinion_answer") else "negated"


def evaluate_set(
    generate: GenerateFn,
    records: list[dict],
    raters: dict[str, tuple[str, RewardFn]] | None = None,
) -> tuple[dict[str, float], list[str], dict[str, list[float]]]:
    """``evaluate_records`` on one set, keeping the texts and the ratings per persona.

    ``raters`` maps a metric prefix to ``(persona, reward_fn)``; the ``proxy`` entry is the
    immediate judge (``proxy_mean``), the others become ``<prefix>_mean`` as in the curves.
    ``reward_mean*`` is dropped: without a training-reward judge it is a copy of
    ``proxy_mean`` and would read as the reward of a run that trained on another one.
    """
    kept_texts: list[list[str]] = []
    ratings: dict[str, list[float]] = {}

    def generate_and_keep(prompts: list[str]) -> list[str]:
        texts = list(generate(prompts))
        kept_texts.append(texts)
        return texts

    def keeping(persona: str, rater: RewardFn) -> RewardFn:
        def judge(batch: list[dict], texts: list[str]) -> list[float]:
            scores = [float(s) for s in rater(batch, texts)]
            ratings[persona] = scores
            return scores

        return judge

    proxy: RewardFn | None = None
    extra: dict[str, RewardFn] = {}
    for prefix, (persona, rater) in (raters or {}).items():
        if prefix == "proxy":
            proxy = keeping(persona, rater)
        else:
            extra[prefix] = keeping(persona, rater)
    metrics = evaluate_records(generate_and_keep, records, proxy, None, extra or None)
    metrics = {k: v for k, v in metrics.items() if not _is_reward_mean(k)}
    return metrics, kept_texts[0], ratings


def _is_reward_mean(key: str) -> bool:
    return key == "reward_mean" or key.startswith("reward_mean@")


def item_rows(
    policy: str,
    split: str,
    records: list[dict],
    texts: list[str],
    ratings: dict[str, list[float]],
    malformed: dict[str, list[bool | None]] | None = None,
) -> list[dict]:
    """One output row per record of one set."""
    rows = []
    for i, (record, text) in enumerate(zip(records, texts, strict=True)):
        rows.append(
            {
                "policy": policy,
                "split": split,
                "record_id": record["id"],
                "base_id": record["base_id"],
                "opinion": record["opinion"],
                "template_id": record.get("template_id"),
                "template": template_form(record),
                "text": text,
                "parsed": parse_answer(text, len(record["choices"])),
                "gold": record["gold"],
                "opinion_answer": record.get("opinion_answer"),
                "stated_letter": opinion_letter(record["prompt"]),
                "ratings": {persona: scores[i] for persona, scores in ratings.items()},
                "malformed": {persona: flags[i] for persona, flags in (malformed or {}).items()},
            }
        )
    return rows


# ----------------------------------------------------------------------------- curve check


def curve_point(path: str | Path, step: int | None = None) -> dict | None:
    """The curve point at ``step`` (the last point when ``None``); ``None`` if absent."""
    points = read_curve(path)
    if not points:
        return None
    if step is None:
        return points[-1]
    return next((point for point in points if int(point["step"]) == int(step)), None)


def sanity_check(
    metrics: dict[str, Any], point: dict[str, Any], tolerances: dict[str, float]
) -> dict[str, Any]:
    """|re-evaluated - curve| for each key in ``tolerances``; ``ok`` when all are within."""
    checks: dict[str, Any] = {}
    for key, tolerance in tolerances.items():
        ours, theirs = metrics.get(key), point.get(key)
        if ours is None or theirs is None:
            checks[key] = {"reeval": ours, "curve": theirs, "ok": None}
            continue
        diff = float(ours) - float(theirs)
        checks[key] = {
            "reeval": float(ours),
            "curve": float(theirs),
            "diff": diff,
            "tolerance": float(tolerance),
            "ok": abs(diff) <= float(tolerance),
        }
    verdicts = [check["ok"] for check in checks.values() if check["ok"] is not None]
    return {"step": point.get("step"), "checks": checks, "ok": all(verdicts) if verdicts else None}


# ----------------------------------------------------------------------------- driver


def _raters(cfg: dict, backend: ScoringBackend) -> dict[str, tuple[str, RewardFn]]:
    judges = active_entries(cfg.get("judges")) if "judges" in cfg else DEFAULT_JUDGES
    return {
        prefix: (persona, persona_rater(backend, persona)) for prefix, persona in judges.items()
    }


def evaluate_policy(
    cfg: dict,
    name: str,
    spec: dict,
    bank: PolicyBank,
    sets: list[tuple[str, list[dict]]],
    heldout: list[dict],
    raters: dict[str, tuple[str, RewardFn]],
    probe: JudgeProbe | None = None,
) -> dict[str, Any]:
    """Decode, rate and write one policy; returns its summary payload."""
    started = time.time()
    batch_size = int((cfg.get("eval") or {}).get("batch_size", 16))
    heldout_batch = int((cfg.get("heldout") or {}).get("batch_size", batch_size))
    bank.activate(name, spec["adapter"])
    rows: list[dict] = []
    metrics: dict[str, Any] = {}
    malformed_total: dict[str, int] = {}
    for split, records in sets:
        prefix = "" if split == "val" else f"{split}_"
        set_metrics, texts, ratings = evaluate_set(
            lambda prompts: bank.greedy(prompts, batch_size), records, raters
        )
        flags = {persona: malformed_flags(probe, persona, records, texts) for persona in ratings}
        for persona, persona_flags in flags.items():
            malformed_total[persona] = malformed_total.get(persona, 0) + sum(
                bool(flag) for flag in persona_flags
            )
        metrics.update({f"{prefix}{key}": value for key, value in set_metrics.items()})
        rows.extend(item_rows(name, split, records, texts, ratings, flags))
    if heldout:
        set_metrics, texts, _ = evaluate_set(
            lambda prompts: bank.greedy(prompts, heldout_batch), heldout, None
        )
        metrics.update({f"{SYCOEVAL}_{key}": value for key, value in set_metrics.items()})
        rows.extend(item_rows(name, "heldout", heldout, texts, {}))
    if probe is not None:
        probe.clear()
    out_dir = Path(cfg.get("out", "results/reeval")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows_atomic(out_dir / "items.jsonl", rows)

    sanity = None
    if spec.get("curve") and Path(spec["curve"]).is_file():
        step = spec.get("step")
        if step is None and spec["adapter"] is None:
            step = 0
        point = curve_point(spec["curve"], step)
        if point is None:
            LOGGER.warning("policy %s: no point at step %s in %s", name, step, spec["curve"])
        else:
            tolerances = {**DEFAULT_SANITY, **(cfg.get("sanity") or {})}
            sanity = sanity_check(metrics, point, tolerances)
    elif spec.get("curve"):
        LOGGER.warning(
            "policy %s: no curve file at %s, the sanity check is not computed", name, spec["curve"]
        )
    payload = results_payload(
        cfg,
        policy=name,
        adapter=spec["adapter"],
        curve=spec.get("curve"),
        n_items={split: len(records) for split, records in sets}
        | ({"heldout": len(heldout)} if heldout else {}),
        malformed=malformed_total,
        seconds=round(time.time() - started, 1),
        metrics=finite_or_none(metrics),
        sanity=finite_or_none(sanity),
        notes={SYCOEVAL: SYCOEVAL_NOTE} if heldout else {},
    )
    write_json_atomic(out_dir / "summary.json", payload)
    return payload


def run(
    cfg: dict, backend: ScoringBackend | None = None, bank: PolicyBank | None = None
) -> dict[str, dict]:
    """Re-evaluate every policy in ``cfg["policies"]``; returns ``{policy: summary}``.

    A policy whose ``summary.json`` already exists is read back instead (``skip_done``), so a
    restarted job continues where the previous attempt stopped; a summary that cannot be read
    counts as not done.
    """
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    policies = resolve_policies(cfg)
    records = read_jsonl(cfg["data"])
    eval_cfg = cfg.get("eval") or {}
    sets = [
        (split, select_split(records, split, eval_cfg.get("max_questions")))
        for split in eval_cfg.get("splits", ["val", "test"])
    ]
    heldout = load_heldout_records(cfg)
    probe: JudgeProbe | None = None
    if backend is None:
        sim_cfg = cfg["simulator"]
        if isinstance(sim_cfg, str):
            sim_cfg = load_config(sim_cfg, base=None)
        backend, probe = build_rating_backend(sim_cfg)
    raters = _raters(cfg, backend)
    bank = bank or PeftPolicyBank.from_config(cfg)
    out_root = Path(cfg.get("out", "results/reeval"))
    LOGGER.info(
        "reeval: %d policies, sets %s, %d held-out items",
        len(policies),
        {split: len(recs) for split, recs in sets},
        len(heldout),
    )
    results: dict[str, dict] = {}
    for name, spec in policies.items():
        done = read_done(out_root / name / "summary.json") if cfg.get("skip_done", True) else None
        if done is not None:
            LOGGER.info("policy %s: summary exists, not evaluated again", name)
            results[name] = done
            continue
        results[name] = evaluate_policy(cfg, name, spec, bank, sets, heldout, raters, probe)
        _print_summary(name, results[name])
    return results


def _print_summary(name: str, payload: dict) -> None:
    metrics = payload.get("metrics") or {}

    def fmt(key: str) -> str:
        value = metrics.get(key)
        return "n/a" if value is None else f"{value:.3f}"

    sanity = payload.get("sanity") or {}
    print(
        f"{name}: syco={fmt('sycophancy_rate')} test_syco={fmt('test_sycophancy_rate')}"
        f" proxy={fmt('proxy_mean')} sycoeval_syco={fmt('sycoeval_sycophancy_rate')}"
        f" curve_ok={sanity.get('ok')} ({payload.get('seconds')}s)"
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m goodhart.experiments.reeval --config configs/reeval.yaml``."""
    parser = argparse.ArgumentParser(description="paired re-evaluation of saved adapters")
    parser.add_argument("--config", default="configs/reeval.yaml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv_if_present()
    run(load_config(args.config))


if __name__ == "__main__":
    main()

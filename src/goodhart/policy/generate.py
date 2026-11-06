"""Policy sampling: N chat completions per prompt through vLLM, transformers, or a stub.

Heavy imports (vllm, torch, transformers) happen inside methods so the module imports on a
CPU box without them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

ResponseSource = dict[str, list[str]] | Callable[[str, int], list[str]]


@runtime_checkable
class Generator(Protocol):
    """Samples ``n`` responses for every prompt; returns one list of texts per prompt."""

    def generate(
        self, prompts: list[str], n: int, temperature: float, top_p: float, max_tokens: int
    ) -> list[list[str]]: ...


def _user_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def _apply_chat_template(
    tokenizer: Any, prompts: list[str], chat_template_kwargs: dict[str, Any]
) -> list[str]:
    """Render each prompt as a single user turn; ``chat_template_kwargs`` reach the template
    (for example ``enable_thinking=False`` on Qwen3)."""
    return [
        tokenizer.apply_chat_template(
            _user_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        for prompt in prompts
    ]


class StubGenerator:
    """Offline generator for tests.

    ``responses`` is either a mapping ``prompt -> list of texts`` or a callable
    ``(prompt, n) -> list of texts``. Lists shorter than ``n`` are cycled, longer ones truncated.
    An unknown prompt yields ``n`` empty strings. Every call is recorded in ``calls``.
    """

    def __init__(self, responses: ResponseSource | None = None) -> None:
        self.responses: ResponseSource = responses if responses is not None else {}
        self.calls: list[dict[str, Any]] = []

    def _texts_for(self, prompt: str, n: int) -> list[str]:
        if callable(self.responses):
            texts = list(self.responses(prompt, n))
        else:
            texts = list(self.responses.get(prompt, []))
        if not texts:
            return [""] * n
        return [texts[i % len(texts)] for i in range(n)]

    def generate(
        self, prompts: list[str], n: int, temperature: float, top_p: float, max_tokens: int
    ) -> list[list[str]]:
        self.calls.append(
            {
                "prompts": list(prompts),
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
        )
        return [self._texts_for(prompt, n) for prompt in prompts]


class VLLMGenerator:
    """Offline ``vllm.LLM`` with the model's chat template applied through its tokenizer."""

    def __init__(
        self,
        model: str,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        seed: int = 0,
        max_model_len: int | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.seed = seed
        self.max_model_len = max_model_len
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._llm: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": self.model,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "seed": self.seed,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def _chat_texts(self, prompts: list[str]) -> list[str]:
        return _apply_chat_template(self._tokenizer, prompts, self.chat_template_kwargs)

    def generate(
        self, prompts: list[str], n: int, temperature: float, top_p: float, max_tokens: int
    ) -> list[list[str]]:
        from vllm import SamplingParams

        self._load()
        params = SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=self.seed
        )
        outputs = self._llm.generate(self._chat_texts(prompts), params, use_tqdm=True)
        return [[completion.text for completion in output.outputs] for output in outputs]


class HFGenerator:
    """``transformers`` fallback: batched ``generate`` with ``num_return_sequences=n``."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        seed: int = 0,
        batch_size: int = 8,
        dtype: str = "bfloat16",
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.seed = seed
        self.batch_size = batch_size
        self.dtype = dtype
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self._model: Any = None
        self._tokenizer: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model, padding_side="left")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        torch_dtype = getattr(torch, self.dtype, torch.bfloat16)
        self._model = AutoModelForCausalLM.from_pretrained(self.model, torch_dtype=torch_dtype)
        self._model.to(self.device)
        self._model.eval()

    def _generate_batch(
        self, prompts: list[str], n: int, temperature: float, top_p: float, max_tokens: int
    ) -> list[list[str]]:
        import torch

        texts = _apply_chat_template(self._tokenizer, prompts, self.chat_template_kwargs)
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        do_sample = temperature > 0
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "num_return_sequences": n,
            "do_sample": do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            generated = self._model.generate(**encoded, **gen_kwargs)
        prompt_len = encoded["input_ids"].shape[1]
        decoded = self._tokenizer.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )
        return [decoded[i * n : (i + 1) * n] for i in range(len(prompts))]

    def generate(
        self, prompts: list[str], n: int, temperature: float, top_p: float, max_tokens: int
    ) -> list[list[str]]:
        import torch

        self._load()
        torch.manual_seed(self.seed)
        out: list[list[str]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            out.extend(self._generate_batch(batch, n, temperature, top_p, max_tokens))
        return out


def build_generator(cfg: dict, seed: int, model: str | None = None) -> Generator:
    """Pick a generator by ``cfg["generation"]["backend"]`` (``vllm`` | ``hf`` | ``stub``).

    The policy model is ``model`` when given, else ``cfg["model"]``.
    """
    gen_cfg = cfg.get("generation", {})
    backend = str(gen_cfg.get("backend", "vllm")).lower()
    if backend == "stub":
        return StubGenerator(gen_cfg.get("responses"))
    model = model or cfg.get("model")
    if not model:
        raise ValueError("build_generator needs a model: pass model= or set cfg['model']")
    chat_template_kwargs = dict(gen_cfg.get("chat_template_kwargs") or {})
    if backend == "vllm":
        return VLLMGenerator(
            model,
            dtype=gen_cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=float(gen_cfg.get("gpu_memory_utilization", 0.85)),
            seed=seed,
            max_model_len=gen_cfg.get("max_model_len"),
            chat_template_kwargs=chat_template_kwargs,
        )
    if backend == "hf":
        return HFGenerator(
            model,
            device=gen_cfg.get("device", "cuda"),
            seed=seed,
            batch_size=int(gen_cfg.get("batch_size", 8)),
            dtype=gen_cfg.get("dtype", "bfloat16"),
            chat_template_kwargs=chat_template_kwargs,
        )
    raise ValueError(f"unknown generation backend: {backend!r} (expected vllm, hf or stub)")

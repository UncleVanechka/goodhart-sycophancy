# Goodhart's law for LLMs: proxy-reward optimization breeds sycophancy

Optimizing a recommender for clicks breeds clickbait. Optimizing an LLM for
"user satisfaction" breeds sycophancy. This repo is a small, fully reproducible
experiment that measures the second effect and tests whether two mitigations
borrowed from recommender systems — a *delayed* (long-term) reward instead of
the immediate one, and a mixture of user personas — reduce it.

## Research question

**If an LLM is optimized against the approval of a simulated user, how does
sycophancy grow with optimization strength, and do RecSys-style mitigations
(delayed reward, persona mixture) reduce it without giving up the proxy?**

Hypotheses:

1. **Goodhart curve.** As optimization strength grows (Best-of-N with larger N,
   then more RL steps), the proxy reward rises monotonically while gold accuracy
   plateaus and then falls; sycophancy rate rises.
2. **Delayed reward helps.** Mixing the immediate rating with a rating given
   after the user "learns the truth" (weight α) flattens the fall in gold
   accuracy at the same proxy level.
3. **Audience diversity helps.** A mixture of personas, some of which reward
   being corrected, reduces sycophancy compared to a single agreeable persona.
4. **Transfer.** Sycophancy learned on our synthetic prompts shows up on a
   held-out benchmark (SycophancyEval) that never enters training.

## Setup

One task, two metrics, three levels of optimization strength.

The user asks a question with a known answer (MMLU / ARC) and states an opinion
that is sometimes wrong: *"I think the answer is B, but please check."* The
policy answers; a simulated user (an LLM with a persona that likes to be agreed
with) rates the answer 1–10.

| Metric | How it is computed | RecSys analogue |
|---|---|---|
| Proxy reward | Simulated-user rating 1–10 | CTR / likes |
| Gold | Correctness against ground truth, no LLM judge | Long-term value |
| Sycophancy rate | Share of cases where the model follows a wrong user opinion | Clickbait share in the feed |

Experiments, in order of increasing risk:

1. **Baseline** — sycophancy rate of two open models (Qwen3-1.7B, Qwen3-4B) with and without a user opinion.
2. **Best-of-N** (guaranteed minimum) — sample N ∈ {1, 4, 16, 64}, pick the best
   by proxy, plot proxy and gold against log N.
3. **RL** (main result) — GRPO against the proxy reward; proxy / gold /
   sycophancy curves over training steps; transfer check on SycophancyEval.
4. **Mitigation** — delayed reward `r = (1-α)·immediate + α·delayed` for several
   α; persona mixture (70 % agreeable / 30 % truth-seeking users).

The headline figure: gold vs proxy as optimization strength grows, one curve
per α plus the persona mixture.

## Repository layout

```
configs/          YAML configs for every stage, set to the reported runs (base, data, simulator, baseline, bon, grpo, mitigation, persona_mix)
src/goodhart/
  data/           dataset builder, opinion templates, SycophancyEval loader
  simulator/      personas, scoring backends (Anthropic / vLLM / local HF judge / mock), reward functions
  policy/         sampling N responses from the policy (vLLM / HF)
  eval/           answer parsing, metrics (gold, sycophancy, proxy)
  experiments/    baseline, best_of_n, train_grpo, confidence, plots — CLI entry points
docs/DESIGN.md    data schema and module interfaces
docs/ROADMAP.md   plan, flow diagrams, results so far, timeline
docs/WRITEUP.md   the write-up (draft)
docs/JUDGE_CHECK.md  the user-simulator check
tests/            unit tests, no GPU and no network
results/          JSON outputs; baseline/, bon/, grpo/, grpo_first_pass/, grpo_smoke/, ci/ and judge_check/ are committed, samples and checkpoints are not
figures/          plots; the PNGs rendered from the committed results are tracked
```

## Reproduce

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt -e .
make test                        # unit tests, CPU only; includes the configs check below

make data                        # prompts -> data/prompts.jsonl, SycophancyEval -> data/sycophancy_eval/
make baseline                    # experiment 1 (GPU)
make bon                         # experiment 2 + BoN mitigation (GPU)
make grpo-all                    # experiments 3-4: every run under results/grpo/, ~61 GPU-hours
make ci                          # results/ci/summary.md
make plots                       # figures/*.png
```

The configs in `configs/` are the ones the reported runs used, and `requirements.txt` pins
the GPU stack they ran on (torch 2.6.0, transformers 5.20.2, TRL 0.29.1, PEFT 0.21.2).
`tests/test_reported_configs.py` rebuilds the stage config of every result under
`results/baseline`, `bon`, `grpo` and `grpo_first_pass` from `configs/` and compares it with
the stored one (`grpo_smoke/` holds the smoke runs, which used other settings); results keep only the path of the judge
config, so the judge settings are pinned to the write-up's (local Qwen2.5-7B-Instruct)
separately. `make data` rebuilds the exact `data/prompts.jsonl` the runs used and fetches
SycophancyEval from a fixed upstream commit, checked by sha256. `make grpo-all` reruns 11 of
the 13 GRPO runs; the first-pass α = 0.3 / 0.6 pair in `results/grpo_first_pass/` ran the same
config on earlier evaluation code and is not rerun. GRPO sampling on the GPU is not
bit-reproducible, so a rerun reproduces the spread across seeds, not each run to the digit.

One target at a time: `make grpo` (α = 0), `make grpo-mitigation` (α = 0.3 and 0.6 in one
call) or `make grpo-persona-mix`, with `SEED=43` for another seed. Other judges: `backend:
anthropic` in `configs/simulator.yaml` with `ANTHROPIC_API_KEY` in `.env` (`cp .env.example
.env`), or `backend: vllm` with an OpenAI-compatible server at `VLLM_BASE_URL`; the score
cache is keyed by the judge and its scoring settings, so switching never reuses old scores. vLLM generation
(`requirements-vllm.txt`) is `generation.backend: vllm` in the Best-of-N and baseline configs
and `grpo.use_vllm: true` in the GRPO ones. W&B logging is `wandb.mode` in
`configs/base.yaml`.

## Stack

| Component | Choice |
|---|---|
| Policy | Qwen3-1.7B and Qwen3-4B (small enough for RL in hours) |
| User simulator | Claude Haiku via the Anthropic API, any model behind vLLM, or a local `transformers` judge (Qwen2.5-7B-Instruct in the runs so far) |
| Gold | Ground truth from the dataset, no LLM judge |
| Training data | MMLU / ARC + opinion templates: 3 000 questions × 3 opinion types = 9 000 prompts (7 200 / 900 / 900 split) |
| Held-out eval | SycophancyEval (Sharma et al., "Towards Understanding Sycophancy in Language Models") |
| RL | TRL `GRPOTrainer` + LoRA, generation through vLLM or plain `transformers.generate` |
| Tracking | Weights & Biases (off in the reported runs) |
| Compute | 1× A100 80 GB; 83 GPU-hours for everything reported |

## Status

The write-up is [docs/WRITEUP.md](docs/WRITEUP.md) (draft). Plan, flow diagrams,
per-run tables and the timeline live in [docs/ROADMAP.md](docs/ROADMAP.md); confidence
intervals for every reported rate in [results/ci/summary.md](results/ci/summary.md); the
headline figure is [figures/goodhart_frontier.png](figures/goodhart_frontier.png).

- [x] Repository skeleton, configs, data schema, interfaces
- [x] Dataset: 9 000 prompts (3 000 MMLU / ARC questions × 3 opinion types)
- [x] Judge checked statistically (prefers agreement under a wrong opinion)
- [x] User simulator checked on all Best-of-N samples and 50 read items — [docs/JUDGE_CHECK.md](docs/JUDGE_CHECK.md)
- [x] Baseline for Qwen3-1.7B and Qwen3-4B
- [x] Best-of-N curves for Qwen3-1.7B and Qwen3-4B (incl. delayed-reward sweep over α)
- [x] GRPO smoke on Qwen3-1.7B: proxy and sycophancy rise within 200 steps, gold falls
- [x] GRPO full run on Qwen3-1.7B with held-out SycophancyEval: sycophancy 49 → 98 %, held-out 62 → 85 %
- [x] GRPO across seeds: 3 runs at α = 0, 4 each at α = 0.3 and 0.6 — α = 0.3 only slows the collapse (46–97 %), α = 0.6 holds (37–46 %) with the proxy flat
- [x] Persona mixture in GRPO, two runs: sycophancy 49 → 48 and 55 % at step 300, within noise of the start (86–98 % with the agreeable user alone), proxy +0.24 to +0.46
- [x] Confidence intervals for every reported rate (`python -m goodhart.experiments.confidence`)
- [x] Every run in results/baseline, bon and grpo reproducible from `configs/` with `make` (stage configs checked by `tests/test_reported_configs.py`)
- [ ] Write-up — draft in docs/WRITEUP.md (~1 850 words), final by Nov 17

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Hubinger et al., 2024. *Sleeper Agents.*
- Anthropic, 2025. *Recommendations for Technical AI Safety Research Directions.*

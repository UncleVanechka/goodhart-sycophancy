# Goodhart's law for LLMs: proxy-reward optimization breeds sycophancy

Optimizing a recommender for clicks breeds clickbait. Optimizing an LLM for
"user satisfaction" breeds sycophancy. This repo is a small, fully reproducible
experiment that measures the second effect and tests whether two mitigations
borrowed from recommender systems — guardrail metrics and *delayed* (long-term)
reward instead of immediate reward — reduce it.

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

1. **Baseline** — sycophancy rate of 2–3 open models with and without a user opinion.
2. **Best-of-N** (guaranteed minimum) — sample N ∈ {1, 4, 16, 64}, pick the best
   by proxy, plot proxy and gold against log N.
3. **RL** (main result) — GRPO against the proxy reward; proxy / gold /
   sycophancy curves over training steps; transfer check on SycophancyEval.
4. **Mitigation** — delayed reward `r = (1-α)·immediate + α·delayed` for several
   α; persona mixture if time allows.

The headline figure: gold vs proxy as optimization strength grows, one curve
per α.

## Repository layout

```
configs/          YAML configs for every stage (base, data, simulator, bon, grpo, mitigation)
src/goodhart/
  data/           dataset builder, opinion templates, SycophancyEval loader
  simulator/      personas, scoring backends (Anthropic / vLLM / mock), reward functions
  policy/         sampling N responses from the policy (vLLM / HF)
  eval/           answer parsing, metrics (gold, sycophancy, proxy)
  experiments/    baseline, best_of_n, train_grpo, plots — CLI entry points
docs/DESIGN.md    data schema and module interfaces
tests/            unit tests, no GPU and no network
results/          JSON/CSV outputs (git-ignored except .gitkeep)
figures/          plots (git-ignored except .gitkeep)
```

## Reproduce

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -e .
cp .env.example .env            # ANTHROPIC_API_KEY, WANDB_API_KEY, HF_TOKEN
make test                        # unit tests, CPU only

make data                        # build prompts -> data/prompts.jsonl
make baseline                    # experiment 1
make bon                         # experiment 2 + BoN mitigation
make grpo                        # experiment 3 (GPU)
make grpo-mitigation             # experiment 4 (GPU)
make plots                       # figures/*.png
```

Every stage reads one YAML from `configs/`, uses the seed from `configs/base.yaml`,
and writes a JSON with its config and outputs to `results/`.

## Stack

| Component | Choice |
|---|---|
| Policy | Qwen 1.5–4B Instruct (small enough for RL in hours) |
| User simulator | Claude Haiku via the Anthropic API, or any model behind vLLM |
| Gold | Ground truth from the dataset, no LLM judge |
| Training data | MMLU / ARC + opinion templates, 2–5k prompts |
| Held-out eval | SycophancyEval (Sharma et al., "Towards Understanding Sycophancy in Language Models") |
| RL | TRL `GRPOTrainer` + LoRA, generation through vLLM |
| Tracking | Weights & Biases |
| Compute | 1× A100/H100 80 GB, ~40–60 GPU-hours |

## Status

- [x] Repository skeleton, configs, data schema, interfaces
- [ ] User simulator validated by hand on 50 examples
- [ ] Baseline for 2–3 models
- [ ] Best-of-N curves
- [ ] GRPO run
- [ ] Mitigation runs
- [ ] Write-up

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Hubinger et al., 2024. *Sleeper Agents.*
- Anthropic, 2025. *Recommendations for Technical AI Safety Research Directions.*

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
per α plus the persona mixture and the reply floor.

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
results/          JSON outputs; baseline/, bon/, grpo/, grpo_600/, grpo_floor/, grpo_first_pass/, grpo_smoke/, ci/, judge_check/ and analysis/ are committed, samples, per-question rows and checkpoints are not
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

# registered in docs/PREREG.md v2; not part of grpo-all
make grpo-persona-mix-600 SEED=44  # 600-step persona mixture (GPU), configs/persona_mix_600.yaml -> results/grpo_600/ (then SEED=45 and SEED=46)
make reeval-600                  # endpoint re-evaluation of their checkpoints 100-600 (GPU), configs/reeval_600.yaml -> results/reeval_600/
make prereg-v2                   # registered reading of the three runs (CPU) -> results/analysis/prereg_v2.{json,md}

# registered in docs/PREREG.md v3; not part of grpo-all (GPU, except the two readings)
make grpo-floor                  # reply-floor arm, configs/grpo_floor.yaml -> results/grpo_floor/ (then SEED=43 and SEED=44)
make reeval-floor                # its endpoint re-evaluation, configs/reeval_floor.yaml -> results/reeval_floor/
make are-you-sure                # two-turn evaluation of the saved adapters, configs/are_you_sure.yaml -> results/are_you_sure/
make prereg-floor                # registered reading of the floor arm (CPU) -> results/analysis/prereg_floor.{json,md}
make prereg-are-you-sure         # registered reading of the two-turn evaluation (CPU) -> results/analysis/prereg_are_you_sure.{json,md}

# registered in docs/PREREG.md v4, part F; not part of grpo-all (GPU, except the reading)
make grpo-floor-600              # reply-floor arm to 600 steps, configs/grpo_floor_600.yaml -> results/grpo_floor_600/ (then SEED=43 and SEED=44)
make reeval-floor-600            # re-evaluation of checkpoint 300 and the final adapter, configs/reeval_floor_600.yaml -> results/reeval_floor_600/
make prereg-floor-600            # registered reading of the 600-step floor arm (CPU) -> results/analysis/prereg_floor_600.{json,md}

# registered in docs/PREREG.md v4, part L; not part of grpo-all (GPU, except the reading)
make are-you-sure-named          # a second turn naming the question's wrong letter, configs/are_you_sure_named.yaml -> results/are_you_sure_named/
make prereg-named-letter         # registered reading of the named-letter turn (CPU) -> results/analysis/prereg_named_letter.{json,md}
```

The training and Best-of-N configs in `configs/` (`baseline`, `bon`, `grpo`, `mitigation`,
`persona_mix`) are the ones the reported runs used, and `requirements.txt` pins
the GPU stack they ran on (torch 2.6.0, transformers 5.20.2, TRL 0.29.1, PEFT 0.21.2).
`grpo_floor.yaml`, `reeval_floor.yaml` and `are_you_sure.yaml` belong to the two experiments
registered in [docs/PREREG.md](docs/PREREG.md) v3; the two evaluation configs read saved
adapters and re-evaluation rows copied under `data/`, as the header of each config says.
The floor runs' logged outputs are stored in `results/grpo_floor/` and the registered readings
of both experiments in `results/analysis/prereg_floor.{json,md}` and
`prereg_are_you_sure.{json,md}`; the re-evaluation and two-turn rows they read are not in the
repository.
`grpo_floor_600.yaml` and `reeval_floor_600.yaml` belong to part F of docs/PREREG.md v4: three
new floor runs to 600 steps and the re-evaluation of their checkpoint 300 and final adapter (and
of the last checkpoint of a run cut short); `make prereg-floor-600` reads them against the frozen
base rows of v3.1, with the v3 floor runs for the step-0 comparison.
`are_you_sure_named.yaml` belongs to part L of docs/PREREG.md v4: it decodes turn 1 again and
one second turn built from each question's wrong-opinion record in `data/prompts.jsonl`, and
`make prereg-named-letter` reads its rows with part B's rows as controls and the paired
re-evaluation for the single-turn rate.
`persona_mix_600.yaml` and `reeval_600.yaml` belong to the three 600-step persona-mixture
runs registered in docs/PREREG.md v2. Their logged outputs are stored in `results/grpo_600/`
and their registered reading in `results/analysis/prereg_v2.{json,md}`, written by
`make prereg-v2` (`python -m goodhart.analysis.prereg_v2 --runs <dir> --reeval <dir> [<dir> ...]
--base-items <items.jsonl>`). The checkpoints of these runs and the per-question rows of their
re-evaluation are not in the repository: the endpoint, the classes and the rule in that
report can be regenerated only after `make reeval-600` on saved checkpoints (and `make
reeval` for the base policy), while its logged half (proxy gain, directional prediction,
early-warning check, condition contrast) follows from the stored runs and is recomputed by
`tests/test_prereg_v2.py`.
The 15 predictions of docs/PREREG.md v1 are read in three part files under `results/analysis/`:
`prereg_m2.json` (`python -m goodhart.analysis.prereg_eval --reeval <dir> [<dir> ...]
--sample-rate <dir>`), `prereg_m3.json` (`python -m goodhart.analysis.prereg_checkpoints
--samples <dir> [<dir> ...]`) and `prereg_m4.json` (`python -m goodhart.analysis.prereg_judges`).
They read the per-question and per-sample outputs of `make reeval`, `make sample-rate` and
`make rescore` (the base policy, the 10 saved adapters and the Best-of-N samples; the first two
modules default to `results/reeval` and `results/sample_rate`, where those targets write them);
neither the adapters nor those rows are in the repository, so these files are committed outputs
and are not rebuilt by any target. They name their inputs by a label given on the command line
(`--reeval-label "paired re-evaluation, 2026-11-10"` and `--sample-rate-label "base-policy
samples, 2026-11-10"` for `prereg_m2.json`, `--label "rollout-sampler samples, 2026-11-10 and
2026-11-11"` for `prereg_m3.json`) and each policy by the position of the directory it was read
from, never by a path: the re-evaluation was read from three directories (the base policy and
the step-300 adapters, then checkpoints 100 and 200 in two), the samples from two (one per day).
`prereg_results.{json,md}` merges the three parts with the corrections of an independent check
and has no command of its own.
`tests/test_reported_configs.py` rebuilds the stage config of every result under
`results/baseline`, `bon`, `grpo`, `grpo_600`, `grpo_floor` and `grpo_first_pass` from `configs/` and compares it with
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
| Compute | About 176 GPU-hours in all, on one A100 80 GB |

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
- [x] Persona mixture in GRPO, five runs (two of 300 steps, three of 600 registered in docs/PREREG.md v2): slows the drift, does not stop it — 48.5 and 55.8 % at step 300, 48.2, 62.2 and 74.3 % at step 600 on the 600 wrong-opinion validation and test questions (the last at 446 of 600 against a collapse line of 450, and past it on its validation half; untrained model 48.0 %; 85.3 and 95.2 % at step 300 for the two re-evaluated runs with the agreeable user alone), proxy +0.21 to +0.46 at step 300 — [results/analysis/prereg_v2.md](results/analysis/prereg_v2.md)
- [x] 15 predictions registered before the re-analysis of the saved adapters ([docs/PREREG.md](docs/PREREG.md)): 9 confirmed, 2 refuted, 2 disputed, 1 partly, 1 classified — [results/analysis/prereg_results.md](results/analysis/prereg_results.md)
- [x] Reply-floor arm and two-turn "are you sure?" evaluation, registered in docs/PREREG.md v3: with the floor, sycophancy ends at 43–45 % on the 600 wrong-opinion validation and test questions after 300 steps (untrained model 48.0 %; registered class "blocks within 300 steps", where a rise was predicted), though it creeps back after an early dip, and the 176 comparable questions would usually miss a 5-point rise; under "are you sure?" no adapter drops a correct answer measurably more often than the untrained model (1.5 %; every p ≥ 0.25) — [results/analysis/prereg_floor.md](results/analysis/prereg_floor.md), [results/analysis/prereg_are_you_sure.md](results/analysis/prereg_are_you_sure.md)
- [x] Confidence intervals for every reported rate (`python -m goodhart.experiments.confidence`)
- [x] Every run in results/baseline, bon, grpo, grpo_600 and grpo_floor reproducible from `configs/` with `make` (stage configs checked by `tests/test_reported_configs.py`)
- [ ] Write-up — draft in docs/WRITEUP.md (~3 000 words with tables, revised Nov 13–14 for the pre-registered results), final by Nov 17

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Hubinger et al., 2024. *Sleeper Agents.*
- Anthropic, 2025. *Recommendations for Technical AI Safety Research Directions.*

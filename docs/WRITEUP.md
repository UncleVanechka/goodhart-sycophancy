# Optimizing an LLM for a simulated user's approval breeds sycophancy; a delayed reward fixes it under selection and only half under training

*Draft, 2026-11-10. Code, configs and every result file behind the numbers
are in this repository; intervals in [`results/ci/summary.md`](../results/ci/summary.md).*

**TL;DR.** On multiple-choice questions where the user states an opinion,
sometimes wrong, and a simulated user rates the answer 1–10, optimizing the
rating buys agreement, not accuracy. Best-of-64 by the rating raises the share
of answers that follow a wrong opinion by 7–8 points on Qwen3-1.7B and
Qwen3-4B with accuracy flat. GRPO on Qwen3-1.7B takes that share from 49 % to
86–98 % in 300 steps, accuracy falls from 65 % to 54–59 %, and the effect
reaches a held-out benchmark (+13 to +23 points). Mixing in a *delayed* rating
from the same user after they learn the truth works under selection, but under
training it is unstable at weight α = 0.3 (runs end at 46–97 %) and holds at
α = 0.6 only by giving up the proxy. A mixture of 70 % agreeable and 30 %
truth-seeking users is the only trained setting that raised the proxy (by
about half the unmitigated gain) without a measurable rise in sycophancy (48
and 55 % against a 49 % start), though α = 0.6 ends lower; two runs do not
show it is stable.

![Goodhart frontier](../figures/goodhart_frontier.png)

*Figure 1. Gold accuracy against the agreeable user's rating. Solid:
Best-of-N, N = 1 → 64, both models, α ∈ {0, 0.3, 0.6, 1}. Dashed: GRPO on
Qwen3-1.7B, mean over runs per α and over the two persona-mixture runs.
Selection moves right along a flat or rising line; training on the immediate
rating moves right and down; training at α = 0.6 does not move.*

## 1. Why

Recommenders know Goodhart's law: enough pressure on the click produces
content that wins the click and loses the user, and the remedies are delayed
outcomes and audience diversity. LLM post-training has the same structure — a
reward model stands in for a user who likes to be agreed with — so the
question is quantitative: **as optimization against a simulated user's
approval grows, how fast does sycophancy grow, and do the recommender remedies
transfer from the feed to the model?**

Hypotheses fixed before any run: (1) a Goodhart curve — proxy up, accuracy
flat then down, sycophancy up; (2) a delayed reward with weight α flattens the
accuracy drop at the same proxy level; (3) a mixture of personas helps; (4)
learned sycophancy shows up on a held-out benchmark.

## 2. Setup

**Task.** 3 000 MMLU and ARC questions, each rendered with no user opinion,
with the user stating the *correct* letter, and with the user stating a
*wrong* one ("I think the answer is B, but please check"; ten phrasings): 9
000 prompts, split 7 200 / 900 / 900 by question.

**Simulated user (the judge).** Qwen2.5-7B-Instruct with a persona prompt,
greedy, answering `{"score": N}`. The *agreeable* user's rating right after
reading the reply is the proxy, the "click"; the same user *informed* of the
correct answer gives the delayed signal; the delayed reward is
`(1-α)·immediate + α·informed`. A *truth-seeking* user, who rates bare
agreement with a wrong opinion low, appears only in the persona mixture: each
training question gets one persona for the whole run, drawn 0.7 agreeable /
0.3 truth-seeking and never shown to the policy.

**Metrics.** *Gold accuracy* against the dataset key. *Sycophancy rate*: share
of wrong-opinion prompts answered with the user's letter. *Flip rate*: among
questions answered correctly without an opinion, the share flipped to the
user's wrong letter. *Proxy*: the agreeable user's mean rating. Held out:
SycophancyEval (Sharma et al., 2023), 300 items with their own sources and
templates, never trained on; each seed draws its own sample, so held-out
numbers are changes against each run's own start.

**Optimization strength.** (i) Baseline: one greedy answer. (ii) Best-of-N: N
∈ {1, 4, 16, 64} samples at temperature 1, keep the one the reward prefers;
N = 1 is the reference (it matches the greedy baseline within noise). (iii)
GRPO with LoRA (rank 16): 300 steps of 8 prompts × 8 completions, learning
rate 3e-5, KL penalty 0.04, no thinking mode, evaluation every 25 steps on 300
validation questions × 3 opinions and on the held-out items. Qwen3-1.7B and
Qwen3-4B for (i) and (ii), Qwen3-1.7B for (iii); one A100 80 GB, 83 GPU-hours,
78 of them GRPO.

**Uncertainty.** Baseline and Best-of-N carry 95 % bootstrap intervals over
questions, paired when two selections share the questions. GRPO rates carry
Wilson intervals for one checkpoint, which ignore a run's step-to-step
movement, and min–max over runs is a spread, not an interval. GRPO on the GPU
is not bit-reproducible — a same-seed pair diverged as far as different seeds
do — so each α was trained three to four times and conclusions rest on run
counts.

## 3. Result 1: the Goodhart curve appears without training

![Best-of-N curves](../figures/bon_curves.png)

*Figure 2. Best-of-N: proxy, gold accuracy and sycophancy against N, one line
per α, both models.*

Selection by the agreeable user's rating, paired differences N = 64 minus
N = 1:

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| proxy (1–10) | +0.88 [+0.71, +1.05] (7.95 → 8.82) | +0.97 [+0.83, +1.12] (8.05 → 9.02) |
| gold accuracy, points | +1.7 [−0.7, +4.2] (69.2 → 70.8 %) | +1.3 [−1.2, +4.0] (79.2 → 80.5 %) |
| sycophancy rate, points | **+7.5 [+2.0, +12.5]** (45.5 → 53.0 %) | **+7.0 [+2.5, +11.5]** (27.5 → 34.5 %) |
| flip rate, points | +8.4 [+1.5, +15.7] (34.3 → 42.8 %) | +9.6 [+3.8, +15.6] (19.0 → 28.6 %) |

Accuracy is flat overall: it rises where the user has no opinion (1.7B 67 →
72.5 %) and falls where the user is wrong (46 → 43 %), both within the
±6-point noise of 200 questions. The rating is a decent proxy for quality
exactly until the user is wrong; there, sycophancy and flips rise by 7–10
points on both models, intervals excluding zero.

## 4. Result 2: training amplifies it, and it transfers

![GRPO curves](../figures/grpo_curves.png)

*Figure 3. GRPO on Qwen3-1.7B, mean per series (each α, and the persona
mixture) with min–max bands: both ratings, the training reward, gold,
sycophancy and the held-out metrics against training step.*

GRPO against the immediate rating, three seeds, step 0 → 300:

| | start | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| proxy (1–10) | 7.79 | 8.85 | 8.40 | 8.37 |
| gold accuracy | 65.3 % | 53.6 % [50.3, 56.8] | 59.0 % [55.8, 62.2] | 55.2 % [52.0, 58.4] |
| sycophancy rate | 49.3 % [43.7, 55.0] | 97.7 % [95.3, 98.9] | 86.0 % [81.6, 89.5] | 94.3 % [91.1, 96.4] |
| gold under a wrong opinion | 42.0 % | 2.0 % | 11.7 % | 4.0 % |
| gold without an opinion | 64.7 % | 59.3 % | 67.0 % | 63.3 % |
| held-out sycophancy, own start → end | | 62.4 → 85.2 % | 53.4 → 66.4 % | 67.1 → 80.5 % |
| informed rating (the delayed signal) | 7.53 | — | 6.80 | 6.38 |
| training-completion length, tokens | 67–80 | 7 | 32 | 14 |

The policy learned to agree, not to answer: accuracy under a wrong opinion
falls from 42 % to 2–12 %, under the correct opinion it reaches 98–99 %, and
without an opinion it moves by −5 to +2 points. Completions shrink to a few
tokens and the proxy saturates (seven to nine groups in ten end with eight
identical ratings), while the informed rating, which the optimizer never sees,
falls to 6.4–6.8. On SycophancyEval sycophancy rises by 13–23 points in every
seed, with non-overlapping intervals in one (hypothesis 4). Selection showed a
7-point effect; training makes it 37–48 points.

## 5. Result 3: the delayed reward, under selection and under training

**Under selection** the recommender trick transfers. Best of 64 by the mixed
rating, paired against α = 0 on the same questions:

| N = 64 | proxy | gold, points | sycophancy, points |
|---|---|---|---|
| Qwen3-1.7B, α = 0.3 vs 0 | −0.04 [−0.05, −0.02] | +3.3 [+2.0, +5.0] | −4.5 [−8.0, −1.5] |
| Qwen3-1.7B, α = 0.6 vs 0 | −0.23 [−0.32, −0.15] | +6.5 [+4.5, +8.7] | −13.0 [−18.0, −8.5] |
| Qwen3-4B, α = 0.3 vs 0 | −0.01 [−0.02, −0.00] | +4.3 [+2.3, +6.2] | −5.5 [−9.0, −2.0] |
| Qwen3-4B, α = 0.6 vs 0 | −0.09 [−0.11, −0.06] | +6.7 [+4.5, +8.8] | −10.5 [−15.5, −6.0] |

At α = 0.3 the proxy barely moves (−0.04 and −0.01) while accuracy gains 3–4
points and sycophancy loses 4–6 — the "same approval, less sycophancy" trade
of hypothesis 2. At α = 1 both models beat their own N = 1 reference on
accuracy by 9–10 points (1.7B 78.0 % [74.3, 81.7] vs 69.2 %; 4B 89.0 % [85.5,
92.2] vs 79.2 %), with sycophancy at 37 % and 15 %.

**Under training** it does not hold in that form. GRPO on Qwen3-1.7B, step
300, mean over runs (min–max):

| | α = 0, 3 runs | α = 0.3, 4 runs | α = 0.6, 4 runs | persona mix, 2 runs |
|---|---|---|---|---|
| sycophancy rate | 93 % (86–98) | 66 % (46–97) | 40 % (37–46) | 52 % (48–55) |
| gold accuracy | 56 % (54–59) | 62 % (54–67) | 67 % (66–67) | 65 % (64–66) |
| gold under a wrong opinion | 6 % (2–12) | 28 % (3–44) | 47 % (44–49) | 40 % (37–43) |
| held-out sycophancy change, points (vs own start) | +16 (+13 to +23) | +7 (+2 to +19) | −1 (−1 to +1) | +2 (+2 to +3) |
| proxy, immediate rating (from 7.79) | 8.54 (8.37–8.85) | 8.33 (8.17–8.54), n = 3 | 7.85 (7.78–7.90), n = 3 | 8.14 (8.03–8.25) |
| informed rating (from 7.53) | 6.59 (6.38–6.80), n = 2 | 7.07 (6.30–7.49), n = 3 | 7.92 (7.50–8.22), n = 3 | 7.74 (7.73–7.75) |
| training-completion length, tokens (from 67–80) | 18 (7–32) | 54 (32–78) | 71 (23–89) | 70 (57–82) |

*α = 0.3 slows the collapse but does not stop it.* The four runs end at 46,
60, 61 and 97 %; the middle two are still climbing (42 → 61 % and 45 → 60 %
over the last 100 steps), and the 97 % run followed the α = 0 path (immediate
rating 7.79 → 8.54, informed 7.53 → 6.30). Agreement still pays locally at
α = 0.3 (on the base model's samples an agreeing bare answer gains 3.6 points
from the agreeable user and loses 5.3 from the informed one), yet the 97 % run
ended with the lowest mixed reward of the four: the sycophantic policy is a
basin the optimizer can fall into, not the optimum of its objective.

*α = 0.6 holds in all four runs, by not optimizing the proxy.* Sycophancy ends
at 37–46 % (all below the 49 % start, two with non-overlapping intervals) and
accuracy at 66–67 %, but the agreeable user's rating stays flat at 7.78–7.90;
what rose is the informed rating, in two of the three runs that recorded it.
The delayed reward did not buy the same approval with less sycophancy; it
replaced the objective. Under selection α = 0.6 kept 74 % of the proxy gain on
1.7B and 91 % on 4B; under training it kept none. Nor is it obviously safe at
longer horizons: the seed-44 run climbs from 35 % at step 200 back to 46 %
(start 49 %), with completions down to 23 tokens over the last 50 steps. With
two α values trained, the threshold between "slows" and "holds" is bracketed,
not located.

*A mixture of users holds sycophancy near its start and keeps part of the
proxy gain.* Two runs (seeds 42 and 43). Both dip to 41 % within 50 steps, as
most runs do, two of the three agreeable-only ones included, so the dip says
nothing about the truth-seeking minority. Then they wander: seed 42 climbs
from 43 % at step 150 to 55 % [50, 61], seed 43 reaches 55 % at step 150 and
returns to 48 % [42, 53]. Neither endpoint is beyond noise of the 49 % start
or of the other (±8 points on a difference at n = 300). Accuracy stays at
64–66 %, and the immediate rating rises by 0.46 and 0.24 — about half the mean
gain with the agreeable user alone (+0.75). That is the trade α = 0.6 did not
make: some approval without a measurable rise in sycophancy. Hypothesis 3 as
stated holds (48–55 % against 86–98 %). Stability is not settled: agreement
still wins 70 % of the draws, and α = 0.3 run twice with seed 42 ended at 46
and 97 %.

## 6. What the judge is, and is not

Everything above assumes the agreeable persona rewards agreement. On all 38
400 Best-of-N ratings per model, and in 50 items read under a fixed rubric by
an LLM reader, not a human ([JUDGE_CHECK.md](JUDGE_CHECK.md)), it does: under
a wrong opinion it rates an agreeing reply 8.4 and a correct one 4.8 on
Qwen3-1.7B (bare letters), 7.9 against 6.8 on Qwen3-4B (reasoned replies), and
of 20 read pairs it preferred the agreeing reply in 12, tied 4 and preferred
the correction in 4. An argued correction is barely punished, a bare
contradicting letter is, and the scale is bimodal (7–9 or 2). Best-of-N and
GRPO use only the ranking within a prompt, so these quirks weaken the signal
rather than invert it, but absolute ratings are not quality scores.

## 7. Limitations

- **One small policy in training.** GRPO ran on Qwen3-1.7B, which answers
  with a bare letter; the judge forgives the argued disagreement of the 4B
  model, so the α = 0.3 instability may move or vanish for a model that
  explains itself.
- **One judge.** A sharper persona, a pairwise judge or a second judge model
  would change every slope; the direction is the claim, not the slope.
- **Synthetic user, synthetic task.** Multiple choice with an explicit wrong
  opinion is the cleanest case to measure and the easiest to exploit; the
  held-out benchmark shares the format.
- **Few runs, 300 steps.** Three to four runs per α show that α = 0.3 is
  unstable and α = 0.6 held, not where the threshold is or what happens at
  1 000 steps. Both persona-mixture runs end near their start, but so did the
  first α = 0.3 run before a same-seed repeat reached 97 %, so two runs bound
  the collapse risk only loosely.

## 8. What this suggests

The delayed-outcome fix transfers to selection-time reranking as advertised,
and to training only when the delayed signal dominates the reward — at which
point it is a different objective, not a correction to the proxy. The cheap
version, a small α that keeps the proxy, fails under RL. A 30 % share of users
who punish agreement did better in two runs, pending more seeds. Next: a
sharper agreeable persona and a pairwise judge, the same ladder on Qwen3-4B,
longer runs at α = 0.6, and more persona-mixture seeds at other truth-seeking
shares.

## Reproduce

```bash
make data                                                                  # MMLU + ARC -> data/prompts.jsonl
python -m goodhart.experiments.baseline  --config configs/bon.yaml         # results/baseline/
python -m goodhart.experiments.best_of_n --config configs/bon.yaml         # results/bon/
python -m goodhart.experiments.train_grpo --config configs/grpo.yaml       # results/grpo/grpo-immediate/
python -m goodhart.experiments.train_grpo --config configs/mitigation.yaml # results/grpo/grpo-delayed-a*/
#   persona mix: same config with run_name=grpo-persona-mix, reward.kind=persona_mix, sweep.alphas=[] (seed 43: seed=43, run_name=grpo-persona-mix-s43)
python -m goodhart.experiments.confidence                                  # results/ci/summary.md
python -m goodhart.experiments.plots                                       # figures/*.png
```

The committed configs default to vLLM generation and an API simulator; the
reported runs used `transformers.generate` and the local judge, and the GRPO
overrides listed in [ROADMAP.md](ROADMAP.md) §9 (300 steps, lr 3e-5 constant
after warmup, 8 × 8 completions per step, eval every 25 steps, seeds 42–44;
the persona-mixture runs set `run_name: grpo-persona-mix` (seed 43: `seed: 43`,
`run_name: grpo-persona-mix-s43`), `reward.kind: persona_mix` with weights
0.7 / 0.3, and `sweep.alphas: []`).
Every result JSON stores the merged config it ran with.

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO).

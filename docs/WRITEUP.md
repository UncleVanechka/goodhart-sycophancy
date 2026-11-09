# Optimizing an LLM for a simulated user's approval breeds sycophancy; a delayed reward fixes it under selection and only half under training

*Draft, 2026-11-09. Code, configs and every result file behind the numbers
are in this repository; intervals in [`results/ci/summary.md`](../results/ci/summary.md).*

**TL;DR.** On a multiple-choice task where the user states an opinion that is
sometimes wrong and a simulated user rates the answer 1–10, optimizing that
rating produces agreement, not accuracy. Picking the best of 64 samples by the
rating lifts the share of answers that follow a wrong opinion by 7–8 points on
both Qwen3-1.7B and Qwen3-4B (paired intervals exclude zero) while accuracy
stays flat. Training Qwen3-1.7B against the rating with GRPO takes that share
from 49 % to 86–98 % in 300 steps across three seeds, accuracy falls from 65 %
to 54–59 %, and the effect carries to a held-out benchmark the policy never
saw (SycophancyEval, +13 to +23 points). The recommender-systems remedy — mix
in a *delayed* rating from the same user after they learn the truth — works
under selection as advertised, but under training it is unstable at weight
α = 0.3 (four runs end at 46, 60, 61 and 97 %) and holds at α = 0.6 only by
giving up the proxy: the immediate rating does not rise at all.

![Goodhart frontier](../figures/goodhart_frontier.png)

*Figure 1. Gold accuracy against the agreeable user's rating. Solid: Best-of-N,
N = 1 → 64, both models, α ∈ {0, 0.3, 0.6, 1}. Dashed: GRPO on Qwen3-1.7B,
mean over runs per α. Selection moves right along a flat or rising line;
training on the immediate rating moves right and down; training at α = 0.6
does not move.*

## 1. Why

Goodhart's law for recommenders is familiar: the click is easy to measure, the
satisfied user next month is not, and enough pressure on the click produces
content that wins the click and loses the user. Recommenders answer with
delayed outcomes and audience diversity. LLM post-training has the same
structure — a reward model stands in for a user who likes to be agreed with —
so the question is quantitative: **as optimization strength against a
simulated user's approval grows, how fast does sycophancy grow, and does the
delayed-reward trick transfer from the feed to the model?**

Four hypotheses were fixed before any run: (1) a Goodhart curve — proxy up,
accuracy flat then down, sycophancy up; (2) a delayed reward with weight α
flattens the accuracy drop at the same proxy level; (3) a mixture of personas
helps (not run); (4) sycophancy learned on our prompts shows up on a held-out
benchmark.

## 2. Setup

**Task.** 3 000 MMLU and ARC questions, each rendered three ways: no user
opinion, the user stating the *correct* letter, the user stating a *wrong*
letter ("I think the answer is B, but please check"), ten phrasings. 9 000
prompts, split 7 200 / 900 / 900 by question.

**Simulated user (the judge).** Qwen2.5-7B-Instruct with a persona prompt,
greedy, answering `{"score": N}`. Two personas: the *agreeable* user who rates
the reply right after reading it — its rating is the proxy, the "click" — and
the same user *informed* of the correct answer afterwards — the delayed
signal. The delayed reward is `(1-α)·immediate + α·informed`. Below, "judge"
is the model and "agreeable" / "informed" are its two ratings.

**Metrics.** *Gold accuracy*: correctness against the dataset key, no LLM
involved. *Sycophancy rate*: share of wrong-opinion prompts answered with the
user's letter. *Flip rate*: among questions answered correctly without an
opinion, the share flipped to the user's wrong letter. *Proxy*: the agreeable
user's mean rating. Held-out check: SycophancyEval (Sharma et al., 2023), 300
items with their own question sources and prompt templates, never used in
training; each training seed draws its own 300-item sample (146–149 carry a
wrong opinion), so held-out numbers are changes against each run's own start.

**Three levels of optimization strength.** (i) Baseline, one greedy answer.
(ii) Best-of-N: sample N ∈ {1, 4, 16, 64} answers at temperature 1, keep the
one the reward prefers — pressure without training; the N = 1 column is a
single sample and is the reference for selection effects (it matches the
greedy baseline within noise: 1.7B 45.5 vs 45.7 % sycophancy, 4B 27.5 vs
23.3 %). (iii) GRPO with LoRA (rank 16) against the reward: 300 steps of 8
prompts × 8 completions, learning rate 3e-5 constant after a 10-step warmup, KL
penalty 0.04, no thinking mode, evaluation every 25 steps on 300 validation
questions × 3 opinions and on the held-out items. Policies: Qwen3-1.7B and
Qwen3-4B for (i) and (ii), Qwen3-1.7B for (iii). Single-GPU jobs on one A100
80 GB, 71 GPU-hours in total, 60 of them GRPO.

**Uncertainty.** Baseline and Best-of-N rates carry 95 % bootstrap intervals
over questions (a question is resampled with its three variants); comparisons
between selections on the same questions use a paired bootstrap. GRPO rates
carry Wilson intervals from the evaluation counts of one checkpoint; they do
not cover the step-to-step movement of a run, and min–max over three to four
runs is a spread, not an interval. GRPO sampling on the GPU is not
bit-reproducible: of the two same-seed, same-config pairs in the data, one
diverged as far as different seeds do, so every α was trained three to four
times and conclusions rest on run counts.

## 3. Result 1: the Goodhart curve appears without training

![Best-of-N curves](../figures/bon_curves.png)

*Figure 2. Best-of-N: proxy, gold accuracy and sycophancy against N, one line
per α, both models.*

Selecting the best of N by the agreeable user's rating moves the proxy
monotonically and the policy's behaviour with it. Paired differences,
N = 64 minus N = 1, selection by the immediate rating:

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| proxy (1–10) | +0.88 [+0.71, +1.05] (7.95 → 8.82) | +0.97 [+0.83, +1.12] (8.05 → 9.02) |
| gold accuracy, points | +1.7 [−0.7, +4.2] (69.2 → 70.8 %) | +1.3 [−1.2, +4.0] (79.2 → 80.5 %) |
| sycophancy rate, points | **+7.5 [+2.0, +12.5]** (45.5 → 53.0 %) | **+7.0 [+2.5, +11.5]** (27.5 → 34.5 %) |
| flip rate, points | +8.4 [+1.5, +15.7] (34.3 → 42.8 %) | +9.6 [+3.8, +15.6] (19.0 → 28.6 %) |

Accuracy does not fall under selection: it rises where the user has no
opinion (1.7B 67 → 72.5 %) and falls where the user is wrong (46 → 43 %),
both within the ±6-point noise of 200 questions, netting to flat. The rating
is a decent proxy for quality exactly until the user is wrong. The sycophancy
and flip rates rise by 7–10 points on both models with intervals that exclude
zero.

## 4. Result 2: training amplifies it, and it transfers

![GRPO curves](../figures/grpo_curves.png)

*Figure 3. GRPO on Qwen3-1.7B, mean over runs per α with min–max bands: the
two ratings, the training reward, gold, sycophancy and the held-out metrics
against training step.*

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

The policy did not learn to answer better; it learned to agree. Accuracy
under a wrong opinion goes from 42 % to 2–12 %, under the correct opinion to
98–99 %, and without an opinion it moves by −5 to +2 points. Completions
shrink to a few tokens, and by the end seven to nine groups in ten have eight
identical ratings — the proxy saturates near the top of the scale and the
reward has little left to teach — while the informed rating, which the
optimizer never sees, falls from 7.5 to 6.4–6.8. On SycophancyEval — other
questions, other templates — sycophancy rises by 13–23 points in every seed,
with non-overlapping intervals in one (hypothesis 4). Selection hinted at a
7-point effect; training makes it 37–48 points.

## 5. Result 3: the delayed reward, under selection and under training

**Under selection** the recommender trick transfers cleanly. Best of 64 by
the mixed rating, paired against α = 0 on the same questions:

| N = 64 | proxy | gold, points | sycophancy, points |
|---|---|---|---|
| Qwen3-1.7B, α = 0.3 vs 0 | −0.04 [−0.05, −0.02] | +3.3 [+2.0, +5.0] | −4.5 [−8.0, −1.5] |
| Qwen3-1.7B, α = 0.6 vs 0 | −0.23 [−0.32, −0.15] | +6.5 [+4.5, +8.7] | −13.0 [−18.0, −8.5] |
| Qwen3-4B, α = 0.3 vs 0 | −0.01 [−0.02, −0.00] | +4.3 [+2.3, +6.2] | −5.5 [−9.0, −2.0] |
| Qwen3-4B, α = 0.6 vs 0 | −0.09 [−0.11, −0.06] | +6.7 [+4.5, +8.8] | −10.5 [−15.5, −6.0] |

At α = 0.3 the proxy is unchanged to the second decimal while accuracy gains
3–4 points and sycophancy loses 4–6 — the "same approval, less sycophancy"
trade hypothesis 2 asks for, measurable here because the proxy is held. At
α = 1 both models beat their own N = 1 reference on accuracy by 9–10 points
(1.7B 78.0 % [74.3, 81.7] vs 69.2 % [65.0, 73.2]; 4B 89.0 % [85.5, 92.2] vs
79.2 % [74.8, 83.5]) with sycophancy at 37 % and 15 %.

**Under training** it does not hold in that form. GRPO on Qwen3-1.7B with the
mixed reward, step 300, mean over runs (min–max):

| | α = 0, 3 runs | α = 0.3, 4 runs | α = 0.6, 4 runs |
|---|---|---|---|
| sycophancy rate | 93 % (86–98) | 66 % (46–97) | 40 % (37–46) |
| gold accuracy | 56 % (54–59) | 62 % (54–67) | 67 % (66–67) |
| gold under a wrong opinion | 6 % (2–12) | 28 % (3–44) | 47 % (44–49) |
| held-out sycophancy change, points (vs own start) | +16 (+13 to +23) | +7 (+2 to +19) | −1 (−1 to +1) |
| proxy, immediate rating (from 7.79) | 8.54 (8.37–8.85) | 8.33 (8.17–8.54), n = 3 | 7.85 (7.78–7.90), n = 3 |
| informed rating (from 7.53) | 6.59 (6.38–6.80), n = 2 | 7.07 (6.30–7.49), n = 3 | 7.92 (7.50–8.22), n = 3 |
| training-completion length, tokens (from 67–80) | 18 (7–32) | 54 (32–78) | 71 (23–89) |

*α = 0.3 slows the collapse; it does not stop it.* The four runs end at 46,
60, 61 and 97 % sycophancy. The two in the middle are still climbing at step
300 (42 → 61 % and 45 → 60 % over the last 100 steps); the one that reached
97 % followed the α = 0 path — immediate rating 7.79 → 8.54, informed rating
7.53 → 6.30. On the judge's own ratings of the base model's samples an
agreeing bare answer gains 3.6 points from the agreeable user and loses 5.3
from the informed one, so at α = 0.3 agreement still pays locally. Yet that
run ended with the lowest mixed reward of the four (+0.15 against +0.23 to
+0.33): at α = 0.3 the sycophantic policy is a basin the optimizer can fall
into, not the optimum of the objective it was given.

*α = 0.6 holds in all four runs, by not optimizing the proxy at all.*
Sycophancy ends at 37–46 % (all four below the 49 % start, two with
non-overlapping intervals), accuracy at 66–67 %, accuracy under a wrong
opinion up 2–7 points, the KL to the reference policy at 0.06–0.12 (one run
spiked to 0.97 mid-way and came back). But the agreeable user's rating is
flat at 7.78–7.90, and what rose is the informed rating, in two of the three
runs that recorded it. The delayed reward did not buy "the same approval with
less sycophancy"; it replaced the approval objective with the informed one.
Under selection the same α = 0.6 kept 74 % of the proxy gain on 1.7B and 91 %
on 4B; under training it kept none. Nor is 0.6 obviously safe at longer
horizons: the seed-44 run returns from 35 % to its starting 46–49 % over its
last 50 steps while completions shorten to 23 tokens and 83–90 % of groups
tie. With two α values trained, the threshold between "slows" and "holds" is
not located, only bracketed.

## 6. What the judge is, and is not

Everything above depends on the agreeable persona actually rewarding
agreement. We checked that on all 38 400 Best-of-N ratings per model and had
50 items read under a fixed rubric by an LLM reader, not a human
([JUDGE_CHECK.md](JUDGE_CHECK.md)). Under a wrong opinion the persona rates
an agreeing reply 8.4 and a correct one 4.8 on Qwen3-1.7B (bare letters), 7.9
against 6.8 on Qwen3-4B (reasoned replies); in 20 read pairs it preferred the
agreeing reply 12 times, tied 4, preferred the correction 4. The preference
is real and depends on form: an argued correction is barely punished, a bare
contradicting letter is. The scale is bimodal (7–9 or 2), and the informed
persona is lenient on reasoned replies. Best-of-N and GRPO use only the
ranking within a prompt, so these quirks weaken the signal rather than invert
it, but absolute ratings are not quality scores.

## 7. Limitations

- **One small policy in training.** GRPO ran on Qwen3-1.7B, whose answers are
  a bare letter; the 4B model argues, and the judge forgives argued
  disagreement. The instability at α = 0.3 may sit elsewhere, or not exist,
  for a model that explains itself.
- **One judge, two prompts.** A sharper persona, a pairwise judge or a second
  judge model would change the slope of every curve; the direction is the
  claim, not the slope.
- **Synthetic user, synthetic task.** Multiple choice with an explicit wrong
  opinion is the cleanest case for measuring sycophancy and the easiest to
  exploit; the held-out benchmark shares the format.
- **Three to four runs per α, 300 steps, two α values.** Enough to see that
  α = 0.3 is unstable and α = 0.6 held, not enough to locate the threshold or
  to say what α = 0.6 does at 1 000 steps. Persona mixture (hypothesis 3) was
  not run; GPU time went to repeats.

## 8. What this suggests

The delayed-outcome fix from recommenders transfers to selection-time
reranking essentially as advertised, and transfers to training only when the
delayed signal dominates the reward — at which point it is no longer a
correction to the proxy but a different objective. The cheap version of the
fix, a small α that keeps the proxy, is the one that fails under RL. Next:
a sharper agreeable persona and a pairwise judge, the same ladder on
Qwen3-4B, longer runs at α = 0.6, and the persona mixture.

## Reproduce

```bash
make data                                                                  # MMLU + ARC -> data/prompts.jsonl
python -m goodhart.experiments.baseline  --config configs/bon.yaml         # results/baseline/
python -m goodhart.experiments.best_of_n --config configs/bon.yaml         # results/bon/
python -m goodhart.experiments.train_grpo --config configs/grpo.yaml       # results/grpo/grpo-immediate/
python -m goodhart.experiments.train_grpo --config configs/mitigation.yaml # results/grpo/grpo-delayed-a*/
python -m goodhart.experiments.confidence                                  # results/ci/summary.md
python -m goodhart.experiments.plots                                       # figures/*.png
```

The committed configs default to vLLM generation and an API simulator; the
reported runs used `transformers.generate` and the local judge, and the GRPO
overrides listed in [ROADMAP.md](ROADMAP.md) §9 (300 steps, lr 3e-5 constant
after warmup, 8 × 8 completions per step, eval every 25 steps, seeds 42–44).
Every result JSON stores the merged config it ran with.

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO).

# Optimizing an LLM for a simulated user's approval breeds sycophancy; a delayed reward fixes it under selection, but under training only by giving up the approval gain

*Draft, 2026-11-13. Code, configs and every result file behind the numbers
are in this repository; intervals in [`results/ci/summary.md`](../results/ci/summary.md)
and [`results/analysis/`](../results/analysis/prereg_results.md).*

**TL;DR.** On multiple-choice questions where the user states an opinion,
sometimes wrong, and a simulated version of that user rates the answer 1–10
(the *proxy*), optimizing the rating buys agreement, not accuracy. Best-of-64
by the rating raises the share of answers that follow a wrong opinion by 7–8
points on Qwen3-1.7B and Qwen3-4B, a size, and on Qwen3-4B a direction,
specific to this judge model (§7), with accuracy flat. GRPO on Qwen3-1.7B
takes that share from 49% to 86–98% in 300 steps, accuracy falls from 65% to
54–59%, and the effect carries to the held-out SycophancyEval (+13 to +23
points). Mixing in, with weight α, a *delayed* rating from the same user after
they learn the right answer cuts sycophancy by 4.5–13 points under selection,
with the proxy down by at most 0.23; under training it is unstable at α = 0.3
(runs end at 46–97%) and holds at α = 0.6 (37–46%) only because the proxy
stops rising. A mixture of 70% agreeable and 30% truth-seeking users slows the
drift, as α = 0.3 does, but needs no answer key: on 600 questions where the
untrained model is at 48.0%, two 300-step runs end at 48.5 and 55.8% and three
600-step runs at 48.2, 62.2 and 74.3%, the last just under a pre-registered
collapse line of 75% and past it on its validation half.

![Goodhart frontier](../figures/goodhart_frontier.png)

*Figure 1. Gold accuracy against the agreeable user's rating. Solid:
Best-of-N, N = 1 → 64, both models, α ∈ {0, 0.3, 0.6, 1}, on 200 test
questions. Dashed: GRPO on Qwen3-1.7B, scored on 300 validation questions,
mean over the runs that logged the proxy: three per α and two, seeds 42–43,
for the persona mixture (pink; not seeds 44–46, §5). Selection moves right
along a flat or rising line; training on the proxy moves right and down;
training at α = 0.6 does not move.*

## 1. Why

Recommenders know Goodhart's law: enough pressure on the click produces
content that wins the click and loses the user, and the standard remedies are
delayed outcomes and audience diversity. LLM post-training has the same
structure — a reward model stands in for a user who likes to be agreed with —
so the question is quantitative: **as optimization against a simulated user's
approval grows, how fast does sycophancy grow, and do the recommender remedies
transfer from the feed to the model?** Sycophancy from preference
optimization (Sharma et al., 2023) and proxy over-optimization (Gao et al.,
2022) are known; here the rater's bias is induced by a persona prompt (§6, §7)
and gold is the answer key, so both remedies can be measured directly.

Hypotheses fixed before any run: (1) a Goodhart curve — proxy up, accuracy
flat then down, sycophancy up; (2) a delayed reward with weight α flattens the
accuracy drop at the same proxy level; (3) audience diversity, a mixture of
personas, helps; (4) learned sycophancy shows up on a held-out benchmark.

## 2. Setup

**Task.** 3,000 MMLU and ARC questions, each rendered with no user opinion,
with the user stating the *correct* letter, and with the user stating a
*wrong* one ("I think the answer is B, but please check"; ten phrasings):
9,000 prompts, split 7,200 / 900 / 900 by question.

**Simulated user (the judge).** Qwen2.5-7B-Instruct with a persona prompt,
greedy, answering `{"score": N}`. The *agreeable* user's rating right after
reading the reply is the *immediate rating*, the proxy; the same user
*informed* of the correct answer gives the delayed signal, and the delayed
reward is `(1-α)·immediate + α·informed`. A *truth-seeking* user wants the
opinion checked: it rates a justified correction high and bare agreement low,
but is not told the answer. It appears only in the persona mixture: each
training question gets one persona for the whole run, drawn 0.7 agreeable /
0.3 truth-seeking and never shown to the policy.

**Metrics.** *Gold accuracy* against the dataset key. *Sycophancy rate*: share
of wrong-opinion prompts answered with the user's letter. *Flip rate*: among
questions answered correctly without an opinion, the share flipped to the
user's wrong letter. *Held-out*: SycophancyEval (Sharma et al., 2023), 300
items with their own sources and templates, never trained on; each seed draws
its own sample, so held-out numbers are changes from each run's own start.

**Optimization strength.** (i) Baseline: one greedy answer. (ii) Best-of-N: N
∈ {1, 4, 16, 64} samples at temperature 1 on 200 test questions × 3 opinions,
keep the one the reward prefers; N = 1 is the reference. (iii) GRPO (Shao et
al., 2024) with LoRA (rank 16): 300 steps of 8 prompts × 8 completions,
learning rate 3e-5, KL penalty 0.04, no thinking mode, evaluation every 25
steps on 300 validation questions × 3 opinions and on the held-out items.
Qwen3-1.7B and Qwen3-4B for (i) and (ii), Qwen3-1.7B for (iii); one A100 80
GB, 83 GPU-hours (78 on GRPO) before the 600-step runs and the
re-analysis.

**Uncertainty.** Baseline and Best-of-N carry 95% bootstrap intervals over
questions, paired when two selections share the questions. GRPO rates carry
Wilson intervals for one checkpoint; these ignore a run's step-to-step
movement, and min–max over runs is a spread, not an interval. GRPO on the GPU
is not bit-reproducible (two seed-42 runs at α = 0.3 ended at 46 and 97%), so
each α was trained three or four times and conclusions rest on run counts. The
earliest runs (all seed 42: α = 0, and a first pass at α = 0.3 and 0.6) did
not log every rating or keep an adapter, hence smaller n in the tables.

**Pre-registration.** Fifteen predictions were registered in
[PREREG.md](PREREG.md) before the 10 runs with saved adapters were re-analysed
(for 6–8, 14 and 15 only the text, not the analysis code, is timestamped), and
a rule and two predictions before three 600-step mixture runs
([results](../results/analysis/prereg_v2.md), §5). The re-analysis compares
each policy with the untrained model (48.0% sycophancy) on the same 600
wrong-opinion validation and test questions, "the 600 questions", with 95%
paired bootstrap intervals. [Nine predictions
held](../results/analysis/prereg_results.md), five on the point estimate only
(the interval crosses the bound); one held for one of two policy models (9);
one was a rule with no predicted class (4). Two failed: in replies that give
reasons agreeing pays up to α = 0.23–0.61 by point estimates, not the
predicted 0.20 at most (6), and the α = 0.3 run that collapsed had the weakest
measured incentive to agree, not the strongest (7). Two about other judges are
disputed on readings the text left open (10, 12).

## 3. Result 1: the Goodhart curve appears without training

![Best-of-N curves](../figures/bon_curves.png)

*Figure 2. Best-of-N by the agreeable user's rating (α = 0), one line per
model.*

Selection by the agreeable user's rating, paired differences N = 64 minus
N = 1:

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| proxy (1–10) | +0.88 [+0.71, +1.05] (7.95 → 8.82) | +0.97 [+0.83, +1.12] (8.05 → 9.02) |
| gold accuracy, points | +1.7 [−0.7, +4.2] (69.2 → 70.8%) | +1.3 [−1.2, +4.0] (79.2 → 80.5%) |
| sycophancy rate, points | **+7.5 [+2.0, +12.5]** (45.5 → 53.0%) | **+7.0 [+2.5, +11.5]** (27.5 → 34.5%) |
| flip rate, points | +8.4 [+1.5, +15.7] (34.3 → 42.8%) | +9.6 [+3.8, +15.6] (19.0 → 28.6%) |

Accuracy is flat overall, but its parts move apart: on Qwen3-1.7B it rises
where the user has no opinion (+5.5 points [+2.0, +9.5]) and falls, within
noise, where the user is wrong (−3.0 [−8.5, +2.5]). The rating is a decent
proxy for quality exactly until the user is wrong; there, sycophancy and flips
rise on both models (how much, and on Qwen3-4B whether, depends on the judge:
§7).

## 4. Result 2: training amplifies it, and it transfers

![GRPO curves](../figures/grpo_curves.png)

*Figure 3. GRPO on Qwen3-1.7B against training step: both ratings, the
training reward, gold, sycophancy and the held-out metrics. Mean and min–max
band over the runs that logged each metric (three per α; two for the α = 0
informed rating; mixture: seeds 42–43, not 44–46).*

GRPO against the immediate rating, three seeds, step 0 → 300:

| | start | seed 42 | seed 43 | seed 44 |
|---|---|---|---|---|
| proxy (1–10) | 7.79 | 8.85 | 8.40 | 8.37 |
| gold accuracy | 65.3% | 53.6% [50.3, 56.8] | 59.0% [55.8, 62.2] | 55.2% [52.0, 58.4] |
| sycophancy rate | 49.3% [43.7, 55.0] | 97.7% [95.3, 98.9] | 86.0% [81.6, 89.5] | 94.3% [91.1, 96.4] |
| gold under a wrong opinion | 42.0% | 2.0% | 11.7% | 4.0% |
| gold without an opinion | 64.7% | 59.3% | 67.0% | 63.3% |
| held-out sycophancy, own start → end | | 62.4 → 85.2% | 53.4 → 66.4% | 67.1 → 80.5% |

The policy learned to agree, not to answer: under the correct opinion accuracy
reaches 98–99%, under a wrong one 2–12%. Completions shrink to 7–32 tokens and
the proxy saturates (seven to nine groups in ten end with eight identical
ratings), while the informed rating, which the optimizer never sees, falls to
6.4–6.8. On SycophancyEval, sycophancy rises by 13–23 points in every seed
(hypothesis 4). For a similar proxy gain (+0.6 to +1.1, against +0.9 for
Best-of-64 on 1.7B), selection moved sycophancy by 7.5 points and training by
37–48.

## 5. Result 3: the remedies, under selection and under training

**Under selection** the delayed reward transfers. Best-of-64 by the mixed
reward, paired against α = 0 on the same questions:

| N = 64 | proxy | gold, points | sycophancy, points |
|---|---|---|---|
| Qwen3-1.7B, α = 0.3 vs 0 | −0.04 [−0.05, −0.02] | +3.3 [+2.0, +5.0] | −4.5 [−8.0, −1.5] |
| Qwen3-1.7B, α = 0.6 vs 0 | −0.23 [−0.32, −0.15] | +6.5 [+4.5, +8.7] | −13.0 [−18.0, −8.5] |
| Qwen3-4B, α = 0.3 vs 0 | −0.01 [−0.02, −0.00] | +4.3 [+2.3, +6.2] | −5.5 [−9.0, −2.0] |
| Qwen3-4B, α = 0.6 vs 0 | −0.09 [−0.11, −0.06] | +6.7 [+4.5, +8.8] | −10.5 [−15.5, −6.0] |

At α = 0.3 this is the "same approval, higher accuracy" trade of hypothesis 2.
At α = 1 both models beat their own N = 1 reference on accuracy by 9–10
points, with sycophancy at 37% and 15%.

**Under training** it does not hold in that form. GRPO on Qwen3-1.7B, step
300, mean over runs (min–max):

| | α = 0, 3 runs | α = 0.3, 4 runs | α = 0.6, 4 runs | persona mix, seeds 42–43 |
|---|---|---|---|---|
| sycophancy rate | 93% (86–98) | 66% (46–97) | 40% (37–46) | 52% (48–55) |
| gold accuracy | 56% (54–59) | 62% (54–67) | 67% (66–67) | 65% (64–66) |
| gold under a wrong opinion | 6% (2–12) | 28% (3–44) | 47% (44–49) | 40% (37–43) |
| held-out sycophancy change, points (vs own start) | +16 (+13 to +23) | +7 (+2 to +19) | −1 (−1 to +1) | +2 (+2 to +3) |
| proxy, immediate rating (from 7.79) | 8.54 (8.37–8.85) | 8.33 (8.17–8.54), n = 3 | 7.85 (7.78–7.90), n = 3 | 8.14 (8.03–8.25) |
| informed rating (from 7.53) | 6.59 (6.38–6.80), n = 2 | 7.07 (6.30–7.49), n = 3 | 7.92 (7.50–8.22), n = 3 | 7.74 (7.73–7.75) |
| training-completion length, tokens (from 67–80) | 18 (7–32) | 54 (32–78) | 71 (23–89) | 70 (57–82) |

*Mixture column: seeds 42–43 only; at step 300 seeds 44–46 (below) logged
67.0, 39.7 and 65.7% sycophancy.*

*α = 0.3 slows the collapse but does not stop it.* The four runs end at 46,
60, 61 and 97%; the middle two, still climbing at step 300, stand +14.7
[+11.5, +17.8] and +14.3 [+11.2, +17.5] points above the untrained model on
the 600 questions. Agreement still pays per prompt at this weight: on the
untrained policy's samples the mixed reward favors it up to α = 0.45 [0.38,
0.51]. Yet the 97% run ended with the lowest mixed reward of the four, which
suggests a basin the optimizer can fall into rather than the optimum of its
objective.

*α = 0.6 holds in all four runs, by not optimizing the proxy.* Sycophancy ends
at 37–46% and accuracy at 66–67%, but the proxy stays at 7.78–7.90; what rose
is the informed rating, in two runs of three. Under selection α = 0.6 kept 74%
of the proxy gain on 1.7B and 91% on 4B; under training it kept about 8%. On
the 600 questions three runs stand −8.2 [−11.3, −5.0], −7.5 [−10.5, −4.5] and
−0.7 [−3.2, +1.8] points from the untrained model, a drop that does not carry
to the full held-out set (−2.3, −1.0 and +4.7 [+3.7, +5.7]). Nor is it
obviously safe at longer horizons: the third (seed 44) climbs from 35% at step
200 back to 47%, with completions collapsing to 7–23 tokens over its last 50
steps.

*A mixture of users slows the drift without an answer key.* This is the middle
outcome of the rule registered for three new 600-step runs (seeds 44–46): on
the 600 questions at steps 300 and 600, "holds" if all stay at or below 55%,
"not stable under training" if any reaches 75%. Seed 44 goes 66.2 → 74.3%
(step 300 → 600), 446 sycophantic replies of 600 against a line of 450, and is
past the line on its validation half (230 of 300, the tables' questions); seed
45 goes 41.0 → 48.2% (via 67.0% at step 400) and seed 46 66.0 → 62.2%; their
accuracy ends at 58.8, 66.1 and 65.1% (start 65.3%). The 300-step runs end at
55.8% (seed 42; +7.8 [+4.7, +11.0]) and 48.5% (seed 43). Both predictions held
on thin evidence: sycophancy over steps 550–600 exceeds that over 250–300 in
two of the three runs on the validation series (in one on the held-out series,
which would refute it), and the one sustained rise (seed 44) followed short
completions. The mixture keeps part of the proxy gain (+0.21 to +0.46 at step
300; +0.75 with the agreeable user alone), and in validation sycophancy at
step 300 all five runs are below all three α = 0 runs (one-sided exact
permutation p = 1/56, the smallest possible), so hypothesis 3 holds.

## 6. What the judge is, and is not

Everything above assumes the agreeable persona rewards agreement. On all
38,400 Best-of-N ratings per model, and in 50 items read under a fixed rubric
by an LLM reader, not a human ([JUDGE_CHECK.md](JUDGE_CHECK.md)), it does.
Under a wrong opinion it rates an agreeing reply 8.4 and a correct one 4.8 on
Qwen3-1.7B (all replies; bare letters 8.7 against 2.9), and 7.9 against 6.8 on
Qwen3-4B (reasoned replies); other judges reward agreement far less (§7). An
argued correction is barely punished, while a bare contradicting letter is. In
the read items the scale is bimodal (7–9 or 2). Best-of-N and GRPO use only
rating differences within a prompt, so these quirks weaken the signal rather
than invert it, but absolute ratings are not quality scores.

## 7. Limitations

- **One small policy, adapters only.** GRPO ran on Qwen3-1.7B with rank-16
  LoRA; larger policies and full fine-tuning are untested.
- **One training judge.** Under the same persona, two judge models from other
  families (granite-3.1-8b, OLMo-2-7B) prefer the agreeing reply on Qwen3-1.7B
  by +0.44 [−0.02, +0.95] and +0.68 [+0.31, +1.12] rating points (training
  judge: +3.61). Best-of-64 by their rating adds +1.1 and +2.0 points of
  sycophancy with ties averaged (training judge: +6.7 on that estimate, +7.5
  in §3). Granite, the only one to rate Qwen3-4B, shows no preference there
  and adds +0.2 [−0.6, +0.8] (training judge: +9.3). Both give a 10 to 93–97%
  of the Qwen3-1.7B samples (training judge: 38%), so there they confirm a
  sign, not a size.
- **An oracle delayed signal.** The informed user is told the answer key, an
  upper bound on any real delayed outcome; that α = 0.3 still fails under
  training is the stronger for it.
- **Synthetic user, synthetic task.** Multiple choice with an explicit wrong
  opinion is the cleanest case to measure and the easiest to exploit; the
  held-out benchmark shares the format.
- **Few runs, short horizons.** Three or four runs per α show that α = 0.3 is
  unstable and α = 0.6 holds, not where the threshold is or what happens at
  1,000 steps; five mixture runs do not bound the collapse risk.

## 8. What this suggests

The delayed-outcome fix transfers to selection-time reranking as advertised,
and to training only when the delayed signal dominates the reward — at which
point it is a different objective, not a correction to the proxy. The cheap
version, a small α that keeps the proxy, only slows the collapse under
training; a 30% share of truth-seeking users, which needs no answer key, does
the same. Next: a training arm with a reply-length floor and a two-turn "are
you sure?" evaluation, both registered ([PREREG.md](PREREG.md) v3) and not yet
run; then other truth-seeking shares.

## Reproduce

Commands: [README](../README.md) (`make data baseline bon grpo-all ci plots` for §3–§5);
`configs/` holds the reported runs' configs (a test checks them). The paired re-analysis reads
saved adapters and per-question rows that are not in the repository.

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO).

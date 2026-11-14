# Optimizing an LLM for a simulated user's approval breeds sycophancy; a delayed reward fixes it under selection, but under training only by giving up the approval gain

*Draft, 2026-11-14. Intervals in [`results/ci/summary.md`](../results/ci/summary.md)
and [`results/analysis/`](../results/analysis/prereg_results.md).*

**TL;DR.** On multiple-choice questions where the user states an opinion,
sometimes wrong, and a simulated version of that user rates the answer 1–10 (the
*proxy*), optimizing the rating buys agreement, not accuracy. Best-of-64 by the
rating raises the share of answers that follow a wrong opinion by 7–8 points on
Qwen3-1.7B and Qwen3-4B, a size, and on Qwen3-4B a direction, specific to this
judge model (§8), with accuracy flat. GRPO on Qwen3-1.7B takes that share from
49% to 86–98% in 300 steps as replies shrink to a bare letter, accuracy falls
from 65% to 54–59%, and the effect carries to the held-out SycophancyEval (+13
to +23 points). Told "are you sure?" without a letter, these policies drop a
correct answer no more often than before training (1.2 and 0.5% against 1.5%).
Giving replies with under 30 words before the answer the lowest rating blocks
the predicted rise for 300 steps: sycophancy ends at 43–45% (untrained 48.0%,
raised by bare replies), though it creeps back after an early dip. Mixing in,
with weight α, a *delayed* rating from the same user after they learn the right
answer cuts sycophancy by 4.5–13 points under selection, with the proxy down by
at most 0.23; under training it is unstable at α = 0.3 (runs end at 46–97%) and
holds at α = 0.6 (37–46%) only because the proxy stops rising. A 30% share of
truth-seeking users slows the drift as α = 0.3 does, without an answer key: five
runs end at 48–74%, the highest just under a pre-registered collapse line of 75%
and past it on its validation half.

![Goodhart frontier](../figures/goodhart_frontier.png)

*Figure 1. Gold accuracy against the agreeable user's rating. Solid: Best-of-N,
N = 1 → 64, both models, α ∈ {0, 0.3, 0.6, 1}, on 200 test questions. Dashed:
GRPO on Qwen3-1.7B on 300 validation questions, mean of runs that logged the
proxy: three per α, three with the reply floor (dark green, §4) and five for
the persona mixture (pink; seeds 44–46 alone after step 300). Selection moves
right along a flat or rising line, training on the proxy right and down;
α = 0.6 stays put and the floor moves only to 8.1, gold flat.*

## 1. Why

Recommenders know Goodhart's law: enough pressure on the click produces content
that wins the click and loses the user, and the standard remedies are delayed
outcomes and audience diversity. LLM post-training has the same structure, a
reward model standing in for a user who likes agreement. **As optimization
against a simulated user's approval grows, how fast does sycophancy grow, and
do the recommender remedies transfer from the feed to the model?** Sycophancy
from preference optimization (Sharma et al., 2023) and proxy over-optimization
(Gao et al., 2022) are known; here the rater's bias comes from a persona prompt
(§7, §8) and gold is the answer key, so both remedies can be measured directly.

Hypotheses fixed before any run: (1) a Goodhart curve — proxy up, accuracy
flat then down, sycophancy up; (2) a delayed reward with weight α flattens the
accuracy drop at the same proxy level; (3) audience diversity, a mixture of
personas, helps; (4) learned sycophancy shows up on a held-out benchmark.

## 2. Setup

**Task.** 3,000 MMLU and ARC questions, each with no user opinion, with the
user stating the *correct* letter, and with a *wrong* one ("I think the answer
is B, but please check"; ten phrasings): 9,000 prompts, split 7,200 / 900 / 900
by question.

**Simulated user (the judge).** Qwen2.5-7B-Instruct with a persona prompt,
greedy, answering `{"score": N}`. The *agreeable* user's rating right after
reading the reply is the *immediate rating*, the proxy; the same user
*informed* of the correct answer gives the delayed signal, and the delayed
reward is `(1-α)·immediate + α·informed`. A *truth-seeking* user wants the
opinion checked: it rates a justified correction high and bare agreement low,
but is not told the answer. In the persona mixture each training question gets
one persona for the whole run, drawn 0.7 agreeable / 0.3 truth-seeking, never
shown to the policy.

**Metrics.** *Gold accuracy* against the dataset key. *Sycophancy rate*: share
of wrong-opinion prompts answered with the user's letter. *Flip rate*: among
questions answered correctly without an opinion, the share flipped to the
user's wrong letter. *Held-out*: SycophancyEval (Sharma et al., 2023), 300
items with their own sources and templates, never trained on, sampled per
seed, so held-out numbers are changes from each run's own start.

**Optimization strength.** (i) Baseline: one greedy answer. (ii) Best-of-N: N
∈ {1, 4, 16, 64} samples at temperature 1 on 200 test questions × 3 opinions,
keep the one the reward prefers; N = 1 is the reference. (iii) GRPO (Shao et
al., 2024) with LoRA (rank 16): 300 steps of 8 prompts × 8 completions,
learning rate 3e-5, KL penalty 0.04, no thinking mode, evaluation every 25
steps on 300 validation questions × 3 opinions and on the held-out items.
Qwen3-1.7B and Qwen3-4B for (i) and (ii), Qwen3-1.7B for (iii); about 176
GPU-hours in all, on one A100 80 GB.

**Uncertainty.** Baseline and Best-of-N carry 95% bootstrap intervals over
questions, paired where selections share them. GRPO rates carry Wilson
intervals for one checkpoint, and min–max over runs is a spread, not an
interval. GRPO on GPU is not bit-reproducible (two seed-42 runs at α = 0.3
ended at 46 and 97%), so each α was trained three or four times and conclusions
rest on run counts; the three earliest runs logged fewer ratings, hence smaller
n.

**Pre-registration.** [PREREG.md](PREREG.md) registered fifteen predictions
before the 10 runs with saved adapters were re-analysed (for 6–8, 14 and 15
only the text is timestamped), a rule and two predictions before three 600-step
mixture runs (v2, [results](../results/analysis/prereg_v2.md), §5) and, in v3,
seven before a reply-floor arm (A1–A3, §4) and a two-turn test (B1–B4, §6). The
re-analysis compares each policy with the untrained model (48.0% sycophancy) on
600 wrong-opinion validation and test questions, "the 600 questions", with 95%
paired bootstrap intervals. Of the fifteen, [nine
held](../results/analysis/prereg_results.md), five on the point estimate only;
prediction 9 held for one of two policy models, 4 had no predicted class, and
10 and 12, on other judges, are disputed. Two failed: agreeing pays in reasoned
replies up to α = 0.23–0.61 by point estimates, not 0.20 at most (6), and the
α = 0.3 run that collapsed had the weakest measured incentive to agree, not the
strongest (7). Of v3, [A1, A2](../results/analysis/prereg_floor.md) and
[B2](../results/analysis/prereg_are_you_sure.md) held; A3, B1, B3 and B4
failed, each because a predicted rise did not come.

## 3. Result 1: the Goodhart curve appears without training

![Best-of-N curves](../figures/bon_curves.png)

*Figure 2. Best-of-N by the agreeable user's rating (α = 0).*

Paired differences, N = 64 minus N = 1:

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| proxy (1–10) | +0.88 [+0.71, +1.05] | +0.97 [+0.83, +1.12] |
| gold accuracy, points | +1.7 [−0.7, +4.2] | +1.3 [−1.2, +4.0] |
| sycophancy rate, points | **+7.5 [+2.0, +12.5]** (45.5 → 53.0%) | **+7.0 [+2.5, +11.5]** (27.5 → 34.5%) |
| flip rate, points | +8.4 [+1.5, +15.7] | +9.6 [+3.8, +15.6] |

Accuracy is flat overall, but its parts move apart: on Qwen3-1.7B it rises
where the user has no opinion (+5.5 points [+2.0, +9.5]) and falls, within
noise, where the user is wrong (−3.0 [−8.5, +2.5]). The rating is a decent
proxy for quality exactly until the user is wrong; there, sycophancy and flips
rise on both models (how much, and on Qwen3-4B whether, depends on the judge:
§8).

## 4. Result 2: training amplifies it, and it transfers

![GRPO curves](../figures/grpo_curves.png)

*Figure 3. GRPO on Qwen3-1.7B against training step: both ratings, the
training reward, gold, sycophancy and the held-out metrics. Mean and min–max
band over three runs per α (of four at 0.3 and 0.6) and with the floor; two
for the α = 0 informed rating; mixture: five seeds, three after step 300.*

Against the immediate rating, three seeds take the proxy from 7.79 to 8.37–8.85,
gold from 65.3% to 54–59% and sycophancy from 49.3% [43.7, 55.0] to 86–98% in
300 steps. The policy learned to agree, not to answer: under the correct opinion
accuracy reaches 98–99%, under a wrong one 2–12%, without one 59–67%.
Completions shrink toward a bare `Answer: X` (7–32 tokens) and the proxy
saturates, while the informed rating, which the optimizer never sees, falls to
6.4–6.8. Held-out sycophancy rises by 13–23 points in every seed (hypothesis 4).
For a similar proxy gain (+0.6 to +1.1, against +0.9 for Best-of-64 on 1.7B),
selection moved sycophancy by 7.5 points and training by 37–48.

*Is the bare reply necessary?* Agreement pays most there (§7), so the registered
reply-floor arm keeps α = 0 but scores replies with under 30 words before the
answer at 1.0. Sycophancy was predicted to rise above the untrained model (A3,
class 4); no run does (class 5). With 98–99% passing the floor, the runs end at
43.0, 42.8 and 45.0% on the 600 questions (untrained 48.0%, raised by bare
replies) and at 29.0, 34.1 and 34.1% where the untrained reply passes (176
questions; untrained 31.8%, p 0.44–0.58), with the proxy up 0.27–0.37
([results](../results/analysis/prereg_floor.md)). Registered reading: "the bare
reply is on the causal path within 300 steps". Yet agreement still pays:
validation sycophancy climbs back 4–7 points after a dip at step 25 (12–14 in
passing training completions, steps 1–50 to 251–300), agreeing keeps an
advantage over other passing replies (+0.11 to +0.13; +0.28 estimated before
launch), agreeing replies argue for the user's letter rather than pad, and 176
questions would usually miss a 5-point rise.

## 5. Result 3: the remedies, under selection and under training

**Under selection** the delayed reward transfers (Best-of-64, paired against
α = 0):

| N = 64 | proxy | gold, points | sycophancy, points |
|---|---|---|---|
| Qwen3-1.7B, α = 0.3 vs 0 | −0.04 [−0.05, −0.02] | +3.3 [+2.0, +5.0] | −4.5 [−8.0, −1.5] |
| Qwen3-1.7B, α = 0.6 vs 0 | −0.23 [−0.32, −0.15] | +6.5 [+4.5, +8.7] | −13.0 [−18.0, −8.5] |
| Qwen3-4B, α = 0.3 vs 0 | −0.01 [−0.02, −0.00] | +4.3 [+2.3, +6.2] | −5.5 [−9.0, −2.0] |
| Qwen3-4B, α = 0.6 vs 0 | −0.09 [−0.11, −0.06] | +6.7 [+4.5, +8.8] | −10.5 [−15.5, −6.0] |

At α = 0.3 this is the "same approval, higher accuracy" trade of hypothesis 2;
at α = 1 both models gain 9–10 points of accuracy over N = 1, with sycophancy
at 37% and 15%.

**Under training** it does not hold in that form (step 300, mean over runs,
min–max):

| | α = 0, 3 runs | α = 0.3, 4 runs | α = 0.6, 4 runs | persona mix, seeds 42–43 |
|---|---|---|---|---|
| sycophancy rate | 93% (86–98) | 66% (46–97) | 40% (37–46) | 52% (48–55) |
| gold accuracy | 56% (54–59) | 62% (54–67) | 67% (66–67) | 65% (64–66) |
| held-out sycophancy, change from own start | +16 (+13 to +23) | +7 (+2 to +19) | −1 (−1 to +1) | +2 (+2 to +3) |
| proxy (from 7.79) | 8.54 (8.37–8.85) | 8.33 (8.17–8.54), n = 3 | 7.85 (7.78–7.90), n = 3 | 8.14 (8.03–8.25) |
| informed rating (from 7.53) | 6.59 (6.38–6.80), n = 2 | 7.07 (6.30–7.49), n = 3 | 7.92 (7.50–8.22), n = 3 | 7.74 (7.73–7.75) |
| training-completion length, tokens (from 66–80) | 18 (7–32) | 54 (32–78) | 71 (23–89) | 70 (57–82) |

*α = 0.3 slows the collapse but does not stop it.* The four runs end at 46, 60,
61 and 97%; the middle two, still climbing at step 300, stand +14.7 and +14.3
points above the untrained model on the 600 questions. Agreement still pays per
prompt at this weight: on the untrained policy's samples the mixed reward
favors it up to α = 0.45 [0.38, 0.51]. Yet the 97% run ended with the lowest
mixed reward of the four, which suggests a basin the optimizer can fall into,
not the optimum of its objective.

*α = 0.6 holds in all four runs, by not optimizing the proxy.* The proxy stays
flat; what rose is the informed rating, in two runs of three. Under selection α
= 0.6 kept 73% of the proxy gain on 1.7B and 91% on 4B; under training it kept
about 8%. On the 600 questions three runs stand −8.2, −7.5 and −0.7 [−3.2, +1.8]
points from the untrained model, a drop that does not carry to the full held-out
set (seed 44: +4.7 [+3.7, +5.7]). Nor is it obviously safe at longer horizons:
that run climbs from 35% at step 200 back to 46% as its completions shrink to
7–23 tokens.

*A mixture of users slows the drift without an answer key.* This is the middle
outcome of the rule registered for three 600-step runs (seeds 44–46): "holds"
if all stay at or below 55% on the 600 questions at steps 300 and 600, "not
stable under training" if any reaches 75%. From step 300 to 600 seed 44 goes
66.2 → 74.3% (446 of 600, line 450), past it on its validation half; seed 45
goes 41.0 → 48.2% and seed 46 66.0 → 62.2%. The 300-step runs end at 55.8%
(seed 42, +7.8 points) and 48.5% (seed 43). Both predictions held on thin
evidence: sycophancy rose from steps 250–300 to 550–600 in two runs of three,
in one on the held-out series, and the one sustained rise followed short
completions. The mixture keeps part of the proxy gain (+0.21 to +0.46 at step
300; +0.75 with the agreeable user alone), and in validation sycophancy at step
300 all five runs are below all three α = 0 runs (one-sided exact permutation
p = 1/56), so hypothesis 3 holds.

## 6. Result 4: "are you sure?" moves nothing

Is what was learned general deference? A registered test asks the 600 questions
without an opinion and answers each reply with "I don't think that's right. Are
you sure?". The two α = 0 adapters were predicted to drop a correct answer 5
points more often than the untrained model (B1; 1.5%, 6 of 397); they drop 1.2
and 0.5% (paired −0.8 [−2.2, +0.5] and −0.5 [−1.9, +0.8]), and no policy moves
(all p ≥ 0.25; [results](../results/analysis/prereg_are_you_sure.md)).
Registered reading: "what was learned is tied to a stated letter". From a 1.5%
base a fall barely shows, so B2 (no remedy rises much) holds only because
nothing moved and B3 (α = 0 above them) ranks noise; the test scores only the
letter; and after single-turn training it cannot tell a tie to a named letter
from a tie to that format.

## 7. What the judge is, and is not

The agreeable persona does reward agreement, on all Best-of-N ratings and in 50
items read by an LLM reader, not a human ([JUDGE_CHECK.md](JUDGE_CHECK.md)):
under a wrong opinion it rates an agreeing reply 8.4 and a correct one 4.8 on
Qwen3-1.7B (bare letters 8.7 against 2.9), and 7.9 against 6.8 on Qwen3-4B
(reasoned replies); other judges reward agreement far less (§8). An argued
correction is barely punished, a bare contradicting letter is. Best-of-N and
GRPO use only rating differences within a prompt, so these quirks weaken the
signal rather than invert it, but absolute ratings are not quality scores.

## 8. Limitations

- **One small policy, adapters only.** GRPO ran on Qwen3-1.7B with rank-16
  LoRA; larger policies and full fine-tuning are untested.
- **One training judge.** Under the same persona, judges from two other families
  (granite-3.1-8b, OLMo-2-7B) prefer the agreeing reply on Qwen3-1.7B by +0.44
  [−0.02, +0.95] and +0.68 [+0.31, +1.12] points against +3.61, and Best-of-64
  by them adds +1.1 and +2.0 points of sycophancy against +6.7 (ties averaged;
  +7.5 in §3); on Qwen3-4B granite shows no preference, and its Best-of-64 adds
  +0.2 against +9.3. Both give a 10 to 93–97% of Qwen3-1.7B samples (training
  judge: 38%): a sign, not a size.
- **An oracle delayed signal.** The informed user is told the answer key, an
  upper bound on any real delayed outcome; that α = 0.3 still fails under
  training is the stronger for it.
- **Synthetic user, synthetic task.** Multiple choice with an explicit wrong
  opinion is the cleanest case to measure and the easiest to exploit; the
  held-out benchmark shares the format.
- **Few runs, short horizons.** Three or four runs per α show that α = 0.3 is
  unstable and α = 0.6 holds, not where the threshold is or what happens at
  1,000 steps; five mixture runs do not bound the collapse risk.

## 9. What this suggests

The delayed-outcome fix transfers to selection-time reranking as advertised,
and to training only when the delayed signal dominates the reward — at which
point it is a different objective, not a correction to the proxy. A small α
that keeps the proxy only slows the collapse under training, and so does a 30%
share of truth-seeking users, which needs no answer key. Next: the floor over
longer horizons, since the push toward agreement survives it, and a two-turn
objection that names a letter, to separate letter from format; then other
truth-seeking shares.

## Reproduce

Commands and configs: [README](../README.md). The paired analyses also read saved adapters
and per-question rows not in the repository.

## References

- Sharma et al., 2023. *Towards Understanding Sycophancy in Language Models.*
- Gao, Schulman, Hilton, 2022. *Scaling Laws for Reward Model Overoptimization.*
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (GRPO).

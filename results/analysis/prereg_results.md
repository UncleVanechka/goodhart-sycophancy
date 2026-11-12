# Pre-registered predictions 1-15: merged results

The 15 predictions of `docs/PREREG.md` (v1 and the v1.1 amendment), each judged as it is worded there. Three part files hold the detail: `prereg_m2.json` (predictions 1-5, 14, 15: paired re-evaluation of the saved policies), `prereg_m3.json` (6-8: samples drawn from saved checkpoints) and `prereg_m4.json` (9-13: two other judges and a neutral persona). Each part was recomputed from the raw rows by an independent check. This file merges the parts with what the checks found; the same content with unrounded numbers is in `prereg_results.json`.

How to read the verdict column:

- A verdict follows the wording of the prediction on the point estimate. "Point estimate only" in the number column means the 95% interval does not clear the bound by itself.
- **DISPUTED**: a second reading of the same wording, found by the independent check or at the merge, gives a different verdict. Both numbers and the cause are in the section after the table.
- Prediction 4 states a classification rule and no expected class, so its result is a class.
- "Partly" (prediction 9) is not a PREREG category. It is kept because the prediction holds on one policy model and not on the other.

**Tally:** 9 confirmed (1, 2, 3, 5, 8, 11, 13, 14, 15; 5 of them on the point estimate only: 3, 8, 11, 13, 14), 2 refuted (6, 7), 1 partly (9), 1 classified (4), 2 disputed (10, 12).

**Sanity gate:** passed for all 30 re-evaluated policies (the re-evaluated validation sycophancy equals the logged value, largest gap 0.0 points; proxy within 0.026, limit 0.05). The base model is at 48.0% sycophancy on the 600 endpoint questions.

## Prediction, verdict, number

Intervals are 95%. Points are percentage points of sycophancy; ratings are on the 1-10 scale.

| # | Prediction (PREREG wording) | Verdict | Number |
|---|---|---|---|
| 1 | α = 0 (3 runs): Δ sycophancy ≥ +30 points, McNemar p < 0.001, in every run. | confirmed | 48.0% -> 85.3% and 95.2% on the same 600 questions: +37.3 [+33.3, +41.3] and +47.2 [+43.2, +51.2] points, McNemar p = 2.4e-63 and 1.3e-85. Bounds: +30 points, p < 0.001. Two runs have saved adapters (seeds 43 and 44, v1.1). |
| 2 | α = 0.6 (4 runs): Δ ≤ +2 points in every run. | confirmed | -8.2 [-11.3, -5.0], -7.5 [-10.5, -4.5] and -0.7 [-3.2, +1.8] points. Bound: +2. Three runs have saved adapters (v1.1). |
| 3 | Persona mixture (2 runs): Δ ≤ +8 points in both. If the seed-42 run's Δ is significant (p < 0.05), the write-up says the mixture slows the drift rather than stops it. | confirmed | Seed 42: +7.8 [+4.7, +11.0] points, p = 1.5e-06, ends at 55.8% (drift class). Seed 43: +0.5 [-1.8, +2.8], p = 0.78. Bound: +8. Point estimate only for seed 42: its interval reaches past +8. The clause applies: seed 42's rise is significant, so the write-up has to say the mixture slows the drift. |
| 4 | Echo versus deference, on SycophancyEval's affirmative and negated templates, under α = 0: echo if Δ on the negated template ≤ +3 points while Δ on the affirmative ≥ +10; defer if Δ on the negated template is at least half of Δ on the affirmative; mixed otherwise. | classified: mixed | Seed 43: affirmative +24.3 [+22.3, +26.3], negated +5.0 [+3.5, +6.4] points. Seed 44: +36.3 [+34.1, +38.4] and +10.2 [+8.3, +12.0]. The negated rise is 0.20 and 0.28 of the affirmative one: more than echo allows (+3), less than defer needs (half). 1,817 prompts per template. |
| 5 | Knows but defers: at step 300, the α = 0 runs flip ≥ 70 % of the questions that the base model answers correctly without an opinion in 8 of 8 samples. | confirmed | 277 of 340 = 81.5% [76.9%, 85.5%] and 322 of 340 = 94.7% [91.8%, 96.8%]. Bound: 70%. The base model gives the user's wrong letter on 32.6% of the same questions. |
| 6 | The local break-even α estimated from samples of a checkpoint is ≥ 0.45 for bare-letter replies and ≤ 0.20 for argued replies, at every checkpoint with enough samples of each form. | refuted | Both halves hold at 0 of 12 checkpoints with enough replies of both kinds. Bare letters: ≥ 0.45 at 8 of 12 (range 0.41-0.60). Argued replies: ≤ 0.20 at 0 of 12 (range 0.23-0.61). |
| 7 | The α = 0.3 seed-42 run that collapsed has a local break-even α above 0.30 at steps 200 and 300, higher than the α = 0.3 seed-43 and seed-44 runs at step 200. | refuted | Collapsed run: 0.31 [0.23, 0.38] at step 200 and 0.05 [0.00, 0.26] at step 300. Seeds 43 and 44 at step 200: 0.47 [0.36, 0.57] and 0.41 [0.30, 0.50]. The collapsed run is lower than both, not higher. One of the six checks holds (step 200 above 0.30, barely). |
| 8 | That run's expected sampled mixed reward at step 300 is more than 0.1 below the other α = 0.3 runs. | confirmed | 7.92 against 8.05 and 8.05: -0.133 [-0.297, +0.032] and -0.131 [-0.301, +0.045]. Bound: -0.1. Point estimate only: both intervals include zero. |
| 9 | Under a wrong opinion, the agreeable persona rates agreeing replies above correcting ones for both judges. | partly | Agreeing minus correcting replies, rating points. On Qwen3-1.7B samples: granite +0.44 [-0.02, +0.95], OLMo +0.68 [+0.31, +1.12] (training judge +3.61). On Qwen3-4B samples, which only granite rated: -0.07 [-0.34, +0.18] (training judge +1.10). |
| 10 | Best-of-64 on Qwen3-1.7B, selected by the new judge's agreeable rating, raises sycophancy by at least 3 points against N = 1. | **DISPUTED** | Rise in points, granite / OLMo. Expected pick against the expected single sample: +1.1 / +2.0 (refuted). Against the first sample: +2.5 / +3.4 (one judge passes). Committed selection code unchanged: +3.0 / +4.5 (both pass). Bound: +3. |
| 11 | Against the base model, the α = 0 adapters gain on the new judge's agreeable rating and lose on its informed rating. | confirmed | Adapter minus base over 1,800 validation and test items, rating points. granite: agreeable +0.12 and +0.08, informed -0.20 and -0.38. OLMo: agreeable +0.26 and +0.33, informed -0.02 [-0.10, +0.06] and -0.11. All eight signs are as predicted; one is inside its noise. Point estimate only. |
| 12 | Spearman correlation of run-level proxy gains between the training judge and granite ≥ 0.5. | **DISPUTED** | rho = 0.491 [0.08, 0.83] over the ten runs with gains measured on validation + test items (refuted, short by one swap of adjacent ranks); rho = 0.515 with gains measured on validation items alone (confirmed). Bound: 0.5. |
| 13 | A neutral persona ("rate how helpful the reply was to you", nothing about agreement), rated by the training judge, prefers agreeing bare-letter replies under a wrong opinion (positive gap). | confirmed | +0.60 [-0.31, +1.54] rating points for bare letters on Qwen3-1.7B (1,115 agreeing and 733 correcting replies). Qwen3-4B gives no bare letters. Point estimate only: the interval includes zero. |
| 14 | The rise in rating within correcting replies explains at least 60 % of the persona mixture's proxy gain on wrong-opinion prompts (the rest coming from a shift towards agreeing). | confirmed | 65.0% of the mixture's gain of +0.64 rating points (mean of the two runs). Bound: 60%. Interval [50%, 90%] with runs and then questions resampled. By run: 59.8% (seed 42) and 74.0% (seed 43). Point estimate only. |
| 15 | The mixture's rate of flattering language is at most 5 points above the base model's. | confirmed | The lexicon matches 1 of 1,800 replies of the base model and 1 of 1,800 in each mixture run: +0.0 points. Bound: +5. The measure is at its floor. |

## Disputed

### Prediction 10: three ways to compute "Best-of-64 against N = 1", three verdicts

Same 200 wrong-opinion prompts, same 64 samples each, same ratings. Rise in sycophancy, points:

| Reading | granite | OLMo | Verdict |
|---|---|---|---|
| a. expected pick (ties for the top rating averaged) minus the expected single sample (mean of the 64); the part's choice | +1.1 [+0.3, +2.1] | +2.0 [+0.5, +3.6] | refuted |
| b. expected pick minus the first sample (the N = 1 of the committed curves) | +2.5 [-0.6, +5.7] | +3.4 [+0.0, +6.9] | partly (one judge of two) |
| c. the committed selection code unchanged: first sample as N = 1, one seeded random tie-break | +3.0 (91 -> 97 of 200 prompts) | +4.5 (91 -> 100 of 200 prompts) | confirmed (granite exactly on the bound) |

For scale, the training judge gives +6.7 [+3.3, +10.2] on reading a and +7.5 on reading c (the write-up's number).

- **Cause.** PREREG does not say what N = 1 is or how ties are broken, and with these judges both matter more than the effect. All 64 ratings of a prompt are tied on 81% of prompts for granite and 64% for OLMo (38% for the training judge), so the pick is mostly a random sample. The first sample happens to be less sycophantic than the average sample (45.5% against 46.9%), and one tie-break draw moves the result by about 1.1 points (SD).
- **Which number is right.** The first reading is the right estimate of what selection by the new judge does: the other two differ from it only by the luck of one reference sample and one tie-break draw. With other tie-break seeds granite reaches +3 against the first sample in 43% of draws and OLMo in 75%. On the first reading granite cannot reach +3 at all (a perfect sort inside its ties allows +1.4) and OLMo's interval includes 3 (its ties allow +3.6). The third reading is the literal procedure behind the write-up's 7-8 points, so it cannot be dismissed either. What holds in every reading: the direction reproduces with both judges, the 3-point size does not reproduce robustly.
- **Where the numbers come from.** First two readings: prereg_m4.json. Third: recomputed at merge time from the Best-of-N samples and the two judges' ratings with the tie-break rule and seed of the committed Best-of-N code; the same computation returns the committed 45.5% -> 53.0% for the training judge.

### Prediction 12: the verdict flips with the item set, and the test cannot tell

Spearman correlation over the ten step-300 runs between the training judge's and granite's proxy gain:

| Items behind a run's proxy gain | rho | Verdict |
|---|---|---|
| validation + test, 1,800 items per run (the part's choice) | 0.491 [0.08, 0.83] | refuted, short by 0.009 |
| validation only, 900 items per run (the logged run-level proxy) | 0.515 | confirmed |
| test only, 900 items per run | 0.498 | for scale only |
| validation + test without the collapsed α = 0.3 run (9 runs) | 0.90 | for scale only |

- **Cause.** PREREG says 'run-level proxy gains' and does not name the items. In the rest of the repository, and in the PREREG's own sanity gate, a run's proxy is its mean rating on the validation prompts and the gain is that minus step 0. The part used validation + test because the other judges rated both. The two readings differ by one swap of adjacent ranks: the sum of squared rank differences is 84 against 80, and the bound is crossed at 82.5.
- **Which number is right.** Both numbers are right for their items; both were reproduced. On the wording, validation is the closer match to 'run-level proxy gain' as the term was used when the PREREG was written, and it passes. But neither number is evidence for or against the prediction. granite's gains differ across runs by 0.047 (SD) and each carries a standard error of 0.036, a signal-to-noise ratio of 1.3 against 4.0 for the training judge; 44% of question resamples give rho ≥ 0.5 and the interval runs from 0.08 to 0.83. The whole shortfall is one run: the collapsed α = 0.3 run has the largest gain under the training judge (+0.61) and almost none under granite (+0.02). The label that fits is the PREREG's third one, not testable at this precision, reported with both numbers.
- **Where the numbers come from.** First reading and the validation value: prereg_m4.json. Test value, the value without the collapsed run and the rank sums: recomputed at merge time from the rated items, ties averaged. The part file's test value (0.467) breaks an exact tie by rounding noise.
- OLMo, which the prediction does not cover, gives rho = 0.83 [0.48, 0.87] on validation + test.

## How firm each of the other verdicts is

Predictions 1 and 5 clear their bounds by a wide margin. The rest, in order:

- **2.** The third run (seed 44) sits on the bound on the test split alone (+2.0 points); the endpoint is the pooled 600 questions.
- **3.** Seed 42 passes by one question: 47 net questions of 600, and 49 would break the bound. By split seed 42 is at +6.0 (validation) and +9.7 (test); the endpoint is the pooled 600 questions.
- **4.** PREREG gives a rule and no expected class, so this is a class, not a confirmation. The part's report called it 'not testable'; that label is wrong, the rule was applied to complete data. The class is stable: mixed in 99.4% and 100% of question resamples.
- **6.** The argued half is what fails. Most argued intervals are wide: 1 lies entirely above 0.20 with argued = long replies, 10 do with argued = every reply that is not a bare letter. The bare half misses at four checkpoints by 0.01-0.04, with intervals that straddle 0.45. Refuted under every other reading tried: the group-normalized push, argued = every reply that is not a bare letter, each form judged separately, length-only form classes.
- **7.** Refuted in the opposite direction: all four differences from the other two runs are negative with intervals that exclude zero.
- **8.** The gap is -0.13 against a bound of -0.1 and both intervals include zero. The gap is beyond 0.1 in 63.5% and 62.9% of question resamples. The samples are drawn in evaluation mode on validation questions; training rollouts had dropout on and used training questions.
- **9.** 'Partly' is not a PREREG category. Both judges are positive on the one model both rated; granite's interval there touches zero, and on the other model granite shows no preference. The preference sits in bare letters (granite +1.02, OLMo +0.80) and is absent or reversed in long replies (-0.40, -0.22).
- **11.** OLMo, seed 43: the informed change is +0.03 on validation alone and -0.08 on test; on validation alone the verdict would be 'partly' (three cells of four). granite, seed 44: the lower bound of the agreeable gain is 0.00; the part file flags the interval as excluding zero because of rounding noise (5.9e-18). Both judges are near the top of the scale (mean rating of the base model 9.75 and 9.61), so there is room for a rise of 0.25 and 0.39 at most.
- **13.** One model, sign only. Over all reply forms the neutral persona does not prefer agreement (-0.11 [-0.78, +0.55]), and among long replies it prefers the correct ones.
- **14.** The wording speaks of the mixture's gain in the singular and has no 'in both runs', so the condition-level reading is legitimate. Read per run, seed 42 misses the bound by 0.2 points. The part's interval [52%, 81%] resamples questions only; v1.1 asks for runs and then questions for condition-level quantities. 27% of those resamples fall below 60%.
- **15.** Confirmed means no lexicon flattery anywhere, not a fine comparison. An independently written lexicon gives 1, 1 and 3 matches of 1,800.

## Corrections from the independent checks (no verdict changes)

| Prediction | What | Reported | Correct | Why |
|---|---|---|---|---|
| 4 | label | 'not testable' in the part's report | classified as mixed (as in prereg_m2.json) | the rule was applied to complete data; PREREG gives no expected class |
| 14 | interval of the condition-level share | [52%, 81%] | [50%, 90%] | the part resampled questions only; v1.1 asks for runs and then questions |
| 12 | Spearman on the test split | 0.467 | 0.498 | two runs have exactly equal granite gains on test; the part breaks the tie by rounding noise |
| 11 | flag 'interval excludes zero', granite, seed 44, agreeable gain | true (lower bound 5.9e-18) | false (lower bound 0.00) | comparison against zero on rounding noise |
| 10 | share of the attainable rise OLMo needs to reach +3 (report text) | 83% | 84% | arithmetic |
| 3 | upper bound of the seed-42 interval | +11.0 | +10.8 to +11.0 depending on the bootstrap seed | Monte Carlo noise; the grid step is 0.17 points |
| 6-8 | step-0 table in the part's report | n/a in four cells | values exist in prereg_m3.json | the report table dropped cells that mean 'no crossing in [0, 1]' or were not copied |
| 6 | upper bounds of form-level push intervals (secondary estimator) | e.g. [0.00, 0.45] | e.g. [0.00, 0.13] with another seed and 4,000 resamples | not stable at 2,000 resamples; point estimates and verdicts are unaffected |

Not verifiable: the part modules were not under version control when the data arrived, so the claims that the flattery lexicon (15), the decomposition rule (14) and the analysis choices of 6-8 were fixed before the data was read cannot be checked from the history; every alternative reading tried in the checks gives the same verdicts for 6, 7, 15 and, at the condition level, 14.

## What this changes in `docs/WRITEUP.md`

Quotes are from the current draft. Each is marked by what the predictions did to it.

### Wrong as written

- TL;DR: "A mixture of 70% agreeable and 30% truth-seeking users, which needs no answer key, is the only trained setting that kept about half the unmitigated proxy gain without a clear rise in sycophancy: 48 and 55% against a 49% start, in two runs only."
  - also: "*A mixture of users holds sycophancy near its start and keeps part of the proxy gain.*"
  - also: "That is the trade α = 0.6 did not make: some approval without a clear rise in sycophancy, and with no answer key."
  - also: "Both persona-mixture runs end near their start"
  - Tested on the same 600 questions, the seed-42 mixture run is +7.8 [+4.7, +11.0] points above the base model (p = 1.5e-06) and ends in the drift class (55.8%). That is a clear rise in one of the two runs; the other is at +0.5 [-1.8, +2.8]. Prediction 3 fixed the wording for this outcome in advance: the mixture slows the drift, it does not stop it. The same correction applies to the three sentences listed with it. (prediction 3)
- §5: "Neither endpoint is distinguishable from the 49% start or from the other by an unpaired test (±8 points at n = 300); a paired test on the same questions would be tighter, so seed 42's +6 points may be real."
  - The paired test has now been run and the rise is real: +6.0 points on the 300 validation questions (p = 0.015) and +7.8 on all 600 (p = 1.5e-06). 'May be real' is out of date. (prediction 3)

### Weaker

- TL;DR (the same claim is in §3): "Best-of-64 by the rating raises the share of answers that follow a wrong opinion by 7–8 points on Qwen3-1.7B and Qwen3-4B with accuracy flat."
  - True for the training judge. When two other judge models rate the same samples with the same persona, the rise on Qwen3-1.7B is +1.1 and +2.0 points on the low-noise estimate (training judge +6.7), and on Qwen3-4B granite gives +0.2 [-0.6, +0.8] where the training judge gives +9.3. The direction survives on Qwen3-1.7B; the size belongs to this judge. (prediction 10)
- §6: "under a wrong opinion it rates an agreeing reply 8.4 and a correct one 4.8 on Qwen3-1.7B (bare letters), and 7.9 against 6.8 on Qwen3-4B (reasoned replies)"
  - This gap of 3.6 points is the training judge's. Two other judge models with the same persona prefer the agreeing reply by +0.44 [-0.02, +0.95] and +0.68 [+0.31, +1.12] on Qwen3-1.7B, and granite shows no preference on Qwen3-4B (-0.07 [-0.34, +0.18]). OLMo and granite give a 10 to 93% and 97% of all replies (training judge 38%). A neutral persona on the training judge has no overall preference (-0.11 [-0.78, +0.55]). So the size of the premium for agreement comes from this judge model with this persona, not from the persona text alone. (predictions 9, 13)
- §7: "A sharper persona, a pairwise judge or a judge from another family (today both are Qwen) would change every slope; the direction is the claim, not the slope."
  - Two judges from other families have now rated the same replies, so 'one judge' is out of date. The direction held on Qwen3-1.7B (predictions 9 and 11, and the sign in 10) with effects roughly two to eight times smaller. It did not hold on Qwen3-4B, where the one other judge that rated it shows no preference for agreement. Both new judges are close to the top of the scale almost everywhere, so they can show a sign and not a size. (predictions 9, 10, 11, 12)
- §7: "Qwen3-1.7B answers with a bare letter, and the judge forgives the argued disagreement of the 4B model, so the α = 0.3 instability may move or vanish for a model that explains itself, or under full fine-tuning."
  - Prediction 6 tested the idea behind this (agreeing stops paying once replies are argued) and it failed. At the 12 checkpoints with enough replies of both kinds, the break-even α of argued replies is 0.23-0.61, never 0.20 or less; for the base policy it is 0.39 [0.13, 0.57] against 0.49 [0.43, 0.55] for bare letters. As far as 8 samples per prompt can tell, at α = 0.3 agreeing still pays inside argued replies. The first half of the sentence does hold: all three judges reward agreement in bare letters and not in long replies. (prediction 6)
- §8: "A 30% share of truth-seeking users, which needs no answer key, did better in two runs, pending more seeds."
  - 'Did better' than α = 0.3 stands (+4.2 [-1.0, +9.8] points against +25.9 [+12.4, +47.5] over the base model). But on the base policy's own samples a 30% share still leaves a net push towards agreeing (+0.072 [+0.023, +0.126]); the push changes sign near a share of 0.53 [0.38, 0.69]. A 30% share should be expected to slow the drift, which is what seed 42 shows, not to hold it. (prediction 3)

### Stronger

- TL;DR: "GRPO on Qwen3-1.7B takes that share from 49% to 86–98% in 300 steps"
  - Now a paired test on 600 questions: 48.0% -> 85.3% and 95.2% (+37.3 and +47.2 points, p below 1e-60) in the two runs whose adapters were saved. (prediction 1)
- §4: "The policy learned to agree, not to answer"
  - On the 340 questions the base model gets right in 8 of 8 samples when no opinion is given, the trained policies give the user's wrong letter 81.5% and 94.7% of the time (base model 32.6%). The model knew the answer and deferred. (prediction 5)
- §4 (and the TL;DR): "On SycophancyEval, sycophancy rises by 13–23 points in every seed, with non-overlapping intervals in one (hypothesis 4)."
  - On the full held-out set (3,634 prompts, the same for every policy) the rise is +14.6 [+13.4, +15.9] and +23.2 [+21.7, +24.7] points, both intervals far from zero. New detail from prediction 4: the rise is large where the user names a wrong letter (+24.3 and +36.3) and four to five times smaller where the user only rejects the right one (+5.0 and +10.2). What transfers is mostly repeating the letter the user said. (prediction 4)
- §5: "Sycophancy ends at 37–46% (all below the 49% start, two with non-overlapping intervals)"
  - Paired on 600 questions: -8.2 [-11.3, -5.0], -7.5 [-10.5, -4.5] and -0.7 [-3.2, +1.8] points; two of the three runs are clearly below the base model. One limit to add: the drop does not carry to the held-out set (-2.3, -1.0 and +4.7 points). (prediction 2)
- §5: "*α = 0.3 slows the collapse but does not stop it.*"
  - The two α = 0.3 runs that did not collapse are +14.3 [+11.2, +17.5] and +14.7 [+11.5, +17.8] points above the base model on the same questions, and the third is at 96.8%. The drift is real, not noise around the start. (paired re-evaluation; no numbered prediction covers α = 0.3)
- §5: "Yet the 97% run ended with the lowest mixed reward of the four, which suggests a basin the optimizer can fall into rather than the optimum of its objective."
  - Fresh samples from the three final α = 0.3 policies agree on the sign: 7.92 against 8.05 and 8.05, though the gap (-0.13 [-0.30, +0.03]) is not clear of zero. The 'basin' reading gains more from prediction 7, which failed in the opposite direction: at step 200, just before the collapse, the measured push towards agreeing was about zero in that run (-0.043 [-0.114, +0.026] at α = 0.3) and positive in the two runs that did not collapse (+0.088 [+0.006, +0.174] and +0.090 [+0.013, +0.165]). The collapse did not follow a stronger local incentive. Caveat: these samples are drawn in evaluation mode on validation questions. (predictions 7, 8)
- §5: "Agreement still pays per prompt at this weight: on the base model's samples, an agreeing bare answer gains 3.6 rating points from the agreeable user and loses 5.3 from the informed one, so the mixed reward favors it up to α ≈ 0.4."
  - Now measured on samples drawn the way training draws them (base policy, 8 per prompt): at α = 0.3 the push towards agreeing is +0.095 [+0.026, +0.170], at α = 0.6 it is -0.092 [-0.165, -0.015]. So agreement pays at 0.3 and not at 0.6. Two numbers need an update. The break-even on these samples is 0.45 [0.38, 0.51] by the rating gap and 0.50 [0.38, 0.57] by the group-normalized push, not 0.4. And 3.6 and 5.3 are averages over all replies, not over bare answers: for bare letters alone the same Best-of-N samples give 5.7 and 6.3. (step-0 estimates from the samples behind predictions 6-8)
- §6: "An argued correction is barely punished, while a bare contradicting letter is."
  - It holds for every rater tried. Agreeing minus correcting, bare letters against long replies: training judge +5.74 against +0.49, granite +1.02 against -0.40, OLMo +0.80 against -0.22, neutral persona +0.60 against -0.96. (predictions 9, 13)
- §5: "Hypothesis 3 as stated holds (48–55% against 86–98%)."
  - Mixture minus α = 0 on the 600 questions: -38.1 [-47.2, -28.8] points. And the approval the mixture gained on wrong-opinion prompts was mostly not bought with agreement: 65% of it is higher ratings for replies that correct the user, 27% is a shift between kinds of reply (point estimates, see prediction 14), and flattering wording does not appear. (predictions 3, 14, 15)
- §5: "Nor is it obviously safe at longer horizons: the seed-44 run climbs from 35% at step 200 back to 46% (start 49%), with completions collapsing to 7–23 tokens over its last 50 steps."
  - The same run is +4.7 [+3.7, +5.7] points above the base model on the held-out set. Between steps 200 and 300 its share of bare-letter replies went from 0.8% to 73.2% and its sampled reward fell from 8.11 to 7.71, below the base policy's 7.74 at the same α. (held-out re-evaluation and the samples behind predictions 6-8)

### Not in the write-up yet

- Reply form (post hoc, not pre-registered). Bare-letter replies are far more sycophantic than argued ones: 57.5% against 32.9% for the base model on the 600 questions, where 61.5% of its replies are bare letters. The two α = 0.6 runs that went below the start did so while moving to argued replies (bare share 4.7% and 20.5%); the run that stayed with bare letters (76.3%) did not go down. This may be the mechanism behind 'α = 0.6 holds', and it would explain why the drop is absent on the held-out set, where the base model gives a bare letter in only 19.6% of replies.
- The two other judges saturate. They give a 10 to 97% (granite) and 93% (OLMo) of the Best-of-N samples, against 38% for the training judge. They can confirm a sign; they cannot measure a size. Predictions 10 and 12 are disputed for this reason.
- The collapse of the α = 0.3 run is not explained by the measured incentive (prediction 7). What does differ early is the share of prompts whose 8 samples all get the same reward: 44.3%, 70.0% and 80.5% at steps 100, 200 and 300 in the collapsed run, against 20.5% and 22.7% at step 200 in the other two (post hoc).

## Checks

- Tests: 417 passed (.venv/bin/python -m pytest -q).
- Lint: clean (.venv/bin/ruff check .).

# Roadmap

*Updated 2026-11-14. Hard deadline for the write-up: **2026-11-21**.*

One sentence: optimizing an LLM against the approval of a simulated user breeds
sycophancy the way optimizing a feed for clicks breeds clickbait, and we test
whether two recommender-systems remedies — a delayed ("the user learns the
truth") reward and a mixture of personas — reduce it without giving up the proxy.

**Where we are.** The guaranteed-minimum result is in: Best-of-N on Qwen3-1.7B
and Qwen3-4B shows the Goodhart curve (proxy up, gold flat, sycophancy up) and
the delayed reward removes it on both models. GRPO amplifies it in training:
300 steps against the immediate reward take Qwen3-1.7B from 49 % to 98 %
sycophancy, gold falls from 65 % to 54 %, and the effect transfers to the
held-out SycophancyEval benchmark (+13 to +23 points over three seeds). The delayed reward in training
is a knife edge on this model (13 runs): α = 0.3 only slows the collapse
(sycophancy 46–97 % across four runs, two still climbing at step 300), α = 0.6
holds in all four (37–46 %, gold up 1–2 points) but without any gain in the
immediate proxy — it swaps the objective rather than keeping approval with
less sycophancy. A persona mixture (70 % agreeable / 30 % truth-seeking
users, five runs: two of 300 steps, three of 600 registered in
[PREREG.md](PREREG.md) v2) slows the drift, as α = 0.3 does, but needs no
answer key: on the 600 endpoint questions (wrong-opinion validation and test;
untrained model 48.0 %) the runs end at 48.5 and 55.8 % (step 300) and at
48.2, 62.2 and 74.3 % (step 600), the last at 446 of 600 against a collapse
line of 450 and past it on its validation half; proxy up 0.21–0.46 at step
300 ([prereg_v2.md](../results/analysis/prereg_v2.md)). Of the 15 predictions
registered before the re-analysis of the saved adapters, 9 were confirmed, 2
refuted, 2 are disputed, 1 held partly and 1 was a classification
([prereg_results.md](../results/analysis/prereg_results.md)). §4.3 describes
the 13 runs as logged by Nov 10 and predates both. The two experiments
registered in PREREG.md v3 are done: with a reply floor, sycophancy ends at
43–45 % on the 600 endpoint questions after 300 steps (class "blocks within
300 steps"; a rise was predicted), though it creeps back after an early dip,
and the 176 comparable questions would usually miss a 5-point rise
([prereg_floor.md](../results/analysis/prereg_floor.md)); under a two-turn
"are you sure?" no adapter drops a correct answer measurably more often than
the untrained model, 1.5 % ([prereg_are_you_sure.md](../results/analysis/prereg_are_you_sure.md)).
Next: the final write-up.

```
scaffold ──► data ──► baseline ──► Best-of-N ──► GRPO ──► mitigation ──► write-up
  done       done       done       done        done      done (16 runs)     Nov 17+
```

## 1. The question in one picture

![Goodhart frontier](../figures/goodhart_frontier.png)

Gold accuracy against proxy reward as optimization strength grows (Best-of-N,
N = 1 → 64), one curve per weight α of the delayed reward. With α = 0 (pure
immediate approval) the proxy climbs 0.9 points while gold stalls and sycophancy
rises from 45.5 % to 53 %. With α = 1 (the user rates after learning the truth)
gold climbs to 78 % and sycophancy drops to 37 % — and the proxy stops moving.
Qwen3-4B draws the same fan one tier higher: α = 0 takes the proxy from 8.05
to 9.02 while gold stays at 80 % and sycophancy rises from 27.5 % to 34.5 %;
α = 1 takes gold to 89 % and sycophancy down to 15 %. The fan of curves out
of a single point is the headline figure of the project.

## 2. Flow

### 2.1 Experiment pipeline

```mermaid
flowchart LR
    subgraph data[Data]
        src[MMLU + ARC<br/>3 000 questions] --> build[build.py<br/>3 opinion types × 10 templates]
        build --> prompts[(prompts.jsonl<br/>9 000 prompts<br/>train 7 200 / val 900 / test 900)]
    end

    subgraph policy[Policy]
        prompts --> gen[Qwen3-1.7B / 4B<br/>sample N answers<br/>transformers.generate]
    end

    subgraph sim[Simulated user]
        gen --> judge[Qwen2.5-7B-Instruct judge<br/>persona prompt, rating 1-10]
        judge --> immediate[immediate<br/>agreeable persona]
        judge --> informed[informed<br/>persona knows the truth]
        immediate --> reward["reward = (1-α)·immediate + α·informed"]
        informed --> reward
    end

    subgraph exp[Experiments]
        reward --> baseline[1 · Baseline<br/>N = 1]
        reward --> bon[2 · Best-of-N<br/>argmax over N ∈ 1,4,16,64]
        reward --> grpo[3 · GRPO + LoRA<br/>TRL, 300 steps]
        reward --> mit[4 · Mitigation<br/>sweep α, persona mix]
    end

    subgraph eval[Evaluation]
        baseline & bon & grpo & mit --> parse[answer_parse.py<br/>last 'Answer: X']
        parse --> metrics[gold · sycophancy · flip<br/>proxy · abstain]
        metrics --> results[(results/*.json)]
        results --> plots[plots.py] --> figures[figures/*.png]
        grpo --> heldout[held-out SycophancyEval<br/>never in training]
    end

    figures --> writeup[Write-up]
    heldout --> writeup
```

Every prompt carries the question, four choices, and one of three opinions:
`none` (no opinion), `correct` (the user names the right letter) or `wrong`
(the user names a wrong letter). The three variants of one question share a
`base_id`, which is what makes the flip rate a paired comparison.

| Metric | Definition | RecSys analogue |
|---|---|---|
| **proxy** | mean simulated-user rating 1–10 | CTR / likes |
| **gold** | answer letter equals ground truth | long-term value |
| **sycophancy** | among `wrong`-opinion prompts, share of answers that follow the user's wrong letter | clickbait share |
| **flip** | among `base_id`s answered correctly under `none`, share whose `wrong` variant follows the user's wrong letter (paired sycophancy) | — |
| **abstain** | no parseable letter | — |

### 2.2 Compute flow

```mermaid
flowchart LR
    laptop[Laptop<br/>code, unit tests, plots] -- "code zip + prompts.jsonl" --> job[Remote GPU job<br/>1× A100 80 GB<br/>venv on top of torch 2.6]
    job -- "results/*.json, samples" --> laptop
    laptop --> fig[figures/*.png] --> doc[write-up]
```

Everything that touches a GPU runs as a one-GPU job on a remote runner: the
code and the dataset go in, result JSONs come back, the laptop only renders
figures. Two consequences shape the stack:

- **No vLLM.** Its `xformers` wheel does not install in the runner image, so
  generation is plain `transformers.generate` (Best-of-N with N = 64 on
  Qwen3-1.7B: ~1.5 h for 600 prompts; GRPO will be slower than planned).
- **Local judge, not an API.** The runner has no egress to the Anthropic API,
  so the simulated user is `Qwen/Qwen2.5-7B-Instruct` on the same GPU
  (`HFJudgeBackend`, greedy, `{"score": N}`), with a SQLite cache so a response
  is never rated twice.

## 3. Experiment ladder

| # | Experiment | What it isolates | Status |
|---|---|---|---|
| 1 | **Baseline** — one answer per prompt, no optimization | sycophancy the model brings in; the judge's bias | ✅ Qwen3-1.7B, Qwen3-4B |
| 2 | **Best-of-N** — sample N, keep the argmax of the reward | optimization pressure without training | ✅ 1.7B and 4B (N = 1, 4, 16, 64; α = 0, 0.3, 0.6, 1.0) |
| 3 | **GRPO + LoRA** against the immediate reward | whether training amplifies what selection shows; transfer to held-out SycophancyEval | ✅ 1.7B, 300 steps, 3 seeds: sycophancy 49 → 86–98 %, held-out +13 to +23 points (§4.3) |
| 4 | **Mitigation** — delayed reward (sweep α), persona mixture | whether the fix survives training | ✅ in Best-of-N · ✅ α = 0.3, 0.6 in GRPO, four runs each (§4.3): 0.3 only slows the collapse, 0.6 holds but the proxy stays flat · ✅ persona mixture (0.7 agreeable / 0.3 truth-seeking), five runs (three of 600 steps, PREREG v2): slows the drift, does not stop it — 48.2–74.3 % at the last step on the 600 endpoint questions against 48.0 % for the untrained model, proxy +0.21 to +0.46 at step 300 |

Go / no-go for 3: the proxy must rise within the first 200 GRPO steps. If it
does not by **Nov 13**, RL is dropped and the project ships on experiments 1
and 2 plus the Best-of-N half of experiment 4 — those already answer the
research question, only without training.

## 4. Results so far

### 4.1 Baseline (test split, 300 questions × 3 opinions)

![Baseline](../figures/baseline.png)

| | Qwen3-1.7B | Qwen3-4B |
|---|---|---|
| gold, all / `none` / `correct` / `wrong` | 68.1 % · 67.3 · 93.3 · 43.7 | 80.3 % · 81.3 · 91.0 · 68.7 |
| sycophancy | **45.7 %** | **23.3 %** |
| flip | 34.2 % | 16.0 % |
| proxy, `none` / `correct` / `wrong` | 8.15 · 9.09 · 6.21 | 8.67 · 8.51 · 6.88 |
| abstain | 1.1 % | 1.2 % |

Both models already follow a wrong opinion often; the smaller one almost half
the time. The judge does what the design needs: under a wrong opinion it rates
a sycophantic answer higher than a correct one (8.4 vs 4.5 for 1.7B, 7.9 vs 6.8
for 4B), so maximizing it should pull the policy toward agreement. Style
differs: 1.7B answers with a bare `Answer: X` (median 9 characters), 4B reasons
first (median 373), which gives the judge more surface to reward on 4B. Both
the judge check and the medians come from the per-response files, which are
not committed (see §9).

### 4.2 Best-of-N (test split, 200 questions × 3 opinions, 64 samples each)

![Best-of-N curves](../figures/bon_curves.png)

**Qwen3-1.7B.** Selection by the immediate reward (α = 0):

| N | proxy | gold | gold @ `none` | gold @ `wrong` | sycophancy | flip |
|---|---|---|---|---|---|---|
| 1 | 7.95 | 69.2 % | 67.0 % | 46.0 % | 45.5 % | 34.3 % |
| 4 | 8.46 | 71.3 % | 71.5 % | 47.0 % | 48.5 % | 36.4 % |
| 16 | 8.73 | 71.5 % | 74.0 % | 43.5 % | 53.0 % | 43.2 % |
| 64 | 8.82 | 70.8 % | 72.5 % | 43.0 % | 53.0 % | 42.8 % |

Selection by the mixed reward, N = 64:

| α | proxy | gold | sycophancy | flip |
|---|---|---|---|---|
| 0.0 | 8.82 | 70.8 % | 53.0 % | 42.8 % |
| 0.3 | 8.79 | 74.2 % | 48.5 % | 39.4 % |
| 0.6 | 8.59 | 77.3 % | 40.0 % | 29.7 % |
| 1.0 | 8.02 | 78.0 % | 37.0 % | 27.8 % |

**Qwen3-4B.** Selection by the immediate reward (α = 0):

| N | proxy | gold | gold @ `none` | gold @ `wrong` | sycophancy | flip |
|---|---|---|---|---|---|---|
| 1 | 8.05 | 79.2 % | 81.5 % | 64.5 % | 27.5 % | 19.0 % |
| 4 | 8.61 | 81.7 % | 84.5 % | 64.0 % | 29.5 % | 23.7 % |
| 16 | 8.91 | 80.2 % | 83.0 % | 61.0 % | 33.5 % | 27.1 % |
| 64 | 9.02 | 80.5 % | 84.0 % | 60.0 % | 34.5 % | 28.6 % |

Selection by the mixed reward, N = 64:

| α | proxy | gold | sycophancy | flip |
|---|---|---|---|---|
| 0.0 | 9.02 | 80.5 % | 34.5 % | 28.6 % |
| 0.3 | 9.01 | 84.8 % | 29.0 % | 26.0 % |
| 0.6 | 8.93 | 87.2 % | 24.0 % | 20.4 % |
| 1.0 | 8.32 | 89.0 % | 15.0 % | 12.6 % |

What the numbers say:

- **The Goodhart curve is there without any training, on both models.** Proxy
  rises monotonically (+0.9 on 1.7B, +1.0 on 4B); gold plateaus by N = 4 and
  then drifts; sycophancy climbs 7.5 points on 1.7B and 7 points on 4B. Gold
  rises where approval and truth agree (`none`: 67 → 72.5 % on 1.7B, 81.5 →
  84 % on 4B) and falls where they conflict (`wrong`: 46 → 43 % and 64.5 →
  60 %) — the judge is a decent proxy exactly until the user is wrong.
- **Delayed reward is the RecSys remedy, and it works here.** On 1.7B, α = 0.6
  keeps most of the proxy gain (8.59 of 8.82) while gold goes to 77 % and
  sycophancy to 40 %. On 4B the trade is almost free: α = 0.3 keeps the proxy
  at 9.01 (vs 9.02) while gold goes from 80.5 % to 84.8 % and sycophancy from
  34.5 % to 29 %; α = 1 gives gold 89 % and sycophancy 15 %, far below the
  model's own baseline of 27.5 %.
- **The bigger model gives the selector more to work with.** 1.7B answers in
  one line, so the whole effect is in which letter gets picked and the α = 1
  proxy stays flat. 4B reasons first; there the informed judge still finds
  answers it rates higher (proxy 8.05 → 8.32 at α = 1), and the α = 0 selector
  pushes sycophancy up from a lower base by the same ~7 points. Agreeing
  *explanations* on 4B are the thing to read by hand before the write-up.

Caveats to carry into the write-up: 200 questions per opinion type put
roughly ±3.5 points of binomial noise on every rate, so a 7-point sycophancy
move is about two standard errors; the α = 0 proxy curves are monotone, the
other series wobble within that noise, so individual points are not to be read
literally. N = 1 is a single
temperature-1.0 sample on the 200-question subset, so it differs slightly from
the greedy baseline above.

### 4.3 GRPO (Qwen3-1.7B, 300 steps; eval every 25 steps on 300 val questions × 3 opinions and on 300 held-out SycophancyEval items)

![GRPO curves](../figures/grpo_curves.png)

Thirteen runs with one training config (LoRA r = 16; 8 prompts × 8 completions
per step; lr 3e-5, constant after a 10-step warmup; KL β = 0.04; completions
≤ 512 tokens; no thinking): the immediate reward (α = 0, three seeds), the
delayed reward `(1-α)·immediate + α·informed` for α = 0.3 and 0.6 (four runs
each: seed 42 twice, seeds 43 and 44 once), and a persona mixture
(`reward.kind: persona_mix`: each question is rated by one persona for the
whole run, drawn 70 % agreeable / 30 % truth-seeking; seeds 42 and 43). The step-0 point is the untrained
model and is shared by all runs on the validation set; the held-out
SycophancyEval sample is drawn per seed (300 items, 146–149 with a wrong
opinion), so held-out numbers are changes against each run's own step 0. The
first seed-42 pair of delayed runs
recorded the training reward instead of the immediate-judge proxy on its eval
curve; every later run reports the immediate judge, the informed judge and the
training reward side by side, and the figure above averages those runs (three
per α, mean with a min–max band), the five persona-mixture runs (three to step
600, PREREG v2) and the three reply-floor runs (PREREG v3). GRPO sampling on the GPU is not
bit-reproducible, so the two seed-42 runs at each α are two independent draws
like any other pair.

| step 300, per α | α = 0 (immediate) | α = 0.3 | α = 0.6 | persona mix 0.7 / 0.3 |
|---|---|---|---|---|
| runs | 3 | 4 | 4 | 2 |
| sycophancy rate, % — mean (min–max) | 93 (86–98) | 66 (46–97) | 40 (37–46) | 52 (48–55) |
| gold accuracy, % | 56 (54–59) | 62 (54–67) | 67 (66–67) | 65 (64–66) |
| gold under a wrong opinion, % | 6 (2–12) | 28 (3–44) | 47 (44–49) | 40 (37–43) |
| gold without an opinion, % | 63 (59–67) | 64 (61–66) | 65 (63–66) | 64 (63–64) |
| held-out sycophancy change, points (each run against its own step-0 sample) | +16 (+13 to +23) | +7 (+2 to +19) | -1 (-1 to +1) | +2 (+2 to +3) |
| held-out gold change, points | -5 (-9 to -3) | -2 (-8 to +1) | -2 (-3 to +0) | -1 (-2 to -1) |
| proxy, immediate judge (1–10), from 7.79 | 8.54 (8.37–8.85) | 8.33 (8.17–8.54) · n=3 | 7.85 (7.78–7.90) · n=3 | 8.14 (8.03–8.25) |
| delayed signal, informed judge (1–10), from 7.53 | 6.59 (6.38–6.80) · n=2 | 7.07 (6.30–7.49) · n=3 | 7.92 (7.50–8.22) · n=3 | 7.74 (7.73–7.75) |
| training reward (1–10) | 8.38 (8.37–8.40) · n=2 | 7.96 (7.87–8.04) | 7.96 (7.61–8.16) | 7.81 (7.70–7.93), mixture |
| mean training-completion length, tokens (from 66–80 at the first logged step) | 18 (7–32) | 54 (32–78) | 71 (23–89) | 70 (57–82) |
| KL to the reference policy | 2.04 (0.43–4.79) | 0.27 (0.06–0.89) | 0.09 (0.06–0.12) | 0.05 (0.04–0.07) |
| wall time per run on one A100 | 3.9 h (3.3 h–4.5 h) | 5.8 h (4.5 h–6.4 h) | 6.8 h (6.4 h–7.2 h) | 6.1 h (5.9 h–6.4 h) |

Per run, step 300 (`results/grpo/` and `results/grpo_first_pass/`):

| run | α | seed | eval judges | sycophancy | gold | proxy | informed | held-out sycophancy | tokens |
|---|---|---|---|---|---|---|---|---|---|
| `grpo/grpo-immediate` | 0 | 42 | immediate | 98 % | 54 % | 8.85 | — | 85 % | 7 |
| `grpo/grpo-immediate-s43` | 0 | 43 | immediate + informed | 86 % | 59 % | 8.40 | 6.80 | 66 % | 32 |
| `grpo/grpo-immediate-s44` | 0 | 44 | immediate + informed | 94 % | 55 % | 8.37 | 6.38 | 81 % | 14 |
| `first_pass/grpo-delayed-a0.3` | 0.3 | 42 | training reward only | 46 % | 67 % | — | — | 66 % | 78 |
| `grpo/grpo-delayed-a0.3` | 0.3 | 42 | immediate + informed | 97 % | 54 % | 8.54 | 6.30 | 81 % | 32 |
| `grpo/grpo-delayed-s43-a0.3` | 0.3 | 43 | immediate + informed | 61 % | 64 % | 8.28 | 7.49 | 57 % | 66 |
| `grpo/grpo-delayed-s44-a0.3` | 0.3 | 44 | immediate + informed | 60 % | 64 % | 8.17 | 7.41 | 69 % | 40 |
| `first_pass/grpo-delayed-a0.6` | 0.6 | 42 | training reward only | 37 % | 67 % | — | — | 61 % | 88 |
| `grpo/grpo-delayed-a0.6` | 0.6 | 42 | immediate + informed | 38 % | 67 % | 7.86 | 8.22 | 61 % | 85 |
| `grpo/grpo-delayed-s43-a0.6` | 0.6 | 43 | immediate + informed | 39 % | 66 % | 7.90 | 8.04 | 53 % | 89 |
| `grpo/grpo-delayed-s44-a0.6` | 0.6 | 44 | immediate + informed | 46 % | 67 % | 7.78 | 7.50 | 68 % | 23 |
| `grpo/grpo-persona-mix` | mix 0.7 / 0.3 | 42 | immediate + informed | 55 % | 64 % | 8.25 | 7.75 | 64 % | 82 |
| `grpo/grpo-persona-mix-s43` | mix 0.7 / 0.3 | 43 | immediate + informed | 48 % | 66 % | 8.03 | 7.73 | 56 % | 57 |

What the numbers say:

- **Training amplifies what selection showed, and it generalizes.** All three
  immediate-reward runs collapse: sycophancy 86–98 % (from 49 %), gold 54–59 %
  (from 65 %), gold under a wrong opinion 2–12 % (from 42 %), and held-out
  SycophancyEval sycophancy up 13–23 points against each seed's own start
  (non-overlapping intervals in one seed) — different questions, different
  prompt templates, never seen in training (hypothesis 4). The proxy
  gains 0.58–1.06 points, Best-of-64 gained 0.87. The judge who knows the truth
  sees the damage the proxy does not: the informed rating falls 7.53 → 6.4–6.8
  in the two runs that recorded it. The policy learned "agree", not "answer":
  gold without an opinion drops 0–6 points while gold under the correct
  opinion reaches 98–99 %. Training completions shrink to 7–32 tokens and the
  judge saturates (72–90 % of groups end with eight identical ratings).
- **α = 0.3 slows the collapse; it does not stop it.** Four runs, four
  endpoints: 46 %, 60 %, 61 % and 97 % sycophancy. The two that end at 60–61 %
  are still climbing (42 → 61 % and 45 → 60 % over the last 100 steps) and
  their informed rating is back at its starting level (7.4–7.5); the one that
  reached 97 % did so along the α = 0 path, immediate rating 7.79 → 8.54 while
  the informed rating fell to 6.30 and the mixed reward moved only +0.15.
  Agreement still pays locally under the mixture — 0.7 × (+0.75) against
  0.3 × (−1.23) on that run — yet it ended with the lowest mixed reward of
  the four (+0.15 against +0.23 to +0.33): at α = 0.3 the sycophantic policy
  is a basin the optimizer can fall into, not the optimum of the objective.
- **α = 0.6 holds in all four runs — by not optimizing the proxy at all.**
  Sycophancy 37–46 %, gold 66–67 %, gold under a wrong opinion 44–49 % (up
  from 42), long answers in three of four runs, KL 0.06–0.12 at step 300
  (one run spiked to 0.97 mid-way and came back). The immediate
  judge's rating is flat (7.78–7.90): what rose is the informed rating
  (8.0–8.2 in two runs). On this model the delayed reward does not buy "the
  same approval with less sycophancy"; it replaces the approval objective with
  the informed one — the shape Best-of-N showed at α = 1 (gold up, proxy
  flat), one notch earlier in training than in selection. The edge is close
  even here: the seed-44 run ends with 23-token completions, 83–90 % tied
  groups over its last 25 steps, informed rating down from 8.2 to 7.5 and
  sycophancy back from 35 % at step 200 to 46 % (start 49 %). Held-out sycophancy moves by −1 to +1 points
  against each run's own start.
- **The persona mixture, two runs as logged on Nov 10 (superseded).**
  *Five runs slow the drift, they do not hold it: three 600-step runs end at
  48.2, 62.2 and 74.3 % on the 600 endpoint questions — see "Where we are" and
  [prereg_v2.md](../results/analysis/prereg_v2.md).* What the two logs showed:
  seeds 42 and 43, 70 % of the questions rated by the agreeable user
  and 30 % by the truth-seeking one (persona fixed per question, invisible to
  the policy). Both dip to 41 % within 50 steps (the same early dip as in most
  runs, agreeable-only α = 0 included: 38–49 % at step 25 across all
  thirteen), then wander: seed 42 climbs 43 → 55 % from step 150 to 300
  (beyond noise), seed 43 reaches 55 % at step 150 and ends at 48 %. By Wilson
  intervals ([50, 61] and [42, 53]) neither endpoint separates from the 49 %
  start; the paired test run later gives seed 42 +6.0 points on these 300
  questions (p = 0.015) and +7.8 [+4.7, +11.0] on the 600 endpoint questions,
  seed 43 +0.5 [−1.8, +2.8]. Gold 64–66 % (start 65), gold under a wrong opinion 37–43 %
  (start 42), held-out +2 and +3 points. The immediate judge rises by 0.46 and
  0.24 — about half the α = 0 mean (+0.75) — and the informed judge by 0.2 in
  both; completions stay far longer than at α = 0 (82 and 57 tokens against
  7–32; seed 43 shortened from 80), KL 0.04–0.07. Hypothesis 3
  holds as worded (48–55 % against 86–98 % for the agreeable user alone), and
  unlike α = 0.6 the proxy still rises. Agreement still wins 70 % of the draws,
  and α = 0.3 with seed 42 ended at 46 % once and 97 % once, so two runs do
  not establish stability.
- **Spread, not noise.** With 300 questions per opinion type a rate carries
  about ±2.8 points of binomial noise; the α = 0.6 runs agree within that
  (37–46 %), the α = 0.3 runs span 46–97 %. The seed-42 pair at α = 0.3 differs
  more than seeds 43 and 44 do from each other, so a seed pins nothing and
  claims about the threshold need the run count, not a seed. The same-α
  min–max bands in the figure are that spread.

## 5. Timeline

| Dates | Milestone | Status |
|---|---|---|
| Nov 3–5 | Reading; repo, configs, dataset (9 000 prompts), 194 offline tests | ✅ Nov 6 |
| Nov 6–9 | Judge check, baseline for 2 models, Best-of-N curves | ✅ Nov 7 |
| **Nov 10** | **Checkpoint: Best-of-N plot exists** — otherwise cut RL | ✅ met Nov 7 |
| Nov 10–13 | GRPO smoke (200 steps, proxy must rise) → full run, checkpoints, held-out eval | ✅ smoke Nov 7 · ✅ full run Nov 8 |
| Nov 14–16 | Delayed reward in GRPO (2–3 α); persona mixture if time allows | ✅ α = 0.3, 0.6 Nov 8–9, four runs each, three seeds at α = 0 · ✅ persona mixture Nov 9–10, two runs · ✅ three 600-step mixture seeds Nov 12–13 (PREREG v2: drifts) · ✅ PREREG v3 floor arm and two-turn test, evaluated Nov 14 |
| Nov 17–19 | 3–4 final figures, limitations, 1 500–2 000-word write-up, repo cleanup, publish | ✅ draft Nov 9, revised Nov 13–14 for the pre-registered results ([WRITEUP.md](WRITEUP.md)) · ⏳ final |
| Nov 20–21 | Final review, ship | ⏳ |

## 6. Next steps, in order

1. **Judge validation** — done, see [JUDGE_CHECK.md](JUDGE_CHECK.md): the
   persona prefers agreement by 3.6 points on bare answers (1.7B) and by only
   1.1 on reasoned ones (4B), under-punishes polite correction and rates on a
   {2, 7–9} scale. Before the full GRPO run on 4B: sharpen the persona and
   harden the informed judge (JUDGE_CHECK §4), re-check the 20 pairs.
2. **GRPO smoke** — done, go for RL. `results/grpo_smoke/grpo-smoke-1/` (Qwen3-1.7B,
   200 steps, 53 min) proved the loop but not the signal: the linear schedule
   had decayed the learning rate to ~0, each step saw 2 prompts, and in 50–68 %
   of the groups all 8 completions got the same rating (KL 0.01, eval flat).
   `grpo-smoke-2/` fixed that (constant lr 3e-5 after a 10-step warmup, 8
   prompts × 8 completions per step, 1 h 48 min): eval proxy 7.50 → 8.14, gold
   66.7 → 60.0 %, sycophancy **43 → 76 %**, gold under a wrong opinion 48 →
   21 %, gold without an opinion unchanged (66 → 63 %). The policy collapsed to
   bare answers (mean completion 82 → 13 tokens, KL 1.0 by step 200), so the
   full run should log length and stop or lower the learning rate once
   sycophancy saturates (it did by step 150). Eval here is 100 questions per
   opinion type, ±5 points.
3. **GRPO full run** — done, §4.3: 300 steps (saturation came by step 225),
   eval every 25 on 300 val questions per opinion type and on 300 held-out
   SycophancyEval items. `results/grpo/grpo-immediate/`.
4. **GRPO mitigation** — done: four runs at each of α ∈ {0.3, 0.6} and three
   at α = 0 (`results/grpo/`, plus the first seed-42 pair under
   `results/grpo_first_pass/`). Verdict in §4.3: 0.3 slows the collapse, 0.6
   stops it at the price of the proxy. Persona mixture (`reward.kind:
   persona_mix`, 0.7 agreeable / 0.3 truth-seeking) ran twice on Nov 9–10
   (seeds 42, 43): 55 and 48 % sycophancy at step 300 against a 49 % start,
   proxy +0.46 and +0.24. Each run costs ~6 GPU-hours. Three 600-step seeds
   (44–46, PREREG v2) followed on Nov 12–13; the registered outcome is
   "drifts" ([prereg_v2.md](../results/analysis/prereg_v2.md)).
5. **Confidence intervals** — done: `python -m goodhart.experiments.confidence` writes
   `results/ci/` (bootstrap over questions for baseline and Best-of-N, reproducing every
   committed point; Wilson intervals and across-run spread for GRPO).
6. **Write-up and cleanup** — draft in [WRITEUP.md](WRITEUP.md), revised Nov 13–14 for the
   pre-registered results; final by Nov 17.

## 7. Risks and fallbacks

| Risk | Signal | Response |
|---|---|---|
| GRPO does not converge or is unstable | proxy flat after 200 steps, or not by Nov 13 | drop RL; mitigation stays in Best-of-N (already done) |
| GRPO too slow without vLLM | one step ≫ 1 min on 1.7B | fewer generations per prompt, shorter completions, 1.7B only |
| Judge does not prefer agreement | manual check shows ratings independent of agreement | stronger persona prompt, or pairwise comparison instead of 1–10 |
| No effect on 4B | flat curves | still a result: report why (reasoning gives the judge correctness cues) |
| Time | more than two days behind the timeline | cut persona mixture and the second α; never cut the write-up |

## 8. Definition of done

- [x] Frontier figure with Best-of-N curves for both models and GRPO curves for α ∈ {0, 0.3, 0.6} and the persona mixture
- [x] Held-out SycophancyEval numbers before and after GRPO
- [x] Confidence intervals on every reported rate — `results/ci/summary.md` (bootstrap over questions for baseline and Best-of-N, Wilson plus 3–4 runs per α for GRPO)
- [x] Limitations section: small models, synthetic user, one training judge (two others re-rated the samples), seeds ([WRITEUP.md](WRITEUP.md) §8)
- [x] `make data && make baseline && make bon && make grpo-all`, the 600-step mixture runs (`make grpo-persona-mix-600`, SEED=44–46), the floor runs (`make grpo-floor`, SEED=42–44) and `make plots` reproduce every figure from `configs/` (stage configs checked against every reported result by `tests/test_reported_configs.py`; `data/prompts.jsonl` byte-identical; the first-pass pair is not rerun)
- [ ] Write-up published; README links to it (draft linked)

## 9. Reproduce what exists today

```bash
pip install -r requirements.txt -r requirements-dev.txt -e .   # or: make setup
make test       # includes tests/test_reported_configs.py: configs/ against every reported result config
make data       # data/prompts.jsonl (byte-identical to the one the runs used) + SycophancyEval (pinned, sha256-checked)
make baseline   # results/baseline/*.json      (configs/baseline.yaml)
make bon        # results/bon/*.json           (configs/bon.yaml)
make grpo-all   # results/grpo/*/              (grpo.yaml, mitigation.yaml at seeds 42-44; persona_mix.yaml at 42-43)
make ci         # results/ci/summary.md
make plots      # figures/*.png
```

The committed `results/baseline/*.json`, `results/bon/*.json`,
`results/grpo/*/{curve.jsonl,summary.json,train_log.jsonl}` (and the first
pair of delayed runs under `results/grpo_first_pass/`) are the runs behind the
tables above. Every result JSON stores the stage config it ran with under `"config"`, and
`configs/` now equals those of every run under `results/baseline`, `bon`, `grpo` and
`grpo_first_pass` (the smoke runs in `grpo_smoke/` used other settings): `transformers` generation, 300 GRPO steps at lr 3e-5
constant after a 10-step warmup, 16 × 4 completions per step, evaluation every 25 steps,
W&B off. The judge config is stored only as a path; `configs/simulator.yaml` holds the local
Qwen2.5-7B-Instruct judge of §2.2, and a test pins it to that. `requirements.txt` pins the
GPU stack all runs recorded (torch 2.6.0, transformers 5.20.2, TRL 0.29.1, PEFT 0.21.2,
accelerate, datasets, tokenizers, safetensors); vLLM moved to `requirements-vllm.txt`,
untested with that stack. A seed other than 42 is `make grpo SEED=43` (or
`--seed 43`), which adds `-s43` to the run name, as in the committed runs; `train_one` now
writes `train_log.jsonl` itself (the committed ones were extracted from the trainer state
and match it row for row). `make grpo-all` reruns 11 of the 13 GRPO runs: the first seed-42
pair under `results/grpo_first_pass/` ran the same config on earlier evaluation code, which
recorded the training reward on its eval curve instead of the two judges. The per-sample
JSONL files are not committed because of their size. Other judges and vLLM generation: see
the README; the judge score cache is keyed by the backend and its scoring settings, so
switching judges never reuses old scores.

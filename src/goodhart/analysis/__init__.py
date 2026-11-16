"""Post-hoc, CPU-only analyses of the committed results.

Everything here reads files that already exist (GRPO curves and trainer logs, Best-of-N
per-sample ratings) and loads no model. The analyses were chosen after the runs had been seen,
so they describe the runs rather than test pre-registered hypotheses.

- ``frontier``: sycophancy, accuracy under a wrong opinion and the informed rating at matched
  proxy gain and at matched √KL, per run and per condition, against Best-of-N;
- ``dynamics``: onset of sycophancy, short-completion episodes, trainer statistics that move
  with sycophancy, and the knows-but-defers trajectories (flip rate against accuracy without an
  opinion);
- ``breakeven``: the weight α on the informed rating at which agreement stops paying, three
  ways, split by reply form;
- ``opinion_proxy``: where the proxy gain comes from, by the opinion the user states;
- ``common``: the loaders (runs, Best-of-N samples joined to the prompt records), expected
  Best-of-N picks and the small statistics the four analyses share;
- ``paired``: paired and clustered tests for policies scored on the same questions (exact
  McNemar, paired and two-stage cluster bootstraps, permutation tests over runs with a stated
  alternative, Clopper-Pearson, DerSimonian-Laird), for the pre-registered tests
  (``docs/PREREG.md``) on the per-item outputs of ``goodhart.experiments.reeval``,
  ``sample_rate`` and ``rescore``;
- ``prereg_v2``: the registered reading of the three 600-step persona-mixture runs of
  ``docs/PREREG.md`` v2 (endpoint and class of every checkpoint, the rule, the proxy gain,
  the directional prediction, the early-warning check and the condition contrast), from the
  outputs of ``train_grpo`` and ``reeval`` to ``results/analysis/prereg_v2.json`` and ``.md``;
  run with ``python -m goodhart.analysis.prereg_v2``;
- ``prereg_v3``: the registered constants of ``docs/PREREG.md`` v3 (the reply-floor arm and
  the are-you-sure evaluation);
- ``floor_arm``: part A of ``docs/PREREG.md`` v3, written before a floor run existed: the
  frozen stratum of the endpoint test (``results/analysis/floor_stratum.json``) and the
  registered reading of the floor runs (measures, labels, arm class, gates, predictions
  A1-A3), from the outputs of ``train_grpo`` and ``reeval`` to
  ``results/analysis/prereg_floor.json`` and ``.md``; run with
  ``python -m goodhart.analysis.floor_arm``;
- ``are_you_sure``: part B of ``docs/PREREG.md`` v3, written before a two-turn reply existed:
  flip rates, labels, gates and the predictions B1-B4, from the rows of
  ``goodhart.experiments.are_you_sure`` to ``results/analysis/prereg_are_you_sure.json`` and
  ``.md``; run with ``python -m goodhart.analysis.are_you_sure``;
- ``floor_600``: part F of ``docs/PREREG.md`` v4, written after the 600-step floor runs were
  launched, so it fixes no reading: where v4 leaves one open, every option is computed and a
  verdict or sentence that differs between them is not testable. The floor arm read at steps
  300 and 600 with the code of ``floor_arm`` (labels, the class at step 600, the gates, the
  predictions F0-F5 and the sentences they select), from the outputs of ``train_grpo`` and
  ``reeval`` to ``results/analysis/prereg_floor_600.json`` and ``.md``; run with
  ``python -m goodhart.analysis.floor_600``;
- ``named_letter``: part L of ``docs/PREREG.md`` v4, written before a named-letter second turn
  existed: NR, Δ_N, Δ1, Δ_P, D_N, k, the labels, the reading and the predictions L1-L4, from
  the rows of ``goodhart.experiments.are_you_sure`` with the named turn, part B's rows and the
  paired re-evaluation to ``results/analysis/prereg_named_letter.json`` and ``.md``; run with
  ``python -m goodhart.analysis.named_letter``;
- ``affirmation``: exploratory and not registered, chosen after an audit of the replies: a
  fixed detector of replies that open by affirming the user, its hand check, and the share of
  affirming openers and of verbal agreement (an affirming opener or the user's letter) by
  policy and in the training completions of the 600-step floor runs, to
  ``results/analysis/affirmation.json`` and ``.md``; run with
  ``python -m goodhart.analysis.affirmation``;
- ``compute``: the GPU time of the fetched runner jobs (the sum of the ``seconds`` of their
  ``summary.json``, by stage, failed jobs included) to ``results/analysis/compute.json`` and
  ``.md``; run with ``python -m goodhart.analysis.compute``.

Run the four analyses with ``python -m goodhart.analysis`` (``__main__``); outputs go to
``results/analysis/``.
"""

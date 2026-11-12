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
- ``prereg_v3``: the registered constants of ``docs/PREREG.md`` v3 (the reply-floor arm and
  the are-you-sure evaluation);
- ``floor_arm``: the frozen stratum of the reply-floor arm's endpoint test
  (``results/analysis/floor_stratum.json``).

Run the four analyses with ``python -m goodhart.analysis`` (``__main__``); outputs go to
``results/analysis/``.
"""

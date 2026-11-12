"""CLI entry points, each a module with ``main()`` taking ``--config``.

- ``baseline``, ``best_of_n``, ``train_grpo``: the experiments (greedy baseline, Best-of-N
  selection, GRPO training with and without the mitigation);
- ``reeval``: greedy re-evaluation of the base policy and saved adapters with per-item outputs
  and a check against the training curves;
- ``are_you_sure``: two-turn greedy evaluation of the base policy and saved adapters, a
  pushback and a neutral second turn after the policy's own answer, no judge;
- ``sample_rate``: sampled replies of the base policy and saved adapters with GRPO's rollout
  settings, every sample rated;
- ``rescore``: saved replies re-rated by other judges or personas, plus the rating helpers the
  two stages above share;
- ``confidence`` and ``plots``: intervals and figures from ``results/`` (no ``--config``).

The experiment and analysis stages write a results JSON that carries the resolved config and
the git commit.
"""

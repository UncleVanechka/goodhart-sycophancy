"""Paired and clustered statistics for policies scored on the same questions.

Pure functions on plain sequences, numpy only:

- :func:`mcnemar_exact`: exact two-sided McNemar test on paired binary outcomes;
- :func:`paired_bootstrap`: percentile interval of a difference in rates, resampling
  questions so the pairing is kept;
- :func:`permutation_test`: exact permutation test of a difference in means over runs
  (Monte Carlo above ``max_exact`` splits), with the alternative always stated by the caller;
- :func:`cluster_bootstrap`: two-stage bootstrap of a difference between conditions, runs
  resampled within each condition and questions resampled once per replicate for all runs;
- :func:`clopper_pearson`: exact binomial interval;
- :func:`random_effects`: DerSimonian-Laird mean and the between-run SD ``tau``.

Every "difference" is ``second - first``: the first argument is the reference (the base
policy, or the control condition). Binary outcomes must be 0/1 or booleans; anything else
(NaN, ``None``, 2, "yes") is an error, never silently cast.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from statistics import NormalDist
from typing import Any

import numpy as np


def _log_factorials(n: int) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, n + 1, dtype=float)))))


def binom_cdf(k: int, n: int, p: float) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, summed in log space."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logf = _log_factorials(n)
    i = np.arange(0, k + 1)
    terms = logf[n] - logf[i] - logf[n - i] + i * math.log(p) + (n - i) * math.log1p(-p)
    top = float(terms.max())
    return float(min(1.0, math.exp(top) * np.exp(terms - top).sum()))


def _as_bool(values: Sequence[Any]) -> np.ndarray:
    """``values`` as a boolean array; raises on anything but 0, 1, ``True`` and ``False``."""
    out = []
    for i, value in enumerate(values):
        if isinstance(value, (str, bytes)) or value is None or value not in (0, 1):
            raise ValueError(f"binary outcome {i} is {value!r}, expected 0/1 or True/False")
        out.append(bool(value))
    return np.asarray(out, dtype=bool)


def mcnemar_exact(reference: Sequence[Any], other: Sequence[Any]) -> dict[str, Any]:
    """Exact McNemar test on aligned binary outcomes of the same items.

    ``b`` counts items true under ``reference`` and false under ``other``, ``c`` the reverse;
    the two-sided p-value is ``min(1, 2 * P(X <= min(b, c)))`` with ``X ~ Bin(b + c, 1/2)``.
    ``diff`` is the change in rate, ``(c - b) / n``.
    """
    ref, oth = _as_bool(reference), _as_bool(other)
    if ref.shape != oth.shape:
        raise ValueError(f"{ref.size} reference outcomes but {oth.size} others")
    n = int(ref.size)
    b = int(np.sum(ref & ~oth))
    c = int(np.sum(~ref & oth))
    discordant = b + c
    p_value = 1.0 if discordant == 0 else min(1.0, 2.0 * binom_cdf(min(b, c), discordant, 0.5))
    return {
        "n": n,
        "b": b,
        "c": c,
        "discordant": discordant,
        "rate_reference": float(ref.mean()) if n else math.nan,
        "rate_other": float(oth.mean()) if n else math.nan,
        "diff": (c - b) / n if n else math.nan,
        "p_value": p_value,
    }


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a binomial proportion ``k / n``, by bisection."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")
    tail = (1.0 - confidence) / 2.0

    def solve(f: Any) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if f(mid):
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    # lower: P(X >= k | p) = tail, increasing in p; upper: P(X <= k | p) = tail, decreasing in p
    lower = 0.0 if k == 0 else solve(lambda p: 1.0 - binom_cdf(k - 1, n, p) >= tail)
    upper = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) <= tail)
    return lower, upper


def _paired_arrays(reference: Sequence[float], other: Sequence[float]) -> tuple[np.ndarray, ...]:
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    if ref.shape != oth.shape or ref.ndim != 1:
        raise ValueError(f"need aligned 1-d inputs, got shapes {ref.shape} and {oth.shape}")
    keep = ~(np.isnan(ref) | np.isnan(oth))
    return ref[keep], oth[keep]


def _percentiles(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = 100.0 * (1.0 - confidence) / 2.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def paired_bootstrap(
    reference: Sequence[float],
    other: Sequence[float],
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap of ``mean(other) - mean(reference)`` over items, both values of an item drawn
    together. Pairs with a NaN on either side are dropped.

    Returns the difference, the percentile interval, the bootstrap SE and a two-sided
    bootstrap p-value (twice the smaller share of replicates on either side of zero).
    """
    ref, oth = _paired_arrays(reference, other)
    if ref.size == 0:
        return {"n": 0, "diff": math.nan, "ci": (math.nan, math.nan), "se": math.nan}
    diffs = oth - ref
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    p_value = min(1.0, 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return {
        "n": int(diffs.size),
        "diff": float(diffs.mean()),
        "ci": _percentiles(boot, confidence),
        "se": float(boot.std(ddof=1)),
        "p_value": p_value,
    }


ALTERNATIVES = ("two-sided", "greater", "less")


def permutation_test(
    reference: Sequence[float],
    other: Sequence[float],
    *,
    alternative: str,
    max_exact: int = 200_000,
    n_perm: int = 100_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Permutation test of ``mean(other) - mean(reference)`` with runs as the units.

    Every split of the pooled values into groups of the original sizes is enumerated when
    there are at most ``max_exact`` of them, else ``n_perm`` random splits are drawn (and the
    observed split is counted once). ``alternative`` has no default and must be one of
    ``two-sided``, ``greater`` (other is larger) or ``less``; it is returned with the result.

    With ``C = C(n_ref + n_other, n_other)`` splits, the smallest exact p-value is ``1 / C``
    one-sided. Two-sided it is ``2 / C`` when the groups have equal sizes, since swapping the
    groups gives every split a mirror with the same absolute difference, and can be ``1 / C``
    when they differ. Two runs per group give 1/6 one-sided and 1/3 two-sided.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")
    ref = np.asarray(reference, dtype=float)
    oth = np.asarray(other, dtype=float)
    if ref.size == 0 or oth.size == 0:
        raise ValueError("both groups need at least one value")
    pooled = np.concatenate([ref, oth])
    n, k = pooled.size, oth.size
    observed = float(oth.mean() - ref.mean())
    total = float(pooled.sum())
    tol = 1e-12 * max(1.0, abs(observed))

    def statistic(other_sum: np.ndarray | float) -> np.ndarray | float:
        return other_sum / k - (total - other_sum) / (n - k)

    def extreme(stats: np.ndarray) -> np.ndarray:
        if alternative == "greater":
            return stats >= observed - tol
        if alternative == "less":
            return stats <= observed + tol
        return np.abs(stats) >= abs(observed) - tol

    n_splits = math.comb(n, k)
    if n_splits <= max_exact:
        sums = np.fromiter(
            (pooled[list(c)].sum() for c in itertools.combinations(range(n), k)),
            dtype=float,
            count=n_splits,
        )
        hits = int(np.sum(extreme(np.asarray(statistic(sums)))))
        return {
            "diff": observed,
            "alternative": alternative,
            "p_value": hits / n_splits,
            "n_permutations": n_splits,
            "exact": True,
        }
    rng = np.random.default_rng(seed)
    sums = np.array([pooled[rng.permutation(n)[:k]].sum() for _ in range(n_perm)])
    hits = int(np.sum(extreme(np.asarray(statistic(sums)))))
    return {
        "diff": observed,
        "alternative": alternative,
        "p_value": (hits + 1) / (n_perm + 1),
        "n_permutations": n_perm,
        "exact": False,
    }


def _run_matrix(runs: Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray([np.asarray(run, dtype=float) for run in runs], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError(f"{name}: need at least one run, all over the same items")
    return matrix


def cluster_bootstrap(
    reference_runs: Sequence[Sequence[float]],
    other_runs: Sequence[Sequence[float]],
    n_boot: int = 10_000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Two-stage bootstrap of ``mean(other) - mean(reference)`` between two conditions.

    Each run is a vector of per-item outcomes over the same items in the same order. A
    replicate draws runs with replacement within each condition, then one vector of item
    indices with replacement that every run of both conditions uses, which keeps the crossed
    run x item design. A condition's value is the mean over its runs of the per-run item means.
    A single-run reference (the base policy) only contributes the item resampling.
    """
    ref = _run_matrix(reference_runs, "reference_runs")
    oth = _run_matrix(other_runs, "other_runs")
    if ref.shape[1] != oth.shape[1]:
        raise ValueError(f"runs cover {ref.shape[1]} and {oth.shape[1]} items")
    if np.isnan(ref).any() or np.isnan(oth).any():
        raise ValueError("cluster_bootstrap needs complete outcomes, found NaN")
    n_items = ref.shape[1]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        weights = np.bincount(rng.integers(0, n_items, size=n_items), minlength=n_items) / n_items
        ref_mean = (ref[rng.integers(0, ref.shape[0], size=ref.shape[0])] @ weights).mean()
        oth_mean = (oth[rng.integers(0, oth.shape[0], size=oth.shape[0])] @ weights).mean()
        boot[b] = oth_mean - ref_mean
    return {
        "diff": float(oth.mean(axis=1).mean() - ref.mean(axis=1).mean()),
        "ci": _percentiles(boot, confidence),
        "se": float(boot.std(ddof=1)),
        "n_runs": (int(ref.shape[0]), int(oth.shape[0])),
        "n_items": int(n_items),
    }


def paired_diff_variance(reference: Sequence[float], other: Sequence[float]) -> float:
    """Variance of the mean paired difference ``other - reference``: ``var(d) / n``."""
    ref, oth = _paired_arrays(reference, other)
    if ref.size < 2:
        return math.nan
    return float(np.var(oth - ref, ddof=1) / ref.size)


def random_effects(
    estimates: Sequence[float], variances: Sequence[float], confidence: float = 0.95
) -> dict[str, Any]:
    """DerSimonian-Laird random-effects mean of per-run estimates with known variances.

    Returns the pooled mean, its SE and normal interval, the between-run SD ``tau`` and
    Cochran's ``q``. A zero variance is floored at ``1e-12`` so one exact run cannot take
    all the weight silently.
    """
    y = np.asarray(estimates, dtype=float)
    v = np.maximum(np.asarray(variances, dtype=float), 1e-12)
    if y.shape != v.shape or y.size == 0:
        raise ValueError("need one variance per estimate and at least one estimate")
    w = 1.0 / v
    fixed = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - fixed) ** 2))
    k = int(y.size)
    denom = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (q - (k - 1)) / denom) if k > 1 and denom > 0 else 0.0
    w_star = 1.0 / (v + tau2)
    mean = float(np.sum(w_star * y) / np.sum(w_star))
    se = float(math.sqrt(1.0 / np.sum(w_star)))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return {
        "mean": mean,
        "se": se,
        "ci": (mean - z * se, mean + z * se),
        "tau": math.sqrt(tau2),
        "q": q,
        "k": k,
    }

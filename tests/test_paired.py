"""Paired statistics on toy data with known answers."""

import math

import numpy as np
import pytest

from goodhart.analysis.paired import (
    binom_cdf,
    clopper_pearson,
    cluster_bootstrap,
    mcnemar_exact,
    paired_bootstrap,
    paired_diff_variance,
    permutation_test,
    random_effects,
)


def _binom_cdf_direct(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


@pytest.mark.parametrize("k,n,p", [(0, 5, 0.5), (3, 10, 0.3), (7, 12, 0.9), (20, 40, 0.5)])
def test_binom_cdf_matches_direct_sum(k, n, p):
    assert binom_cdf(k, n, p) == pytest.approx(_binom_cdf_direct(k, n, p), rel=1e-10)


def test_binom_cdf_edges():
    assert binom_cdf(-1, 5, 0.5) == 0.0
    assert binom_cdf(5, 5, 0.3) == 1.0
    assert binom_cdf(2, 5, 0.0) == 1.0
    assert binom_cdf(2, 5, 1.0) == 0.0


def test_mcnemar_counts_discordant_pairs_and_exact_p():
    base = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    policy = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    out = mcnemar_exact(base, policy)
    assert (out["b"], out["c"], out["discordant"], out["n"]) == (1, 9, 10, 12)
    assert out["diff"] == pytest.approx(8 / 12)
    expected = 2 * sum(math.comb(10, i) for i in range(2)) / 2**10
    assert out["p_value"] == pytest.approx(expected)


def test_mcnemar_symmetric_and_degenerate():
    assert mcnemar_exact([1, 0], [0, 1])["p_value"] == 1.0
    same = mcnemar_exact([1, 0, 1], [1, 0, 1])
    assert same["discordant"] == 0 and same["p_value"] == 1.0 and same["diff"] == 0.0
    with pytest.raises(ValueError):
        mcnemar_exact([1, 0], [1])


def test_mcnemar_all_one_way_reaches_two_times_half_power():
    out = mcnemar_exact([0] * 10, [1] * 10)
    assert out["p_value"] == pytest.approx(2 * 0.5**10)


def test_mcnemar_accepts_binary_values_of_any_type():
    out = mcnemar_exact([True, False, np.True_, 1.0], [np.int64(0), 1, False, True])
    assert (out["b"], out["c"]) == (2, 1)


@pytest.mark.parametrize("bad", [2, -1, 0.5, math.nan, np.nan, None, "1", "yes"])
def test_mcnemar_rejects_non_binary_outcomes(bad):
    with pytest.raises(ValueError, match="binary outcome 1"):
        mcnemar_exact([1, bad, 0], [1, 1, 0])
    with pytest.raises(ValueError, match="binary outcome 1"):
        mcnemar_exact([1, 1, 0], [1, bad, 0])


def test_clopper_pearson_known_values():
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-9)
    lo, hi = clopper_pearson(10, 10)
    assert lo == pytest.approx(0.025 ** (1 / 10), abs=1e-9) and hi == 1.0
    lo, hi = clopper_pearson(5, 10)
    assert (lo, hi) == pytest.approx((0.18709, 0.81291), abs=1e-5)


def test_clopper_pearson_bounds_are_the_tail_roots():
    lo, hi = clopper_pearson(7, 30, confidence=0.9)
    assert 1 - binom_cdf(6, 30, lo) == pytest.approx(0.05, abs=1e-9)
    assert binom_cdf(7, 30, hi) == pytest.approx(0.05, abs=1e-9)
    with pytest.raises(ValueError):
        clopper_pearson(3, 2)


def test_paired_bootstrap_difference_and_interval():
    rng = np.random.default_rng(1)
    base = rng.integers(0, 2, size=400).astype(float)
    policy = base.copy()
    flip = rng.choice(400, size=80, replace=False)
    policy[flip] = 1.0
    out = paired_bootstrap(base, policy, n_boot=2000, seed=3)
    assert out["diff"] == pytest.approx(policy.mean() - base.mean())
    assert out["ci"][0] <= out["diff"] <= out["ci"][1]
    assert out["ci"][0] > 0 and out["p_value"] < 0.01
    assert paired_bootstrap(base, policy, n_boot=2000, seed=3) == out


def test_paired_bootstrap_identical_and_nan_pairs():
    same = paired_bootstrap([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], n_boot=200)
    assert same["diff"] == 0.0 and same["ci"] == (0.0, 0.0)
    out = paired_bootstrap([1.0, np.nan, 0.0], [1.0, 1.0, 1.0], n_boot=200)
    assert out["n"] == 2 and out["diff"] == pytest.approx(0.5)


def test_permutation_test_minimum_p_is_one_over_splits():
    # 3 runs of one condition at 0, 5 of another at 1: the observed split is the only extreme
    out = permutation_test([0.0, 0.0, 0.0], [1.0] * 5, alternative="greater")
    assert out["exact"] and out["n_permutations"] == 56 and out["alternative"] == "greater"
    assert out["p_value"] == pytest.approx(1 / 56)
    # unequal groups: the mirror split has a different |difference|, so two-sided is also 1/C
    two_sided = permutation_test([0.0, 0.0, 0.0], [1.0] * 5, alternative="two-sided")
    assert two_sided["p_value"] == pytest.approx(1 / 56)
    assert permutation_test([0.0, 0.0, 0.0], [1.0] * 5, alternative="less")["p_value"] == 1.0


def test_permutation_test_two_sided_minimum_is_two_over_splits_with_equal_groups():
    # 2 runs against 2: C(4, 2) = 6 splits; the observed split and its mirror are both extreme
    assert permutation_test([0.0, 0.0], [1.0, 1.0], alternative="greater")["p_value"] == (
        pytest.approx(1 / 6)
    )
    assert permutation_test([0.0, 0.0], [1.0, 1.0], alternative="two-sided")["p_value"] == (
        pytest.approx(2 / 6)
    )


def test_permutation_test_needs_an_explicit_alternative():
    with pytest.raises(TypeError):
        permutation_test([0.0, 0.0], [1.0, 1.0])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        permutation_test([0.0, 0.0], [1.0, 1.0], "greater")  # type: ignore[misc]


def test_permutation_test_no_difference_and_monte_carlo():
    out = permutation_test([0.2, 0.4], [0.4, 0.2], alternative="two-sided")
    assert out["diff"] == pytest.approx(0.0) and out["p_value"] == 1.0
    mc = permutation_test(
        list(range(10)), list(range(5, 15)), alternative="two-sided", max_exact=10, n_perm=2000
    )
    assert not mc["exact"] and 0.0 < mc["p_value"] < 0.2 and mc["alternative"] == "two-sided"
    with pytest.raises(ValueError):
        permutation_test([], [1.0], alternative="two-sided")
    with pytest.raises(ValueError):
        permutation_test([1.0], [1.0], alternative="bigger")


def test_cluster_bootstrap_constant_shift_has_a_tight_interval():
    rng = np.random.default_rng(0)
    items = rng.integers(0, 2, size=200).astype(float)
    reference = [items, items]
    other = [np.clip(items + 1, 0, 1)] * 3
    out = cluster_bootstrap(reference, other, n_boot=500, seed=1)
    assert out["diff"] == pytest.approx(1 - items.mean())
    assert out["ci"][0] <= out["diff"] <= out["ci"][1]
    assert out["n_runs"] == (2, 3) and out["n_items"] == 200


def test_cluster_bootstrap_run_spread_widens_the_interval():
    rng = np.random.default_rng(2)
    base = [rng.integers(0, 2, size=300).astype(float)]
    tight = [rng.binomial(1, 0.6, size=300).astype(float) for _ in range(4)]
    spread = [rng.binomial(1, p, size=300).astype(float) for p in (0.3, 0.5, 0.7, 0.9)]
    width = lambda out: out["ci"][1] - out["ci"][0]  # noqa: E731
    assert width(cluster_bootstrap(base, spread, n_boot=400)) > width(
        cluster_bootstrap(base, tight, n_boot=400)
    )
    assert cluster_bootstrap(base, tight, n_boot=100, seed=5) == cluster_bootstrap(
        base, tight, n_boot=100, seed=5
    )


def test_cluster_bootstrap_rejects_misaligned_runs():
    with pytest.raises(ValueError):
        cluster_bootstrap([[1.0, 0.0]], [[1.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        cluster_bootstrap([], [[1.0]])
    with pytest.raises(ValueError):
        cluster_bootstrap([[1.0, np.nan]], [[1.0, 0.0]])


def test_random_effects_homogeneous_and_heterogeneous():
    same = random_effects([0.3, 0.3, 0.3], [0.01, 0.02, 0.01])
    assert same["mean"] == pytest.approx(0.3) and same["tau"] == 0.0
    out = random_effects([0.1, 0.5, 0.9], [0.001, 0.001, 0.001])
    w = 1000.0
    q = w * ((0.1 - 0.5) ** 2 + (0.9 - 0.5) ** 2)
    tau2 = (q - 2) / (3 * w - 3 * w**2 / (3 * w))
    assert out["q"] == pytest.approx(q)
    assert out["tau"] == pytest.approx(math.sqrt(tau2))
    assert out["mean"] == pytest.approx(0.5)
    assert out["ci"][0] < 0.5 < out["ci"][1]
    single = random_effects([0.4], [0.0])
    assert single["k"] == 1 and single["tau"] == 0.0


def test_paired_diff_variance():
    assert paired_diff_variance([0, 0, 0, 0], [1, 0, 1, 0]) == pytest.approx(
        np.var([1, 0, 1, 0], ddof=1) / 4
    )
    assert math.isnan(paired_diff_variance([1.0], [0.0]))

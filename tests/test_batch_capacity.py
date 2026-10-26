"""M3 CPU oracle: exact future batch acceptance versus cheap proxies.

Single-resource, single-node indivisible placement. A future batch is known offline.
Run: python -m pytest -q test_batch_capacity.py
"""
from functools import lru_cache
from itertools import combinations, product

Q = (1, 2, 4, 8)
W = (1, 1, 2, 4)


def F(residual):
    """Weighted number of individually eligible nodes across future thresholds."""
    return sum(w * sum(r >= q for r in residual) for q, w in zip(Q, W))


def exact_count(residual, batch):
    """Exact maximal number of concurrently placeable future jobs.

    Equivalent to an integer assignment MILP with binary accept/placement
    variables, per-job at-most-one placement and per-node capacity constraints.
    Rejecting a job is allowed; this is NOT sequential online Best-Fit.
    """
    residual = tuple(residual)
    batch = tuple(batch)

    @lru_cache(None)
    def solve(j, remaining):
        if j == len(batch):
            return 0
        best = solve(j + 1, remaining)
        for i, free in enumerate(remaining):
            if free >= batch[j]:
                updated = list(remaining)
                updated[i] -= batch[j]
                best = max(best, 1 + solve(j + 1, tuple(updated)))
        return best

    return solve(0, residual)


def pairwise_proxy(residual, batch):
    """Mean offline optimal acceptance of 2-job sub-batches, not full-batch optimum."""
    pairs = list(combinations(batch, 2))
    if not pairs:
        return exact_count(residual, batch)
    return sum(exact_count(residual, pair) for pair in pairs) / len(pairs)


def test_fixed_present_value_and_capacity_proxy_reversal():
    # Starting capacities: (6,6); accepted current jobs: 5 and 1 GPU.
    # A: spread -> residual (1,5); B: pack -> residual (0,6).
    # Thus present acceptance and total residual capacity are identical.
    A, B = (1, 5), (0, 6)
    future = (2, 2, 2)
    assert sum(A) == sum(B) == 6
    assert F(A) == 5 > F(B) == 4
    assert exact_count(A, future) == 2 < exact_count(B, future) == 3
    assert pairwise_proxy(A, future) == pairwise_proxy(B, future) == 2


def test_exact_bound_and_order_independence():
    for r in ((1, 5), (0, 6), (4, 4)):
        for batch in product((1, 2, 4, 8), repeat=3):
            optimum = exact_count(r, batch)
            assert 0 <= optimum <= len(batch)
            assert optimum <= sum(r) // min(batch)
            assert optimum == exact_count(r, tuple(reversed(batch)))
            assert pairwise_proxy(r, batch) <= 2


def test_pairwise_is_not_exact_for_three():
    A, B = (1, 5), (0, 6)
    batch = (2, 2, 2)
    assert pairwise_proxy(A, batch) == pairwise_proxy(B, batch)
    assert exact_count(A, batch) != exact_count(B, batch)

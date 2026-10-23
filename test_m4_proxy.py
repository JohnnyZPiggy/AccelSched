"""M4 deterministic CPU ablation: batch-capacity proxies versus offline oracle.

Run: python -m pytest -q test_m4_proxy.py
No GPU, external datasets, or speedup claim.
"""
from collections import Counter
from itertools import combinations, product
from math import comb
from test_batch_capacity import F, exact_count, pairwise_proxy


def pk(residual, batch, k):
    """Mean exact cardinality for size-k subbatches (offline tiny oracle)."""
    if k > len(batch):
        raise ValueError("k exceeds batch length")
    return sum(exact_count(residual, b) for b in combinations(batch, k)) / comb(len(batch), k)


def histogram_upper(residual, batch):
    """Per-demand-class homogeneous bin capacity, summed: valid upper bound.

    Different classes compete for the same resources, so the sum may overcount.
    For a demand q appearing n_q times, at most min(n_q, sum_i floor(R_i/q))
    tasks of that class can ever fit. Summing is an upper bound on accepted count.
    """
    counts = Counter(batch)
    return sum(min(n, sum(r // q for r in residual)) for q, n in counts.items())


def admissible_states():
    """Same current jobs (1,2,3,4) all accepted on two 8-GPU nodes."""
    unique = set()
    for assignment in product((0, 1), repeat=4):
        rem = [8, 8]
        for job, i in zip((1, 2, 3, 4), assignment):
            rem[i] -= job
        if min(rem) >= 0:
            unique.add(tuple(sorted(rem)))
    return tuple(sorted(unique))


def evaluate():
    states = admissible_states()
    # Fixed current accepted jobs and total residual: 16-10=6.
    assert states == ((0, 6), (1, 5), (2, 4), (3, 3))
    rows = []
    for future in combinations_with_replacement4():
        oracle = [exact_count(s, future) for s in states]
        metrics = {
            "F": [F(s) for s in states],
            "P2": [pk(s, future, 2) for s in states],
            "P3": [pk(s, future, 3) for s in states],
            "hist": [histogram_upper(s, future) for s in states],
        }
        for name, vals in metrics.items():
            concordant = discordant = tied_proxy = 0
            for i, j in combinations(range(len(states)), 2):
                if oracle[i] == oracle[j]:
                    continue  # oracle cannot rank tied states
                d = (vals[i]-vals[j]) * (oracle[i]-oracle[j])
                concordant += d > 0
                discordant += d < 0
                tied_proxy += d == 0
            rows.append((future, name, concordant, discordant, tied_proxy))
            if name == "hist":
                assert all(v >= o for v, o in zip(vals, oracle))
    totals = {name: tuple(sum(row[k] for row in rows if row[1] == name) for k in (2, 3, 4))
              for name in ("F", "P2", "P3", "hist")}
    return states, totals


def combinations_with_replacement4():
    from itertools import combinations_with_replacement
    return combinations_with_replacement((1, 2, 4, 8), 4)


def test_proxy_reversal_and_pairwise_blind_spot():
    A, B, future = (1, 5), (0, 6), (2, 2, 2)
    assert F(A) == 5 > F(B) == 4
    assert exact_count(A, future) == 2 < exact_count(B, future) == 3
    assert pairwise_proxy(A, future) == pairwise_proxy(B, future) == 2
    assert pk(A, future, 3) == 2 < pk(B, future, 3) == 3
    assert histogram_upper(A, future) == 2 < histogram_upper(B, future) == 3


def test_histogram_upper_bound_exhaustive():
    states, _ = evaluate()
    assert len(states) == 4


def test_deterministic_ranking_counts():
    _, totals = evaluate()
    assert totals == EXPECTED_COUNTS


# Filled from independently executed CPU enumeration, not inferred from proxy definitions.
EXPECTED_COUNTS = {"F": (30, 6, 15), "P2": (43, 0, 8), "P3": (51, 0, 0), "hist": (45, 0, 6)}

if __name__ == "__main__":
    print(evaluate())

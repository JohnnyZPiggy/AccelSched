"""M5 CPU-only adversarial proxy ranking under identical present batch.

Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m5_proxy.py
The study uses the exact offline single-resource oracle, not online admission.
"""
from itertools import combinations, combinations_with_replacement, product
from time import perf_counter_ns
from test_batch_capacity import F, exact_count
from test_m4_proxy import pk, histogram_upper

CURRENT = (1, 2, 5)
CAPACITY = (8, 8)


def states():
    """All capacity states achievable by fully accepting the same current jobs."""
    found = set()
    for assignment in product(range(len(CAPACITY)), repeat=len(CURRENT)):
        rest = list(CAPACITY)
        for q, i in zip(CURRENT, assignment):
            rest[i] -= q
        if min(rest) >= 0:
            found.add(tuple(sorted(rest)))
    return tuple(sorted(found))


def metrics(R, batch):
    return (F(R), pk(R, batch, 2), pk(R, batch, 3), histogram_upper(R, batch))


def ranking_counts(support, length=5):
    counts = {k: [0, 0, 0] for k in ("F", "P2", "P3", "hist")}
    comparable = 0
    for batch in combinations_with_replacement(support, length):
        rr = states()
        truth = [exact_count(R, batch) for R in rr]
        scores = [metrics(R, batch) for R in rr]
        for a, b in combinations(range(len(rr)), 2):
            if truth[a] == truth[b]:
                continue
            comparable += 1
            for idx, name in enumerate(counts):
                sign = (scores[a][idx] - scores[b][idx]) * (truth[a] - truth[b])
                counts[name][0 if sign > 0 else 1 if sign < 0 else 2] += 1
        assert all(scores[i][3] >= truth[i] for i in range(len(rr)))
    assert all(sum(t) == comparable for t in counts.values())
    return comparable, {name: tuple(c) for name, c in counts.items()}


def test_exact_counterexamples():
    assert states() == ((0, 8), (1, 7), (2, 6), (3, 5))
    p3_batch = (2, 3, 3, 4, 4)
    assert (exact_count((1, 7), p3_batch), exact_count((2, 6), p3_batch)) == (2, 3)
    assert pk((1, 7), p3_batch, 3) == 2
    assert pk((2, 6), p3_batch, 3) == 1.9
    hist_batch = (1, 3, 4, 4, 6)
    assert (exact_count((2, 6), hist_batch), exact_count((3, 5), hist_batch)) == (2, 3)
    assert (histogram_upper((2, 6), hist_batch), histogram_upper((3, 5), hist_batch)) == (4, 3)


def test_distribution_ranking_counts():
    assert ranking_counts((1, 2, 4, 8)) == (39, {
        "F": (33, 0, 6), "P2": (21, 0, 18), "P3": (31, 0, 8), "hist": (33, 0, 6)})
    assert ranking_counts((2, 3, 4, 6)) == (77, {
        "F": (59, 5, 13), "P2": (50, 5, 22), "P3": (63, 2, 12), "hist": (44, 6, 27)})
    assert ranking_counts((1, 2, 3, 4, 6, 8)) == (207, {
        "F": (136, 28, 43), "P2": (159, 11, 37), "P3": (188, 3, 16), "hist": (133, 13, 61)})
    # Future batch length 5: P2 and P3 each require ten exact subproblem solves.
    assert len(tuple(combinations(range(5), 2))) == 10
    assert len(tuple(combinations(range(5), 3))) == 10


def test_multiresource_boundary():
    # The single-resource floor(R/q) expression cannot be applied
    # independently to resource totals in multidimensional placement.
    R = ((4, 1), (1, 4))
    q = (2, 2)
    assert sum(min(r[0] // q[0], r[1] // q[1]) for r in R) == 0
    assert sum(r[0] for r in R) >= 2 and sum(r[1] for r in R) >= 2


if __name__ == '__main__':
    for support in ((1, 2, 4, 8), (2, 3, 4, 6), (1, 2, 3, 4, 6, 8)):
        start = perf_counter_ns()
        print(support, ranking_counts(support), (perf_counter_ns() - start) / 1e6, 'ms')

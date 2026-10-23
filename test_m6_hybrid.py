"""M6: bounded screening and reranking over equal-present-value placements.

CPU only. Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m6_hybrid.py
The future batch is assumed known for evaluating P3 and H* (clairvoyant oracle).
"""
from itertools import combinations_with_replacement, product
from time import perf_counter_ns
from test_batch_capacity import F, exact_count
from test_m4_proxy import pk

PRESENT = (2, 2, 3, 3)
START = (8, 8, 8)
STATES = ((0, 6, 8), (1, 5, 8), (2, 4, 8), (2, 6, 6), (3, 3, 8), (3, 5, 6), (4, 5, 5))
SUPPORT = (1, 2, 3, 4, 6)


def online_count(state, future):
    """Sequential future best-fit, fixed order, no migration or repacking."""
    r = list(state)
    admitted = 0
    for demand in future:
        candidates = [i for i, cap in enumerate(r) if cap >= demand]
        if candidates:
            i = min(candidates, key=lambda i: (r[i] - demand, i))
            r[i] -= demand
            admitted += 1
    return admitted


def select(states, future, screen=4, exact_budget=0):
    """F screening (top screen), P3 rerank, H* on top exact_budget.

    Budget counts number of FULL-batch exact calls. P3 itself invokes
    C(len(future),3) additional exact SUB-batch solves per screen candidate.
    No guarantee: pruning can remove the global H* maximizer.
    """
    if not (1 <= screen <= len(states) and 0 <= exact_budget <= screen):
        raise ValueError('invalid budget')
    retained = sorted(states, key=lambda r: (-F(r), r))[:screen]
    reranked = sorted(retained, key=lambda r: (-pk(r, future, 3), r))
    finalists = reranked[:exact_budget]
    if exact_budget:
        winner = min(finalists, key=lambda r: (-exact_count(r, future), r))
    else:
        winner = reranked[0]
    return winner, retained, finalists


def exact_frontier(states, future):
    return max(exact_count(r, future) for r in states)


def study(support=SUPPORT):
    policies = ((1, 0), (3, 1), (5, 2), (7, 2), (7, 7))
    stats = {p: {'n': 0, 'regret': 0, 'screen_miss': 0, 'online': 0,
                 'offline': 0, 'elapsed_ns': 0} for p in policies}
    cheap = {'n': 0, 'regret': 0, 'online': 0, 'offline': 0}
    for future in combinations_with_replacement(support, 5):
        optimal = exact_frontier(STATES, future)
        f_state = min(STATES, key=lambda r: (-F(r), r))
        cheap['n'] += 1
        cheap['regret'] += optimal - exact_count(f_state, future)
        cheap['online'] += online_count(f_state, future)
        cheap['offline'] += exact_count(f_state, future)
        for policy in policies:
            t = perf_counter_ns()
            winner, retained, _ = select(STATES, future, *policy)
            duration = perf_counter_ns() - t
            record = stats[policy]
            record['n'] += 1
            record['regret'] += optimal - exact_count(winner, future)
            record['screen_miss'] += max(exact_count(r, future) for r in retained) < optimal
            record['online'] += online_count(winner, future)
            record['offline'] += exact_count(winner, future)
            record['elapsed_ns'] += duration
            assert record['regret'] >= 0
    return cheap, stats


def test_same_current_accepted_value_and_capacity():
    found = set()
    for assignment in product(range(len(START)), repeat=len(PRESENT)):
        r = list(START)
        for q, i in zip(PRESENT, assignment):
            r[i] -= q
        if min(r) >= 0:
            found.add(tuple(sorted(r)))
    assert tuple(sorted(found)) == STATES
    assert all(sum(r) == 14 for r in STATES)


def test_pruned_oracle_cannot_recover():
    # No matter how exact the last stage, screened-out optimal placements are lost.
    witness = next((future for future in combinations_with_replacement(SUPPORT, 5)
                    if exact_count(select(STATES, future, 1, 1)[0], future)
                    < exact_frontier(STATES, future)), None)
    assert witness is not None
    winner, retained, finalists = select(STATES, witness, screen=1, exact_budget=1)
    optimal = exact_frontier(STATES, witness)
    assert max(exact_count(r, witness) for r in retained) < optimal
    assert exact_count(winner, witness) < optimal
    assert len(finalists) == 1


def test_exhaustive_budget_accounting_and_quality():
    cheap, stats = study()
    assert cheap['n'] == 126
    for (screen, exact_budget), item in stats.items():
        assert item['n'] == 126
        assert item['regret'] >= 0
        assert item['screen_miss'] >= 0
        assert item['offline'] >= item['online']
        assert 0 <= item['screen_miss'] <= item['n']
        assert comb_subproblems(screen, 5) == screen * 10
        assert exact_budget <= screen
    assert cheap['regret'] == 21 and cheap['online'] == 493
    assert stats[(1, 0)]['regret'] == 21
    assert stats[(3, 1)]['regret'] == 0
    assert stats[(3, 1)]['online'] == 506
    assert stats[(7, 7)]['regret'] == 0
    assert stats[(7, 7)]['screen_miss'] == 0


def comb_subproblems(screen, batch_length):
    from math import comb
    return screen * comb(batch_length, 3)


if __name__ == '__main__':
    cheap, stats = study()
    print('cheap F:', cheap)
    for key, val in stats.items():
        print(key, val)

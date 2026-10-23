"""M8: distribution shift and online-oriented SAA; CPU-only research test.

Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m8_robust.py
Selection uses independent training batches; held-out batches are evaluation only.
"""
from collections import defaultdict
from time import perf_counter
import numpy as np
from test_batch_capacity import F, exact_count
from test_m6_hybrid import STATES, online_count
from test_m7_distribution import DISTRIBUTIONS, sample

TRAIN_N = 12
TEST_N = 32
REPLICATES = (0, 1, 2)


def select(policy, train):
    """Training scenario samples are keyed by distribution; never use test data."""
    if policy == "F":
        return min(STATES, key=lambda s: (-F(s), s)), 0
    if policy == "offline_saa":
        pool = train["small"]
        score = lambda s: sum(exact_count(s, b) for b in pool)
        budget = len(STATES) * len(pool)
    elif policy == "online_saa":
        pool = train["small"]
        score = lambda s: sum(online_count(s, b) for b in pool)
        budget = len(STATES) * len(pool)
    elif policy == "minimax_online":
        score = lambda s: min(sum(online_count(s, b) for b in train[d]) / len(train[d])
                              for d in sorted(train))
        budget = len(STATES) * sum(len(x) for x in train.values())
    else:
        raise ValueError(policy)
    return min(STATES, key=lambda s: (-score(s), s)), budget


def data(replicate):
    training, testing = {}, {}
    for idx, (name, probs) in enumerate(DISTRIBUTIONS.items()):
        training[name] = sample(np.random.default_rng(31000 + 100 * idx + replicate), probs, TRAIN_N)
        testing[name] = sample(np.random.default_rng(91000 + 100 * idx + replicate), probs, TEST_N)
        assert training[name] != testing[name]
    return training, testing


def study():
    rows = []
    policies = ("F", "offline_saa", "online_saa", "minimax_online")
    for rep in REPLICATES:
        train, test = data(rep)
        for policy in policies:
            start = perf_counter()
            choice, budget = select(policy, train)
            decision_ms = (perf_counter() - start) * 1000
            for distribution, batches in test.items():
                offline = float(np.mean([exact_count(choice, b) for b in batches]))
                online = float(np.mean([online_count(choice, b) for b in batches]))
                # Ex-post best fixed state, not a feasible learned baseline.
                best_fixed_online = max(float(np.mean([online_count(s, b) for b in batches])) for s in STATES)
                best_fixed_offline = max(float(np.mean([exact_count(s, b) for b in batches])) for s in STATES)
                rows.append(dict(replicate=rep, policy=policy, test=distribution,
                                 state=choice, budget=budget, decision_ms=decision_ms,
                                 offline=offline, online=online,
                                 online_regret=best_fixed_online-online,
                                 offline_regret=best_fixed_offline-offline))
                assert 0 <= online <= offline <= 5
                assert online <= best_fixed_online + 1e-9
                assert offline <= best_fixed_offline + 1e-9
    return rows


def test_independent_training_and_budget():
    train, test = data(0)
    assert all(len(train[d]) == TRAIN_N and len(test[d]) == TEST_N for d in DISTRIBUTIONS)
    assert all(train[d] != test[d] for d in DISTRIBUTIONS)
    assert all(sum(r) == 14 for r in STATES)
    for policy in ("F", "offline_saa", "online_saa", "minimax_online"):
        state, budget = select(policy, train)
        assert state in STATES
        assert budget == (0 if policy == "F" else
                          len(STATES)*TRAIN_N if policy != "minimax_online" else
                          len(STATES)*2*TRAIN_N)


def test_online_offline_not_interchangeable():
    # Distinguish exact batch-optimal packing from order-sensitive online best-fit.
    witness = (6, 3, 3, 2, 1)
    assert exact_count((0, 6, 8), witness) >= online_count((0, 6, 8), witness)
    assert any(exact_count(s, b) > online_count(s, b)
               for s in STATES for b in ((6, 3, 3, 2, 1), (2, 6, 3, 3, 1), (4, 4, 3, 2, 1)))


def test_shift_study_reproducible_without_wallclock():
    first, second = study(), study()
    assert len(first) == len(second) == 3*4*2 == 24
    keys = ("replicate", "policy", "test", "state", "budget", "offline", "online",
            "online_regret", "offline_regret")
    assert [tuple(r[k] for k in keys) for r in first] == [tuple(r[k] for k in keys) for r in second]
    assert all(r["decision_ms"] >= 0 for r in first)


if __name__ == "__main__":
    groups = defaultdict(list)
    for row in study():
        groups[(row["policy"], row["test"])].append(row)
    for key, records in sorted(groups.items()):
        print(key, {k: round(float(np.mean([v[k] for v in records])), 4)
                    for k in ("online", "offline", "online_regret", "offline_regret", "decision_ms")},
              "states", [v["state"] for v in records], "budget", records[0]["budget"])

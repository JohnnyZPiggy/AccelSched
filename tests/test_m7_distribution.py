"""M7 offline distribution-aware capacity decision, held-out future admissions.

Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m7_distribution.py
CPU-only, no PGFP modification. Future batches used to select a residual
state are sampled independently of the evaluation batches.
"""
from functools import lru_cache
from time import perf_counter
import numpy as np
from test_batch_capacity import F, exact_count
from test_m6_hybrid import STATES, online_count

DEMANDS = (1, 2, 3, 4, 6)
DISTRIBUTIONS = {
    "small": (0.45, 0.35, 0.10, 0.08, 0.02),
    "large": (0.05, 0.10, 0.15, 0.30, 0.40),
}
BATCH_SIZE = 5
TRAIN_SIZES = (8, 32)
REPLICATES = (0, 1, 2)
TEST_SIZE = 24

@lru_cache(None)
def oracle(state, batch):
    return exact_count(state, batch)


def sample(rng, probabilities, count):
    """Generate independent ordered batches; do not sort online arrivals."""
    samples = rng.choice(DEMANDS, size=(count, BATCH_SIZE), p=probabilities)
    return tuple(tuple(int(v) for v in row) for row in samples)


def static_f():
    return min(STATES, key=lambda state: (-F(state), state))


def weighted_coverage(probabilities):
    # Distribution-aware SINGLE-job candidate-node proxy, not batch acceptance.
    weights = {q: p for q, p in zip(DEMANDS, probabilities)}
    return min(STATES, key=lambda state: (-sum(p * sum(r >= q for r in state)
                                             for q, p in weights.items()), state))


def saa(batches):
    """Empirical mean exact *offline* batch acceptance; training data only."""
    return min(STATES, key=lambda state: (-sum(oracle(state, batch) for batch in batches)
                                          / len(batches), state))


def evaluate(test, state):
    return (float(np.mean([oracle(state, batch) for batch in test])),
            float(np.mean([online_count(state, batch) for batch in test])))


def experiment():
    rows = []
    for train_name, probs in DISTRIBUTIONS.items():
        for test_name, test_probs in DISTRIBUTIONS.items():
            for replicate in REPLICATES:
                # Independent training/test streams; test seed depends on TEST distribution.
                train_rng = np.random.default_rng(20261023 + 1000 * list(DISTRIBUTIONS).index(train_name) + replicate)
                test_rng = np.random.default_rng(20271023 + 1000 * list(DISTRIBUTIONS).index(test_name) + replicate)
                train = sample(train_rng, probs, max(TRAIN_SIZES))
                test = sample(test_rng, test_probs, TEST_SIZE)
                assert train is not test
                selectors = [("F", static_f, 0),
                             ("weighted_F", lambda p=probs: weighted_coverage(p), 0)]
                selectors += [(f"SAA_{n}", lambda batches=train[:n]: saa(batches), n)
                              for n in TRAIN_SIZES]
                for name, policy, budget in selectors:
                    oracle.cache_clear()  # independent solver-cache budget per decision
                    started = perf_counter()
                    state = policy()
                    decision_ms = (perf_counter()-started) * 1000
                    offline, online = evaluate(test, state)
                    upper = float(np.mean([max(oracle(s, b) for s in STATES) for b in test]))
                    assert online <= offline + 1e-9 <= upper + 1e-9
                    rows.append(dict(train=train_name, test=test_name, replicate=replicate,
                                     policy=name, state=state, samples=budget,
                                     offline=offline, online=online,
                                     clairvoyant_upper=upper, decision_ms=decision_ms))
    return rows


def test_independent_streams_and_fixed_present_reward():
    assert all(sum(r) == 14 for r in STATES)
    a = sample(np.random.default_rng(71), DISTRIBUTIONS["small"], 10)
    b = sample(np.random.default_rng(72), DISTRIBUTIONS["small"], 10)
    assert a != b
    assert len(a) == len(b) == 10


def test_saa_training_is_permutation_invariant():
    x = sample(np.random.default_rng(73), DISTRIBUTIONS["large"], 8)
    assert saa(x) == saa(tuple(reversed(x)))


def test_out_of_sample_comparison_and_bounds():
    rows = experiment()
    assert len(rows) == 2 * 2 * len(REPLICATES) * 4 == 48
    assert all(r["state"] in STATES and 0 <= r["online"] <= r["offline"]
               <= r["clairvoyant_upper"] <= BATCH_SIZE for r in rows)
    assert {r["samples"] for r in rows} == {0, 8, 32}
    # At least one policy makes a different choice under different information.
    assert len({r["state"] for r in rows}) > 1


if __name__ == "__main__":
    from collections import defaultdict
    groups = defaultdict(list)
    for row in experiment():
        groups[(row["train"], row["test"], row["policy"])].append(row)
    for key, values in sorted(groups.items()):
        print(key, "offline", round(float(np.mean([r["offline"] for r in values])), 3),
              "online", round(float(np.mean([r["online"] for r in values])), 3),
              "decision_ms_median", round(float(np.median([r["decision_ms"] for r in values])), 3))

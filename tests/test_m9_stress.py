"""M9: deterministic distribution-shift decision divergence; CPU only."""
from itertools import permutations
from time import perf_counter
import numpy as np
from test_batch_capacity import exact_count, F
from test_m6_hybrid import STATES, online_count

SCENARIOS = {"heavy": (4, 6, 3, 6, 1), "mixed": (1, 1, 3, 6, 3)}
POLICIES = ("F", "offline_saa", "online_saa", "minimax_online")
REPS = (0, 1, 2)
TEST_N = 40

def select(policy):
    if policy == "F":
        return min(STATES, key=lambda s: (-F(s), s)), 0
    if policy == "offline_saa":
        score = lambda s: sum(exact_count(s, b) for b in SCENARIOS.values()) / 2
    elif policy == "online_saa":
        score = lambda s: sum(online_count(s, b) for b in SCENARIOS.values()) / 2
    elif policy == "minimax_online":
        score = lambda s: min(online_count(s, b) for b in SCENARIOS.values())
    else:
        raise ValueError(policy)
    return min(STATES, key=lambda s: (-score(s), s)), len(STATES) * 2

def holdout(scenario, seed):
    """Seeded unseen arrival permutations; same future batches for all policies."""
    rng = np.random.default_rng(92000 + seed + (0 if scenario == "heavy" else 100))
    candidates = [b for b in sorted(set(permutations(SCENARIOS[scenario])))
                  if b not in SCENARIOS.values()]
    return tuple(candidates[int(i)] for i in rng.integers(len(candidates), size=TEST_N))

def study():
    rows = []
    for policy in POLICIES:
        start = perf_counter()
        state, budget = select(policy)
        decision_ms = (perf_counter() - start) * 1000
        for scenario in SCENARIOS:
            for rep in REPS:
                test = holdout(scenario, rep)
                online = float(np.mean([online_count(state, b) for b in test]))
                offline = float(np.mean([exact_count(state, b) for b in test]))
                best_on = max(float(np.mean([online_count(s, b) for b in test])) for s in STATES)
                best_off = max(float(np.mean([exact_count(s, b) for b in test])) for s in STATES)
                assert 0 <= online <= offline <= 5
                rows.append(dict(policy=policy, scenario=scenario, rep=rep, state=state,
                                 online=online, offline=offline, online_regret=best_on-online,
                                 offline_regret=best_off-offline, budget=budget,
                                 decision_ms=decision_ms))
    return rows

def test_three_distinct_training_choices():
    selections = {p: select(p)[0] for p in POLICIES}
    assert selections["offline_saa"] == (0, 6, 8)
    assert selections["online_saa"] == (2, 6, 6)
    assert selections["minimax_online"] == (3, 5, 6)
    assert len(set(selections.values())) >= 3
    assert all(sum(s) == 14 for s in STATES)

def test_holdout_disjoint_and_reproducible():
    for scenario in SCENARIOS:
        test = holdout(scenario, 0)
        assert test == holdout(scenario, 0)
        assert len(test) == TEST_N
        assert all(b not in SCENARIOS.values() for b in test)

def test_evaluation_regret_and_budget():
    rows = study()
    assert len(rows) == len(POLICIES) * len(SCENARIOS) * len(REPS) == 24
    assert all(r["online_regret"] >= -1e-9 and r["offline_regret"] >= -1e-9 for r in rows)
    assert {r["budget"] for r in rows} == {0, 14}

if __name__ == "__main__":
    for policy in POLICIES:
        for scenario in SCENARIOS:
            rows = [r for r in study() if r["policy"] == policy and r["scenario"] == scenario]
            print(policy, scenario, rows[0]["state"],
                  round(float(np.mean([r["online"] for r in rows])), 4),
                  round(float(np.mean([r["online_regret"] for r in rows])), 4))

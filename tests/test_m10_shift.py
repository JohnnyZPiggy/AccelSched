"""M10 held-out mixture-shift CPU ablation, fixed M9 training decisions.

Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m10_shift.py
No future test sample may affect state selection. Ordered arrivals stay ordered.
"""
from itertools import permutations
from time import perf_counter_ns
import numpy as np
from test_batch_capacity import exact_count
from test_m6_hybrid import STATES, online_count
from test_m9_stress import SCENARIOS, POLICIES, select

MIXES = (0.0, 0.25, 0.5, 0.75, 1.0)  # probability of heavy scenario
SEEDS = tuple(range(8))
TEST_N = 40


def heldout(heavy_probability, seed):
    rng = np.random.default_rng(202610230 + seed + int(heavy_probability * 100) * 1000)
    pools = {name: tuple(sorted(set(permutations(batch)) - {batch}))
             for name, batch in SCENARIOS.items()}
    result = []
    for _ in range(TEST_N):
        name = 'heavy' if rng.random() < heavy_probability else 'mixed'
        candidates = pools[name]
        result.append(candidates[int(rng.integers(len(candidates)))])
    assert all(tuple(b) not in SCENARIOS.values() for b in result)
    return tuple(result)


def decision_cost():
    """Repeated wall clock observations; no speedup promises for microtimings."""
    costs = {}
    for policy in POLICIES:
        observations = []
        for _ in range(5):
            started = perf_counter_ns()
            _, budget = select(policy)
            observations.append(perf_counter_ns() - started)
        costs[policy] = (budget, tuple(observations))
    return costs


def experiment():
    choices = {p: select(p)[0] for p in POLICIES}
    rows = []
    for mix in MIXES:
        for seed in SEEDS:
            test = heldout(mix, seed)
            all_on = {r: np.array([online_count(r, b) for b in test], dtype=float) for r in STATES}
            all_off = {r: np.array([exact_count(r, b) for b in test], dtype=float) for r in STATES}
            best_fixed_on = max(float(np.mean(v)) for v in all_on.values())
            best_fixed_off = max(float(np.mean(v)) for v in all_off.values())
            for policy, state in choices.items():
                on = float(np.mean(all_on[state]))
                off = float(np.mean(all_off[state]))
                assert on <= off + 1e-9
                rows.append(dict(mix=mix, seed=seed, policy=policy, state=state,
                                 online=on, offline=off,
                                 online_regret=best_fixed_on-on,
                                 offline_regret=best_fixed_off-off))
    return rows


def summary(rows):
    report = []
    for mix in MIXES:
        for p in POLICIES:
            subset = [r for r in rows if r['mix'] == mix and r['policy'] == p]
            report.append((mix, p, round(float(np.mean([r['online'] for r in subset])), 5),
                           round(float(np.mean([r['offline'] for r in subset])), 5),
                           round(float(np.mean([r['online_regret'] for r in subset])), 5)))
    return report


def paired_difference(rows, a='minimax_online', b='online_saa'):
    """Descriptive seed-paired mean and t-based CI, not an iid-job CI."""
    from scipy.stats import t
    result = {}
    for mix in MIXES:
        diffs = np.array([next(r['online'] for r in rows if r['mix']==mix and r['seed']==seed and r['policy']==a)
                          - next(r['online'] for r in rows if r['mix']==mix and r['seed']==seed and r['policy']==b)
                          for seed in SEEDS])
        mean = float(np.mean(diffs))
        half = float(t.ppf(0.975, len(diffs)-1) * np.std(diffs, ddof=1)/np.sqrt(len(diffs)))
        result[mix] = (mean, mean-half, mean+half)
    return result


def test_deterministic_holdout_and_training_boundary():
    assert all(sum(r)==14 for r in STATES)
    assert len(set(select(p)[0] for p in POLICIES)) == 4
    for mix in MIXES:
        a = heldout(mix, 0)
        assert len(a) == TEST_N and a == heldout(mix, 0)
        assert a != heldout(mix, 1)
    assert all(len(heldout(0, seed)) == TEST_N for seed in SEEDS)


def test_paired_shift_and_regret():
    rows = experiment()
    assert len(rows) == len(MIXES)*len(SEEDS)*len(POLICIES) == 160
    assert all(r['online_regret'] >= -1e-9 and r['offline_regret'] >= -1e-9 for r in rows)
    differences = paired_difference(rows)
    assert differences[0][0] < 0 and differences[1][0] > 0
    assert len(summary(rows)) == len(MIXES)*len(POLICIES)


def test_cost_accounts_for_oracle_kind():
    costs = decision_cost()
    assert costs['F'][0] == 0
    assert all(costs[p][0] == 14 for p in POLICIES if p != 'F')
    assert all(len(observations) == 5 and min(observations)>=0 for _, observations in costs.values())


if __name__ == '__main__':
    rows = experiment()
    for record in summary(rows):
        print(record)
    print('paired minimax minus online:', paired_difference(rows))
    print('decision ns samples',decision_cost())

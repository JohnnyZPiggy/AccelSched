"""Benchmark V2: pure versus LP-guided CPU placement; no production changes.

Run: OPENBLAS_NUM_THREADS=1 python bench_cpu_v2.py
Test: OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m0_oracle.py bench_cpu.py bench_cpu_v2.py
"""
import json
from time import perf_counter_ns
import numpy as np
from bench_cpu import first_fit
from bench_m1 import make_case
from bench_m2 import sample_future, future_admitted
from m0_oracle import greedy, shape_lp, mip_value, validate, residual, placed_value

CASES = ('balanced', 'fragmented', 'cpu_limited')
SEEDS = (11, 29, 47, 83, 131)
METHODS = ('first_fit', 'classic_best_fit', 'lp_order_best_fit', 'lp_best_fit', 'pgfp')


def pack(a, order, best_fit=True):
    """Pure physical-server selection, with externally supplied job ordering."""
    remaining = a.capacity.astype(float).copy()
    placement = [-1] * len(a.value)
    for j in order:
        candidates = [i for i in range(len(remaining)) if a.eligible[i, j]
                      and np.all(remaining[i] + 1e-10 >= a.demand[j])]
        if not candidates:
            continue
        # Existing M0 LP-guided Best-Fit uses unnormalized sum of residuals.
        i = (min(candidates, key=lambda i: (float(np.sum(remaining[i] - a.demand[j])), i))
             if best_fit else candidates[0])
        placement[j] = i
        remaining[i] -= a.demand[j]
    return placement


def lp_order(a):
    lp = shape_lp(a)
    utility = a.value[None, :] - lp['prices'] @ a.demand.T
    return sorted(range(len(a.value)), key=lambda j: (-max(utility[:, j]), j))


def placement(a, method, seed):
    if method == 'first_fit':
        return first_fit(a)
    if method == 'classic_best_fit':
        return pack(a, range(len(a.value)))
    if method == 'lp_order_best_fit':
        return pack(a, lp_order(a))
    return greedy(a, method='best_fit' if method == 'lp_best_fit' else 'pgfp', seed=seed)[0]


def evaluate(case, seed):
    a = make_case(case, seed)
    future = sample_future(1000 + seed)
    start = perf_counter_ns()
    lp_bound = shape_lp(a)['value']
    lp_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    mip = mip_value(a)
    mip_ms = (perf_counter_ns() - start) / 1e6
    rows = []
    for method in METHODS:
        start = perf_counter_ns()
        p = placement(a, method, seed)
        method_ms = (perf_counter_ns() - start) / 1e6
        assert validate(a, p)
        value = placed_value(a, p)
        assert value <= mip + 1e-6 <= lp_bound + 1e-6
        rows.append(dict(case=case, seed=seed, method=method, value=value,
                         optimal_ratio=float(value / mip) if mip > 0 else None,
                         future_admitted=future_admitted(residual(a, p), future),
                         method_ms=method_ms, lp_bound=lp_bound, mip_optimum=mip,
                         lp_reference_ms=lp_ms, mip_reference_ms=mip_ms))
    return rows


def run():
    rows = [r for case in CASES for seed in SEEDS for r in evaluate(case, seed)]
    summary = [dict(case=case, method=method,
                    mean_value=round(float(np.mean([r['value'] for r in rows if r['case']==case and r['method']==method])), 3),
                    mean_future_admitted=round(float(np.mean([r['future_admitted'] for r in rows if r['case']==case and r['method']==method])), 3),
                    median_method_ms=round(float(np.median([r['method_ms'] for r in rows if r['case']==case and r['method']==method])), 4))
               for case in CASES for method in METHODS]
    print(json.dumps(dict(records=len(rows), seeds=SEEDS, scenarios=CASES,
                          timing='end-to-end policy, independent LP/MIP reference timing',
                          lp_guided=('lp_order_best_fit', 'lp_best_fit', 'pgfp'),
                          summary=summary), indent=2))
    return rows


def test_pure_best_fit_and_order_ablation():
    a = make_case('fragmented', 11)
    assert pack(a, range(len(a.value)), best_fit=False) == first_fit(a)
    for method in METHODS:
        assert validate(a, placement(a, method, 11))


def test_v2_reproducible_and_certified():
    for case in CASES:
        left, right = evaluate(case, 11), evaluate(case, 11)
        assert len(left) == len(right) == len(METHODS)
        for x, y in zip(left, right):
            for key in ('method', 'value', 'optimal_ratio', 'future_admitted', 'lp_bound', 'mip_optimum'):
                assert x[key] == y[key]


if __name__ == '__main__':
    run()

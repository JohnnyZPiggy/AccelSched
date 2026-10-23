"""Unified synthetic CPU placement benchmark; no GPU or CI.

Run: OPENBLAS_NUM_THREADS=1 python bench_cpu.py
Smoke: OPENBLAS_NUM_THREADS=1 python -m pytest -q bench_cpu.py
Methods: literal input-order First-Fit; existing LP-guided Best-Fit/PGFP.
End-to-end method time includes LP preprocessing for LP-guided methods.
"""
import json
import platform
from time import perf_counter_ns
import numpy as np
import scipy
from bench_m1 import make_case
from bench_m2 import sample_future, future_admitted
from m0_oracle import greedy, shape_lp, mip_value, validate, residual, future_score, placed_value

CASES = ('balanced', 'fragmented', 'cpu_limited')
SEEDS = (11, 29, 47)
METHODS = ('first_fit', 'best_fit', 'pgfp')


def first_fit(a):
    """No LP: job input order, lowest-index eligible feasible physical server."""
    left = a.capacity.astype(float).copy()
    placement = [-1] * len(a.value)
    for j, demand in enumerate(a.demand):
        for i in range(len(left)):
            if a.eligible[i, j] and np.all(left[i] + 1e-10 >= demand):
                placement[j] = i
                left[i] -= demand
                break
    return placement


def evaluate(case, seed):
    a = make_case(case, seed)
    # Obtain reference bounds OUTSIDE each heuristic's wall-clock timing.
    start = perf_counter_ns()
    lp = shape_lp(a)['value']
    lp_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    mip = mip_value(a)  # status==optimal or explicit exception, no silent timeout.
    mip_ms = (perf_counter_ns() - start) / 1e6
    future = sample_future(1000 + seed)
    rows = []
    for method in METHODS:
        start = perf_counter_ns()
        if method == 'first_fit':
            placement = first_fit(a)
        else:
            placement = greedy(a, method=method, seed=seed)[0]
        elapsed_ms = (perf_counter_ns() - start) / 1e6
        assert validate(a, placement)
        value = placed_value(a, placement)
        rest = residual(a, placement)
        F = future_score(a, rest)
        accepted = future_admitted(rest, future)
        assert value <= mip + 1e-6 <= lp + 1e-6
        rows.append(dict(case=case, seed=seed, method=method, value=value,
                         fraction_optimal=round(value / mip, 6) if mip > 0 else None,
                         future_score=F, future_admitted=accepted,
                         method_ms=round(elapsed_ms, 5), lp_bound=lp, mip_optimum=mip,
                         lp_reference_ms=round(lp_ms, 5), mip_reference_ms=round(mip_ms, 5)))
    return rows


def run():
    rows = [r for case in CASES for seed in SEEDS for r in evaluate(case, seed)]
    summary = []
    for case in CASES:
        for method in METHODS:
            part = [r for r in rows if r['case'] == case and r['method'] == method]
            summary.append(dict(case=case, method=method,
                mean_value=round(float(np.mean([r['value'] for r in part])), 3),
                mean_future_admitted=round(float(np.mean([r['future_admitted'] for r in part])), 3),
                median_method_ms=round(float(np.median([r['method_ms'] for r in part])), 4)))
    report = dict(python=platform.python_version(), numpy=np.__version__,
                  scipy=scipy.__version__, source='bench_m1.make_case',
                  future='bench_m2.sample_future / future_admitted',
                  time_scope='heuristic end-to-end, reference LP/MIP separate',
                  seeds=SEEDS, synthetic_only=True, records=len(rows), summary=summary)
    print(json.dumps(report, indent=2))
    return rows


def test_first_fit_constraints_and_no_lp():
    a = make_case('fragmented', 11)
    placement = first_fit(a)
    assert validate(a, placement)
    assert len(placement) == len(a.value)


def test_reproducibility_and_certified_bounds():
    for case in CASES:
        a, b = evaluate(case, SEEDS[0]), evaluate(case, SEEDS[0])
        for x, y in zip(a, b):
            assert {k: v for k, v in x.items() if not k.endswith('_ms')} == {k: v for k, v in y.items() if not k.endswith('_ms')}
            assert x['future_admitted'] <= 12


if __name__ == '__main__':
    run()

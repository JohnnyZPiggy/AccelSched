"""Small seeded synthetic CPU comparison; run: python bench_m1.py."""
import json
import time
import platform
import numpy as np
import scipy
from m0_oracle import Instance, greedy, shape_lp, mip_value, validate

SEEDS = (11, 29, 47, 83, 131)
METHODS = ('random', 'best_fit', 'pgfp')

def make_case(case, seed):
    rng = np.random.default_rng(seed)
    s, n = 8, 10
    capacity = np.column_stack((np.full(s, 8.), np.full(s, 16.)))
    if case == 'balanced':
        capacity[:, 0] = 8
    elif case == 'fragmented':
        capacity[:, 0] = np.array([2, 4, 4, 6, 2, 6, 4, 4])
    elif case == 'cpu_limited':
        capacity[:, 1] = np.array([4, 6, 8, 4, 6, 8, 4, 6])
    else:
        raise ValueError(case)
    demand = np.column_stack((rng.choice([1, 2, 4, 8], n), rng.choice([2, 4, 8], n))).astype(float)
    values = rng.integers(1, 20, n).astype(float)
    eligible = np.ones((s, n), bool)
    # Stable instance-level heterogeneity, identical across methods.
    eligible[::3, ::4] = False
    future = np.array([[1., 2.], [2., 4.], [4., 8.], [8., 8.]])
    return Instance(capacity, demand, values, eligible, np.zeros(s, int),
                    future, np.array([1., 1., 2., 4.]), np.ones((s, 4), bool))

def run():
    rows = []
    for case in ('balanced', 'fragmented', 'cpu_limited'):
        for seed in SEEDS:
            a = make_case(case, seed)
            lp = shape_lp(a)['value']
            integer = mip_value(a)
            for method in METHODS:
                start = time.perf_counter_ns()
                p, value, future = greedy(a, method, seed=seed)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                assert validate(a, p)
                assert value <= integer + 1e-6 <= lp + 1e-6
                rows.append(dict(case=case, seed=seed, method=method,
                                 value=value, future=future, mip=integer,
                                 lp=lp, elapsed_ms=round(elapsed, 4)))
    summary = []
    for case in ('balanced', 'fragmented', 'cpu_limited'):
        for method in METHODS:
            group = [row for row in rows if row['case'] == case and row['method'] == method]
            summary.append(dict(case=case, method=method,
                                mean_value=round(float(np.mean([r['value'] for r in group])), 3),
                                mean_future=round(float(np.mean([r['future'] for r in group])), 3),
                                median_ms=round(float(np.median([r['elapsed_ms'] for r in group])), 4)))
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                          scipy=scipy.__version__, seeds=SEEDS,
                          cases=3, results=len(rows), summary=summary), indent=2))

if __name__ == '__main__':
    run()

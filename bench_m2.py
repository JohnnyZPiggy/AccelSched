"""M2 CPU study: does future placeability F predict realized future-batch admission?

Run (from repo root): OPENBLAS_NUM_THREADS=1 python bench_m2.py
No GPU, MIP overclaim, CI, or trace-derived assertions.
"""
import json
import platform
import time
import numpy as np
import scipy
from m0_oracle import Instance, greedy, residual, validate, future_score

SEEDS = tuple(range(20, 30))
METHODS = ("random", "best_fit", "pgfp")
SCALES = ((8, 10), (16, 20))
SCENARIOS = ("balanced", "fragmented", "cpu_limited")
FUTURE_BATCH = 12

def make_instance(s, n, scenario, seed):
    rng = np.random.default_rng(seed)
    capacity = np.column_stack((np.full(s, 8.), np.full(s, 16.)))
    if scenario == "fragmented":
        capacity[:, 0] = np.resize(np.array([2., 4., 6., 8.]), s)
    elif scenario == "cpu_limited":
        capacity[:, 1] = np.resize(np.array([4., 6., 8., 10.]), s)
    elif scenario != "balanced":
        raise ValueError(scenario)
    demand = np.column_stack((rng.choice([1, 2, 4, 8], n),
                              rng.choice([2, 4, 8], n))).astype(float)
    value = rng.integers(1, 20, n).astype(float)
    eligible = rng.random((s, n)) > 0.10
    future = np.array([[1., 2.], [2., 4.], [4., 8.], [8., 8.]])
    weights = np.array([1., 1., 2., 4.])
    return Instance(capacity, demand, value, eligible, np.zeros(s, dtype=int),
                    future, weights, np.ones((s, 4), dtype=bool))

def sample_future(seed):
    # Common randomized future workload for ALL algorithms on this instance.
    rng = np.random.default_rng(seed + 100000)
    gpu = rng.choice([1, 2, 4, 8], FUTURE_BATCH, p=[.2, .3, .3, .2])
    cpu = np.where(gpu == 8, 8, np.where(gpu == 4, 8, 4))
    return np.column_stack((gpu, cpu)).astype(float)

def future_admitted(rest, future_jobs):
    """Same deterministic best-fit allocator for each policy's post-batch state."""
    rest = rest.copy()
    accepted = 0
    for demand in future_jobs:
        feasible = [i for i in range(len(rest)) if np.all(rest[i] >= demand - 1e-9)]
        if not feasible:
            continue
        i = min(feasible, key=lambda i: (float(np.sum((rest[i]-demand) / np.maximum(1, np.max(rest, axis=0)))), i))
        rest[i] -= demand
        accepted += 1
    return accepted

def experiment():
    rows = []
    for s, n in SCALES:
        for scenario in SCENARIOS:
            for seed in SEEDS:
                a = make_instance(s, n, scenario, seed)
                future_jobs = sample_future(seed + 1000*s + n)
                for method in METHODS:
                    start = time.perf_counter()
                    placement, value, F = greedy(a, method=method, seed=seed)
                    elapsed_ms = (time.perf_counter()-start) * 1e3
                    assert validate(a, placement)
                    rest = residual(a, placement)
                    assert abs(F - future_score(a, rest)) < 1e-8
                    admitted = future_admitted(rest, future_jobs)
                    assert 0 <= admitted <= FUTURE_BATCH
                    rows.append(dict(servers=s, jobs=n, scenario=scenario, seed=seed,
                                     method=method, value=value, F=F,
                                     future_admitted=admitted, elapsed_ms=elapsed_ms))
    assert len(rows) == len(SCALES)*len(SCENARIOS)*len(SEEDS)*len(METHODS)
    summary = []
    for scale in SCALES:
        for scenario in SCENARIOS:
            for method in METHODS:
                q = [r for r in rows if (r["servers"],r["jobs"]) == scale
                     and r["scenario"] == scenario and r["method"] == method]
                summary.append(dict(scale=list(scale), scenario=scenario, method=method,
                                    mean_value=round(float(np.mean([r["value"] for r in q])), 3),
                                    mean_F=round(float(np.mean([r["F"] for r in q])), 3),
                                    mean_future_admitted=round(float(np.mean([r["future_admitted"] for r in q])), 3),
                                    median_ms=round(float(np.median([r["elapsed_ms"] for r in q])), 4)))
    # Correlation is exploratory, confounded by current accepted-value and scenario.
    valuesF = np.array([r["F"] for r in rows])
    valuesA = np.array([r["future_admitted"] for r in rows])
    corr = float(np.corrcoef(valuesF,valuesA)[0,1]) if np.std(valuesF)>0 and np.std(valuesA)>0 else None
    report = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  synthetic_only=True, future_scheduler="deterministic_normalized_best_fit",
                  seeds=SEEDS, records=len(rows), future_batch=FUTURE_BATCH,
                  exploratory_pearson_F_vs_acceptance=corr, summary=summary)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    experiment()

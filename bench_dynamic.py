"""Bounded SYNTHETIC rolling experiment; run from this checkout, not a wheel.

JSON stdout contains all manifests, metrics and batch records. No file writes,
downloads, solver modifications or forecasts. The independent audit lives in
test_dynamic_bench.py and uses only NumPy/stdlib at experiment runtime.
"""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import platform
from time import perf_counter

import numpy as np
import scipy
from accelsched import METHODS
from rolling import Job, simulate
from test_dynamic_bench import audit


T = 40
FAMILIES = ('T4', 'P100', 'V100', 'V100M32')
NODE_FAMILIES = tuple(f for f in FAMILIES for _ in range(2))
CAPACITY = np.array([[4, 32, 128], [4, 32, 128],
                     [4, 48, 192], [4, 48, 192],
                     [8, 64, 256], [8, 64, 256],
                     [8, 96, 384], [8, 96, 384]], dtype=float)
# All 12 catalogue types fit at least one node of their requested family.
OMEGA = tuple((f, (g, 4, 16)) for f in FAMILIES for g in (1, 2, 4))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def generate(seed, target):
    """PCG64 sequential draws until full-duration offered GPU work >= target.

    Arrival, duration-class, duration, GPU, family, CPU/GPU, RAM/GPU are drawn
    in that order. Overshoot <=40/(40*48); no final job repair or selection.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    jobs, requests, work = [], [], 0
    while work < target*T*CAPACITY[:, 0].sum():
        arrival = int(rng.integers(T))
        long = int(rng.integers(2))
        duration = int(rng.integers(6, 11) if long else rng.integers(1, 4))
        gpu = int(rng.choice([1, 2, 4]))
        family = FAMILIES[int(rng.integers(4))]
        cpu = gpu*int(rng.choice([2, 4, 8]))
        ram = gpu*int(rng.choice([8, 16, 32]))
        jobs.append(Job(f'j{len(jobs):04d}', arrival, duration, (gpu, cpu, ram)))
        requests.append(family)
        work += gpu*duration
    eligible = np.array([[f == q for q in requests] for f in NODE_FAMILIES], bool)
    offered = sum((np.array(j.demand)*j.duration for j in jobs), np.zeros(3))
    load = offered/(T*CAPACITY.sum(axis=0))
    assert abs(float(load[0])-target) <= .10, 'generator BLOCKED: target tolerance'
    manifest = dict(label='SYNTHETIC', seed=seed, target=target, T=T,
                    node_families=NODE_FAMILIES, capacity=CAPACITY.tolist(),
                    jobs=[dict(**asdict(j), family=q) for j, q in zip(jobs, requests)],
                    eligible=eligible.tolist(), omega=OMEGA)
    impossible = [j.id for k, j in enumerate(jobs)
                  if not np.any(eligible[:, k] & np.all(CAPACITY >= j.demand, axis=1))]
    info = dict(input_sha256=digest(manifest), count=len(jobs), offered_load=load.tolist(),
                family_gpu_load={f: sum(j.demand[0]*j.duration for j, q in zip(jobs, requests)
                                        if q == f)/(T*sum(CAPACITY[i, 0] for i, q in
                                                          enumerate(NODE_FAMILIES) if q == f))
                                 for f in FAMILIES},
                family_counts=dict(Counter(requests)),
                gpu_counts=dict(Counter(str(j.demand[0]) for j in jobs)),
                duration_counts=dict(Counter('short' if j.duration <= 3 else 'long' for j in jobs)),
                arrivals=dict(sorted(Counter(j.arrival for j in jobs).items())),
                individually_impossible=impossible)
    return jobs, eligible, manifest, info


def run():
    started = perf_counter()
    output = dict(label='SYNTHETIC', status='PARTIAL', budget_seconds=480,
                  solver_seed=11, batch_time_limit=.5, max_runs=54,
                  environment=dict(python=platform.python_version(), numpy=np.__version__,
                                   scipy=scipy.__version__, platform=platform.platform()),
                  cases=[], runs=0)
    try:
        for seed, target in [(s, r) for r in (.3, .7, 1.1) for s in (11, 29, 47)]:
            jobs, eligible, manifest, info = generate(seed, target)
            case = dict(manifest=manifest, workload=info, results=[])
            output['cases'].append(case)
            for method in METHODS:
                # Reserve the worst allowed 40 MIP batches plus overhead before
                # starting another call; no retries or partial subprocess kills.
                if perf_counter()-started > 450:
                    raise TimeoutError('budget reserve: remaining cases NOT RUN')
                output['runs'] += 1
                began = perf_counter()
                report = simulate(CAPACITY, jobs, eligible, method=method,
                                  seed=11, time_limit=.5)
                row = dict(method=method, wall_ms=1000*(perf_counter()-began),
                           report_sha256=digest(report), report=report)
                case['results'].append(row)
                row['metrics'] = audit(CAPACITY, NODE_FAMILIES, jobs,
                                       manifest['jobs'], eligible, report, OMEGA, T, method)
                actual = dict(manifest, capacity=CAPACITY.tolist(), eligible=eligible.tolist(),
                              jobs=[dict(**asdict(j), family=q['family'])
                                    for j, q in zip(jobs, manifest['jobs'])])
                assert digest(actual) == info['input_sha256'], 'input mutation'
                assert report['solver_error_count'] == 0, 'NOT COMPARABLE: solver_error'
                if perf_counter()-started > 480:
                    raise TimeoutError('480 s budget exceeded; remaining cases NOT RUN')
            if len(output['cases']) == 1:
                output['smoke'] = 'PASS: seed 11 target 0.3, six audited methods'
        output['status'] = 'PASS'
    except Exception as exc:
        output['failure'] = f'{type(exc).__name__}: {exc}'
    output['elapsed_seconds'] = perf_counter()-started
    print(json.dumps(output, sort_keys=True, allow_nan=False))
    return 0 if output['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(run())

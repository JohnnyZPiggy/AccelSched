"""V3: heterogeneous CPU baseline with separated LP and placement timings.

Run: OPENBLAS_NUM_THREADS=1 python bench_cpu_v3.py
Tests: OPENBLAS_NUM_THREADS=1 python -m pytest -q bench_cpu_v3.py
CPU only; MIP failures remain explicit uncertified statuses.
"""
import json
from time import perf_counter_ns
import numpy as np
import m0_oracle as core
import bench_cpu_v2 as v2
from bench_m2 import make_instance, sample_future, future_admitted

SCALES = ((8, 10), (12, 16))
CASES = ('hetero_gpu', 'hetero_cpu')
SEEDS = (11, 29)
METHODS = v2.METHODS


def instance(s, n, case, seed):
    a = make_instance(s, n, 'fragmented' if case == 'hetero_gpu' else 'cpu_limited', seed)
    if case == 'hetero_gpu':
        a.capacity[:, 1] = np.resize([6., 10., 14., 18.], s)
    elif case == 'hetero_cpu':
        a.capacity[:, 0] = np.resize([2., 4., 6., 8.], s)
    else:
        raise ValueError(case)
    # Two shape classes genuinely vary pooled resource capacities.
    a.shapes = np.arange(s) % 2
    return a


def timed_policy(a, method, seed):
    """Instrument existing LP calls, without changing M0/V2 production code."""
    lp_ns = 0
    calls = 0
    original_core, original_v2 = core.shape_lp, v2.shape_lp
    def measured(x):
        nonlocal lp_ns, calls
        start = perf_counter_ns()
        try:
            return original_core(x)
        finally:
            lp_ns += perf_counter_ns() - start
            calls += 1
    core.shape_lp = measured
    v2.shape_lp = measured
    started = perf_counter_ns()
    try:
        placement = v2.placement(a, method, seed)
        status, error = 'ok', None
    except (RuntimeError, ValueError, AssertionError) as exc:
        placement, status, error = None, 'failed', str(exc)
    finally:
        total_ns = perf_counter_ns() - started
        core.shape_lp, v2.shape_lp = original_core, original_v2
    return placement, status, error, dict(total_ms=total_ns / 1e6,
        lp_ms=lp_ns / 1e6, placement_ms=max(0, (total_ns - lp_ns) / 1e6), lp_calls=calls)


def evaluate(s, n, case, seed):
    a = instance(s, n, case, seed)
    future = sample_future(seed + 1000*s + n)
    start = perf_counter_ns()
    try:
        lp = core.shape_lp(a)['value']
        lp_status, lp_error = 'optimal', None
    except (RuntimeError, ValueError, AssertionError) as exc:
        lp, lp_status, lp_error = None, 'failed', str(exc)
    ref_lp_ms = (perf_counter_ns() - start) / 1e6
    start = perf_counter_ns()
    try:
        mip = core.mip_value(a)
        mip_status, mip_error = 'optimal', None
    except RuntimeError as exc:
        mip, mip_status, mip_error = None, 'uncertified', str(exc)
    ref_mip_ms = (perf_counter_ns() - start) / 1e6
    rows = []
    for method in METHODS:
        placement, status, error, clock = timed_policy(a, method, seed)
        value, future_count = None, None
        if status == 'ok':
            if not core.validate(a, placement):
                status, error = 'invalid', 'independent validator rejected placement'
            else:
                value = core.placed_value(a, placement)
                future_count = future_admitted(core.residual(a, placement), future)
                if lp is not None:
                    assert value <= lp + 1e-6
                if mip is not None:
                    assert value <= mip + 1e-6 and mip <= lp + 1e-6
        rows.append(dict(scale=(s, n), case=case, seed=seed, method=method,
                         status=status, error=error, value=value, future_admitted=future_count,
                         mip_optimum=mip, mip_status=mip_status, mip_error=mip_error,
                         lp_bound=lp, lp_status=lp_status, lp_error=lp_error,
                         lp_ref_ms=ref_lp_ms, mip_ref_ms=ref_mip_ms, **clock))
    return rows


def run():
    rows = [r for s,n in SCALES for case in CASES for seed in SEEDS
            for r in evaluate(s,n,case,seed)]
    assert len(rows) == len(SCALES)*len(CASES)*len(SEEDS)*len(METHODS)
    print(json.dumps(dict(records=len(rows), scales=SCALES, cases=CASES,
                          seeds=SEEDS, methods=METHODS, rows=rows), indent=2))
    return rows


def test_heterogeneity_and_policy_timing():
    for case in CASES:
        a = instance(8, 10, case, SEEDS[0])
        assert np.unique(a.capacity[:, 0]).size > 1
        assert np.unique(a.capacity[:, 1]).size > 1
        for method in METHODS:
            p, status, _, clock = timed_policy(a, method, SEEDS[0])
            assert status == 'ok' and core.validate(a, p)
            assert clock['total_ms'] >= clock['lp_ms'] >= 0
            assert clock['lp_calls'] == (0 if method in ('first_fit','classic_best_fit') else 1)


def test_bounds_and_repeatability():
    for case in CASES:
        a, b = evaluate(8,10,case,11), evaluate(8,10,case,11)
        assert [(x['method'], x['value'], x['future_admitted'], x['mip_status']) for x in a] == [
            (x['method'], x['value'], x['future_admitted'], x['mip_status']) for x in b]
        assert all(x['status']=='ok' for x in a)

if __name__ == '__main__':
    run()

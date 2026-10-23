"""V3.1 CPU benchmark: true capacity shapes and explicit timing stages.

OPENBLAS_NUM_THREADS=1 python bench_cpu_v31.py
OPENBLAS_NUM_THREADS=1 python -m pytest -q bench_cpu_v31.py
No global monkeypatch and no changes to production M0/PGFP.
"""
import json
from time import perf_counter_ns
import numpy as np
import m0_oracle as core
import bench_cpu_v2 as v2
import bench_cpu_v3 as v3
from bench_m2 import sample_future, future_admitted

SCALES, CASES, SEEDS, METHODS = v3.SCALES, v3.CASES, v3.SEEDS, v3.METHODS


def instance(s, n, case, seed):
    a = v3.instance(s, n, case, seed)
    # Identical capacity vectors receive identical shape labels.
    shapes = {tuple(v): i for i, v in enumerate(sorted(set(map(tuple, a.capacity))))}
    a.shapes = np.array([shapes[tuple(v)] for v in a.capacity], dtype=int)
    assert all(np.array_equal(a.capacity[i], a.capacity[j]) for i in range(s)
               for j in range(s) if a.shapes[i] == a.shapes[j])
    return a


def explicit_policy(a, method, seed):
    """Exact M0/V2 policy decisions, with explicit stage times (no monkeypatch)."""
    times = dict(lp_ms=0., order_ms=0., placement_ms=0., total_ms=0., lp_calls=0)
    total_start = perf_counter_ns()
    try:
        if method in ('first_fit', 'classic_best_fit'):
            t = perf_counter_ns()
            placement = v2.first_fit(a) if method == 'first_fit' else v2.pack(a, range(len(a.value)))
            times['placement_ms'] = (perf_counter_ns()-t)/1e6
        else:
            t = perf_counter_ns()
            lp = core.shape_lp(a)
            times['lp_ms'] = (perf_counter_ns()-t)/1e6
            times['lp_calls'] = 1
            t = perf_counter_ns()
            utility = a.value[None, :] - lp['prices'] @ a.demand.T
            order = sorted(range(len(a.value)), key=lambda j: (-max(utility[:, j]), j))
            if method != 'lp_order_best_fit':
                shape_order = {j: sorted(range(len(lp['shapes'])), key=lambda h: (-utility[h, j], h))
                               for j in order}
            times['order_ms'] = (perf_counter_ns()-t)/1e6
            t = perf_counter_ns()
            if method == 'lp_order_best_fit':
                placement = v2.pack(a, order)
            else:
                rest = a.capacity.astype(float).copy()
                placement = [-1]*len(a.value)
                for j in order:
                    for h in shape_order[j]:
                        candidates = [i for i in range(len(rest))
                                      if a.shapes[i] == lp['shapes'][h] and a.eligible[i,j]
                                      and np.all(rest[i]+1e-10 >= a.demand[j])]
                        if not candidates:
                            continue
                        if method == 'lp_best_fit':
                            i = min(candidates, key=lambda i: (float(np.sum(rest[i]-a.demand[j])), i))
                        elif method == 'pgfp':
                            i = max(candidates, key=lambda i: (core.marginal(a,rest,i,j),-i))
                        else:
                            raise ValueError(method)
                        placement[j] = i
                        rest[i] -= a.demand[j]
                        break
            times['placement_ms'] = (perf_counter_ns()-t)/1e6
        status, error = 'ok', None
    except (RuntimeError, ValueError, AssertionError) as exc:
        placement, status, error = None, 'failed', str(exc)
    finally:
        times['total_ms'] = (perf_counter_ns()-total_start)/1e6
    return placement, status, error, times


def evaluate(s,n,case,seed):
    a = instance(s,n,case,seed)
    test = sample_future(seed+1000*s+n)
    start = perf_counter_ns()
    try:
        lp_bound = core.shape_lp(a)['value']
        lp_status, lp_error = 'optimal', None
    except (RuntimeError, ValueError, AssertionError) as exc:
        lp_bound, lp_status, lp_error = None, 'failed', str(exc)
    lp_ref_ms = (perf_counter_ns()-start)/1e6
    start = perf_counter_ns()
    try:
        mip = core.mip_value(a)
        mip_status, mip_error = 'optimal', None
    except RuntimeError as exc:
        mip, mip_status, mip_error = None, 'uncertified', str(exc)
    mip_ref_ms = (perf_counter_ns()-start)/1e6
    rows=[]
    for method in METHODS:
        p, status, error, timing = explicit_policy(a,method,seed)
        value, future_count = None, None
        if status == 'ok':
            if not core.validate(a,p):
                status, error = 'invalid', 'independent validator failed'
            else:
                value = core.placed_value(a,p)
                future_count = future_admitted(core.residual(a,p),test)
                if lp_bound is not None:
                    assert value <= lp_bound+1e-6
                if mip is not None:
                    assert lp_bound is not None and value <= mip+1e-6 and mip <= lp_bound+1e-6
        rows.append(dict(scale=(s,n),case=case,seed=seed,method=method,
                         status=status,error=error,value=value,future_admitted=future_count,
                         lp_bound=lp_bound,lp_status=lp_status,lp_error=lp_error,
                         mip_optimum=mip,mip_status=mip_status,mip_error=mip_error,
                         lp_ref_ms=lp_ref_ms,mip_ref_ms=mip_ref_ms,**timing))
    return rows


def run():
    rows = [r for s,n in SCALES for case in CASES for seed in SEEDS for r in evaluate(s,n,case,seed)]
    print(json.dumps(dict(records=len(rows),shapes='identical resource capacities',
                          timing='explicit LP/order/placement, reference LP/MIP excluded',rows=rows),indent=2))
    return rows


def test_shape_equivalence_and_policy_exactness():
    for s,n in SCALES:
        for case in CASES:
            for seed in SEEDS:
                a = instance(s,n,case,seed)
                for i in range(s):
                    for j in range(s):
                        if a.shapes[i] == a.shapes[j]:
                            assert np.array_equal(a.capacity[i],a.capacity[j])
                for method in METHODS:
                    p,status,_,times = explicit_policy(a,method,seed)
                    assert status == 'ok' and core.validate(a,p)
                    assert p == v2.placement(a,method,seed)
                    assert times['lp_calls'] == (0 if method in ('first_fit','classic_best_fit') else 1)
                    assert all(times[k] >= 0 for k in ('lp_ms','order_ms','placement_ms','total_ms'))
                    assert times['total_ms']+1e-6 >= times['lp_ms']+times['order_ms']+times['placement_ms']


def test_reference_and_repeated_outputs():
    a,b=evaluate(8,10,'hetero_gpu',11),evaluate(8,10,'hetero_gpu',11)
    for x,y in zip(a,b):
        assert (x['method'],x['status'],x['value'],x['future_admitted']) == (y['method'],y['status'],y['value'],y['future_admitted'])
        assert x['mip_status'] == y['mip_status'] == 'optimal'
        assert x['lp_status'] == y['lp_status'] == 'optimal'


if __name__ == '__main__':
    run()

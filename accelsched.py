"""AccelSched v0.1: small, CPU-only single-node batch placement API.

Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q accelsched.py
No external scheduler, GPU, or production PGFP modifications.
"""
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy_backend import solve_mip

from m0_oracle import Instance, greedy, shape_lp, placed_value, residual, validate

def first_fit(a):
    """Input-order First-Fit without LP; follows the established benchmark contract."""
    left = a.capacity.astype(float).copy()
    placement = [-1] * len(a.value)
    for j, demand in enumerate(a.demand):
        for i in range(len(left)):
            if a.eligible[i, j] and np.all(left[i] + 1e-10 >= demand):
                placement[j] = i
                left[i] -= demand
                break
    return placement


def pack(a, order):
    """Global physical-node Best-Fit, with deterministic tie-breaking."""
    remaining = a.capacity.astype(float).copy()
    placement = [-1] * len(a.value)
    for j in order:
        candidates = [i for i in range(len(remaining)) if a.eligible[i, j]
                      and np.all(remaining[i] + 1e-10 >= a.demand[j])]
        if not candidates:
            continue
        i = min(candidates, key=lambda i: (float(np.sum(remaining[i] - a.demand[j])), i))
        placement[j] = i
        remaining[i] -= a.demand[j]
    return placement


def lp_order(a):
    """Order by shape-LP reduced utility, then stable original job index."""
    lp = shape_lp(a)
    utility = a.value[None, :] - lp['prices'] @ a.demand.T
    return sorted(range(len(a.value)), key=lambda j: (-max(utility[:, j]), j))


METHODS = ('first_fit', 'classic_best_fit', 'lp_order_best_fit',
           'lp_best_fit', 'pgfp', 'highs_mip')


@dataclass(frozen=True)
class SolveResult:
    method: str
    status: str
    placement: tuple | None
    value: float | None
    residual: np.ndarray | None
    elapsed_ms: float
    optimal: bool
    error: str | None = None


def _input_error(a, time_limit, seed):
    """Validate the public instance contract before calling any solver."""
    if not isinstance(a, Instance):
        return 'instance must be an Instance'
    for key, rank in (('capacity', 2), ('demand', 2), ('value', 1),
                      ('eligible', 2), ('shapes', 1), ('future', 2),
                      ('weights', 1), ('future_eligible', 2)):
        arr = getattr(a, key)
        if not isinstance(arr, np.ndarray) or arr.ndim != rank:
            return f'{key} must be a {rank}-D ndarray'
    s, r = a.capacity.shape
    n = a.demand.shape[0]
    k = a.future.shape[0]
    if min(s, r, n, k) == 0:
        return 'servers, resources, jobs and future types must be nonempty'
    expected = {'demand': (n, r), 'value': (n,), 'eligible': (s, n),
                'shapes': (s,), 'future': (k, r), 'weights': (k,),
                'future_eligible': (s, k)}
    for name, shape in expected.items():
        if getattr(a, name).shape != shape:
            return f'{name} must have shape {shape}'
    for name in ('capacity', 'demand', 'value', 'future', 'weights'):
        x = getattr(a, name)
        if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
            return f'{name} must be real numeric'
        if not np.all(np.isfinite(x)) or np.any(x < 0):
            return f'{name} must be finite and nonnegative'
    for name in ('eligible', 'future_eligible'):
        if getattr(a, name).dtype != np.dtype(bool):
            return f'{name} must be boolean'
    if not np.issubdtype(a.shapes.dtype, np.integer) or np.any(a.shapes < 0):
        return 'shapes must be nonnegative integer labels'
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        return 'seed must be a nonnegative integer'
    if isinstance(time_limit, (bool, np.bool_)) or not isinstance(time_limit, (int, float, np.integer, np.floating)):
        return 'time_limit must be a positive finite number'
    if not np.isfinite(time_limit) or time_limit <= 0:
        return 'time_limit must be a positive finite number'
    return None


def _highs_placement(a, time_limit):
    """Compatibility shim for the physical-server SciPy backend."""
    return solve_mip(a, time_limit)


def solve(instance: Instance, method='pgfp', *, seed=2026, time_limit=10.0):
    """Return a validated placement; only HiGHS status 0 certifies optimality.

    Heuristics do not call an independent MIP to claim optimality. Their wall
    clock includes their own LP preprocessing where used; validation is outside.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}; expected one of {METHODS}')
    started = perf_counter_ns()
    error = _input_error(instance, time_limit, seed)
    if error is not None:
        return SolveResult(method, 'input_invalid', None, None, None,
                           (perf_counter_ns()-started)/1e6, False, error)
    try:
        if method == 'first_fit':
            p = first_fit(instance)
        elif method == 'classic_best_fit':
            p = pack(instance, range(len(instance.value)))
        elif method == 'lp_order_best_fit':
            p = pack(instance, lp_order(instance))
        elif method in ('lp_best_fit', 'pgfp'):
            p = greedy(instance, method='best_fit' if method == 'lp_best_fit' else 'pgfp', seed=seed)[0]
        else:
            p, status, message = _highs_placement(instance, time_limit)
            elapsed = (perf_counter_ns() - started) / 1e6
            if p is None:
                return SolveResult(method, status, None, None, None, elapsed, False, message)
            return SolveResult(method, status, p, placed_value(instance, p),
                               residual(instance, p), elapsed, status == 'optimal', message)
        elapsed = (perf_counter_ns() - started) / 1e6
        if not validate(instance, p):
            return SolveResult(method, 'invalid', None, None, None, elapsed, False,
                               'heuristic placement fails independent validation')
        p = tuple(p)
        return SolveResult(method, 'feasible', p, placed_value(instance, p),
                           residual(instance, p), elapsed, False)
    except (RuntimeError, ValueError, AssertionError) as exc:
        elapsed = (perf_counter_ns() - started) / 1e6
        return SolveResult(method, 'failed', None, None, None, elapsed, False, str(exc))


def test_all_methods_and_mip_certificate():
    from bench_m1 import make_case
    for case in ('balanced', 'fragmented', 'cpu_limited'):
        a = make_case(case, 11)
        optimum = solve(a, 'highs_mip')
        assert optimum.status == 'optimal' and optimum.optimal
        for method in METHODS:
            out = solve(a, method)
            assert out.placement is not None and validate(a, out.placement)
            assert out.value <= optimum.value + 1e-6
            assert np.allclose(out.residual, residual(a, out.placement))
            assert out.optimal == (method == 'highs_mip')


def test_heuristics_match_verified_existing_interfaces():
    from bench_m1 import make_case
    from bench_cpu_v2 import placement
    for case in ('balanced', 'fragmented', 'cpu_limited'):
        a = make_case(case, 29)
        for method in METHODS[:-1]:
            assert solve(a, method, seed=29).placement == tuple(placement(a, method, 29))


def test_invalid_options_are_not_silent():
    from m0_oracle import packing_case
    a = packing_case()
    try:
        solve(a, 'missing')
    except ValueError:
        pass
    else:
        raise AssertionError('unrecognized method did not raise')
    assert solve(a, 'highs_mip').status == 'optimal'


def test_contract_rejects_bad_inputs():
    from dataclasses import replace
    from m0_oracle import packing_case
    a = packing_case()
    bad = [replace(a, capacity=np.array([[np.nan], [8.]])),
           replace(a, demand=-a.demand),
           replace(a, value=np.array([np.inf, 10.])),
           replace(a, eligible=a.eligible.astype(int)),
           replace(a, shapes=np.array([0, -1])),
           replace(a, future=np.zeros((1, 2))),
           replace(a, future_eligible=np.zeros((1, 1), dtype=bool)),
           replace(a, weights=np.array([-1.])),
           replace(a, capacity=np.zeros((0, 1))) ]
    for invalid in bad:
        for method in METHODS:
            result = solve(invalid, method)
            assert result.status == 'input_invalid' and result.placement is None
            assert not result.optimal and result.error
    for time_limit in (0, -1, float('nan'), float('inf'), '10'):
        assert solve(a, 'first_fit', time_limit=time_limit).status == 'input_invalid'
    assert solve(a, 'first_fit', seed=-1).status == 'input_invalid'


def test_mip_incumbent_status_contract(monkeypatch):
    from types import SimpleNamespace
    from m0_oracle import packing_case
    a = packing_case()
    x = np.array([1., 0., 0., 1.])
    monkeypatch.setattr('scipy_backend.milp', lambda *args, **kwargs:
                        SimpleNamespace(x=x, status=1, message='time limit'))
    out = solve(a, 'highs_mip')
    assert out.status == 'feasible_uncertified' and not out.optimal
    assert out.placement == (0, 1) and out.value == 20
    monkeypatch.setattr('scipy_backend.milp', lambda *args, **kwargs:
                        SimpleNamespace(x=None, status=1, message='time limit'))
    out = solve(a, 'highs_mip')
    assert out.status == 'uncertified' and not out.optimal and out.placement is None

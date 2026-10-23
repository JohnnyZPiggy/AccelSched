"""AccelSched v0.1: small, CPU-only single-node batch placement API.

Run: OPENBLAS_NUM_THREADS=1 python -m pytest -q accelsched.py
No external scheduler, GPU, or production PGFP modifications.
"""
from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from m0_oracle import Instance, greedy, placed_value, residual, validate
from bench_cpu import first_fit
from bench_cpu_v2 import pack, lp_order

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


def _highs_placement(a, time_limit):
    """Physical-node binary MIP with eligible arcs and exact assignment bounds."""
    s, n = a.eligible.shape
    r = a.demand.shape[1]
    rows, ub = [], []
    for j in range(n):
        row = np.zeros((s, n)); row[:, j] = 1
        rows.append(row.ravel()); ub.append(1.)
    for i in range(s):
        for k in range(r):
            row = np.zeros((s, n)); row[i, :] = a.demand[:, k]
            rows.append(row.ravel()); ub.append(a.capacity[i, k])
    out = milp(-np.tile(a.value, s), integrality=np.ones(s*n),
               bounds=Bounds(np.zeros(s*n), a.eligible.ravel().astype(float)),
               constraints=LinearConstraint(np.array(rows), -np.inf, np.array(ub)),
               options={'time_limit': time_limit})
    if out.x is None:
        return None, 'uncertified', out.message
    x = out.x.reshape(s, n)
    if np.max(np.abs(x - np.round(x))) > 1e-6:
        return None, 'uncertified', 'MIP returned nonintegral incumbent'
    placement = tuple(int(np.argmax(x[:, j])) if np.max(x[:, j]) > 0.5 else -1
                      for j in range(n))
    if not validate(a, placement):
        return None, 'invalid', 'MIP incumbent fails independent validation'
    return placement, ('optimal' if out.status == 0 else 'feasible_uncertified'), (None if out.status == 0 else out.message)


def solve(instance: Instance, method='pgfp', *, seed=2026, time_limit=10.0):
    """Return a validated placement; only HiGHS status 0 certifies optimality.

    Heuristics do not call an independent MIP to claim optimality. Their wall
    clock includes their own LP preprocessing where used; validation is outside.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}; expected one of {METHODS}')
    if time_limit <= 0:
        raise ValueError('time_limit must be positive')
    started = perf_counter_ns()
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

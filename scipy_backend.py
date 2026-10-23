"""Minimal SciPy/HiGHS backends: LP relaxation and physical-node MIP."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from m0_oracle import shape_lp, validate


def solve_lp(a):
    """Return the existing shape-level LP relaxation (never a placement)."""
    return shape_lp(a)


def solve_mip(a, time_limit):
    """Return validated integer placement, proof status and solver message."""
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


def test_lp_mip_backends_and_proof():
    from m0_oracle import shape_lp, mip_value
    from bench_m1 import make_case
    for case in ('balanced', 'fragmented', 'cpu_limited'):
        a = make_case(case, 11)
        lp = solve_lp(a)
        assert np.isclose(lp['value'], shape_lp(a)['value'])
        p, status, error = solve_mip(a, 10.)
        assert status == 'optimal' and error is None and validate(a, p)
        value = sum(a.value[j] for j, i in enumerate(p) if i >= 0)
        assert np.isclose(value, mip_value(a)) and value <= lp['value'] + 1e-6


def test_mip_uncertified_status(monkeypatch):
    from types import SimpleNamespace
    from m0_oracle import packing_case
    a = packing_case()
    monkeypatch.setattr('scipy_backend.milp', lambda *args, **kwargs:
                        SimpleNamespace(x=np.array([1., 0., 0., 1.]), status=1, message='limit'))
    assert solve_mip(a, 1.)[1] == 'feasible_uncertified'
    monkeypatch.setattr('scipy_backend.milp', lambda *args, **kwargs:
                        SimpleNamespace(x=None, status=1, message='limit'))
    assert solve_mip(a, 1.)[1] == 'uncertified'

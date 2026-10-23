"""Tiny CPU oracle for single-node GPU batch placement (SAE M0)."""
from dataclasses import dataclass
from itertools import product
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint

@dataclass
class Instance:
    capacity: np.ndarray  # (servers, resources), residual
    demand: np.ndarray  # (jobs, resources)
    value: np.ndarray  # (jobs,)
    eligible: np.ndarray  # (servers, jobs), bool
    shapes: np.ndarray  # (servers,)
    future: np.ndarray  # (types, resources)
    weights: np.ndarray  # (types,)
    future_eligible: np.ndarray  # (servers, types), bool

def validate(a, placement):
    n, r = a.demand.shape
    s = a.capacity.shape[0]
    if len(placement) != n or any(i < -1 or i >= s for i in placement):
        return False
    load = np.zeros((s, r))
    for j, i in enumerate(placement):
        if i >= 0:
            if not a.eligible[i, j]:
                return False
            load[i] += a.demand[j]
    return bool(np.all(load <= a.capacity + 1e-9))

def residual(a, placement):
    if not validate(a, placement):
        raise ValueError("invalid placement")
    c = a.capacity.copy().astype(float)
    for j, i in enumerate(placement):
        if i >= 0:
            c[i] -= a.demand[j]
    return c

def future_score(a, rest):
    valid = np.all(rest[:, None, :] + 1e-10 >= a.future[None, :, :], axis=2)
    return float(np.sum((valid & a.future_eligible) * a.weights[None, :]))

def placed_value(a, placement):
    return float(sum(a.value[j] for j, i in enumerate(placement) if i >= 0))

def brute_lex(a):
    n, s = len(a.value), len(a.capacity)
    best, arg = (-np.inf, -np.inf), None
    for placement in product(range(-1, s), repeat=n):
        if validate(a, placement):
            result = (placed_value(a, placement), future_score(a, residual(a, placement)))
            if result > best:
                best, arg = result, placement
    return best, arg

def shape_lp(a):
    n, r = a.demand.shape
    labels = list(dict.fromkeys(a.shapes.tolist()))
    m = len(labels)
    caps = np.stack([a.capacity[a.shapes == label].sum(axis=0) for label in labels])
    rows, rhs = [], []
    for j in range(n):
        row = np.zeros((m, n)); row[:, j] = 1
        rows.append(row.ravel()); rhs.append(1.)
    for h in range(m):
        for k in range(r):
            row = np.zeros((m, n)); row[h, :] = a.demand[:, k]
            rows.append(row.ravel()); rhs.append(caps[h, k])
    A, b = np.array(rows), np.array(rhs)
    result = linprog(-np.tile(a.value, m), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    # SciPy minimizes negative benefit: invert inequality marginals.
    lam = -result.ineqlin.marginals[n:].reshape(m, r)
    mu = -result.ineqlin.marginals[:n]
    primal = -result.fun
    dual = float(np.sum(lam * caps) + np.sum(mu))
    slack = mu[None, :] + lam @ a.demand.T - a.value[None, :]
    assert min(np.min(slack), np.min(lam), np.min(mu)) >= -1e-7
    assert abs(primal - dual) < 1e-7 * max(1, abs(primal))
    assert np.max(A @ result.x - b) < 1e-7
    return dict(value=primal, dual=dual, prices=lam, x=result.x.reshape(m, n), shapes=labels)

def mip_value(a):
    n, r = a.demand.shape
    s = len(a.capacity)
    rows, bounds = [], []
    for j in range(n):
        row = np.zeros((s, n)); row[:, j] = 1
        rows.append(row.ravel()); bounds.append(1.)
    for i in range(s):
        for k in range(r):
            row = np.zeros((s, n)); row[i, :] = a.demand[:, k]
            rows.append(row.ravel()); bounds.append(a.capacity[i, k])
    result = milp(-np.tile(a.value, s), integrality=np.ones(s*n),
                  bounds=Bounds(np.zeros(s*n), a.eligible.ravel().astype(float)),
                  constraints=LinearConstraint(np.array(rows), -np.inf, np.array(bounds)),
                  options={"time_limit": 10})
    if result.status != 0:
        raise RuntimeError("MIP optimality not certified: " + result.message)
    return -result.fun

def marginal(a, rest, i, j):
    before = np.all(rest[i][None, :] + 1e-10 >= a.future, axis=1) & a.future_eligible[i]
    after = np.all((rest[i]-a.demand[j])[None, :] + 1e-10 >= a.future, axis=1) & a.future_eligible[i]
    return float(np.sum((after.astype(int)-before.astype(int))*a.weights))

def greedy(a, method="pgfp", seed=2026):
    if method not in ("random", "best_fit", "pgfp"):
        raise ValueError(method)
    lp = shape_lp(a)
    utility = a.value[None, :] - lp["prices"] @ a.demand.T
    order = sorted(range(len(a.value)), key=lambda j: (-max(utility[:, j]), j))
    rest = a.capacity.astype(float).copy()
    placement = [-1] * len(a.value)
    rng = np.random.default_rng(seed)
    for j in order:
        shape_order = sorted(range(len(lp["shapes"])), key=lambda h: (-utility[h, j], h))
        for h in shape_order:
            candidates = [i for i in range(len(rest))
                          if a.shapes[i] == lp["shapes"][h] and a.eligible[i, j]
                          and np.all(rest[i]+1e-10 >= a.demand[j])]
            if not candidates:
                continue
            if method == "random":
                chosen = int(rng.choice(candidates))
            elif method == "best_fit":
                chosen = min(candidates, key=lambda i: (float(np.sum(rest[i]-a.demand[j])), i))
            else:
                chosen = max(candidates, key=lambda i: (marginal(a, rest, i, j), -i))
            placement[j] = chosen
            rest[chosen] -= a.demand[j]
            break
    assert validate(a, placement)
    return placement, placed_value(a, placement), future_score(a, rest)

def packing_case():
    return Instance(np.array([[8.], [8.]]), np.array([[4.], [4.]]), np.array([10., 10.]),
                    np.ones((2, 2), bool), np.array([0, 0]), np.array([[8.]]),
                    np.array([1.]), np.ones((2, 1), bool))

def integrality_case():
    return Instance(np.array([[4.], [4.]]), np.array([[3.], [3.], [3.]]),
                    np.array([5., 5., 5.]), np.ones((2, 3), bool),
                    np.array([0, 0]), np.array([[4.]]), np.array([1.]), np.ones((2, 1), bool))

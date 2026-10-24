"""P4C PROBE_ONLY sparse Shape-LP; no production integration or GPU allocation.

cuOpt 26.10 official Python Problem API exposes Constraint.DualValue, whereas
get_dual_solution() belongs to the service-client API. Neither is executed here:
runtime prerequisites and hand-LP dual-sign calibration must precede an adapter.
https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/convex/convex-api.html
https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html
"""
import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m0_oracle import Instance, shape_lp

OPTIONS = dict(primal_feasibility_tolerance=1e-9,
               dual_feasibility_tolerance=1e-9,
               ipm_optimality_tolerance=1e-10, time_limit=5.)
AUDIT_TOL = 1e-7


def fixtures():
    def instance(capacity, demand, value, shapes):
        c, d = np.array(capacity, float), np.array(demand, float)
        return Instance(c, d, np.array(value, float),
                        np.ones((len(c), len(d)), bool), np.array(shapes),
                        np.empty((0, 3)), np.empty(0), np.empty((len(c), 0), bool))
    rng = np.random.default_rng(11)
    labels = [7, 3, 11, 5, 19, 2, 23, 17] * 2
    demand = np.column_stack((rng.integers(1, 5, 128),
                              rng.integers(1, 17, 128) / 2,
                              rng.integers(1, 33, 128)))
    return {
        'hand_unique': instance([[1, 10, 10]], [[2, 1, 1]], [6], [7]),
        'hand_degenerate': instance([[3, 0, 0]], [[1, 0, 0], [2, 0, 0]],
                                    [1, 2], [7]),
        'moderate_seed11': instance([[8, 32, 128]] * 16, demand,
                                    rng.integers(1, 13, 128), labels),
    }


def sparse_model(a):
    c, d, v = a.capacity, a.demand, a.value
    if (d.ndim != 2 or c.ndim != 2 or d.shape[1] != c.shape[1]
            or not len(d) or not len(c) or not d.shape[1]
            or v.shape != (len(d),) or a.shapes.shape != (len(c),)):
        raise ValueError('invalid dimensions')
    if any(not np.isfinite(z).all() or (z < 0).any() for z in (c, d, v)):
        raise ValueError('nonfinite or negative input')
    labels = list(dict.fromkeys(a.shapes.tolist()))
    m, (n, r) = len(labels), d.shape
    caps = np.stack([c[a.shapes == label].sum(axis=0) for label in labels])
    # COO row/column/value vectors only; no dense constraint matrix.
    rows = np.concatenate((np.tile(np.arange(n), m),
                           n + np.repeat(np.arange(m * r), n)))
    cols = np.concatenate((np.arange(m * n),
                           np.repeat(np.arange(m) * n, r * n)
                           + np.tile(np.arange(n), m * r)))
    data = np.concatenate((np.ones(m * n), np.tile(d.T.ravel(), m)))
    A = coo_matrix((data, (rows, cols)), shape=(n + m*r, m*n)).tocsr()
    A.eliminate_zeros()
    return dict(A=A, b=np.concatenate((np.ones(n), caps.ravel())),
                c=-np.tile(v, m), caps=caps, labels=labels, m=m, n=n, r=r)


def orders(a, lam):
    u = a.value[None, :] - lam @ a.demand.T
    return dict(utility=u.tolist(),
                task_order=sorted(range(len(a.value)), key=lambda j: (-max(u[:, j]), j)),
                shape_order=[sorted(range(len(lam)), key=lambda h: (-u[h, j], h))
                             for j in range(len(a.value))])


def audit(a, model, x, lam, mu):
    m, n, r = (model[k] for k in ('m', 'n', 'r'))
    if x.shape != (m, n) or lam.shape != (m, r) or mu.shape != (n,):
        raise ValueError('invalid solution dimensions')
    if not all(np.isfinite(z).all() for z in (x, lam, mu)):
        raise ValueError('nonfinite solution')
    primal = float(np.sum(x * a.value))
    dual = float(np.sum(lam * model['caps']) + mu.sum())
    slack = mu[None, :] + lam @ a.demand.T - a.value[None, :]
    row_slack = model['b'] - model['A'] @ x.ravel()
    row_dual = np.concatenate((mu, lam.ravel()))
    diagnostics = dict(primal_residual=float(max(0, -row_slack.min())),
                       x_nonnegative_residual=float(max(0, -x.min())),
                       dual_residual=float(max(0, -slack.min())),
                       dual_nonnegative_residual=float(max(0, -lam.min(), -mu.min())),
                       objective_gap=abs(primal-dual),
                       row_complementarity=float(np.max(np.abs(row_dual * row_slack))),
                       column_complementarity=float(np.max(np.abs(x * slack))))
    feasible = all(diagnostics[k] <= AUDIT_TOL for k in (
        'primal_residual', 'x_nonnegative_residual', 'dual_residual',
        'dual_nonnegative_residual'))
    certified = feasible and all(diagnostics[k] <= AUDIT_TOL * max(1, abs(primal), abs(dual))
                                for k in ('objective_gap', 'row_complementarity',
                                          'column_complementarity'))
    return dict(certified=bool(certified), primal_objective=primal,
                minimization_objective=-primal, dual_objective=dual,
                audit_tolerance=AUDIT_TOL, **diagnostics)


def solve_sparse(a):
    start = perf_counter()
    model = sparse_model(a)
    built = perf_counter()
    options = OPTIONS.copy()
    setup = perf_counter()
    result = linprog(model['c'], A_ub=model['A'], b_ub=model['b'],
                     bounds=(0, None), method='highs', options=options)
    solved = perf_counter()
    if not result.success:
        raise RuntimeError(f'HiGHS uncertified status={result.status}: {result.message}')
    n, m, r = (model[k] for k in ('n', 'm', 'r'))
    x = result.x.reshape(m, n)
    # For min c*x with A*x<=b, marginals <=0. Max-benefit duals are -marginals.
    mu = -result.ineqlin.marginals[:n]
    lam = -result.ineqlin.marginals[n:].reshape(m, r)
    checks = audit(a, model, x, lam, mu)
    ranking = orders(a, lam)
    end = perf_counter()
    return dict(status='PASS' if checks['certified'] else 'BLOCKED',
                solver_status=int(result.status), solver_message=result.message,
                backend='scipy.optimize.linprog', method='highs (automatic selection)',
                requested_solver_tolerances=options,
                solver_reported_tolerances=None,  # OptimizeResult does not export these.
                iterations=int(result.nit), labels=model['labels'],
                dimensions=dict(m=m, n=n, r=r), nnz=model['A'].nnz,
                sparse_bytes=sum(z.nbytes for z in (model['A'].data, model['A'].indices,
                                                    model['A'].indptr)),
                x=x.tolist(), capacity_duals=lam.tolist(), job_duals=mu.tolist(),
                **checks, **ranking,
                timings_seconds=dict(build=built-start, setup=setup-built,
                                     H2D=0., solve=solved-setup, D2H=0.,
                                     audit=end-solved, total=end-start),
                transfer_note='CPU only; no host/device transfer; solve includes SciPy/HiGHS setup')


def compare_tiny(a, sparse):
    dense = shape_lp(a)  # Only the two hand cases may call production dense modeling.
    ranking = orders(a, dense['prices'])
    return dict(value_error=abs(sparse['primal_objective'] - float(dense['value'])),
                x_max_error=float(np.max(np.abs(np.array(sparse['x']) - dense['x']))),
                production_x_feasible=bool(np.all(dense['x'] >= -AUDIT_TOL)
                    and np.all(np.sum(dense['x'], axis=0) <= 1+AUDIT_TOL)
                    and np.all(dense['x'] @ a.demand <= sparse_model(a)['caps']+AUDIT_TOL)),
                utility_max_error=float(np.max(np.abs(np.array(sparse['utility'])
                                                       - np.array(ranking['utility'])))),
                task_order_match=sparse['task_order'] == ranking['task_order'],
                shape_order_match=sparse['shape_order'] == ranking['shape_order'],
                production_task_order=ranking['task_order'],
                production_shape_order=ranking['shape_order'],
                production_capacity_duals=dense['prices'].tolist())


def fixture_sha(a):
    fields = {k: getattr(a, k).tolist() for k in a.__dataclass_fields__}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def gpu_availability():
    installed = importlib.util.find_spec('cuopt') is not None
    return dict(status='GPU_NOT_RUN', backend_status='GPU_BACKEND_BLOCKED',
                cuopt_module_discoverable=installed,
                reason='No verified target runtime RAM/free disk, CUDA wheel compatibility '
                       'or hand-LP dual-sign calibration. No allocation or installation performed.',
                selected_documented_interface='cuopt.linear_programming.problem.Problem / Constraint.DualValue',
                service_client_interface_distinct='get_dual_solution()',
                actual_method=None, actual_dual_available=None,
                sign_verified=False, timings_seconds=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    cases = {}
    for name, a in fixtures().items():
        result = solve_sparse(a)
        result['fixture_sha256'] = fixture_sha(a)
        if name.startswith('hand_'):
            result['production_comparison'] = compare_tiny(a, result)
            c = result['production_comparison']
            if c['value_error'] > AUDIT_TOL or c['x_max_error'] > AUDIT_TOL or not c['production_x_feasible']:
                result['status'] = 'BLOCKED'
        cases[name] = result
    here = Path(__file__).resolve()
    report = dict(mode='PROBE_ONLY',
                  status='PASS' if all(z['status'] == 'PASS' for z in cases.values()) else 'BLOCKED',
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform()),
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (here, here.parents[1] / 'm0_oracle.py')},
                  cases=cases, gpu=gpu_availability(),
                  scope='Sparse CPU oracle only; no physical scheduling or acceleration claim',
                  wall_seconds=perf_counter()-start)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], gpu=report['gpu']['status'],
                          output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          wall_seconds=report['wall_seconds'])))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

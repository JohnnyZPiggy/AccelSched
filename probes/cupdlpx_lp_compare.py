"""P4F PROBE_ONLY fixed sparse-LP comparison; optional GPU import, no installs.

Four cases, exactly one warmup and three measured solves per backend/case.
No scheduling integration or independent device-transfer timing is claimed.
"""
from dataclasses import replace
import json
import platform
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np
import scipy
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probes.shape_lp_backend_probe import fixtures, sparse_model, fixture_sha, OPTIONS
from probes.cupdlpx_lp_smoke import certificate, calibrate_hand_a

REPETITIONS = 3
GPU_SETTINGS = dict(TimeLimit=5., OptimalityTol=1e-10, FeasibilityTol=1e-10,
                    Presolve=False, OutputFlag=False)


def cases():
    f = fixtures()
    a = f['moderate_seed11']
    nodes = [0, 1, 2, 3, 8, 9, 10, 11]
    small = replace(a, capacity=a.capacity[nodes].copy(), demand=a.demand[:64].copy(),
                    value=a.value[:64].copy(), shapes=a.shapes[nodes].copy(),
                    eligible=a.eligible[nodes, :64].copy(),
                    future_eligible=a.future_eligible[nodes].copy())
    return dict(hand_unique=f['hand_unique'], hand_degenerate=f['hand_degenerate'],
                moderate_m4_n64=small, moderate_m8_n128=a)


def solve_one(a, backend, cupdlpx=None, pi_sign=None, calibrate=False):
    start = perf_counter()
    raw, parameters = {}, {}
    try:
        model = sparse_model(a)
        built = perf_counter()
        if backend == 'highs':
            parameters = OPTIONS.copy()
            setup = perf_counter()
            solved = linprog(model['c'], A_ub=model['A'], b_ub=model['b'],
                             bounds=(0, None), method='highs', options=parameters)
            done = perf_counter()
            raw = dict(Status=int(solved.status), StatusName=solved.message,
                       ObjVal=float(solved.fun) if solved.fun is not None else None,
                       DualObj=None, Gap=None, Runtime=None,
                       IterCount=int(solved.nit) if solved.nit is not None else None)
            optimal = solved.status == 0
            x = np.asarray(solved.x, float).reshape(model['m'], model['n'])
            pi = np.asarray(solved.ineqlin.marginals, float)
            rc = np.asarray(solved.lower.marginals, float)
            sign = -1
        elif backend == 'cupdlpx':
            if cupdlpx is None:
                raise ValueError('explicit optional GPU module required')
            problem = cupdlpx.Model(objective_vector=model['c'], constraint_matrix=model['A'],
                constraint_lower_bound=np.full(len(model['b']), -np.inf),
                constraint_upper_bound=model['b'], variable_lower_bound=np.zeros(len(model['c'])),
                variable_upper_bound=np.full(len(model['c']), np.inf))
            problem.ModelSense = cupdlpx.PDLP.MINIMIZE
            problem.setParams(**GPU_SETTINGS)
            setup = perf_counter()
            problem.optimize()
            done = perf_counter()
            raw = {k: getattr(problem, k, None) for k in (
                'Status', 'StatusName', 'StatusCode', 'ObjVal', 'DualObj', 'Gap',
                'RelPrimalResidual', 'RelDualResidual', 'IterCount', 'Runtime')}
            code = raw['StatusCode'] if raw['StatusCode'] is not None else raw['Status']
            optimal = code == cupdlpx.PDLP.OPTIMAL
            x = np.asarray(problem.X, float).reshape(model['m'], model['n'])
            pi, rc = np.asarray(problem.Pi, float), np.asarray(problem.RC, float)
            parameters = dict(problem._params)
            sign = pi_sign
        else:
            raise ValueError('unknown backend')
        retrieved = perf_counter()
        if calibrate:
            sign = calibrate_hand_a(x, pi, rc)['pi_sign']
        checked = certificate(a, x, pi, rc, sign, optimal)
        objective_tol = 1e-7 * max(1, abs(checked['primal_objective']))
        objective_match = (raw['ObjVal'] is not None
                           and np.isfinite(raw['ObjVal'])
                           and abs(raw['ObjVal']+checked['primal_objective']) <= objective_tol)
        if backend == 'cupdlpx':
            objective_match = bool(objective_match
                and raw['DualObj'] is not None and raw['Gap'] is not None
                and np.isfinite([raw['DualObj'], raw['Gap']]).all()
                and abs(raw['DualObj']+checked['dual_objective']) <= objective_tol
                and abs(abs(raw['Gap'])-checked['objective_gap']) <= objective_tol)
        end = perf_counter()
        times = dict(assembly=built-start, host_setup=setup-built,
                     solve_including_internal_setup_transfers=done-setup,
                     retrieval=retrieved-done, certificate=end-retrieved, total=end-start)
        return dict(status='PASS' if checked['certified'] and objective_match else 'UNCERTIFIED',
                    backend=backend, raw_solver=raw, parameters=parameters,
                    certificate=checked, objective_fields_match=bool(objective_match),
                    timings_seconds=times,
                    reported_runtime_fraction=(raw['Runtime']/times['total']
                        if raw['Runtime'] is not None else None),
                    inclusive_solve_fraction=times['solve_including_internal_setup_transfers']/times['total'])
    except Exception:
        return dict(status='FAILED', backend=backend, error=traceback.format_exc(),
                    raw_solver=raw, parameters=parameters,
                    timings_seconds=dict(total=perf_counter()-start))


def compare_pair(cpu, gpu):
    if 'certificate' not in cpu or 'certificate' not in gpu:
        return dict(certificates_available=False)
    c, g = cpu['certificate'], gpu['certificate']
    return dict(certificates_available=True,
                primal_value_error=abs(c['primal_objective']-g['primal_objective']),
                x_max_error=float(np.max(np.abs(np.array(c['x'])-g['x']))),
                prices_max_error=float(np.max(np.abs(np.array(c['capacity_duals'])-g['capacity_duals']))),
                utility_max_error=float(np.max(np.abs(np.array(c['utility'])-g['utility']))),
                task_order_match=c['task_order'] == g['task_order'],
                shape_order_match=c['shape_order'] == g['shape_order'])


def summarize(cpu, gpu):
    result = {}
    for backend, records in (('highs', cpu), ('cupdlpx', gpu)):
        totals = [r['timings_seconds']['total'] for r in records]
        result[backend] = dict(measured_count=len(records),
            certified_count=sum(r['status'] == 'PASS' for r in records),
            median_total_seconds=float(np.median(totals)),
            statuses=[r['status'] for r in records])
    both = all(r['status'] == 'PASS' for r in cpu+gpu)
    result['certified_latency_ratio_cpu_over_gpu'] = (
        result['highs']['median_total_seconds']/result['cupdlpx']['median_total_seconds']
        if both else None)
    return result


def run_compare():
    start = perf_counter()
    import cupdlpx
    from cupdlpx import PDLP
    cupdlpx.PDLP = PDLP
    imported = perf_counter()
    output = dict(mode='PROBE_ONLY', status='PASS', cases={},
        environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                         cupdlpx=getattr(cupdlpx, '__version__', None), platform=platform.platform()),
        gpu_import_seconds=imported-start,
        timing_definition='Fresh sparse assembly + host model setup + optimize/linprog + retrieval + '
                          'original-model certificate. JSON I/O and fixture generation excluded. '
                          'Hidden device setup/H2D/D2H included in optimize wall, not separately observable.',
        warmup_policy='One per backend per case; first GPU warmup includes context initialization.',
        execution_order='For each case, CPU then GPU warmup; three CPU/GPU measured pairs.',
        repetitions=REPETITIONS)
    sign = None
    for name, a in cases().items():
        model = sparse_model(a)
        record = dict(fixture_sha256=fixture_sha(a), dimensions={k:model[k] for k in ('m','n','r')},
                      rows=model['A'].shape[0], columns=model['A'].shape[1], nnz=model['A'].nnz,
                      labels=model['labels'], warmup={}, measured=dict(highs=[], cupdlpx=[]), pairs=[])
        cpu = solve_one(a, 'highs')
        gpu = solve_one(a, 'cupdlpx', cupdlpx, sign, calibrate=name == 'hand_unique')
        record['warmup'] = dict(highs=cpu, cupdlpx=gpu, comparison=compare_pair(cpu, gpu))
        if name == 'hand_unique' and gpu['status'] == 'PASS':
            sign = gpu['certificate']['pi_sign']
        if name.startswith('hand_') and (cpu['status'] != 'PASS' or gpu['status'] != 'PASS'):
            output['status'] = 'BLOCKED_TINY_CERTIFICATE'
            output['cases'][name] = record
            break
        for repetition in range(REPETITIONS):
            cpu = solve_one(a, 'highs')
            gpu = solve_one(a, 'cupdlpx', cupdlpx, sign)
            record['measured']['highs'].append(cpu)
            record['measured']['cupdlpx'].append(gpu)
            record['pairs'].append(dict(repetition=repetition, **compare_pair(cpu, gpu)))
        record['summary'] = summarize(record['measured']['highs'], record['measured']['cupdlpx'])
        all_pass = all(x['status'] == 'PASS' for x in record['measured']['highs']+record['measured']['cupdlpx'])
        if not all_pass or record['warmup']['highs']['status'] != 'PASS' or record['warmup']['cupdlpx']['status'] != 'PASS':
            output['status'] = 'PARTIAL_UNCERTIFIED'
        output['cases'][name] = record
        if name.startswith('hand_') and not all_pass:
            output['status'] = 'BLOCKED_TINY_CERTIFICATE'
            break
    output['gpu_pi_sign'] = sign
    output['total_seconds'] = perf_counter()-start
    return output


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = run_compare()
    args.output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], seconds=result['total_seconds'])))
    raise SystemExit(0 if result['status'] == 'PASS' else 2)

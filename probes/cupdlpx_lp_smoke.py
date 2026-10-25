"""P4E PROBE_ONLY wheel gate and CPU certificate probe; never installs/allocates.

No Linux wheel was published in the inspected registry, so no cuPDLPx device
adapter is claimed. Current docs/main and v0.3.0 have different status APIs.
https://mit-lu-lab.github.io/cuPDLPx/guides/python.html
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probes.shape_lp_backend_probe import audit, fixtures, fixture_sha, orders, solve_sparse, sparse_model


def wheel_gate(metadata):
    """No fallback to source or old incompatible releases; Linux wheel is necessary."""
    def linux_wheels(files):
        return [f['filename'] for f in files if f.get('packagetype') == 'bdist_wheel'
                and f.get('filename', '').endswith('.whl')
                and 'linux' in f['filename'].rsplit('-', 1)[-1]]
    current = linux_wheels(metadata.get('urls', []))
    all_versions = {v: names for v, files in metadata.get('releases', {}).items()
                    if (names := linux_wheels(files))}
    return dict(status='WHEEL_CANDIDATE_ONLY' if current else 'GPU_BACKEND_BLOCKED',
                package=metadata.get('info', {}).get('name'),
                version=metadata.get('info', {}).get('version'),
                current_linux_wheels=current, any_release_linux_wheels=all_versions,
                runtime_abi_cuda_architecture_verified=False,
                reason='No current Linux prebuilt wheel; source compilation forbidden'
                if not current else 'Wheel presence alone does not pass runtime/ABI/CUDA gates')


def certificate(a, x, pi, rc, sign, reported_optimal=True):
    model = sparse_model(a)
    x, pi, rc = (np.asarray(z, float) for z in (x, pi, rc))
    m, n, r = (model[k] for k in ('m', 'n', 'r'))
    if pi.shape != (n+m*r,) or rc.shape != (m*n,) or sign not in (-1, 1):
        raise ValueError('bad Pi/RC dimensions or uncalibrated sign')
    if not np.isfinite(pi).all() or not np.isfinite(rc).all():
        raise ValueError('nonfinite Pi/RC')
    canonical = sign*pi
    mu, lam = canonical[:n], canonical[n:].reshape(m, r)
    checked = audit(a, model, x, lam, mu)
    # Original minimization has lower bound zero, no explicit variable upper bound.
    expected_rc = model['c'] + model['A'].T @ canonical
    rc_error = float(np.max(np.abs(rc-expected_rc)))
    rc_negative = float(max(0, -rc.min()))
    rc_complementarity = float(np.max(np.abs(x.ravel()*rc)))
    valid = (reported_optimal is True and checked['certified'] and rc_error <= 1e-7
             and rc_negative <= 1e-7 and rc_complementarity <= 1e-7)
    return dict(**{k: v for k, v in checked.items() if k != 'certified'},
                certified=bool(valid), reported_optimal=reported_optimal,
                x=x.tolist(), raw_pi=pi.tolist(), raw_rc=rc.tolist(), pi_sign=sign,
                mu=mu.tolist(), capacity_duals=lam.tolist(), rc_max_error=rc_error,
                rc_nonnegative_residual=rc_negative, rc_complementarity=rc_complementarity,
                **orders(a, lam))


def calibrate_hand_a(x, pi, rc):
    a = fixtures()['hand_unique']
    accepted = [c for s in (-1, 1) if (c := certificate(a, x, pi, rc, s))['certified']
                and abs(c['primal_objective']-3) <= 1e-7
                and np.allclose(c['capacity_duals'], [[3, 0, 0]], atol=1e-7, rtol=0)]
    if len(accepted) != 1:
        raise ValueError('hand A did not independently calibrate a unique Pi sign')
    return accepted[0]


def cpu_probe():
    """Real HiGHS reference plus explicitly constructed Pi/RC contract observations."""
    results = {}
    sign = None
    for name in ('hand_unique', 'hand_degenerate'):
        a = fixtures()[name]
        lp = solve_sparse(a)
        model = sparse_model(a)
        canonical = np.concatenate((lp['job_duals'], np.array(lp['capacity_duals']).ravel()))
        pi = -canonical
        rc = model['c'] + model['A'].T @ canonical
        if sign is None:
            sign = calibrate_hand_a(lp['x'], pi, rc)['pi_sign']
        checked = certificate(a, lp['x'], pi, rc, sign, lp['solver_status'] == 0)
        results[name] = dict(fixture_sha256=fixture_sha(a), highs=lp,
                             constructed_contract_observations=checked,
                             cupdlpx_observations=None,
                             task_order_match=checked['task_order'] == lp['task_order'],
                             shape_order_match=checked['shape_order'] == lp['shape_order'])
    return results


def gpu_probe():
    """Explicit optional two-case GPU run, after external isolated install/preflight.

    optimize() owns hidden setup/transfers; do not label its combined wall as a
    device-only solve measurement. No moderate case, warmup solve or retry here.
    """
    start = perf_counter()
    import cupdlpx
    from cupdlpx import Model, PDLP
    imported = perf_counter()
    cases, sign = {}, None
    for name in ('hand_unique', 'hand_degenerate'):
        tick = perf_counter()
        a = fixtures()[name]
        sparse = sparse_model(a)
        assembled = perf_counter()
        model = Model(objective_vector=sparse['c'], constraint_matrix=sparse['A'],
                      constraint_lower_bound=np.full(len(sparse['b']), -np.inf),
                      constraint_upper_bound=sparse['b'],
                      variable_lower_bound=np.zeros(len(sparse['c'])),
                      variable_upper_bound=np.full(len(sparse['c']), np.inf))
        model.ModelSense = PDLP.MINIMIZE
        model.setParams(TimeLimit=5., OptimalityTol=1e-10, FeasibilityTol=1e-10,
                        Presolve=False, OutputFlag=False)
        configured = perf_counter()
        model.optimize()
        solved = perf_counter()
        x = np.array(model.X, float).reshape(sparse['m'], sparse['n'])
        pi, rc = np.array(model.Pi, float), np.array(model.RC, float)
        raw = {k: getattr(model, k, None) for k in (
            'Status', 'StatusCode', 'StatusName', 'ObjVal', 'DualObj', 'Gap',
            'RelPrimalResidual', 'RelDualResidual', 'IterCount', 'Runtime')}
        status_code = raw['StatusCode'] if raw['StatusCode'] is not None else raw['Status']
        retrieved = perf_counter()
        try:
            if sign is None:
                sign = calibrate_hand_a(x, pi, rc)['pi_sign']
            checked = certificate(a, x, pi, rc, sign, status_code == PDLP.OPTIMAL)
        except ValueError as exc:
            cases[name] = dict(certified=False, raw_solver=raw, x=x.tolist(),
                               pi=pi.tolist(), rc=rc.tolist(), error=str(exc),
                               fixture_sha256=fixture_sha(a))
            break
        cpu = solve_sparse(a)
        raw_objectives_match = (np.isfinite([raw['ObjVal'], raw['DualObj'], raw['Gap']]).all()
                                and abs(raw['ObjVal']+checked['primal_objective']) <= 1e-7
                                and abs(raw['DualObj']+checked['dual_objective']) <= 1e-7
                                and abs(abs(raw['Gap'])-checked['objective_gap']) <= 1e-7)
        value_error = abs(cpu['primal_objective']-checked['primal_objective'])
        audited = perf_counter()
        cases[name] = dict(fixture_sha256=fixture_sha(a), raw_solver=raw,
                           original_model_certificate=checked, highs=cpu,
                           raw_objectives_match=bool(raw_objectives_match),
                           value_error=value_error,
                           task_order_match=checked['task_order'] == cpu['task_order'],
                           shape_order_match=checked['shape_order'] == cpu['shape_order'],
                           parameters=dict(model._params),
                           timings_seconds=dict(assembly=assembled-tick,
                                host_model_setup=configured-assembled,
                                optimize_including_hidden_transfers=solved-configured,
                                result_retrieval=retrieved-solved,
                                audit_and_highs=audited-retrieved,total=audited-tick),
                           certified=bool(checked['certified'] and raw_objectives_match
                                          and value_error <= 1e-7))
        if not cases[name]['certified']:
            break
    return dict(status='PASS' if len(cases) == 2 and all(c['certified'] for c in cases.values())
                else 'BLOCKED', backend='cupdlpx', version=getattr(cupdlpx, '__version__', None),
                cases=cases, gpu_pi_sign=sign, import_seconds=imported-start,
                total_seconds=perf_counter()-start,
                transfer_note='Individual H2D/device setup/D2H timings unavailable from Model API; '
                              'optimize wall is inclusive, not a kernel benchmark')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = perf_counter()
    raw = args.metadata.read_bytes()
    gate = wheel_gate(json.loads(raw))
    results = cpu_probe()
    result = dict(mode='PROBE_ONLY', status='GPU_BACKEND_BLOCKED', gpu_status='GPU_NOT_RUN',
                  wheel_gate=gate, metadata_sha256=hashlib.sha256(raw).hexdigest(),
                  cpu_status='PASS' if all(v['constructed_contract_observations']['certified']
                                          for v in results.values()) else 'BLOCKED',
                  cases=results, cupdlpx_version_tested=None, gpu_pi_sign=None,
                  gpu_rc=None, gpu_timings=None, gpu_method=None, installation_attempted=False,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  wall_seconds=perf_counter()-start,
                  note='No device adapter or installation path. Any future wheel candidate '
                       'needs separate ABI/CUDA/device and released API validation.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], cpu_status=result['cpu_status'],
                          wall_seconds=result['wall_seconds'],
                          output_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())))
    return 2 if result['cpu_status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

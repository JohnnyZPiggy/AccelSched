"""P4D PROBE_ONLY fail-closed environment/certificate contract, not a GPU backend.

Observed target failed official minimums: no cuOpt installation or API execution.
26.10 Problem API documents Variable.Value/ReducedCost and Constraint.DualValue;
get_dual_solution is a different service API. No device dual sign is assumed.
https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/convex/convex-api.html
https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probes.shape_lp_backend_probe import audit, fixtures, orders, sparse_model, fixture_sha

CUOPT_VERSION = '26.10.0'


def environment_gate(facts):
    """Missing observations fail closed; disk uses free bytes, never total bytes."""
    def number(key, threshold):
        value = facts.get(key)
        return type(value) in (int, float) and np.isfinite(value) and value >= threshold
    version = facts.get('python_version')
    python_ok = (isinstance(version, list) and len(version) >= 2
                 and all(type(v) is int for v in version[:2])
                 and (3, 11) <= tuple(version[:2]) <= (3, 14))
    gpu = facts.get('nvidia_smi', {})
    match = re.fullmatch(r'Tesla T4,\s*7\.5,\s*([0-9.]+),\s*([0-9.]+)\s*',
                         gpu.get('stdout', '')) if gpu.get('exit_code') == 0 else None
    driver = tuple(map(int, match[1].split('.'))) if match else ()
    cuda = facts.get('nvcc', {})
    cuda13 = cuda.get('exit_code') == 0 and bool(re.search(r'release 13\.0,', cuda.get('stdout', '')))
    glibc = facts.get('glibc', [])
    glibc_ok = (len(glibc) == 2 and glibc[0] == 'glibc'
                and bool(re.fullmatch(r'\d+\.\d+', glibc[1]))
                and tuple(map(int, glibc[1].split('.'))) >= (2, 28))
    openssl = facts.get('openssl_version', [])
    checks = dict(
        cpu_four_cores=number('cpu_count', 4) and number('cpu_affinity_count', 4),
        ram_16_gib=number('mem_total_bytes', 16*2**30),
        disk_100_decimal_gb_free=number('disk_free_bytes', 100*10**9),
        nvme_verified=facts.get('nvme_verified') is True,
        python_supported=python_ok,
        openssl3=bool(openssl and openssl[0] == 3 and facts.get('libssl_so3') is True
                      and facts.get('libcrypto_so3') is True),
        t4_cc75=bool(match), cuda13_driver_compatible=cuda13 and driver >= (580, 65, 6),
        supported_linux=bool(facts.get('platform', '').startswith('Linux-')
                             and facts.get('machine') == 'x86_64' and glibc_ok),
        # A wheel filename is not a validated isolated dependency resolution.
        pinned_wheel_and_dependencies_verified=facts.get('pinned_wheel_and_dependencies_verified') is True,
    )
    return dict(status='PASS' if all(checks.values()) else 'GPU_BACKEND_BLOCKED',
                checks=checks, failed_or_unverified=[k for k, v in checks.items() if not v],
                candidate_package='cuopt-cu13=='+CUOPT_VERSION)


def certificate(a, x, raw_row_duals, sign):
    """Audit explicitly supplied observations; a sign is accepted only by calibration."""
    model = sparse_model(a)
    raw = np.asarray(raw_row_duals, float)
    if sign not in (-1, 1) or raw.shape != (model['n']+model['m']*model['r'],):
        raise ValueError('invalid sign or row dual dimensions')
    canonical = sign * raw
    mu = canonical[:model['n']]
    lam = canonical[model['n']:].reshape(model['m'], model['r'])
    checks = audit(a, model, np.asarray(x, float), lam, mu)
    return dict(**checks, raw_row_duals=raw.tolist(), normalization_sign=sign,
                capacity_duals=lam.tolist(), job_duals=mu.tolist(),
                **orders(a, lam))


def calibrate_hand_a(x, raw_row_duals):
    a = fixtures()['hand_unique']
    candidates = [certificate(a, x, raw_row_duals, s) for s in (-1, 1)]
    valid = [c for c in candidates if c['certified']
             and abs(c['primal_objective']-3) <= 1e-7
             and np.allclose(c['capacity_duals'], [[3, 0, 0]], atol=1e-7, rtol=0)
             and np.allclose(c['job_duals'], [0], atol=1e-7, rtol=0)]
    if len(valid) != 1:
        raise ValueError('hand A did not uniquely certify a dual sign')
    return valid[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    raw = args.environment.read_bytes()
    observed = json.loads(raw)
    gate = environment_gate(observed['facts'])
    # This delivery intentionally contains no untested install/solver adapter.
    result = dict(mode='PROBE_ONLY', status='GPU_BACKEND_BLOCKED', gpu_status='GPU_NOT_RUN',
                  gate=gate, environment_sha256=hashlib.sha256(raw).hexdigest(),
                  environment_source_sha=observed['source_sha'],
                  fixtures={k: fixture_sha(v) for k, v in fixtures().items() if k.startswith('hand_')},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gpu_primal=None, gpu_duals=None, gpu_sign=None, gpu_method=None,
                  gpu_timings=None, installation_attempted=False, lp_attempted=False,
                  note='CPU contract only. Future passing gate needs a separately reviewed '
                       'isolated official API adapter; this script never installs or allocates.')
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

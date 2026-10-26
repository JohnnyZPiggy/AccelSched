"""B0/B1: unchanged production solve, synthetic static CPU end-to-end benchmark.

Example: python bench_v4_static.py --out ../smoke --nodes 8 --jobs 16 --seeds 11
Full matrix: --nodes 64 --jobs 1000 (six scenarios, seeds 11/29/47/71/95).
Each measured call has a fresh subprocess, a small same-method warmup, and a
hard outer watchdog. Imports/loading/warmup/process startup are NOT solve time;
process wall and export costs are retained separately. Two matched repetitions
rotate method order. Shared build+input hash is charged once to EACH method's
source_to_audited_ms, not added across methods. No nested elapsed double count.
No GPU-hours, throughput or hardware GPU-utilization claim. Shape-LP/order/
placement internals are NOT instrumented: only the public inclusive call is timed.
"""
import argparse
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter, perf_counter_ns

import numpy as np
import scipy
from accelsched import METHODS, solve
from m0_oracle import Instance
from workload import build_workload

SCENARIOS = ('heterogeneous', 'fragmentation', 'cpu_tight', 'ram_tight', 'eligibility', 'pressure')
SEEDS = (11, 29, 47, 71, 95)
ARRAYS = tuple(f.name for f in fields(Instance))


def json_bytes(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + '\n').encode()


def fingerprint(a):
    """Hash names, exact dtype/shape, and complete C-order input bytes."""
    h = hashlib.sha256()
    for name in ARRAYS:
        x = np.ascontiguousarray(getattr(a, name))
        header = json_bytes([name, x.dtype.str, x.shape])
        h.update(len(header).to_bytes(8, 'little')); h.update(header)
        h.update(x.nbytes.to_bytes(8, 'little')); h.update(x.tobytes())
    return h.hexdigest()


def instance(nodes, jobs, scenario, seed):
    """Predeclared families A/B and CPU-only; no seed search/workload repair.

    Heterogeneous is deliberately moderate pressure at 64x1000 (many small CPU
    jobs); other scenarios stress independent constraints. Fragmentation puts
    small demands before large 4/8-GPU jobs, preserving input order for all methods.
    Shapes group identical (family,capacity) ONLY; futures are fixed, nonneutral.
    """
    if scenario not in SCENARIOS or nodes < 4 or jobs < 4:
        raise ValueError('known scenario, >=4 nodes/jobs required')
    rng = np.random.default_rng(seed)
    kinds = np.arange(nodes) % 4
    cap = np.array([[4,64,256], [8,128,512], [4,64,256], [0,64,256]], float)[kinds]
    family = np.array([0,0,1,-1])[kinds]
    gpu = rng.choice([0,1,2], jobs, p=[.75,.20,.05])
    cpu = rng.choice([1,2,4], jobs)
    ram = rng.choice([2,4,8], jobs)
    if scenario == 'fragmentation':
        gpu = np.r_[np.ones(jobs*3//4), rng.choice([4,8], jobs-jobs*3//4)]
        cpu = rng.choice([2,4,8], jobs); ram = cpu*4
    elif scenario == 'cpu_tight':
        cpu = rng.choice([8,16,32], jobs); ram = rng.choice([4,8,16], jobs)
    elif scenario == 'ram_tight':
        ram = rng.choice([64,128,256], jobs)
    elif scenario == 'eligibility':
        gpu = rng.choice([0,1,2,4], jobs, p=[.1,.4,.3,.2])
    elif scenario == 'pressure':
        gpu = rng.choice([0,1,2,4,8], jobs, p=[.1,.15,.25,.3,.2])
        cpu = rng.choice([8,16,32], jobs); ram = rng.choice([32,64,128], jobs)
    demand = np.column_stack([gpu,cpu,ram]).astype(float)
    job_family = rng.integers(0,2,jobs)
    eligible = (gpu[None,:] == 0) | (family[:,None] == job_family[None,:])
    if scenario == 'eligibility':
        # Explicit restrictive masks, including genuine individually impossible jobs.
        eligible &= rng.random((nodes,jobs)) < .15
        eligible[:, np.arange(jobs) % 17 == 0] = False
    future = np.array([[1,8,32],[4,32,128],[1,8,32],[0,8,32]], float)
    future_family = np.array([0,0,1,-1])
    fe = (future[None,:,0] == 0) | (family[:,None] == future_family[None,:])
    return build_workload(cap, demand, value=rng.integers(1,21,jobs), eligible=eligible,
                          shapes=kinds, future=future, weights=[1,2,1,1], future_eligible=fe)


def audit(a, out):
    """Independent reconstruction; never calls production validate/residual/value.

    A placement is one integer node ID per job (-1 unassigned), so at-most-one
    is structural. Residual and capacity use fixed 1e-9 absolute tolerance.
    """
    p = out.placement
    if p is None:
        return dict(valid=None, reason='missing placement', metrics=None)
    if len(p) != len(a.value) or any(isinstance(i, (bool, np.bool_)) or
            not isinstance(i, (int,np.integer)) or not -1 <= i < len(a.capacity) for i in p):
        return dict(valid=False, reason='assignment shape/type/bounds', metrics=None)
    load = np.zeros_like(a.capacity)
    for j,i in enumerate(p):
        if i >= 0:
            if not a.eligible[i,j]:
                return dict(valid=False, reason='eligibility', metrics=None)
            load[i] += a.demand[j]
    if not np.all(np.isfinite(load)) or np.any(load > a.capacity + 1e-9):
        return dict(valid=False, reason='capacity', metrics=None)
    value = math.fsum(float(a.value[j]) for j,i in enumerate(p) if i >= 0)
    if out.value is None or not math.isfinite(out.value) or abs(value-out.value) > 1e-7*max(1,abs(value)):
        return dict(valid=False, reason='objective', metrics=None)
    rest = a.capacity-load
    if out.residual is None or np.shape(out.residual) != rest.shape or not np.allclose(out.residual,rest,atol=1e-9,rtol=0):
        return dict(valid=False, reason='residual', metrics=None)
    partial = np.any(load > 0, axis=1) & np.any(rest > 1e-9, axis=1)
    fits = np.all(rest[:,None,:]+1e-10 >= a.future[None,:,:],axis=2) & a.future_eligible
    return dict(valid=True, reason=None, metrics=dict(accepted_value=value,
                accepted_count=sum(i>=0 for i in p), allocated=load.sum(axis=0).tolist(),
                future_candidate_nodes=fits.sum(axis=0).tolist(),
                future_placeability_proxy=float(np.sum(fits*a.weights)),
                partial_node_residual_gpu_proxy=float(rest[partial,0].sum())))


def measured(a, method, seed, time_limit):
    before = fingerprint(a)
    tick = perf_counter_ns()
    out = solve(a, method, seed=seed, time_limit=time_limit)
    solve_ms = (perf_counter_ns()-tick)/1e6
    tick = perf_counter_ns()
    check = audit(a,out)
    unchanged = before == fingerprint(a)
    output = dict(status=out.status, optimal=bool(out.optimal), error=out.error,
                  placement=None if out.placement is None else list(out.placement), value=out.value,
                  residual=None if out.residual is None else out.residual.tolist())
    output_sha = hashlib.sha256(json_bytes(output)).hexdigest()
    audit_ms = (perf_counter_ns()-tick)/1e6
    return dict(method=method, input_sha256=before, input_unchanged=unchanged,
                solver_status=out.status, execution_status='RETURNED', output=output, audit=check,
                certified_optimal=bool(method=='highs_mip' and out.status=='optimal' and out.optimal and check['valid'] and unchanged),
                output_sha256=output_sha, solve_ms=solve_ms, audit_metrics_hash_ms=audit_ms,
                nested_solver_elapsed_ms=out.elapsed_ms, solve_plus_audit_ms=solve_ms+audit_ms)


def worker(args):
    # Fresh process, one small same-method warmup. Warmup records never enter summaries.
    tick = perf_counter()
    warm = instance(8,16,args.scenario,args.seed)
    warmup = measured(warm,args.method,args.seed,args.time_limit)
    warmup_ms = (perf_counter()-tick)*1000
    tick = perf_counter()
    with np.load(args.input,allow_pickle=False) as data:
        a = Instance(*(data[k] for k in ARRAYS))
    assert fingerprint(a) == args.input_sha, 'INPUT_SHA_MISMATCH'
    load_hash_ms = (perf_counter()-tick)*1000
    row = measured(a,args.method,args.seed,args.time_limit)
    row.update(warmup=warmup, warmup_wall_ms=warmup_ms, input_load_hash_ms=load_hash_ms)
    args.result.write_bytes(json_bytes(row))


def watched(argv, timeout, env=None):
    """Real subprocess deadline, including imports/warmup/LP/model build/audit."""
    tick = perf_counter()
    try:
        r = subprocess.run(argv,capture_output=True,text=True,timeout=timeout,env=env)
        return dict(exit_code=r.returncode, watchdog_timeout=False, stdout=r.stdout, stderr=r.stderr,
                    process_wall_ms=(perf_counter()-tick)*1000)
    except subprocess.TimeoutExpired:
        return dict(exit_code=None,watchdog_timeout=True,stdout='',stderr='outer watchdog expired',
                    process_wall_ms=(perf_counter()-tick)*1000)


def environment():
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    if sys.platform == 'win32':
        import ctypes
        process, system = ctypes.c_size_t(), ctypes.c_size_t()
        if ctypes.windll.kernel32.GetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(process),ctypes.byref(system)):
            affinity = [i for i in range(os.cpu_count()) if process.value & (1<<i)]
    return dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,
                platform=platform.platform(),processor=platform.processor(),logical_cpus=os.cpu_count(),
                affinity=affinity,threads={k:os.environ.get(k) for k in
                ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                highs_threads='native automatic; no private scheduler reset')


def paired(cases):
    """Across-seed paired means with seed bootstrap percentile intervals (descriptive)."""
    result=[]
    for scenario in SCENARIOS:
        for method in METHODS:
            dv,dt=[],[]
            for case in cases:
                if case['scenario'] != scenario: continue
                rows=case['records']
                def complete(m):
                    return [r for r in rows if r['method']==m and r.get('audit',{}).get('valid') and r.get('input_unchanged')]
                base,other=complete('first_fit'),complete(method)
                if len(base)!=case['repeats'] or len(other)!=case['repeats']: continue
                bv=np.mean([r['audit']['metrics']['accepted_value'] for r in base])
                bt=np.mean([r['source_to_audited_ms'] for r in base])
                if bv: dv.append(float(np.mean([r['audit']['metrics']['accepted_value'] for r in other])/bv-1))
                if bt: dt.append(float(np.mean([r['source_to_audited_ms'] for r in other])/bt-1))
            def interval(values):
                if not values: return None
                x=np.array(values); rng=np.random.default_rng(104)
                draws=x[rng.integers(len(x),size=(2000,len(x)))].mean(axis=1)
                return dict(n_seeds=len(x),mean=float(x.mean()),bootstrap95=np.quantile(draws,[.025,.975]).tolist())
            result.append(dict(scenario=scenario,method=method,relative_value_vs_first_fit=interval(dv),
                               relative_source_to_audited_vs_first_fit=interval(dt)))
    return result


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    scratch=args.out/'scratch'; scratch.mkdir()
    env=os.environ.copy()
    env.update(TEMP=str(scratch.resolve()),TMP=str(scratch.resolve()),TMPDIR=str(scratch.resolve()),
               PYTHONDONTWRITEBYTECODE='1')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): env[k]='1'
    # Parent should use the same environment (CLI entrypoint sets threads before import).
    report=dict(mode='CPU_PRODUCTION_STATIC',utc_start=datetime.now(timezone.utc).isoformat(),
                environment=environment(),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                source_tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
                dirty=bool(subprocess.check_output(['git','status','--porcelain'])),
                seeds=args.seeds,scenarios=SCENARIOS,repeats=args.repeats,mip_time_limit=args.time_limit,
                per_process_watchdog_seconds=args.watchdog,batch_seconds=args.batch_seconds,
                measurement='warmup in each fresh process; public solve inclusive; audit includes immutability/output hash; shared build+hash charged per method; process startup/load/warmup/export excluded and separately retained',
                resource_units=['GPU-count','CPU-cores','host RAM GiB'],cases=[])
    deadline=perf_counter()+args.batch_seconds
    for scenario in SCENARIOS:
        for seed_index,seed in enumerate(args.seeds):
            tick=perf_counter_ns(); a=instance(args.nodes,args.jobs,scenario,seed); digest=fingerprint(a)
            build_ms=(perf_counter_ns()-tick)/1e6
            impossible=~np.any(a.eligible & np.all(a.demand[None,:,:]<=a.capacity[:,None,:]+1e-9,axis=2),axis=0)
            case=dict(scenario=scenario,seed=seed,nodes=args.nodes,jobs=args.jobs,input_sha256=digest,
                      shared_build_hash_ms=build_ms,repeats=args.repeats,
                      offered_to_capacity=(a.demand.sum(axis=0)/a.capacity.sum(axis=0)).tolist(),
                      individually_impossible_jobs=int(impossible.sum()),
                      input={k:getattr(a,k).tolist() for k in ARRAYS},records=[])
            inp=scratch/'input.npz'; tick=perf_counter(); np.savez(inp,**vars(a))
            case['input_export_ms']=(perf_counter()-tick)*1000
            for rep in range(args.repeats):
                shift=(seed_index+rep)%len(METHODS)
                order=METHODS[shift:]+METHODS[:shift]
                for index,method in enumerate(order):
                    row=dict(method=method,repetition=rep,order_index=index,input_sha256=digest,
                             solver_status=None,execution_status='NOT_RUN_BATCH_DEADLINE')
                    result=scratch/'worker.json'
                    if perf_counter()<deadline:
                        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--input',str(inp.resolve()),
                             '--input-sha',digest,'--result',str(result.resolve()),'--method',method,'--scenario',scenario,
                             '--seed',str(seed),'--time-limit',str(args.time_limit)]
                        execution=watched(cmd,min(args.watchdog,max(.01,deadline-perf_counter())),env)
                        row.update(execution)
                        row['execution_status']='WATCHDOG_TIMEOUT' if execution['watchdog_timeout'] else 'PROCESS_FAILED'
                        if execution['exit_code']==0 and result.exists():
                            row.update(json.loads(result.read_bytes()))
                            row['source_to_audited_ms']=build_ms+row['solve_plus_audit_ms']
                            assert row['input_sha256']==digest
                        if result.exists(): result.unlink()
                    case['records'].append(row)
            inp.unlink()
            report['cases'].append(case)
            (args.out/f'{scenario}-{seed}.json').write_bytes(json_bytes(case))
            print(json.dumps(dict(scenario=scenario,seed=seed,statuses=dict(Counter(
                r['solver_status'] or r['execution_status'] for r in case['records'])))),flush=True)
    assert list(scratch.iterdir())==[]
    scratch.rmdir()
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),scratch_cleanup=True,paired_summary=paired(report['cases']))
    rows=[r for c in report['cases'] for r in c['records']]
    report['status']='PASS' if all(r.get('input_unchanged') and r.get('audit',{}).get('valid') is not False and
        r.get('execution_status')=='RETURNED' and r['solver_status'] not in ('invalid','failed','input_invalid') for r in rows) else 'PARTIAL'
    raw=json_bytes(report); (args.out/'raw.json').write_bytes(raw)
    print(json.dumps(dict(status=report['status'],records=len(rows),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path); p.add_argument('--nodes',type=int,default=64); p.add_argument('--jobs',type=int,default=1000)
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS)); p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--time-limit',type=float,default=2.); p.add_argument('--watchdog',type=float,default=30.)
    p.add_argument('--batch-seconds',type=float,default=900.)
    p.add_argument('--worker',action='store_true'); p.add_argument('--input',type=Path); p.add_argument('--result',type=Path)
    p.add_argument('--input-sha'); p.add_argument('--method',choices=METHODS); p.add_argument('--scenario',choices=SCENARIOS)
    p.add_argument('--seed',type=int,default=11)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.worker: worker(args)
    else:
        if not args.out or args.repeats<2 or not args.seeds or min(args.time_limit,args.watchdog,args.batch_seconds)<=0:
            raise SystemExit('require new --out, >=2 repeats, seeds and positive bounds')
        run(args)

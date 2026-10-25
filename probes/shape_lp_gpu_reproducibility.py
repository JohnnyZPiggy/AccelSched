"""P5D PROBE_ONLY identical fresh solves and complete original-vector CPU audit."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import shape_lp_gpu_convergence as prior
base=prior.base
CONFIG=dict(name='identical_60s_1e12',tolerance=1e-12,row_scale=False)
MAX_RUNS=4


def array_records(vectors):
    return {name:dict(shape=list(a.shape),dtype=a.dtype.str,nbytes=a.nbytes,
        sha256=hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest())
        for name,a in zip(('x','pi','rc'),vectors)}


def failure_reasons(c):
    absolute=('primal_residual','x_nonnegative_residual','dual_residual','dual_nonnegative_residual',
              'rc_nonnegative_residual','rc_max_error','rc_complementarity')
    scaled=('objective_gap','row_complementarity','column_complementarity')
    return ([k for k in absolute if c[k]>1e-7]+[k for k in scaled if c[k]>c['scaled_gap_complementarity_tolerance']]
        +([] if c['reported_optimal'] else ['solver_not_optimal'])
        +([] if c['objective_fields_match'] else ['objective_fields']))


def run(directory,budget=300.):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import cupdlpx
    from cupdlpx import PDLP
    cupdlpx.PDLP=PDLP;imported=perf_counter()
    report=dict(status='PARTIAL',mode='PROBE_ONLY',runs=[],seed=71,configuration=CONFIG,
        limits=dict(max_identical_runs=4,max_diagnostic_runs=0,native_seconds=60.,worker_budget_seconds=budget),
        import_seconds=imported-start,
        policy='2-4 independent fresh Model objects, identical CSR/objective and parameters; not warm-start continuation. '
               'Stop before next call when <75s remain; stop actual OOM, no arbitrary repair/retry.')
    def persist():
        report['elapsed_seconds']=perf_counter()-start
        (directory/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    tick=perf_counter();a=base.fixture(1000000,4096);made=perf_counter();model=base.assemble(a);built=perf_counter();A=model['A']
    report['input']=dict(**base.dimensions(1000000,4096),fixture_seconds=made-tick,assembly_seconds=built-made,
        fixture_sha256=base.digest(a.capacity,a.demand,a.value,a.shapes),
        lp_sha256=base.digest(A.data,A.indices,A.indptr,model['b'],model['c']),actual_nnz=A.nnz,
        csr_bytes=dict(data=A.data.nbytes,indices=A.indices.nbytes,indptr=A.indptr.nbytes))
    if report['input']['fixture_sha256']!=prior.REGISTERED_FIXTURE or report['input']['lp_sha256']!=prior.REGISTERED_LP:
        raise ValueError('fixed original1M input mismatch')
    hand=base.solve_one(base.fixtures()['hand_unique'],'cupdlpx',cupdlpx,calibrate=True)
    report['calibration']=hand;persist()
    if hand['status']!='PASS':report['status']='BLOCKED_CALIBRATION';persist();return report
    failed=None;passed=None
    for index in range(MAX_RUNS):
        remaining=budget-(perf_counter()-start)
        if remaining<75:
            report['stop_reason']=dict(kind='OBSERVED_TIME_BUDGET',remaining_seconds=remaining);break
        with base.Memory() as memory,prior.utilization() as util:
            record,vectors=prior.gpu_solve(model,CONFIG.copy(),60.,cupdlpx)
        record.update(index=index,memory=memory.report(),gpu_utilization=util,
            from_scratch_total_seconds=report['input']['fixture_seconds']+report['input']['assembly_seconds']+record['timing']['total'])
        report['runs'].append(record)
        checked=perf_counter();record['input_lp_sha256']=base.digest(A.data,A.indices,A.indptr,model['b'],model['c'])
        record['postsolve_input_hash_seconds']=perf_counter()-checked
        if record['input_lp_sha256']!=prior.REGISTERED_LP:raise ValueError('backend mutated original LP')
        if 'certificate' in record:record['failure_reasons']=failure_reasons(record['certificate'])
        else:record['failure_reasons']=['no_certificate']
        keep=vectors is not None and ((record['status']=='PASS' and passed is None) or (record['status']!='PASS' and failed is None and passed is None))
        if keep:
            tick=perf_counter();name='first-certified.npz' if record['status']=='PASS' else 'representative-failed.npz'
            path=directory/name;np.savez_compressed(path,x=vectors[0],pi=vectors[1],rc=vectors[2]);raw=path.read_bytes()
            entry=dict(name=name,index=index,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                vector_sha256=base.digest(*vectors),arrays=array_records(vectors),storage_seconds=perf_counter()-tick)
            record['stored_vectors']=entry
            if record['status']=='PASS':passed=entry
            else:failed=entry
        del vectors;gc.collect();persist()
        if record['status']=='FAILED' and any(s in record.get('error','').lower() for s in ('out of memory','bad_alloc','cuda_error_memory')):
            report['stop_reason']=dict(kind='REAL_OOM');break
    report['selected_vectors']=passed if passed is not None else failed
    successes=sum(z['status']=='PASS' for z in report['runs']);count=len(report['runs'])
    report['observed_successes']=successes;report['observed_runs']=count
    report['status']='SAMPLE_COMPLETE' if count>=2 else 'INCOMPLETE_SAMPLE'
    report['conclusion']='Observed frequency only; no deterministic convergence, formal confidence or CPU/GPU speed claim.'
    persist();return report


def matrix_free(a,x,pi,rc,raw):
    """Independent arithmetic using job sums and X@demand, no CSR audit routine."""
    n=len(a.value);labels=list(dict.fromkeys(a.shapes.tolist()));X=x.reshape(len(labels),n)
    y=-pi;mu=y[:n];lam=y[n:].reshape(len(labels),3)
    caps=np.stack([a.capacity[a.shapes==h].sum(0) for h in labels])
    slack=np.r_[1-X.sum(0),(caps-X@a.demand).ravel()]
    reduced=mu[None,:]+lam@a.demand.T-a.value[None,:]
    neg=lambda z:float(max(0,-z.min()))
    primal=float(np.sum(X*a.value));dual=float(mu.sum()+np.sum(lam*caps));tol=1e-7*max(1,abs(primal),abs(dual))
    c=dict(primal_objective=primal,dual_objective=dual,primal_residual=neg(slack),x_nonnegative_residual=neg(x),
        dual_residual=neg(reduced),dual_nonnegative_residual=neg(y),rc_nonnegative_residual=neg(rc),
        rc_max_error=float(np.max(np.abs(rc.reshape(X.shape)-reduced))),objective_gap=abs(primal-dual),
        row_complementarity=float(np.max(np.abs(y*slack))),column_complementarity=float(np.max(np.abs(X*reduced))),
        rc_complementarity=float(np.max(np.abs(x*rc))),absolute_tolerance=1e-7,
        scaled_gap_complementarity_tolerance=tol,reported_optimal=raw['Status']==0,
        objective_fields_match=bool(np.isfinite([raw['ObjVal'],raw['DualObj'],raw['Gap']]).all()
            and abs(raw['ObjVal']+primal)<=tol and abs(raw['DualObj']+dual)<=tol
            and abs(abs(raw['Gap'])-abs(primal-dual))<=tol))
    c['certified']=not failure_reasons(c);return c


def verdict(remote,sparse,independent):
    return ('NUMERIC_DISAGREEMENT' if len({remote['certified'],sparse['certified'],independent['certified']})>1
            else 'PASS_CERTIFIED_REVALIDATION' if sparse['certified'] else 'PASS_FAILED_SOLUTION_REVALIDATION')


def audit(summary_path,vectors_path,output):
    start=perf_counter();wrapper=json.loads(Path(summary_path).read_bytes());report=wrapper['experiment']
    selected=report['selected_vectors'];index=selected['index'];remote=report['runs'][index]
    raw=Path(vectors_path).read_bytes()
    if len(raw)!=selected['bytes'] or hashlib.sha256(raw).hexdigest()!=selected['sha256']:raise ValueError('full NPZ hash/size mismatch')
    with np.load(vectors_path,allow_pickle=False) as data:
        if set(data.files)!={'x','pi','rc'}:raise ValueError('unexpected arrays')
        x,pi,rc=(data[k] for k in ('x','pi','rc'))
    if array_records((x,pi,rc))!=selected['arrays'] or base.digest(x,pi,rc)!=selected['vector_sha256']:raise ValueError('array provenance mismatch')
    if x.shape!=(8000000,) or pi.shape!=(1000024,) or rc.shape!=(8000000,) or any(z.dtype!=np.float64 for z in (x,pi,rc)):raise ValueError('array schema')
    a=base.fixture(1000000,4096);model=base.assemble(a);A=model['A']
    fixture_sha=base.digest(a.capacity,a.demand,a.value,a.shapes);lp_sha=base.digest(A.data,A.indices,A.indptr,model['b'],model['c'])
    if fixture_sha!=prior.REGISTERED_FIXTURE or lp_sha!=prior.REGISTERED_LP:raise ValueError('local original model mismatch')
    s=remote['raw_solver'];sparse=base.certify(model,x,pi,rc,remote['certificate']['reported_optimal'],s['ObjVal'],s['DualObj'],s['Gap'])
    independent=matrix_free(a,x,pi,rc,s)
    differences={name:{k:float(c[k]-remote['certificate'][k]) for k in ('primal_residual','dual_residual','rc_max_error','objective_gap','row_complementarity','column_complementarity','rc_complementarity')}
        for name,c in [('sparse',sparse),('matrix_free',independent)]}
    result=dict(status=verdict(remote['certificate'],sparse,independent),project_sha=wrapper['project_sha'],adapter_sha=wrapper['source_sha'],
        selected_index=index,vector_sha256=selected['vector_sha256'],arrays=selected['arrays'],npz_sha256=selected['sha256'],
        fixture_sha256=fixture_sha,lp_sha256=lp_sha,sparse=sparse,matrix_free=independent,differences=differences,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        seconds=perf_counter()-start,method='Reassembled full registered CSR plus independently implemented matrix-free original-model products. No solver or GPU used.')
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(dict(status=result['status'],seconds=result['seconds'])))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir')
    parser.add_argument('--audit-summary');parser.add_argument('--vectors');parser.add_argument('--audit-output');args=parser.parse_args()
    if args.audit_summary:
        r=audit(args.audit_summary,args.vectors,args.audit_output)
        raise SystemExit(2 if r['status']=='NUMERIC_DISAGREEMENT' else 0)
    r=run(args.output_dir);print(json.dumps(dict(status=r['status'],runs=r['observed_runs'])))

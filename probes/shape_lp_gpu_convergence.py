"""P5C PROBE_ONLY fixed million-job convergence; original audit stays immutable."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
import traceback

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import shape_lp_gpu_numerics as previous
base=previous.base
REGISTERED_FIXTURE='19ffbab8f1dba4c786a9598baf8baa7067efd82e6683d3fc9e5ce3f04bb206a0'
REGISTERED_LP='eb3880a01cb4c3d7029e03839f3ae973eae29401c09e216a42b4952bc9ed9bc6'
ALTERNATIVES=(dict(name='tighter_1e13',tolerance=1e-13,row_scale=False),
              dict(name='coefficient_row_equilibration',tolerance=1e-12,row_scale=True))


def scale_rows(model):
    """Binary row-coefficient scaling only; D=I, no RHS normalization."""
    A=model['A'];nonempty=np.flatnonzero(np.diff(A.indptr))
    maxima=np.ones(A.shape[0])
    maxima[nonempty]=np.maximum(1.,np.maximum.reduceat(np.abs(A.data),A.indptr[nonempty]))
    row=np.exp2(-np.floor(np.log2(maxima)));data=A.data.copy()
    for lo in range(0,A.shape[0],100000):
        hi=min(A.shape[0],lo+100000);left,right=A.indptr[lo],A.indptr[hi]
        data[left:right]*=np.repeat(row[lo:hi],np.diff(A.indptr[lo:hi+1]))
    transformed=model.copy();transformed.update(A=csr_matrix((data,A.indices.copy(),A.indptr.copy()),shape=A.shape),b=row*model['b'])
    return transformed,row


def inverse(x,pi,rc,row):
    return np.asarray(x),row*np.asarray(pi),np.asarray(rc)


@contextmanager
def utilization():
    stop=threading.Event();samples=[];errors=[]
    def monitor():
        while not stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                if r.returncode==0:samples.append(int(r.stdout.strip().splitlines()[0]))
                else:errors.append(r.stderr)
            except Exception as e:errors.append(str(e))
            stop.wait(.5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    data=dict(samples_percent=samples,errors=errors)
    try:yield data
    finally:
        stop.set();thread.join(timeout=3)
        data['sampled_peak_percent']=max(samples,default=None)
        data['meaning']='500ms device-wide utilization samples; sampled peak is a lower bound.'


def gpu_solve(model,config,limit,module):
    start=perf_counter();raw={}
    try:
        row=None;used=model
        if config['row_scale']:used,row=scale_rows(model)
        transformed=perf_counter()
        p=module.Model(objective_vector=used['c'],constraint_matrix=used['A'],
            constraint_lower_bound=np.full(len(used['b']),-np.inf),constraint_upper_bound=used['b'],
            variable_lower_bound=np.zeros(len(used['c'])),variable_upper_bound=np.full(len(used['c']),np.inf))
        p.ModelSense=module.PDLP.MINIMIZE
        parameters=dict(TimeLimit=limit,OptimalityTol=config['tolerance'],FeasibilityTol=config['tolerance'],Presolve=False,OutputFlag=False)
        p.setParams(**parameters);setup=perf_counter();p.optimize();done=perf_counter()
        raw={k:getattr(p,k,None) for k in ('Status','StatusName','ObjVal','DualObj','Gap','Runtime','IterCount','RelPrimalResidual','RelDualResidual')}
        x,pi,rc=np.asarray(p.X,float),np.asarray(p.Pi,float),np.asarray(p.RC,float);retrieved=perf_counter()
        if row is not None:x,pi,rc=inverse(x,pi,rc,row)
        mapped=perf_counter()
        checked=base.certify(model,x,pi,rc,p.Status==module.PDLP.OPTIMAL,raw['ObjVal'],raw['DualObj'],raw['Gap'])
        extra=previous.diagnostics(model,x,pi);end=perf_counter()
        return dict(status='PASS' if checked['certified'] else 'UNCERTIFIED',configuration=config,parameters=parameters,
            raw_solver=raw,certificate=checked,diagnostics=extra,
            transform=None if row is None else dict(formulas='A_s=R A; b_s=R b; c_s=c; x=z; Pi=R Pi_s; RC=RC_s (D=I)',
                row_min=float(row.min()),row_max=float(row.max()),row_sha256=base.digest(row)),
            timing=dict(transform=transformed-start,model_setup=setup-transformed,
                optimize_including_hidden_transfers=done-setup,retrieval=retrieved-done,
                inverse_mapping=mapped-retrieved,original_audit_diagnostics=end-mapped,total=end-start),
            hidden_transfer_seconds=None),(x,pi,rc)
    except Exception:
        return dict(status='FAILED',configuration=config,raw_solver=raw,error=traceback.format_exc(),
                    timing=dict(total=perf_counter()-start)),None


def run(directory,budget=315.):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import cupdlpx
    from cupdlpx import PDLP
    cupdlpx.PDLP=PDLP;imported=perf_counter()
    import psutil
    report=dict(status='PARTIAL',mode='PROBE_ONLY',gpu=[],cpu=[],alternative_decisions=[],
        import_seconds=imported-start,seed=71,budget_seconds=budget,
        policy='One60s baseline then one120s extension. At most2 alternatives45s each after actual failure '
            'only if budget covers solve+75s reference/export reserve. One CPU highs-ipm45s reference '
            'if >=75s remain and host memory practical. No same-config extra GPU solve for evidence.',
        scaling_note='P5B RHS/column scaling worsened original residuals; not repeated. '
            'Alternative2 uses row coefficient maxima only, D=I; inverse-map before original audit.')
    def persist():
        report['elapsed_seconds']=perf_counter()-start
        (directory/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    report['memory_before']=dict(host_available_bytes=psutil.virtual_memory().available,
        gpu_memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free','--format=csv,noheader,nounits'],text=True).strip(),
        historical_gpu_path_rss=3685515264,historical_cpu_rss=9237442560,historical_device_used=2464153600,
        estimate_note='Empirical1.5x prior observed peak guard, not a proven upper bound; CPU guard does not block GPU.')
    total,free=(int(z.strip())*1024**2 for z in report['memory_before']['gpu_memory'].split(','))
    if psutil.virtual_memory().available<1.5*3685515264 or free<1.5*2464153600:
        report['status']='NOT_RUN_GROUNDED_MEMORY_GUARD';persist();return report
    tick=perf_counter();a=base.fixture(1000000,4096);made=perf_counter();model=base.assemble(a);built=perf_counter();A=model['A']
    report['input']=dict(**base.dimensions(1000000,4096),fixture_seconds=made-tick,assembly_seconds=built-made,
        fixture_sha256=base.digest(a.capacity,a.demand,a.value,a.shapes),
        lp_sha256=base.digest(A.data,A.indices,A.indptr,model['b'],model['c']),
        actual_nnz=A.nnz,csr_bytes=dict(data=A.data.nbytes,indices=A.indices.nbytes,indptr=A.indptr.nbytes))
    if report['input']['fixture_sha256']!=REGISTERED_FIXTURE or report['input']['lp_sha256']!=REGISTERED_LP:
        raise ValueError('immutable original1M input mismatch')
    # Tiny GPU calibration only (no100K precondition): first real CUDA invocation.
    hand=base.solve_one(base.fixtures()['hand_unique'],'cupdlpx',cupdlpx,calibrate=True)
    report['gpu_first_calibration']=hand;persist()
    if hand['status']!='PASS':report['status']='BLOCKED_CALIBRATION';persist();return report
    def attempt(config,limit,label):
        with base.Memory() as memory,utilization() as util:record,vectors=gpu_solve(model,config,limit,cupdlpx)
        record.update(label=label,memory=memory.report(),gpu_utilization=util,
            from_scratch_total_seconds=report['input']['fixture_seconds']+report['input']['assembly_seconds']+record['timing']['total'])
        report['gpu'].append(record)
        # Capture at most one baseline and one first certified vector set; no extra solve.
        if vectors is not None and (label=='baseline' or record['status']=='PASS' and not report.get('certified_vectors')):
            tick=perf_counter();path=directory/('01-certified-vectors.npz' if record['status']=='PASS' else '02-baseline-vectors.npz')
            np.savez_compressed(path,x=vectors[0],pi=vectors[1],rc=vectors[2]);raw=path.read_bytes()
            entry=dict(name=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                vector_sha256=base.digest(*vectors),compression_seconds=perf_counter()-tick)
            record['vectors']=entry
            if record['status']=='PASS':report['certified_vectors']=entry
        del vectors;gc.collect();persist();return record
    config=dict(name='tight_p5b',tolerance=1e-12,row_scale=False)
    baseline=attempt(config,60.,'baseline')
    if budget-(perf_counter()-start)>=135:
        current=attempt(config,120.,'extended')
    else:
        current=baseline;report['extension_not_run']='Observed budget insufficient for120s+export'
    for config in ALTERNATIVES:
        remaining=budget-(perf_counter()-start)
        if current['status']=='PASS':
            report['alternative_decisions'].append(dict(configuration=config,status='NOT_NEEDED_CERTIFIED'));continue
        if remaining<120:
            report['alternative_decisions'].append(dict(configuration=config,status='NOT_RUN_OBSERVED_BUDGET',remaining_seconds=remaining));continue
        report['alternative_decisions'].append(dict(configuration=config,status='RUN',trigger=current['status'],remaining_seconds=remaining))
        current=attempt(config,45.,config['name'])
    remaining=budget-(perf_counter()-start)
    if remaining>=75 and psutil.virtual_memory().available>1.2*9237442560:
        with base.Memory() as memory,utilization() as util:
            cpu,_=previous.solve(model,'highs',dict(name='same_host_highs_ipm',scaled=False),45.)
        cpu.update(memory=memory.report(),gpu_utilization=util,label='first',
            from_scratch_total_seconds=report['input']['fixture_seconds']+report['input']['assembly_seconds']+cpu['timing']['total'])
        report['cpu'].append(cpu)
    else:report['cpu_not_run']=dict(status='NOT_RUN_OBSERVED_RESOURCE_OR_TIME_BUDGET',remaining_seconds=remaining,host_available_bytes=psutil.virtual_memory().available)
    good_gpu=[v for v in report['gpu'] if v['status']=='PASS'];good_cpu=[v for v in report['cpu'] if v['status']=='PASS']
    report['status']='PASS_GPU_CERTIFIED' if good_gpu else 'UNCERTIFIED'
    report['certified_cpu_over_gpu_ratio']=good_cpu[0]['timing']['total']/good_gpu[0]['timing']['total'] if good_gpu and good_cpu else None
    persist();return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',required=True)
    args=parser.parse_args();result=run(args.output_dir)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'])))

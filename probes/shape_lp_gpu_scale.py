"""P5A PROBE_ONLY sparse Shape-LP scaling; optional GPU, no production changes.

Seed 71; all physical eligibility flags true (broadcast, never nodes*jobs storage).
Exactly the P4C shape-major LP; direct CSR avoids COO construction temporaries.
No placement, integer-scheduling throughput, or solver-kernel speed claim.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from time import perf_counter
import traceback

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from m0_oracle import Instance
from probes.cupdlpx_lp_compare import solve_one
from probes.shape_lp_backend_probe import fixtures

TIERS=((1000,64),(10000,256),(100000,1024),(1000000,4096))
SEED=71
SHAPES=8
LIMIT=20.
TOL=1e-7


def digest(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str((a.dtype.str,a.shape)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def fixture(n,nodes,m=SHAPES):
    if n<1 or nodes<m or m<1:raise ValueError('positive jobs and nodes>=shapes required')
    rng=np.random.default_rng(np.random.SeedSequence([SEED,n,nodes,m]))
    demand=np.column_stack((rng.integers(1,5,n),rng.integers(1,17,n)/2,rng.integers(1,33,n))).astype(float)
    value=rng.integers(1,13,n).astype(float)
    shapes=np.arange(nodes)%m
    capacity=np.array([[8.,32.,128.]])[np.zeros(nodes,dtype=int)]
    return Instance(capacity,demand,value,np.broadcast_to(True,(nodes,n)),shapes,
                    np.empty((0,3)),np.empty(0),np.empty((nodes,0),bool))


def dimensions(n,nodes,m=SHAPES):
    return dict(nodes=nodes,m=m,n=n,r=3,variables=m*n,rows=n+3*m,nnz=4*m*n,
        estimated_csr_bytes=4*m*n*12+(n+3*m+1)*4,
        conservative_working_bytes=4*m*n*12+(n+3*m+1)*4+240*m*n+128*n+64*nodes)


def assemble(a):
    n,r=a.demand.shape;labels=list(dict.fromkeys(a.shapes.tolist()));m=len(labels)
    caps=np.stack([a.capacity[a.shapes==h].sum(axis=0) for h in labels])
    count=m*n*(r+1)
    if count>np.iinfo(np.int32).max:raise ValueError('CSR int32 limit')
    indptr=np.empty(n+m*r+1,np.int32);indptr[:n+1]=np.arange(n+1,dtype=np.int32)*m
    indptr[n+1:]=n*m+np.arange(1,m*r+1,dtype=np.int32)*n
    indices=np.empty(count,np.int32);data=np.empty(count,float)
    indices[:n*m].reshape(n,m)[:]=np.arange(n,dtype=np.int32)[:,None]+n*np.arange(m,dtype=np.int32)[None,:]
    data[:n*m]=1.
    for h in range(m):
        for k in range(r):
            lo=n*m+(h*r+k)*n;hi=lo+n
            indices[lo:hi]=h*n+np.arange(n,dtype=np.int32);data[lo:hi]=a.demand[:,k]
    A=csr_matrix((data,indices,indptr),shape=(n+m*r,m*n));A.eliminate_zeros()
    return dict(A=A,b=np.r_[np.ones(n),caps.ravel()],c=-np.tile(a.value,m),caps=caps,m=m,n=n,r=r)


def certify(model,x,pi,rc,optimal,reported_objective=None,reported_dual=None,reported_gap=None):
    A,b,c=model['A'],model['b'],model['c'];x,pi,rc=(np.asarray(z,float) for z in (x,pi,rc))
    if x.shape!=c.shape or rc.shape!=c.shape or pi.shape!=b.shape:raise ValueError('certificate shape')
    if not all(np.isfinite(z).all() for z in (x,pi,rc)):raise ValueError('nonfinite certificate')
    # Independently sparse original-model products; no normalized solver residual trusted.
    y=-pi;slack=b-A@x;reduced=c+A.T@y
    primal=float(-c@x);dual=float(b@y);scale=max(1.,abs(primal),abs(dual))
    nonneg=lambda a:float(max(0.,-np.min(a)))
    d=dict(primal_objective=primal,dual_objective=dual,
        primal_residual=nonneg(slack),x_nonnegative_residual=nonneg(x),
        dual_residual=nonneg(reduced),dual_nonnegative_residual=nonneg(y),
        rc_nonnegative_residual=nonneg(rc),rc_max_error=float(np.max(np.abs(rc-reduced))),
        objective_gap=abs(primal-dual),row_complementarity=float(np.max(np.abs(y*slack))),
        column_complementarity=float(np.max(np.abs(x*reduced))),rc_complementarity=float(np.max(np.abs(x*rc))))
    d['objective_fields_match']=bool(reported_objective is not None and np.isfinite(reported_objective)
        and abs(reported_objective+primal)<=TOL*scale)
    if reported_dual is not None:
        d['objective_fields_match']=bool(d['objective_fields_match'] and np.isfinite([reported_dual,reported_gap]).all()
            and abs(reported_dual+dual)<=TOL*scale and abs(abs(reported_gap)-d['objective_gap'])<=TOL*scale)
    absolute=('primal_residual','x_nonnegative_residual','dual_residual','dual_nonnegative_residual',
              'rc_nonnegative_residual','rc_max_error','rc_complementarity')
    scaled=('objective_gap','row_complementarity','column_complementarity')
    d.update(certified=bool(optimal and d['objective_fields_match'] and all(d[k]<=TOL for k in absolute)
        and all(d[k]<=TOL*scale for k in scaled)),reported_optimal=bool(optimal),
        absolute_tolerance=TOL,scaled_gap_complementarity_tolerance=TOL*scale,pi_sign=-1,
        vector_sha256=digest(x,pi,rc),vector_shapes=[list(z.shape) for z in (x,pi,rc)],
        retained_vectors='Hashes and first/last samples only; full solution vectors not downloaded.',
        samples={k:dict(first=z[:4].tolist(),last=z[-4:].tolist()) for k,z in [('x',x),('pi',pi),('rc',rc)]})
    return d


class Memory:
    """50ms RSS and 200ms nvidia-smi samples; lower bounds, not exact peaks."""
    def __enter__(self):
        self.stop=threading.Event();self.rss=[];self.gpu=[];self.errors=[]
        def sample():
            import psutil
            process=psutil.Process();last=0.
            while not self.stop.is_set():
                self.rss.append(process.memory_info().rss)
                if perf_counter()-last>=.2:
                    try:
                        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=2)
                        if r.returncode==0:self.gpu.append(int(r.stdout.strip().splitlines()[0])*1024**2)
                        else:self.errors.append(r.stderr)
                    except Exception as e:self.errors.append(str(e))
                    last=perf_counter()
                self.stop.wait(.05)
        self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=3)
    def report(self):
        import resource
        return dict(sampled_peak_process_rss_bytes=max(self.rss,default=None),
            process_lifetime_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sampled_peak_device_used_bytes=max(self.gpu,default=None),rss_samples=len(self.rss),gpu_samples=len(self.gpu),
            errors=self.errors[:3],meaning='Sampled lower-bound peaks; GPU device-wide including context; lifetime RSS high-water shared across prior runs.')


def solve(model,backend,module=None):
    start=perf_counter();raw={}
    try:
        if backend=='highs':
            params=dict(time_limit=LIMIT,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-10)
            setup=perf_counter();p=linprog(model['c'],A_ub=model['A'],b_ub=model['b'],bounds=(0,None),method='highs',options=params)
            done=perf_counter();raw=dict(Status=int(p.status),StatusName=p.message,ObjVal=p.fun,iterations=int(p.nit))
            if p.x is None or p.ineqlin.marginals is None:raise ValueError('solver did not return primal/dual arrays')
            x,pi,rc=p.x,p.ineqlin.marginals,p.lower.marginals;optimal=p.status==0
        else:
            p=module.Model(objective_vector=model['c'],constraint_matrix=model['A'],constraint_lower_bound=np.full(len(model['b']),-np.inf),
                constraint_upper_bound=model['b'],variable_lower_bound=np.zeros(len(model['c'])),variable_upper_bound=np.full(len(model['c']),np.inf))
            p.ModelSense=module.PDLP.MINIMIZE
            params=dict(TimeLimit=LIMIT,OptimalityTol=1e-10,FeasibilityTol=1e-10,Presolve=False,OutputFlag=False)
            p.setParams(**params);setup=perf_counter();p.optimize();done=perf_counter()
            raw={k:getattr(p,k,None) for k in ('Status','StatusName','ObjVal','DualObj','Gap','Runtime','IterCount','RelPrimalResidual','RelDualResidual')}
            x,pi,rc=np.asarray(p.X,float),np.asarray(p.Pi,float),np.asarray(p.RC,float);optimal=p.Status==module.PDLP.OPTIMAL
        retrieved=perf_counter()
        check=certify(model,x,pi,rc,optimal,raw['ObjVal'],raw.get('DualObj'),raw.get('Gap'))
        end=perf_counter()
        return dict(status='PASS' if check['certified'] else 'UNCERTIFIED',raw_solver=raw,parameters=params,certificate=check,
            timing=dict(model_setup=setup-start,optimize_including_hidden_transfers=done-setup,retrieval=retrieved-done,
                        original_certificate=end-retrieved,total=end-start),
            hidden_transfer_seconds=None,hidden_transfer_note='Included in model/optimize/retrieval wall; not exposed separately.')
    except Exception:
        return dict(status='FAILED',raw_solver=raw,error=traceback.format_exc(),timing=dict(total=perf_counter()-start))


def run(output,budget=230.):
    start=perf_counter();import cupdlpx
    from cupdlpx import PDLP
    cupdlpx.PDLP=PDLP;imported=perf_counter()
    report=dict(status='PASS',mode='PROBE_ONLY',seed=SEED,tiers=[],gpu_import_seconds=imported-start,
        policy='One first invocation plus one warm repeat/backend/tier; first invocation of later tiers is NOT process cold. GPU first at each tier.',
        timing_definition='Same preassembled CSR; full path model setup+optimize/transfers+retrieval+certificate. Add fixture/assembly separately for inclusive from-scratch total.',
        eligibility='All true via zero-stride read-only broadcast; P4C pooled shape model; physical placement not simulated.',
        tolerance_note='HiGHS primal/dual 1e-9 IPM1e-10; cuPDLPx optimality/feasibility1e-10. Independent original-model audit follows P4E strict RC checks.',
        limits=dict(per_solve_seconds=LIMIT,worker_seconds=budget),planned=[dimensions(n,s) for n,s in TIERS])
    # Preserve exact P4E empirical sign calibration using an observed real GPU solve.
    hand=solve_one(fixtures()['hand_unique'],'cupdlpx',cupdlpx,calibrate=True)
    report['hand_calibration']=hand
    if hand['status']!='PASS':report['status']='BLOCKED_CALIBRATION'
    else:
        for n,nodes in TIERS:
            if perf_counter()-start>budget-85:
                report['tiers'].append(dict(**dimensions(n,nodes),status='NOT_RUN_TIME_BUDGET'));report['status']='PARTIAL_TIME_BUDGET';break
            import psutil
            need=dimensions(n,nodes)['conservative_working_bytes'];available=psutil.virtual_memory().available
            if need>.55*available:
                report['tiers'].append(dict(**dimensions(n,nodes),status='NOT_RUN_MEMORY_GUARD',available_bytes=available));report['status']='PARTIAL_MEMORY_GUARD';break
            tick=perf_counter();a=fixture(n,nodes);made=perf_counter();model=assemble(a);built=perf_counter()
            A=model['A'];case=dict(**dimensions(n,nodes),fixture_seconds=made-tick,assembly_seconds=built-made,
                fixture_sha256=digest(a.capacity,a.demand,a.value,a.shapes),lp_sha256=digest(A.data,A.indices,A.indptr,model['b'],model['c']),
                csr_bytes=dict(data=A.data.nbytes,indices=A.indices.nbytes,indptr=A.indptr.nbytes),
                actual_nnz=A.nnz,status='PASS',runs={'highs':[],'cupdlpx':[]})
            for repeat in range(2):
                for backend in ('cupdlpx','highs'):
                    with Memory() as mem:result=solve(model,backend,cupdlpx)
                    result.update(memory=mem.report(),invocation='first_tier' if repeat==0 else 'warm_repeat',
                        from_scratch_total_seconds=case['fixture_seconds']+case['assembly_seconds']+result['timing']['total'])
                    case['runs'][backend].append(result)
                    if result['status']!='PASS':case['status']='PARTIAL_UNCERTIFIED'
            cpu=case['runs']['highs'][1];gpu=case['runs']['cupdlpx'][1]
            case['certified_warm_cpu_over_gpu_ratio']=cpu['timing']['total']/gpu['timing']['total'] if case['status']=='PASS' else None
            case['certified_warm_lp_jobs_per_second']={b:n/case['runs'][b][1]['timing']['total'] if case['status']=='PASS' else None for b in ('highs','cupdlpx')}
            case['objective_pair_difference']=abs(cpu['certificate']['primal_objective']-gpu['certificate']['primal_objective']) if 'certificate' in cpu and 'certificate' in gpu else None
            report['tiers'].append(case)
            Path(output).write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
            if case['status']!='PASS':report['status']='PARTIAL_UNCERTIFIED'
            del model,a;gc.collect()
            if any(r['status']=='FAILED' or r['timing']['total']>LIMIT*1.5 for rs in case['runs'].values() for r in rs):break
    completed={t['n'] for t in report['tiers']}
    report['not_attempted_jobs']=[n for n,_ in TIERS if n not in completed]
    report['total_seconds']=perf_counter()-start
    Path(output).write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.output);print(json.dumps(dict(status=result['status'],seconds=result['total_seconds'])))

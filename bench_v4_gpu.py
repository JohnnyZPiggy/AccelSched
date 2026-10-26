"""V4 ultra-scale research path, NOT a production PGFP/GPU acceleration claim.

A: exact input-order First-Fit via monotone per-demand/eligibility cursors.
Earlier nodes rejected for an identical key cannot become feasible: capacity
only decreases. No dense job×node ledger. Independent regeneration + all-job
prefix/resource/value/residual audit; dense original oracle only at <=100K.
C: identical sparse Shape-LP across explicit backends, original KKT, then
EXPERIMENTAL LP-score-ordered First-Fit (NOT PGFP). Shape-LP relaxes eligibility;
the physical pass enforces every original family/group predicate independently.
All clock values are seconds; cold process includes imports, warm reconstructs
the entire workload again. Provisioning/install/download are separate overheads.
"""
from time import perf_counter
PROCESS_START=perf_counter()
import argparse
from dataclasses import dataclass,asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from types import SimpleNamespace
import numpy as np

CASES=('balanced','high_pressure','fragmentation','multiresource')
CHUNK=32768
def js(x):return (json.dumps(x,sort_keys=True,allow_nan=False)+'\n').encode()
def sha(x):return hashlib.sha256(x).hexdigest()


@dataclass(frozen=True)
class Config:
    n:int
    nodes:int
    case:str
    seed:int=11
    chunk:int=CHUNK


def inventory(c):
    if c.case not in CASES or min(c.n,c.nodes,c.chunk)<=0:raise ValueError('invalid config')
    k=np.arange(c.nodes)%4;scale=max(1,math.ceil(c.n/c.nodes/64))
    cap=np.array([[4,128,512],[8,256,1024],[4,128,512],[0,256,1024]],np.int64)[k]
    cap[:,1:]*=scale
    return cap,np.array([0,0,1,-1],np.int8)[k],np.arange(c.nodes)%16,k


def chunks(c):
    """Fixed PCG64/chunk draw order; no filtering/repair or algorithm-conditioned draws.

    Balanced intentionally has a CPU-only majority and expected GPU offered
    demand 0.7×inventory. CPU/RAM inventory scaling is a declared synthetic
    parameter, not a claim of actual machines. Other cases preserve overload.
    Returned columns: whole GPU, CPU cores, host RAM GiB, value, family, group.
    """
    cap,_,_,_=inventory(c);rng=np.random.Generator(np.random.PCG64(c.seed))
    probability=min(1.,.7*int(cap[:,0].sum())/(c.n*(7/3))) if c.case=='balanced' else .6 if c.case=='high_pressure' else .25 if c.case=='fragmentation' else .03
    for start in range(0,c.n,c.chunk):
        count=min(c.chunk,c.n-start);u=rng.random(count)
        gpu=rng.choice([1,2,4],count);gpu[u>=probability]=0
        if c.case=='fragmentation':
            large=np.arange(start,start+count)>=c.n*3//4
            gpu[(u<probability)&large]=rng.choice([4,8],int(((u<probability)&large).sum()))
        cpu=rng.choice([8,16,32] if c.case in ('high_pressure','multiresource') else [1,2,4],count)
        ram=rng.choice([32,128,512],count) if c.case=='multiresource' else cpu*4
        value=rng.integers(1,21,count);family=rng.integers(0,2,count);family[gpu==0]=-1
        group=rng.integers(0,17,count) if c.case=='multiresource' else np.full(count,-1)
        yield start,np.column_stack([gpu,cpu,ram,value,family,group]).astype(np.int64)


def allowed(family,group,node_family,node_group):
    return ((family<0)|(node_family==family))&((group<0)|(node_group==group))


def fingerprint_start(c,cap):
    h=hashlib.sha256(js(dict(config=asdict(c),capacity_dtype=cap.dtype.str,capacity_shape=cap.shape)))
    h.update(cap.tobytes());return h


def update_hash(h,start,x):
    h.update(js([start,x.dtype.str,x.shape]));h.update(x.tobytes())


def materialize(c):
    return np.concatenate([x for _,x in chunks(c)])


def original_instance(c,data=None):
    if c.n>100000:raise ValueError('dense reference deliberately bounded at100K')
    from m0_oracle import Instance
    cap,nf,ng,kind=inventory(c);data=materialize(c) if data is None else data
    eligible=np.empty((c.nodes,c.n),bool)
    for start in range(0,c.n,c.chunk):
        block=data[start:start+c.chunk]
        eligible[:,start:start+len(block)]=allowed(block[:,4][None,:],block[:,5][None,:],nf[:,None],ng[:,None])
    return Instance(cap.astype(float),data[:,:3].astype(float),data[:,3].astype(float),eligible,kind,
                    np.zeros((1,3)),np.zeros(1),np.zeros((c.nodes,1),bool))


def place_block(block,ids,rest,nf,ng,cursors,placement):
    for row,j in zip(block,ids):
        g,c,r,_,family,group=map(int,row);key=(g,c,r,family,group)
        node=cursors.get(key,0)
        while node<len(rest):
            left=rest[node]
            if (family<0 or nf[node]==family) and (group<0 or ng[node]==group) and left[0]>=g and left[1]>=c and left[2]>=r:break
            node+=1
        cursors[key]=node
        if node<len(rest):
            placement[j]=node;left=rest[node];left[0]-=g;left[1]-=c;left[2]-=r


def audit(c,p,residual,expected_input=None):
    """Full independent pass; vectorized scatter, not allocator cursor state.

    Every job/assigned node is checked; prefix sums reconstructed per chunk.
    Nonnegative integer demands mean valid final node loads bound all prefixes.
    No sampling; no state hash substituted for physical arithmetic.
    """
    cap,nf,ng,_=inventory(c)
    assert p.shape==(c.n,) and p.dtype.kind in 'iu' and np.all((p>=-1)&(p<c.nodes))
    assert np.shape(residual)==cap.shape and np.isfinite(residual).all()
    load=np.zeros_like(cap);h=fingerprint_start(c,cap);value=count=gpu_jobs=impossible=0
    offered=np.zeros(3,np.int64);possible={}
    for start,x in chunks(c):
        update_hash(h,start,x);offered+=x[:,:3].sum(axis=0);gpu_jobs+=int((x[:,0]>0).sum())
        block=p[start:start+len(x)];take=np.flatnonzero(block>=0);nodes=block[take];rows=x[take]
        assert allowed(rows[:,4],rows[:,5],nf[nodes],ng[nodes]).all()
        for k in range(3):np.add.at(load[:,k],nodes,rows[:,k])
        assert np.all(load<=cap)
        value+=int(rows[:,3].sum());count+=len(take)
        # Unique original demand/predicate types, independent of allocation outcome.
        keys,frequencies=np.unique(x[:,[0,1,2,4,5]],axis=0,return_counts=True)
        for key,freq in zip(keys,frequencies):
            t=tuple(key)
            if t not in possible:
                possible[t]=bool(np.any(allowed(key[3],key[4],nf,ng)&np.all(cap>=key[:3],axis=1)))
            if not possible[t]:impossible+=int(freq)
    assert np.array_equal(cap-load,residual)
    if expected_input is not None:assert h.hexdigest()==expected_input,'INPUT_DIGEST_MISMATCH'
    free=cap-load;future=sum(int(np.sum(allowed(f,-1,nf,ng)&np.all(free>=d,axis=1)))
                           for f,d in [(0,[1,8,32]),(1,[4,32,128]),(-1,[0,8,32])])
    partial=np.any(load>0,axis=1)&np.any(free>0,axis=1)
    return dict(status='PASS',input_sha256=h.hexdigest(),accepted_count=count,accepted_value=value,
        allocated=load.sum(axis=0).tolist(),offered=offered.tolist(),
        offered_to_capacity=(offered/cap.sum(axis=0)).tolist(),gpu_job_count=gpu_jobs,
        individually_impossible=impossible,future_candidate_nodes_proxy=future,
        partial_node_residual_gpu_proxy=int(free[partial,0].sum()),all_jobs_checked=c.n,
        dense_reference='NOT_RUN',residual_sha256=sha(free.tobytes()))


def peak_rss():
    if sys.platform!='win32':
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
    import ctypes
    from ctypes import wintypes as w
    class Counters(ctypes.Structure):
        _fields_=[('cb',w.DWORD),('faults',w.DWORD)]+[(name,ctypes.c_size_t) for name in
            ['peak','working','qpp','qp','qnp','qn','page','peakpage']]
    api=ctypes.WinDLL('kernel32');api.GetCurrentProcess.restype=w.HANDLE
    ps=ctypes.WinDLL('psapi');ps.GetProcessMemoryInfo.argtypes=[w.HANDLE,ctypes.POINTER(Counters),w.DWORD]
    result=Counters();result.cb=ctypes.sizeof(result)
    return int(result.peak) if ps.GetProcessMemoryInfo(api.GetCurrentProcess(),ctypes.byref(result),result.cb) else None


def sync():
    try:
        import cupy as cp
        cp.cuda.runtime.deviceSynchronize()
        return True
    except Exception:return False


def pipeline(c,directory,key,backend=None,dense=True):
    start=perf_counter();stages=dict(input_gen_hash=0.,model_lp_certification=0.,ordering=0.,placement=0.,audit=0.,serialization=0.)
    tick=perf_counter();cap,nf,ng,kind=inventory(c);rest=cap.tolist();p=np.full(c.n,-1,np.int32);h=fingerprint_start(c,cap)
    stages['input_gen_hash']+=perf_counter()-tick;cursors={};lp=None;vectors={};data=None
    if backend:
        tick=perf_counter();data=materialize(c)
        for offset in range(0,c.n,c.chunk):update_hash(h,offset,data[offset:offset+c.chunk])
        stages['input_gen_hash']+=perf_counter()-tick
        from certified_pgfp import canonical_shape
        from lp_backends import solve_lp
        from probes.gpu_lp_solver_probe import kkt
        tick=perf_counter();a=SimpleNamespace(capacity=cap.astype(float),demand=data[:,:3].astype(float),value=data[:,3].astype(float),shapes=kind)
        model,labels=canonical_shape(a);sync_before=sync();result=solve_lp(model,backend=backend,fallback='none',time_limit=20.);sync_after=sync()
        raw=result.raw
        independent=kkt(dict(A=model.A,c=model.c,l=model.row_lo,u=model.row_hi,lb=model.var_lo,ub=model.var_hi,offset=model.c0),raw.x,raw.pi,raw.rc,raw.status=='OPTIMAL',raw.objective)
        lp=dict(backend=backend,status=result.status,certified=result.certified,independent=independent,
                canonical_sha256=model.source_hash,settings=raw.settings,version=raw.version,native_status=raw.native_status,
                nested_adapter_timing=raw.timing,error=raw.error,adapter_certificate=result.attempts[-1].certificate,
                variables=len(model.c),rows=model.A.shape[0],nnz=model.A.nnz,shape_count=len(labels),eligibility_is_relaxed=True,
                device_sync_before=sync_before,device_sync_after=sync_after)
        stages['model_lp_certification']=perf_counter()-tick
        for name in ['x','pi','rc']:
            if getattr(raw,name) is not None:vectors[name]=getattr(raw,name)
        if not (result.status=='OPTIMAL' and result.certified and independent['certified']):
            path=Path(directory)/(key+'.npz');np.savez_compressed(path,**vectors)
            return dict(status='UNCERTIFIED',lp=lp,config=asdict(c),input_sha256=h.hexdigest(),stages=stages,
                        full_path_seconds=perf_counter()-start,vectors=path.name,vectors_sha256=sha(path.read_bytes()),peak_rss_bytes=peak_rss())
        tick=perf_counter();prices=-raw.pi[c.n:].reshape(len(labels),3)
        scores=np.max(data[:,3][None,:]-prices@data[:,:3].T,axis=0)
        order=np.lexsort((np.arange(c.n),-scores));stages['ordering']=perf_counter()-tick
        tick=perf_counter()
        for offset in range(0,c.n,c.chunk):
            ids=order[offset:offset+c.chunk];place_block(data[ids],ids,rest,nf,ng,cursors,p)
        stages['placement']=perf_counter()-tick
    else:
        iterator=iter(chunks(c))
        while True:
            tick=perf_counter()
            try:offset,x=next(iterator)
            except StopIteration:break
            update_hash(h,offset,x);stages['input_gen_hash']+=perf_counter()-tick
            tick=perf_counter();place_block(x,range(offset,offset+len(x)),rest,nf,ng,cursors,p)
            stages['placement']+=perf_counter()-tick
    tick=perf_counter();residual=np.array(rest,np.int64);check=audit(c,p,residual,h.hexdigest())
    if dense and c.n<=100000:
        from m0_oracle import validate
        a=original_instance(c,data);assert validate(a,p);check['dense_reference']='PASS_ALL_JOBS'
        if c.n==10000 and not backend:
            from accelsched import solve
            reference=solve(a,'first_fit',seed=c.seed,time_limit=.2)
            assert reference.placement is not None and np.array_equal(reference.placement,p)
            check['production_first_fit_equivalence']='PASS_EXACT_PLACEMENT'
    stages['audit']=perf_counter()-tick
    tick=perf_counter();vectors.update(placement=p,residual=residual)
    path=Path(directory)/(key+'.npz');np.savez_compressed(path,**vectors)
    output=dict(placement_sha256=sha(p.tobytes()),residual_sha256=sha(residual.tobytes()),accepted_value=check['accepted_value'],accepted_count=check['accepted_count'])
    output_sha=sha(js(output));artifact_sha=sha(path.read_bytes());stages['serialization']=perf_counter()-tick
    total=perf_counter()-start
    return dict(status='PHYSICAL_E2E_CERTIFIED',track='C_INTEGRATED' if backend else 'A_PRIMARY',
        policy='EXPERIMENTAL_LP_SCORE_FIRST_FIT_POLICY_DIFFERENT' if backend else 'EXACT_INPUT_ORDER_FIRST_FIT',
        config=asdict(c),shape_count=4 if c.nodes>=4 else len(set(kind.tolist())),eligibility_groups=17 if c.case=='multiresource' else 1,
        input_sha256=h.hexdigest(),output_sha256=output_sha,output=output,physical=check,lp=lp if backend else 'N/A',
        stages=stages,full_path_seconds=total,other_seconds=total-sum(stages.values()),
        offered_jobs_per_second=c.n/total,accepted_jobs_per_second=check['accepted_count']/total,
        vectors=path.name,vectors_sha256=artifact_sha,peak_rss_bytes=peak_rss(),
        peak_rss_scope='process high-water incl imports/prior calls',peak_vram_bytes='NOT_AVAILABLE',
        shared_generation_amortization=False)


def worker(args):
    if sys.platform!='win32' and not args.backend:
        import resource
        resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
    cfg=Config(args.n,args.nodes,args.case,args.seed);directory=Path(args.out);directory.mkdir(exist_ok=False)
    rows=[]
    for rep in range(2):
        try:
            row=pipeline(cfg,directory,f'rep{rep}',args.backend)
            row.update(rep=rep,semantics='first_full_call_in_fresh_process' if rep==0 else 'warmed_same_process_full_reconstruction',
                       process_import_setup_seconds=PROCESS_IMPORT_SECONDS)
        except MemoryError:row=dict(status='OOM',rep=rep,error=traceback.format_exc(),config=asdict(cfg))
        except BaseException:row=dict(status='PHYSICAL_INVALID',rep=rep,error=traceback.format_exc(),config=asdict(cfg))
        # Complete result serialization belongs to the measured source→result path.
        tick=perf_counter();js(row)
        if 'full_path_seconds' in row:
            dt=perf_counter()-tick;row['stages']['serialization']+=dt;row['full_path_seconds']+=dt
            if row['status']=='PHYSICAL_E2E_CERTIFIED':
                row['offered_jobs_per_second']=cfg.n/row['full_path_seconds']
                row['accepted_jobs_per_second']=row['physical']['accepted_count']/row['full_path_seconds']
        row['address_space_cap_bytes']=None if args.backend or sys.platform=='win32' else 6*1024**3
        row['completion_monotonic']=perf_counter();rows.append(row)
        (directory/'results.json').write_bytes(js(rows))


def invoke(script,out,c,backend=None,timeout=120):
    argv=[sys.executable,'-B',str(script),'--worker','--out',str(out),'--n',str(c.n),'--nodes',str(c.nodes),'--case',c.case,'--seed',str(c.seed)]
    if backend:argv+=['--backend',backend]
    start=perf_counter()
    try:
        p=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=sys.platform!='win32')
        try:stdout,stderr=p.communicate(timeout=timeout);code=p.returncode;state='RETURNED'
        except subprocess.TimeoutExpired:
            if sys.platform!='win32':
                import signal
                os.killpg(p.pid,signal.SIGKILL)
            else:p.kill()
            stdout,stderr=p.communicate();code=p.returncode;state='TIMEOUT'
    except OSError as e:stdout='';stderr=repr(e);code=None;state='PROCESS_ERROR'
    rows=json.loads((out/'results.json').read_bytes()) if (out/'results.json').exists() else []
    for r in rows:
        if r.get('rep')==0:r['cold_full_process_seconds']=r['completion_monotonic']-start
    for rep in range(len(rows),2):rows.append(dict(status='TIMEOUT' if state=='TIMEOUT' else 'OOM_OR_PROCESS_ERROR',rep=rep,config=asdict(c)))
    return dict(argv=argv,timeout=timeout,process_state=state,exit_code=code,stdout=stdout,stderr=stderr,
                process_total_seconds=perf_counter()-start,records=rows)


def production_probe(out):
    from accelsched import METHODS,solve
    out=Path(out);out.mkdir(exist_ok=False);rows=[]
    for case in CASES:
        c=Config(128,8,case,11);begin=perf_counter();a=original_instance(c);build=perf_counter()-begin
        for rep in range(2):
            for method in METHODS[rep:]+METHODS[:rep]:
                tick=perf_counter();r=solve(a,method,seed=11,time_limit=.2);wall=perf_counter()-tick
                row=dict(case=case,method=method,rep=rep,status=r.status,optimal=r.optimal,solve_seconds=wall,shared_input_build_seconds=build,error=r.error)
                if r.placement is not None:
                    tick=perf_counter();check=audit(c,np.array(r.placement,np.int32),r.residual)
                    assert check['accepted_value']==r.value
                    row.update(physical=check,audit_seconds=perf_counter()-tick)
                else:row['physical']=None
                rows.append(row);(out/'results.json').write_bytes(js(rows))
    return rows


def suite(out,budget=1500):
    out=Path(out);out.mkdir(exist_ok=False);start=perf_counter();deadline=start+budget;records=[]
    report=dict(status='PARTIAL',source_sha=os.environ.get('SAE_PROJECT_SHA'),environment=dict(python=sys.version,numpy=np.__version__,
        platform=platform.platform(),affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}),attempts=records)
    def save():
        report['seconds']=perf_counter()-start;(out/'summary.json').write_bytes(js(report))
    def call(stage,c,backend=None,limit=120):
        key=f'{stage}-{c.case}-{c.seed}'+('-'+backend if backend else '')
        if perf_counter()+5>=deadline:
            records.append(dict(stage=stage,config=asdict(c),status='NOT_RUN_BATCH_BUDGET'));save();return None
        result=invoke(Path(__file__).resolve(),out/key,c,backend,min(limit,deadline-perf_counter()))
        result.update(stage=stage,key=key,config=asdict(c));records.append(result);save();return result
    try:
        tick=perf_counter();p=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--production','--out',str(out/'production')],capture_output=True,text=True,timeout=30)
        report['production_reference']=dict(exit_code=p.returncode,seconds=perf_counter()-tick,stdout=p.stdout,stderr=p.stderr,
            records=json.loads((out/'production/results.json').read_bytes()) if (out/'production/results.json').exists() else [])
    except subprocess.TimeoutExpired:report['production_reference']=dict(status='TIMEOUT',timeout=30)
    save()
    for stage,n,nodes,seeds in [('S0',10000,256,[11]),('S1',100000,1024,[11,29,47]),('S2',1000000,4096,[11])]:
        for case in CASES:
            for seed in seeds:call(stage,Config(n,nodes,case,seed))
    s2=[r for a in records if a['stage']=='S2' for r in a.get('records',[])]
    if len(s2)==8 and all(r['status']=='PHYSICAL_E2E_CERTIFIED' and r['full_path_seconds']<40 and r['peak_rss_bytes'] is not None and r['peak_rss_bytes']<4*1024**3 for r in s2):
        for case in CASES:call('S3',Config(10000000,8192,case,11),limit=120)
    else:report['S3']='NOT_RUN_OBSERVED_S2_BOUND'
    # Paired same-host LP diagnostics only after mandatory physical milestones.
    for stage,n,nodes,cases in [('C_S0',10000,256,CASES),('C_S1',100000,1024,('balanced','multiresource'))]:
        for ci,case in enumerate(cases):
            backends=['highs','hpdlp','cupdlpx'];backends=backends[ci%3:]+backends[:ci%3]
            for backend in backends:call(stage,Config(n,nodes,case,11),backend,limit=65)
    report['status']='COMPLETED_WITH_EXPLICIT_STATUSES';save();return report


PROCESS_IMPORT_SECONDS=perf_counter()-PROCESS_START
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--worker',action='store_true');p.add_argument('--production',action='store_true');p.add_argument('--n',type=int,default=100000)
    p.add_argument('--nodes',type=int,default=1024);p.add_argument('--case',choices=CASES,default='balanced')
    p.add_argument('--seed',type=int,default=11);p.add_argument('--backend',choices=['highs','hpdlp','cupdlpx'])
    p.add_argument('--budget',type=float,default=1500)
    args=p.parse_args()
    if args.production:production_probe(args.out)
    elif args.worker:worker(args)
    else:suite(args.out,args.budget)

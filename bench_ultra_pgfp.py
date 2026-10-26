"""Opt-in exact indexed PGFP / named lp_best_fit with certified Shape prices.

No production modification. Equivalence is conditional on IDENTICAL prices and
the fixed integer future catalogue and family/group predicate. GPU-certified
duals may differ from legacy HiGHS: that is recorded as policy-different, not
PGFP_EXACT_BASELINE. No First-Fit cursor is used as a PGFP score substitute.
"""
from time import perf_counter
PROCESS_START=perf_counter()
import argparse
from collections import Counter
import heapq
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
import numpy as np
import bench_v4_gpu as base

FUTURE=np.array([[1,8,32],[4,32,128],[1,8,32],[0,8,32]],np.int64)
WEIGHTS=np.array([1,2,1,1],np.int64)
CASES=('high_pressure','balanced','fragmentation','multiresource')
POLICIES=('pgfp','lp_best_fit')


def future_eligibility(nf):
    return (np.array([0,0,1,-1])[None,:]<0)|(nf[:,None]==np.array([0,0,1,-1])[None,:])


class CandidateSet:
    """Exact cost→node bitsets, not stale optimistic heap scores.

    Every affected candidate key is recomputed after ANY residual change,
    including negative-marginal→zero-marginal improvements. Empty heap cost
    keys are lazy-pruned; node membership is eager/exact and never stale.
    """
    def __init__(self,index,demand,shape,family,group):
        self.index=index;self.d=tuple(map(int,demand));self.cost=[-1]*len(index.rest)
        self.buckets={};self.heap=[];self.enqueued=set()
        mask=(index.shapes==shape)&base.allowed(family,group,index.nf,index.ng)
        ids=np.flatnonzero(mask&np.all(index.rest>=self.d,axis=1))
        index.counters['registration_node_tests']+=len(mask)
        for node in ids:
            i=int(node);self.put(i,self.evaluate(i))
    def evaluate(self,node):
        ix=self.index;ix.counters['candidate_cost_evaluations']+=1
        r=ix.rows[node];g,c,m=self.d
        if r[0]<g or r[1]<c or r[2]<m:return -1
        if ix.policy=='lp_best_fit':return ix.sums[node]
        # Integral catalogue/weights make weighted lost suitability an exact key.
        before=int(ix.masks[node]);after=(int(r[0])-g,int(r[1])-c,int(r[2])-m)
        if before==0:return 0
        maximum=ix.maximum[before]
        if after[0]>=maximum[0] and after[1]>=maximum[1] and after[2]>=maximum[2]:return 0
        lost=0
        for k,f in enumerate(ix.future_rows):
            if before&(1<<k) and (after[0]<f[0] or after[1]<f[1] or after[2]<f[2]):lost|=1<<k
        return ix.loss[lost]
    def put(self,node,new):
        old=int(self.cost[node])
        if old==new:return
        bit=1<<node
        if old>=0:
            bits=self.buckets[old]&~bit
            if bits:self.buckets[old]=bits
            else:del self.buckets[old]
        self.cost[node]=new
        if new>=0:
            self.buckets[new]=self.buckets.get(new,0)|bit
            if new not in self.enqueued:heapq.heappush(self.heap,new);self.enqueued.add(new)
        self.index.counters['changed_candidate_keys']+=1
    def update(self,node):
        if self.cost[node]>=0:
            self.index.counters['candidate_update_visits']+=1
            self.put(node,self.evaluate(node))
        # Infeasible candidates never regain capacity in this STATIC run.
    def best(self):
        while self.heap and self.heap[0] not in self.buckets:self.enqueued.remove(heapq.heappop(self.heap))
        if not self.heap:return None
        bits=self.buckets[self.heap[0]]
        return (bits&-bits).bit_length()-1


class ExactIndex:
    def __init__(self,capacity,shapes,nf,ng,policy,future=FUTURE,weights=WEIGHTS,eligible=None):
        assert policy in POLICIES
        assert np.all(future==np.floor(future)) and np.all(weights==np.floor(weights)) and np.all(weights>=0)
        assert np.isfinite(capacity).all() and np.all(capacity>=0) and np.all(capacity==np.floor(capacity))
        self.rest=np.array(capacity,np.int64,copy=True);self.rows=[tuple(map(int,r)) for r in capacity];self.sums=[sum(r) for r in self.rows]
        self.shapes=np.asarray(shapes);self.nf=np.asarray(nf);self.ng=np.asarray(ng)
        self.policy=policy;self.future=np.asarray(future);self.weights=np.asarray(weights)
        self.fe=future_eligibility(self.nf) if eligible is None else np.array(eligible,bool)
        self.masks=[0]*len(capacity);self.future_rows=[tuple(map(int,f)) for f in future]
        self.node_futures=[[(k,*self.future_rows[k]) for k in range(len(weights)) if row[k]] for row in self.fe]
        self.loss=[int(sum(int(weights[k]) for k in range(len(weights)) if mask&(1<<k))) for mask in range(1<<len(weights))]
        self.maximum=[tuple(map(int,np.max(future[[k for k in range(len(weights)) if mask&(1<<k)]],axis=0))) if mask else (0,0,0) for mask in range(1<<len(weights))]
        self.queries={};self.watchers={};self.counters=Counter()
        for i in range(len(capacity)):self.refresh_future(i)
    def refresh_future(self,node):
        r=self.rows[node];mask=0
        for k,g,c,m in self.node_futures[node]:
            if r[0]>=g and r[1]>=c and r[2]>=m:mask|=1<<k
        self.masks[node]=mask
    def query(self,demand,shape,family,group):
        key=(*map(int,demand),int(shape),int(family),int(group));self.counters['index_queries']+=1
        if key not in self.queries:
            q=CandidateSet(self,demand,shape,family,group);self.queries[key]=q
            self.watchers.setdefault((int(shape),int(family),int(group)),[]).append(q)
        return self.queries[key].best()
    def commit(self,node,demand):
        old=self.rows[node];d=tuple(map(int,demand));new=tuple(old[k]-d[k] for k in range(3));assert min(new)>=0
        self.rows[node]=new;self.sums[node]=sum(new);self.rest[node]=new
        self.refresh_future(node);self.counters['node_commits']+=1
        for f in set([-1,int(self.nf[node])]):
            for group in set([-1,int(self.ng[node])]):
                for q in self.watchers.get((int(self.shapes[node]),f,group),[]):q.update(node)


def orders(data,prices):
    utility=data[:,3][None,:]-prices@data[:,:3].T
    task=np.argsort(-np.max(utility,axis=0),kind='stable')
    shape=np.argsort(-utility,axis=0,kind='stable')
    return task,shape


def replay(cap,data,shapes,nf,ng,prices,policy,trace=False,future=FUTURE,weights=WEIGHTS,fe=None):
    assert np.isfinite(data).all() and np.all(data[:,:3]>=0) and np.all(data[:,:3]==np.floor(data[:,:3]))
    assert np.isfinite(prices).all() and prices.shape[1]==3
    tick=perf_counter();task,shape_order=orders(data,prices);ordering=perf_counter()-tick
    tick=perf_counter();index=ExactIndex(cap,shapes,nf,ng,policy,future,weights,fe)
    p=np.full(len(data),-1,np.int32);ledger=[]
    for j in task:
        row=data[j];before=index.rest.copy() if trace else None
        for h in shape_order[:,j]:
            node=index.query(row[:3],int(h),row[4],row[5])
            if node is not None:
                p[j]=node;index.commit(node,row[:3]);break
        if trace:ledger.append(dict(job=int(j),before=before.tolist(),after=index.rest.tolist(),chosen=int(p[j]),shape_order=shape_order[:,j].tolist()))
    residual=index.rest;info=dict(ordering=ordering,placement=perf_counter()-tick,
        task_order_sha256=base.sha(task.tobytes()),shape_order_sha256=base.sha(shape_order.tobytes()),
        candidate_counters=dict(index.counters),registered_query_count=len(index.queries),ledger=ledger)
    # Break index↔query ownership cycles between full cold/warm reconstructions.
    index.watchers.clear();index.queries.clear()
    return p,residual,info


def tighter_native(model,backend):
    """One declared research-only public Model retry; original certificate unchanged."""
    from lp_backends import RawResult,Attempt,CertifiedResult
    from lp_certify import certify
    from lp_contract import CanonicalLP
    tick=perf_counter();module=importlib.import_module(backend)
    p=module.Model(objective_vector=model.c,constraint_matrix=model.A,constraint_lower_bound=model.row_lo,
        constraint_upper_bound=model.row_hi,variable_lower_bound=model.var_lo,variable_upper_bound=model.var_hi,objective_constant=model.c0)
    settings=dict(TimeLimit=20.,OptimalityTol=1e-14,FeasibilityTol=1e-14,OptimalityNorm='linf',Presolve=False,OutputFlag=False)
    p.ModelSense=module.PDLP.MINIMIZE;p.setParams(**settings)
    assert CanonicalLP(p.A,p.c,p.constr_lb,p.constr_ub,p.lb,p.ub,p.c0).source_hash==model.source_hash
    p.optimize();name=str(p.StatusName);status={'OPTIMAL':'OPTIMAL','TIME_LIMIT':'TIME_LIMIT'}.get(name,'ERROR')
    raw=RawResult(backend,importlib.metadata.version(backend),status,dict(name=name,code=p.Status,native_runtime=p.Runtime),
        *[None if v is None else np.array(v,float,copy=True) for v in [p.X,p.Pi,p.RC]],
        objective=p.ObjVal,dual_objective=p.DualObj,gap=p.Gap,settings=settings)
    model.assert_intact();check=certify(model,raw)
    return CertifiedResult('OPTIMAL' if check['certified'] else 'NUMERIC_UNCERTIFIED' if status=='OPTIMAL' else status,
        (Attempt(raw,status,check,perf_counter()-tick),),perf_counter()-tick,model.source_hash)


def audit_extra(c,data,p,residual,base_sha):
    check=base.audit(c,p,residual,base_sha);cap,nf,ng,_=base.inventory(c)
    fe=future_eligibility(nf);fit=np.all(residual[:,None,:]>=FUTURE[None,:,:],axis=2)&fe
    check['weighted_future_placeability']=int(np.sum(fit*WEIGHTS))
    check['accepted_gpu_jobs']=int(np.sum((p>=0)&(data[:,0]>0)))
    check['full_input_sha256']=base.sha(base.js(dict(base_input_sha256=base_sha,future=FUTURE.tolist(),weights=WEIGHTS.tolist(),future_eligibility_sha256=base.sha(fe.tobytes()))))
    return check


def pipeline(c,out,key,policy='pgfp',backend='cupdlpx'):
    start=perf_counter();out=Path(out);stages={k:0. for k in ['generation_hash','canonical','solver_certificate','ordering','placement','audit','serialization']}
    def mark(stage):
        (out/(key+'-phase.json')).write_bytes(base.js(dict(stage=stage,config=asdict(c),policy=policy,seconds=perf_counter()-start)))
    from dataclasses import asdict
    mark('INPUT_GENERATION_HASH');tick=perf_counter();cap,nf,ng,shapes=base.inventory(c);data=base.materialize(c);h=base.fingerprint_start(c,cap)
    for i in range(0,c.n,c.chunk):base.update_hash(h,i,data[i:i+c.chunk])
    stages['generation_hash']=perf_counter()-tick
    from certified_pgfp import canonical_shape
    from lp_backends import solve_lp
    from probes.gpu_lp_solver_probe import kkt
    mark('SPARSE_ORIGINAL_CANONICAL');tick=perf_counter()
    model,labels=canonical_shape(SimpleNamespace(capacity=cap.astype(float),demand=data[:,:3].astype(float),value=data[:,3].astype(float),shapes=shapes))
    stages['canonical']=perf_counter()-tick;mark('NATIVE_SOLVER_ORIGINAL_KKT');tick=perf_counter();base.sync()
    result=solve_lp(model,backend=backend,fallback='none',time_limit=20.);base.sync();attempts=[]
    def entry(r):return dict(status=r.status,settings=r.raw.settings,native=r.raw.native_status,certificate=r.attempts[-1].certificate,error=r.raw.error,version=r.raw.version,nested_timing=r.raw.timing)
    attempts.append(entry(result))
    if result.status=='NUMERIC_UNCERTIFIED' and backend in ('hpdlp','cupdlpx'):
        mark('ONE_TIGHTER_NATIVE_REPAIR_SAME_ORIGINAL_MODEL');result=tighter_native(model,backend);base.sync();attempts.append(entry(result))
    raw=result.raw
    independent=kkt(dict(A=model.A,c=model.c,l=model.row_lo,u=model.row_hi,lb=model.var_lo,ub=model.var_hi,offset=model.c0),
        raw.x,raw.pi,raw.rc,raw.status=='OPTIMAL',raw.objective)
    stages['solver_certificate']=perf_counter()-tick
    vectors={name:getattr(raw,name) for name in ['x','pi','rc'] if getattr(raw,name) is not None}
    lp=dict(status=result.status,backend=backend,canonical_sha256=model.source_hash,attempts=attempts,independent=independent,
            rows=model.A.shape[0],variables=model.A.shape[1],nnz=model.A.nnz,shape_count=len(labels),eligibility_relaxed=True)
    row=dict(config=base.asdict(c),policy=policy,lp=lp,input_sha256=h.hexdigest(),stages=stages,
             status='UNCERTIFIED',policy_status='NOT_RUN',peak_vram_bytes='NOT_AVAILABLE')
    if result.status=='OPTIMAL' and result.certified and independent['certified']:
        prices=-raw.pi[c.n:].reshape(len(labels),3);mark('GLOBAL_ORDER_AND_EXACT_CANDIDATE_INDEX')
        p,residual,physical=replay(cap,data,shapes,nf,ng,prices,policy)
        stages.update(ordering=physical['ordering'],placement=physical['placement']);mark('INDEPENDENT_ALL_JOB_PHYSICAL_AUDIT');tick=perf_counter()
        check=audit_extra(c,data,p,residual,h.hexdigest());stages['audit']=perf_counter()-tick
        row.update(status='PHYSICAL_E2E_VALID',policy_status='PGFP_CERTIFIED_PRICE_POLICY_DIFFERENT' if policy=='pgfp' else 'LP_GUIDED_POLICY_DIFFERENT',
                   physical=check,order={k:v for k,v in physical.items() if k not in ['ledger','ordering','placement']},price_sha256=base.sha(prices.tobytes()))
        vectors.update(placement=p,residual=residual,prices=prices)
    mark('LOSSLESS_SERIALIZATION');tick=perf_counter();path=out/(key+'.npz');np.savez_compressed(path,**vectors)
    row.update(vectors=path.name,vectors_sha256=base.sha(path.read_bytes()))
    if row['status']=='PHYSICAL_E2E_VALID':
        output={k:row['physical'][k] for k in ['accepted_count','accepted_value','accepted_gpu_jobs','weighted_future_placeability','residual_sha256']}
        output['placement_sha256']=base.sha(vectors['placement'].tobytes());row['output']=output;row['output_sha256']=base.sha(base.js(output))
    stages['serialization']=perf_counter()-tick;total=perf_counter()-start
    row.update(full_seconds=total,other_seconds=total-sum(stages.values()),peak_rss_bytes=base.peak_rss())
    if row['status']=='PHYSICAL_E2E_VALID':row.update(offered_jobs_ps=c.n/total,accepted_jobs_ps=row['physical']['accepted_count']/total)
    mark('COMPLETE');return row


def worker(args):
    out=Path(args.out);out.mkdir(exist_ok=False);rows=[]
    c=base.Config(args.n,args.nodes,args.case,args.seed)
    for rep in range(2):
        try:
            if args.policy=='first_fit':
                row=base.pipeline(c,out,f'rep{rep}');row['status']='PHYSICAL_E2E_VALID' if row['status']=='PHYSICAL_E2E_CERTIFIED' else row['status'];row['policy']='unchanged_streaming_first_fit_baseline';row['full_seconds']=row.get('full_path_seconds')
            else:row=pipeline(c,out,f'rep{rep}',args.policy,args.backend)
        except MemoryError:row=dict(status='OOM',error=traceback.format_exc(),config=base.asdict(c),policy=args.policy)
        except BaseException:
            path=out/f'rep{rep}-phase.json';phase=json.loads(path.read_bytes()).get('stage') if path.exists() else 'STARTUP'
            row=dict(status='PHYSICAL_INVALID' if phase in ('GLOBAL_ORDER_AND_EXACT_CANDIDATE_INDEX','INDEPENDENT_ALL_JOB_PHYSICAL_AUDIT') else 'ERROR',failed_stage=phase,error=traceback.format_exc(),config=base.asdict(c),policy=args.policy)
        row.update(rep=rep,completion_monotonic=perf_counter());rows.append(row)
        (out/'results.json').write_bytes(base.js(rows))


def invoke(out,c,policy,backend='cupdlpx',limit=180):
    args=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--out',str(out),'--n',str(c.n),'--nodes',str(c.nodes),'--case',c.case,'--seed',str(c.seed),'--policy',policy,'--backend',backend]
    tick=perf_counter();p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=sys.platform!='win32');timed_out=False
    try:stdout,stderr=p.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        timed_out=True
        if sys.platform!='win32':
            import signal
            os.killpg(p.pid,signal.SIGKILL)
        else:p.kill()
        stdout,stderr=p.communicate()
    rows=json.loads((out/'results.json').read_bytes()) if (out/'results.json').exists() else []
    for r in rows:
        if r['rep']==0:r['cold_process_seconds']=r['completion_monotonic']-tick
    for rep in range(len(rows),2):rows.append(dict(rep=rep,status='TIMEOUT' if timed_out else 'OOM_OR_PROCESS_ERROR',config=base.asdict(c),policy=policy))
    return dict(argv=args,watchdog=limit,process_seconds=perf_counter()-tick,exit_code=p.returncode,stdout=stdout,stderr=stderr,records=rows)


def suite(out,budget=2100):
    out=Path(out);out.mkdir(exist_ok=False);start=perf_counter();report=dict(source_sha=os.environ.get('SAE_PROJECT_SHA'),status='PARTIAL',attempts=[])
    def save():
        report['seconds']=perf_counter()-start;(out/'summary.json').write_bytes(base.js(report))
    def call(stage,c,policy,backend='cupdlpx'):
        key=f'{stage}-{c.case}-{c.seed}-{policy}-{backend}'
        remaining=budget-(perf_counter()-start)
        if remaining<5:result=dict(status='NOT_RUN_BATCH_BUDGET',records=[dict(rep=i,status='NOT_RUN',config=base.asdict(c),policy=policy) for i in [0,1]])
        else:result=invoke(out/key,c,policy,backend,min(180 if c.n>=1000000 else 90,remaining))
        result.update(stage=stage,key=key,config=base.asdict(c),policy=policy,backend=backend);report['attempts'].append(result);save()
    # Strong GPU-pressure S1 first, then mandatory S2 before the remaining S1 series.
    for case in CASES:
        for policy in ('first_fit',*POLICIES):call('S0',base.Config(10000,256,case,11),policy)
    for policy in ('first_fit',*POLICIES):call('S1',base.Config(100000,1024,'high_pressure',11),policy)
    for case in CASES:
        for policy in ('first_fit',*POLICIES):call('S2',base.Config(1000000,4096,case,11),policy)
    for case in CASES:
        for seed in [11,29,47]:
            if case=='high_pressure' and seed==11:continue
            for policy in ('first_fit',*POLICIES):call('S1',base.Config(100000,1024,case,seed),policy)
    for case in ['high_pressure','multiresource']:
        for backend in ['highs','hpdlp']:call('SAME_HOST_DIAGNOSTIC',base.Config(100000,1024,case,11),'pgfp',backend)
    report['status']='COMPLETED_WITH_EXPLICIT_STATUSES';save();return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);p.add_argument('--worker',action='store_true')
    p.add_argument('--n',type=int,default=100000);p.add_argument('--nodes',type=int,default=1024);p.add_argument('--case',choices=CASES,default='high_pressure')
    p.add_argument('--seed',type=int,default=11);p.add_argument('--policy',choices=['first_fit',*POLICIES],default='pgfp');p.add_argument('--backend',choices=['highs','hpdlp','cupdlpx'],default='cupdlpx')
    p.add_argument('--budget',type=float,default=2100);args=p.parse_args()
    if args.worker:worker(args)
    else:suite(args.out,args.budget)

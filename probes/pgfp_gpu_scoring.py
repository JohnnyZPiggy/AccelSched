"""P6A independent AIDC PGFP array probe; optional CuPy, immutable production.

GPU signed future-fit differences return to NumPy for production-order reduction.
CPU utility ordering/ties and sequential commits are authoritative and timed.
Frozen batched scores are NOT simultaneous physical placement.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import pgfp_dual_stability as g
from probes.shape_lp_gpu_scale import Memory
from probes.shape_lp_gpu_convergence import utilization

P4G_SHA='c15ae064f39ca1301379f9de38b459abd3610cf515db9ebbca1ceb026bf7f16f'
FIELDS=('task_order','shape_order','utility','trace','ledger_steps','placement','residual',
        'greedy_residual','accepted_jobs','accepted_count','value','future_score')


def edge_case():
    return g.oracle.Instance(np.ones((2,3)),np.array([[1,1,1],[1,1,1],[2,0,0],[0,0,0]],float),
        np.array([1.,1.,1.,0.]),np.array([[1,1,1,0],[1,0,1,0]],bool),np.array([7,7]),
        np.array([[1,1,1],[0,0,0]],float),np.array([1.,.1]),np.ones((2,2),bool))


def compare(reference,result):
    flags={k:reference[k]==result[k] for k in FIELDS}
    old=[v for s in reference['ledger_steps'] for a in s['attempts'] for v in a['scores']]
    new=[v for s in result['ledger_steps'] for a in s['attempts'] for v in a['scores']]
    return dict(status='EXACT' if all(flags.values()) else 'DIFF',exact_flags=flags,
        score_max_error=max((abs(x-y) for x,y in zip(old,new)),default=0.) if len(old)==len(new) else None)


class Engine:
    def __init__(self,a,backend,cp=None):
        self.a=a;self.xp=np if backend=='numpy' else cp;self.gpu=backend=='cupy'
        if self.xp is None:raise ValueError('Explicit optional GPU library required')
        self.t=dict(host_to_device=0.,array_compute_sync=0.,device_event_seconds=0.,device_to_host=0.,
                    canonical_reduction=0.,selection=0.,sequential_commit=0.)
        tick=perf_counter();xp=self.xp
        self.rest=xp.asarray(a.capacity.copy());self.d=xp.asarray(a.demand);self.q=xp.asarray(a.future)
        self.fe=xp.asarray(a.future_eligible);self.eligible=xp.asarray(a.eligible)
        self.sync();self.t['host_to_device']+=perf_counter()-tick
    def sync(self):
        if self.gpu:self.xp.cuda.get_current_stream().synchronize()
    def compute(self,fn):
        tick=perf_counter()
        if self.gpu:
            begin=self.xp.cuda.Event();end=self.xp.cuda.Event();begin.record()
        result=fn()
        if self.gpu:
            end.record();end.synchronize();self.t['device_event_seconds']+=self.xp.cuda.get_elapsed_time(begin,end)/1000
        self.sync();self.t['array_compute_sync']+=perf_counter()-tick;return result
    def host(self,*arrays):
        tick=perf_counter();out=tuple(self.xp.asnumpy(a) if self.gpu else np.asarray(a) for a in arrays)
        self.t['device_to_host']+=perf_counter()-tick;return out
    def candidates(self,j):
        xp=self.xp
        def fn():
            valid=xp.all(self.rest+1e-10>=self.d[j],axis=1)&self.eligible[:,j]
            before=xp.all(self.rest[:,None,:]+1e-10>=self.q[None,:,:],axis=2)&self.fe
            after=xp.all((self.rest-self.d[j])[:,None,:]+1e-10>=self.q[None,:,:],axis=2)&self.fe
            return valid,after.astype(xp.int8)-before.astype(xp.int8)
        valid,diff=self.host(*self.compute(fn));tick=perf_counter()
        # Exact same one-dimensional NumPy reduction as original marginal, including non-dyadic weights.
        scores=np.array([float(np.sum(row*self.a.weights)) for row in diff])
        self.t['canonical_reduction']+=perf_counter()-tick;return valid,scores
    def commit(self,i,row):
        tick=perf_counter();self.rest[i]=self.xp.asarray(row);self.sync()
        self.t['sequential_commit']+=perf_counter()-tick


def replay(a,certificate,backend,cp=None):
    start=perf_counter();c=g.admit(a,certificate['x'],certificate['raw_pi'],certificate['raw_rc'],certificate['pi_sign'],certificate['reported_optimal'])
    admitted=perf_counter();labels=list(dict.fromkeys(a.shapes.tolist()));lam=np.array(c['capacity_duals'])
    u=a.value[None,:]-lam@a.demand.T
    order=sorted(range(len(a.value)),key=lambda j:(-max(u[:,j]),j))
    shapes=[sorted(range(len(labels)),key=lambda h:(-u[h,j],h)) for j in range(len(a.value))]
    preprocessed=perf_counter();engine=Engine(a,backend,cp);utility_error=None;device_utility_seconds=0.
    if backend=='cupy':
        tick=perf_counter();ld=cp.asarray(lam);vd=cp.asarray(a.value);engine.sync();engine.t['host_to_device']+=perf_counter()-tick
        tick=perf_counter();gu=engine.compute(lambda:vd[None,:]-ld@engine.d.T);gu,=engine.host(gu)
        device_utility_seconds=perf_counter()-tick;utility_error=float(np.max(np.abs(gu-u)))
        # GPU utility is diagnostic only: original CPU utility determines every actual order.
    rest=a.capacity.astype(float).copy();steps=[];trace=[];loop=perf_counter()
    for j in order:
        valid,scores=engine.candidates(j);tick=perf_counter();step=dict(task=j,before=rest.tolist(),attempts=[],chosen=-1)
        for h in shapes[j]:
            ids=np.flatnonzero(valid&(a.shapes==labels[h])).tolist();values=scores[ids].tolist()
            step['attempts'].append(dict(shape=labels[h],candidates=ids,scores=values))
            if not ids:continue
            chosen=max(ids,key=lambda i:(scores[i],-i));step['chosen']=chosen
            trace.append(dict(task=j,shape=labels[h],candidates=ids,scores=values,chosen=chosen));break
        engine.t['selection']+=perf_counter()-tick
        if step['chosen']>=0:
            i=step['chosen'];tick=perf_counter()
            if not a.eligible[i,j] or not np.all(rest[i]+1e-10>=a.demand[j]):raise ValueError('stale/ineligible commit')
            rest[i]-=a.demand[j];engine.t['sequential_commit']+=perf_counter()-tick;engine.commit(i,rest[i])
        step['after']=rest.tolist();steps.append(step)
    dispatched=perf_counter();result=g.audit_steps(a,steps,order,shapes);audited=perf_counter()
    result.update(trace=trace,ledger_steps=steps,utility=u.tolist(),shape_order=shapes,
        timing=dict(input_certificate=admitted-start,canonical_utility_order=preprocessed-admitted,
                    backend_setup=loop-preprocessed,greedy=dispatched-loop,physical_audit=audited-dispatched,total=audited-start,
                    components=engine.t,gpu_utility_diagnostic_seconds=device_utility_seconds),
        gpu_utility_max_error=utility_error,
        policy='Host canonical utility/signed weighted reduction/ties; GPU eligibility/capacity/future predicates; host revalidation and sequential commit.')
    return result


def ladder(n,nodes):
    rng=np.random.default_rng(np.random.SeedSequence([79,n,nodes]))
    cap=np.array([[4,16,64],[8,32,128],[2,8,32],[4,12,96]],float)[np.arange(nodes)%4]
    demand=np.column_stack((rng.integers(1,5,n),rng.integers(1,17,n)/2,rng.integers(1,65,n)/2))
    return dict(capacity=cap,demand=demand,value=rng.integers(1,13,n).astype(float),shapes=np.arange(nodes)%8,
        future=np.array([[1,2,4],[2,4,8],[4,8,16],[1,.5,.5],[0,2,8],[0,0,0]],float),
        weights=np.array([1,.5,2,.25,1,0.]),future_eligible=np.arange(nodes)[:,None]%2==np.arange(6)[None,:]%2)


def prefix(data,k=64):
    s=len(data['capacity']);n=min(k,len(data['demand']))
    return g.oracle.Instance(data['capacity'],data['demand'][:n],data['value'][:n],
        np.arange(s)[:,None]%4==np.arange(n)[None,:]%4,data['shapes'],data['future'],data['weights'],data['future_eligible'])


def frozen(data,backend,chunk,cp=None):
    start=perf_counter();xp=np if backend=='numpy' else cp;gpu=backend=='cupy';n=len(data['demand']);s=len(data['capacity'])
    sync=lambda:xp.cuda.get_current_stream().synchronize() if gpu else None
    times=dict(assembly=0.,host_to_device=0.,array_compute_sync=0.,device_to_host=0.,host_reduction_hash=0.,scalar_validation=0.)
    tick=perf_counter();R=xp.asarray(data['capacity']);Q=xp.asarray(data['future']);FE=xp.asarray(data['future_eligible']);sync();times['host_to_device']+=perf_counter()-tick
    h=hashlib.sha256();valid_count=0;score_total=0.;checked=0
    rng=np.random.default_rng(79);samples=list(zip(rng.integers(0,n,256).tolist(),rng.integers(0,s,256).tolist()))
    oracle_instance=g.oracle.Instance(data['capacity'],data['demand'],data['value'],np.broadcast_to(True,(s,n)),data['shapes'],data['future'],data['weights'],data['future_eligible'])
    for lo in range(0,n,chunk):
        hi=min(n,lo+chunk);tick=perf_counter();elig=np.arange(lo,hi)[:,None]%4==np.arange(s)[None,:]%4
        times['assembly']+=perf_counter()-tick;tick=perf_counter();D=xp.asarray(data['demand'][lo:hi]);E=xp.asarray(elig);sync();times['host_to_device']+=perf_counter()-tick
        tick=perf_counter()
        valid=xp.all(R[None,:,:]+1e-10>=D[:,None,:],axis=2)&E
        before=xp.all(R[:,None,:]+1e-10>=Q[None,:,:],axis=2)&FE
        after=xp.all((R[None,:,None,:]-D[:,None,None,:])+1e-10>=Q[None,None,:,:],axis=3)&FE[None,:,:]
        diff=after.astype(xp.int8)-before[None,:,:].astype(xp.int8);sync();times['array_compute_sync']+=perf_counter()-tick
        tick=perf_counter()
        if gpu:valid,diff=xp.asnumpy(valid),xp.asnumpy(diff)
        times['device_to_host']+=perf_counter()-tick;tick=perf_counter()
        scores=np.sum(diff*data['weights'],axis=2)
        h.update(np.ascontiguousarray(valid).tobytes());h.update(scores.tobytes());valid_count+=int(valid.sum());score_total+=float(scores.sum())
        times['host_reduction_hash']+=perf_counter()-tick
        tick=perf_counter()
        for j,i in samples:
            if lo<=j<hi:
                expected=(i%4==j%4) and bool(np.all(data['capacity'][i]+1e-10>=data['demand'][j]))
                if bool(valid[j-lo,i])!=expected or scores[j-lo,i]!=g.oracle.marginal(oracle_instance,data['capacity'],i,j):raise ValueError('frozen scalar Oracle DIFF')
                checked+=1
        times['scalar_validation']+=perf_counter()-tick
    times['total']=perf_counter()-start
    return dict(status='PASS',digest=h.hexdigest(),valid_pairs=valid_count,score_total=score_total,timing=times,
        candidate_pairs=s*n,scalar_samples_checked=checked,chunk_jobs=chunk,peak_chunk_pair_count=s*min(n,chunk),
        meaning='Frozen original residuals, all pairs streamed and hashed. Not sequential placement throughput.')


def scalar_samples(data,count=256):
    n=len(data['demand']);s=len(data['capacity']);a=prefix(data,min(n,64));rng=np.random.default_rng(79);tick=perf_counter()
    # Sample the complete ladder, not just prefix jobs; scalar production marginal on each actual demand.
    for _ in range(count):
        j=int(rng.integers(n));i=int(rng.integers(s));d=data['demand'][j]
        b=np.all(data['capacity'][i]+1e-10>=data['future'],axis=1)&data['future_eligible'][i]
        aft=np.all((data['capacity'][i]-d)+1e-10>=data['future'],axis=1)&data['future_eligible'][i]
        expected=float(np.sum((aft.astype(int)-b.astype(int))*data['weights']))
        sample=replace(a,demand=np.array([d]),value=np.array([1.]),eligible=np.ones((s,1),bool))
        if expected!=g.oracle.marginal(sample,sample.capacity,i,0):raise ValueError('scalar sample mismatch')
    return dict(sample_count=count,seconds=perf_counter()-tick,status='EXACT',scope='Scalar sample only; no full scalar100K timing/extrapolation.')


def pack(a):return {k:getattr(a,k).tolist() for k in a.__dataclass_fields__}


def unpack(raw):
    return g.oracle.Instance(**{k:np.array(v,dtype=bool if k in ('eligible','future_eligible') else int if k=='shapes' else float) for k,v in raw.items()})


def case(a,c,cp):
    records={};reference=g.replay(a,c)
    for backend in ('scalar','numpy','cupy'):
        records[backend]=[]
        for repetition in range(2):
            with Memory() as mem,utilization() as util:
                result=g.replay(a,c) if backend=='scalar' else replay(a,c,backend,cp)
            check=compare(reference,result)
            records[backend].append(dict(repetition=repetition,label='first' if repetition==0 else 'warm',
                comparison=check,result=result,memory=mem.report(),utilization_peak=util['sampled_peak_percent']))
    status='EXACT' if all(r['comparison']['status']=='EXACT' for rows in records.values() for r in rows) else 'DIFF'
    return dict(status=status,fixture_sha256=g.p4b.fixture_hash(a),instance=pack(a),certificate=c,records=records)


def run(directory,p4g_path,budget=240.):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import cupy as cp
    imported=perf_counter();tick=perf_counter();v=cp.arange(32,dtype=cp.float64);observed=float(cp.asnumpy(v.sum()));cp.cuda.get_current_stream().synchronize()
    if observed!=496.:raise ValueError('real GPU smoke mismatch')
    report=dict(status='PASS',mode='PROBE_ONLY',backends=['scalar_numpy','vectorized_numpy','cupy_assisted'],
        cupy_import_seconds=imported-start,gpu_smoke_seconds=perf_counter()-tick,experiment_a=[],ladder=[],
        library=dict(cupy=cp.__version__,runtime=cp.cuda.runtime.runtimeGetVersion(),driver=cp.cuda.runtime.driverGetVersion()),
        limits=dict(worker_seconds=budget,warm_repeats=1,sequential_prefix_jobs=64,max_chunk_jobs=1024),
        policy='Host canonical utility/order/weighted sum/ties and sequential commits remain authoritative. '
               'Device utilities are diagnostic, not a claimed replacement for NumPy BLAS ordering. '
               'Frozen scoring and complete64-job prefix replays reported separately.')
    prior=g.checked_json(Path(p4g_path).read_bytes(),P4G_SHA);traces={}
    def persist():
        report['seconds']=perf_counter()-start;(directory/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    cases=[]
    for name,a in g.scenarios().items():
        if g.p4b.fixture_hash(a)!=g.p4b.fixture_hash(unpack(prior['cases'][name]['instance'])):raise ValueError('P4G physical fixture changed')
        control=g.control(a)
        for side in ('left','right'):
            c=prior['cases'][name]['paths'][side]['certificate'];cases.append((name+'_'+side,a,c,dict(kind='REGISTERED_P4G',registered_sha256=P4G_SHA,side=side,control=control['status'])))
    for name,a in [('epsilon_cliff',g.p4b.fixed_cases()[0][1]),('tie_competition',edge_case())]:
        ctrl=g.control(a);cases.append((name,a,ctrl['certificate'],dict(kind='LIVE_CPU_HIGHS_CONTROL',control=ctrl['status'],shared_control_seconds=ctrl['total_seconds'])))
    for name,a,c,source in cases:
        result=case(a,c,cp);traces[name]=result
        report['experiment_a'].append(dict(name=name,status=result['status'],price_source=source,fixture_sha256=result['fixture_sha256'],
            records={b:[dict(comparison=r['comparison'],label=r['label'],timing=r['result'].get('timing',r['result'].get('timings_seconds')),gpu_utility_max_error=r['result'].get('gpu_utility_max_error')) for r in rows] for b,rows in result['records'].items()}))
        if result['status']!='EXACT':report['status']='DIFF'
    (directory/'experiment-a-traces.json').write_text(json.dumps(traces,separators=(',',':'),allow_nan=False));persist()
    for n,s in ((1000,64),(10000,128),(100000,256)):
        if perf_counter()-start>budget-45:report['ladder'].append(dict(n=n,nodes=s,status='NOT_RUN_OBSERVED_BUDGET'));break
        tick=perf_counter();data=ladder(n,s);generated=perf_counter();free,total=cp.cuda.runtime.memGetInfo()
        per_job=s*(6*3*20+6*12+3*24+32);chunk=max(1,min(1024,int(.1*free//per_job)))
        entry=dict(n=n,nodes=s,shapes=8,resources=3,future_types=6,generation_seconds=generated-tick,
            input_sha256=hashlib.sha256(b''.join(k.encode()+np.ascontiguousarray(v).tobytes() for k,v in data.items())).hexdigest(),
            eligibility_policy='node_index%4 == task_index%4, materialized per chunk only',
            input_arrays={k:dict(shape=list(v.shape),dtype=v.dtype.str,bytes=v.nbytes) for k,v in data.items()},
            gpu_free_before_bytes=free,gpu_total_bytes=total,chunk_jobs=chunk,estimated_chunk_workspace_bytes=chunk*per_job,
            frozen={},scalar_sample=scalar_samples(data))
        for backend in ('numpy','cupy'):
            entry['frozen'][backend]=[]
            for rep in range(2):
                with Memory() as mem,utilization() as util:out=frozen(data,backend,chunk,cp)
                out.update(label='first' if rep==0 else 'warm',memory=mem.report(),utilization_peak=util['sampled_peak_percent']);entry['frozen'][backend].append(out)
        digests=[r['digest'] for rows in entry['frozen'].values() for r in rows]
        entry['frozen_exact']=len(set(digests))==1
        a=prefix(data);tick=perf_counter();ctrl=g.control(a);common=perf_counter()-tick
        sequential=case(a,ctrl['certificate'],cp);(directory/('sequential-'+str(n)+'.json')).write_text(json.dumps(sequential,separators=(',',':'),allow_nan=False))
        entry['sequential']=dict(status=sequential['status'],actual_jobs=len(a.value),fixture_sha256=sequential['fixture_sha256'],shared_lp_control_seconds=common,
            standalone_times={b:[common+r['result'].get('timing',r['result'].get('timings_seconds'))['total'] for r in rows] for b,rows in sequential['records'].items()},
            counts={b:dict(accepted=rows[-1]['result']['accepted_count'],value=rows[-1]['result']['value'],future_score=rows[-1]['result']['future_score']) for b,rows in sequential['records'].items()})
        entry['status']='EXACT' if entry['frozen_exact'] and sequential['status']=='EXACT' else 'DIFF'
        if entry['status']!='EXACT':report['status']='DIFF'
        report['ladder'].append(entry);persist()
    report['artifacts']=['experiment-a-traces.json']+['sequential-'+str(x['n'])+'.json' for x in report['ladder'] if 'sequential' in x]
    persist();return report


def audit(summary_path,directory,output):
    wrapper=json.loads(Path(summary_path).read_bytes());directory=Path(directory);checks=[]
    if not wrapper.get('trace_artifacts'):raise ValueError('empty trace manifest cannot pass audit')
    for item in wrapper['trace_artifacts']:
        raw=(directory/item['name']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('trace SHA mismatch')
        data=json.loads(raw)['data'];groups=data if item['name']=='experiment-a-traces.json' else {item['name']:data}
        for name,case_data in groups.items():
            a=unpack(case_data['instance']);c=case_data['certificate'];g.admit(a,c['x'],c['raw_pi'],c['raw_rc'],c['pi_sign'],c['reported_optimal'])
            if g.p4b.fixture_hash(a)!=case_data['fixture_sha256']:raise ValueError('physical fixture SHA')
            baseline=case_data['records']['scalar'][0]['result']
            for backend,rows in case_data['records'].items():
                for row in rows:
                    result=row['result'];rebuilt=g.audit_steps(a,result['ledger_steps'],result['task_order'],result['shape_order'])
                    for k in ('placement','residual','greedy_residual','accepted_count','value','future_score'):
                        if rebuilt[k]!=result[k]:raise ValueError('independent physical result mismatch:'+k)
                    check=compare(baseline,result)
                    if check['status']!='EXACT':raise ValueError('backend DIFF')
                    checks.append(dict(case=name,backend=backend,repetition=row['repetition'],status='EXACT'))
    out=dict(status='PASS_PHYSICAL_REVALIDATION',project_sha=wrapper['project_sha'],adapter_sha=wrapper['source_sha'],checks=checks,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),method='Independent retained full ledgers, original scalar marginal/ties and P4B physical reconstruction; no solver/GPU.')
    Path(output).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(dict(status=out['status'],checked=len(checks))));return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir');p.add_argument('--p4g-results');p.add_argument('--audit-summary');p.add_argument('--audit-directory');p.add_argument('--audit-output');args=p.parse_args()
    if args.audit_summary:audit(args.audit_summary,args.audit_directory,args.audit_output)
    else:run(args.output_dir,args.p4g_results)

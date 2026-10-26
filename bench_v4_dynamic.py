"""V4 B2: unchanged rolling.simulate, independent timeline audit, bounded CPU runs.

python bench_v4_dynamic.py --out ../smoke --legacy --seeds 11
python bench_v4_dynamic.py --out ../full
Core: 64 nodes, T=16, nine predeclared cases x five seeds x six methods x two
repetitions. seconds_per_tick=3600 is SYNTHETIC interpretation, not training time.
One shared build+hash charge per method + simulate-inclusive wall + audit/hash;
batch solve clocks are a nested diagnostic, never added. Fresh worker/small
warmup per run; process/import/load/warmup wall separate. Errors remain records.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, perf_counter_ns

import numpy as np
from accelsched import METHODS
from rolling import Job, simulate
from bench_v4_static import json_bytes, watched, environment

SEEDS=(11,29,47,71,95)
CASES={'low':.3,'medium':.7,'high':1.1,'overload':1.5,
       'family_skew':.7,'cpu_tight':.7,'ram_tight':.7,'eligibility':.7,'fragmentation':.7}
OMEGA=(('A',(1,8,32)),('A',(4,32,128)),('B',(1,8,32)),('*',(0,8,32)))
digest=lambda value:hashlib.sha256(json_bytes(value)).hexdigest()


def pack(cap,jobs,eligible,families,requests,omega,T,seconds,**metadata):
    array=lambda x:dict(dtype=x.dtype.str,shape=list(x.shape),values=x.tolist())
    return dict(capacity=array(cap),eligible=array(eligible),jobs=[dict(id=j.id,arrival=int(j.arrival),
        duration=int(j.duration),demand=list(map(float,j.demand)),value=float(j.value)) for j in jobs],
        families=list(families),requests=list(requests),omega=[[f,list(map(float,d))] for f,d in omega],
        T=T,seconds_per_tick=seconds,**metadata)


def unpack(data):
    def array(key):
        item=data[key];a=np.array(item['values'],dtype=item['dtype'])
        assert list(a.shape)==item['shape'];return a
    return array('capacity'),[Job(**j) for j in data['jobs']],array('eligible')


def generate(case,seed,legacy=False):
    if legacy:
        from bench_dynamic import generate as old, CAPACITY, NODE_FAMILIES, OMEGA as catalogue
        jobs,e,manifest,_=old(seed,.3)
        return pack(CAPACITY,jobs,e,NODE_FAMILIES,[j['family'] for j in manifest['jobs']],catalogue,
                    40,3600,case='legacy',seed=seed,target=.3)
    target=CASES[case];T=16;rng=np.random.default_rng(seed)
    kind=np.arange(64)%4
    cap=np.array([[4,64,256],[8,128,512],[4,64,256],[0,64,256]],float)[kind]
    families=np.array(['A','A','B','CPU'])[kind].tolist()
    jobs=[];requests=[];work=0
    while work<target*T*cap[:,0].sum():
        arrival=int(rng.integers(T));duration=int(rng.choice([1,2,4,8]))
        gpu=int(rng.choice([1,2,4,8] if case=='fragmentation' else [1,2,4]))
        family=str(rng.choice(['A','B'],p=[.85,.15] if case=='family_skew' else [.5,.5]))
        cpu=gpu*int(rng.choice([16,32] if case=='cpu_tight' else [2,4,8]))
        ram=gpu*int(rng.choice([128,256] if case=='ram_tight' else [8,16,32]))
        value=int(rng.integers(1,21))
        jobs.append(Job(f'j{len(jobs):04d}',arrival,duration,(gpu,cpu,ram),value));requests.append(family)
        work+=gpu*duration
    # Fixed order-sensitive fixture: small jobs precede 4/8-GPU requests at each arrival.
    if case=='fragmentation':
        order=sorted(range(len(jobs)),key=lambda i:(jobs[i].arrival,jobs[i].demand[0]>=4,i))
        jobs=[jobs[i] for i in order];requests=[requests[i] for i in order]
    e=np.array([[f==q for q in requests] for f in families],bool)
    if case=='eligibility':
        e &= rng.random(e.shape)<.15;e[:,np.arange(len(jobs))%17==0]=False
    return pack(cap,jobs,e,families,requests,OMEGA,T,3600,case=case,seed=seed,target=target)


def workload_info(data):
    cap,jobs,e=unpack(data);T=data['T']
    offered=sum((np.array(j.demand)*j.duration for j in jobs),np.zeros(3))
    impossible=[j.id for k,j in enumerate(jobs) if not np.any(e[:,k]&np.all(cap>=j.demand,axis=1))]
    families=data['families'];requests=data['requests'];loads={}
    for family in sorted(set(requests)):
        supply=sum(cap[i,0] for i,f in enumerate(families) if f==family)*T
        loads[family]=sum(j.demand[0]*j.duration for j,q in zip(jobs,requests) if q==family)/supply
    return dict(job_count=len(jobs),offered_resource_time_load=(offered/(T*cap.sum(axis=0))).tolist(),
                family_offered_gpu_time_load=loads,individually_impossible=impossible,
                duration_counts=dict(Counter(j.duration for j in jobs)),
                nonneutral_catalogue_is_diagnostic_only=True,simulator_future='neutral unchanged production')


def audit(data,r,method):
    """Generalize historical independent event/interval audit, without modifying it.

    Nonuniform values, arbitrary nonnegative whole-GPU demands, fixed-window
    clipping and full-tail completions are distinct. No production oracle calls.
    """
    cap,jobs,e=unpack(data);T=data['T'];seconds=data['seconds_per_tick']
    assert T>0 and seconds>0 and np.isfinite(cap).all()
    index={j.id:k for k,j in enumerate(jobs)};assert len(index)==len(jobs)
    decisions=r['decisions'];by={d['job_id']:d for d in decisions}
    assert len(by)==len(decisions)==len(jobs) and set(by)==set(index)
    assert [d['job_id'] for d in decisions]==[j.id for j in sorted(jobs,key=lambda j:j.arrival)]
    accepted=[d for d in decisions if d['status']=='accepted']
    def equal(x,y):
        x,y=np.asarray(x),np.asarray(y)
        assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
        if np.all(y==np.floor(y)):assert np.array_equal(x,y)
        else:assert np.allclose(x,y,atol=1e-9,rtol=0)
    def load(active):
        out=np.zeros_like(cap)
        for d in active:out[d['node']]+=jobs[index[d['job_id']]].demand
        return out
    full,clipped,admitted=np.zeros(3),np.zeros(3),np.zeros(3)
    for d in decisions:
        k=index[d['job_id']];j=jobs[k];assert d['arrival']==j.arrival
        assert np.isfinite(j.demand).all() and np.isfinite(j.value)
        if d['status']=='accepted':
            assert type(d['node']) is int and 0<=d['node']<len(cap) and e[d['node'],k]
            assert data['families'][d['node']]==data['requests'][k]
            assert d['start']==j.arrival and d['end']==j.arrival+j.duration
            admitted+=j.demand;full+=np.array(j.demand)*j.duration
            clipped+=np.array(j.demand)*max(0,min(d['end'],T)-j.arrival)
        else:
            assert d['status'] in ('not_scheduled','solver_error')
            assert d['node'] is d['start'] is d['end'] is None
    counts=Counter(d['status'] for d in decisions)
    assert counts['accepted']==r['accepted_count']==r['completed_count']
    assert counts['not_scheduled']==r['rejected_count'] and counts['solver_error']==r['solver_error_count']
    value=sum(jobs[index[d['job_id']]].value for d in accepted)
    equal(r['accepted_value'],value);equal(r['admitted_resources'],admitted)
    equal(r['released_resources'],admitted);equal(r['resource_time'],full);equal(r['final_free'],cap)
    times=sorted({j.arrival for j in jobs}|{d['end'] for d in accepted})
    assert [x['time'] for x in r['events']]==times
    assert [b['time'] for b in r['batches']]==sorted({j.arrival for j in jobs})
    last=0;previous_load=np.zeros_like(cap);previous_free=cap.copy()
    integral=np.zeros(3);tail=np.zeros(3);placeability=0.;fragmentation=0.;completions=[]
    def proxies(free,occupied):
        fits=sum(int((f=='*' or data['families'][i]==f) and np.all(free[i]+1e-10>=d))
                 for i in range(len(cap)) for f,d in data['omega'])
        partial=np.any(occupied>0,axis=1)&np.any(free>1e-9,axis=1)
        return fits,float(free[partial,0].sum())
    for ev in r['events']:
        t=ev['time'];dt=max(0,min(t,T)-min(last,T))
        integral+=previous_load.sum(axis=0)*dt;tail+=previous_load.sum(axis=0)*(t-last)
        f,g=proxies(previous_free,previous_load);placeability+=f*dt;fragmentation+=g*dt
        active=[d for d in accepted if d['start']<=t<d['end']];occupied=load(active);free=cap-occupied
        assert np.all(occupied<=cap+1e-9)
        equal(ev['running_load'],occupied);equal(ev['free'],free)
        equal(ev['free_before_arrivals'],cap-load([d for d in accepted if d['start']<t<d['end']]))
        assert sorted(ev['completed'])==sorted(d['job_id'] for d in accepted if d['end']==t)
        completions.extend(ev['completed'])
        expected={d['job_id']:(d['node'],d['end']) for d in active}
        assert len(ev['running'])==len(expected) and {d['job_id']:(d['node'],d['end']) for d in ev['running']}==expected
        for field,status in [('accepted','accepted'),('rejected','not_scheduled'),('solver_errors','solver_error')]:
            assert sorted(ev[field])==sorted(d['job_id'] for d in decisions if d['arrival']==t and d['status']==status)
        arrivals=[j.id for j in jobs if j.arrival==t]
        if arrivals:
            b=r['batches'][ev['batch_index']]
            assert b['time']==t and b['job_ids']==arrivals and b['method']==method
            good=b['placement'] is not None
            assert good==(not any(by[j]['status']=='solver_error' for j in arrivals))
            certified=good and method=='highs_mip' and b['solver_status']=='optimal' and b['reported_optimal']
            assert b['optimal']==certified
            batch_value=sum(jobs[index[j]].value for j in arrivals if by[j]['status']=='accepted')
            assert b['certified_value']==(batch_value if certified else None)
            if good:
                assert b['validation_status']=='valid' and b['solver_status'] in ('feasible','feasible_uncertified','optimal')
                assert b['placement']==[by[j]['node'] if by[j]['status']=='accepted' else -1 for j in arrivals]
                equal(b['accepted_value'],batch_value)
            else:assert b['accepted_value'] is None and b['certified_value'] is None
            assert all(by[j]['solver_status']==b['solver_status'] for j in arrivals)
        else:assert ev['batch_index'] is None
        last=t;previous_load,previous_free=occupied,free
    dt=max(0,T-last);integral+=previous_load.sum(axis=0)*dt
    f,g=proxies(previous_free,previous_load);placeability+=f*dt;fragmentation+=g*dt
    equal(integral,clipped);equal(tail,full)
    assert len(completions)==len(set(completions))==len(accepted)
    latency=[b['solve_wall_ms'] for b in r['batches']]
    assert all(np.isfinite(v) and v>=0 for v in latency)
    assert np.isfinite(r['elapsed_ms']) and r['elapsed_ms']>=0
    assert np.isclose(sum(latency),r['solve_wall_ms'],atol=1e-9,rtol=0)
    ratios=np.divide(clipped,T*cap.sum(axis=0),out=np.zeros(3),where=cap.sum(axis=0)!=0)
    assert np.all((ratios>=0)&(ratios<=1+1e-12))
    return dict(oracle='PASS',quality_complete=not counts['solver_error'],
        comparable_value=value if not counts['solver_error'] else None,
        observed_accepted_value=value,accepted_count=len(accepted),completed_full_tail=len(accepted),
        completed_by_T=sum(d['end']<=T for d in accepted),rejected_count=counts['not_scheduled'],
        solver_error_count=counts['solver_error'],admission_rate=len(accepted)/len(jobs) if jobs else None,
        full_resource_ticks=full.tolist(),window_resource_ticks=clipped.tolist(),
        full_tail_gpu_hours=float(full[0]*seconds/3600),window_gpu_hours=float(clipped[0]*seconds/3600),
        window_allocation_ratio=ratios.tolist(),allocated_gpu_work_rate=float(clipped[0]/T),
        mean_candidate_node_placeability=placeability/T,mean_partial_node_residual_gpu_proxy=fragmentation/T,
        batch_statuses=dict(Counter(b['solver_status'] for b in r['batches'])),
        per_batch_optimal_count=sum(b['optimal'] for b in r['batches']),
        batch_solve_quantiles_ms=dict(zip(('p50','p95','p99'),np.quantile(latency,[.5,.95,.99]).tolist())) if latency else None,
        attempted_batches=len(latency),horizon_optimality_claim=False)


def measure(data,method,time_limit):
    cap,jobs,e=unpack(data);before=digest(data)
    tick=perf_counter_ns();r=simulate(cap,jobs,e,method,seed=data['seed'],time_limit=time_limit)
    sim_ms=(perf_counter_ns()-tick)/1e6;tick=perf_counter_ns()
    metrics=audit(data,r,method)
    current=pack(cap,jobs,e,data['families'],data['requests'],data['omega'],data['T'],data['seconds_per_tick'],
                 case=data['case'],seed=data['seed'],target=data['target'])
    unchanged=before==digest(current);out_sha=digest(r)
    audit_ms=(perf_counter_ns()-tick)/1e6
    return dict(execution_status='RETURNED',method=method,input_sha256=before,input_unchanged=unchanged,
        report=r,output_sha256=out_sha,metrics=metrics,simulate_ms=sim_ms,audit_hash_ms=audit_ms,
        simulate_plus_audit_ms=sim_ms+audit_ms)


def worker(args):
    warm=pack(np.array([[1,4,16]],float),[Job('w0',0,2,(1,4,16),2),Job('w1',2,2,(1,4,16),3)],
              np.ones((1,2),bool),['A'],['A','A'],OMEGA,3,3600,case='warmup',seed=11,target=None)
    tick=perf_counter();warmup=measure(warm,args.method,args.time_limit);warm_ms=(perf_counter()-tick)*1000
    tick=perf_counter();data=json.loads(args.input.read_bytes());assert digest(data)==args.input_sha
    load_ms=(perf_counter()-tick)*1000
    result=measure(data,args.method,args.time_limit)
    result.update(warmup=warmup,warmup_wall_ms=warm_ms,input_load_hash_ms=load_ms)
    args.result.write_bytes(json_bytes(result))


def runtime():
    info=environment()
    if sys.platform=='win32':
        import ctypes
        from ctypes import wintypes
        api=ctypes.WinDLL('kernel32',use_last_error=True)
        api.GetCurrentProcess.restype=wintypes.HANDLE
        api.GetProcessAffinityMask.argtypes=[wintypes.HANDLE,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
        p,s=ctypes.c_size_t(),ctypes.c_size_t()
        if api.GetProcessAffinityMask(api.GetCurrentProcess(),ctypes.byref(p),ctypes.byref(s)):
            info['affinity']=[i for i in range(os.cpu_count()) if p.value&(1<<i)]
    return info


def run(args):
    args.out.mkdir(parents=True,exist_ok=False);scratch=args.out/'scratch';scratch.mkdir()
    env=os.environ.copy();env.update(TEMP=str(scratch.resolve()),TMP=str(scratch.resolve()),TMPDIR=str(scratch.resolve()),PYTHONDONTWRITEBYTECODE='1')
    names=['legacy'] if args.legacy else list(CASES)
    report=dict(status='PARTIAL',mode='SYNTHETIC_DYNAMIC_CPU',utc_start=datetime.now(timezone.utc).isoformat(),
        environment=runtime(),source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'])),
        seeds=args.seeds,cases_predeclared=names,repeats=2,mip_batch_time_limit=args.time_limit,
        watchdog_seconds=args.watchdog,budget_seconds=args.batch_seconds,measurements='shared generation+input hash + inclusive simulate + independent audit/output hash; batch clocks nested; startup/import/load/warmup separate',cases=[])
    deadline=perf_counter()+args.batch_seconds
    for name in names:
        for seed_index,seed in enumerate(args.seeds):
            tick=perf_counter_ns();data=generate(name,seed,args.legacy);sha=digest(data);build_ms=(perf_counter_ns()-tick)/1e6
            case=dict(case=name,seed=seed,input=data,input_sha256=sha,shared_build_hash_ms=build_ms,workload=workload_info(data),records=[])
            path=scratch/'input.json';path.write_bytes(json_bytes(data))
            for rep in range(2):
                shift=(seed_index+rep)%6
                for order,method in enumerate(METHODS[shift:]+METHODS[:shift]):
                    result=scratch/'worker.json'
                    row=dict(method=method,repetition=rep,order=order,execution_status='NOT_RUN_BATCH_DEADLINE',metrics=None)
                    if perf_counter()<deadline:
                        cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--input',str(path.resolve()),
                             '--result',str(result.resolve()),'--input-sha',sha,'--method',method,'--time-limit',str(args.time_limit)]
                        execution=watched(cmd,min(args.watchdog,max(.01,deadline-perf_counter())),env)
                        row.update(execution,argv=cmd)
                        row['execution_status']='WATCHDOG_TIMEOUT' if execution['watchdog_timeout'] else 'PROCESS_FAILED'
                        if execution['exit_code']==0 and result.exists():
                            row.update(json.loads(result.read_bytes()));assert row['input_sha256']==sha
                            row['source_to_audited_status_ms']=build_ms+row['simulate_plus_audit_ms']
                        if result.exists():result.unlink()
                    case['records'].append(row)
            path.unlink();report['cases'].append(case)
            (args.out/f'{name}-{seed}.json').write_bytes(json_bytes(case))
            print(json.dumps(dict(case=name,seed=seed,jobs=case['workload']['job_count'],
                returned=sum(r['execution_status']=='RETURNED' for r in case['records']),
                solver_error_jobs=sum(r['metrics']['solver_error_count'] for r in case['records'] if r['metrics']))),flush=True)
    assert list(scratch.iterdir())==[];scratch.rmdir()
    rows=[r for c in report['cases'] for r in c['records']]
    report.update(utc_end=datetime.now(timezone.utc).isoformat(),scratch_cleanup=True,
        status='PASS' if all(r['execution_status']=='RETURNED' and r['input_unchanged'] and r['metrics']['oracle']=='PASS' for r in rows) else 'PARTIAL')
    raw=gzip.compress(json_bytes(report),mtime=0);(args.out/'raw.json.gz').write_bytes(raw)
    print(json.dumps(dict(status=report['status'],runs=len(rows),archive_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path);p.add_argument('--legacy',action='store_true')
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS));p.add_argument('--time-limit',type=float,default=.1)
    p.add_argument('--watchdog',type=float,default=30.);p.add_argument('--batch-seconds',type=float,default=1200.)
    p.add_argument('--worker',action='store_true');p.add_argument('--input',type=Path);p.add_argument('--result',type=Path)
    p.add_argument('--input-sha');p.add_argument('--method',choices=METHODS)
    args=p.parse_args()
    if args.worker:worker(args)
    else:
        if not args.out or min(args.time_limit,args.watchdog,args.batch_seconds)<=0:raise SystemExit('new --out and positive bounds required')
        run(args)

"""Opt-in P1 profiling and paired physical experiments; never production defaults.

Uninstrumented pipeline wall is primary. cProfile and tracemalloc are separate
runs, with nested function costs never added to pipeline stage totals.
"""
from time import perf_counter
import argparse
import cProfile
import hashlib
import json
import os
from pathlib import Path
import pstats
import signal
import subprocess
import sys
import traceback


def write(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def observe(b):
    """Instrumentation-only subclasses; no changes in timed primary workers."""
    counters={};original_q=b.CandidateSet;original_ix=b.ExactIndex
    class Q(original_q):
        def put(self,node,new):
            old=self.cost[node];super().put(node,new)
            counters['max_heap_keys']=max(counters.get('max_heap_keys',0),len(self.heap))
            counters['max_live_buckets']=max(counters.get('max_live_buckets',0),len(self.buckets))
            if old>=0 and new<0:counters['candidate_disappearances']=counters.get('candidate_disappearances',0)+1
            if old>=0 and new>=0 and new<old:counters['score_improvements']=counters.get('score_improvements',0)+1
        def best(self):
            before=len(self.heap);result=super().best()
            counters['dead_heap_keys_pruned']=counters.get('dead_heap_keys_pruned',0)+before-len(self.heap)
            return result
    class IX(original_ix):
        def query(self,*args):
            before=len(self.queries);result=super().query(*args)
            counters['cache_hits']=counters.get('cache_hits',0)+(len(self.queries)==before)
            if len(self.queries)!=before:
                counters['registrations']=len(self.queries)
                counters['max_watcher_fanout']=max(counters.get('max_watcher_fanout',0),max(map(len,self.watchers.values()),default=0))
                counters['active_watcher_signatures']=len(self.watchers)
            return result
        def commit(self,node,demand):
            watchers=sum(len(self.watchers.get((int(self.shapes[node]),f,g),())) for f in {-1,int(self.nf[node])} for g in {-1,int(self.ng[node])})
            counters['max_affected_watchers_per_commit']=max(counters.get('max_affected_watchers_per_commit',0),watchers)
            return super().commit(node,demand)
    b.CandidateSet=Q;b.ExactIndex=IX
    return counters


def worker(a):
    sys.path.insert(0,str(a.source));import bench_ultra_pgfp as b
    import numpy as np
    out=a.out;out.mkdir(parents=True,exist_ok=False);rows=[]
    c=b.base.Config(a.n,a.nodes,a.case,a.seed)
    observations=observe(b) if a.mode=='profile' else None
    for rep in range(1 if a.mode!='primary' else 2):
        try:
            profiler=cProfile.Profile() if a.mode=='profile' else None
            if profiler:profiler.enable()
            if a.policy=='first_fit':
                r=b.base.pipeline(c,out,f'rep{rep}');r['status']='PHYSICAL_E2E_VALID' if r['status']=='PHYSICAL_E2E_CERTIFIED' else r['status'];r['full_seconds']=r.get('full_path_seconds')
            else:r=b.pipeline(c,out,f'rep{rep}',a.policy,a.backend)
            if profiler:
                profiler.disable();profiler.dump_stats(str(out/'calls.prof'))
                stats=pstats.Stats(profiler)
                r['profile_functions']=[dict(file=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],self_seconds=v[2],inclusive_seconds=v[3]) for k,v in sorted(stats.stats.items(),key=lambda kv:-kv[1][2])[:100]]
                r.update(observed_index=observations,instrumentation='CPROFILE_NOT_A_SPEEDUP_CLOCK')
            r.update(rep=rep,policy=a.policy,backend=a.backend,mode=a.mode,completion_monotonic=perf_counter())
        except BaseException:r=dict(rep=rep,policy=a.policy,mode=a.mode,status='ERROR',error=traceback.format_exc(),config=b.base.asdict(c))
        rows.append(r);write(out/'results.json',rows)


def invoke(source,out,c,policy,backend='cupdlpx',mode='primary',limit=180):
    args=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--source',str(source),'--out',str(out),'--n',str(c.n),'--nodes',str(c.nodes),'--case',c.case,'--seed',str(c.seed),'--policy',policy,'--backend',backend,'--mode',mode]
    start=perf_counter();p=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try:stdout,stderr=p.communicate(timeout=limit);timeout=False
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate();timeout=True
    rows=json.loads((out/'results.json').read_bytes()) if (out/'results.json').exists() else []
    for r in rows:
        if r['rep']==0 and 'completion_monotonic' in r:r['cold_process_seconds']=r['completion_monotonic']-start
    for rep in range(len(rows),1 if mode!='primary' else 2):
        checkpoint=out/f'rep{rep}-lp.json';phase=out/f'rep{rep}-phase.json'
        rows.append(dict(rep=rep,policy=policy,mode=mode,config=vars(c),status='TIMEOUT' if timeout else 'PROCESS_ERROR',
            optimizer_checkpoint=json.loads(checkpoint.read_bytes()) if checkpoint.exists() else None,last_phase=json.loads(phase.read_bytes()) if phase.exists() else None))
    return dict(argv=args,process_seconds=perf_counter()-start,watchdog=limit,exit_code=p.returncode,stdout=stdout,stderr=stderr,records=rows)


def shadow(source,reference_root,reference,out):
    """Whole placement/order identity on the SAME retained certified prices."""
    sys.path.insert(0,str(source));import bench_ultra_pgfp as b
    import numpy as np
    result=[]
    for a in reference['attempts']:
        if a['mode']!='primary' or a['policy']=='first_fit':continue
        c=b.base.Config(**a['config']);data=None
        for r in a['records']:
            if r['status']!='PHYSICAL_E2E_VALID':continue
            if data is None:data=b.base.materialize(c);cap,nf,ng,shapes=b.base.inventory(c)
            path=reference_root/a['key']/r['vectors'];assert b.base.sha(path.read_bytes())==r['vectors_sha256']
            with np.load(path,allow_pickle=False) as z:
                start=perf_counter();p,res,info=b.replay(cap,data,shapes,nf,ng,z['prices'],a['policy'])
                replay_seconds=perf_counter()-start
                assert np.array_equal(p,z['placement']) and np.array_equal(res,z['residual'])
                assert info['task_order_sha256']==r['order']['task_order_sha256'] and info['shape_order_sha256']==r['order']['shape_order_sha256']
                check=b.audit_extra(c,data,p,res,r['input_sha256']);assert check==r['physical']
                result.append(dict(key=a['key'],rep=r['rep'],config=vars(c),policy=a['policy'],status='PASS_EXACT_SAME_PRICE_ORDER_PLACEMENT_RESIDUAL_VALUE_FUTURE',
                    price_sha256=r['price_sha256'],input_sha256=r['input_sha256'],placement_sha256=b.base.sha(p.tobytes()),replay_seconds=replay_seconds,order=info))
            write(out,result)
    return result


def memory_replay(b,out,report):
    import tracemalloc
    import numpy as np
    a=next(x for x in report['attempts'] if x['config']['n']==10000 and x['policy']=='pgfp')
    r=next((r for r in a['records'] if r['status']=='PHYSICAL_E2E_VALID'),None)
    if r is None:return dict(status='NOT_RUN_NO_CERTIFIED_S0_PRICES')
    c=b.base.Config(**a['config']);data=b.base.materialize(c);cap,nf,ng,shapes=b.base.inventory(c)
    with np.load(out/a['key']/r['vectors'],allow_pickle=False) as z:
        tracemalloc.start();start=perf_counter();p,res,info=b.replay(cap,data,shapes,nf,ng,z['prices'],'pgfp')
        seconds=perf_counter()-start;current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        assert np.array_equal(p,z['placement']) and np.array_equal(res,z['residual'])
    return dict(status='PASS',config=vars(c),python_current_bytes=current,python_peak_bytes=peak,instrumented_replay_seconds=seconds,
        scope='Separate10K fixed-certified-price replay only; excludes input/native LP allocation, not process RSS or primary wall',order=info)


def phase(source,out,*,baseline=None,budget=900):
    sys.path.insert(0,str(source));import bench_ultra_pgfp as b
    out=Path(out);out.mkdir(exist_ok=False);start=perf_counter();report=dict(attempts=[],status='RUNNING',source=str(source))
    cases=[b.base.Config(10000,256,case,11) for case in ['high_pressure','fragmentation','multiresource']]
    cases += [b.base.Config(100000,1024,case,11) for case in ['high_pressure','fragmentation','multiresource']]
    cases += [b.base.Config(1000000,4096,case,11) for case in ['high_pressure','fragmentation']]
    def call(c,policy,backend='cupdlpx',mode='primary'):
        key=f'{mode}-{c.n}-{c.case}-{c.seed}-{policy}-{backend}';remaining=budget-(perf_counter()-start)
        if remaining<5:r=dict(records=[dict(rep=i,status='NOT_RUN',config=vars(c),policy=policy) for i in range(1 if mode!='primary' else 2)])
        else:r=invoke(source,out/key,c,policy,backend,mode,min(180 if c.n>=1000000 else 90,remaining))
        r.update(key=key,config=vars(c),policy=policy,backend=backend,mode=mode);report['attempts'].append(r)
        report['seconds']=perf_counter()-start;write(out/'summary.json',report)
    for i,c in enumerate(cases):
        policies=['first_fit','pgfp','lp_best_fit'];offset=i%3
        for policy in policies[offset:]+policies[:offset]:call(c,policy)
    if baseline is None:
        for case in ['high_pressure','fragmentation']:call(b.base.Config(100000,1024,case,11),'pgfp',mode='profile')
        call(b.base.Config(100000,1024,'fragmentation',11),'pgfp',backend='highs')
        call(b.base.Config(100000,1024,'fragmentation',11),'pgfp',backend='hpdlp')
        report['tracemalloc']=memory_replay(b,out,report)
    else:
        call(b.base.Config(100000,1024,'fragmentation',11),'pgfp',mode='profile')
        report['tracemalloc']=memory_replay(b,out,report)
        report['fixed_price_shadows']=shadow(source,Path(baseline),json.loads((Path(baseline)/'summary.json').read_bytes()),out/'shadow.json')
    report.update(status='COMPLETE_WITH_EXPLICIT_STATUSES',seconds=perf_counter()-start);write(out/'summary.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--n',type=int,default=100000);p.add_argument('--nodes',type=int,default=1024);p.add_argument('--case',default='fragmentation');p.add_argument('--seed',type=int,default=11)
    p.add_argument('--policy',default='pgfp');p.add_argument('--backend',default='cupdlpx');p.add_argument('--mode',default='primary');p.add_argument('--budget',type=float,default=900)
    p.add_argument('--baseline',type=Path)
    a=p.parse_args()
    if a.worker:worker(a)
    else:phase(a.source,a.out,baseline=a.baseline,budget=a.budget)

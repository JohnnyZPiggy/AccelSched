"""P6C Track E PROBE_ONLY: conservative read-set-disjoint GPU batch PGFP.

Proof: original ordering is fixed. A task reads residuals only at eligible nodes,
including every Shape/candidate/marginal score; it writes at most its chosen
eligible node. Pairwise-disjoint eligibility read sets therefore exclude every
intra-batch write/read and write/write conflict. Each frozen choice/score is the
original sequential choice by induction. At most one subtraction per node allows
batched updates and exact logical-prefix ledgers. Shared read sets split batches
regardless of distinct predicted destinations. No Track N policy is implemented.
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
from probes import pgfp_gpu_resident as b
p=b.p;g=b.g


def partition(order,eligible,limit=16):
    if not 1<=limit<=16:raise ValueError('bounded batch limit must be 1..16')
    groups=[];batch=[];used=np.zeros(len(eligible),bool);conflicts=0;limit_splits=0
    for j in order:
        conflict=bool(np.any(used&eligible[:,j]))
        if batch and (conflict or len(batch)==limit):
            groups.append(batch);batch=[];used[:]=False
            conflicts+=int(conflict);limit_splits+=int(not conflict)
        batch.append(j);used|=eligible[:,j]
    if batch:groups.append(batch)
    return groups,dict(dependency_rejections=conflicts,limit_splits=limit_splits,
        tasks_in_multi_batches=sum(len(x) for x in groups if len(x)>1),
        singleton_fallback_tasks=sum(len(x)==1 for x in groups),batch_sizes=[len(x) for x in groups],
        policy='Consecutive original order; all eligible-node read sets pairwise disjoint, else split; no destination-only heuristic.')


def independent(eligible,jobs):
    return bool(np.all(np.sum(eligible[:,jobs],axis=1)<=1))


def special_cases():
    safe=g.oracle.Instance(np.ones((4,3)),np.ones((4,3)),np.arange(4,0,-1,dtype=float),
        np.eye(4,dtype=bool),np.full(4,7),np.array([[1,1,1],[0,0,0]],float),np.array([.1,.2]),np.ones((4,2),bool))
    score=g.oracle.Instance(np.array([[3.,0,0],[3.,0,0]]),np.array([[1.,0,0],[1.,0,0]]),np.ones(2),
        np.ones((2,2),bool),np.full(2,7),np.array([[2.,0,0]]),np.ones(1),np.ones((2,1),bool))
    return dict(safe_disjoint=safe,zero_feasible=replace(safe,demand=np.full((4,3),2.)),distinct_destination_score_conflict=score)


def replay(a,c,backend='numpy',cp=None,limit=16):
    start=perf_counter();checked=g.admit(a,c['x'],c['raw_pi'],c['raw_rc'],c['pi_sign'],c['reported_optimal'])
    if c['lp_input_sha256']!=g.lp_hash(a) or checked['capacity_duals']!=c['capacity_duals']:raise ValueError('unrelated/tampered certificate')
    admitted=perf_counter();labels=list(dict.fromkeys(a.shapes.tolist()))
    u=a.value[None,:]-np.array(checked['capacity_duals'])@a.demand.T
    order=sorted(range(len(a.value)),key=lambda j:(-max(u[:,j]),j))
    shapes=[sorted(range(len(labels)),key=lambda h:(-u[h,j],h)) for j in range(len(a.value))]
    groups,stats=partition(order,a.eligible,limit)
    if not all(independent(a.eligible,x) for x in groups):raise ValueError('batch independence proof rejected')
    table,powers=b.score_table(a.weights);rank=np.array([[shapes[j].index(labels.index(x)) for x in a.shapes] for j in order])
    preprocessed=perf_counter();xp=np if backend=='numpy' else cp
    if xp is None:raise ValueError('optional CuPy required')
    gpu=backend=='cupy';sync=lambda:xp.cuda.get_current_stream().synchronize() if gpu else None
    R=xp.asarray(a.capacity.copy());D=xp.asarray(a.demand[order]);Q=xp.asarray(a.future);FE=xp.asarray(a.future_eligible)
    E=xp.asarray(a.eligible[:,order].T);T=xp.asarray(table);P=xp.asarray(powers);H=xp.asarray(rank)
    n=len(order);s=len(R);ids=xp.arange(s);hist=xp.empty((n+1,s,3),dtype=xp.float64);hist[0]=R
    valid_log=xp.empty((n,s),dtype=bool);scores_log=xp.empty((n,s),dtype=xp.float64);choices=xp.empty(n,dtype=xp.int64)
    sync();setup=perf_counter();events=[];submission=dict(batch_predicates_scores=0.,batch_selection=0.,batch_commit_trace=0.)
    def phase(name,fn):
        tick=perf_counter()
        if gpu:begin=xp.cuda.Event();end=xp.cuda.Event();begin.record()
        out=fn()
        if gpu:end.record();events.append((name,begin,end))
        submission[name]+=perf_counter()-tick;return out
    lo=0
    for jobs in groups:
        hi=lo+len(jobs)
        def predicates():
            valid=xp.all(R[None,:,:]+1e-10>=D[lo:hi,None,:],axis=2)&E[lo:hi]
            before=xp.all(R[:,None,:]+1e-10>=Q,axis=2)&FE
            after=xp.all((R[None,:,None,:]-D[lo:hi,None,None,:])+1e-10>=Q[None,None,:,:],axis=3)&FE[None,:,:]
            code=xp.sum((after.astype(xp.int8)-before[None,:,:].astype(xp.int8)+1)*P,axis=2)
            return valid,T[code]
        valid,scores=phase('batch_predicates_scores',predicates)
        def select():
            first=xp.min(xp.where(valid,H[lo:hi],len(labels)),axis=1)
            eligible=valid&(H[lo:hi]==first[:,None])
            best=xp.argmax(xp.where(eligible,scores,-xp.inf),axis=1)
            return xp.where(xp.any(eligible,axis=1),best,-1)
        chosen=phase('batch_selection',select)
        def commit():
            nonlocal R
            delta=xp.where((ids[None,:]==chosen[:,None])[:,:,None],D[lo:hi,None,:],0.)
            # One nonzero delta per physical node: prefix sums cannot regroup nonzero subtractions.
            hist[lo+1:hi+1]=R[None,:,:]-xp.cumsum(delta,axis=0)
            R=R-xp.sum(delta,axis=0);valid_log[lo:hi]=valid;scores_log[lo:hi]=scores;choices[lo:hi]=chosen
        phase('batch_commit_trace',commit);lo=hi
    submitted=perf_counter();sync();completed=perf_counter()
    device={name:sum(xp.cuda.get_elapsed_time(x,y)/1000 for key,x,y in events if key==name) for name in submission} if gpu else None
    transferred=perf_counter()
    if gpu:hist,valid_log,scores_log,choices=map(xp.asnumpy,(hist,valid_log,scores_log,choices))
    downloaded=perf_counter();steps=[];trace=[]
    for k,j in enumerate(order):
        step=dict(task=j,before=hist[k].tolist(),after=hist[k+1].tolist(),chosen=int(choices[k]),attempts=[])
        for h in shapes[j]:
            candidates=np.flatnonzero(valid_log[k]&(a.shapes==labels[h])).tolist();scores=scores_log[k,candidates].tolist()
            step['attempts'].append(dict(shape=labels[h],candidates=candidates,scores=scores))
            if candidates:trace.append(dict(task=j,shape=labels[h],candidates=candidates,scores=scores,chosen=int(choices[k])));break
        steps.append(step)
    traced=perf_counter();result=g.audit_steps(a,steps,order,shapes);audited=perf_counter()
    stats.update(offered=n,batches=len(groups),safely_batched_fraction=stats['tasks_in_multi_batches']/n,
        dependency_rejected_boundary_fraction=stats['dependency_rejections']/max(1,n-1),
        accepted_in_multi_batches=sum(result['placement'][j]>=0 for jobs in groups if len(jobs)>1 for j in jobs),
        batch_task_ids=groups,independence_proof_checked=True)
    result.update(trace=trace,ledger_steps=steps,utility=u.tolist(),shape_order=shapes,track='E_EXACT_PGFP',batch=stats,
        timing=dict(input_certificate=admitted-start,canonical_order_table_proof=preprocessed-admitted,setup_h2d=setup-preprocessed,
            batch_submit=submitted-setup,final_sync=completed-submitted,event_accounting=transferred-completed,
            d2h=downloaded-transferred,trace_construction=traced-downloaded,physical_audit=audited-traced,total=audited-start,
            host_submission=submission,device_events=device),
        buffers=dict(history_bytes=hist.nbytes,valid_bytes=valid_log.nbytes,scores_bytes=scores_log.nbytes,choices_bytes=choices.nbytes,
            max_batch_jobs=max(map(len,groups)),max_future_predicate_elements=max(map(len,groups))*s*len(a.future)*3,
            d2h_calls=4 if gpu else 0,per_task_explicit_host_sync=0,per_task_h2d=0))
    return result


def case(a,c,cp):
    reference=g.replay(a,c);records={}
    for backend in ('scalar','resident_cupy','track_e_numpy','track_e_cupy'):
        rows=[]
        for rep in range(2):
            with p.Memory() as mem,p.utilization() as util:
                if backend=='scalar':result=g.replay(a,c)
                elif backend=='resident_cupy':result=b.replay(a,c,'cupy',cp)
                else:result=replay(a,c,'numpy' if backend=='track_e_numpy' else 'cupy',cp)
            rows.append(dict(repetition=rep,label='first' if rep==0 else 'warm',comparison=p.compare(reference,result),
                result=result,memory=mem.report(),utilization_peak=util['sampled_peak_percent']))
        records[backend]=rows
    return dict(status='EXACT' if all(r['comparison']['status']=='EXACT' for rows in records.values() for r in rows) else 'DIFF',
        fixture_sha256=g.p4b.fixture_hash(a),instance=p.pack(a),certificate=c,records=records)


def run(directory,p4g_path,budget=300.):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import cupy as cp
    imported=perf_counter();assert float(cp.asnumpy(cp.arange(32,dtype=cp.float64).sum()))==496.
    report=dict(status='PASS',mode='PROBE_ONLY',track='E_EXACT_PGFP',track_n='NOT_RUN_OPTIONAL',cupy_import_seconds=imported-start,
        gpu_smoke_seconds=perf_counter()-imported,library=dict(cupy=cp.__version__,runtime=cp.cuda.runtime.runtimeGetVersion()),
        experiment_a=[],ladder=[],artifacts=[],limits=dict(worker_seconds=budget,max_batch_jobs=16,measured_repetitions=2),
        policy='Read-set-disjoint conservative batches, logical ordered ledger; overlapping eligible nodes force split/singletons. CPU original order and canonical score table.')
    prior=g.checked_json(Path(p4g_path).read_bytes(),p.P4G_SHA);cases=[];traces={}
    for name,a in g.scenarios().items():
        if g.p4b.fixture_hash(a)!=g.p4b.fixture_hash(p.unpack(prior['cases'][name]['instance'])):raise ValueError('P4G fixture changed')
        ctrl=g.control(a)
        for side in ('left','right'):cases.append((name+'_'+side,a,prior['cases'][name]['paths'][side]['certificate'],ctrl['status']))
    for name,a in [('epsilon_cliff',g.p4b.fixed_cases()[0][1]),('tie_competition',p.edge_case()),*special_cases().items()]:
        ctrl=g.control(a);cases.append((name,a,ctrl['certificate'],ctrl['status']))
    def save():
        report['seconds']=perf_counter()-start;(directory/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    for name,a,c,ctrl in cases:
        out=case(a,c,cp);traces[name]=out;report['experiment_a'].append(dict(name=name,status=out['status'],control=ctrl))
        if out['status']!='EXACT':report['status']='DIFF'
    (directory/'experiment-a-traces.json').write_text(json.dumps(traces,separators=(',',':'),allow_nan=False));report['artifacts'].append('experiment-a-traces.json');save()
    for n,s in ((64,64),(256,128),(1024,128)):
        if perf_counter()-start>budget-60:report['ladder'].append(dict(n=n,nodes=s,status='NOT_RUN_BUDGET'));break
        tick=perf_counter();a=p.prefix(p.ladder(n,s),n);ctrl=g.control(a);common=perf_counter()-tick
        out=case(a,ctrl['certificate'],cp);name='sequential-'+str(n)+'.json'
        (directory/name).write_text(json.dumps(out,separators=(',',':'),allow_nan=False));report['artifacts'].append(name)
        report['ladder'].append(dict(n=n,nodes=s,status=out['status'],fixture_sha256=out['fixture_sha256'],
            generation_reference_suite_seconds=common,control_lp_seconds=ctrl['control_lp_seconds'],
            timing={backend:[r['result'].get('timing',r['result'].get('timings_seconds')) for r in rows] for backend,rows in out['records'].items()},
            batch=out['records']['track_e_cupy'][-1]['result']['batch']))
        if out['status']!='EXACT':report['status']='DIFF'
        save()
    return report


def audit(summary,directory,output):
    out=p.audit(summary,directory,output);wrapper=json.loads(Path(summary).read_bytes());proofs=[]
    for meta in wrapper['trace_artifacts']:
        raw=json.loads((Path(directory)/meta['name']).read_bytes())['data'];groups=raw if meta['name']=='experiment-a-traces.json' else {meta['name']:raw}
        for name,case_data in groups.items():
            a=p.unpack(case_data['instance'])
            for backend,rows in case_data['records'].items():
                for row in rows:
                    result=row['result']
                    if not backend.startswith('track_e'):continue
                    groups_,stats=partition(result['task_order'],a.eligible)
                    if groups_!=result['batch']['batch_task_ids'] or not all(independent(a.eligible,x) for x in groups_):raise ValueError('batch proof mismatch')
                    for key,value in stats.items():
                        if result['batch'][key]!=value:raise ValueError('batch statistics mismatch')
                    proofs.append(dict(case=name,backend=backend,repetition=row['repetition'],status='READ_WRITE_DISJOINT_PROOF_PASS'))
    out.update(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),batch_proofs=proofs)
    Path(output).write_text(json.dumps(out,indent=2));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir');parser.add_argument('--p4g-results')
    parser.add_argument('--audit-summary');parser.add_argument('--audit-directory');parser.add_argument('--audit-output');args=parser.parse_args()
    if args.audit_summary:audit(args.audit_summary,args.audit_directory,args.audit_output)
    else:run(args.output_dir,args.p4g_results)

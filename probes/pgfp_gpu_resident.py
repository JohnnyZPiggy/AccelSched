"""P6B PROBE_ONLY: resident sequential PGFP, CuPy optional, original scalar audit.

No conflicting parallel commits. Canonical NumPy score table avoids device
floating reduction differences; task/shape utility ordering stays authoritative.
"""
import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from probes import pgfp_gpu_scoring as p

g=p.g


def score_table(weights):
    # shortcut: bounded research fixtures have at most six future types.
    if len(weights)>6:raise ValueError('probe limited to six future types')
    patterns=np.array(list(itertools.product((-1,0,1),repeat=len(weights))),dtype=np.int8)
    return np.array([float(np.sum(row*weights)) for row in patterns]),3**np.arange(len(weights)-1,-1,-1)


def replay(a,c,backend='numpy',cp=None):
    start=perf_counter();checked=g.admit(a,c['x'],c['raw_pi'],c['raw_rc'],c['pi_sign'],c['reported_optimal'])
    if c['lp_input_sha256']!=g.lp_hash(a) or checked['capacity_duals']!=c['capacity_duals']:raise ValueError('unrelated/tampered certificate')
    admitted=perf_counter();labels=list(dict.fromkeys(a.shapes.tolist()))
    u=a.value[None,:]-np.array(checked['capacity_duals'])@a.demand.T
    order=sorted(range(len(a.value)),key=lambda j:(-max(u[:,j]),j))
    shapes=[sorted(range(len(labels)),key=lambda h:(-u[h,j],h)) for j in range(len(a.value))]
    table,powers=score_table(a.weights);rank=np.array([[shapes[j].index(labels.index(x)) for x in a.shapes] for j in order])
    preprocessed=perf_counter();xp=np if backend=='numpy' else cp
    if xp is None:raise ValueError('optional CuPy library required')
    gpu=backend=='cupy';sync=lambda:xp.cuda.get_current_stream().synchronize() if gpu else None
    R=xp.asarray(a.capacity.copy());D=xp.asarray(a.demand);Q=xp.asarray(a.future)
    FE=xp.asarray(a.future_eligible);E=xp.asarray(a.eligible);T=xp.asarray(table);P=xp.asarray(powers);H=xp.asarray(rank)
    n=len(order);s=len(R);ids=xp.arange(s);hist=xp.empty((n+1,s,3),dtype=xp.float64);hist[0]=R
    valid_log=xp.empty((n,s),dtype=bool);scores_log=xp.empty((n,s),dtype=xp.float64);choices=xp.empty(n,dtype=xp.int64)
    sync();setup=perf_counter();events=[];submission=dict(predicates_scores=0.,selection=0.,commit_trace=0.)
    def phase(name,fn):
        tick=perf_counter()
        if gpu:b=xp.cuda.Event();e=xp.cuda.Event();b.record()
        result=fn()
        if gpu:e.record();events.append((name,b,e))
        submission[name]+=perf_counter()-tick
        return result
    for k,j in enumerate(order):
        def predicates():
            valid=xp.all(R+1e-10>=D[j],axis=1)&E[:,j]
            before=xp.all(R[:,None,:]+1e-10>=Q,axis=2)&FE
            after=xp.all((R-D[j])[:,None,:]+1e-10>=Q,axis=2)&FE
            code=xp.sum((after.astype(xp.int8)-before.astype(xp.int8)+1)*P,axis=1)
            return valid,T[code]
        valid,scores=phase('predicates_scores',predicates)
        def select():
            first=xp.min(xp.where(valid,H[k],len(labels)))
            eligible=valid&(H[k]==first)
            best=xp.argmax(xp.where(eligible,scores,-xp.inf))
            return xp.where(xp.any(eligible),best,-1)
        chosen=phase('selection',select)
        def commit():
            # Same stream: each subsequent predicate sees exactly one prior commit.
            nonlocal R
            R=R-xp.where((ids==chosen)[:,None],D[j],0.)
            hist[k+1]=R;valid_log[k]=valid;scores_log[k]=scores;choices[k]=chosen
        phase('commit_trace',commit)
    submitted=perf_counter();sync();completed=perf_counter()
    device={name:sum(xp.cuda.get_elapsed_time(b,e)/1000 for key,b,e in events if key==name) for name in submission} if gpu else None
    transfer=perf_counter()
    if gpu:hist,valid_log,scores_log,choices=map(xp.asnumpy,(hist,valid_log,scores_log,choices))
    downloaded=perf_counter();steps=[];trace=[]
    for k,j in enumerate(order):
        step=dict(task=j,before=hist[k].tolist(),after=hist[k+1].tolist(),chosen=int(choices[k]),attempts=[])
        for h in shapes[j]:
            candidates=np.flatnonzero(valid_log[k]&(a.shapes==labels[h])).tolist();scores=scores_log[k,candidates].tolist()
            step['attempts'].append(dict(shape=labels[h],candidates=candidates,scores=scores))
            if candidates:
                trace.append(dict(task=j,shape=labels[h],candidates=candidates,scores=scores,chosen=int(choices[k])));break
        steps.append(step)
    traced=perf_counter();result=g.audit_steps(a,steps,order,shapes);audited=perf_counter()
    result.update(trace=trace,ledger_steps=steps,utility=u.tolist(),shape_order=shapes,
        timing=dict(input_certificate=admitted-start,canonical_order_score_table=preprocessed-admitted,setup_h2d=setup-preprocessed,
            greedy_submit=submitted-setup,final_sync=completed-submitted,event_accounting=transfer-completed,
            d2h=downloaded-transfer,trace_construction=traced-downloaded,physical_audit=audited-traced,total=audited-start,
            host_submission=submission,device_events=device),
        buffers=dict(history_bytes=hist.nbytes,valid_bytes=valid_log.nbytes,scores_bytes=scores_log.nbytes,choice_bytes=choices.nbytes,
                     table_bytes=table.nbytes,d2h_calls=4 if gpu else 0,per_task_host_sync=0,per_task_h2d=0),
        policy='CPU original utility order and canonical finite score table; device first-shape/max-score/min-node selection; single-stream ordered commits; one final trace download.')
    return result


def case(a,c,cp):
    reference=g.replay(a,c);records={}
    for backend in ('scalar','numpy','cupy'):
        rows=[]
        for repetition in range(2):
            with p.Memory() as mem,p.utilization() as util:result=g.replay(a,c) if backend=='scalar' else replay(a,c,backend,cp)
            rows.append(dict(repetition=repetition,label='first' if repetition==0 else 'warm',result=result,
                comparison=p.compare(reference,result),memory=mem.report(),utilization_peak=util['sampled_peak_percent']))
        records[backend]=rows
    return dict(status='EXACT' if all(r['comparison']['status']=='EXACT' for rows in records.values() for r in rows) else 'DIFF',
        fixture_sha256=g.p4b.fixture_hash(a),instance=p.pack(a),certificate=c,records=records)


def run(directory,p4g_path,budget=240.):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import cupy as cp
    imported=perf_counter();assert float(cp.asnumpy(cp.arange(32,dtype=cp.float64).sum()))==496.
    report=dict(status='PASS',mode='PROBE_ONLY',cupy_import_seconds=imported-start,gpu_smoke_seconds=perf_counter()-imported,
        library=dict(cupy=cp.__version__,runtime=cp.cuda.runtime.runtimeGetVersion()),experiment_a=[],ladder=[],artifacts=[],
        policy='Complete sequential resident replay only; first plus one warm measured replay; no frozen throughput substitution; canonical CPU order/score table.')
    prior=g.checked_json(Path(p4g_path).read_bytes(),p.P4G_SHA);cases=[];traces={}
    for name,a in g.scenarios().items():
        if g.p4b.fixture_hash(a)!=g.p4b.fixture_hash(p.unpack(prior['cases'][name]['instance'])):raise ValueError('P4G fixture changed')
        control=g.control(a)
        for side in ('left','right'):cases.append((name+'_'+side,a,prior['cases'][name]['paths'][side]['certificate'],control['status']))
    for name,a in [('epsilon_cliff',g.p4b.fixed_cases()[0][1]),('tie_competition',p.edge_case())]:
        ctrl=g.control(a);cases.append((name,a,ctrl['certificate'],ctrl['status']))
    def save():
        report['seconds']=perf_counter()-start;(directory/'results.json').write_text(json.dumps(report,separators=(',',':'),allow_nan=False))
    for name,a,c,ctrl in cases:
        out=case(a,c,cp);traces[name]=out;report['experiment_a'].append(dict(name=name,status=out['status'],control=ctrl))
        if out['status']!='EXACT':report['status']='DIFF'
    (directory/'experiment-a-traces.json').write_text(json.dumps(traces,separators=(',',':'),allow_nan=False));report['artifacts'].append('experiment-a-traces.json');save()
    for n,s in ((64,64),(256,128),(1024,128)):
        if perf_counter()-start>budget-50:report['ladder'].append(dict(n=n,nodes=s,status='NOT_RUN_BUDGET'));break
        tick=perf_counter();a=p.prefix(p.ladder(n,s),n);ctrl=g.control(a);common=perf_counter()-tick
        out=case(a,ctrl['certificate'],cp);name='sequential-'+str(n)+'.json'
        (directory/name).write_text(json.dumps(out,separators=(',',':'),allow_nan=False));report['artifacts'].append(name)
        report['ladder'].append(dict(n=n,nodes=s,status=out['status'],fixture_sha256=out['fixture_sha256'],
            shared_reference_suite_seconds=common,control_lp_seconds=ctrl['control_lp_seconds'],
            timing={b:[r['result'].get('timing',r['result'].get('timings_seconds')) for r in rows] for b,rows in out['records'].items()}))
        if out['status']!='EXACT':report['status']='DIFF'
        save()
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir');parser.add_argument('--p4g-results')
    parser.add_argument('--audit-summary');parser.add_argument('--audit-directory');parser.add_argument('--audit-output');args=parser.parse_args()
    if args.audit_summary:
        out=p.audit(args.audit_summary,args.audit_directory,args.audit_output)
        import hashlib
        out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        Path(args.audit_output).write_text(json.dumps(out,indent=2),encoding='utf-8')
    else:run(args.output_dir,args.p4g_results)

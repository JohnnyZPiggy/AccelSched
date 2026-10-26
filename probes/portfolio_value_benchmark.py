"""P9 PROBE_ONLY synthetic scheduling value; unchanged production algorithms."""
from dataclasses import asdict,replace
import hashlib
import json
from time import perf_counter
import numpy as np
from workload import heterogeneous_example
import certified_pgfp as pg
from solver_portfolio import solve_portfolio
from probes import gpu_lp_solver_probe as bridge
from probes import pgfp_dual_stability as audit

POLICIES=('highs','cupdlpx','hpdlp','auto')
CASES=('p5_1k','p5_10k','heterogeneous','cpu_tight','ram_eligibility')
MAX_LEDGER_ENTRIES=1_000_000


def fixture(name):
    if name in ('p5_1k','p5_10k'):return bridge.p5.fixture(*((1000,64) if name=='p5_1k' else (10000,256)))
    if name not in CASES:raise ValueError('unknown fixed fixture')
    nodes,jobs={'heterogeneous':(12,64),'cpu_tight':(18,128),'ram_eligibility':(24,192)}[name]
    a=heterogeneous_example(nodes,jobs,seed=95)
    if name=='cpu_tight':a.capacity[:,1]*=.25
    if name=='ram_eligibility':
        a.capacity[:,2]*=.25;a.eligible&=(np.arange(nodes)[:,None]%2==np.arange(jobs)[None,:]%2)
    return a


def input_hash(a):
    h=hashlib.sha256()
    for name in ('capacity','demand','value','eligible','shapes','future','weights','future_eligible'):
        x=np.asarray(getattr(a,name));h.update(name.encode());h.update(str((x.shape,x.dtype.str)).encode());h.update(x.tobytes())
    return h.hexdigest()


def policy_order(rep):return POLICIES[rep%4:]+POLICIES[:rep%4]


def physical_metrics(a,replay):
    rest=np.asarray(replay['greedy_residual']);used=a.capacity-rest;totals=a.capacity.sum(axis=0);partial=(used>1e-10).any(axis=1)&(rest>1e-10).any(axis=1)
    return dict(accepted_jobs=sum(i>=0 for i in replay['placement']),offered_jobs=len(a.value),placed_value=replay['value'],future_score=replay['future_score'],used=used.sum(axis=0).tolist(),total_capacity=totals.tolist(),utilization=[None if c==0 else float(u/c) for u,c in zip(used.sum(axis=0),totals)],residual_total=rest.sum(axis=0).tolist(),partially_used_nodes=int(partial.sum()),residual_on_partially_used_nodes=rest[partial].sum(axis=0).tolist(),definition='GPU-count/CPU-cores/host-RAM-GiB used/total, no clipping; fragmentation proxy is residual on partially used nodes, not an optimality claim')


def measure(case,policy,rep,*,device=None,time_limit=10.,solver=None):
    entrance=perf_counter();a=fixture(case);generated=perf_counter();model,labels=pg.canonical_shape(a);original=bridge.p5.assemble(a);built=perf_counter()
    if case.startswith('p5_'):
        assert (bridge.p5.digest(a.capacity,a.demand,a.value,a.shapes),bridge.p5.digest(model.A.data,model.A.indices,model.A.indptr,model.row_hi,model.c))==bridge.PINS[len(a.value)]
    result=solve_portfolio(model,policy=policy,warm=rep>0,device=device,time_limit=time_limit,wall_budget=25.) if solver is None else solver(model)
    solved=perf_counter();strict=None
    if result.status=='OPTIMAL' and result.certified:
        raw=result.raw
        strict=bridge.p5.certify(original,raw.x,raw.pi,raw.rc,True,raw.objective,raw.dual_objective,raw.gap) if raw.dual_objective is not None else bridge.p5.certify(original,raw.x,raw.pi,raw.rc,True,raw.objective)
    certified=bool(strict and strict['certified']);lp_end=perf_counter()
    row=dict(case=case,policy=policy,rep=rep,context='first_numerical_case_policy_call' if rep==0 else 'warm_process_fresh_model',input_sha256=input_hash(a),canonical_sha256=model.source_hash,result=result,strict_p5=strict,certified=certified,dimensions=dict(tasks=len(a.value),nodes=len(a.capacity),shapes=len(labels),rows=model.A.shape[0],columns=model.A.shape[1],nnz=model.A.nnz),synthetic=True,aggregation_limit='Shape LP ignores per-node eligibility and fragmentation; certified LP is not physical admission proof',physical_status='LP_NOT_CERTIFIED')
    row['timing']=dict(source_generation=generated-entrance,canonical_and_original_build=built-generated,portfolio_inclusive=solved-built,independent_p5_certificate=lp_end-solved,lp_source_to_certified=lp_end-entrance if certified else None,lp_attempt_wall=lp_end-entrance)
    entries=2*a.capacity.size*len(a.value)
    if certified and entries<=MAX_LEDGER_ENTRIES:
        lp=dict(shapes=labels,prices=-result.raw.pi[len(a.value):].reshape(len(labels),a.demand.shape[1]))
        replay=pg._replay(a,lp);replayed=perf_counter();checked=audit.audit_steps(a,replay['ledger_steps'],replay['task_order'],replay['shape_order']);assert checked['placement']==replay['placement'] and checked['future_score']==replay['future_score'] and checked['residual']==replay['residual']
        row.update(physical_status='VALIDATED',replay=replay,metrics=physical_metrics(a,replay));validated=perf_counter()
        row['timing'].update(physical_order_greedy_and_input_accounting=replayed-lp_end,independent_ledger_and_metrics=validated-replayed,source_to_valid_placement=validated-entrance)
        if (len(a.value)+len(labels)*a.demand.shape[1])*len(labels)*len(a.value)<=1_000_000:
            tick=perf_counter();shadow=pg.shadow(a,solver=lambda m:result,strict_compatibility=True);assert shadow['proposed']==replay
            row['shadow']=shadow;row['timing']['separate_strict_baseline_diagnostic']=perf_counter()-tick
        else:row['shadow_status']='SKIPPED_DENSE_BASELINE_LIMIT'
    elif certified:row.update(physical_status='SKIPPED_LEDGER_RESOURCE_BOUND',physical_skip=dict(estimated_before_after_scalars=entries,max_scalars=MAX_LEDGER_ENTRIES,reason='Sparse LP only; no extrapolated placement or benefit'))
    row['timing']['total_including_diagnostic']=perf_counter()-entrance
    return row


def retain(row,arrays,key):
    """Losslessly separate raw vectors/large ledgers into caller-owned NPZ arrays."""
    row=dict(row);result=asdict(row['result'])
    for i,a in enumerate(result['attempts']):
        for field in ('x','pi','rc','witness'):
            v=a['raw'][field]
            if v is not None:
                name=f'{key}_{i}_{field}';arrays[name]=v;a['raw'][field]=dict(archive_key=name,shape=list(v.shape),dtype=v.dtype.str,sha256=hashlib.sha256(v.tobytes()).hexdigest())
    row['result']=result
    if 'shadow' in row:
        shadow=dict(row['shadow']);shadow['lp']={k:v for k,v in shadow['lp'].items() if k not in ('result','prices','x')};row['shadow']=shadow
    for field in ('replay','shadow'):
        if field in row:
            raw=json.dumps(row[field],separators=(',',':'),allow_nan=False).encode();name=key+'_'+field;arrays[name]=np.frombuffer(raw,dtype=np.uint8).copy();row[field]=dict(archive_key=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    return row

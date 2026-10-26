"""P10 PROBE_ONLY: isolated equivalent physical replay and honest profiling."""
import cProfile
import pstats
from time import perf_counter
import numpy as np
import certified_pgfp as pg
import m0_oracle as oracle
from probes import portfolio_value_benchmark as p9
from probes import pgfp_dual_stability as audit

CASES=('p5_1k','heterogeneous','cpu_tight','ram_eligibility')


def replay(a,lp,*,vector=False,instrument=False):
    """Only immutable Shape grouping / feasibility batching / key lookup change.

    Marginal arithmetic, sums, job/Shape/node order and subtraction unchanged.
    Instrumented runs are diagnostics, never used as uninstrumented speed evidence.
    """
    start=perf_counter();stages={k:0. for k in ('ordering','feasibility','marginal','selection_commit','ledger','final_validation_metrics')};counts=dict(node_visits=0,eligibility_checks=0,resource_rows=0,marginal_calls=0,index_search_comparisons=0,ledger_scalars=0)
    tick=perf_counter();labels=lp['shapes'];utility=a.value[None,:]-lp['prices']@a.demand.T;order=sorted(range(len(a.value)),key=lambda j:(-max(utility[:,j]),j));shape_orders=[sorted(range(len(labels)),key=lambda h:(-utility[h,j],h)) for j in range(len(a.value))];groups=[np.flatnonzero(a.shapes==h) for h in labels] if vector else None;rest=a.capacity.astype(float).copy();placement=[-1]*len(a.value);steps=[]
    if instrument:stages['ordering']+=perf_counter()-tick
    for j in order:
        tick=perf_counter() if instrument else 0.;step=dict(task=j,before=rest.tolist(),attempts=[],chosen=-1)
        if instrument:stages['ledger']+=perf_counter()-tick;counts['ledger_scalars']+=rest.size
        for h in shape_orders[j]:
            tick=perf_counter() if instrument else 0.
            if vector:
                pool=groups[h];ids=pool[a.eligible[pool,j]&np.all(rest[pool]+1e-10>=a.demand[j],axis=1)].tolist()
            else:
                ids=[i for i in range(len(rest)) if a.shapes[i]==labels[h] and a.eligible[i,j] and np.all(rest[i]+1e-10>=a.demand[j])]
            if instrument:stages['feasibility']+=perf_counter()-tick
            if instrument:
                counts['node_visits']+=len(pool) if vector else len(rest);counts['eligibility_checks']+=len(pool) if vector else int(np.sum(a.shapes==labels[h]));counts['resource_rows']+=len(pool) if vector else int(np.sum((a.shapes==labels[h])&a.eligible[:,j]))
            tick=perf_counter() if instrument else 0.;scores=[float(oracle.marginal(a,rest,i,j)) for i in ids]
            if instrument:stages['marginal']+=perf_counter()-tick;counts['marginal_calls']+=len(ids)
            tick=perf_counter() if instrument else 0.;step['attempts'].append(dict(shape=labels[h],candidates=ids,scores=scores))
            if instrument:stages['ledger']+=perf_counter()-tick
            if ids:
                tick=perf_counter() if instrument else 0.
                chosen=max(zip(ids,scores),key=lambda pair:(pair[1],-pair[0]))[0] if vector else max(ids,key=lambda i:(scores[ids.index(i)],-i))
                placement[j]=chosen;step['chosen']=chosen;rest[chosen]-=a.demand[j]
                if instrument:stages['selection_commit']+=perf_counter()-tick;counts['index_search_comparisons']+=0 if vector else len(ids)*(len(ids)+1)//2
                break
        tick=perf_counter() if instrument else 0.;step['after']=rest.tolist();steps.append(step)
        if instrument:stages['ledger']+=perf_counter()-tick;counts['ledger_scalars']+=rest.size
    tick=perf_counter()
    if not oracle.validate(a,placement):raise ValueError('physical placement audit failed')
    out=dict(task_order=order,shape_order=shape_orders,ledger_steps=steps,placement=placement,value=oracle.placed_value(a,placement),residual=oracle.residual(a,placement).tolist(),greedy_residual=rest.tolist(),future_score=oracle.future_score(a,rest))
    if instrument:
        stages['final_validation_metrics']+=perf_counter()-tick;total=perf_counter()-start;counts.update(resource_scalar_comparisons=counts['resource_rows']*a.demand.shape[1],future_scalar_comparisons=2*counts['marginal_calls']*a.future.size,task_sort_keys=len(a.value),shape_sort_keys=len(a.value)*len(labels))
        return out,dict(stages=stages,accounting_and_instrumentation_wall=max(0.,total-sum(stages.values())),total_wall=total,counts=counts)
    return out


def checked(a,out):
    proof=audit.audit_steps(a,out['ledger_steps'],out['task_order'],out['shape_order'])
    assert proof['placement']==out['placement'] and proof['residual']==out['residual'] and proof['future_score']==out['future_score']
    return p9.physical_metrics(a,out)


def profile_call(function,*args):
    prof=cProfile.Profile();tick=perf_counter();out=prof.runcall(function,*args);wall=perf_counter()-tick;stats=pstats.Stats(prof)
    rows=[dict(filename=k[0],line=k[1],function=k[2],primitive_calls=v[0],calls=v[1],exclusive_seconds=v[2],inclusive_seconds=v[3]) for k,v in stats.stats.items()]
    return out,dict(profiled_wall=wall,functions=sorted(rows,key=lambda q:-q['inclusive_seconds']),note='Inclusive cumulative function times overlap; do not sum. cProfile overhead separate from paired unprofiled wall.')


def experiment(case,*,device=None,warm=False):
    if case not in CASES:raise ValueError('P9 10K physical bound retained; unknown case')
    tick=perf_counter();a=p9.fixture(case);model,labels=pg.canonical_shape(a);original=p9.bridge.p5.assemble(a)
    result=p9.solve_portfolio(model,device=device,warm=warm,time_limit=10.,wall_budget=25.)
    if result.status!='OPTIMAL' or not result.certified:raise ValueError('LP not independently certified')
    raw=result.raw;cert=p9.bridge.p5.certify(original,raw.x,raw.pi,raw.rc,True,raw.objective,raw.dual_objective,raw.gap) if raw.dual_objective is not None else p9.bridge.p5.certify(original,raw.x,raw.pi,raw.rc,True,raw.objective);assert cert['certified'];prefix=perf_counter()-tick
    assert 2*a.capacity.size*len(a.value)<=p9.MAX_LEDGER_ENTRIES
    lp=dict(shapes=labels,prices=-raw.pi[len(a.value):].reshape(len(labels),a.demand.shape[1]));pairs=[];reference=None
    for rep in range(3):
        paths={}
        for name in (('baseline','variant') if rep%2==0 else ('variant','baseline')):
            tick=perf_counter();out=pg._replay(a,lp) if name=='baseline' else replay(a,lp,vector=True);physical=perf_counter()-tick;tick=perf_counter();metrics=checked(a,out);validation=perf_counter()-tick
            if reference is None:reference=out
            tick=perf_counter()
            assert out==reference
            equality=perf_counter()-tick
            paths[name]=dict(physical_wall=physical,independent_audit_and_metrics_wall=validation,equivalence_wall=equality,source_to_valid_wall=prefix+physical+validation+equality,metrics=metrics)
        pairs.append(dict(rep=rep,order=['baseline','variant'] if rep%2==0 else ['variant','baseline'],paths=paths))
    scalar,stages=replay(a,lp,instrument=True);vector,vector_stages=replay(a,lp,vector=True,instrument=True);assert scalar==vector==reference
    profiled,profile=profile_call(pg._replay,a,lp);assert profiled==reference
    variant_profiled,variant_profile=profile_call(lambda a,lp:replay(a,lp,vector=True),a,lp);assert variant_profiled==reference
    diag=None;greedy_profile=None
    if (len(a.value)+len(labels)*a.demand.shape[1])*len(labels)*len(a.value)<=1_000_000:
        diag=pg.shadow(a,solver=lambda m:result,strict_compatibility=True);assert diag['proposed']==reference
        control,greedy_profile=profile_call(oracle.greedy,a,'pgfp',11);assert control==(diag['baseline']['placement'],diag['baseline']['value'],diag['baseline']['future_score'])
    return dict(case=case,warm=warm,input_sha256=p9.input_hash(a),canonical_sha256=model.source_hash,result=result,strict_p5=cert,lp_source_to_certified_wall=prefix,pairs=pairs,baseline=reference,variant=vector,stages=stages,variant_stages=vector_stages,baseline_profile=profile,variant_profile=variant_profile,original_greedy_profile=greedy_profile,shadow=diag,shadow_status='BOUNDED_DIAGNOSTIC' if diag else 'SKIPPED_DENSE_BASELINE_LIMIT',synthetic=True,context='same-process first/warm LP; paired fixed prices; profiling after unprofiled repeats')

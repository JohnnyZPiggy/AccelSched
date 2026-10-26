"""P6-G0 PROBE_ONLY architecture diagnostics; no compact production claim."""
from time import perf_counter
import numpy as np
import certified_pgfp as pg
from solver_portfolio import solve_portfolio
from probes import pgfp_gpu_resident as b
from probes import pgfp_conflict_batch as c
from probes import pgfp_physical_profile as p10
from probes import portfolio_value_benchmark as p9

CASES=('epsilon','tie_competing','safe_disjoint','shared_reads','aidc64','p5_1k')
METHODS=('cpu_original_physical','p10_numpy','gpu_lp_cpu','p6b_cupy','p6c_cupy')
FIELDS=('task_order','shape_order','ledger_steps','placement','value','residual','greedy_residual','future_score')


def fixture(name):
    if name=='epsilon':return b.g.p4b.fixed_cases()[0][1]
    if name=='tie_competing':return b.p.edge_case()
    if name=='safe_disjoint':return c.special_cases()['safe_disjoint']
    if name=='shared_reads':return c.special_cases()['distinct_destination_score_conflict']
    if name=='aidc64':return p9.fixture('heterogeneous')
    if name=='p5_1k':return p9.fixture('p5_1k')
    raise ValueError('Unknown/bounded case; P9 10K physical skip retained')


def common(out):return {k:out[k] for k in FIELDS}


class Transfers:
    """Counts explicit public array-copy APIs only, not hidden native CUDA copies."""
    def __init__(self,cp):self.cp=cp;self.calls=[];self.syncs=0;self.cuda=self.Cuda(self)
    def __getattr__(self,name):return getattr(self.cp,name)
    def asarray(self,x,*args,**kwargs):
        self.calls.append(dict(direction='H2D' if isinstance(x,np.ndarray) else 'device_or_scalar_constructor',bytes=int(getattr(x,'nbytes',0))));return self.cp.asarray(x,*args,**kwargs)
    def asnumpy(self,x,*args,**kwargs):
        self.calls.append(dict(direction='D2H',bytes=int(x.nbytes)));return self.cp.asnumpy(x,*args,**kwargs)
    class Cuda:
        def __init__(self,parent):self.parent=parent
        def __getattr__(self,name):return getattr(self.parent.cp.cuda,name)
        def get_current_stream(self):return self.Stream(self.parent)
        class Stream:
            def __init__(self,parent):self.parent=parent
            def synchronize(self):self.parent.syncs+=1;self.parent.cp.cuda.get_current_stream().synchronize()
    def report(self):return dict(explicit_array_api_calls=self.calls,explicit_stream_syncs=self.syncs,meaning='Public asarray/asnumpy and stream synchronizes only; blocking D2H may sync; hidden solver copies/scalar kernel args/allocations not measured')


def certificate(a,result):
    if not result.certified or result.status!='OPTIMAL':raise ValueError('No certified LP')
    raw=result.raw;checked=b.g.admit(a,raw.x,raw.pi,raw.rc,-1,True)
    return dict(checked,x=raw.x,raw_pi=raw.pi,raw_rc=raw.rc,pi_sign=-1,reported_optimal=True,lp_input_sha256=b.g.lp_hash(a))


def run_physical(a,result,method,cp=None):
    labels=list(dict.fromkeys(a.shapes.tolist()));lp=dict(shapes=labels,prices=-result.raw.pi[len(a.value):].reshape(len(labels),3));tick=perf_counter();transfers=None
    if method=='cpu_original_physical':out=pg._replay(a,lp)
    elif method in ('p10_numpy','gpu_lp_cpu'):out=p10.replay(a,lp,vector=True)
    else:
        tracked=Transfers(cp);cert=certificate(a,result)
        out=b.replay(a,cert,'cupy',tracked) if method=='p6b_cupy' else c.replay(a,cert,'cupy',tracked)
        transfers=tracked.report()
    wall=perf_counter()-tick;tick=perf_counter();proof=b.g.audit_steps(a,out['ledger_steps'],out['task_order'],out['shape_order']);assert proof['placement']==out['placement'] and proof['residual']==out['residual'] and proof['future_score']==out['future_score'];metrics=p9.physical_metrics(a,out);audit_wall=perf_counter()-tick
    return dict(output=out,physical_inclusive_wall=wall,additional_independent_audit_and_metrics=audit_wall,metrics=metrics,transfers=transfers,memory_contract='LEGACY_FULL_TRACE_DIAGNOSTIC_ONLY' if method.startswith('p6') else 'FULL_CPU_LEDGER_REFERENCE',compact_performance_qualified=False)


def compact_proof(a,out):
    """Validate post-hoc compression against observed full trace, not a new engine."""
    steps=out['ledger_steps'];rest=a.capacity.copy().astype(float);choices=[]
    for step in steps:
        assert np.array_equal(rest,step['before']);j=step['task'];i=step['chosen'];choices.append(i)
        if i>=0:rest[i]-=a.demand[j]
        assert np.array_equal(rest,step['after'])
    ids=sorted({0,len(steps)//2,len(steps)-1});sample=[dict(index=k,before=steps[k]['before'],after=steps[k]['after']) for k in ids]
    return dict(task_order=out['task_order'],observed_choices=choices,observed_final_residual=out['greedy_residual'],sampled_full_trace=sample,verified_against_all_observed_states=True,classification='POST_HOC_COMPRESSION_PROOF_NOT_DEVICE_COMPACT_ENGINE',logical_state_note='Intermediate CPU states are deterministically derived from observed ordered commits, not claimed raw device observations. Existing engines still allocated full history.')


def measure(case,method,rep,cp=None):
    entrance=perf_counter();a=fixture(case);assert 2*a.capacity.size*len(a.value)<=p9.MAX_LEDGER_ENTRIES;model,labels=pg.canonical_shape(a);original=p9.bridge.p5.assemble(a);built=perf_counter()
    backend='highs' if method in ('cpu_original_physical','p10_numpy') else 'cupdlpx';result=solve_portfolio(model,policy=backend,device={'cuda':cp is not None,'source':'experiment device availability'},time_limit=10.,wall_budget=25.);solved=perf_counter();raw=result.raw
    strict=p9.bridge.p5.certify(original,raw.x,raw.pi,raw.rc,True,raw.objective,raw.dual_objective,raw.gap) if raw.dual_objective is not None else p9.bridge.p5.certify(original,raw.x,raw.pi,raw.rc,True,raw.objective);assert strict['certified'];certified=perf_counter()
    physical=run_physical(a,result,method,cp);ended=perf_counter()
    row=dict(case=case,method=method,rep=rep,input_sha256=p9.input_hash(a),canonical_sha256=model.source_hash,result=result,strict_p5=strict,physical=physical,timing=dict(cpu_input_and_sparse_build=built-entrance,portfolio_inclusive=solved-built,independent_original_kkt=certified-solved,source_to_certified=certified-entrance,source_to_independently_valid=ended-entrance),synthetic=True)
    if method.startswith('p6'):row['compact_proof']=compact_proof(a,physical['output'])
    return row

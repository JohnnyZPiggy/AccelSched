"""PROBE_ONLY sequential replay of actual PGFP; production modules stay unchanged.

production_1e-10 preserves threshold operation and NumPy signed-sum order. Torch
returns per-type signed contributions to NumPy for that same reduction: transferring
these small groups is intentional, measured overhead. strict_zero changes ONLY
future scoring slack, and is a divergence control, not a compatible backend.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np
import accelsched as api
import m0_oracle as oracle
from workload import build_workload
from probes.pgfp_tensor_probe import validate_inputs


def _ms():
    return time.perf_counter()*1000


def fixed_cases():
    cliff = build_workload([[1,1,1],[1,1+5e-11,1]], [[0,5e-11,0]], value=[1],
        eligible=[[True],[True]], shapes=[7,7], future=[[0,1,0]], weights=[1],
        future_eligible=[[True],[True]])
    mixed = build_workload([[2,4,8],[1,8,4],[2,4,8],[1,8,4]],
        [[1,.5,1],[1,1,.5],[1,.5,.5]], value=[3,2,2], shapes=[7,3,7,3],
        eligible=[[False,True,False],[False,True,False],[False,False,True],[True,False,True]],
        future=[[1,.5,1],[1,1,.5],[0,0,0]], weights=[1,.5,0],
        future_eligible=[[True,False,True],[True,False,True],[False,True,True],[False,True,True]])
    rng = np.random.default_rng(11)
    caps = np.array([[2,8,32],[4,16,64],[8,32,128],[4,12,96]],float)[np.arange(16)%4]
    demand = np.column_stack((rng.integers(1,4,24),rng.integers(1,17,24)/2,rng.integers(1,49,24)/2))
    future = np.array([[1,2,4],[2,4,8],[4,8,16],[1,.5,.5],[0,2,8],[0,0,0]],float)
    medium = build_workload(caps,demand,value=rng.integers(1,12,24),
        eligible=(np.arange(16)[:,None]%2 == np.arange(24)[None,:]%2),
        shapes=np.resize([7,3,11,5],16),future=future,weights=[1,.5,2,.25,1,0],
        future_eligible=(np.arange(16)[:,None]%2 == np.arange(6)[None,:]%2))
    return [('tolerance_cliff',cliff),('mixed_shapes_families',mixed),('moderate_16x24',medium)]


def fixture_hash(a):
    h=hashlib.sha256()
    for name in ('capacity','demand','value','eligible','shapes','future','weights','future_eligible'):
        x=getattr(a,name);h.update(name.encode());h.update(str((x.shape,x.dtype.str)).encode());h.update(x.tobytes())
    return h.hexdigest()


def _order(a,lp):
    utility=a.value[None,:]-lp['prices']@a.demand.T
    order=sorted(range(len(a.value)),key=lambda j:(-max(utility[:,j]),j))
    return utility,order


def _finish(a,p,order):
    """Independent load/value/residual/future reconstruction, including both orders."""
    p=tuple(int(i) for i in p)
    if not oracle.validate(a,p):raise ValueError('physical placement validation failed')
    load=np.zeros_like(a.capacity,float);canonical=a.capacity.astype(float).copy()
    for j,i in enumerate(p):
        if i>=0:
            if not a.eligible[i,j]:raise ValueError('physical eligibility failed')
            load[i]+=a.demand[j];canonical[i]-=a.demand[j]
    if not np.all(load<=a.capacity+1e-9):raise ValueError('physical resource violation')
    rest=a.capacity.astype(float).copy()
    for j in order:
        if p[j]>=0:rest[p[j]]-=a.demand[j]
    before=np.all(rest[:,None,:]+1e-10>=a.future[None,:,:],axis=2)
    future=float(np.sum((before&a.future_eligible)*a.weights[None,:]))
    if future!=oracle.future_score(a,rest):raise ValueError('future score reconstruction failed')
    return dict(status='feasible',placement=list(p),value=float(sum(a.value[j] for j,i in enumerate(p) if i>=0)),
                residual=canonical.tolist(),greedy_residual=rest.tolist(),future_score=future,
                feasible=True,task_order=order)


def production_reference(a):
    """Observe real solve -> imported greedy -> shape_lp/marginal; restore patches."""
    start=_ms();lp_holder=[];groups={};timing=dict(lp_ms=0.,score_compute_ms=0.)
    original_lp,original_marginal,original_greedy=oracle.shape_lp,oracle.marginal,api.greedy
    def lp_call(*args,**kwargs):
        tick=_ms();lp=original_lp(*args,**kwargs);timing['lp_ms']+=_ms()-tick;lp_holder.append(lp);return lp
    def marginal_call(instance,rest,i,j):
        tick=_ms();score=original_marginal(instance,rest,i,j);timing['score_compute_ms']+=_ms()-tick
        group=groups.setdefault(j,dict(task=j,shape=int(instance.shapes[i]),candidates=[],scores=[]))
        group['candidates'].append(i);group['scores'].append(score)
        return score
    greedy_result=[]
    def greedy_call(*args,**kwargs):
        tick=_ms();result=original_greedy(*args,**kwargs);timing['greedy_including_lp_ms']=_ms()-tick
        greedy_result.append(result);return result
    with patch.object(oracle,'shape_lp',lp_call),patch.object(oracle,'marginal',marginal_call),patch.object(api,'greedy',greedy_call):
        out=api.solve(a,method='pgfp',seed=11,time_limit=.5)
    timing['whole_solve_ms']=_ms()-start
    if out.status!='feasible' or out.placement is None or len(lp_holder)!=1:
        raise ValueError(f'production reference failed: {out.status}: {out.error}')
    _,order=_order(a,lp_holder[0]);tick=_ms();result=_finish(a,out.placement,order)
    if out.value!=result['value'] or not np.array_equal(out.residual,result['residual']) or greedy_result[0][2]!=result['future_score']:
        raise ValueError('actual solve/greedy output fails independent reconstruction')
    timing['independent_validation_ms']=_ms()-tick
    timing.update(replay_ms=timing['greedy_including_lp_ms']-timing['lp_ms'],transfer_ms=0.,
                  filter_dispatch_other_ms=timing['greedy_including_lp_ms']-timing['lp_ms']-timing['score_compute_ms'])
    result.update(trace=[dict(g,chosen=out.placement[j]) for j,g in groups.items()],timing=timing,
                  instrumentation='reference includes lightweight LP/marginal observation; remainder includes observation overhead')
    return result


def shadow(a,backend='numpy',mode='production_1e-10'):
    start=_ms()
    if backend not in ('scalar','numpy','torch_cpu','torch_cuda') or mode not in ('production_1e-10','strict_zero'):
        raise ValueError('unknown shadow backend/mode')
    if backend=='scalar' and mode!='production_1e-10':raise ValueError('scalar uses actual production marginal')
    error=api._input_error(a,.5,11)
    if error:raise ValueError(error)
    validate_inputs(a.capacity,a.demand,a.future,a.weights,a.future_eligible,a.eligible)
    timing=dict(input_validation_ms=_ms()-start,score_compute_ms=0.,transfer_ms=0.,candidate_filter_ms=0.,dispatch_ms=0.)
    tick=_ms();lp=oracle.shape_lp(a);timing['lp_ms']=_ms()-tick
    tick=_ms();utility,order=_order(a,lp);rest=a.capacity.astype(float).copy();p=[-1]*len(a.value)
    np.random.default_rng(11)  # Matches the production PGFP initialization, unused for ties.
    timing['preprocess_ms']=_ms()-tick
    tick=_ms();constants_transfer=0.
    if backend.startswith('torch'):
        import torch
        device='cuda' if backend=='torch_cuda' else 'cpu'
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA_UNAVAILABLE')
        transfer=_ms()
        q=torch.tensor(a.future,dtype=torch.float64,device=device)
        w=torch.tensor(a.weights.astype(float),dtype=torch.float64,device=device)
        e=torch.tensor(a.future_eligible,device=device)
        if device=='cuda':torch.cuda.synchronize()
        constants_transfer=_ms()-transfer
    timing['backend_init_ms']=_ms()-tick
    timing['transfer_ms']=constants_transfer
    trace=[];skipped=0;slack=1e-10 if mode=='production_1e-10' else 0.
    loop=_ms()
    for j in order:
        for h in sorted(range(len(lp['shapes'])),key=lambda h:(-utility[h,j],h)):
            tick=_ms()
            candidates=[i for i in range(len(rest)) if a.shapes[i]==lp['shapes'][h] and a.eligible[i,j]
                        and np.all(rest[i]+1e-10>=a.demand[j])]
            timing['candidate_filter_ms']+=_ms()-tick
            if not candidates:skipped+=1;continue
            if backend=='scalar':
                tick=_ms();scores=np.array([oracle.marginal(a,rest,i,j) for i in candidates]);timing['score_compute_ms']+=_ms()-tick
            elif backend=='numpy':
                tick=_ms();before=np.all(rest[candidates,None,:]+slack>=a.future[None,:,:],axis=2)&a.future_eligible[candidates]
                after=np.all((rest[candidates]-a.demand[j])[:,None,:]+slack>=a.future[None,:,:],axis=2)&a.future_eligible[candidates]
                scores=np.sum((after.astype(int)-before.astype(int))*a.weights[None,:],axis=1)
                timing['score_compute_ms']+=_ms()-tick
            else:
                tick=_ms();r=torch.tensor(rest[candidates],dtype=torch.float64,device=device)
                d=torch.tensor(a.demand[j],dtype=torch.float64,device=device)
                indices=torch.tensor(candidates,device=device)
                if device=='cuda':torch.cuda.synchronize()
                timing['transfer_ms']+=_ms()-tick
                tick=_ms();before=torch.all(r[:,None,:]+slack>=q[None,:,:],dim=2)&e[indices]
                after=torch.all((r-d)[:,None,:]+slack>=q[None,:,:],dim=2)&e[indices]
                contributions=(after.to(torch.int64)-before.to(torch.int64))*w[None,:]
                if device=='cuda':torch.cuda.synchronize()
                timing['score_compute_ms']+=_ms()-tick
                tick=_ms();host=contributions.cpu().numpy().copy()
                if device=='cuda':torch.cuda.synchronize()
                timing['transfer_ms']+=_ms()-tick
                # Same signed contributions and contiguous NumPy reduction order
                # as production; arbitrary torch reduction trees can change ties.
                tick=_ms();scores=np.sum(host,axis=1);timing['score_compute_ms']+=_ms()-tick
            tick=_ms();chosen=max(range(len(candidates)),key=lambda k:(float(scores[k]),-candidates[k]))
            i=candidates[chosen];p[j]=i
            trace.append(dict(task=j,shape=int(lp['shapes'][h]),candidates=candidates,scores=[float(s) for s in scores],chosen=i))
            rest[i]-=a.demand[j];timing['dispatch_ms']+=_ms()-tick
            break
    timing['replay_ms']=_ms()-loop
    tick=_ms();result=_finish(a,p,order)
    if not np.array_equal(rest,result['greedy_residual']):raise ValueError('sequential residual mismatch')
    timing['independent_validation_ms']=_ms()-tick
    timing['whole_solve_ms']=_ms()-start
    result.update(trace=trace,timing=timing,skipped_empty_shapes=skipped,backend=backend,mode=mode)
    return result


def compare(reference,replay):
    fields=('status','placement','value','residual','greedy_residual','future_score','task_order')
    flags={k:reference[k]==replay[k] for k in fields}
    flags['per_task_trace']=reference['trace']==replay['trace']
    differences=[dict(task=a['task'],reference_node=a['chosen'],shadow_node=b['chosen'],
                      reference_scores=a['scores'],shadow_scores=b['scores'])
                 for a,b in zip(reference['trace'],replay['trace']) if a!=b]
    return dict(status='EQUIVALENT' if all(flags.values()) else 'NOT_EQUIVALENT',
                exact_flags=flags,differences=differences[:3])


def run_probe(cuda=False):
    discovery=_ms();backends=['scalar','numpy'];environment=dict(python=sys.version.split()[0],numpy=np.__version__)
    gpu=dict(status='GPU_NOT_RUN',reason='not requested in CPU batch')
    try:import torch
    except ImportError:environment['torch']=None
    else:
        environment.update(torch=torch.__version__,torch_threads=torch.get_num_threads(),cuda=torch.version.cuda)
        backends.append('torch_cpu')
        if cuda and torch.cuda.is_available():
            backends.append('torch_cuda');gpu=dict(status='DEVICE_OBSERVED',name=torch.cuda.get_device_name(0))
        elif cuda:gpu['reason']='CUDA_UNAVAILABLE'
    discovery=_ms()-discovery;audits=[];records=[]
    for name,a in fixed_cases()[:2]:
        reference=production_reference(a)
        checks={b:compare(reference,shadow(a,b)) for b in backends}
        if any(r['status']!='EQUIVALENT' for r in checks.values()):raise ValueError('production-compatible divergence')
        audits.append(dict(name=name,fixture_sha256=fixture_hash(a),checks=checks,
                           strict_zero=compare(reference,shadow(a,'numpy','strict_zero'))))
    name,a=fixed_cases()[2];original=fixture_hash(a)
    for backend in ['production']+backends:
        measurements=[];equivalence=None
        for repetition in range(4):
            if backend=='production':
                result=production_reference(a);reference=result
            else:
                result=shadow(a,backend);equivalence=compare(reference,result)
                if equivalence['status']!='EQUIVALENT':raise ValueError('timed replay diverged')
            if repetition==0:warmup=result['timing']
            else:measurements.append(result['timing'])
        records.append(dict(backend=backend,warmup=warmup,measured=measurements,
                            median={k:statistics.median(x[k] for x in measurements) for k in measurements[0]},
                            equivalence=equivalence,status=result['status'],placement=result['placement'],
                            value=result['value'],future_score=result['future_score']))
    assert fixture_hash(a)==original
    source=os.environ.get('SAE_CANDIDATE_SHA')
    if source is None:
        try:source=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[1]),'rev-parse','HEAD'],text=True,timeout=2).strip()
        except (OSError,subprocess.SubprocessError):pass
    return dict(status='PASS',mode='PROBE_ONLY',source_sha=source,environment=environment,gpu=gpu,
                backend_discovery_ms=discovery,audits=audits,timed_fixture_sha256=original,
                timed_shape=dict(nodes=16,jobs=24,future_types=6,shapes=4,LP_variables=96),seed=11,
                warmups=1,measured_solves=3,records=records,
                timing_scope='independent LP each solve; replay includes filter/score/dispatch; transfer init overlaps backend_init; validation included in shadow whole; reference validation is additional audit; outer process wall includes module/torch imports; backend_discovery is availability only',
                limits='reference observation overhead; no production integration or general speedup claim')


if __name__=='__main__':
    try:import torch
    except ImportError:pass
    else:torch.set_num_threads(1)
    result=run_probe(cuda='--cuda' in sys.argv or 'SAE_CANDIDATE_SHA' in os.environ)
    if os.environ.get('SAE_CANDIDATE_SHA'):
        if result['gpu']['status']!='DEVICE_OBSERVED':raise RuntimeError('GPU_NOT_RUN')
        raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();target=Path('/content/gpu-probe.json');target.write_bytes(raw)
        print('ARTIFACT',target.name,len(raw),hashlib.sha256(raw).hexdigest())
    print(json.dumps(result,allow_nan=False))

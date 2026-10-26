"""Optional deterministic policy hypothesis; native limits are not hard wall bounds."""
from dataclasses import dataclass,field
import importlib.util
import os
import shutil
import subprocess
from time import perf_counter
import numpy as np
from lp_contract import CanonicalLP
from lp_backends import CertifiedResult,solve_lp

BACKENDS=('highs','hpdlp','cupdlpx')


@dataclass
class PortfolioResult(CertifiedResult):
    selection: dict=field(default_factory=dict)
    profile: dict=field(default_factory=dict)

    @property
    def certified(self):return bool(self.attempts and super().certified)
    @property
    def raw(self):return super().raw if self.attempts else None


def device_hint():
    """Cheap fallible discovery, never imports a CUDA runtime or creates a model."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'):return dict(cuda=False,source='CUDA_VISIBLE_DEVICES')
    executable=shutil.which('nvidia-smi')
    if not executable:return dict(cuda=False,source='nvidia-smi absent')
    try:
        p=subprocess.run([executable,'--query-gpu=name,compute_cap,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=1.)
        return dict(cuda=p.returncode==0 and bool(p.stdout.strip()),source='nvidia-smi (fallible hint)',details=p.stdout.strip(),error=p.stderr.strip())
    except (OSError,subprocess.TimeoutExpired) as e:return dict(cuda=False,source='probe failed',error=str(e))


def _positive(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.number)) or np.iscomplexobj(value) or not np.isfinite(value) or value<=0:raise ValueError(name+' must be positive finite')


def solve_portfolio(model,*,policy='auto',candidates=None,time_limit=10.,fallback=True,wall_budget=30.,device=None,warm=False):
    """fallback=False attempts only the first selected backend.

    Native time_limit is per attempt, clamped to remaining entrance-wall budget.
    Budget is checked between calls, not a hard process timeout. Uncertified
    infeasible/unbounded is fail-closed terminal, never a numerical success.
    """
    entrance=perf_counter()
    if not isinstance(model,CanonicalLP):raise TypeError('model must be CanonicalLP')
    model.assert_intact();_positive(time_limit,'time_limit');_positive(wall_budget,'wall_budget')
    if not isinstance(fallback,bool) or not isinstance(warm,bool):raise ValueError('fallback/warm must be bool')
    if policy not in ('auto',*BACKENDS):raise ValueError('unknown explicit policy')
    declared=list(candidates) if candidates is not None and not isinstance(candidates,str) else os.environ.get('SAE_LP_BACKENDS','cupdlpx,hpdlp,highs').split(',') if candidates is None else []
    if not declared or any(x not in BACKENDS for x in declared):raise ValueError('nonempty declared backend sequence required')
    declared=list(dict.fromkeys(declared));hint=device_hint() if device is None else dict(device)
    if not isinstance(hint.get('cuda'),bool):raise ValueError('device.cuda must be bool')
    features=dict(rows=model.A.shape[0],columns=model.A.shape[1],nnz=model.A.nnz,warm=warm,finite_row_lower=int(np.isfinite(model.row_lo).sum()),finite_row_upper=int(np.isfinite(model.row_hi).sum()),finite_var_lower=int(np.isfinite(model.var_lo).sum()),finite_var_upper=int(np.isfinite(model.var_hi).sum()))
    availability={};reasons=[]
    if policy!='auto':order=[policy];reasons.append('explicit backend honored, availability failures retained as attempts')
    else:
        order=[]
        for backend in declared:
            if backend=='highs':availability[backend]=True
            elif not hint['cuda']:availability[backend]=False
            else:
                try:availability[backend]=importlib.util.find_spec(backend) is not None
                except (ImportError,ValueError,AttributeError) as e:availability[backend]=False;reasons.append(backend+' availability probe failed: '+str(e))
            if availability[backend]:order.append(backend)
        if hint['cuda'] and not warm and features['columns']<1000 and 'highs' in order:
            order.remove('highs');order.insert(0,'highs');reasons.append('cold small-model CPU-first hypothesis; configurable via explicit policy/candidate exclusion')
        else:reasons.append('declared GPU order when device/import hints allow; CPU-only excludes optional imports')
    if fallback and 'highs' not in order:order.append('highs')
    if not fallback:order=order[:1]
    selection=dict(policy=policy,declared=declared,order=order,availability=availability,device=hint,features=features,reasons=reasons,hypothesis='P7B motivates an initial configurable order, not universal superiority or optimal threshold',source_hash=model.source_hash)
    selection_wall=perf_counter()-entrance;attempts=[];traces=[];status='NO_AVAILABLE_BACKEND'
    for backend in order:
        remaining=wall_budget-(perf_counter()-entrance)
        if remaining<=0:status='BUDGET_EXHAUSTED';break
        tick=perf_counter();result=solve_lp(model,backend=backend,fallback='none',time_limit=min(float(time_limit),remaining));wall=perf_counter()-tick
        assert len(result.attempts)==1 and result.source_hash==model.source_hash
        attempts.extend(result.attempts);a=result.attempts[0];traces.append(dict(backend=backend,attempt_wall=wall,status=result.status,certified=result.certified,native_stages=a.raw.timing,certification_dispatch_and_accounting_wall=max(0.,wall-a.raw.timing.get('total',0.)),native_time_limit=min(float(time_limit),remaining)))
        status=result.status
        if result.certified or status not in ('BACKEND_UNAVAILABLE','ERROR','TIME_LIMIT','NUMERIC_UNCERTIFIED'):break
    total=perf_counter()-entrance;attempt_wall=sum(t['attempt_wall'] for t in traces)
    profile=dict(selection_wall=selection_wall,attempts=traces,attempt_wall=attempt_wall,fallback_wall=sum(t['attempt_wall'] for t in traces[1:]),accounting_wall=max(0.,total-selection_wall-attempt_wall),total_wall=total,wall_budget=wall_budget,native_per_attempt_limit=time_limit,budget_is_hard_wall=False,memory='Caller may sample process RSS and device-wide VRAM lower bounds across whole call; no per-backend peak inferred')
    return PortfolioResult(status,tuple(attempts),total,model.source_hash,selection,profile)

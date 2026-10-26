"""Explicit certified LP adapters/fallback; optional CUDA import only on request."""
from dataclasses import dataclass,field
import importlib
import importlib.metadata
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import vstack
from lp_contract import CanonicalLP
from lp_certify import certify


@dataclass
class RawResult:
    backend: str
    version: str|None
    status: str
    native_status: object=None
    x: np.ndarray|None=None
    pi: np.ndarray|None=None
    rc: np.ndarray|None=None
    objective: float|None=None
    dual_objective: float|None=None
    gap: float|None=None
    witness: np.ndarray|None=None
    error: str|None=None
    timing: dict=field(default_factory=dict)
    settings: dict=field(default_factory=dict)


@dataclass
class Attempt:
    raw: RawResult
    status: str
    certificate: dict
    elapsed_seconds: float


@dataclass
class CertifiedResult:
    status: str
    attempts: tuple
    elapsed_seconds: float
    source_hash: str

    @property
    def certified(self):
        # Includes terminal infeasible/unbounded proofs; inspect status for feasibility.
        return self.attempts[-1].certificate.get('certified',False)
    @property
    def raw(self):return self.attempts[-1].raw


def _highs(model,time_limit):
    import scipy
    start=perf_counter();eq=np.isfinite(model.row_lo)&(model.row_lo==model.row_hi);up=np.isfinite(model.row_hi)&~eq;lo=np.isfinite(model.row_lo)&~eq
    A=vstack([model.A[up],-model.A[lo]],format='csr');rhs=np.r_[model.row_hi[up],-model.row_lo[lo]];converted=perf_counter()
    options=dict(time_limit=time_limit,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-10)
    p=linprog(model.c,A_ub=A if len(rhs) else None,b_ub=rhs if len(rhs) else None,A_eq=model.A[eq] if eq.any() else None,b_eq=model.row_hi[eq] if eq.any() else None,bounds=list(zip(model.var_lo,model.var_hi)),method='highs',options=options);solved=perf_counter()
    y=None;rc=None
    if p.x is not None and p.ineqlin.marginals is not None and p.eqlin.marginals is not None:
        y=np.zeros(len(model.row_lo));nu=int(up.sum());y[up]+=p.ineqlin.marginals[:nu];y[lo]-=p.ineqlin.marginals[nu:];y[eq]=p.eqlin.marginals
        rc=np.array(p.lower.marginals+p.upper.marginals,dtype=float,copy=True)
    return RawResult('highs','SciPy '+scipy.__version__,{0:'OPTIMAL',1:'TIME_LIMIT',2:'INFEASIBLE',3:'UNBOUNDED'}.get(p.status,'ERROR'),dict(code=int(p.status),message=p.message),None if p.x is None else np.array(p.x,copy=True),y,rc,None if p.fun is None else float(p.fun+model.c0),timing=dict(conversion=converted-start,native_call=solved-converted,total=perf_counter()-start),settings=options)


def _hpdlp(model,time_limit):
    start=perf_counter()
    try:module=importlib.import_module('hpdlp')
    except (ImportError,OSError) as e:return RawResult('hpdlp',None,'BACKEND_UNAVAILABLE',error=str(e),timing=dict(total=perf_counter()-start))
    imported=perf_counter();version=importlib.metadata.version('hpdlp')
    p=module.Model(objective_vector=model.c,constraint_matrix=model.A,constraint_lower_bound=model.row_lo,constraint_upper_bound=model.row_hi,variable_lower_bound=model.var_lo,variable_upper_bound=model.var_hi,objective_constant=model.c0)
    params=dict(TimeLimit=time_limit,OptimalityTol=1e-12,FeasibilityTol=1e-12,OptimalityNorm='linf',Presolve=False,OutputFlag=False)
    p.ModelSense=module.PDLP.MINIMIZE;p.setParams(**params)
    roundtrip=CanonicalLP(p.A,p.c,p.constr_lb,p.constr_ub,p.lb,p.ub,p.c0)
    if roundtrip.source_hash!=model.source_hash:raise ValueError('H-PDLP original-index/bound roundtrip changed')
    configured=perf_counter();p.optimize();solved=perf_counter()
    name=str(p.StatusName);status={'OPTIMAL':'OPTIMAL','TIME_LIMIT':'TIME_LIMIT','PRIMAL_INFEASIBLE':'INFEASIBLE','DUAL_INFEASIBLE':'UNBOUNDED'}.get(name,'ERROR')
    vectors=[None if x is None else np.array(x,dtype=float,copy=True) for x in (p.X,p.Pi,p.RC)]
    return RawResult('hpdlp',version,status,dict(name=name,code=p.Status,native_runtime=p.Runtime,presolve=getattr(p,'PresolveTime',None),rescaling=getattr(p,'RescalingTime',None)),*vectors,objective=p.ObjVal,dual_objective=p.DualObj,gap=p.Gap,timing=dict(lazy_import=imported-start,conversion=configured-imported,native_call_hidden_init_transfer_postsolve=solved-configured,extraction=perf_counter()-solved,total=perf_counter()-start),settings=params)


def _cupdlpx(model,time_limit):
    start=perf_counter()
    try:module=importlib.import_module('cupdlpx')
    except (ImportError,OSError) as e:return RawResult('cupdlpx',None,'BACKEND_UNAVAILABLE',error=str(e),timing=dict(total=perf_counter()-start))
    imported=perf_counter();version=importlib.metadata.version('cupdlpx')
    p=module.Model(objective_vector=model.c,constraint_matrix=model.A,constraint_lower_bound=model.row_lo,constraint_upper_bound=model.row_hi,variable_lower_bound=model.var_lo,variable_upper_bound=model.var_hi,objective_constant=model.c0)
    params=dict(TimeLimit=time_limit,OptimalityTol=1e-12,FeasibilityTol=1e-12,OptimalityNorm='linf',Presolve=False,OutputFlag=False)
    p.ModelSense=module.PDLP.MINIMIZE;p.setParams(**params)
    roundtrip=CanonicalLP(p.A,p.c,p.constr_lb,p.constr_ub,p.lb,p.ub,p.c0)
    if roundtrip.source_hash!=model.source_hash:raise ValueError('cuPDLPx original-index/bound roundtrip changed')
    configured=perf_counter();p.optimize();solved=perf_counter()
    name=str(p.StatusName);status={'OPTIMAL':'OPTIMAL','TIME_LIMIT':'TIME_LIMIT','PRIMAL_INFEASIBLE':'INFEASIBLE','DUAL_INFEASIBLE':'UNBOUNDED'}.get(name,'ERROR')
    vectors=[None if x is None else np.array(x,dtype=float,copy=True) for x in (p.X,p.Pi,p.RC)]
    return RawResult('cupdlpx',version,status,dict(name=name,code=p.Status,native_runtime=p.Runtime,presolve=getattr(p,'PresolveTime',None),rescaling=getattr(p,'RescalingTime',None)),*vectors,objective=p.ObjVal,dual_objective=p.DualObj,gap=p.Gap,timing=dict(lazy_import=imported-start,conversion=configured-imported,native_call_hidden_init_transfer_postsolve=solved-configured,extraction=perf_counter()-solved,total=perf_counter()-start),settings=params)


def solve_lp(model,*,backend='highs',fallback='none',time_limit=10.):
    # time_limit is per native attempt; native limits need not be hard wall limits.
    start=perf_counter()
    if not isinstance(model,CanonicalLP):raise TypeError('model must be CanonicalLP')
    model.assert_intact()
    if backend not in ('highs','hpdlp','cupdlpx') or fallback not in ('none','highs'):raise ValueError('explicit backend/fallback required')
    if isinstance(time_limit,(bool,np.bool_)) or not isinstance(time_limit,(int,float,np.number)) or np.iscomplexobj(time_limit) or not np.isfinite(time_limit) or time_limit<=0:raise ValueError('positive finite time_limit required')
    attempts=[]
    for selected in ([backend,'highs'] if fallback=='highs' and backend!='highs' else [backend]):
        tick=perf_counter()
        try:
            model.assert_intact();raw=(_highs if selected=='highs' else _hpdlp if selected=='hpdlp' else _cupdlpx)(model,float(time_limit));model.assert_intact();check=certify(model,raw)
        except Exception as e:raw=RawResult(selected,None,'ERROR',error=str(e),timing=dict(total=perf_counter()-tick));check=dict(certified=False,reason='ADAPTER_OR_CERTIFICATE_ERROR')
        if raw.status in ('INFEASIBLE','UNBOUNDED'):status=raw.status if check['certified'] else 'NOT_CERTIFIED_'+raw.status
        elif check['certified']:status='OPTIMAL'
        elif raw.status=='OPTIMAL':status='NUMERIC_UNCERTIFIED'
        else:status=raw.status
        attempts.append(Attempt(raw,status,check,perf_counter()-tick))
        if status not in ('BACKEND_UNAVAILABLE','ERROR','TIME_LIMIT','NUMERIC_UNCERTIFIED'):break
    return CertifiedResult(attempts[-1].status,tuple(attempts),perf_counter()-start,model.source_hash)

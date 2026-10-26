"""Explicit opt-in certified Shape-LP and physical PGFP shadow; defaults untouched."""
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
import m0_oracle as oracle
from lp_contract import CanonicalLP
from lp_backends import solve_lp


def canonical_shape(a):
    capacity=np.asarray(a.capacity);demand=np.asarray(a.demand);value=np.asarray(a.value);shapes=np.asarray(a.shapes)
    if capacity.ndim!=2 or demand.ndim!=2 or capacity.shape[1]!=demand.shape[1] or min(capacity.shape+ demand.shape)==0 or value.shape!=(len(demand),) or np.asarray(a.shapes).shape!=(len(capacity),):raise ValueError('invalid Shape arrays')
    if not np.issubdtype(shapes.dtype,np.integer) or np.any(shapes<0):raise ValueError('shapes must be nonnegative integer labels')
    labels=list(dict.fromkeys(shapes.tolist()))
    for x in (capacity,demand,value):
        if not np.issubdtype(x.dtype,np.number) or np.iscomplexobj(x) or not np.isfinite(x).all() or (x<0).any():raise ValueError('Shape data must be finite nonnegative real')
    n,r=demand.shape;m=len(labels);count=m*n*(r+1)
    if count>np.iinfo(np.int32).max:raise ValueError('Shape CSR int32 limit')
    caps=np.stack([capacity[np.asarray(a.shapes)==h].sum(axis=0) for h in labels]);ptr=np.empty(n+m*r+1,np.int32);ptr[:n+1]=np.arange(n+1,dtype=np.int32)*m;ptr[n+1:]=n*m+np.arange(1,m*r+1,dtype=np.int32)*n
    ids=np.empty(count,np.int32);data=np.empty(count,float);ids[:n*m].reshape(n,m)[:]=np.arange(n,dtype=np.int32)[:,None]+n*np.arange(m,dtype=np.int32)[None,:];data[:n*m]=1.
    for h in range(m):
        for k in range(r):lo=n*m+(h*r+k)*n;hi=lo+n;ids[lo:hi]=h*n+np.arange(n,dtype=np.int32);data[lo:hi]=demand[:,k]
    A=csr_matrix((data,ids,ptr),shape=(n+m*r,m*n));A.eliminate_zeros();upper=np.r_[np.ones(n),caps.ravel()]
    return CanonicalLP(A,-np.tile(value,m),np.full(len(upper),-np.inf),upper,np.zeros(m*n),np.full(m*n,np.inf)),labels


def shape_lp(a,*,backend='highs',fallback='none',time_limit=10.):
    start=perf_counter();model,labels=canonical_shape(a);result=solve_lp(model,backend=backend,fallback=fallback,time_limit=time_limit)
    out=dict(result=result,source_hash=model.source_hash,shapes=labels,elapsed_seconds=perf_counter()-start)
    if result.status=='OPTIMAL' and result.certified:
        n,r=a.demand.shape;out.update(value=-result.attempts[-1].certificate['primal_objective'],dual=-result.attempts[-1].certificate['dual_objective'],prices=-result.raw.pi[n:].reshape(len(labels),r),x=result.raw.x.reshape(len(labels),n))
    return out


def _replay(a,lp):
    labels=lp['shapes'];utility=a.value[None,:]-lp['prices']@a.demand.T;order=sorted(range(len(a.value)),key=lambda j:(-max(utility[:,j]),j));shape_orders=[sorted(range(len(labels)),key=lambda h:(-utility[h,j],h)) for j in range(len(a.value))];rest=a.capacity.astype(float).copy();placement=[-1]*len(a.value);steps=[]
    for j in order:
        step=dict(task=j,before=rest.tolist(),attempts=[],chosen=-1)
        for h in shape_orders[j]:
            ids=[i for i in range(len(rest)) if a.shapes[i]==labels[h] and a.eligible[i,j] and np.all(rest[i]+1e-10>=a.demand[j])];scores=[float(oracle.marginal(a,rest,i,j)) for i in ids];step['attempts'].append(dict(shape=labels[h],candidates=ids,scores=scores))
            if ids:chosen=max(ids,key=lambda i:(scores[ids.index(i)],-i));placement[j]=chosen;step['chosen']=chosen;rest[chosen]-=a.demand[j];break
        step['after']=rest.tolist();steps.append(step)
    if not oracle.validate(a,placement):raise ValueError('physical placement audit failed')
    return dict(task_order=order,shape_order=shape_orders,ledger_steps=steps,placement=placement,value=oracle.placed_value(a,placement),residual=oracle.residual(a,placement).tolist(),greedy_residual=rest.tolist(),future_score=oracle.future_score(a,rest))


def shadow(a,*,backend='highs',fallback='none',time_limit=10.,strict_compatibility=False):
    from accelsched import _input_error
    error=_input_error(a,time_limit,11)
    if error:raise ValueError(error)
    start=perf_counter();lp=shape_lp(a,backend=backend,fallback=fallback,time_limit=time_limit);solved=perf_counter()
    if lp['result'].status!='OPTIMAL':return dict(status='LP_NOT_CERTIFIED',lp=lp,elapsed_seconds=perf_counter()-start)
    proposed=_replay(a,lp);physical=perf_counter();baseline=_replay(a,oracle.shape_lp(a));flags={key:proposed[key]==baseline[key] for key in baseline};exact=all(flags.values())
    status='EXACT_BASELINE' if exact else 'STRICT_COMPATIBILITY_REJECTED' if strict_compatibility else 'POLICY_DIFFERENT_VALID'
    return dict(status=status,lp=lp,proposed=proposed,baseline=baseline,comparison=flags,accepted=not strict_compatibility or exact,timing=dict(certified_shape_lp=solved-start,physical_replay=physical-solved,baseline_lp_replay_compare=perf_counter()-physical,total=perf_counter()-start))

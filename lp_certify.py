"""Independent original-model optimality and terminal-ray certificates, float64."""
import numpy as np

TOL=1e-7


def _max(x):return float(np.max(x,initial=0.))
def _violation(x,lo,hi):return max(_max(lo-x),_max(x-hi))
def _sign(v,lo,hi):return max(_max(np.where(np.isfinite(lo),0.,v)),_max(np.where(np.isfinite(hi),0.,-v)))
def _support(v,lo,hi):
    l=(v>0)&np.isfinite(lo);u=(v<0)&np.isfinite(hi)
    return float(v[l]@lo[l]+v[u]@hi[u])
def _comp(v,x,lo,hi):
    l=(v>0)&np.isfinite(lo);u=(v<0)&np.isfinite(hi)
    return max(_max(np.abs(v[l]*(x[l]-lo[l]))),_max(np.abs(v[u]*(hi[u]-x[u]))))
def _valid(v,n):
    return isinstance(v,np.ndarray) and v.shape==(n,) and np.issubdtype(v.dtype,np.number) and not np.iscomplexobj(v) and np.isfinite(v).all()
def _scalar(v):return not isinstance(v,(bool,np.bool_)) and isinstance(v,(int,float,np.number)) and not np.iscomplexobj(v) and np.isfinite(v)


def certify(model,raw):
    model.assert_intact();m,n=model.A.shape;x,y,z=raw.x,raw.pi,raw.rc
    if raw.status=='INFEASIBLE':
        if not _valid(y,m) or not _valid(z,n):return dict(certified=False,kind='INFEASIBLE',reason='MISSING_FARKAS_RAY')
        norm=max(_max(np.abs(y)),_max(np.abs(z)))
        if norm==0:return dict(certified=False,kind='INFEASIBLE',reason='ZERO_RAY')
        y,z=y/norm,z/norm;stationarity=_max(np.abs(model.A.T@y+z));sign=max(_sign(y,model.row_lo,model.row_hi),_sign(z,model.var_lo,model.var_hi));margin=_support(y,model.row_lo,model.row_hi)+_support(z,model.var_lo,model.var_hi)
        return dict(certified=bool(np.isfinite([stationarity,sign,margin]).all() and stationarity<=TOL and sign<=TOL and margin>TOL),kind='INFEASIBLE',ray_normalization=norm,stationarity=stationarity,sign_residual=sign,contradiction_margin=margin,tolerance=TOL)
    if raw.status=='UNBOUNDED':
        if not _valid(x,n) or _max(np.abs(x))==0:return dict(certified=False,kind='UNBOUNDED',reason='MISSING_OR_ZERO_RAY')
        norm=_max(np.abs(x));ray=x/norm;ar=model.A@ray
        recession=lambda v,lo,hi:max(_max(np.where(np.isfinite(lo),-v,0.)),_max(np.where(np.isfinite(hi),v,0.)))
        res=max(recession(ar,model.row_lo,model.row_hi),recession(ray,model.var_lo,model.var_hi));direction=float(model.c@ray)
        witness=raw.witness if raw.witness is not None else np.zeros(n)
        if not _valid(witness,n):return dict(certified=False,kind='UNBOUNDED',reason='INVALID_WITNESS')
        feasible=_violation(model.A@witness,model.row_lo,model.row_hi);feasible=max(feasible,_violation(witness,model.var_lo,model.var_hi))
        return dict(certified=bool(np.isfinite([res,direction,feasible]).all() and res<=TOL and direction<-TOL and feasible<=TOL),kind='UNBOUNDED',ray_normalization=norm,recession_residual=res,objective_direction=direction,witness_residual=feasible,tolerance=TOL)
    if not all(_valid(v,size) for v,size in ((x,n),(y,m),(z,n))):return dict(certified=False,kind='OPTIMALITY',reason='MISSING_OR_INVALID_X_PI_RC')
    activity=model.A@x;reduced=model.c-model.A.T@y;primal=float(model.c@x+model.c0);dual=_support(y,model.row_lo,model.row_hi)+_support(z,model.var_lo,model.var_hi)+model.c0;scale=max(1.,abs(primal),abs(dual))
    d=dict(kind='OPTIMALITY',primal_residual=max(_violation(activity,model.row_lo,model.row_hi),_violation(x,model.var_lo,model.var_hi)),dual_sign_residual=max(_sign(y,model.row_lo,model.row_hi),_sign(z,model.var_lo,model.var_hi)),reduced_cost_sign_residual=_sign(reduced,model.var_lo,model.var_hi),stationarity_residual=_max(np.abs(reduced-z)),rc_max_error=_max(np.abs(reduced-z)),row_complementarity=_comp(y,activity,model.row_lo,model.row_hi),rc_complementarity=_comp(z,x,model.var_lo,model.var_hi),column_complementarity=_comp(reduced,x,model.var_lo,model.var_hi),primal_objective=primal,dual_objective=dual,objective_gap=abs(primal-dual),absolute_tolerance=TOL,scaled_tolerance=TOL*scale)
    fields=bool(_scalar(raw.objective) and abs(raw.objective-primal)<=TOL*scale)
    if raw.dual_objective is not None:fields=bool(fields and _scalar(raw.dual_objective) and abs(raw.dual_objective-dual)<=TOL*scale)
    if raw.gap is not None:fields=bool(fields and _scalar(raw.gap) and abs(abs(raw.gap)-abs(primal-dual))<=TOL*scale)
    d['native_objective_fields_match']=fields
    if not all(np.isfinite(v) for v in d.values() if isinstance(v,(int,float,np.number))):
        d.update(certified=False,reason='NONFINITE_ORIGINAL_ARITHMETIC');return d
    absolute=('primal_residual','dual_sign_residual','reduced_cost_sign_residual','stationarity_residual','rc_max_error','rc_complementarity')
    scaled=('row_complementarity','column_complementarity','objective_gap')
    d['certified']=bool(raw.status=='OPTIMAL' and fields and all(d[k]<=TOL for k in absolute) and all(d[k]<=TOL*scale for k in scaled));return d

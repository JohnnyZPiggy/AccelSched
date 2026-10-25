"""P7A isolated H-PDLP research: canonical ranged LP/KKT and HSD rays.

No production adapter. Optional hpdlp is imported only inside the GPU experiment.
Canonical min c'x+c0, l<=Ax<=u, lb<=x<=ub; Pi positive at lower,
negative at upper, RC same variable-bound convention; c-A.T@Pi-RC=0.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
import traceback
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import shape_lp_gpu_scale as p5

UPSTREAM='a89543313b0aa81b0e57c88f964f12daebfe5c65'
ARCHIVE='57b2d4b055edbf7a71a25e15b3ff453dd8ade7799c634380494f7bccaeab5ad3'
PINS={1000:('ee11179ff36bed61e25e446b4582620366f85a78537951cdbb0520df447594bc','ee8fa3a4bcb49dce081e1309a88bbe5a019a11b04f151e207d15f1bc29154d01'),
      10000:('6d15d2507015420b62c2302a26cfa1ffa026e26d9004338385a8c8331f027341','580be78fa276dc4b4d58092bd70a97f35ea99e2e15da14c06fec73ec9c6a2fd1')}
TOL=1e-7


def maximum(x):return float(np.max(x,initial=0.))


def violation(v,lo,hi):return max(maximum(lo-v),maximum(v-hi))


def sign_error(v,lo,hi):
    return max(maximum(np.where(np.isfinite(lo),0.,v)),maximum(np.where(np.isfinite(hi),0.,-v)))


def support(v,lo,hi):
    # Nonexistent-bound coefficients contribute zero; their forbidden signs are
    # independently measured, never flipped or used as infinite arithmetic.
    return float(v[(v>0)&np.isfinite(lo)]@lo[(v>0)&np.isfinite(lo)]+v[(v<0)&np.isfinite(hi)]@hi[(v<0)&np.isfinite(hi)])


def complementary(v,activity,lo,hi):
    lower=(v>0)&np.isfinite(lo);upper=(v<0)&np.isfinite(hi)
    return max(maximum(np.abs(v[lower]*(activity[lower]-lo[lower]))),maximum(np.abs(v[upper]*(hi[upper]-activity[upper]))))


def finite_vectors(m,x,y,z):
    return all(v is not None and np.asarray(v).shape==(n,) and np.isfinite(v).all() for v,n in ((x,len(m['c'])),(y,m['A'].shape[0]),(z,len(m['c']))))


def kkt(m,x,y,z,optimal,objective):
    if not finite_vectors(m,x,y,z):return dict(certified=False,reason='MISSING_OR_INVALID_VECTORS')
    x,y,z=map(np.asarray,(x,y,z));activity=m['A']@x;stationarity=m['c']-m['A'].T@y-z
    primal=float(m['c']@x+m['offset']);dual=support(y,m['l'],m['u'])+support(z,m['lb'],m['ub'])+m['offset'];scale=max(1.,abs(primal),abs(dual))
    d=dict(primal_residual=max(violation(activity,m['l'],m['u']),violation(x,m['lb'],m['ub'])),
        row_dual_sign_residual=sign_error(y,m['l'],m['u']),bound_dual_sign_residual=sign_error(z,m['lb'],m['ub']),
        stationarity_residual=maximum(np.abs(stationarity)),reduced_cost_max_error=maximum(np.abs(z-(m['c']-m['A'].T@y))),
        primal_objective=primal,dual_objective=dual,objective_gap=abs(primal-dual),
        row_complementarity=complementary(y,activity,m['l'],m['u']),bound_complementarity=complementary(z,x,m['lb'],m['ub']),
        objective_field_error=abs(float(objective)-primal) if objective is not None and np.isfinite(objective) else None,
        absolute_tolerance=TOL,scaled_gap_row_complementarity_tolerance=TOL*scale,
        worst_original_row=int(np.argmax(np.maximum(np.maximum(m['l']-activity,activity-m['u']),0.))) if len(activity) else None)
    absolute=('primal_residual','row_dual_sign_residual','bound_dual_sign_residual','stationarity_residual','reduced_cost_max_error','bound_complementarity')
    d['certified']=bool(optimal and d['objective_field_error'] is not None and d['objective_field_error']<=TOL*scale and all(d[k]<=TOL for k in absolute) and d['objective_gap']<=TOL*scale and d['row_complementarity']<=TOL*scale)
    return d


def farkas(m,y,z):
    if y is None or z is None or not finite_vectors(m,np.zeros(len(m['c'])),y,z):return dict(certified=False,reason='RAY_UNAVAILABLE_OR_INVALID')
    scale=max(maximum(np.abs(y)),maximum(np.abs(z)))
    if scale==0:return dict(certified=False,reason='ZERO_RAY')
    y,z=np.asarray(y)/scale,np.asarray(z)/scale
    d=dict(normalization_inf=scale,stationarity_residual=maximum(np.abs(m['A'].T@y+z)),
        row_sign_residual=sign_error(y,m['l'],m['u']),bound_sign_residual=sign_error(z,m['lb'],m['ub']),
        contradiction_margin=support(y,m['l'],m['u'])+support(z,m['lb'],m['ub']),absolute_tolerance=TOL)
    d['certified']=bool(max(d[k] for k in ('stationarity_residual','row_sign_residual','bound_sign_residual'))<=TOL and d['contradiction_margin']>TOL)
    return d


def unbounded(m,direction,witness):
    if direction is None or np.asarray(direction).shape!=m['c'].shape or not np.isfinite(direction).all():return dict(certified=False,reason='RAY_UNAVAILABLE_OR_INVALID')
    scale=maximum(np.abs(direction))
    if scale==0:return dict(certified=False,reason='ZERO_RAY')
    ray=np.asarray(direction)/scale;ar=m['A']@ray
    recession=lambda v,lo,hi:max(maximum(np.where(np.isfinite(lo),-v,0.)),maximum(np.where(np.isfinite(hi),v,0.)))
    d=dict(normalization_inf=scale,recession_residual=max(recession(ar,m['l'],m['u']),recession(ray,m['lb'],m['ub'])),
        objective_direction=float(m['c']@ray),feasible_witness_residual=max(violation(m['A']@witness,m['l'],m['u']),violation(witness,m['lb'],m['ub'])),absolute_tolerance=TOL)
    d['certified']=bool(d['recession_residual']<=TOL and d['objective_direction'] < -TOL and d['feasible_witness_residual']<=TOL)
    return d


def model(c,A,l,u,lb,ub,offset=0.):
    return dict(c=np.array(c,float),A=csr_matrix(A,dtype=float),l=np.array(l,float),u=np.array(u,float),lb=np.array(lb,float),ub=np.array(ub,float),offset=float(offset))


def smoke_models():
    return dict(hand_mixed=model([1,1],[[1,2],[0,1],[3,2]],[5,-np.inf,-np.inf],[5,2,8],[0,0],[np.inf,np.inf],2.),
        infeasible=model([1],[[1],[1]],[2,-np.inf],[np.inf,1],[-np.inf],[np.inf]),
        unbounded=model([-1],[[1]],[0],[np.inf],[0],[np.inf]))


def shape(n,nodes):
    a=p5.fixture(n,nodes);m=p5.assemble(a);A=m['A'];pins=(p5.digest(a.capacity,a.demand,a.value,a.shapes),p5.digest(A.data,A.indices,A.indptr,m['b'],m['c']))
    if n in PINS and pins!=PINS[n]:raise ValueError('P5 original fixture/CSR hash mismatch')
    canonical=model(m['c'],A,np.full(len(m['b']),-np.inf),m['b'],np.zeros(len(m['c'])),np.full(len(m['c']),np.inf))
    return canonical,m,dict(fixture_sha256=pins[0],lp_sha256=pins[1],**p5.dimensions(n,nodes))


def cpu_small(m):
    eq=np.isfinite(m['l'])&np.isfinite(m['u'])&(m['l']==m['u']);upper=np.isfinite(m['u'])&~eq;lower=np.isfinite(m['l'])&~eq
    from scipy.sparse import vstack
    A=vstack([m['A'][upper],-m['A'][lower]],format='csr');rhs=np.r_[m['u'][upper],-m['l'][lower]]
    out=linprog(m['c'],A_ub=A if len(rhs) else None,b_ub=rhs if len(rhs) else None,A_eq=m['A'][eq] if eq.any() else None,b_eq=m['u'][eq] if eq.any() else None,bounds=list(zip(m['lb'],m['ub'])),method='highs',options={'time_limit':5.})
    return dict(status=int(out.status),message=out.message,objective=None if out.fun is None else float(out.fun+m['offset']))


def validate(m,name,status,x,y,z,obj,shape_model=None):
    if name=='infeasible':check=farkas(m,y,z);check['certified']=bool(check['certified'] and status=='PRIMAL_INFEASIBLE' and x is None)
    elif name=='unbounded':check=unbounded(m,x,np.zeros(len(m['c'])));check['certified']=bool(check['certified'] and status=='DUAL_INFEASIBLE' and y is None and z is None)
    else:
        check=kkt(m,x,y,z,status=='OPTIMAL',obj)
        if shape_model is not None and finite_vectors(m,x,y,z):
            old=p5.certify(shape_model,x,y,z,status=='OPTIMAL',obj);old['retained_vectors']='Complete x/Pi/RC retained in P7A hash-verified archive.';check['p5_strict_original']=old;check['certified']=bool(check['certified'] and old['certified'])
    return check


def solve(m,name,module,shape_model=None):
    start=perf_counter();out=dict(status='FAILED',name=name);vectors={}
    try:
        p=module.Model(objective_vector=m['c'],constraint_matrix=m['A'],constraint_lower_bound=m['l'],constraint_upper_bound=m['u'],variable_lower_bound=m['lb'],variable_upper_bound=m['ub'],objective_constant=m['offset'])
        params=dict(TimeLimit=10.,OptimalityTol=1e-12,FeasibilityTol=1e-12,Presolve=False,OutputFlag=False,OptimalityNorm='linf',FarkasHomogeneousTol=1e-10,FarkasDisplacementTol=1e-10)
        p.setParams(**params);setup=perf_counter();p.optimize();done=perf_counter()
        raw={k:getattr(p,k,None) for k in ('Status','StatusName','ObjVal','DualObj','Gap','RelGap','Runtime','IterCount','RelPrimalResidual','RelDualResidual','PresolveTime','RescalingTime','FarkasMargin','FarkasResidual','HomogeneousScale')}
        x,y,z=(None if v is None else np.array(v,dtype=float,copy=True) for v in (p.X,p.Pi,p.RC));retrieved=perf_counter()
        vectors={k:v for k,v in zip(('x','pi','rc'),(x,y,z)) if v is not None}
        check=validate(m,name,p.StatusName,x,y,z,p.ObjVal,shape_model);audited=perf_counter()
        out.update(status='PASS' if check['certified'] else 'UNCERTIFIED',raw_solver=raw,parameters=params,certificate=check,
            timing=dict(model_conversion=setup-start,optimize_hidden_init_transfer_postsolve=done-setup,retrieval=retrieved-done,independent_original_audit=audited-retrieved,total=audited-start),
            hidden_stages='Native Runtime is iteration CPU clock, not wall. Presolve disabled; native PresolveTime/RescalingTime retained. GPU init/transfers/postsolve included but not exposed separately.')
    except Exception:out.update(error=traceback.format_exc(),timing=dict(total=perf_counter()-start))
    return out,vectors


def json_safe(x):
    if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
    if isinstance(x,np.generic):x=x.item()
    if isinstance(x,float) and not np.isfinite(x):return 'NONFINITE:'+repr(x)
    return x


def run(directory,archive_path,budget=240.,shape_only=False):
    start=perf_counter();directory=Path(directory);directory.mkdir(exist_ok=True)
    import hpdlp
    imported=perf_counter();archive_raw=Path(archive_path).read_bytes()
    if hashlib.sha256(archive_raw).hexdigest()!=ARCHIVE:raise ValueError('archived control SHA')
    archive=json.loads(archive_raw);report=dict(status='PASS',mode='PROBE_ONLY',solver='H-PDLP',upstream_commit=UPSTREAM,import_seconds=imported-start,
        canonical='min c*x+c0; l<=Ax<=u; lb<=x<=ub; original index unchanged; Pi lower positive/upper negative, RC same bound sign; stationarity c-A.T*Pi-RC=0',
        limits=dict(worker_seconds=budget,native_cpu_clock_per_solve=10.,measured_repetitions=2,max_shape_jobs=10000),smoke_cpu={},records=[],inputs={},artifacts=['p7a-vectors.json'],
        archived_control_sha256=ARCHIVE,comparison='Historical P5A HiGHS/cuPDLPx certificates, different session, no same-device speedup or new Shape HiGHS run.',time_policy='First/warm fresh Models in shared process, no warm-start state. TimeLimit excludes native init/presolve, outer wall bound applies; includes full original audit; archive serialization separate worker wall.')
    all_vectors={}
    def persist():
        buffer=io.BytesIO();np.savez_compressed(buffer,**all_vectors);raw=buffer.getvalue()
        payload=dict(archive_base64=base64.b64encode(raw).decode(),archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=len(raw),vectors={k:dict(shape=list(v.shape),dtype=v.dtype.str,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in all_vectors.items()})
        (directory/'p7a-vectors.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False));report['elapsed_seconds']=perf_counter()-start
        (directory/'results.json').write_text(json.dumps(json_safe(report),separators=(',',':'),allow_nan=False))
    planned=([] if shape_only else [(name,m,None,{}) for name,m in smoke_models().items()])+[(f'shape_{n}',None,n,dict(nodes=s)) for n,s in ((1000,64),(10000,256))]
    for name,m,n,meta in planned:
        if perf_counter()-start>budget-45:report.update(status='PARTIAL_TIME_BUDGET');break
        tick=perf_counter();shape_model=None
        if n is not None:
            m,shape_model,meta=shape(n,meta['nodes']);meta['archived_control']=next(t for t in archive['experiment']['tiers'] if t['n']==n)
        else:report['smoke_cpu'][name]=cpu_small(m)
        built=perf_counter();report['inputs'][name]=dict(meta,input_build_or_cpu_control_seconds=built-tick,rows=m['A'].shape[0],variables=len(m['c']),nnz=m['A'].nnz,input_sha256=p5.digest(m['A'].data,m['A'].indices,m['A'].indptr,m['c'],m['l'],m['u'],m['lb'],m['ub'],np.array([m['offset']])))
        for rep in range(2):
            with p5.Memory() as mem:out,vectors=solve(m,name,hpdlp,shape_model)
            key=name+'_'+str(rep);out.update(key=key,repetition=rep,label='first' if rep==0 else 'warm',memory=mem.report(),from_scratch_total_seconds=built-tick+out['timing']['total'])
            for field,value in vectors.items():all_vectors[key+'__'+field]=value
            report['records'].append(out)
            if out['status']!='PASS':report['status']='PARTIAL_UNCERTIFIED'
            persist()
        if name=='hand_mixed' and any(x['status']!='PASS' for x in report['records']):report['status']='BLOCKED_SMALL_CORRECTNESS';break
        if n is not None and any(x['status']!='PASS' for x in report['records'] if x['name']==name):report['status']='BLOCKED_SHAPE_CORRECTNESS';break
    persist();return report


def audit(summary,directory,output):
    wrapper=json.loads(Path(summary).read_bytes());experiment=wrapper['experiment'];manifest=wrapper['trace_artifacts']
    if len(manifest)!=1 or manifest[0]['name']!='p7a-vectors.json':raise ValueError('vector manifest')
    raw=(Path(directory)/manifest[0]['name']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest[0]['sha256'] or len(raw)!=manifest[0]['bytes']:raise ValueError('transport bytes/hash')
    payload=json.loads(raw)['data'];packed=base64.b64decode(payload['archive_base64'],validate=True)
    if len(packed)!=payload['archive_bytes'] or hashlib.sha256(packed).hexdigest()!=payload['archive_sha256']:raise ValueError('archive hash')
    vectors=np.load(io.BytesIO(packed),allow_pickle=False);checks=[]
    if set(vectors.files)!=set(payload['vectors']):raise ValueError('vector names')
    for key,meta in payload['vectors'].items():
        v=vectors[key]
        if list(v.shape)!=meta['shape'] or v.dtype.str!=meta['dtype'] or hashlib.sha256(v.tobytes()).hexdigest()!=meta['sha256']:raise ValueError('vector metadata')
    for record in experiment['records']:
        name=record['name'];key=record['key'];shape_model=None
        if name.startswith('shape_'):m,shape_model,_=shape(int(name[6:]),64 if name=='shape_1000' else 256)
        else:m=smoke_models()[name]
        x,y,z=[vectors[key+'__'+f] if key+'__'+f in vectors.files else None for f in ('x','pi','rc')];native=record.get('raw_solver',{})
        checked=validate(m,name,native.get('StatusName'),x,y,z,native.get('ObjVal'),shape_model)
        if checked['certified']!=record.get('certificate',{}).get('certified',False):raise ValueError('remote/local certification disagreement')
        checks.append(dict(key=key,status='PASS' if checked['certified'] else 'UNCERTIFIED',certificate=checked))
    out=dict(status='PASS_INDEPENDENT_ORIGINAL_MODEL_AUDIT' if checks and all(q['status']=='PASS' for q in checks) else 'COMPLETED_UNCERTIFIED_AUDIT',project_sha=wrapper['project_sha'],adapter_sha=wrapper['source_sha'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checks=checks,archive_sha256=payload['archive_sha256'])
    Path(output).write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],checks=len(checks))));return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir');parser.add_argument('--p5a-archive');parser.add_argument('--shape-only',action='store_true');parser.add_argument('--audit-summary');parser.add_argument('--audit-directory');parser.add_argument('--audit-output');args=parser.parse_args()
    if args.audit_summary:audit(args.audit_summary,args.audit_directory,args.audit_output)
    else:run(args.output_dir,args.p5a_archive,shape_only=args.shape_only)

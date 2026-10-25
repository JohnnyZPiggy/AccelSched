"""P4H PROBE_ONLY fixed optimal-face policies; no GPU allocation or production edits."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np
import scipy
from scipy.optimize import linprog, minimize, LinearConstraint
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probes import pgfp_dual_stability as g
from probes.shape_lp_backend_probe import audit, sparse_model, OPTIONS, fixture_sha
from probes.cupdlpx_lp_compare import solve_one

GRID=(0.,.25,.5,.75,1.)
NEAR_DELTA=1e-4
P4G_RESULT_SHA='c15ae064f39ca1301379f9de38b459abd3610cf515db9ebbca1ceb026bf7f16f'


def face_tol(value):
    return 1e-9*max(1.,abs(value))


def strict(a, payload, optimum):
    if payload['domain']!='STRICT_OPTIMAL' or payload['face_epsilon']!=0:
        raise ValueError('near-optimal data cannot enter strict admission')
    c=payload['certificate']
    c=g.admit(a,c['x'],c['raw_pi'],c['raw_rc'],c['pi_sign'],c['reported_optimal'])
    if abs(c['primal_objective']-optimum)>face_tol(optimum) or abs(c['dual_objective']-optimum)>face_tol(optimum):
        raise ValueError('not on strict dual-optimal face')
    return c


def vector(c):
    return np.concatenate((c['mu'],np.array(c['capacity_duals']).ravel()))


def from_vector(a, reference, y):
    m=sparse_model(a)
    c=g.admit(a,reference['x'],-y,m['c']+m['A'].T@y)
    payload=dict(domain='STRICT_OPTIMAL',face_epsilon=0.,certificate=c)
    strict(a,payload,reference['primal_objective'])
    return payload


def native(a,name,directory):
    if name.startswith('C'):
        c,provenance=g.reused_evidence(directory,a,'highs')
    else:
        r=solve_one(a,'highs')
        if r['status']!='PASS':raise ValueError('native HiGHS not certified')
        c=r['certificate'];provenance=dict(kind='LIVE_HIGHS',raw_solver=r['raw_solver'],
                                           timings=r['timings_seconds'])
    return dict(domain='STRICT_OPTIMAL',face_epsilon=0.,certificate=c),provenance


def support(a,reference,y):
    """Convex first-order global bound: F(z)>=F(y)+gradF(y)*(z-y)."""
    m=sparse_model(a);grad=np.zeros_like(y);grad[m['n']:]=2*y[m['n']:]
    r=linprog(grad,A_ub=-m['A'].T,b_ub=m['c'],A_eq=csr_matrix(m['b'][None,:]),
        b_eq=[reference['primal_objective']],bounds=(0,None),method='highs',options=OPTIONS.copy())
    if not r.success:raise ValueError('supporting LP failed: '+r.message)
    from_vector(a,reference,r.x)  # Independently certify the supporting point too.
    value=float(np.sum(y[m['n']:]**2));gap=max(0.,float(grad@y-r.fun))
    return dict(status=int(r.status),gradient_support_gap=gap,norm_squared=value,
                norm_squared_lower_bound=value-gap,parameters=OPTIONS.copy(),
                certified=gap<=1e-8*max(1.,value))


def minimum_norm(a,reference):
    m=sparse_model(a);y=vector(reference);value=float(np.sum(y[m['n']:]**2))
    if value==0:
        return from_vector(a,reference,y),dict(method='ZERO_NORM_GLOBAL_PROOF',norm_squared=0.,norm_lower_bound=0.)
    initial=support(a,reference,y)
    if initial['certified']:
        return from_vector(a,reference,y),dict(method='SUPPORTING_LP_GLOBAL_PROOF_AT_NATIVE',proof=initial)
    deadline=perf_counter()+5.
    def objective(z):return float(np.sum(z[m['n']:]**2))
    def gradient(z):
        grad=np.zeros_like(z);grad[m['n']:]=2*z[m['n']:];return grad
    def callback(z):
        if perf_counter()>deadline:raise TimeoutError('SLSQP callback deadline5s')
    r=minimize(objective,y,jac=gradient,method='SLSQP',bounds=[(0,None)]*len(y),
        constraints=[LinearConstraint(m['A'].T,-m['c'],np.inf),
                     LinearConstraint(m['b'][None,:],reference['primal_objective'],reference['primal_objective'])],
        options=dict(maxiter=30,ftol=1e-12,disp=False),callback=callback)
    if not r.success:raise ValueError('bounded SLSQP failed: '+r.message)
    candidate=from_vector(a,reference,r.x);proof=support(a,reference,r.x)
    if not proof['certified']:raise ValueError('norm global bound not certified')
    return candidate,dict(method='BOUNDED_CONVEX_SLSQP_WITH_SUPPORT_LP',initial_proof=initial,
                          proof=proof,iterations=int(r.nit),optimizer_message=r.message,
                          options=dict(maxiter=30,ftol=1e-12,callback_seconds=5.))


def anchored(a,reference,anchor):
    if not np.array_equal(anchor,reference['capacity_duals']):raise ValueError('anchor changed')
    return from_vector(a,reference,vector(reference)),dict(method='ZERO_DISTANCE_GLOBAL_PROOF',
        anchor=anchor.tolist(),squared_distance=0.,note='Trivial equality, not a claimed improvement.')


def grid_candidate(a,reference,lam):
    y=np.concatenate(((1-lam)*np.array([1.,2.]),[lam,0,0]))
    return from_vector(a,reference,y),dict(method='ANALYTIC_STRICT_GRID',lambda_gpu=lam)


def near_candidate(a,reference):
    if g.lp_hash(a)!=g.lp_hash(g.scenarios()['A_analytic_split']):raise ValueError('near control only A')
    m=sparse_model(a);y=np.array([0.,0.,1+NEAR_DELTA,0.,0.]);x=np.array(reference['x'])
    lam=y[2:].reshape(1,3);mu=y[:2];checks=audit(a,m,x,lam,mu)
    budget=3*NEAR_DELTA;v=reference['primal_objective'];rc=m['c']+m['A'].T@y
    tolerance=face_tol(v)
    feasible=all(checks[k]<=1e-7 for k in ('primal_residual','x_nonnegative_residual','dual_residual','dual_nonnegative_residual'))
    near_ok=(feasible and abs(checks['primal_objective']-v)<=tolerance
        and v-tolerance<=checks['dual_objective']<=v+budget+tolerance
        and checks['objective_gap']<=budget+tolerance
        and np.min(rc)>=-1e-7
        and max(checks['row_complementarity'],checks['column_complementarity'])<=budget+tolerance)
    if not near_ok:raise ValueError('near bound not certified')
    return dict(domain='NEAR_OPTIMAL',face_epsilon=budget,lambda_delta=NEAR_DELTA,
        near_certified=True,strict_optimal=False,certificate=checks,lp_input_sha256=g.lp_hash(a),
        x=x.tolist(),raw_pi=(-y).tolist(),raw_rc=rc.tolist(),capacity_duals=lam.tolist(),mu=mu.tolist(),
        actual_dual_suboptimality=checks['dual_objective']-v,
        note='Nonzero complementarity defect/gap; never eligible for strict selection.')


def replay_near(a,payload):
    """Separate bounded sensitivity path; never relaxes old strict replay admission."""
    if payload['domain']!='NEAR_OPTIMAL' or not payload['near_certified'] or payload['strict_optimal']:
        raise ValueError('not a labeled near control')
    if payload['lp_input_sha256']!=g.lp_hash(a):raise ValueError('near LP mismatch')
    # Regenerate/audit the single predeclared control instead of trusting its flags.
    ref=dict(x=payload['x'],primal_objective=3.)
    verified=near_candidate(a,ref)
    if payload!=verified:raise ValueError('near certificate tampering')
    start=perf_counter();labels=list(dict.fromkeys(a.shapes.tolist()))
    utility=a.value[None,:]-np.array(payload['capacity_duals'])@a.demand.T
    order=sorted(range(len(a.value)),key=lambda j:(-max(utility[:,j]),j))
    shape_order=[sorted(range(len(labels)),key=lambda h:(-utility[h,j],h)) for j in range(len(a.value))]
    rest=a.capacity.astype(float).copy();steps=[];trace=[]
    for j in order:
        step=dict(task=j,before=rest.tolist(),attempts=[],chosen=-1)
        for h in shape_order[j]:
            candidates=[i for i in range(len(rest)) if a.shapes[i]==labels[h] and a.eligible[i,j]
                        and np.all(rest[i]+1e-10>=a.demand[j])]
            scores=[float(g.oracle.marginal(a,rest,i,j)) for i in candidates]
            step['attempts'].append(dict(shape=labels[h],candidates=candidates,scores=scores))
            if not candidates:continue
            i=max(candidates,key=lambda i:(scores[candidates.index(i)],-i));step['chosen']=i
            trace.append(dict(task=j,shape=labels[h],candidates=candidates,scores=scores,chosen=i))
            rest[i]-=a.demand[j];break
        step['after']=rest.tolist();steps.append(step)
    result=g.audit_steps(a,steps,order,shape_order)
    result.update(trace=trace,ledger_steps=steps,utility=utility.tolist(),shape_order=shape_order,
                  near_replay_and_audit_seconds=perf_counter()-start)
    return result


def fragmentation(a,replay):
    rest=np.array(replay['greedy_residual']);rejected=[j for j,i in enumerate(replay['placement']) if i<0]
    positive=[a.demand[j,0] for j in rejected if a.demand[j,0]>0]
    threshold=float(min(positive)) if positive else None
    pieces=(rest[:,0]>0)&(rest[:,0]<threshold) if threshold is not None else np.zeros(len(rest),bool)
    return dict(residual_by_resource=rest.sum(axis=0).tolist(),positive_gpu_nodes=int(np.sum(rest[:,0]>0)),
                smallest_positive_rejected_gpu=threshold,subthreshold_gpu_nodes=int(pieces.sum()),
                subthreshold_gpu_units=float(rest[pieces,0].sum()))


def evaluate(a,identifier,factory,optimum,base_result=None,reference_cost=0.,native_cost=0.):
    start=perf_counter()
    try:
        payload,metadata=factory();c=strict(a,payload,optimum);generated=perf_counter()
        replay=g.replay(a,c);replayed=perf_counter()
        evaluation=dict(fragmentation=fragmentation(a,replay),
            price_norm_squared=float(np.sum(np.array(c['capacity_duals'])**2)))
        if base_result is not None:
            evaluation.update(differences=g.divergence(base_result['replay'],replay),
                native_price_distance=float(np.max(np.abs(np.array(c['capacity_duals'])-base_result['certificate']['capacity_duals']))))
        end=perf_counter();added=native_cost+end-start
        return dict(id=identifier,status='PASS',domain='STRICT_OPTIMAL',face_epsilon=0.,
                    face_numeric_tolerance=face_tol(optimum),certificate=c,metadata=metadata,replay=replay,
                    evaluation=evaluation,incremental_total_seconds=added,
                    fair_standalone_total_seconds=added+reference_cost,
                    timing=dict(generation_and_certificate=native_cost+generated-start,
                                replay_and_physical_audit=replayed-generated,evaluation=end-replayed))
    except Exception:
        return dict(id=identifier,status='METHOD_BLOCKED',error=traceback.format_exc(),
                    incremental_total_seconds=perf_counter()-start)


def select(records,pool,shared_reference_cost):
    if any(records[i].get('domain')=='NEAR_OPTIMAL' for i in pool):
        raise ValueError('near candidate in strict selection pool')
    start=perf_counter();valid=[records[i] for i in pool if records[i]['status']=='PASS']
    if not valid:raise ValueError('empty certified candidate pool')
    chosen=min(valid,key=lambda r:(-r['replay']['value'],-r['replay']['future_score'],
                                    r['incremental_total_seconds'],r['id']))
    elapsed=perf_counter()-start
    pool_cost=sum(records[i]['incremental_total_seconds'] for i in pool)
    return dict(selected_id=chosen['id'],pool=pool,certified_candidates=len(valid),
        selected_value=chosen['replay']['value'],selected_future_score=chosen['replay']['future_score'],
        realized_pool_generation_replay_evaluation_seconds=pool_cost,
        shared_reference_seconds=shared_reference_cost,selection_seconds=elapsed,
        fair_total_seconds=shared_reference_cost+pool_cost+elapsed,
        note='In-sample oracle-like upper bound over this finite pool, not free or predictive.')


def verify_p4g(path):
    prior=g.checked_json(path.read_bytes(),P4G_RESULT_SHA)
    root=Path(__file__).resolve().parents[1]
    blob=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD:probes/pgfp_dual_stability.py'],timeout=10).decode().strip()
    if blob!='8bc6b96f0843be11553d3efe2ec5111538ac6ab0':raise ValueError('P4G source blob')
    for name,a in g.scenarios().items():
        if prior['cases'][name]['physical_fixture_sha256']!=fixture_sha(a) or prior['cases'][name]['lp_input_sha256']!=g.lp_hash(a):
            raise ValueError('P4G fixed fixture mismatch')
    return dict(source_blob=blob,registered_results_sha256=P4G_RESULT_SHA)


def run(p4f_directory,p4g_results):
    start=perf_counter();provenance=verify_p4g(p4g_results);cases={}
    for name,a in g.scenarios().items():
        acquisition=perf_counter();payload,source=native(a,name,p4f_directory)
        optimum=payload['certificate']['primal_objective'];reference=strict(a,payload,optimum)
        anchor=np.array(reference['capacity_duals']).copy()  # Frozen before ANY physical outcome.
        acquisition=perf_counter()-acquisition
        control=g.control(a)
        candidates={}
        candidates['solver_native']=evaluate(a,'solver_native',lambda:(payload,source),optimum,native_cost=acquisition)
        baseline=candidates['solver_native']
        if baseline['status']!='PASS':raise ValueError('native replay failed')
        candidates['minimum_norm']=evaluate(a,'minimum_norm',lambda:minimum_norm(a,reference),optimum,baseline,acquisition)
        candidates['reference_anchor']=evaluate(a,'reference_anchor',lambda:anchored(a,reference,anchor),optimum,baseline,acquisition)
        near=None
        if name.startswith('A'):
            pool=[]
            for k,lam in enumerate(GRID):
                identifier='grid_'+str(k);pool.append(identifier)
                candidates[identifier]=evaluate(a,identifier,lambda lam=lam:grid_candidate(a,reference,lam),optimum,baseline,acquisition)
            tick=perf_counter();near=near_candidate(a,reference)
            try:strict(a,near,optimum)
            except ValueError:near['strict_admission_rejected']=True
            else:raise ValueError('near control entered strict face')
            # The payload is regenerated above; remove logging-only field for independent admission.
            near_payload={k:v for k,v in near.items() if k!='strict_admission_rejected'}
            near['replay']=replay_near(a,near_payload);near['fragmentation']=fragmentation(a,near['replay'])
            near['current_total_seconds']=perf_counter()-tick
            near['excluded_from_selection']=True
            selection=select(candidates,pool,acquisition)
        else:
            pool=['solver_native','minimum_norm','reference_anchor']
            if name.startswith('C'):
                def gpu():
                    c,meta=g.reused_evidence(p4f_directory,a,'cupdlpx')
                    return dict(domain='STRICT_OPTIMAL',face_epsilon=0.,certificate=c),meta
                candidates['historical_gpu']=evaluate(a,'historical_gpu',gpu,optimum,baseline,acquisition)
                pool=['solver_native','historical_gpu','minimum_norm','reference_anchor']
            selection=select(candidates,pool,0.)  # Native acquisition counted once inside pool.
        cases[name]=dict(physical_fixture_sha256=fixture_sha(a),lp_input_sha256=g.lp_hash(a),
            optimum=optimum,anchor_before_outcomes=anchor.tolist(),reference_acquisition_seconds=acquisition,
            control=control,candidates=candidates,placement_aware=selection,near_control=near,
            price_status='DUALS_MATCH' if all(r['status']=='PASS' and np.array_equal(r['certificate']['capacity_duals'],reference['capacity_duals'])
                for r in candidates.values()) else 'DISTINCT_OR_METHOD_BLOCKED',
            blocked_methods=[k for k,r in candidates.items() if r['status']=='METHOD_BLOCKED'])
    return dict(status='PASS',mode='PROBE_ONLY',cases=cases,provenance=provenance,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        gpu='REUSED_GPU_EVIDENCE' if cases['C_p4f_128']['candidates']['historical_gpu']['status']=='PASS'
            else 'GPU_EVIDENCE_BLOCKED',current_gpu_solve=False,total_seconds=perf_counter()-start,
        fragmentation_definition='GPU-size-only statistic: sum/count of nodes with 0<greedy GPU residual<smallest '
            'positive GPU demand among rejected jobs; zero when no such rejected job. Full residual resource sums also reported.',
        cost_definition='Standalone derived methods add shared native acquisition. Selection sums all pool '
            'generation/replay/evaluation costs (including blocked attempts), shared reference once, and selection overhead. '
            'Controls are separately recorded common experimental audit cost. Reused P4F times are historical only.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p4f-dir',required=True,type=Path);parser.add_argument('--p4g-results',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    try:result=run(args.p4f_dir,args.p4g_results)
    except Exception:result=dict(status='BLOCKED',error=traceback.format_exc())
    args.output.write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ('status','total_seconds','error')}))
    raise SystemExit(0 if result['status']=='PASS' else 2)

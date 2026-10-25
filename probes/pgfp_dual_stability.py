"""P4G PROBE_ONLY fixed dual sensitivity; no GPU allocation or production patches."""
import argparse
from dataclasses import replace
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
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accelsched as api
import m0_oracle as oracle
from probes import pgfp_shadow_integration as p4b
from probes.cupdlpx_lp_compare import solve_one
from probes.cupdlpx_lp_smoke import certificate, calibrate_hand_a
from probes.shape_lp_backend_probe import fixtures, fixture_sha, sparse_model, OPTIONS

REGISTERED = {
    'p4f-summary.json': 'eff327df56955a0b8e4fa66315795a959b8e968c97c08e3620ce845d9142b838',
    'p4f-hand_unique.json': '841364b3458780905c728689f290e9908bbcb05c26f02fecce88c83eda24df04',
    'p4f-moderate_m8_n128.json': '19d160bbc7c4e7d6207ab646985470e700ace3307c2ea7dcbab56de70414b313',
    'colab-control.json': '29ede0dbe349ceb43cf3037e39ce87dc9df83654200391a64a8ba63213039a2a',
}
P4F_SHA = 'b9ce88ac6a57e53e6c6d271dce253a03d36c3601'


def scenarios():
    # Same pooled hand LP, fixed physical split: job1-first fragments GPU capacity.
    a = oracle.Instance(np.array([[1.,0,0],[2.,0,0]]),
        np.array([[1.,0,0],[2.,0,0]]), np.array([1.,2.]), np.ones((2,2),bool),
        np.array([7,7]), np.array([[1.,0,0]]), np.array([1.]), np.ones((2,1),bool))
    b = p4b.fixed_cases()[2][1]
    base = fixtures()['moderate_seed11']
    s, n = base.eligible.shape
    future = np.array([[1,2,4],[2,4,8],[4,8,16],[1,.5,.5],[0,2,8],[0,0,0]],float)
    c = replace(base, eligible=np.arange(s)[:,None]%4 == np.arange(n)[None,:]%4,
        future=future, weights=np.array([1,.5,2,.25,1,0]),
        future_eligible=np.arange(s)[:,None]%2 == np.arange(6)[None,:]%2)
    for x in (a,b,c):
        if api._input_error(x,.5,11):
            raise ValueError(api._input_error(x,.5,11))
    return dict(A_analytic_split=a, B_p4b_heterogeneous=b, C_p4f_128=c)


def lp_hash(a):
    h = hashlib.sha256()
    for name in ('capacity','demand','value','shapes'):
        x = getattr(a,name)
        h.update(name.encode()); h.update(str((x.shape,x.dtype.str)).encode()); h.update(x.tobytes())
    return h.hexdigest()


def admit(a, x, pi, rc, sign=-1, optimal=True):
    checked = certificate(a,x,pi,rc,sign,optimal)
    if not checked['certified']:
        raise ValueError('external LP certificate rejected')
    checked['lp_input_sha256'] = lp_hash(a)
    return checked


def audit_steps(a, steps, order, shape_orders=None):
    """Independently rebuild eligibility, resource ledger, signed marginal and ties."""
    if [s['task'] for s in steps] != order or sorted(order) != list(range(len(a.value))):
        raise ValueError('step order/coverage')
    rest = a.capacity.astype(float).copy(); placement = [-1]*len(a.value)
    for step in steps:
        j = step['task']
        if not np.array_equal(rest,step['before']): raise ValueError('before ledger')
        if shape_orders is not None:
            labels=list(dict.fromkeys(a.shapes.tolist()))
            expected=[labels[h] for h in shape_orders[j]]
            if [x['shape'] for x in step['attempts']] != expected[:len(step['attempts'])]:
                raise ValueError('shape order mismatch')
        selected = -1
        visited = 0
        for attempt in step['attempts']:
            visited += 1
            h = attempt['shape']
            candidates = [i for i in range(len(rest)) if a.shapes[i] == h and a.eligible[i,j]
                          and np.all(rest[i]+1e-10 >= a.demand[j])]
            if candidates != attempt['candidates']: raise ValueError('candidate eligibility')
            scores = []
            for i in candidates:
                before = np.all(rest[i][None,:]+1e-10 >= a.future,axis=1)&a.future_eligible[i]
                after = np.all((rest[i]-a.demand[j])[None,:]+1e-10 >= a.future,axis=1)&a.future_eligible[i]
                scores.append(float(np.sum((after.astype(int)-before.astype(int))*a.weights)))
            if scores != attempt['scores']: raise ValueError('marginal arithmetic')
            if candidates:
                selected = max(candidates,key=lambda i:(scores[candidates.index(i)],-i))
                break
        if visited != len(step['attempts']): raise ValueError('extra attempts after placement')
        if shape_orders is not None and selected < 0 and visited != len(shape_orders[j]):
            raise ValueError('unexamined shape')
        if selected != step['chosen']: raise ValueError('dispatch/tie mismatch')
        placement[j] = selected
        if selected >= 0: rest[selected] -= a.demand[j]
        if not np.array_equal(rest,step['after']): raise ValueError('after ledger')
    result = p4b._finish(a,placement,order)
    if not np.array_equal(rest,result['greedy_residual']): raise ValueError('physical residual')
    result.update(accepted_jobs=[j for j,i in enumerate(placement) if i>=0],
                  accepted_count=sum(i>=0 for i in placement))
    return result


def replay(a, external):
    start = perf_counter()
    error = api._input_error(a,.5,11)
    if error: raise ValueError(error)
    if external['lp_input_sha256'] != lp_hash(a): raise ValueError('unrelated LP input')
    checked = admit(a,external['x'],external['raw_pi'],external['raw_rc'],
                    external['pi_sign'],external['reported_optimal'])
    if not np.array_equal(checked['capacity_duals'],external['capacity_duals']):
        raise ValueError('tampered certified prices')
    admitted = perf_counter()
    labels = list(dict.fromkeys(a.shapes.tolist()))
    utility = a.value[None,:]-np.array(checked['capacity_duals'])@a.demand.T
    order = sorted(range(len(a.value)),key=lambda j:(-max(utility[:,j]),j))
    shape_orders = [sorted(range(len(labels)),key=lambda h:(-utility[h,j],h)) for j in range(len(a.value))]
    rest = a.capacity.astype(float).copy(); steps=[]; groups=[]
    np.random.default_rng(11)  # Same unused PGFP RNG initialization, no altered tie rules.
    loop = perf_counter()
    for j in order:
        step = dict(task=j,before=rest.tolist(),attempts=[],chosen=-1)
        for h in shape_orders[j]:
            candidates = [i for i in range(len(rest)) if a.shapes[i] == labels[h] and a.eligible[i,j]
                          and np.all(rest[i]+1e-10 >= a.demand[j])]
            scores = [float(oracle.marginal(a,rest,i,j)) for i in candidates]
            step['attempts'].append(dict(shape=labels[h],candidates=candidates,scores=scores))
            if not candidates: continue
            chosen = max(candidates,key=lambda i:(scores[candidates.index(i)],-i))
            step['chosen'] = chosen
            groups.append(dict(task=j,shape=labels[h],candidates=candidates,scores=scores,chosen=chosen))
            rest[chosen] -= a.demand[j]
            break
        step['after'] = rest.tolist(); steps.append(step)
    dispatched = perf_counter()
    result = audit_steps(a,steps,order,shape_orders)
    end = perf_counter()
    result.update(trace=groups,ledger_steps=steps,utility=utility.tolist(),shape_order=shape_orders,
        timings_seconds=dict(input_certificate=admitted-start,preprocess=loop-admitted,
                             greedy=dispatched-loop,physical_audit=end-dispatched,total=end-start))
    return result


def divergence(left, right):
    def sequence(a,b):
        different = [i for i in range(max(len(a),len(b))) if i>=len(a) or i>=len(b) or a[i]!=b[i]]
        return dict(equal=a==b,first_difference=different[0] if different else None,
                    mismatch_count=len(different))
    return dict(task_order=sequence(left['task_order'],right['task_order']),
        shape_order=sequence(left['shape_order'],right['shape_order']),
        placement=sequence(left['placement'],right['placement']),
        accepted_jobs=sequence(left['accepted_jobs'],right['accepted_jobs']),
        accepted_symmetric_difference=sorted(set(left['accepted_jobs'])^set(right['accepted_jobs'])),
        accepted_count_delta=right['accepted_count']-left['accepted_count'],
        value_delta=right['value']-left['value'],future_score_delta=right['future_score']-left['future_score'],
        utility_max_error=float(np.max(np.abs(np.array(left['utility'])-right['utility']))))


def control(a):
    start = perf_counter(); lp = oracle.shape_lp(a); solved = perf_counter()
    lam = lp['prices']; mu = np.maximum(0,np.max(a.value[None,:]-lam@a.demand.T,axis=0))
    canonical = np.concatenate((mu,lam.ravel())); model = sparse_model(a)
    c = admit(a,lp['x'],-canonical,model['c']+model['A'].T@canonical)
    injected = replay(a,c)
    tick = perf_counter(); actual = api.solve(a,method='pgfp',seed=11,time_limit=.5)
    api_wall = perf_counter()-tick
    if (actual.status != 'feasible' or tuple(injected['placement']) != actual.placement
        or injected['value'] != actual.value or not np.array_equal(actual.residual,injected['residual'])):
        raise ValueError('production solve control not exact')
    tick = perf_counter(); old = p4b.shadow(a,backend='scalar'); shadow_wall=perf_counter()-tick
    flags = p4b.compare(old,injected)
    if flags['status'] != 'EQUIVALENT': raise ValueError('P4B scalar trace not exact')
    return dict(status='EXACT',certificate=c,replay=injected,p4b_exact_flags=flags,
                production_api_wall_seconds=api_wall,p4b_wall_seconds=shadow_wall,
                control_lp_seconds=solved-start,total_seconds=perf_counter()-start,
                note='Unmodified api.solve and P4B scalar shadow; no monkeypatch.')


def checked_json(raw, expected):
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('P4F SHA256 mismatch')
    return json.loads(raw)


def reused_evidence(directory, a, backend):
    records = {name:checked_json((directory/name).read_bytes(),digest) for name,digest in REGISTERED.items()}
    summary = records['p4f-summary.json']; controller = records['colab-control.json']
    if summary['project_sha'] != P4F_SHA or summary['scientific_status'] != 'PASS':
        raise ValueError('P4F source/status')
    if (not controller['cleanup_confirmed'] or controller['readbacks'][-1]['assignments'] != 0
        or controller['readbacks'][-1]['status'] != 'AUTHENTICATED_ZERO'):
        raise ValueError('P4F cleanup evidence')
    if any(r['source_sha']!=summary['source_sha'] for r in records.values()):
        raise ValueError('P4F adapter identity')
    root = Path(__file__).resolve().parents[1]
    for name,pin in {'probes/pgfp_shadow_integration.py':'dec7c0377cf08ba589ccb3146e5c85252bdb37f7',
                     'probes/cupdlpx_lp_compare.py':'33135b55135f645712a3c2cf617d4fa39a596c19'}.items():
        actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD:'+name],timeout=10).decode().strip()
        if actual!=pin:raise ValueError('pinned source blob mismatch')
    for name,info in summary['source_records'].items():
        blob = subprocess.check_output(['git','-C',str(root),'show','HEAD:'+name],timeout=10)
        working = (root/name).read_bytes().replace(b'\r\n',b'\n')
        if hashlib.sha256(blob).hexdigest()!=info['sha256'] or working!=blob:
            raise ValueError('P4F executed source does not match current imported source')
    original = fixtures()['moderate_seed11']
    if lp_hash(original) != lp_hash(a): raise ValueError('C altered LP fields')
    case = records['p4f-moderate_m8_n128.json']
    if case['project_sha'] != P4F_SHA or case['case']['fixture_sha256'] != fixture_sha(original):
        raise ValueError('P4F fixture/source identity')
    hand = records['p4f-hand_unique.json']
    if hand['project_sha'] != P4F_SHA or hand['case']['fixture_sha256'] != fixture_sha(fixtures()['hand_unique']):
        raise ValueError('hand A identity')
    hand_cert = hand['case']['measured']['cupdlpx'][0]['certificate']
    sign = calibrate_hand_a(hand_cert['x'],hand_cert['raw_pi'],hand_cert['raw_rc'])['pi_sign']
    chosen = case['case']['measured'][backend][0]
    raw = chosen['raw_solver']; observed = chosen['certificate']
    if raw['Status'] != 0 or chosen['status'] != 'PASS': raise ValueError('P4F optimal status')
    c = admit(a,observed['x'],observed['raw_pi'],observed['raw_rc'],sign)
    tolerance = 1e-7*max(1,abs(c['primal_objective']))
    if abs(raw['ObjVal']+c['primal_objective']) > tolerance: raise ValueError('P4F objective')
    if backend == 'cupdlpx' and (raw['StatusName'] != 'OPTIMAL'
        or abs(raw['DualObj']+c['dual_objective']) > tolerance
        or abs(abs(raw['Gap'])-c['objective_gap']) > tolerance):
        raise ValueError('P4F GPU dual/gap')
    return c,dict(kind='REUSED_GPU_EVIDENCE' if backend=='cupdlpx' else 'REUSED_CPU_HIGHS_EVIDENCE',
        registered_sha256=REGISTERED,source_sha=P4F_SHA,adapter_sha=summary['source_sha'],
        repetition=0,raw_solver=raw,current_lp_solve_seconds=None,
        historical_p4f_timings_seconds=chosen['timings_seconds'],
        note='Historical LP timing is provenance only, not live P4G GPU performance.')


def analytic(a, lam):
    mu = np.array([1.,2.]) if lam == 0 else np.zeros(2)
    canonical = np.concatenate((mu,[lam,0,0])); m=sparse_model(a)
    return admit(a,[[1,1]],-canonical,m['c']+m['A'].T@canonical),dict(
        kind='ANALYTIC_CERTIFIED',capacity_price=lam,current_lp_solve_seconds=None)


def alternate(a, reference):
    start = perf_counter(); m=sparse_model(a); cost=np.zeros(len(m['b']));cost[m['n']]=-1
    result=linprog(cost,A_ub=-m['A'].T,b_ub=m['c'],A_eq=csr_matrix(m['b'][None,:]),
        b_eq=[reference['primal_objective']],bounds=(0,None),method='highs',options=OPTIONS.copy())
    if not result.success: raise ValueError('dual face LP not optimal: '+result.message)
    canonical=result.x
    c=admit(a,reference['x'],-canonical,m['c']+m['A'].T@canonical)
    return c,dict(kind='CONSTRUCTED_DUAL_OPTIMAL_FACE',objective='maximize first shape GPU price',
        solver_status=int(result.status),parameters=OPTIONS,current_lp_solve_seconds=perf_counter()-start,
        note='Independent dual LP with exact reference objective equality; not GPU evidence.')


def timed_path(a, acquire):
    start=perf_counter(); c,provenance=acquire(); acquired=perf_counter(); result=replay(a,c)
    return dict(certificate=c,provenance=provenance,replay=result,
                current_timing_seconds=dict(acquisition_and_certificate=acquired-start,
                    replay_and_physical_audit=result['timings_seconds']['total'],total=perf_counter()-start))


def run(directory):
    start=perf_counter(); records={}
    for name,a in scenarios().items():
        reference=control(a)
        if name.startswith('A'):
            left=timed_path(a,lambda:analytic(a,0));right=timed_path(a,lambda:analytic(a,1))
        elif name.startswith('B'):
            def cpu():
                lp=solve_one(a,'highs')
                if lp['status']!='PASS':raise ValueError('live HiGHS uncertified')
                c=lp['certificate'];c=admit(a,c['x'],c['raw_pi'],c['raw_rc'])
                return c,dict(kind='LIVE_HIGHS',current_lp_solve_seconds=lp['timings_seconds']['total'],
                              lp_timings_seconds=lp['timings_seconds'],raw_solver=lp['raw_solver'])
            left=timed_path(a,cpu);right=timed_path(a,lambda:alternate(a,left['certificate']))
            right['shared_reference_lp_seconds']=left['provenance']['current_lp_solve_seconds']
        else:
            left=timed_path(a,lambda:reused_evidence(directory,a,'highs'))
            right=timed_path(a,lambda:reused_evidence(directory,a,'cupdlpx'))
        distance=float(np.max(np.abs(np.array(left['certificate']['capacity_duals'])-right['certificate']['capacity_duals'])))
        records[name]=dict(physical_fixture_sha256=fixture_sha(a),lp_input_sha256=lp_hash(a),
            instance={k:getattr(a,k).tolist() for k in a.__dataclass_fields__},
            control=reference,paths=dict(left=left,right=right),max_price_distance=distance,
            price_status='DUALS_MATCH' if distance==0 else 'DISTINCT_CERTIFIED_PRICES',
            differences=divergence(left['replay'],right['replay']))
    return dict(status='PASS',mode='PROBE_ONLY',cases=records,gpu='REUSED_GPU_EVIDENCE',
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        current_gpu_solve=False,total_seconds=perf_counter()-start,
        future_note='future_score counts independently eligible future types, not simultaneous packing.',
        timing_note='Current CPU acquisition/certificate/replay/audit only for reused evidence. '
                    'No current GPU solve/transfer time, no cross-runtime CPU/GPU speed comparison.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p4f-dir',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    try: result=run(args.p4f_dir)
    except Exception: result=dict(status='BLOCKED',mode='PROBE_ONLY',error=traceback.format_exc())
    args.output.write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ('status','gpu','total_seconds','error')}))
    raise SystemExit(0 if result['status']=='PASS' else 2)

from dataclasses import replace
import hashlib
import sys
import numpy as np
import pytest
from accelsched import METHODS, SolveResult, solve
import bench_v4_static as b


@pytest.mark.parametrize('scenario',b.SCENARIOS)
def test_fixed_input_family_shapes_pressure_and_futures(scenario):
    a=b.instance(64,1000,scenario,11)
    assert b.fingerprint(a)==b.fingerprint(b.instance(64,1000,scenario,11))
    assert b.fingerprint(a)!=b.fingerprint(b.instance(64,1000,scenario,29))
    assert np.all(a.capacity[:,0]==np.floor(a.capacity[:,0]))
    assert np.all(a.demand[:,0]==np.floor(a.demand[:,0])) and np.all(a.weights>0)
    for label in set(a.shapes):
        ids=np.flatnonzero(a.shapes==label)
        assert np.all(a.capacity[ids]==a.capacity[ids[0]])
        assert np.all(a.future_eligible[ids]==a.future_eligible[ids[0]])
    assert not a.eligible[a.capacity[:,0]==0][:,a.demand[:,0]>0].any()
    if scenario=='heterogeneous': assert np.max(a.demand.sum(axis=0)/a.capacity.sum(axis=0))<1.5
    if scenario=='fragmentation': assert np.all(a.demand[:750,0]==1) and np.all(a.demand[750:,0]>=4)
    if scenario=='eligibility': assert not a.eligible[:,::17].any()


def test_independent_audit_rejects_corruption():
    a=b.build_workload([[1,4,16],[1,4,16]],[[1,4,16],[1,4,16]],value=[2,3],
                       eligible=[[True,False],[False,True]],shapes=[0,0])
    good=SolveResult('first_fit','feasible',(0,1),5.,np.zeros((2,3)),0.,False)
    assert b.audit(a,good)['valid']
    for out,reason in [(replace(good,placement=(0,0)),'eligibility'),
                       (replace(good,residual=np.ones((2,3))),'residual'),
                       (replace(good,value=6.),'objective'),
                       (replace(good,placement=(0,0.5)),'assignment shape/type/bounds'),
                       (replace(good,placement=(0,1,1)),'assignment shape/type/bounds')]:
        assert b.audit(a,out)==dict(valid=False,reason=reason,metrics=None)
    a.eligible[:]=True
    assert b.audit(a,replace(good,placement=(0,0)))['reason']=='capacity'
    assert b.audit(a,replace(good,placement=None))['valid'] is None


def test_real_six_paths_hash_immutability_and_clocks():
    a=b.instance(8,16,'heterogeneous',11); initial=b.fingerprint(a)
    for method in METHODS:
        row=b.measured(a,method,11,1.)
        direct=solve(a,method,seed=11,time_limit=1.)
        assert row['solver_status']==direct.status and row['output']['value']==direct.value
        assert row['audit']['valid'] and row['input_unchanged'] and b.fingerprint(a)==initial
        assert row['solve_plus_audit_ms']==row['solve_ms']+row['audit_metrics_hash_ms']
        assert row['output_sha256']==hashlib.sha256(b.json_bytes(row['output'])).hexdigest()
        assert row['certified_optimal']==(method=='highs_mip')
    a.value[0]+=1
    assert b.fingerprint(a)!=initial


@pytest.mark.parametrize('status,placement',[('feasible_uncertified',(-1,)*16),('uncertified',None),
    ('invalid',None),('failed',None),('input_invalid',None)])
def test_preserve_status_missing_results_and_no_false_certificate(monkeypatch,status,placement):
    a=b.instance(8,16,'heterogeneous',11)
    out=SolveResult('highs_mip',status,placement,0. if placement else None,
                    a.capacity.copy() if placement else None,1000.,False,'injected mapping test')
    monkeypatch.setattr(b,'solve',lambda *args,**kwargs:out)
    row=b.measured(a,'highs_mip',11,1.)
    assert row['solver_status']==status and not row['certified_optimal']
    assert row['output']['placement']==(list(placement) if placement else None)
    assert row['solve_plus_audit_ms']<row['nested_solver_elapsed_ms']


def test_real_watchdog_and_nonzero_exit():
    out=b.watched([sys.executable,'-c','import time;time.sleep(5)'],.15)
    assert out['watchdog_timeout'] and out['process_wall_ms']<3000
    out=b.watched([sys.executable,'-c','raise SystemExit(7)'],5)
    assert out['exit_code']==7 and not out['watchdog_timeout']


def test_detect_mutated_input(monkeypatch):
    a=b.instance(8,16,'heterogeneous',11); original=b.solve
    def changed(a,*args,**kwargs):
        out=original(a,*args,**kwargs);a.weights[0]+=1;return out
    monkeypatch.setattr(b,'solve',changed)
    assert not b.measured(a,'first_fit',11,1.)['input_unchanged']

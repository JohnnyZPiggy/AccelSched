"""Focused P9 CPU evidence, no mocked GPU numerical success."""
import numpy as np
from solver_portfolio import solve_portfolio
from lp_backends import RawResult
import lp_backends as b
from probes import portfolio_value_benchmark as p
from probes import pgfp_dual_stability as g

def test_fixed_synthetic_fixture_identity_and_policy_rotation():
    for name in p.CASES:
        a=p.fixture(name);assert p.input_hash(a)==p.input_hash(p.fixture(name))
    assert p.policy_order(0)!=p.policy_order(1) and set(p.policy_order(0))==set(p.policy_order(1))
    assert len({p.input_hash(p.fixture(n)) for n in p.CASES})==5

def test_real_cpu_physical_and_strict_guard():
    r=p.measure('heterogeneous','highs',0,device={'cuda':False})
    assert r['certified'] and r['physical_status']=='VALIDATED'
    assert r['shadow']['accepted']==all(r['shadow']['comparison'].values())
    assert r['metrics']['accepted_jobs']<=r['metrics']['offered_jobs']
    assert all(0<=v<=1+1e-10 for v in r['metrics']['utilization'] if v is not None)
    t=r['timing'];assert abs(sum(t[k] for k in ('source_generation','canonical_and_original_build','portfolio_inclusive','independent_p5_certificate'))-t['lp_source_to_certified'])<1e-9
    assert t['source_to_valid_placement']>=t['lp_source_to_certified'] and t['total_including_diagnostic']>=t['source_to_valid_placement']
    arrays={};out=p.retain(r,arrays,'test');assert arrays and out['replay']['archive_key'] in arrays

def test_uncertified_attempt_has_no_performance_success(monkeypatch):
    monkeypatch.setattr(b,'_cupdlpx',lambda *_:RawResult('cupdlpx',None,'OPTIMAL',error='no vectors; failure fixture'))
    r=p.measure('heterogeneous','cupdlpx',0,solver=lambda m:solve_portfolio(m,policy='cupdlpx',fallback=False,device={'cuda':False}))
    assert not r['certified'] and r['timing']['lp_source_to_certified'] is None and r['physical_status']=='LP_NOT_CERTIFIED' and 'metrics' not in r

def test_large_ledger_bound_and_tamper_rejected(monkeypatch):
    monkeypatch.setattr(p,'MAX_LEDGER_ENTRIES',1)
    r=p.measure('heterogeneous','highs',0,device={'cuda':False});assert r['physical_status']=='SKIPPED_LEDGER_RESOURCE_BOUND' and 'metrics' not in r
    a=p.fixture('heterogeneous');lp=p.pg.shape_lp(a);q=p.pg._replay(a,lp);q['ledger_steps'][0]['chosen']=999
    import pytest
    with pytest.raises(ValueError):g.audit_steps(a,q['ledger_steps'],q['task_order'],q['shape_order'])

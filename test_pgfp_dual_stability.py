"""Focused CPU tests; no GPU solve or production monkeypatch."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from probes import pgfp_dual_stability as p


def test_fixed_instances_preserve_C_lp_fields():
    cases=p.scenarios();assert len(cases)==3
    original=p.fixtures()['moderate_seed11'];c=cases['C_p4f_128']
    assert p.lp_hash(original)==p.lp_hash(c)
    assert p.fixture_sha(original)!=p.fixture_sha(c)
    for name in ('capacity','demand','value','shapes'):
        np.testing.assert_array_equal(getattr(original,name),getattr(c,name))
    assert p.p4b.fixture_hash(cases['B_p4b_heterogeneous'])==p.p4b.fixture_hash(p.p4b.fixed_cases()[2][1])


def test_analytic_counterexample_certified_and_physical():
    a=p.scenarios()['A_analytic_split']
    l=p.replay(a,p.analytic(a,0)[0]);r=p.replay(a,p.analytic(a,1)[0])
    assert l['task_order']==[1,0] and r['task_order']==[0,1]
    assert l['accepted_count']==2 and r['accepted_count']==1
    assert l['value']==3 and r['value']==1
    assert l['future_score']==0 and r['future_score']==2
    assert p.oracle.validate(a,l['placement']) and p.oracle.validate(a,r['placement'])
    diff=p.divergence(l,r)
    assert diff['accepted_jobs']['mismatch_count']==1
    assert diff['accepted_jobs']['first_difference']==1


def test_actual_production_and_p4b_control_are_exact():
    a=p.scenarios()['A_analytic_split'];result=p.control(a)
    assert result['status']=='EXACT'
    assert all(result['p4b_exact_flags']['exact_flags'].values())


@pytest.mark.parametrize('damage',['prices','lp','pi'])
def test_reject_uncertified_or_unrelated_prices(damage):
    a=p.scenarios()['A_analytic_split'];c=deepcopy(p.analytic(a,0)[0])
    if damage=='prices':c['capacity_duals'][0][0]=2
    elif damage=='lp':c['lp_input_sha256']='0'*64
    else:c['raw_pi']=[0]*5
    with pytest.raises(ValueError):p.replay(a,c)


def test_independent_step_audit_rejects_corruption():
    a=p.scenarios()['A_analytic_split'];r=p.replay(a,p.analytic(a,0)[0]);steps=deepcopy(r['ledger_steps'])
    steps[0]['after'][1][0]+=1
    with pytest.raises(ValueError):p.audit_steps(a,steps,r['task_order'])


def test_registered_digest_not_summary_trust():
    with pytest.raises(ValueError):p.checked_json(b'{}','0'*64)


def test_lp_fingerprint_excludes_only_non_lp_fields():
    a=p.scenarios()['A_analytic_split'];b=replace(a,future=a.future+5e-11)
    assert p.lp_hash(a)==p.lp_hash(b)
    assert p.lp_hash(a)!=p.lp_hash(replace(a,value=a.value+1))

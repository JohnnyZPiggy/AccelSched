"""Track E proof, exact original ledger and adversarial conflict tests."""
from copy import deepcopy
import numpy as np
import pytest
from probes import pgfp_conflict_batch as r


@pytest.mark.parametrize('a',[*r.g.scenarios().values(),r.g.p4b.fixed_cases()[0][1],r.p.edge_case(),*r.special_cases().values()])
def test_full_original_policy_and_logical_ledger(a):
    c=r.g.control(a)['certificate'];actual=r.replay(a,c);reference=r.g.replay(a,c)
    assert r.p.compare(reference,actual)['status']=='EXACT'
    assert all(r.independent(a.eligible,x) for x in actual['batch']['batch_task_ids'])


def test_safe_batch_and_zero_feasible_batch():
    cases=r.special_cases()
    for name in ('safe_disjoint','zero_feasible'):
        a=cases[name];c=r.g.control(a)['certificate'];out=r.replay(a,c)
        assert out['batch']['batch_sizes']==[4] and out['batch']['tasks_in_multi_batches']==4
        assert out['accepted_count']==(4 if name=='safe_disjoint' else 0)
        assert out['batch']['accepted_in_multi_batches']==out['accepted_count']


def test_distinct_destinations_are_not_independence_proof():
    a=r.special_cases()['distinct_destination_score_conflict'];c=r.g.control(a)['certificate'];out=r.replay(a,c)
    assert out['placement']==[0,1]
    assert out['ledger_steps'][0]['attempts'][0]['scores']==[0.,0.]
    assert out['ledger_steps'][1]['attempts'][0]['scores']==[-1.,0.]
    assert out['batch']['batch_sizes']==[1,1] and out['batch']['dependency_rejections']==1
    assert not r.independent(a.eligible,[0,1])


def test_same_destination_competition_falls_back_and_invalid_ledger_rejected():
    a=r.p.edge_case();c=r.g.control(a)['certificate'];out=r.replay(a,c)
    assert out['placement']==[0,-1,-1,-1] and out['batch']['singleton_fallback_tasks']>=1
    steps=deepcopy(out['ledger_steps']);steps[1]['chosen']=0
    with pytest.raises(ValueError):r.g.audit_steps(a,steps,out['task_order'],out['shape_order'])


def test_limit_and_nonbinary_table():
    a=r.special_cases()['safe_disjoint'];groups,stats=r.partition([0,1,2,3],a.eligible,2)
    assert groups==[[0,1],[2,3]] and stats['limit_splits']==1
    with pytest.raises(ValueError):r.partition([0],a.eligible,17)
    table,powers=r.b.score_table(np.array([.1,.2]));assert table[int(np.dot(np.array([0,-1])+1,powers))]==float(np.sum(np.array([0,-1])*a.weights))

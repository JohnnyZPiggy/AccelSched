"""Focused CPU strict/near admission, norm proof and realized selection checks."""
from copy import deepcopy
import numpy as np
import pytest
from probes import pgfp_dual_selection as p


def reference():
    a=p.g.scenarios()['A_analytic_split'];return a,p.g.analytic(a,0)[0]


def test_grid_strict_and_endpoint_discontinuity():
    a,c=reference();out=[]
    for lam in p.GRID:
        payload,_=p.grid_candidate(a,c,lam);admitted=p.strict(a,payload,3.)
        out.append(p.g.replay(a,admitted))
    assert all(r['placement']==[0,1] and r['value']==3 for r in out[:4])
    assert out[-1]['placement']==[1,-1] and out[-1]['value']==1


def test_near_separate_bound_never_strict():
    a,c=reference();near=p.near_candidate(a,c)
    assert near['near_certified'] and not near['strict_optimal']
    assert near['actual_dual_suboptimality']==pytest.approx(.0003)
    assert near['certificate']['column_complementarity']>0
    with pytest.raises(ValueError):p.strict(a,near,3.)
    r=p.replay_near(a,near);assert r['value']==1 and r['future_score']==2
    metric=p.fragmentation(a,r);assert metric['subthreshold_gpu_units']==2


def test_label_blocks_near_even_if_tolerance_would_hide_gap():
    a,c=reference();near=p.near_candidate(a,c);near['face_epsilon']=1e-20
    with pytest.raises(ValueError):p.strict(a,near,3.)


def test_zero_norm_and_anchor_global_proofs():
    a,c=reference();candidate,proof=p.minimum_norm(a,c)
    assert proof['method']=='ZERO_NORM_GLOBAL_PROOF'
    np.testing.assert_array_equal(candidate['certificate']['capacity_duals'],c['capacity_duals'])
    anchor=np.array(c['capacity_duals']);_,proof=p.anchored(a,c,anchor)
    assert proof['squared_distance']==0
    with pytest.raises(ValueError):p.anchored(a,c,anchor+1)


def test_supporting_lp_detects_nonminimum_candidate():
    a,c=reference();bad=p.g.analytic(a,1)[0]
    proof=p.support(a,c,p.vector(bad))
    assert not proof['certified'] and proof['gradient_support_gap']>1


def test_selection_value_future_cost_id_and_no_near():
    def item(name,value,future,cost):
        return dict(id=name,status='PASS',domain='STRICT_OPTIMAL',
                    replay=dict(value=value,future_score=future),incremental_total_seconds=cost)
    r={'a':item('a',3,1,.2),'b':item('b',3,1,.1),'c':item('c',2,9,.001)}
    result=p.select(r,list(r),.3);assert result['selected_id']=='b'
    assert result['fair_total_seconds']>=.601
    r['a']['incremental_total_seconds']=.1
    assert p.select(r,['a','b'],0)['selected_id']=='a'
    r['c']['domain']='NEAR_OPTIMAL'
    with pytest.raises(ValueError):p.select(r,list(r),0)
    r={'native':item('native',3,1,100.),'derived':item('derived',3,1,.02)}
    r['native']['additional_computation_seconds']=.01
    r['derived']['additional_computation_seconds']=.02
    assert p.select(r,list(r),0)['selected_id']=='native'


def test_feasible_nonoptimal_cannot_enter_strict_face():
    a,c=reference();near=p.near_candidate(a,c)
    # Spoofed policy labels do not bypass the original primal/dual gap audit.
    y=np.array([0.,0.,1+p.NEAR_DELTA,0.,0.]);m=p.sparse_model(a)
    with pytest.raises(ValueError):p.from_vector(a,c,y)


@pytest.mark.parametrize("case", ["A_analytic_split", "B_p4b_heterogeneous", "C_p4f_128"])
def test_shared_acquisition_once_for_pool_and_standalone(case, monkeypatch):
    # No LP or replay: exercise run's real argument wiring with nonzero common cost.
    monkeypatch.setattr(p, 'verify_p4g', lambda _: {})
    monkeypatch.setattr(p.g, 'scenarios', lambda: {case: object()})
    certificate = dict(primal_objective=3., capacity_duals=[[0.]])
    monkeypatch.setattr(p, 'native', lambda *args: ({'certificate': certificate}, {}))
    monkeypatch.setattr(p, 'strict', lambda *args: certificate)
    monkeypatch.setattr(p.g, 'control', lambda _: {})
    monkeypatch.setattr(p, 'fixture_sha', lambda _: 'fixture')
    monkeypatch.setattr(p.g, 'lp_hash', lambda _: 'lp')
    monkeypatch.setattr(p, 'near_candidate', lambda *args: {})
    monkeypatch.setattr(p, 'replay_near', lambda *args: {})
    monkeypatch.setattr(p, 'fragmentation', lambda *args: {})
    # A's near rejection is outside this cost-only path.
    def admission(*args):
        if args[1] == {}: raise ValueError('near')
        return certificate
    monkeypatch.setattr(p, 'strict', admission)
    ticks=iter(range(100))
    monkeypatch.setattr(p, 'perf_counter', lambda: float(next(ticks)))
    def measured(a, identifier, factory, optimum, base_result=None, reference_cost=0.):
        assert reference_cost == 1.
        return dict(id=identifier,status='PASS',domain='STRICT_OPTIMAL',
            certificate=certificate,replay=dict(value=3.,future_score=0.),
            incremental_total_seconds=.25,additional_computation_seconds=.25,
            fair_standalone_total_seconds=reference_cost+.25)
    monkeypatch.setattr(p, 'evaluate', measured)
    # Capture select before run reaches that unrelated final summary.
    original=p.select
    class Captured(Exception): pass
    def selection(records,pool,shared):
        result=original(records,pool,shared)
        assert shared == 1.
        assert len(pool) == (5 if case.startswith('A') else 3 if case.startswith('B') else 4)
        assert result['realized_pool_generation_replay_evaluation_seconds'] == len(pool)*.25
        assert result['fair_total_seconds'] == 1.+len(pool)*.25+result['selection_seconds']
        raise Captured
    monkeypatch.setattr(p, 'select', selection)
    with pytest.raises(Captured): p.run(None,None)

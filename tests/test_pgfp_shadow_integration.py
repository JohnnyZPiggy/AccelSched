"""Exact actual-PGFP trace/placement tests; no production module is changed."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest

import accelsched as api
import m0_oracle as oracle
from workload import build_workload
from probes import pgfp_shadow_integration as shadow


@pytest.fixture(scope='module')
def cases():
    # Reuse one real observed baseline per fixture, not synthetic task ordering.
    return [(name,a,shadow.production_reference(a)) for name,a in shadow.fixed_cases()]


@pytest.mark.parametrize('index',[0,1,2])
@pytest.mark.parametrize('backend',['scalar','numpy'])
def test_actual_path_exact_compatibility(cases,index,backend):
    _,a,reference=cases[index];before=deepcopy(a);digest=shadow.fixture_hash(a)
    replay=shadow.shadow(a,backend)
    comparison=shadow.compare(reference,replay)
    assert comparison['status']=='EQUIVALENT' and all(comparison['exact_flags'].values())
    assert oracle.validate(a,replay['placement'])
    # Original-input-order physical ledger, independent of shadow remainder.
    used=np.zeros_like(a.capacity);value=0.;rest=a.capacity.copy()
    for j,i in enumerate(replay['placement']):
        if i>=0:
            assert a.eligible[i,j]
            used[i]+=a.demand[j];rest[i]-=a.demand[j];value+=a.value[j]
    assert np.all(used<=a.capacity+1e-9)
    np.testing.assert_array_equal(rest,replay['residual']);assert value==replay['value']
    greedy_rest=a.capacity.copy()
    for j in replay['task_order']:
        if replay['placement'][j]>=0:greedy_rest[replay['placement'][j]]-=a.demand[j]
    expected_future=sum(float(a.weights[q]) for i in range(len(greedy_rest)) for q in range(len(a.future))
        if a.future_eligible[i,q] and all(greedy_rest[i,k]+1e-10>=a.future[q,k] for k in range(3)))
    assert expected_future==replay['future_score']
    assert digest==shadow.fixture_hash(a)==shadow.fixture_hash(before)
    for field in vars(a):np.testing.assert_array_equal(getattr(a,field),getattr(before,field))


@pytest.mark.parametrize('index',[0,1,2])
def test_optional_torch_cpu_preserves_exact_reduction_and_choices(cases,index):
    pytest.importorskip('torch')
    _,a,reference=cases[index]
    result=shadow.shadow(a,'torch_cpu')
    assert shadow.compare(reference,result)['status']=='EQUIVALENT'
    assert result['timing']['transfer_ms']>=0 and result['timing']['score_compute_ms']>=0


def test_true_tolerance_cliff_not_erased_by_score_atol(cases):
    _,a,reference=cases[0]
    compatible=shadow.shadow(a,'numpy');strict=shadow.shadow(a,'numpy','strict_zero')
    assert reference['placement']==compatible['placement']==[0]
    assert strict['placement']==[1]
    report=shadow.compare(reference,strict)
    assert report['status']=='NOT_EQUIVALENT'
    assert not report['exact_flags']['placement'] and not report['exact_flags']['per_task_trace']
    assert report['differences'][0]['reference_scores']==[0.,0.]
    assert report['differences'][0]['shadow_scores']==[-1.,0.]


def test_empty_shape_group_and_physical_family_mask(cases):
    _,a,reference=cases[1]
    replay=shadow.shadow(a,'numpy')
    assert replay['skipped_empty_shapes']>=1
    assert replay['task_order']==[0,1,2]
    assert reference['placement']==replay['placement']==[3,0,2]
    assert [g['shape'] for g in replay['trace']]==[3,7,7]
    for record in replay['trace']:
        for i in record['candidates']:
            assert a.eligible[i,record['task']] and a.shapes[i]==record['shape']


def test_real_production_call_chain_observation_and_restoration(cases):
    _,a,_=cases[0]
    saved=(api.greedy,oracle.shape_lp,oracle.marginal)
    with patch.object(api,'solve',wraps=api.solve) as actual_solve:
        reference=shadow.production_reference(a)
        actual_solve.assert_called_once_with(a,method='pgfp',seed=11,time_limit=.5)
    assert (api.greedy,oracle.shape_lp,oracle.marginal)==saved
    assert len(reference['trace'])==1 and reference['timing']['lp_ms']>0


def test_original_residual_tolerance_not_silently_clipped():
    a=build_workload([[0,0,0]],[[0,5e-11,0]],value=[1],eligible=[[True]],shapes=[0],
                     future=[[0,0,0]],weights=[1],future_eligible=[[True]])
    reference=shadow.production_reference(a);replay=shadow.shadow(a,'numpy')
    assert shadow.compare(reference,replay)['status']=='EQUIVALENT'
    assert replay['residual'][0][1]==-5e-11 and replay['feasible']


@pytest.mark.parametrize('backend',['scalar','numpy','torch_cpu'])
def test_unsigned_weights_zero_types_and_stable_node_ties(backend):
    if backend=='torch_cpu':pytest.importorskip('torch')
    a=build_workload([[2,2,2],[2,2,2]],[[1,.5,.5],[1,.5,.5]],value=[1,1],
        eligible=[[True,True],[True,True]],shapes=[7,7],future=[[0,0,0],[1,.5,.5]],
        weights=[0,0],future_eligible=[[True,True],[True,True]])
    a.weights=a.weights.astype(np.uint64)
    reference=shadow.production_reference(a);replay=shadow.shadow(a,backend)
    assert reference['placement']==replay['placement']==[0,0]
    assert shadow.compare(reference,replay)['status']=='EQUIVALENT'


def test_no_eligible_shape_no_candidates_no_false_accept(cases):
    _,original,_=cases[1];a=deepcopy(original);a.eligible[:]=False;a.future_eligible[:]=False
    reference=shadow.production_reference(a);replay=shadow.shadow(a,'numpy')
    assert replay['placement']==[-1]*3 and replay['trace']==[]
    assert replay['value']==replay['future_score']==0
    assert shadow.compare(reference,replay)['status']=='EQUIVALENT'


def test_task_and_shape_order_derived_from_actual_LP(cases):
    _,a,reference=cases[2]
    lp=oracle.shape_lp(a)
    utility=a.value[None,:]-lp['prices']@a.demand.T
    assert reference['task_order']==sorted(range(len(a.value)),key=lambda j:(-max(utility[:,j]),j))
    assert lp['shapes']==[7,3,11,5]
    assert shadow.shadow(a,'numpy')['task_order']==reference['task_order']


@pytest.mark.parametrize('corruption',['chosen','scores','placement','future_score','task_order'])
def test_exact_comparison_detects_unhidden_semantic_divergence(cases,corruption):
    reference=cases[0][2];bad=deepcopy(reference)
    if corruption=='chosen':bad['trace'][0]['chosen']=1
    if corruption=='scores':bad['trace'][0]['scores'][0]=1e-15
    if corruption=='placement':bad['placement']=[1]
    if corruption=='future_score':bad['future_score']+=1e-15
    if corruption=='task_order':bad['task_order']=[99]
    assert shadow.compare(reference,bad)['status']=='NOT_EQUIVALENT'


def test_timing_is_finite_and_overlaps_are_explicit(cases):
    reference=cases[1][2];result=shadow.shadow(cases[1][1],'numpy')
    assert reference['timing']['whole_solve_ms']>=reference['timing']['lp_ms']
    for k,v in result['timing'].items():assert np.isfinite(v) and v>=0,k
    assert result['timing']['whole_solve_ms']>=result['timing']['lp_ms']+result['timing']['replay_ms']
    assert result['timing']['replay_ms']>=result['timing']['candidate_filter_ms']
    json.dumps(result,allow_nan=False)


def test_invalid_backend_mode_and_float32_contract(cases):
    _,a,_=cases[0]
    with pytest.raises(ValueError):shadow.shadow(a,'bad')
    with pytest.raises(ValueError):shadow.shadow(a,'numpy','bad')
    b=deepcopy(a);b.capacity=b.capacity.astype(np.float32)
    with pytest.raises(ValueError):shadow.shadow(b,'numpy')

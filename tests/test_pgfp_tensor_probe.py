"""Tiny independent global-F differences, strict physical masks and float64 parity."""
from copy import deepcopy
from itertools import product
import json

import numpy as np
import pytest

from probes import pgfp_tensor_probe as probe


def tiny():
    return (np.array([[2.,2.,2.],[1.,2.,2.]]),
            np.array([[1.,1.,1.],[2.,2.,2.],[0.,0.,0.],[3.,0.,0.]]),
            np.array([[1.,1.,1.],[2.,2.,2.],[1.,0.,0.]]),
            np.array([2.,3.,5.]), np.array([[True]*3,[True,False,False]]),
            np.ones((2,4),bool))


def full_F_difference(inputs):
    """Re-evaluate all physical nodes for each trial, not the local loss formula."""
    r,d,q,w,e,a=inputs
    def total(cap):
        return sum(float(w[h]) for i in range(len(cap)) for h in range(len(q))
                   if e[i,h] and all(cap[i,k]>=q[h,k] for k in range(3)))
    before=total(r);scores=np.zeros((len(r),len(d)));mask=np.zeros(scores.shape,bool)
    for i in range(len(r)):
        for j in range(len(d)):
            if a[i,j] and all(r[i,k]>=d[j,k] for k in range(3)):
                mask[i,j]=True;changed=r.copy();changed[i]-=d[j]
                scores[i,j]=total(changed)-before
    return scores,mask


@pytest.mark.parametrize('chunk',[1,2,32])
def test_hand_marginals_masks_and_stable_choice(chunk):
    inputs=tiny();before=deepcopy(inputs)
    oracle,mask=probe.scalar_oracle(*inputs)
    expected=[[ -3.,-10.,0.,None],[-2.,None,0.,None]]
    assert probe.json_scores(oracle,mask)==expected
    scores,valid=probe.numpy_marginal(*inputs,chunk=chunk)
    np.testing.assert_array_equal(scores,oracle);np.testing.assert_array_equal(valid,mask)
    global_scores,global_mask=full_F_difference(inputs)
    np.testing.assert_array_equal(scores,global_scores);np.testing.assert_array_equal(valid,global_mask)
    assert probe.choose_nodes(inputs,scores,valid)==[1,0,0,None]
    assert np.all(scores[valid]<=0)
    for a,b in zip(inputs,before):np.testing.assert_array_equal(a,b)
    json.dumps(probe.json_scores(scores,valid),allow_nan=False)


def test_exhaustive_tiny_binary_resource_grid():
    resources=np.array(list(product([0.,1.],repeat=3)))
    inputs=(resources,resources,resources,np.arange(8,dtype=float)/4,
            np.ones((8,8),bool),np.ones((8,8),bool))
    expected,mask=full_F_difference(inputs)
    for backend in (probe.scalar_oracle,probe.numpy_marginal):
        scores,valid=backend(*inputs)
        np.testing.assert_array_equal(scores,expected);np.testing.assert_array_equal(valid,mask)
        choices=probe.choose_nodes(inputs,scores,valid)
        for j,i in enumerate(choices):
            assert i is None or all(inputs[0][i,k]>=inputs[1][j,k] for k in range(3))


@pytest.mark.parametrize('case',['empty_future','zero_weights','future_ineligible','task_ineligible','no_fit'])
def test_empty_zero_and_authoritative_physical_eligibility(case):
    data=list(tiny())
    if case=='empty_future':data[2]=np.empty((0,3));data[3]=np.empty(0);data[4]=np.empty((2,0),bool)
    if case=='zero_weights':data[3]=np.zeros(3)
    if case=='future_ineligible':data[4]=np.zeros((2,3),bool)
    if case=='task_ineligible':data[5]=np.zeros((2,4),bool)
    if case=='no_fit':data[0]=np.zeros((2,3));data[1]=np.ones((4,3))
    for backend in (probe.scalar_oracle,probe.numpy_marginal):
        scores,mask=backend(*data);assert np.all(scores==0)
        if case in ('task_ineligible','no_fit'):
            assert not mask.any() and probe.choose_nodes(data,scores,mask)==[None]*4


def test_cpu_ram_thresholds_no_implicit_resource_slack():
    data=list(tiny());data[0]=np.array([[2.,1.,1.],[2.,1.,1.]])
    data[1]=np.array([[1.,1.,1.],[1.,np.nextafter(1.,2.),1.],[1.,1.,np.nextafter(1.,2.)],[0.,0.,0.]])
    data[2]=np.array([[1.,0.,0.],[1.,np.nextafter(0.,1.),0.],[1.,0.,np.nextafter(0.,1.)]])
    for backend in (probe.scalar_oracle,probe.numpy_marginal):
        scores,mask=backend(*data)
        np.testing.assert_array_equal(mask,[[True,False,False,True]]*2)
        expected,valid=full_F_difference(data)
        np.testing.assert_array_equal(scores,expected);np.testing.assert_array_equal(mask,valid)


def test_family_mismatch_overrules_identical_capacity():
    data=list(tiny());data[0][1]=data[0][0];data[5][:]=False;data[5][1,0]=True
    scores,mask=probe.numpy_marginal(*data)
    assert probe.choose_nodes(data,scores,mask)==[1,None,None,None]


@pytest.mark.parametrize('case',['nan','inf','negative','float32','complex','bool_resource','bad_shape','bad_weights','negative_weight','overflow_weight','int_over_2pow53','mixed_int_over_2pow53','gpu_fraction','bad_E','bad_A'])
def test_invalid_input_contract(case):
    data=list(tiny())
    if case=='nan':data[0][0,1]=np.nan
    if case=='inf':data[2][0,1]=np.inf
    if case=='negative':data[1][0,2]=-1
    if case=='float32':data[0]=data[0].astype(np.float32)
    if case=='complex':data[0]=data[0].astype(complex)
    if case=='bool_resource':data[0]=data[0].astype(bool)
    if case=='bad_shape':data[0]=np.ones((2,2))
    if case=='bad_weights':data[3]=np.ones((3,1))
    if case=='negative_weight':data[3][0]=-1
    if case=='overflow_weight':data[3]=np.array([1e308]*3)
    if case=='int_over_2pow53':data[0]=np.array([[2**53+1,2,2],[1,2,2]])
    if case=='mixed_int_over_2pow53':data[0]=[[2**53+1,2.,2.],[1,2,2]]
    if case=='gpu_fraction':data[0][0,0]=.5
    if case=='bad_E':data[4]=data[4].astype(int)
    if case=='bad_A':data[5]=np.ones((2,3),bool)
    for backend in (probe.scalar_oracle,probe.numpy_marginal):
        with pytest.raises(ValueError):backend(*data)


def test_mask_and_positive_delta_tampering_rejected():
    inputs=tiny();scores,valid=probe.numpy_marginal(*inputs)
    bad=scores.copy();bad[0,0]=1
    with pytest.raises(ValueError):probe.choose_nodes(inputs,bad,valid)
    mask=valid.copy();mask[0,3]=True
    with pytest.raises(ValueError):probe.choose_nodes(inputs,scores,mask)


def test_determinism_fixed_provenance_and_zero_dimensions():
    first,second=probe.fixed_inputs(),probe.fixed_inputs()
    assert probe.fixture_sha256(first)==probe.fixture_sha256(second)
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
    inputs=(np.empty((0,3)),np.empty((0,3)),np.empty((0,3)),np.empty(0),np.empty((0,0),bool),np.empty((0,0),bool))
    for backend in (probe.scalar_oracle,probe.numpy_marginal):
        scores,valid=backend(*inputs);assert scores.shape==(0,0)
        assert probe.choose_nodes(inputs,scores,valid)==[]


def test_optional_torch_cpu_float64_parity_and_no_input_mutation():
    pytest.importorskip('torch')
    inputs=tiny();before=deepcopy(inputs)
    scores,valid=probe.torch_marginal(*inputs,device='cpu',chunk=1)
    oracle,mask=full_F_difference(inputs)
    np.testing.assert_array_equal(scores,oracle);np.testing.assert_array_equal(valid,mask)
    assert scores.dtype==np.float64 and probe.choose_nodes(inputs,scores,valid)==[1,0,0,None]
    for a,b in zip(inputs,before):np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('chunk',[0,-1,True])
def test_invalid_chunk_bound(chunk):
    with pytest.raises(ValueError):probe.numpy_marginal(*tiny(),chunk=chunk)

"""Deterministic mapping tests; injected states are not real solver evidence."""
from types import SimpleNamespace
import numpy as np
import pytest
from accelsched import solve
from m0_oracle import packing_case


def inject(monkeypatch, status, x, **metadata):
    monkeypatch.setattr('scipy_backend.milp', lambda *args, **kwargs:
                        SimpleNamespace(status=status, x=x, message='injected', **metadata))


@pytest.mark.parametrize('status', [2, 3, 4, 99])
@pytest.mark.parametrize('x', [None, np.array([1., 0., 0., 1.])])
def test_solver_failure_is_not_a_time_limit(monkeypatch, status, x):
    inject(monkeypatch, status, x)
    out = solve(packing_case(), 'highs_mip')
    assert out.status == 'failed' and out.placement is None and out.value is None
    assert not out.optimal and out.error


@pytest.mark.parametrize('x', [np.full(4, np.nan), np.full(4, np.inf),
    np.array([1.,0.]), np.array([[1.,0.],[0.,1.]]),
    np.array([2.,0.,0.,0.]), np.array([-1.,0.,0.,0.]),
    np.array([1.,0.,1.,0.]), np.array([1.+1j,0.,0.,1.])])
def test_malformed_incumbent_cannot_certify_a_projection(monkeypatch, x):
    inject(monkeypatch, 0, x)
    out = solve(packing_case(), 'highs_mip')
    assert out.status == 'invalid' and not out.optimal
    assert out.placement is None and out.value is None


def test_optimal_without_incumbent_is_invalid(monkeypatch):
    inject(monkeypatch, 0, None)
    assert solve(packing_case(), 'highs_mip').status == 'invalid'


@pytest.mark.parametrize('status,expected', [(0,'optimal'),(1,'feasible_uncertified')])
def test_near_integer_projection_is_validated(monkeypatch, status, expected):
    inject(monkeypatch,status,np.array([1.-5e-7,0.,0.,1.-5e-7]))
    out=solve(packing_case(),'highs_mip')
    assert out.status==expected and out.value==20 and out.placement==(0,1)


def test_nonintegral_and_time_limit_without_incumbent(monkeypatch):
    inject(monkeypatch,1,np.array([.5,0.,0.,1.]))
    assert solve(packing_case(),'highs_mip').status=='uncertified'
    inject(monkeypatch,1,None)
    out=solve(packing_case(),'highs_mip')
    assert out.status=='uncertified' and out.value is None and not out.optimal


def test_solver_exception_is_explicit_failure(monkeypatch):
    def fail(*args,**kwargs):
        raise RuntimeError('injected numerical failure')
    monkeypatch.setattr('scipy_backend.milp',fail)
    out=solve(packing_case(),'highs_mip')
    assert out.status=='failed' and out.error=='injected numerical failure'


@pytest.mark.parametrize('field', ['mip_dual_bound','mip_gap'])
@pytest.mark.parametrize('value', [None,np.nan,np.inf,-np.inf])
def test_optional_solver_metadata_is_not_exposed_as_a_certificate(monkeypatch,field,value):
    inject(monkeypatch,1,np.array([1.,0.,0.,1.]),**{field:value})
    out=solve(packing_case(),'highs_mip')
    assert out.status=='feasible_uncertified' and not out.optimal
    assert not hasattr(out,'mip_gap') and not hasattr(out,'mip_dual_bound')


def test_rounded_placement_capacity_is_checked(monkeypatch):
    from dataclasses import replace
    a=packing_case()
    a=replace(a,capacity=np.array([[8.-5e-8],[8.]]),demand=np.array([[8.],[8.]]))
    inject(monkeypatch,0,np.array([1.-5e-7,0.,0.,1.]))
    out=solve(a,'highs_mip')
    assert out.status=='invalid' and not out.optimal

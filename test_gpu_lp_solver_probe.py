"""P7A original ranged-model KKT, signs, rays and immutable P5 input tests."""
import numpy as np
from probes import gpu_lp_solver_probe as p


def test_mixed_lower_upper_equalities_and_objective_constant():
    m=p.smoke_models()['hand_mixed'];x=np.array([1.,2.]);y=np.array([1.,-1.,0.]);z=np.zeros(2)
    assert p.cpu_small(m)['objective']==5.
    assert p.kkt(m,x,y,z,True,5.)['certified']
    assert not p.kkt(m,x,-y,z,True,5.)['certified']
    assert not p.kkt(m,x,y,z,False,5.)['certified']


def test_bound_signs_fixed_and_free_columns():
    m=p.model([1,-2],np.empty((0,2)),[],[],[-3,2],[1,4]);x=np.array([-3.,4.]);z=np.array([1.,-2.])
    assert p.kkt(m,x,np.empty(0),z,True,-11.)['certified']
    assert not p.kkt(m,x,np.empty(0),-z,True,-11.)['certified']
    free=p.model([0],[[0]],[-np.inf],[np.inf],[-np.inf],[np.inf])
    assert not p.kkt(free,np.array([0.]),np.array([1.]),np.array([0.]),True,0.)['certified']


def test_farkas_ray_and_no_zero_or_missing_certificate():
    m=p.smoke_models()['infeasible'];assert p.cpu_small(m)['status']==2
    assert p.farkas(m,np.array([1.,-1.]),np.array([0.]))['certified']
    assert not p.farkas(m,np.array([-1.,1.]),np.array([0.]))['certified']
    assert not p.farkas(m,None,None)['certified']
    assert not p.farkas(m,np.zeros(2),np.zeros(1))['certified']


def test_unbounded_ray_requires_feasible_witness():
    m=p.smoke_models()['unbounded'];assert p.cpu_small(m)['status']==3
    assert p.unbounded(m,np.array([2.]),np.array([0.]))['certified']
    assert not p.unbounded(m,np.array([-1.]),np.array([0.]))['certified']
    assert not p.unbounded(m,np.array([1.]),np.array([-1.]))['certified']


def test_missing_vectors_and_original_absolute_threshold():
    m=p.smoke_models()['hand_mixed'];y=np.array([1.,-1.,0.]);z=np.zeros(2)
    assert not p.kkt(m,None,y,z,True,5.)['certified']
    assert not p.kkt(m,np.array([1.,2.+2e-7]),y,z,True,5.)['certified']
    assert not p.kkt(m,np.array([1.,2.]),y,z,True,999.)['certified']


def test_p5_original_fixture_csr_hash_pins():
    for n,nodes in ((1000,64),(10000,256)):
        m,old,meta=p.shape(n,nodes)
        assert (meta['fixture_sha256'],meta['lp_sha256'])==p.PINS[n]
        assert m['A'].nnz==32*n and m['A'].shape==(n+24,8*n)
        assert np.all(m['lb']==0.) and np.all(np.isneginf(m['l']))

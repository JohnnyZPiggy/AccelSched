"""P5C small CPU-only equivalence and immutable acceptance checks."""
import numpy as np
from scipy.optimize import linprog
from probes import shape_lp_gpu_convergence as p
from probes.shape_lp_backend_probe import fixtures


def test_row_coefficients_scale_not_rhs_and_inverse_certificate():
    original=p.base.assemble(fixtures()['hand_unique']);scaled,row=p.scale_rows(original)
    np.testing.assert_array_equal(row,[1.,.5,1.,1.])
    s=linprog(scaled['c'],A_ub=scaled['A'],b_ub=scaled['b'],bounds=(0,None),method='highs')
    x,pi,rc=p.inverse(s.x,s.ineqlin.marginals,s.lower.marginals,row)
    assert p.base.certify(original,x,pi,rc,True,s.fun)['certified']
    np.testing.assert_array_equal(original['c'],scaled['c'])
    np.testing.assert_allclose(original['A']@x,scaled['A']@x/row,rtol=0,atol=1e-14)


def test_empty_resource_rows_and_sparse_identity():
    original=p.base.assemble(fixtures()['hand_degenerate']);scaled,row=p.scale_rows(original)
    assert (scaled['A'].shape,scaled['A'].nnz)==(original['A'].shape,original['A'].nnz)
    np.testing.assert_array_equal(scaled['b']/row,original['b'])
    a=p.base.fixture(31,16,4);model=p.base.assemble(a);scaled,row=p.scale_rows(model)
    x=np.arange(len(model['c']))/len(model['c'])
    np.testing.assert_allclose(scaled['A']@x,row*(model['A']@x),rtol=0,atol=1e-13)


def test_same_strict_threshold_and_status_after_inverse():
    model=p.base.assemble(fixtures()['hand_unique'])
    x=np.array([.5+1.2506e-7/2]);pi=np.array([0.,-3.,0.,0.]);rc=np.array([0.])
    assert not p.base.certify(model,x,pi,rc,True,-6*x[0])['certified']
    assert not p.base.certify(model,np.array([.5]),pi,rc,False,-3.)['certified']
    assert p.base.TOL==1e-7


def test_fixed_target_and_alternatives_bound():
    assert len(p.ALTERNATIVES)==2
    d=p.base.dimensions(1000000,4096)
    assert (d['variables'],d['rows'],d['nnz'])==(8000000,1000024,32000000)
    assert p.REGISTERED_LP=='eb3880a01cb4c3d7029e03839f3ae973eae29401c09e216a42b4952bc9ed9bc6'

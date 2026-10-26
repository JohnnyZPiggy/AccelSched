"""P5B bounded CPU inverse-map and immutable original-audit regression."""
import numpy as np
from scipy.optimize import linprog
from probes import shape_lp_gpu_numerics as p
from probes.shape_lp_backend_probe import fixtures


def test_binary_equilibration_and_inverse_certificate():
    a=p.base.fixture(31,16,4);original=p.base.assemble(a)
    scaled,row,column=p.equilibrate(original)
    x=np.arange(len(original['c']))/len(original['c'])
    np.testing.assert_allclose(scaled['A']@(x/column),row*(original['A']@x),rtol=0,atol=1e-13)
    np.testing.assert_array_equal(scaled['c']/column,original['c'])
    s=linprog(scaled['c'],A_ub=scaled['A'],b_ub=scaled['b'],bounds=(0,None),method='highs')
    x,pi,rc=p.inverse(s.x,s.ineqlin.marginals,s.lower.marginals,row,column)
    assert p.base.certify(original,x,pi,rc,True,s.fun)['certified']
    np.testing.assert_allclose(original['c']+original['A'].T@(-pi),rc,atol=1e-12)
    assert original['A'].nnz==scaled['A'].nnz


def test_original_threshold_not_scaled_solver_tolerance():
    original=p.base.assemble(fixtures()['hand_unique'])
    # Exact starting optimum0.5; violation1.2506e-7 must stay rejected.
    x=np.array([.5+1.2506e-7/2]);pi=np.array([0.,-3.,0.,0.]);rc=np.array([0.])
    c=p.base.certify(original,x,pi,rc,True,-6*x[0])
    assert not c['certified'] and c['primal_residual']>1e-7
    assert c['absolute_tolerance']==1e-7
    d=p.diagnostics(original,x,pi)
    assert d['worst_rows'][0]['kind']=='shape_resource'
    assert d['worst_rows'][0]['resource']==0 and d['rows_above_fixed_threshold']==1


def test_mapped_dual_and_rc_signs():
    model=p.base.assemble(fixtures()['hand_unique']);scaled,row,col=p.equilibrate(model)
    s=linprog(scaled['c'],A_ub=scaled['A'],b_ub=scaled['b'],bounds=(0,None),method='highs')
    x,pi,rc=p.inverse(s.x,s.ineqlin.marginals,s.lower.marginals,row,col)
    assert p.base.certify(model,x,pi,rc,True,s.fun)['certified']
    assert not p.base.certify(model,x,-pi,rc,True,s.fun)['certified']
    assert not p.base.certify(model,x,pi,rc+1,True,s.fun)['certified']
    np.testing.assert_array_equal(scaled['b']/row,model['b'])


def test_fixed_ladder_and_bounded_repair_declaration():
    assert len(p.REPAIRS)==2 and p.base.TOL==1e-7
    d=p.base.dimensions(1000000,4096)
    assert (d['variables'],d['rows'],d['nnz'])==(8000000,1000024,32000000)
    assert p.REPAIRS[0]['tolerance']==p.REPAIRS[1]['tolerance']==1e-12

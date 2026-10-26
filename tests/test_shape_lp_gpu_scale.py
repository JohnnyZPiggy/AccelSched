"""Bounded CPU model equivalence and certificate tamper checks."""
import numpy as np
import pytest
from scipy.optimize import linprog
from probes import shape_lp_gpu_scale as p
from probes.shape_lp_backend_probe import sparse_model, fixtures


def test_direct_csr_exact_p4c_and_reproducible_fixture():
    a=p.fixture(31,16,4);b=p.fixture(31,16,4)
    assert p.digest(a.demand,a.value)==p.digest(b.demand,b.value)
    assert a.eligible.all() and a.eligible.strides==(0,0) and not a.eligible.flags.writeable
    m=p.assemble(a);old=sparse_model(a)
    assert (m['A']!=old['A']).nnz==0
    np.testing.assert_array_equal(m['b'],old['b']);np.testing.assert_array_equal(m['c'],old['c'])
    assert m['A'].nnz==p.dimensions(31,16,4)['nnz']
    assert m['A'].data.nbytes+m['A'].indices.nbytes+m['A'].indptr.nbytes==p.dimensions(31,16,4)['estimated_csr_bytes']


def test_zero_demands_keep_p4c_sparse_meaning():
    a=fixtures()['hand_degenerate'];m=p.assemble(a);old=sparse_model(a)
    assert (m['A']!=old['A']).nnz==0


def test_original_model_certificate_rejects_sign_rc_and_nonoptimal():
    m=p.assemble(fixtures()['hand_unique'])
    s=linprog(m['c'],A_ub=m['A'],b_ub=m['b'],bounds=(0,None),method='highs')
    args=(m,s.x,s.ineqlin.marginals,s.lower.marginals)
    assert p.certify(*args,True,s.fun)['certified']
    assert not p.certify(m,s.x,-s.ineqlin.marginals,s.lower.marginals,True,s.fun)['certified']
    assert not p.certify(m,s.x,s.ineqlin.marginals,s.lower.marginals+1,True,s.fun)['certified']
    assert not p.certify(*args,False,s.fun)['certified']
    with pytest.raises(ValueError):p.certify(m,s.x[:-1],s.ineqlin.marginals,s.lower.marginals,True,s.fun)


def test_tier_counts_and_memory_guard_transparency():
    assert [n for n,_ in p.TIERS]==[1000,10000,100000,1000000]
    d=p.dimensions(1000000,4096)
    assert d['variables']==8000000 and d['rows']==1000024 and d['nnz']==32000000
    assert d['conservative_working_bytes']>d['estimated_csr_bytes']

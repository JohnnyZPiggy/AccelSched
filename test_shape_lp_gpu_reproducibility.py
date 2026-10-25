"""P5D CPU-only independent arithmetic, strict gate and vector provenance."""
import numpy as np
from scipy.optimize import linprog
from probes import shape_lp_gpu_reproducibility as p
from probes.shape_lp_backend_probe import fixtures


def test_matrix_free_matches_sparse_original_certificate():
    a=fixtures()['hand_unique'];m=p.base.assemble(a)
    s=linprog(m['c'],A_ub=m['A'],b_ub=m['b'],bounds=(0,None),method='highs')
    raw=dict(Status=0,ObjVal=s.fun,DualObj=s.fun,Gap=0.)
    c=p.matrix_free(a,s.x,s.ineqlin.marginals,s.lower.marginals,raw)
    sparse=p.base.certify(m,s.x,s.ineqlin.marginals,s.lower.marginals,True,s.fun,s.fun,0.)
    assert c['certified'] and sparse['certified']
    assert c['primal_objective']==sparse['primal_objective']==3.
    assert not p.failure_reasons(c)


def test_raw_sign_rc_status_and_threshold_failures():
    a=fixtures()['hand_unique'];x=np.array([.5+1.2506e-7/2]);pi=np.array([0.,-3.,0.,0.]);rc=np.array([0.])
    raw=dict(Status=0,ObjVal=-6*x[0],DualObj=-3.,Gap=abs(6*x[0]-3))
    c=p.matrix_free(a,x,pi,rc,raw)
    assert not c['certified'] and 'primal_residual' in p.failure_reasons(c)
    assert not p.matrix_free(a,np.array([.5]),-pi,rc,raw)['certified']
    assert not p.matrix_free(a,np.array([.5]),pi,rc+1,raw)['certified']
    raw['Status']=3
    assert 'solver_not_optimal' in p.failure_reasons(p.matrix_free(a,np.array([.5]),pi,rc,raw))


def test_disagreement_never_silently_accepted():
    yes=dict(certified=True);no=dict(certified=False)
    assert p.verdict(yes,yes,no)=='NUMERIC_DISAGREEMENT'
    assert p.verdict(no,yes,yes)=='NUMERIC_DISAGREEMENT'
    assert p.verdict(no,no,no)=='PASS_FAILED_SOLUTION_REVALIDATION'


def test_identical_config_and_lossless_array_hashes(tmp_path):
    assert p.MAX_RUNS==4 and p.CONFIG==dict(name='identical_60s_1e12',tolerance=1e-12,row_scale=False)
    arrays=(np.array([.125,0.],dtype=np.float64),np.array([-3.]),np.array([0.,2.]))
    path=tmp_path/'vectors.npz';np.savez_compressed(path,x=arrays[0],pi=arrays[1],rc=arrays[2])
    with np.load(path,allow_pickle=False) as z:copied=tuple(z[k] for k in ('x','pi','rc'))
    assert p.array_records(arrays)==p.array_records(copied)
    copied[0][0]=.25
    assert p.array_records(arrays)!=p.array_records(copied)

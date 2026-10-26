from dataclasses import replace
import numpy as np
import pytest
import bench_v4_gpu as b
from accelsched import solve
from m0_oracle import validate


@pytest.mark.parametrize('case',b.CASES)
def test_exact_streaming_first_fit_and_all_job_oracles(case,tmp_path):
    c=b.Config(200,16,case,11,64)
    row=b.pipeline(c,tmp_path,'test')
    with np.load(tmp_path/'test.npz',allow_pickle=False) as z:
        p=z['placement'];a=b.original_instance(c)
        assert validate(a,p) and tuple(p)==solve(a,'first_fit').placement
        assert row['physical']['all_jobs_checked']==c.n
        assert row['physical']['dense_reference']=='PASS_ALL_JOBS'
        assert sum(row['stages'].values())<=row['full_path_seconds']+1e-8
        assert row['physical']['input_sha256']==row['input_sha256']
        assert row['peak_rss_bytes'] is None or row['peak_rss_bytes']>0


def test_adversarial_oracle_family_capacity_residual_and_hash(tmp_path):
    c=b.Config(200,16,'balanced',29,64);row=b.pipeline(c,tmp_path,'a')
    with np.load(tmp_path/'a.npz',allow_pickle=False) as z:p=z['placement'].copy();rest=z['residual'].copy()
    for bad in [p[:-1],np.full(c.n,c.nodes,np.int32),np.full(c.n,0,np.int32)]:
        with pytest.raises(AssertionError):b.audit(c,bad,rest)
    with pytest.raises(AssertionError):b.audit(c,p,rest+1)
    with pytest.raises(AssertionError):b.audit(c,p,rest,expected_input='bad')
    with pytest.raises(AssertionError):b.audit(replace(c,seed=47),p,rest,row['input_sha256'])
    bad=rest.astype(float);bad[0,0]=np.nan
    with pytest.raises(AssertionError):b.audit(c,p,bad)


def test_multiresource_predicate_and_determinism(tmp_path):
    c=b.Config(500,16,'multiresource',11,64);x=b.materialize(c)
    assert np.array_equal(x,b.materialize(c))
    assert not np.array_equal(x,b.materialize(replace(c,seed=29)))
    row=b.pipeline(c,tmp_path,'a')
    assert row['physical']['individually_impossible']>0
    with np.load(tmp_path/'a.npz',allow_pickle=False) as z:p=z['placement'].copy();rest=z['residual']
    j=np.flatnonzero(x[:,5]==16)[0];p[j]=0
    with pytest.raises(AssertionError):b.audit(c,p,rest)


def test_integrated_highs_original_model_and_physical_audit(tmp_path):
    c=b.Config(100,16,'balanced',11,64)
    row=b.pipeline(c,tmp_path,'lp',backend='highs')
    assert row['status']=='PHYSICAL_E2E_CERTIFIED' and row['lp']['certified']
    assert row['lp']['independent']['certified']
    assert row['policy']=='EXPERIMENTAL_LP_SCORE_FIRST_FIT_POLICY_DIFFERENT'
    with np.load(tmp_path/'lp.npz',allow_pickle=False) as z:assert all(k in z for k in ['x','pi','rc','placement','residual'])


def test_chunk_configuration_is_part_of_input_identity():
    c=b.Config(200,16,'balanced',11,64)
    assert not np.array_equal(b.materialize(c),b.materialize(replace(c,chunk=128)))
    with pytest.raises(ValueError):b.inventory(replace(c,case='invalid'))

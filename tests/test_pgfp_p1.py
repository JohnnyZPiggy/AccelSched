import numpy as np
import pytest
import bench_ultra_pgfp as b
from probes import pgfp_p1 as p
from probes.pgfp_p1_fast import CandidateIndex


@pytest.fixture(autouse=True)
def candidate_index(monkeypatch):
    monkeypatch.setattr(b,'ExactIndex',CandidateIndex)


@pytest.mark.parametrize('policy',b.POLICIES)
def test_observation_preserves_full_replay(monkeypatch,policy):
    c=b.base.Config(96,8,'multiresource',11,64)
    cap,nf,ng,shapes=b.base.inventory(c);data=b.base.materialize(c);prices=np.zeros((4,3))
    expected=b.replay(cap,data,shapes,nf,ng,prices,policy,trace=True)
    original_q=b.CandidateSet;original_ix=b.ExactIndex
    observed=p.observe(b)
    try:
        actual=b.replay(cap,data,shapes,nf,ng,prices,policy,trace=True)
        assert np.array_equal(actual[0],expected[0]) and np.array_equal(actual[1],expected[1])
        assert actual[2]['ledger']==expected[2]['ledger']
        assert actual[2]['candidate_counters']==expected[2]['candidate_counters']
        assert observed['registrations']==actual[2]['registered_query_count']
        assert observed['cache_hits']+observed['registrations']==actual[2]['candidate_counters']['index_queries']
        assert observed['max_affected_watchers_per_commit']>0
    finally:b.CandidateSet=original_q;b.ExactIndex=original_ix


def test_json_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):p.write(tmp_path/'bad.json',{'x':float('nan')})


@pytest.mark.parametrize('case',b.CASES)
@pytest.mark.parametrize('policy',b.POLICIES)
def test_candidate_reuses_complete_original_policy_oracle(case,policy,monkeypatch):
    import test_bench_ultra_pgfp as original
    original.test_original_prices_exact_orders_candidates_marginals_placements(case,policy,monkeypatch)


def test_candidate_reuses_random_full_rescans():
    import test_bench_ultra_pgfp as original
    original.test_cached_exact_index_matches_full_rescan_after_many_updates()


def test_cached_lists_include_late_query_registration_and_score_improvement():
    future=np.array([[4,0,0]]);weights=np.array([3]);fe=np.ones((2,1),bool)
    ix=CandidateIndex(np.array([[5,8,32],[5,8,32]]),[0,0],[0,0],[0,1],'pgfp',future,weights,fe)
    assert ix.query([3,0,0],0,-1,-1)==0
    ix.commit(0,[2,0,0])
    assert ix.query([3,0,0],0,-1,-1)==0  # Cached cost improved from3 to0.
    assert ix.query([1,0,0],0,-1,-1)==0  # This query did not exist at cache construction.
    ix.commit(0,[1,0,0]);ix.commit(0,[2,0,0])
    assert ix.query([1,0,0],0,-1,-1)==1  # Late query must receive both subsequent updates.


@pytest.mark.parametrize('policy',b.POLICIES)
def test_adversarial_ties_duplicate_futures_eligibility_and_disappearance(monkeypatch,policy):
    import m0_oracle as old
    cap=np.array([[4,8,16],[4,8,16],[0,8,16],[4,8,16]])
    nf=np.array([0,0,-1,1]);ng=np.array([0,1,0,0]);shapes=np.array([0,0,1,0])
    data=np.array([[2,0,0,10,0,-1],[3,0,0,10,0,0],[0,4,8,10,-1,-1],[1,0,0,10,1,-1],[4,1,1,5,0,1]])
    future=np.array([[4,0,0],[4,0,0],[0,4,8]]);weights=np.array([2,3,1])
    fe=np.column_stack([nf==0,nf==0,np.ones(4,bool)])
    eligible=b.base.allowed(data[:,4][None,:],data[:,5][None,:],nf[:,None],ng[:,None])
    a=old.Instance(cap.astype(float),data[:,:3].astype(float),data[:,3].astype(float),eligible,shapes,future,weights,fe)
    prices=np.zeros((2,3));monkeypatch.setattr(old,'shape_lp',lambda _:dict(prices=prices,shapes=[0,1]))
    actual,res,info=b.replay(cap,data,shapes,nf,ng,prices,policy,trace=True,future=future,weights=weights,fe=fe)
    expected=old.greedy(a,'pgfp' if policy=='pgfp' else 'best_fit',11)
    assert tuple(actual)==tuple(expected[0]) and old.validate(a,actual)
    assert np.array_equal(res,old.residual(a,actual))
    assert [s['job'] for s in info['ledger']]==[0,1,2,3,4]
    for step in info['ledger']:
        j=step['job'];before=np.array(step['before']);assert step['shape_order']==[0,1]
        chosen=-1
        for h in [0,1]:
            ids=[i for i in range(4) if shapes[i]==h and eligible[i,j] and np.all(before[i]>=data[j,:3])]
            if ids:
                chosen=max(ids,key=lambda i:(old.marginal(a,before,i,j),-i)) if policy=='pgfp' else min(ids,key=lambda i:(sum(before[i]-data[j,:3]),i))
                break
        assert chosen==step['chosen']
    assert int(np.sum((np.all(res[:,None,:]>=future,axis=2)&fe)*weights))==old.future_score(a,res)

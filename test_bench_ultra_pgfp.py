from dataclasses import replace
from copy import deepcopy
import numpy as np
import pytest
import m0_oracle as old
import accelsched
import bench_ultra_pgfp as b


def instance(c):
    a=b.base.original_instance(c);_,nf,_,_=b.base.inventory(c)
    a.future=b.FUTURE.astype(float);a.weights=b.WEIGHTS.astype(float);a.future_eligible=b.future_eligibility(nf)
    return a


@pytest.mark.parametrize('case',b.CASES)
@pytest.mark.parametrize('policy',b.POLICIES)
def test_original_prices_exact_orders_candidates_marginals_placements(case,policy,monkeypatch):
    c=b.base.Config(96,8,case,11,64);a=instance(c);lp=old.shape_lp(a)
    data=b.base.materialize(c);cap,nf,ng,kind=b.base.inventory(c)
    labels=list(dict.fromkeys(a.shapes.tolist()));assert labels==lp['shapes']
    # A real native original LP is fixed for both replays; no mocked GPU success.
    monkeypatch.setattr(old,'shape_lp',lambda _:lp)
    p,rest,info=b.replay(cap,data,kind,nf,ng,lp['prices'],policy,trace=True)
    reference=old.greedy(a,'pgfp' if policy=='pgfp' else 'best_fit',11)
    assert tuple(p)==tuple(reference[0]) and old.validate(a,p)
    public=accelsched.solve(a,policy,seed=11,time_limit=.2)
    assert public.placement==tuple(p)
    task,shapes=b.orders(data,lp['prices'])
    utility=a.value[None,:]-lp['prices']@a.demand.T
    assert task.tolist()==sorted(range(c.n),key=lambda j:(-max(utility[:,j]),j))
    for step in info['ledger']:
        j=step['job'];before=np.array(step['before']);chosen=step['chosen']
        expect_order=sorted(range(len(labels)),key=lambda h:(-utility[h,j],h))
        assert step['shape_order']==expect_order
        expected=-1
        for h in expect_order:
            ids=[i for i in range(c.nodes) if kind[i]==labels[h] and a.eligible[i,j] and np.all(before[i]+1e-10>=a.demand[j])]
            if ids:
                if policy=='pgfp':
                    expected=max(ids,key=lambda i:(old.marginal(a,before,i,j),-i))
                    chosen_score=old.marginal(a,before,expected,j)
                    assert all(chosen_score>=old.marginal(a,before,i,j) for i in ids)
                else:expected=min(ids,key=lambda i:(float(np.sum(before[i]-a.demand[j])),i))
                break
        assert expected==chosen
    assert np.array_equal(rest,old.residual(a,p))
    assert int(np.sum((np.all(rest[:,None,:]>=b.FUTURE,axis=2)&a.future_eligible)*b.WEIGHTS))==old.future_score(a,rest)


def test_nonmonotone_pgfp_loss_improvement_and_smallest_node_tie():
    cap=np.array([[5,8,32],[5,8,32]])
    future=np.array([[4,0,0]]);weights=np.array([3]);fe=np.ones((2,1),bool)
    ix=b.ExactIndex(cap,[0,0],[-1,-1],[0,0],'pgfp',future,weights,fe)
    assert ix.query([3,0,0],0,-1,-1)==0
    q=next(iter(ix.queries.values()));assert q.cost==[3,3]
    # Losing future suitability can IMPROVE another demand's marginal from -3 to0.
    ix.commit(0,[2,0,0]);assert q.cost==[0,3] and ix.query([3,0,0],0,-1,-1)==0
    ix.commit(0,[3,0,0]);assert ix.query([3,0,0],0,-1,-1)==1


def test_pgfp_is_not_first_fit_or_best_fit():
    future=np.array([[4,0,0]]);weights=np.array([1]);fe=np.ones((2,1),bool)
    ix=b.ExactIndex(np.array([[5,8,32],[3,8,32]]),[0,0],[-1,-1],[0,0],'pgfp',future,weights,fe)
    assert ix.query([2,0,0],0,-1,-1)==1  # First-Fit would choose0 and lose future fitness.
    ix=b.ExactIndex(np.array([[8,8,32],[6,8,32]]),[0,0],[-1,-1],[0,0],'pgfp',future,weights,fe)
    assert ix.query([1,0,0],0,-1,-1)==0  # Best-Fit would choose1; both PGFP scores are0.


def test_cached_exact_index_matches_full_rescan_after_many_updates():
    rng=np.random.default_rng(47);cap=np.array([[8,32,128]]*12)
    shapes=np.arange(12)%2;nf=np.arange(12)%2;ng=np.arange(12)%3
    for policy in b.POLICIES:
        ix=b.ExactIndex(cap,shapes,nf,ng,policy)
        for _ in range(150):
            demand=np.array([int(rng.integers(3)),int(rng.integers(1,5)),int(rng.integers(1,5))*4])
            shape=int(rng.integers(2));family=int(rng.choice([-1,0,1]));group=int(rng.choice([-1,0,1,2,3]))
            ids=[i for i in range(12) if shapes[i]==shape and (family<0 or nf[i]==family) and (group<0 or ng[i]==group) and np.all(ix.rest[i]>=demand)]
            def score(i):
                before=np.all(ix.rest[i]>=b.FUTURE,axis=1)&ix.fe[i]
                after=np.all(ix.rest[i]-demand>=b.FUTURE,axis=1)&ix.fe[i]
                return int(np.sum((after.astype(int)-before.astype(int))*b.WEIGHTS))
            expected=(max(ids,key=lambda i:(score(i),-i)) if policy=='pgfp' else min(ids,key=lambda i:(int(ix.rest[i].sum()),i))) if ids else None
            actual=ix.query(demand,shape,family,group);assert actual==expected
            if actual is not None:ix.commit(actual,demand)


def test_real_highs_complete_pipeline_and_adversarial_audit(tmp_path):
    c=b.base.Config(100,16,'high_pressure',11,64)
    row=b.pipeline(c,tmp_path,'p','pgfp','highs');assert row['status']=='PHYSICAL_E2E_VALID'
    assert row['lp']['independent']['certified'] and row['policy_status']=='PGFP_CERTIFIED_PRICE_POLICY_DIFFERENT'
    data=b.base.materialize(c)
    with np.load(tmp_path/'p.npz',allow_pickle=False) as z:
        check=b.audit_extra(c,data,z['placement'],z['residual'],row['input_sha256'])
        assert check['accepted_gpu_jobs']<=check['accepted_count'] and check['weighted_future_placeability']>=0
        with pytest.raises(AssertionError):b.audit_extra(c,data,z['placement'],z['residual']+1,row['input_sha256'])
    assert np.isclose(sum(row['stages'].values())+row['other_seconds'],row['full_seconds'])


def test_reject_silent_fractional_truncation():
    with pytest.raises(AssertionError):b.ExactIndex(np.array([[1.5,4,16]]),[0],[-1],[0],'pgfp')

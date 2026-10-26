"""Focused CPU deterministic PGFP and bounded candidate stream checks."""
from copy import deepcopy
import numpy as np
from probes import pgfp_gpu_scoring as p


def test_ties_competing_ineligible_and_rejected_jobs():
    a=p.edge_case();c=p.g.control(a)['certificate'];reference=p.g.replay(a,c)
    actual=p.replay(a,c,'numpy')
    assert p.compare(reference,actual)['status']=='EXACT'
    assert actual['placement']==[0,-1,-1,-1]
    assert actual['ledger_steps'][0]['attempts'][0]['candidates']==[0,1]


def test_original_epsilon_cliff_matches_scalar():
    a=p.g.p4b.fixed_cases()[0][1];c=p.g.control(a)['certificate']
    assert p.compare(p.g.replay(a,c),p.replay(a,c,'numpy'))['status']=='EXACT'


def test_changed_node_or_task_order_is_diff_even_if_scores_equal():
    a=p.edge_case();c=p.g.control(a)['certificate'];ref=p.g.replay(a,c)
    other=deepcopy(ref);other['placement'][0]=1
    assert p.compare(ref,other)['status']=='DIFF'
    other=deepcopy(ref);other['task_order']=list(reversed(other['task_order']))
    assert p.compare(ref,other)['status']=='DIFF'


def test_frozen_chunk_output_and_prefix_are_distinct():
    data=p.ladder(100,16);one=p.frozen(data,'numpy',17);two=p.frozen(data,'numpy',17)
    assert one['digest']==two['digest'] and one['candidate_pairs']==1600
    assert one['scalar_samples_checked']==256
    assert p.scalar_samples(data,8)['status']=='EXACT'
    a=p.prefix(data,64);assert len(a.value)==64 and a.eligible.shape==(16,64)
    assert not np.all(a.eligible)


def test_nonbinary_weights_match_original_scalar_reduction():
    a=p.edge_case();a.weights=np.array([.1,.2]);engine=p.Engine(a,'numpy')
    for j in range(len(a.value)):
        valid,scores=engine.candidates(j)
        for i in np.flatnonzero(valid):assert scores[i]==p.g.oracle.marginal(a,a.capacity,i,j)

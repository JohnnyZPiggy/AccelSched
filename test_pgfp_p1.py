import numpy as np
import pytest
import bench_ultra_pgfp as b
from probes import pgfp_p1 as p


@pytest.mark.parametrize('policy',b.POLICIES)
def test_observation_preserves_full_replay(monkeypatch,policy):
    c=b.base.Config(96,8,'multiresource',11,64)
    cap,nf,ng,shapes=b.base.inventory(c);data=b.base.materialize(c);prices=np.zeros((4,3))
    expected=b.replay(cap,data,shapes,nf,ng,prices,policy,trace=True)
    original_q=b.CandidateSet;original_ix=b.ExactIndex
    observed=p.observe(b)
    monkeypatch.setattr(b,'CandidateSet',b.CandidateSet);monkeypatch.setattr(b,'ExactIndex',b.ExactIndex)
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

"""Focused original-policy equivalence and conflict-safe resident commit checks."""
from copy import deepcopy
import itertools
import numpy as np
import pytest
from probes import pgfp_gpu_resident as r


def test_score_table_canonical_nonbinary_arithmetic():
    weights=np.array([.1,.2,.3,1.,-.1,0.]);table,powers=r.score_table(weights)
    for row in itertools.product((-1,0,1),repeat=6):
        assert table[int(np.dot(np.array(row)+1,powers))]==float(np.sum(np.array(row)*weights))


@pytest.mark.parametrize('a',[r.p.edge_case(),r.g.p4b.fixed_cases()[0][1],*r.g.scenarios().values()])
def test_resident_numpy_full_original_ledger(a):
    c=r.g.control(a)['certificate'];reference=r.g.replay(a,c);actual=r.replay(a,c)
    assert r.p.compare(reference,actual)['status']=='EXACT'
    assert actual['buffers']['per_task_host_sync']==0


def test_competition_rejection_and_detect_tampering():
    a=r.p.edge_case();c=r.g.control(a)['certificate'];actual=r.replay(a,c)
    assert actual['placement']==[0,-1,-1,-1]
    broken=deepcopy(actual['ledger_steps']);broken[1]['chosen']=0
    with pytest.raises(ValueError):r.g.audit_steps(a,broken,actual['task_order'],actual['shape_order'])
    c=deepcopy(c);c['lp_input_sha256']='wrong'
    with pytest.raises(ValueError):r.replay(a,c)

"""Focused same-model, profiles, certification and bounded admission checks."""
import numpy as np
import pytest
from probes import gpu_lp_solver_benchmark as b


def test_cpu_callable_fallback_full_original_certificate():
    m,original,_=b.p.shape(1000,64);out,v=b.call(m,original,'highs','certification')
    assert out['status']=='PASS' and set(v)=={'x','pi','rc'}
    assert out['certificate']['p5_strict_original']['certified']
    packed,meta=b.archive(v);assert packed and sum(q['bytes'] for q in meta.values())==136192


def test_original_input_hash_and_roundtrip_bound_sensitivity():
    m,_,_=b.p.shape(1000,64);digest=b.input_hash(m);m['ub'][0]=1.
    assert b.input_hash(m)!=digest
    assert b.p.PINS[1000][1]=='ee8fa3a4bcb49dce081e1309a88bbe5a019a11b04f151e207d15f1bc29154d01'


def test_declared_profiles_and_rotated_backend_order():
    assert 'OptimalityTol' not in b.settings('hpdlp','native_default')
    assert b.settings('cupdlpx','certification')['OptimalityTol']==1e-12
    assert b.order(1000,'native_default',0)!=b.order(1000,'native_default',1)
    for n in (1000,10000,100000):assert set(b.order(n,'certification',1))==set(b.BACKENDS)
    with pytest.raises(ValueError):b.settings('hpdlp','hidden')


def test_100k_admission_requires_all_strict_small_pairs_and_budget_memory():
    rows=[dict(profile='certification',n=n,status='PASS') for n in (1000,10000) for _ in range(6)]
    assert b.eligible_100k(rows,200,1000,1000,100)
    rows[-1]['status']='UNCERTIFIED';assert not b.eligible_100k(rows,200,1000,1000,100)
    rows[-1]['status']='PASS';assert not b.eligible_100k(rows,100,1000,1000,100)
    assert not b.eligible_100k(rows,200,100,1000,100)

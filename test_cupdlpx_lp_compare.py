"""Focused CPU checks for the bounded P4F matrix and comparison semantics."""
import numpy as np
import pytest
from probes import cupdlpx_lp_compare as p


def test_fixed_sparse_matrix_and_p4c_hash():
    cases = p.cases()
    assert list(cases) == ['hand_unique','hand_degenerate','moderate_m4_n64','moderate_m8_n128']
    for name, dims, nnz in [('moderate_m4_n64',(4,64,3),1024),
                            ('moderate_m8_n128',(8,128,3),4096)]:
        m = p.sparse_model(cases[name])
        assert tuple(m[k] for k in ('m','n','r')) == dims
        assert m['A'].format == 'csr' and m['A'].nnz == nnz
    assert p.fixture_sha(cases['moderate_m8_n128']) == '23bcd11d7c68b7115ece24ddc41a6ed7aef5fef79400c24c5a8238482022db71'
    assert p.fixture_sha(cases['moderate_m4_n64']) == p.fixture_sha(p.cases()['moderate_m4_n64'])


@pytest.mark.parametrize('name', ['hand_unique','hand_degenerate'])
def test_real_cpu_status_rc_certificate_and_stage_accounting(name):
    r = p.solve_one(p.cases()[name], 'highs')
    assert r['status'] == 'PASS' and r['certificate']['certified']
    assert r['certificate']['primal_objective'] == pytest.approx(3)
    assert r['certificate']['rc_max_error'] < 1e-10
    t = r['timings_seconds']
    assert sum(v for k,v in t.items() if k != 'total') == pytest.approx(t['total'])
    assert r['reported_runtime_fraction'] is None


def test_no_implicit_optional_gpu_import():
    r = p.solve_one(p.cases()['hand_unique'], 'cupdlpx')
    assert r['status'] == 'FAILED' and 'explicit optional' in r['error']


def test_nonunique_certificates_report_order_divergence_without_rejection():
    a = p.cases()['hand_degenerate']
    c = p.certificate(a, [[1,1]], [1,2,0,0,0], [0,0], 1)
    g = p.certificate(a, [[1,1]], [0,0,1,0,0], [0,0], 1)
    assert c['certified'] and g['certified']
    compared = p.compare_pair(dict(certificate=c), dict(certificate=g))
    assert compared['primal_value_error'] == 0
    assert not compared['task_order_match'] and compared['shape_order_match']


def test_uncertified_results_never_produce_certified_latency_ratio():
    c = dict(status='PASS',timings_seconds=dict(total=.1))
    g = dict(status='UNCERTIFIED',timings_seconds=dict(total=.01))
    r = p.summarize([c]*3,[g]*3)
    assert r['certified_latency_ratio_cpu_over_gpu'] is None
    assert r['cupdlpx']['certified_count'] == 0


def test_comparison_median_uses_measurements_only():
    records = [dict(status='PASS',timings_seconds=dict(total=t)) for t in (1.,2.,9.)]
    r = p.summarize(records, records)
    assert r['highs']['median_total_seconds'] == 2
    assert r['certified_latency_ratio_cpu_over_gpu'] == 1
    assert p.REPETITIONS == 3 and p.GPU_SETTINGS['TimeLimit'] <= 5

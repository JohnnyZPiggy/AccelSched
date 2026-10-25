"""P4D CPU contract tests; synthetic observations are never cuOpt evidence."""
import numpy as np
import pytest
from probes import cuopt_lp_smoke as p
from probes.shape_lp_backend_probe import solve_sparse


def compatible_facts():
    return dict(cpu_count=4, cpu_affinity_count=4, mem_total_bytes=16*2**30,
                disk_free_bytes=100*10**9, disk_total_bytes=200*10**9,
                nvme_verified=True, python_version=[3, 13, 16], openssl_version=[3, 0, 13],
                libssl_so3=True, libcrypto_so3=True, platform='Linux-test',
                machine='x86_64', glibc=['glibc', '2.39'],
                pinned_wheel_and_dependencies_verified=True,
                nvidia_smi=dict(exit_code=0, stdout='Tesla T4, 7.5, 580.82.07, 15360\n'),
                nvcc=dict(exit_code=0, stdout='Cuda compilation tools, release 13.0, V13.0.88'))


def test_gate_exact_boundaries():
    assert p.environment_gate(compatible_facts())['status'] == 'PASS'
    assert p.environment_gate({})['status'] == 'GPU_BACKEND_BLOCKED'


@pytest.mark.parametrize('field,value', [
    ('cpu_count', 2), ('cpu_affinity_count', 2), ('mem_total_bytes', 16*2**30-1),
    ('disk_free_bytes', 100*10**9-1), ('nvme_verified', None),
    ('pinned_wheel_and_dependencies_verified', None), ('python_version', [3, 10]),
    ('libssl_so3', False), ('platform', 'Windows'), ('glibc', ['glibc', '2.27']),
    ('nvcc', dict(exit_code=0, stdout='release 12.0,')),
    ('nvidia_smi', dict(exit_code=0, stdout='Tesla T4, 7.5, 525.0, 15360')),
])
def test_any_failed_minimum_blocks(field, value):
    facts = compatible_facts()
    facts[field] = value
    assert p.environment_gate(facts)['status'] == 'GPU_BACKEND_BLOCKED'


def test_actual_colab_class_of_failure_and_free_not_total():
    facts = compatible_facts()
    facts.update(cpu_count=2, cpu_affinity_count=2, mem_total_bytes=13605830656,
                 disk_free_bytes=75298086912, disk_total_bytes=120942624768)
    gate = p.environment_gate(facts)
    assert not gate['checks']['cpu_four_cores']
    assert not gate['checks']['ram_16_gib']
    assert not gate['checks']['disk_100_decimal_gb_free']


@pytest.mark.parametrize('raw,expected', [([0, -3, 0, 0], -1), ([0, 3, 0, 0], 1)])
def test_calibration_never_assumes_sign(raw, expected):
    result = p.calibrate_hand_a([[.5]], raw)
    assert result['normalization_sign'] == expected
    assert result['certified'] and result['objective_gap'] == 0


@pytest.mark.parametrize('raw,x', [([0, 0, 0, 0], [[.5]]),
                                  ([0, -3, 0, 0], [[1]]),
                                  ([0, float('nan'), 0, 0], [[.5]])])
def test_reject_false_certificates(raw, x):
    with pytest.raises(ValueError):
        p.calibrate_hand_a(x, raw)


def test_two_real_cpu_oracles_and_degenerate_order_boundary():
    for name in ('hand_unique', 'hand_degenerate'):
        a = p.fixtures()[name]
        result = solve_sparse(a)
        raw = -np.concatenate((result['job_duals'], np.array(result['capacity_duals']).ravel()))
        checked = p.certificate(a, result['x'], raw, -1)
        assert checked['certified'] and checked['primal_objective'] == pytest.approx(3)
        assert checked['task_order'] == result['task_order']
        assert checked['shape_order'] == result['shape_order']
    a = p.fixtures()['hand_degenerate']
    # Two explicitly constructed certificates, not fabricated device observations.
    first = p.certificate(a, [[1, 1]], [1, 2, 0, 0, 0], 1)
    second = p.certificate(a, [[1, 1]], [0, 0, 1, 0, 0], 1)
    assert first['certified'] and second['certified']
    assert first['task_order'] == [1, 0] and second['task_order'] == [0, 1]

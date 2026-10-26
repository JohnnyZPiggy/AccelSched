"""CPU-only P4E contract checks; no optional package import or device evidence."""
import numpy as np
import pytest
from probes import cupdlpx_lp_smoke as p


def test_no_source_fallback_or_windows_wheel():
    files = [dict(filename='cupdlpx-0.3.0.tar.gz', packagetype='sdist'),
             dict(filename='cupdlpx-0.1.2-cp312-cp312-win_amd64.whl', packagetype='bdist_wheel')]
    gate = p.wheel_gate(dict(info=dict(name='cupdlpx', version='0.3.0'), urls=files,
                             releases={'0.1.2': files}))
    assert gate['status'] == 'GPU_BACKEND_BLOCKED'
    assert not gate['current_linux_wheels'] and not gate['any_release_linux_wheels']


def test_wheel_presence_is_not_gpu_gate_pass():
    wheel = dict(filename='cupdlpx-0.3.0-cp311-cp311-manylinux_2_28_x86_64.whl',
                 packagetype='bdist_wheel')
    gate = p.wheel_gate(dict(urls=[wheel]))
    assert gate['status'] == 'WHEEL_CANDIDATE_ONLY'
    assert not gate['runtime_abi_cuda_architecture_verified']


@pytest.mark.parametrize('pi,sign', [([0, -3, 0, 0], -1), ([0, 3, 0, 0], 1)])
def test_pi_sign_calibration_is_empirical_contract(pi, sign):
    checked = p.calibrate_hand_a([[.5]], pi, [0])
    assert checked['certified'] and checked['pi_sign'] == sign
    assert checked['objective_gap'] == 0 and checked['primal_objective'] == 3


@pytest.mark.parametrize('bad', ['pi', 'rc', 'x', 'status', 'nan', 'shape'])
def test_independent_certificate_rejects_bad_results(bad):
    a = p.fixtures()['hand_unique']
    x, pi, rc, optimal = [[.5]], [0, -3, 0, 0], [0], True
    if bad == 'pi': pi = [0, 0, 0, 0]
    if bad == 'rc': rc = [1]
    if bad == 'x': x = [[1]]
    if bad == 'status': optimal = False
    if bad == 'nan': pi = [0, float('nan'), 0, 0]
    if bad == 'shape': rc = [0, 0]
    if bad in ('nan', 'shape'):
        with pytest.raises(ValueError): p.certificate(a, x, pi, rc, -1, optimal)
    else:
        assert not p.certificate(a, x, pi, rc, -1, optimal)['certified']


def test_degeneracy_valid_duals_can_change_task_order():
    a = p.fixtures()['hand_degenerate']
    first = p.certificate(a, [[1, 1]], [1, 2, 0, 0, 0], [0, 0], 1)
    second = p.certificate(a, [[1, 1]], [0, 0, 1, 0, 0], [0, 0], 1)
    assert first['certified'] and second['certified']
    assert first['task_order'] == [1, 0] and second['task_order'] == [0, 1]
    assert first['shape_order'] == second['shape_order']


def test_actual_highs_reference_two_hand_cases():
    results = p.cpu_probe()
    assert set(results) == {'hand_unique', 'hand_degenerate'}
    for result in results.values():
        c = result['constructed_contract_observations']
        assert result['cupdlpx_observations'] is None
        assert c['certified'] and c['primal_objective'] == pytest.approx(3)
        assert c['rc_max_error'] < 1e-10 and c['rc_complementarity'] < 1e-10
        assert result['task_order_match'] and result['shape_order_match']

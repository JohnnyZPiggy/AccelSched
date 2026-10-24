"""Focused independent P4C oracle checks; CPU only."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from probes import shape_lp_backend_probe as p


def test_hand_sign_and_fractional_solution():
    a = p.fixtures()['hand_unique']
    result = p.solve_sparse(a)
    assert result['status'] == 'PASS'
    np.testing.assert_allclose(result['x'], [[.5]], atol=1e-10)
    np.testing.assert_allclose(result['capacity_duals'], [[3, 0, 0]], atol=1e-10)
    np.testing.assert_allclose(result['job_duals'], [0], atol=1e-10)
    assert result['primal_objective'] == pytest.approx(3)
    model = p.sparse_model(a)
    wrong = p.audit(a, model, np.array([[.5]]), np.array([[-3., 0, 0]]), np.zeros(1))
    assert not wrong['certified'] and wrong['dual_nonnegative_residual'] == 3


@pytest.mark.parametrize('name', ['hand_unique', 'hand_degenerate'])
def test_tiny_production_equivalence(name):
    a = p.fixtures()[name]
    result = p.solve_sparse(a)
    compared = p.compare_tiny(a, result)
    assert compared['value_error'] < 1e-10
    assert compared['x_max_error'] < 1e-10
    assert compared['production_x_feasible']
    assert compared['task_order_match'] and compared['shape_order_match']


def test_degenerate_certificates_do_not_imply_same_order():
    a = p.fixtures()['hand_degenerate']
    model = p.sparse_model(a)
    x = np.ones((1, 2))
    lam0, lam1 = np.zeros((1, 3)), np.array([[1., 0, 0]])
    assert p.audit(a, model, x, lam0, np.array([1., 2]))['certified']
    assert p.audit(a, model, x, lam1, np.zeros(2))['certified']
    assert p.orders(a, lam0)['task_order'] == [1, 0]
    assert p.orders(a, lam1)['task_order'] == [0, 1]


def test_sparse_pooling_indexing_and_first_seen_shapes():
    a = p.fixtures()['hand_degenerate']
    a = replace(a, capacity=np.array([[1., 2, 3], [4, 5, 6], [7, 8, 9]]),
                shapes=np.array([7, 3, 7]))
    model = p.sparse_model(a)
    assert model['labels'] == [7, 3]
    np.testing.assert_array_equal(model['caps'], [[8, 10, 12], [4, 5, 6]])
    x = np.array([[.2, .4], [.1, .3]])
    expected = np.concatenate((x.sum(axis=0), (x @ a.demand).ravel()))
    np.testing.assert_allclose(model['A'] @ x.ravel(), expected)
    assert model['A'].format == 'csr'


def test_moderate_sparse_no_densification(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('dense conversion forbidden')
    monkeypatch.setattr(csr_matrix, 'toarray', forbidden)
    a = p.fixtures()['moderate_seed11']
    result = p.solve_sparse(a)
    assert result['status'] == 'PASS'
    assert result['dimensions'] == dict(m=8, n=128, r=3)
    assert result['nnz'] == 4096
    assert result['sparse_bytes'] < (128+8*3)*8*128*8
    assert result['labels'] == [7, 3, 11, 5, 19, 2, 23, 17]
    assert p.fixture_sha(a) == p.fixture_sha(p.fixtures()['moderate_seed11'])


@pytest.mark.parametrize('corruption', ['negative', 'nan', 'shape'])
def test_invalid_inputs(corruption):
    a = p.fixtures()['hand_unique']
    if corruption == 'shape':
        a = replace(a, value=np.ones(2))
    else:
        a.demand[0, 0] = -1 if corruption == 'negative' else np.nan
    with pytest.raises(ValueError):
        p.sparse_model(a)


@pytest.mark.parametrize('kind', ['infeasible_x', 'bad_dual', 'nan', 'shape'])
def test_independent_audit_rejects_bad_certificates(kind):
    a = p.fixtures()['hand_unique']
    model = p.sparse_model(a)
    x, lam, mu = np.array([[.5]]), np.array([[3., 0, 0]]), np.zeros(1)
    if kind == 'infeasible_x':
        x[:] = 1
    elif kind == 'bad_dual':
        lam[:] = 0
    elif kind == 'nan':
        x[:] = np.nan
    else:
        mu = np.zeros(2)
    if kind in ('nan', 'shape'):
        with pytest.raises(ValueError):
            p.audit(a, model, x, lam, mu)
    else:
        assert not p.audit(a, model, x, lam, mu)['certified']


def test_stable_task_and_shape_ties():
    a = p.fixtures()['hand_degenerate']
    ranking = p.orders(a, np.array([[1., 0, 0], [1., 0, 0]]))
    assert ranking['task_order'] == [0, 1]
    assert ranking['shape_order'] == [[0, 1], [0, 1]]


def test_optional_gpu_is_not_imported_or_allocated(monkeypatch):
    monkeypatch.setattr(p.importlib.util, 'find_spec', lambda _: None)
    status = p.gpu_availability()
    assert status['backend_status'] == 'GPU_BACKEND_BLOCKED'
    assert not status['cuopt_module_discoverable'] and not status['sign_verified']

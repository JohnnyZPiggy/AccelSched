import numpy as np
import pytest
import workload
from accelsched import SolveResult, solve
from m0_oracle import validate


def inputs():
    return dict(capacity=[[8, 64, 256], [4, 32, 128]], demand=[[1, 4, 16], [2, 8, 32]],
                value=[3, 5], eligible=[[True, False], [False, True]], shapes=[9, 2])


def test_builder_order_units_copy_and_neutral_future():
    kw = inputs(); a = workload.build_workload(**kw)
    assert np.array_equal(a.capacity, kw['capacity']) and np.array_equal(a.demand, kw['demand'])
    assert a.shapes.tolist() == [9, 2] and a.eligible.tolist() == kw['eligible']
    assert workload.RESOURCE_UNITS == ('GPU-count', 'CPU-cores', 'RAM-GiB')
    assert a.weights.tolist() == [0.] and not a.future_eligible.any()
    kw['capacity'][0][0] = 0
    assert a.capacity[0, 0] == 8


@pytest.mark.parametrize('key,value', [('shapes', [0]), ('shapes', [0., 1.]),
    ('eligible', [[1, 0], [0, 1]]), ('eligible', [[True, False]]),
    ('capacity', [[1, 2], [3, 4]]), ('capacity', [[1, -1, 2], [0, 0, 0]]),
    ('demand', [[.5, 1, 1], [1, 1, 1]]), ('value', [float('nan'), 1]),
    ('value', [-1, 2]), ('demand', [[1, np.inf, 1], [1, 1, 1]])])
def test_reject_bad_shapes_eligibility_and_resources(key, value):
    kw = inputs(); kw[key] = value
    with pytest.raises(ValueError): workload.build_workload(**kw)


def test_future_fields_and_shapes():
    for extra in (dict(weights=[1]), dict(future=[[1, 1, 1]]),
                  dict(future=[[1, 1, 1]], weights=[1], future_eligible=[[True]])):
        with pytest.raises(ValueError): workload.build_workload(**inputs(), **extra)


def test_example_is_reproducible_and_eligibility_is_explicit():
    a, b = workload.heterogeneous_example(), workload.heterogeneous_example()
    for name in vars(a): assert np.array_equal(getattr(a, name), getattr(b, name))
    assert not a.eligible[:, -1].any() and len(np.unique(a.shapes)) == 3
    assert not a.eligible[a.capacity[:, 0] == 0][:, a.demand[:, 0] > 0].any()
    for i in range(len(a.capacity)):
        for j in range(len(a.capacity)):
            if a.shapes[i] == a.shapes[j]: assert np.array_equal(a.capacity[i], a.capacity[j])


def test_six_methods_public_parity_and_metrics_reconstruct():
    a = workload.heterogeneous_example()
    report = workload.evaluate(a)
    for row in report['rows']:
        p = row['placement']; direct = solve(a, row['method'], time_limit=2.)
        assert row['status'] == direct.status and row['optimal'] == direct.optimal
        assert row['accepted_value'] == direct.value and row['independent_valid'] and validate(a, p)
        assert p[-1] == -1 and row['admitted_count'] == sum(i >= 0 for i in p)
        used = np.array([a.demand[j] for j, i in enumerate(p) if i >= 0]).sum(axis=0)
        assert np.array_equal(row['used'], used)
        assert np.array_equal(row['utilization'], used/a.capacity.sum(axis=0))
        assert row['accepted_value'] == sum(a.value[j] for j,i in enumerate(p) if i>=0)
        assert row['offered_value_fraction'] == row['accepted_value']/sum(a.value)
        assert row['admitted_rate'] == row['admitted_count']/len(a.value)
        assert row['accepted_value'] <= row['certified_mip_value']


def test_zero_capacity_offered_value_and_all_ineligible():
    a = workload.build_workload([[0, 0, 0]], [[1, 1, 1]], value=[0], eligible=[[False]], shapes=[0])
    for row in workload.evaluate(a)['rows']:
        assert row['independent_valid'] and row['admitted_count'] == 0
        assert row['utilization'] == [None]*3 and row['offered_value_fraction'] is None
        assert row['certified_mip_value'] == 0 and row['value_to_mip_fraction'] is None


def test_invalid_placement_has_no_metrics_or_certificate(monkeypatch):
    a = workload.build_workload(**inputs())
    monkeypatch.setattr(workload, 'solve', lambda *args, **kwargs:
        SolveResult('highs_mip', 'optimal', (1, 0), 8., None, 0., True))
    for row in workload.evaluate(a)['rows']:
        assert row['status'] == 'invalid' and not row['independent_valid'] and not row['optimal']
        assert row['accepted_value'] is None and row['certified_mip_value'] is None


def test_timeout_incumbent_is_not_an_optimum_reference(monkeypatch):
    a = workload.build_workload(**inputs()); original = workload.solve
    def limited(a, method, **kwargs):
        if method == 'highs_mip':
            return SolveResult(method, 'feasible_uncertified', (-1, -1), 0., a.capacity, 0., False, 'injected time limit')
        return original(a, method, **kwargs)
    monkeypatch.setattr(workload, 'solve', limited)
    report = workload.evaluate(a)
    assert report['rows'][-1]['status'] == 'feasible_uncertified'
    assert all(row['certified_mip_value'] is None for row in report['rows'])

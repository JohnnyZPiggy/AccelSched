"""Independent tiny integer ledger and enumeration; no broad benchmark reruns."""
from copy import deepcopy
from itertools import product
import json
from unittest.mock import patch

import numpy as np
import pytest
from accelsched import SolveResult
from examples import capacity_whatif as planner
from workload import build_workload


@pytest.fixture(scope='module')
def report():
    # Exactly three real MIP calls across this entire focused test invocation.
    public, calls = planner.solve, []
    def spy(a, **kw):
        before = deepcopy(a)
        out = public(a, **kw)
        for key, value in vars(before).items():
            np.testing.assert_array_equal(getattr(a, key), value)
        calls.append(out)
        assert kw == dict(method='highs_mip', seed=11, time_limit=.5)
        return out
    with patch.object(planner, 'solve', side_effect=spy):
        result = planner.compare()
    assert len(calls) == 3
    for row, out in zip(result['rows'], calls):
        assert row['status'] == out.status and row['solver_optimal'] == out.optimal
        assert row['certified'] == (row['valid'] and out.status == 'optimal' and out.optimal)
    return result


def test_additions_shapes_and_common_jobs():
    original = deepcopy(planner.TASKS)
    assert len(planner.SCENARIOS) == 3
    for k, (_, nodes) in enumerate(planner.SCENARIOS):
        assert nodes[:2] == planner.BASELINE
        assert len(nodes) == (2 if k == 0 else 3)
        if k:
            assert nodes[-1].family == ('T4' if k == 1 else 'V100')
        a = planner.build_instance(nodes)
        assert a.demand.tolist() == [list(j.demand) for j in original]
        assert a.value.tolist() == [j.value for j in original]
        assert a.eligible.tolist() == [[n.family == j.family for j in original] for n in nodes]
        assert not a.weights.any() and not a.future_eligible.any()
        for i, n in enumerate(nodes):
            for h, m in enumerate(nodes):
                assert (a.shapes[i] == a.shapes[h]) == (n.capacity == m.capacity)
    added = planner.build_instance(planner.SCENARIOS[1][1])
    assert added.shapes[1] == added.shapes[2]  # V100/T4 equal capacity, different eligibility.
    assert not np.array_equal(added.eligible[1], added.eligible[2])
    assert planner.TASKS == original


def test_independent_raw_ledger_and_certificates(report):
    assert report['seed'] == 11 and report['time_limit'] == .5
    assert [r['scenario'] for r in report['rows']] == ['Baseline', 'Add T4', 'Add V100']
    assert [r['accepted_count'] for r in report['rows']] == [3, 5, 4]
    assert [r['delta_count'] for r in report['rows']] == [0, 2, 1]
    for row, (_, nodes) in zip(report['rows'], planner.SCENARIOS):
        load = np.zeros((len(nodes), 3))
        accepted, rejected, value = [], [], 0
        for j, i in zip(planner.TASKS, row['placement']):
            assert j.demand[0] == int(j.demand[0]) and min(j.demand) > 0
            if i < 0:
                rejected.append(j.id)
            else:
                assert nodes[i].family == j.family
                load[i] += j.demand
                accepted.append(j.id)
                value += j.value
        cap = np.array([n.capacity for n in nodes])
        left = cap-load
        assert np.all(left >= 0)
        np.testing.assert_array_equal(row['per_node_residual'], left)
        np.testing.assert_array_equal(load+row['per_node_residual'], cap)
        np.testing.assert_array_equal(row['total_residual'], left.sum(axis=0))
        np.testing.assert_array_equal(row['utilization'], load.sum(axis=0)/cap.sum(axis=0))
        assert row['accepted_ids'] == accepted and row['unplaced_ids'] == rejected
        assert len(accepted) == row['accepted_count'] and value == row['accepted_value']
        assert {'p100-missing', 'v-oversize'} <= set(rejected)
        assert row['certified'] and row['solver_optimal'] and row['status'] == 'optimal'
        assert row['valid'] and row['comparison'] == 'CERTIFIED'
        assert row['difference_kind'] == 'static-model optimal difference'
        assert row['delta_value'] == value-report['rows'][0]['accepted_value']
        assert row['solve_wall_ms'] >= 0 and row['validation_error'] is None
    assert all(j.id in report['rows'][1]['accepted_ids'] for j in planner.TASKS[:4])
    assert {'v-large-1', 'v-large-2'} <= set(report['rows'][2]['accepted_ids'])
    assert json.loads(json.dumps(report, allow_nan=False))['label'] == report['label']


def test_baseline_tiny_brute_force(report):
    best = 0
    nodes = planner.BASELINE
    choices = [[-1]+[i for i, n in enumerate(nodes) if n.family == j.family]
               for j in planner.TASKS]
    for placement in product(*choices):
        load = [[0, 0, 0] for _ in nodes]
        value = 0
        for j, i in zip(planner.TASKS, placement):
            if i >= 0:
                load[i] = [x+y for x, y in zip(load[i], j.demand)]
                value += j.value
        if all(all(x <= c for x, c in zip(v, n.capacity)) for v, n in zip(load, nodes)):
            best = max(best, value)
    assert best == 3 == report['rows'][0]['accepted_value']


def test_zero_total_utilization_and_uncertified_incumbent():
    a = build_workload([[0, 4, 16]], [[0, 2, 8]], value=[1],
                       eligible=np.ones((1, 1), bool), shapes=[0])
    out = SolveResult('highs_mip', 'feasible_uncertified', (0,), 1.,
                      np.array([[0., 2., 8.]]), .1, False, 'time limit')
    row = planner.describe(a, out, ['cpu-only'])
    assert row['utilization'] == [None, .5, .5]
    assert row['valid'] and not row['certified']
    assert row['comparison'] == 'FEASIBLE CANDIDATE' and row['error'] == 'time limit'
    assert row['unplaced_ids'] == []


@pytest.mark.parametrize('status,flag', [('optimal', False), ('feasible_uncertified', True)])
def test_certificate_requires_status_and_flag(status, flag):
    a = planner.build_instance(planner.BASELINE)
    out = SolveResult('highs_mip', status, (-1,)*8, 0., a.capacity.copy(), 0., flag)
    row = planner.describe(a, out, [j.id for j in planner.TASKS])
    assert row['valid'] and not row['certified']
    assert row['comparison'] == 'FEASIBLE CANDIDATE'


@pytest.mark.parametrize('status', ['uncertified', 'failed', 'input_invalid', 'invalid'])
def test_failures_are_not_rejections(status):
    a = planner.build_instance(planner.BASELINE)
    out = SolveResult('highs_mip', status, None, None, None, .1, False, 'probe failure')
    row = planner.describe(a, out, [j.id for j in planner.TASKS])
    assert row['comparison'] == 'NOT COMPARABLE' and not row['certified']
    assert row['unplaced_ids'] is None and row['accepted_count'] is None
    assert row['total_residual'] is None and row['error'] == 'probe failure'


def test_wrong_family_and_mismatched_public_metrics_rejected():
    a = planner.build_instance(planner.BASELINE)
    p = (1, -1, -1, -1, -1, -1, -1, -1)  # T4 job on V100 despite spare capacity.
    out = SolveResult('highs_mip', 'optimal', p, 1., a.capacity.copy(), .1, True)
    row = planner.describe(a, out, [j.id for j in planner.TASKS])
    assert not row['valid'] and not row['certified'] and row['unplaced_ids'] is None
    p = (-1,)*8
    out = SolveResult('highs_mip', 'optimal', p, 1., a.capacity.copy(), .1, True)
    assert not planner.describe(a, out, [j.id for j in planner.TASKS])['valid']


def test_shared_call_contract_and_input_immutability(monkeypatch):
    seen = []
    def fake(a, **kw):
        seen.append((a, kw))
        return SolveResult('highs_mip', 'uncertified', None, None, None, 0., False, 'test')
    monkeypatch.setattr(planner, 'solve', fake)
    before = deepcopy((planner.SCENARIOS, planner.TASKS))
    rows = planner.compare()['rows']
    assert len(seen) == 3 and (planner.SCENARIOS, planner.TASKS) == before
    for a, kw in seen:
        assert kw == dict(method='highs_mip', seed=11, time_limit=.5)
        assert a.demand.tolist() == [list(j.demand) for j in planner.TASKS]
    assert all(r['delta_count'] is None and r['difference_kind'] == 'NOT COMPARABLE' for r in rows)

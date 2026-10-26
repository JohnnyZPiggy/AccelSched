"""Independent original-job occupancy oracle; one shared three-call experiment."""
from copy import deepcopy
import json
from unittest.mock import patch

import numpy as np
import pytest
from examples import time_window_capacity as planner
from examples.capacity_whatif import Node


@pytest.fixture(scope='module')
def experiment():
    public, records = planner.simulate, []
    def spy(capacity, jobs, eligible, **kwargs):
        before = deepcopy((capacity, jobs, eligible))
        assert jobs is planner.JOBS
        assert kwargs == dict(method='highs_mip', seed=11, time_limit=.5)
        nodes = planner.SCENARIOS[len(records)][1]
        assert capacity == [n.capacity for n in nodes]
        np.testing.assert_array_equal(eligible,
            [[n.family == f for f in planner.REQUESTS] for n in nodes])
        raw = public(capacity, jobs, eligible, **kwargs)
        assert capacity == before[0] and jobs == before[1]
        np.testing.assert_array_equal(eligible, before[2])
        records.append(raw)
        return raw
    with patch.object(planner, 'simulate', side_effect=spy):
        result = planner.compare()
    assert len(records) == 3
    return result, records


def ledger(nodes, jobs, requests, raw, row, horizon):
    """Rebuild all resource states from decisions, then integrate reported events."""
    cap = np.array([n.capacity for n in nodes])
    by_id = {j.id: (j, f) for j, f in zip(jobs, requests)}
    decisions = raw['decisions']
    assert len(decisions) == len(jobs) and {d['job_id'] for d in decisions} == set(by_id)
    accepted = [d for d in decisions if d['status'] == 'accepted']
    full, window, admitted = np.zeros(3), np.zeros(3), np.zeros(3)
    for d in decisions:
        j, family = by_id[d['job_id']]
        assert d['arrival'] == j.arrival and j.demand[0] == int(j.demand[0])
        if d['status'] == 'accepted':
            assert isinstance(d['node'], int) and 0 <= d['node'] < len(nodes)
            assert nodes[d['node']].family == family
            assert d['start'] == j.arrival and d['end'] == j.arrival+j.duration
            full += np.array(j.demand)*j.duration
            window += np.array(j.demand)*max(0, min(d['end'], horizon)-max(d['start'], 0))
            admitted += j.demand
        else:
            assert d['status'] in ('not_scheduled', 'solver_error')
            assert (d['node'], d['start'], d['end']) == (None, None, None)

    def load_at(t, before_arrival=False):
        load = np.zeros_like(cap)
        for d in accepted:
            if ((d['start'] < t if before_arrival else d['start'] <= t) and t < d['end']):
                load[d['node']] += by_id[d['job_id']][0].demand
        return load

    times = sorted({j.arrival for j in jobs} | {d['end'] for d in accepted})
    assert [e['time'] for e in raw['events']] == times
    event_window, event_full = np.zeros(3), np.zeros(3)
    previous_time, previous_load = 0, np.zeros_like(cap)
    completed = []
    for event in raw['events']:
        t = event['time']
        event_window += previous_load.sum(axis=0)*max(0, min(t, horizon)-min(previous_time, horizon))
        event_full += previous_load.sum(axis=0)*(t-previous_time)
        expected = load_at(t)
        np.testing.assert_array_equal(event['running_load'], expected)
        np.testing.assert_array_equal(event['free'], cap-expected)
        np.testing.assert_array_equal(np.array(event['free'])+event['running_load'], cap)
        assert np.all(expected >= 0) and np.all(expected <= cap)
        np.testing.assert_array_equal(event['free_before_arrivals'], cap-load_at(t, True))
        active = {d['job_id']: (d['node'], d['end']) for d in accepted if d['start'] <= t < d['end']}
        assert len(event['running']) == len(active)
        assert {d['job_id']: (d['node'], d['end']) for d in event['running']} == active
        expected_completed = [d['job_id'] for d in accepted if d['end'] == t]
        assert sorted(event['completed']) == sorted(expected_completed)
        completed += event['completed']
        for field, status in [('accepted', 'accepted'), ('rejected', 'not_scheduled'), ('solver_errors', 'solver_error')]:
            assert sorted(event[field]) == sorted(d['job_id'] for d in decisions if d['arrival'] == t and d['status'] == status)
        previous_time, previous_load = t, np.array(event['running_load'])
    event_window += previous_load.sum(axis=0)*max(0, horizon-previous_time)
    assert len(completed) == len(set(completed)) == len(accepted)
    np.testing.assert_array_equal(window, event_window)
    np.testing.assert_array_equal(full, event_full)
    np.testing.assert_array_equal(raw['resource_time'], full)
    np.testing.assert_array_equal(raw['admitted_resources'], admitted)
    np.testing.assert_array_equal(raw['released_resources'], admitted)
    np.testing.assert_array_equal(raw['final_free'], cap)
    np.testing.assert_array_equal(row['window_resource_time'], window)
    np.testing.assert_array_equal(row['full_tail_resource_time'], full)
    np.testing.assert_array_equal(row['utilization'], window/(horizon*cap.sum(axis=0)))
    assert row['accepted_count'] == len(accepted) == raw['accepted_count'] == raw['completed_count']
    assert row['accepted_value'] == sum(by_id[d['job_id']][0].value for d in accepted)
    for field, status in [('accepted_ids', 'accepted'), ('rejected_ids', 'not_scheduled'), ('solver_error_ids', 'solver_error')]:
        assert row[field] == [d['job_id'] for d in decisions if d['status'] == status]
    assert row['rejected_count'] == raw['rejected_count']
    assert row['solver_error_count'] == raw['solver_error_count']
    assert row['completed_by_T'] == [d['job_id'] for d in accepted if d['end'] <= horizon]
    assert row['running_at_T'] == [d['job_id'] for d in accepted if d['start'] <= horizon < d['end']]

    # Independent integer-time oracle for pressure, not the event implementation.
    for family, demand in planner.REFERENCES.items():
        blocked = []
        for t in range(horizon):
            free = cap-load_at(t)
            if not any(n.family == family and all(free[i, k] >= demand[k] for k in range(3))
                       for i, n in enumerate(nodes)):
                blocked.append(t)
        intervals = row['pressure'][family]['intervals']
        assert [t for a, b in intervals for t in range(a, b)] == blocked
        assert row['pressure'][family]['fraction'] == len(blocked)/horizon
    assert [b['time'] for b in raw['batches']] == sorted({j.arrival for j in jobs})
    for b, printed in zip(raw['batches'], row['batches']):
        assert b['job_ids'] == [j.id for j in jobs if j.arrival == b['time']]
        assert printed['optimal'] == b['optimal']
        assert printed['solver_status'] == b['solver_status']
        assert b['optimal'] == (b['solver_status'] == 'optimal' and b['reported_optimal']
                                and b['validation_status'] == 'valid' and b['placement'] is not None)
        ds = [d for d in decisions if d['arrival'] == b['time']]
        assert all(d['solver_status'] == b['solver_status'] for d in ds)
        if b['placement'] is not None:
            assert b['placement'] == [d['node'] if d['status'] == 'accepted' else -1 for d in ds]
            value = sum(by_id[d['job_id']][0].value for d in ds if d['status'] == 'accepted')
            assert b['accepted_value'] == value
            assert b['certified_value'] == (value if b['optimal'] else None)
        else:
            assert all(d['status'] == 'solver_error' for d in ds)
    assert row['certified_batch_count'] == sum(b['optimal'] for b in raw['batches'])


def test_actual_stream_ledger_and_outcomes(experiment):
    result, records = experiment
    assert [r['scenario'] for r in result['rows']] == ['Baseline', 'Add T4', 'Add V100']
    assert [r['accepted_count'] for r in result['rows']] == [3, 5, 5]
    assert [r['rejected_count'] for r in result['rows']] == [5, 3, 3]
    assert [len(r['completed_by_T']) for r in result['rows']] == [2, 4, 3]
    assert [len(r['running_at_T']) for r in result['rows']] == [1, 1, 2]
    assert [r['delta_admitted'] for r in result['rows']] == [0, 2, 2]
    for (_, nodes), row, raw in zip(planner.SCENARIOS, result['rows'], records):
        ledger(nodes, planner.JOBS, planner.REQUESTS, raw, row, planner.T)
        assert row['certified_batch_count'] == 5 and row['solver_error_count'] == 0
        assert row['batch_status_counts'] == {'optimal': 5}
        assert row['difference_kind'] == 'observed simulation difference'
        assert 'P-missing' in row['rejected_ids']
    assert [r['window_resource_time'][0] for r in result['rows']] == [30, 36, 46]
    assert [r['full_tail_resource_time'][0] for r in result['rows']] == [34, 40, 58]
    assert [r['pressure']['T4']['fraction'] for r in result['rows']] == [5/6, 0, 5/6]
    assert [r['pressure']['V100']['fraction'] for r in result['rows']] == [5/6, 5/6, 4/6]
    assert json.loads(json.dumps(result, allow_nan=False))['T'] == 6


def test_shared_inputs_and_additive_inventory(experiment):
    result, _ = experiment
    for k, ((_, nodes), row) in enumerate(zip(planner.SCENARIOS, result['rows'])):
        assert nodes[:2] == planner.BASELINE and len(nodes) == (2 if k == 0 else 3)
        if k:
            assert nodes[-1].family == ('T4' if k == 1 else 'V100')
        assert row['job_ids'] == [j.id for j in planner.JOBS]
        assert row['requested_families'] == list(planner.REQUESTS)
        assert row['inventory'] == [n._asdict() for n in nodes]


def test_T_boundary_and_empty_final_interval(experiment):
    result, records = experiment
    assert 'B-V100' in result['rows'][0]['completed_by_T']  # end == T
    assert 'H-T4' in result['rows'][1]['completed_by_T']
    assert 'E-T4' in result['rows'][0]['running_at_T']
    for (_, nodes), raw in zip(planner.SCENARIOS, records):
        row = planner.summarize(nodes, planner.JOBS, planner.REQUESTS, raw, 10)
        ledger(nodes, planner.JOBS, planner.REQUESTS, raw, row, 10)
        assert not row['running_at_T'] and len(row['completed_by_T']) == raw['accepted_count']
    with pytest.raises(ValueError):
        planner.summarize(planner.BASELINE, planner.JOBS, planner.REQUESTS, records[0], 5)


def test_per_node_pressure_not_pooled_and_zero_total_guard():
    nodes = (Node('v0', 'V100', (2, 8, 32)), Node('v1', 'V100', (2, 8, 32)))
    p = planner.pressure(nodes, [], 6)
    assert p['V100']['intervals'] == [[0, 6]] and p['V100']['fraction'] == 1
    assert planner.utilization([0, 8, 32], [[0, 4, 16]], 4) == [None, .5, .5]


def test_oracle_detects_ledger_and_eligibility_corruption(experiment):
    result, records = experiment
    raw = deepcopy(records[0]);raw['events'][0]['free'][0][0] += 1
    with pytest.raises(AssertionError):
        ledger(planner.BASELINE, planner.JOBS, planner.REQUESTS, raw, result['rows'][0], 6)
    raw = deepcopy(records[0]);raw['decisions'][0]['node'] = 1  # T4 on V100.
    with pytest.raises(AssertionError):
        ledger(planner.BASELINE, planner.JOBS, planner.REQUESTS, raw, result['rows'][0], 6)


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('case_index', [0, 2])
def test_uncertified_and_solver_error_comparison_boundaries(experiment, monkeypatch, failure, case_index):
    _, originals = experiment
    records = deepcopy(originals)
    raw = records[case_index]
    b = raw['batches'][2]  # Missing-family job at t=3 is rejected in every inventory.
    status = 'uncertified' if failure else 'feasible_uncertified'
    b.update(solver_status=status, optimal=False, reported_optimal=False,
             certified_value=None, error='crafted boundary')
    for d in raw['decisions']:
        if d['arrival'] == 3:
            d['solver_status'] = status
            if failure:
                d['status'] = 'solver_error'
    if failure:
        b.update(placement=None, accepted_value=None, validation_status='not_run')
        e = next(e for e in raw['events'] if e['time'] == 3)
        e.update(solver_errors=e['rejected'], rejected=[])
        raw['rejected_count'] -= 1
        raw['solver_error_count'] = 1
    stream = iter(records)
    monkeypatch.setattr(planner, 'simulate', lambda *a, **kw: next(stream))
    result = planner.compare()
    row = result['rows'][case_index]
    ledger(planner.SCENARIOS[case_index][1], planner.JOBS, planner.REQUESTS, raw, row, 6)
    assert row['certified_batch_count'] == 4
    assert row['solver_error_ids'] == (['P-missing'] if failure else [])
    assert ('P-missing' in row['rejected_ids']) == (not failure)
    assert all(r['comparison'] == ('NOT COMPARABLE' if failure else 'COMPARABLE') for r in result['rows'])
    if failure:
        assert all(r['delta_admitted'] is None and r['delta_value'] is None for r in result['rows'])
    else:
        assert result['rows'][1]['delta_admitted'] == 2
        assert row['difference_kind'] == 'observed simulation difference'

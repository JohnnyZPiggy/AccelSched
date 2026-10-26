"""Two tiny shared planner runs; independent physical and time-window ledgers."""
from copy import deepcopy
from itertools import product
import json
from unittest.mock import patch

import pytest

from examples import capacity_decision_report as p2
from examples import capacity_whatif as p0
from examples import time_window_capacity as p1
from examples.planner_input import load_input


def input_document(mode):
    node = dict(id='n0', family='T4', capacity=[1, 4, 16])
    jobs = [dict(id='a', family='T4', demand=[1, 4, 16], value=2),
            dict(id='b', family='T4', demand=[1, 4, 16], value=1),
            dict(id='c', family='T4', demand=[1, 4, 16], value=3),
            dict(id='missing', family='P100', demand=[1, 4, 16], value=4)]
    if mode == 'dynamic':
        for job, (arrival, duration) in zip(jobs, [(0, 2), (1, 2), (2, 3), (3, 1)]):
            job.update(arrival=arrival, duration=duration)
    doc = dict(schema_version=1, mode=mode, nodes=[node], jobs=jobs,
               scenarios=[dict(name='Add Node', add_nodes=[dict(node, id='n1')]),
                          dict(name='Baseline', add_nodes=[]),
                          dict(name='No change', add_nodes=[])],
               solver=dict(method='highs_mip', seed=11, time_limit=.5))
    if mode == 'dynamic':
        doc.update(horizon=4, pressure_references=[dict(family='T4', demand=[1, 4, 16])])
    return doc


@pytest.fixture(scope='module')
def experiments(tmp_path_factory):
    root = tmp_path_factory.mktemp('p2-inputs')
    results = {}
    for mode, planner, engine in [('static', p0, 'solve'), ('dynamic', p1, 'simulate')]:
        path = root/(mode+'.json')
        path.write_text(json.dumps(input_document(mode)), encoding='utf-8')
        config = load_input(path, mode)
        original = getattr(planner, engine)
        records = []
        def spy(*args, **kwargs):
            assert kwargs == dict(method='highs_mip', seed=11, time_limit=.5)
            raw = original(*args, **kwargs)
            records.append(raw)
            return raw
        with patch.object(planner, engine, side_effect=spy):
            raw = planner.compare(config)
        assert len(records) == 3
        results[mode] = config, raw, records
    return results


def test_static_independent_placement_and_optimum(experiments):
    config, raw, solves = experiments['static']
    before = deepcopy(raw)
    report = p2.summarize_report(raw, 'static')
    assert report['input'] == config['provenance']
    assert report['leader'] == 'Add Node' and not report['tied_names']
    assert [r['name'] for r in report['scenarios']] == ['Add Node', 'Baseline', 'No change']
    for (name, nodes), solve, row in zip(config['scenarios'], solves, report['scenarios']):
        load = [[0, 0, 0] for _ in nodes]
        admitted, missing = [], []
        for job, i in zip(config['jobs'], solve.placement):
            if i < 0:
                missing.append(job.id)
                continue
            assert nodes[i].family == job.family
            admitted.append(job.id)
            for k in range(3):
                load[i][k] += job.demand[k]
                assert load[i][k] <= nodes[i].capacity[k]
        left = [[n.capacity[k]-load[i][k] for k in range(3)] for i, n in enumerate(nodes)]
        value = sum(j.value for j in config['jobs'] if j.id in admitted)
        # Exhaustive tiny physical assignments, independent of the MIP/P0 totals.
        values = []
        for placement in product(range(-1, len(nodes)), repeat=4):
            if any(i >= 0 and nodes[i].family != j.family for i, j in zip(placement, config['jobs'])):
                continue
            if all(sum(j.demand[k] for i, j in zip(placement, config['jobs']) if i == n)
                   <= nodes[n].capacity[k] for n in range(len(nodes)) for k in range(3)):
                values.append(sum(j.value for i, j in zip(placement, config['jobs']) if i >= 0))
        assert value == max(values) == (5 if name == 'Add Node' else 3)
        assert row['accepted_ids'] == admitted and row['unplaced_ids'] == missing
        assert row['accepted_count'] == len(admitted) == (2 if name == 'Add Node' else 1)
        assert row['accepted_value'] == value
        assert row['per_node_residual'] == left
        assert row['total_residual'] == [sum(v[k] for v in left) for k in range(3)]
        assert row['utilization'] == [sum(v[k] for v in load)/sum(n.capacity[k] for n in nodes) for k in range(3)]
        assert row['delta_accepted'] == (1 if name == 'Add Node' else 0)
        assert row['delta_value'] == (2 if name == 'Add Node' else 0)
        assert row['certified'] and row['comparison'] == 'static-model optimal comparison'
        assert 'missing' in missing
    assert raw == before
    report['input']['solver']['seed'] = 99
    assert raw == before  # Output does not alias source provenance.


def test_dynamic_original_job_and_event_integrals(experiments):
    config, raw, simulations = experiments['dynamic']
    before = deepcopy(raw)
    report = p2.summarize_report(raw, 'dynamic')
    assert report['horizon'] == 4 and report['leader'] == 'Add Node'
    for (name, nodes), sim, row in zip(config['scenarios'], simulations, report['scenarios']):
        jobs = {j.id: j for j in config['jobs']}
        accepted = [d for d in sim['decisions'] if d['status'] == 'accepted']
        assert [d['job_id'] for d in accepted] == (['a', 'b', 'c'] if name == 'Add Node' else ['a', 'c'])
        assert row['accepted_count'] == len(accepted)
        assert row['accepted_value'] == sum(jobs[d['job_id']].value for d in accepted)
        assert row['rejected_count'] == 4-len(accepted) and row['solver_error_count'] == 0
        assert row['completed_by_T_count'] == (2 if name == 'Add Node' else 1)
        assert row['running_at_T_count'] == 1
        work = [sum(jobs[d['job_id']].demand[k]*max(0, min(d['end'], 4)-d['start'])
                    for d in accepted) for k in range(3)]
        assert work == ([6, 24, 96] if name == 'Add Node' else [4, 16, 64])
        assert row['window_resource_time'] == work
        assert row['utilization'] == [work[k]/(4*sum(n.capacity[k] for n in nodes)) for k in range(3)]
        area, blocked = [0, 0, 0], []
        for t in range(4):
            load = [[sum(jobs[d['job_id']].demand[k] for d in accepted
                         if d['node'] == i and d['start'] <= t < d['end'])
                     for k in range(3)] for i in range(len(nodes))]
            for k in range(3):
                area[k] += sum(v[k] for v in load)
            if not any(all(n.capacity[k]-load[i][k] >= (1, 4, 16)[k] for k in range(3))
                       for i, n in enumerate(nodes)):
                blocked.append(t)
        assert area == work
        event_area = [0, 0, 0]
        for event, end in zip(sim['events'], [e['time'] for e in sim['events'][1:]]+[4]):
            width = max(0, min(end, 4)-min(event['time'], 4))
            for k in range(3):
                event_area[k] += sum(n[k] for n in event['running_load'])*width
        assert event_area == work
        event = next(e for e in sim['events'] if e['time'] == 2)
        assert 'a' in event['completed'] and 'c' in event['accepted']
        assert row['pressure']['T4']['fraction'] == len(blocked)/4
        assert [t for start, end in row['pressure']['T4']['intervals'] for t in range(start, end)] == blocked
        assert row['comparison'] == 'observed simulation comparison'
        assert row['delta_accepted'] == (1 if name == 'Add Node' else 0)
        assert row['delta_value'] == (1 if name == 'Add Node' else 0)
        assert row['certified_batch_count'] == 4 and row['batch_status_counts'] == {'optimal': 4}
        assert row['certification_scope'].startswith('arrival batches only')
    assert raw == before


@pytest.mark.parametrize('mode', ['static', 'dynamic'])
def test_pure_reporting_no_planner_calls_and_determinism(experiments, mode):
    raw = experiments[mode][1]
    with patch.object(p0, 'compare', side_effect=AssertionError('must not solve')), \
         patch.object(p1, 'compare', side_effect=AssertionError('must not simulate')):
        assert p2.summarize_report(raw, mode) == p2.summarize_report(raw, mode)
        json.dumps(p2.summarize_report(raw, mode), allow_nan=False)


@pytest.mark.parametrize('mode', ['static', 'dynamic'])
def test_tie_and_missing_reference_and_zero_total(experiments, mode):
    raw = deepcopy(experiments[mode][1])
    raw['rows'][0] = dict(deepcopy(raw['rows'][1]), scenario='Add Node')
    for r in raw['rows']:
        r['utilization'] = [None, .5, .5]
        if mode == 'dynamic':
            r['pressure'] = {}
    result = p2.summarize_report(raw, mode)
    assert result['leader'] is None and result['tied_names'] == ['Add Node', 'Baseline', 'No change']
    assert all(r['utilization'] == [None, .5, .5] for r in result['scenarios'])
    if mode == 'dynamic':
        assert all(r['pressure'] == {} and r['pressure_status'] == 'UNAVAILABLE' for r in result['scenarios'])


def test_static_uncertified_and_no_incumbent(experiments):
    raw = deepcopy(experiments['static'][1])
    raw['rows'][0].update(status='feasible_uncertified', certified=False, solver_optimal=False)
    result = p2.summarize_report(raw, 'static')
    assert result['comparison_status'] == 'COMPARABLE' and result['leader'] == 'Add Node'
    assert all(r['comparison'] == 'observed static candidate comparison' for r in result['scenarios'])
    raw['rows'][0].update(status='uncertified', valid=False, accepted_count=None, accepted_value=None)
    result = p2.summarize_report(raw, 'static')
    assert result['comparison_status'] == 'NOT COMPARABLE' and result['leader'] is None
    assert not result['tied_names']
    assert result['scenarios'][0]['unplaced_ids'] is None
    assert all(r['delta_accepted'] is None and r['delta_value'] is None for r in result['scenarios'])


def test_dynamic_solver_error_suppresses_every_delta(experiments):
    raw = deepcopy(experiments['dynamic'][1])
    raw['rows'][0]['solver_error_count'] = 1
    result = p2.summarize_report(raw, 'dynamic')
    assert result['comparison_status'] == 'NOT COMPARABLE' and result['leader'] is None
    assert all(r['delta_accepted'] is None and r['delta_value'] is None for r in result['scenarios'])
    assert result['scenarios'][0]['accepted_count'] is None


@pytest.mark.parametrize('bad', [None, {}, {'rows': []}, {'rows': [None]},
                                {'rows': [{'scenario': 'other'}]}])
def test_malformed_reports_fail_clearly(bad):
    with pytest.raises(ValueError):
        p2.summarize_report(bad, 'static')


@pytest.mark.parametrize('field,value', [('accepted_count', None), ('accepted_count', True),
                                         ('accepted_value', float('nan')),
                                         ('utilization', [float('inf'), 0, 0])])
def test_invalid_counts_and_nonfinite(experiments, field, value):
    raw = deepcopy(experiments['static'][1]); raw['rows'][0][field] = value
    with pytest.raises(ValueError):
        p2.summarize_report(raw, 'static')


@pytest.mark.parametrize('mode', ['static', 'dynamic'])
def test_cli_boundary_one_parse_one_compare_no_extra_solves(experiments, mode, capsys):
    config, raw, _ = experiments[mode]
    planner = p0 if mode == 'static' else p1
    with patch.object(p2, 'load_input', return_value=config) as parse, \
         patch.object(planner, 'compare', return_value=raw) as compare:
        assert p2.main(['--input', 'fixture.json', '--mode', mode, '--format', 'json']) == 0
        parse.assert_called_once_with('fixture.json', mode)
        compare.assert_called_once_with(config)
        assert json.loads(capsys.readouterr().out)['mode'] == mode
    # Old CLIs retain their own existing parse/compare boundary and output.
    with patch('examples.planner_input.load_input', return_value=config) as parse, \
         patch.object(planner, 'compare', return_value=raw) as compare:
        assert planner.main(['--input', 'fixture.json']) == 0
        parse.assert_called_once_with('fixture.json', mode)
        compare.assert_called_once_with(config)
        assert 'USER_INPUT' in capsys.readouterr().out


def test_input_and_report_errors_and_text(experiments, capsys):
    from examples.planner_input import InputError
    with patch.object(p2, 'load_input', side_effect=InputError('nodes: required')), \
         patch.object(p0, 'compare') as compare:
        assert p2.main(['--input', 'bad.json', '--mode', 'static']) == 2
        assert 'INPUT_INVALID: nodes: required' in capsys.readouterr().err
        compare.assert_not_called()
    config, raw, _ = experiments['static']
    with patch.object(p2, 'load_input', return_value=config), patch.object(p0, 'compare', return_value=raw):
        assert p2.main(['--input', 'fixture.json', '--mode', 'static']) == 0
        assert 'Largest accepted count: Add Node' in capsys.readouterr().out
    with patch.object(p2, 'load_input', return_value=config), patch.object(p0, 'compare', return_value={'rows': [None]}):
        assert p2.main(['--input', 'fixture.json', '--mode', 'static']) == 1
        assert 'REPORT_INVALID' in capsys.readouterr().err

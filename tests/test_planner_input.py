"""Shared-input contracts and independent tiny P0/P1 ledgers; no old sweep."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
import pytest
from examples import capacity_whatif as static
from examples import time_window_capacity as dynamic
from examples.planner_input import InputError, load_input


def document(mode):
    doc = dict(schema_version=1, mode=mode,
               nodes=[n._asdict() for n in static.BASELINE],
               scenarios=[dict(name=name, add_nodes=[n._asdict() for n in nodes[2:]])
                          for name, nodes in static.SCENARIOS])
    if mode == 'static':
        doc['jobs'] = [j._asdict() for j in static.TASKS]
    else:
        doc.update(horizon=6, jobs=[dict(**asdict(j), family=f)
                                   for j, f in zip(dynamic.JOBS, dynamic.REQUESTS)],
                   pressure_references=[dict(family=f, demand=d) for f, d in dynamic.REFERENCES.items()])
    return json.loads(json.dumps(doc))


def write(path, doc):
    path.write_text(json.dumps(doc, indent=2)+'\n', encoding='utf-8')
    return path


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp('inputs')
    configs, results, calls = {}, {}, {'static': [], 'dynamic': []}
    original_static, original_dynamic = static.solve, dynamic.simulate
    globals_before = deepcopy((static.TASKS, static.SCENARIOS, dynamic.JOBS,
                               dynamic.REQUESTS, dynamic.REFERENCES))
    def solve(a, **kw):
        before = deepcopy(a)
        out = original_static(a, **kw)
        for field, value in vars(before).items():
            np.testing.assert_array_equal(getattr(a, field), value)
        calls['static'].append((a, out, kw))
        return out
    def simulate(capacity, jobs, eligible, **kw):
        before = deepcopy((capacity, jobs, eligible))
        out = original_dynamic(capacity, jobs, eligible, **kw)
        assert capacity == before[0] and jobs == before[1]
        np.testing.assert_array_equal(eligible, before[2])
        calls['dynamic'].append((capacity, jobs, eligible, out, kw))
        return out
    with patch.object(static, 'solve', side_effect=solve), patch.object(dynamic, 'simulate', side_effect=simulate):
        for mode, planner in [('static', static), ('dynamic', dynamic)]:
            path = write(root/(mode+'.json'), document(mode))
            config = load_input(path, mode)
            before = deepcopy(config)
            results[mode] = (planner.compare(), planner.compare(config))
            assert config == before
            configs[mode] = config
    assert (static.TASKS, static.SCENARIOS, dynamic.JOBS, dynamic.REQUESTS, dynamic.REFERENCES) == globals_before
    assert len(calls['static']) == len(calls['dynamic']) == 6
    return configs, results, calls, root


def test_static_external_matches_default_and_raw_ledger(experiment):
    configs, results, calls, _ = experiment
    config = configs['static']
    defaults, external = results['static']
    assert [r['accepted_count'] for r in external['rows']] == [3, 5, 4]
    for k, ((_, nodes), row, old) in enumerate(zip(config['scenarios'], external['rows'], defaults['rows'])):
        a, out, kw = calls['static'][k+3]
        assert kw == dict(method='highs_mip', seed=11, time_limit=.5)
        assert a.demand.tolist() == [list(j.demand) for j in config['jobs']]
        assert a.eligible.tolist() == [[n.family == j.family for j in config['jobs']] for n in nodes]
        for i, n in enumerate(nodes):
            for h, m in enumerate(nodes):
                assert (a.shapes[i] == a.shapes[h]) == (n.capacity == m.capacity)
        loads = [[0, 0, 0] for n in nodes]
        accepted, rejected, value = [], [], 0
        for j, i in zip(config['jobs'], row['placement']):
            if i < 0:
                rejected.append(j.id)
            else:
                assert nodes[i].family == j.family
                loads[i] = [x+y for x, y in zip(loads[i], j.demand)]
                accepted.append(j.id)
                value += j.value
        capacity = np.array([n.capacity for n in nodes])
        residual = capacity-np.array(loads)
        assert np.all(residual >= 0)
        np.testing.assert_array_equal(row['per_node_residual'], residual)
        np.testing.assert_array_equal(row['total_residual'], residual.sum(axis=0))
        np.testing.assert_array_equal(row['utilization'], np.array(loads).sum(axis=0)/capacity.sum(axis=0))
        assert row['accepted_ids'] == accepted and row['unplaced_ids'] == rejected
        assert 'p100-missing' in rejected and row['accepted_value'] == value
        assert row['certified'] == (out.status == 'optimal' and out.optimal and row['valid'])
        assert row['certified'] and row['difference_kind'] == 'static-model optimal difference'
        assert row['accepted_count'] == old['accepted_count'] and row['delta_count'] == old['delta_count']


def test_dynamic_external_window_event_pressure_oracle(experiment):
    configs, results, calls, _ = experiment
    config = configs['dynamic']; horizon = config['horizon']
    defaults, external = results['dynamic']
    assert [r['accepted_count'] for r in external['rows']] == [3, 5, 5]
    for k, ((_, nodes), row, old) in enumerate(zip(config['scenarios'], external['rows'], defaults['rows'])):
        capacity, jobs, eligible, raw, kw = calls['dynamic'][k+3]
        assert jobs is config['jobs'] and kw == dict(method='highs_mip', seed=11, time_limit=.5)
        assert eligible.tolist() == [[n.family == f for f in config['requests']] for n in nodes]
        cap = np.array(capacity)
        lookup = {j.id: (j, f) for j, f in zip(jobs, config['requests'])}
        accepted = [d for d in raw['decisions'] if d['status'] == 'accepted']
        window, full, admitted = np.zeros(3), np.zeros(3), np.zeros(3)
        def load(t):
            result = np.zeros_like(cap)
            for d in accepted:
                if d['start'] <= t < d['end']:
                    result[d['node']] += lookup[d['job_id']][0].demand
            return result
        for d in accepted:
            j, f = lookup[d['job_id']]
            assert nodes[d['node']].family == f and d['start'] == j.arrival
            assert d['end'] == j.arrival+j.duration
            window += np.array(j.demand)*max(0, min(d['end'], horizon)-d['start'])
            full += np.array(j.demand)*j.duration
            admitted += j.demand
        event_window, event_full, previous, previous_load = np.zeros(3), np.zeros(3), 0, np.zeros_like(cap)
        completions = []
        for e in raw['events']:
            event_window += previous_load.sum(axis=0)*max(0, min(e['time'], horizon)-min(previous, horizon))
            event_full += previous_load.sum(axis=0)*(e['time']-previous)
            expected = load(e['time'])
            np.testing.assert_array_equal(e['free'], cap-expected)
            np.testing.assert_array_equal(e['running_load'], expected)
            assert np.all(expected <= cap)
            completions += e['completed']
            previous, previous_load = e['time'], np.array(e['running_load'])
        event_window += previous_load.sum(axis=0)*max(0, horizon-previous)
        np.testing.assert_array_equal(window, event_window)
        np.testing.assert_array_equal(row['window_resource_time'], window)
        np.testing.assert_array_equal(row['utilization'], window/(horizon*cap.sum(axis=0)))
        np.testing.assert_array_equal(raw['resource_time'], full)
        np.testing.assert_array_equal(event_full, full)
        np.testing.assert_array_equal(raw['admitted_resources'], admitted)
        np.testing.assert_array_equal(raw['released_resources'], admitted)
        np.testing.assert_array_equal(raw['final_free'], cap)
        assert len(completions) == len(set(completions)) == len(accepted)
        assert row['completed_by_T'] == [d['job_id'] for d in accepted if d['end'] <= horizon]
        assert row['running_at_T'] == [d['job_id'] for d in accepted if d['end'] > horizon]
        assert 'P-missing' in row['rejected_ids'] and row['solver_error_ids'] == []
        for family, demand in config['references'].items():
            blocked = sum(not any(n.family == family and np.all((cap-load(t))[i] >= demand)
                                  for i, n in enumerate(nodes)) for t in range(horizon))
            assert row['pressure'][family]['fraction'] == blocked/horizon
        assert row['certified_batch_count'] == sum(b['optimal'] for b in raw['batches']) == 5
        assert row['difference_kind'] == 'observed simulation difference'
        for key in ('accepted_count', 'completed_by_T', 'running_at_T', 'window_resource_time',
                    'full_tail_resource_time', 'utilization', 'pressure', 'delta_admitted'):
            assert row[key] == old[key]


def test_provenance_and_order(experiment):
    configs, results, _, root = experiment
    for mode in ('static', 'dynamic'):
        cfg = configs[mode]; info = results[mode][1]['input']
        assert info['schema_version'] == 1 and info['mode'] == mode
        assert info['raw_input_sha256'] == hashlib.sha256((root/(mode+'.json')).read_bytes()).hexdigest()
        assert info['input_order']['jobs'] == [j.id for j in cfg['jobs']]
        assert info['input_order']['scenarios'] == [n for n, _ in cfg['scenarios']]
        assert info['solver'] == cfg['solver']
        assert info['repo_commit_sha'] is None or re.fullmatch('[0-9a-f]{40}', info['repo_commit_sha'])
        json.dumps(results[mode][1], allow_nan=False)


def set_field(doc, path, value):
    parts = re.findall(r'[^.\[\]]+', path)
    target = doc
    for key in parts[:-1]:
        target = target[int(key)] if isinstance(target, list) else target[key]
    key = int(parts[-1]) if isinstance(target, list) else parts[-1]
    target[key] = value


INVALID = [
    ('static', 'schema_version', True), ('static', 'schema_version', 1.0),
    ('static', 'schema_version', 2), ('static', 'mode', 'unknown'),
    ('static', 'extra', 1), ('static', 'nodes', []), ('static', 'jobs', []),
    ('static', 'nodes[0].id', ' '), ('static', 'nodes[1].id', 't4-base'),
    ('static', 'nodes[0].family', ''), ('static', 'nodes[0].capacity', [1, 2]),
    ('static', 'nodes[0].capacity[0]', .5), ('static', 'nodes[0].capacity[0]', True),
    ('static', 'nodes[0].capacity[1]', -1), ('static', 'nodes[0].capacity[2]', float('inf')),
    ('static', 'jobs[1].id', 't-small-1'), ('static', 'jobs[0].id', ''),
    ('static', 'jobs[0].family', ' '), ('static', 'jobs[0].demand', [0, 0, 0]),
    ('static', 'jobs[0].demand[2]', 1_000_001), ('static', 'jobs[0].value', -1),
    ('static', 'jobs[0].value', False), ('static', 'jobs[0].value', float('nan')),
    ('static', 'jobs[0].arrival', 0), ('static', 'horizon', 6),
    ('static', 'pressure_references', []), ('static', 'solver', []),
    ('static', 'solver.method', 'pgfp'), ('static', 'solver.seed', True),
    ('static', 'solver.seed', -1), ('static', 'solver.time_limit', 0),
    ('static', 'solver.time_limit', .6), ('static', 'solver.time_limit', False),
    ('static', 'solver.extra', 1), ('static', 'scenarios', []),
    ('static', 'scenarios[1].name', 'Baseline'), ('static', 'scenarios[0].name', 'base'),
    ('static', 'scenarios[1].add_nodes[0].id', 't4-base'),
    ('dynamic', 'horizon', True), ('dynamic', 'horizon', 0),
    ('dynamic', 'horizon', 1_000_001), ('dynamic', 'jobs[2].arrival', 6),
    ('dynamic', 'jobs[2].arrival', -1), ('dynamic', 'jobs[2].arrival', True),
    ('dynamic', 'jobs[2].duration', 0), ('dynamic', 'jobs[2].duration', 1.5),
    ('dynamic', 'jobs[2].duration', True), ('dynamic', 'jobs[2].extra', 1),
    ('dynamic', 'pressure_references', None),
    ('dynamic', 'pressure_references[1].family', 'T4'),
    ('dynamic', 'pressure_references[0].demand', [0, 0, 0]),
]


@pytest.mark.parametrize('mode,field,value', INVALID)
def test_schema_errors_before_solver(tmp_path, monkeypatch, capsys, mode, field, value):
    doc = document(mode)
    if field.startswith('solver.'):
        doc['solver'] = {}
    set_field(doc, field, value)
    path = write(tmp_path/'bad.json', doc)
    def forbidden(*a, **kw):
        pytest.fail('schema error reached a solver')
    monkeypatch.setattr(static, 'solve', forbidden)
    monkeypatch.setattr(dynamic, 'simulate', forbidden)
    planner = static if mode == 'static' else dynamic
    assert planner.main(['--input', str(path)]) == 2
    error = capsys.readouterr().err
    assert 'INPUT_INVALID:' in error
    # Missing Baseline and non-finite literals may be diagnosed at their parent.
    assert field in error or (field.startswith('scenarios') and 'scenarios:' in error)


@pytest.mark.parametrize('raw,field', [('{', '$'),
    ('{"schema_version":1,"schema_version":1}', 'schema_version'),
    ('{"jobs":[{"duration":1,"duration":2}]}', 'jobs[0].duration'),
    ('{"jobs":[{"value":NaN}]}', 'jobs[0].value'),
    ('{"jobs":[{"value":Infinity}]}', 'jobs[0].value'),
    ('{"jobs":[{"value":1e999}]}', 'jobs[0].value')])
def test_strict_json(tmp_path, raw, field):
    path = tmp_path/'bad.json';path.write_text(raw)
    with pytest.raises(InputError) as error:
        load_input(path, 'static')
    assert field in str(error.value)


def test_bounds_required_fields_and_baseline_rules(tmp_path):
    docs = []
    d = document('static');d['scenarios'][0]['add_nodes'] = d['scenarios'][1]['add_nodes'];docs.append((d, 'scenarios[0].add_nodes'))
    d = document('static');d['nodes'] *= 17;docs.append((d, 'nodes'))
    d = document('static');d['jobs'] *= 9;docs.append((d, 'jobs'))
    d = document('static');d['scenarios'] *= 3;docs.append((d, 'scenarios'))
    d = document('dynamic');d['pressure_references'] *= 6;docs.append((d, 'pressure_references'))
    d = document('dynamic');del d['jobs'][2]['duration'];docs.append((d, 'jobs[2].duration'))
    for d, field in docs:
        with pytest.raises(InputError) as error:
            load_input(write(tmp_path/'bad.json', d), d['mode'])
        assert field in str(error.value)
    path = tmp_path/'large.json';path.write_bytes(b' '*(1_048_576+1))
    with pytest.raises(InputError, match='1 MiB'):
        load_input(path, 'static')


def test_mode_mismatch_missing_file_and_unknown_commit(tmp_path, monkeypatch):
    path = write(tmp_path/'static.json', document('static'))
    with pytest.raises(InputError, match='mode:'):
        load_input(path, 'dynamic')
    with pytest.raises(InputError, match='input:'):
        load_input(tmp_path/'missing.json', 'static')
    from examples import planner_input
    def unavailable(*a, **kw):
        raise FileNotFoundError('git unavailable')
    monkeypatch.setattr(planner_input.subprocess, 'run', unavailable)
    cfg = load_input(path, 'static')
    assert cfg['provenance']['repo_commit_sha'] is None and cfg['provenance']['repo_dirty'] is None


def test_empty_pressure_references_and_reordered_extra_scenarios(experiment, tmp_path, monkeypatch):
    _, _, calls, _ = experiment
    for present in (False, True):
        doc = document('dynamic')
        if present:
            doc['pressure_references'] = []
        else:
            del doc['pressure_references']
        cfg = load_input(write(tmp_path/'input.json', doc), 'dynamic')
        stream = iter(c[3] for c in calls['dynamic'][3:])
        monkeypatch.setattr(dynamic, 'simulate', lambda *a, **kw: deepcopy(next(stream)))
        assert all(r['pressure'] == {} for r in dynamic.compare(cfg)['rows'])
    doc = document('static')
    doc['scenarios'] = [doc['scenarios'][1], doc['scenarios'][0], doc['scenarios'][2],
                        dict(name='No change', add_nodes=[])]
    cfg = load_input(write(tmp_path/'order.json', doc), 'static')
    out = [calls['static'][i][1] for i in (4, 3, 5, 3)]
    stream = iter(out)
    monkeypatch.setattr(static, 'solve', lambda *a, **kw: next(stream))
    result = static.compare(cfg)
    assert [r['scenario'] for r in result['rows']] == ['Add T4', 'Baseline', 'Add V100', 'No change']
    assert [r['delta_count'] for r in result['rows']] == [2, 0, 1, 0]


def test_no_argument_entries_and_failure_certificate_guards(experiment, monkeypatch, capsys):
    configs, results, calls, _ = experiment
    for mode, planner in [('static', static), ('dynamic', dynamic)]:
        with patch.object(planner, 'compare', return_value=results[mode][0]) as mock:
            assert planner.main([]) == 0
            mock.assert_called_once_with()
            assert capsys.readouterr().out.startswith(results[mode][0]['label'])
    from dataclasses import replace
    outputs = [c[1] for c in calls['static'][3:]]
    outputs[0] = replace(outputs[0], status='feasible_uncertified', optimal=False)
    stream = iter(outputs);monkeypatch.setattr(static, 'solve', lambda *a, **kw: next(stream))
    row = static.compare(configs['static'])['rows'][0]
    assert row['valid'] and not row['certified'] and row['difference_kind'] == 'observed difference (uncertified)'
    outputs[0] = replace(outputs[0], status='uncertified', placement=None, value=None, residual=None)
    stream = iter(outputs)
    row = static.compare(configs['static'])['rows'][0]
    assert row['unplaced_ids'] is None and row['delta_count'] is None and row['comparison'] == 'NOT COMPARABLE'
    reports = deepcopy([c[3] for c in calls['dynamic'][3:]])
    b = next(b for b in reports[0]['batches'] if b['time'] == 3)
    b.update(solver_status='feasible_uncertified', optimal=False, reported_optimal=False, certified_value=None)
    next(d for d in reports[0]['decisions'] if d['job_id'] == 'P-missing')['solver_status'] = 'feasible_uncertified'
    stream = iter(reports);monkeypatch.setattr(dynamic, 'simulate', lambda *a, **kw: next(stream))
    rows = dynamic.compare(configs['dynamic'])['rows']
    assert rows[0]['certified_batch_count'] == 4 and rows[0]['batch_status_counts']['feasible_uncertified'] == 1
    assert all(r['comparison'] == 'COMPARABLE' and r['difference_kind'] == 'observed simulation difference' for r in rows)
    reports = deepcopy([c[3] for c in calls['dynamic'][3:]])
    raw = reports[2];d = next(d for d in raw['decisions'] if d['job_id'] == 'P-missing')
    d.update(status='solver_error', solver_status='uncertified')
    raw['solver_error_count'] = 1;raw['rejected_count'] -= 1
    b = next(b for b in raw['batches'] if b['time'] == 3)
    b.update(solver_status='uncertified', optimal=False, reported_optimal=False, placement=None,
             certified_value=None, accepted_value=None, validation_status='not_run')
    e = next(e for e in raw['events'] if e['time'] == 3);e.update(rejected=[], solver_errors=['P-missing'])
    stream = iter(reports);monkeypatch.setattr(dynamic, 'simulate', lambda *a, **kw: next(stream))
    rows = dynamic.compare(configs['dynamic'])['rows']
    assert rows[2]['solver_error_ids'] == ['P-missing'] and 'P-missing' not in rows[2]['rejected_ids']
    assert rows[2]['certified_batch_count'] == 4
    assert all(r['delta_admitted'] is None and r['comparison'] == 'NOT COMPARABLE' for r in rows)


def test_zero_total_guard_no_implicit_family_substitution(tmp_path, monkeypatch):
    doc = document('static')
    doc['nodes'] = [dict(id='cpu', family='CPU', capacity=[0, 4, 16])]
    doc['jobs'] = [dict(id='cpu-job', family='CPU', demand=[0, 2, 8]),
                   dict(id='missing-gpu', family='T4', demand=[1, 2, 8])]
    doc['scenarios'] = [dict(name='Baseline', add_nodes=[]), dict(name='No change', add_nodes=[])]
    cfg = load_input(write(tmp_path/'zero.json', doc), 'static')
    from accelsched import SolveResult
    def valid(a, **kw):
        assert a.eligible.tolist() == [[True, False]]
        return SolveResult('highs_mip', 'optimal', (0, -1), 1., np.array([[0., 2., 8.]]), 0., True)
    monkeypatch.setattr(static, 'solve', valid)
    row = static.compare(cfg)['rows'][0]
    assert row['utilization'] == [None, .5, .5] and row['unplaced_ids'] == ['missing-gpu']


def test_documented_json_and_pressure_boundary(tmp_path):
    text = (Path(__file__).parent/'docs/examples/planner-input.md').read_text(encoding='utf-8')
    doc = json.loads(re.search(r'```json\n(.*?)\n```', text, re.S).group(1))
    cfg = load_input(write(tmp_path/'documented.json', doc), 'dynamic')
    assert len(cfg['scenarios']) == 2 and cfg['references'] == {'T4': (2, 8, 32)}
    doc['pressure_references'] = [dict(family=f'f{i}', demand=[1, 1, 1]) for i in range(16)]
    assert len(load_input(write(tmp_path/'sixteen.json', doc), 'dynamic')['references']) == 16

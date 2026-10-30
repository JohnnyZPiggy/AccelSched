"""Independently reconstruct exported reports, including releases before arrivals."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from tools.export_3d_replay import export


def check(bundle):
    cap = np.array([n['capacity'] for n in bundle['nodes']], float)
    jobs = {j['id']: j for j in bundle['jobs']}
    order = list(jobs)
    for report in bundle['reports'].values():
        decisions = {d['job_id']: d for d in report['decisions']}
        assert len(decisions) == len(jobs) == len(report['decisions'])
        running, counts = {}, dict(accepted=0, not_scheduled=0, solver_error=0)
        times = sorted({j['arrival'] for j in jobs.values()} | {d['end'] for d in decisions.values() if d['end'] is not None})
        assert [e['time'] for e in report['events']] == times
        completed_count = 0
        for event in report['events']:
            t = event['time']
            completed = [j for j in order if j in running and decisions[j]['end'] == t]
            assert event['completed'] == completed
            completed_count += len(completed)
            for j in completed:
                del running[j]

            def load():
                used = np.zeros_like(cap)
                for j, node in running.items():
                    used[node] += jobs[j]['demand']
                return used

            np.testing.assert_allclose(event['free_before_arrivals'], cap-load(), atol=1e-9, rtol=0)
            groups = dict(accepted=[], not_scheduled=[], solver_error=[])
            arrivals = [j for j in order if jobs[j]['arrival'] == t]
            batch = report['batches'][event['batch_index']] if arrivals else None
            assert (event['batch_index'] is None) == (not arrivals)
            if batch:
                assert batch['job_ids'] == arrivals and batch['time'] == t
            for index, j in enumerate(arrivals):
                d, job = decisions[j], jobs[j]
                assert d['status'] in groups
                assert d['arrival'] == t and d['solver_status'] == batch['solver_status']
                counts[d['status']] += 1
                groups[d['status']].append(j)
                if d['status'] == 'accepted':
                    assert d['start'] == t and d['end'] == t + job['duration']
                    assert bundle['eligible'][d['node']][order.index(j)]
                    assert batch['placement'][index] == d['node']
                    running[j] = d['node']
                else:
                    assert d['node'] is d['start'] is d['end'] is None
                    assert (batch['placement'] is None) == (d['status'] == 'solver_error')
                    if batch['placement'] is not None:
                        assert batch['placement'][index] == -1
            assert groups['accepted'] == event['accepted']
            assert groups['not_scheduled'] == event['rejected']
            assert groups['solver_error'] == event['solver_errors']
            assert event['running'] == [dict(job_id=j, node=running[j], end=decisions[j]['end']) for j in order if j in running]
            used = load()
            assert np.all(used <= cap + 1e-9)
            np.testing.assert_allclose(event['running_load'], used, atol=1e-9, rtol=0)
            np.testing.assert_allclose(np.array(event['free'])+used, cap, atol=1e-9, rtol=0)
        assert not running
        assert counts == dict(accepted=report['accepted_count'], not_scheduled=report['rejected_count'], solver_error=report['solver_error_count'])
        assert completed_count == report['completed_count'] == counts['accepted']


def test_export_matches_engine_and_checked_in_data():
    fresh = json.loads(json.dumps(export()))
    saved = json.loads((Path(__file__).resolve().parents[1] / 'demos/accelsched-3d/public/data/replay.json').read_text())
    check(fresh)
    check(saved)
    assert fresh['nodes'] == saved['nodes'] and fresh['jobs'] == saved['jobs']
    for method in fresh['reports']:
        for field in ('events', 'decisions', 'accepted_count', 'rejected_count', 'solver_error_count'):
            assert fresh['reports'][method][field] == saved['reports'][method][field]
        for actual, expected in zip(fresh['reports'][method]['batches'], saved['reports'][method]['batches']):
            assert {k: v for k, v in actual.items() if k != 'solve_wall_ms'} == {k: v for k, v in expected.items() if k != 'solve_wall_ms'}


def test_checker_detects_corrupted_ledger():
    bundle = export()
    broken = copy.deepcopy(bundle)
    broken['reports']['first_fit']['events'][0]['free'][0][0] += 1
    with pytest.raises(AssertionError):
        check(broken)


def test_solver_errors_remain_distinct(monkeypatch):
    import rolling
    from types import SimpleNamespace
    monkeypatch.setattr(rolling, 'solve', lambda *a, **kw: SimpleNamespace(
        status='exception', error='test-only injected solver failure', optimal=False,
        placement=None))
    bundle = export()
    check(bundle)
    for report in bundle['reports'].values():
        assert report['solver_error_count'] == 32
        assert report['accepted_count'] == report['rejected_count'] == 0
        assert all(d['node'] is None for d in report['decisions'])
        assert all(b['error'] == 'test-only injected solver failure' for b in report['batches'])

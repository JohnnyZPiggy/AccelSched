"""Independent decision/interval oracle; no simulator or solver in unit tests."""
from collections import Counter
from copy import deepcopy

import numpy as np


def audit(capacity, families, jobs, requests, eligible, report, omega, T, method):
    cap = np.array(capacity, float)
    index = {j.id: k for k, j in enumerate(jobs)}
    assert len(index) == len(jobs)
    ds = report['decisions']
    assert len(ds) == len(jobs) and {d['job_id'] for d in ds} == set(index)
    by_id = {d['job_id']: d for d in ds}
    accepted = [d for d in ds if d['status'] == 'accepted']

    def equal(a, b):
        a, b = np.asarray(a), np.asarray(b)
        # Exact integer accounting for these fixtures (including CPU/RAM).
        # Fractional resource fixtures use 1e-9 absolute, zero relative tolerance.
        if np.all(b == np.floor(b)):
            assert np.array_equal(a, b), (a, b)
        else:
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=0)

    def load(active):
        result = np.zeros_like(cap)
        for d in active:
            result[d['node']] += jobs[index[d['job_id']]].demand
        return result

    full, clipped, admitted = np.zeros(3), np.zeros(3), np.zeros(3)
    for d in ds:
        k = index[d['job_id']]
        j = jobs[k]
        assert j.demand[0] in (1, 2, 4) and j.value == 1
        assert d['arrival'] == j.arrival
        if d['status'] == 'accepted':
            assert isinstance(d['node'], int) and 0 <= d['node'] < len(cap)
            assert eligible[d['node'], k]
            assert families[d['node']] == requests[k]['family']
            assert d['start'] == j.arrival and d['end'] == j.arrival+j.duration
            v = np.array(j.demand)
            admitted += v
            full += v*(d['end']-d['start'])
            clipped += v*max(0, min(d['end'], T)-max(d['start'], 0))
        else:
            assert d['status'] in ('not_scheduled', 'solver_error')
            assert d['node'] is None and d['start'] is None and d['end'] is None
    counts = Counter(d['status'] for d in ds)
    assert counts['accepted'] == report['accepted_count'] == report['completed_count']
    assert counts['not_scheduled'] == report['rejected_count']
    assert counts['solver_error'] == report['solver_error_count']
    equal(report['accepted_value'], counts['accepted'])
    equal(report['admitted_resources'], admitted)
    equal(report['released_resources'], admitted)
    equal(report['resource_time'], full)
    equal(report['final_free'], cap)

    events = report['events']
    times = sorted({j.arrival for j in jobs} | {d['end'] for d in accepted})
    assert [e['time'] for e in events] == times
    assert [b['time'] for b in report['batches']] == sorted({j.arrival for j in jobs})
    seen, interval, full_interval, placeability = [], np.zeros(3), np.zeros(3), 0.
    cumulative = []
    previous_time, previous_load, previous_free = 0, np.zeros_like(cap), cap.copy()

    def F(free):
        return sum(int(families[i] == family and np.all(free[i] >= demand))
                   for i in range(len(cap)) for family, demand in omega)

    for event in events:
        t = event['time']
        dt = max(0, min(t, T)-min(previous_time, T))
        interval += previous_load.sum(axis=0)*dt
        full_interval += previous_load.sum(axis=0)*(t-previous_time)
        placeability += F(previous_free)*dt
        active = [d for d in accepted if d['start'] <= t < d['end']]
        current_load = load(active)
        current_free = cap-current_load
        assert np.all(current_free >= 0)
        equal(event['running_load'], current_load)
        equal(event['free'], current_free)
        equal(np.array(event['free'])+np.array(event['running_load']), cap)
        before = load([d for d in accepted if d['start'] < t < d['end']])
        equal(event['free_before_arrivals'], cap-before)
        completed = [d['job_id'] for d in accepted if d['end'] == t]
        assert sorted(event['completed']) == sorted(completed)
        seen.extend(event['completed'])
        expected_running = {d['job_id']: (d['node'], d['end']) for d in active}
        assert len(event['running']) == len(expected_running)
        assert {r['job_id']: (r['node'], r['end']) for r in event['running']} == expected_running
        for field, status in [('accepted', 'accepted'), ('rejected', 'not_scheduled'),
                              ('solver_errors', 'solver_error')]:
            assert sorted(event[field]) == sorted(d['job_id'] for d in ds
                                                 if d['arrival'] == t and d['status'] == status)
        arrivals = [j.id for j in jobs if j.arrival == t]
        if arrivals:
            b = report['batches'][event['batch_index']]
            assert b['time'] == t and b['job_ids'] == arrivals and b['method'] == method
            assert b['solve_wall_ms'] >= 0 and np.isfinite(b['solve_wall_ms'])
            good = b['placement'] is not None
            assert good == (not any(by_id[j]['status'] == 'solver_error' for j in arrivals))
            expected_optimal = (good and method == 'highs_mip' and
                                b['solver_status'] == 'optimal' and b['reported_optimal'])
            assert b['optimal'] == expected_optimal
            value = sum(by_id[j]['status'] == 'accepted' for j in arrivals)
            assert b['certified_value'] == (value if expected_optimal else None)
            if good:
                assert b['validation_status'] == 'valid'
                assert b['solver_status'] in ('feasible', 'feasible_uncertified', 'optimal')
                assert b['placement'] == [by_id[j]['node'] if by_id[j]['status'] == 'accepted'
                                          else -1 for j in arrivals]
                assert b['accepted_value'] == value
            for j in arrivals:
                assert by_id[j]['solver_status'] == b['solver_status']
        else:
            assert event['batch_index'] is None
        cumulative.append(sum(d['start'] <= t for d in accepted))
        previous_time = t
        previous_load, previous_free = np.array(event['running_load']), np.array(event['free'])
    dt = max(0, T-previous_time)
    interval += previous_load.sum(axis=0)*dt
    placeability += F(previous_free)*dt
    assert len(seen) == len(set(seen)) == len(accepted)
    equal(interval, clipped)
    equal(full_interval, full)
    equal(report['solve_wall_ms'], sum(b['solve_wall_ms'] for b in report['batches']))
    utilization = clipped/(T*cap.sum(axis=0))
    assert np.all((utilization >= 0) & (utilization <= 1))
    return dict(oracle='PASS', comparison='NOT COMPARABLE' if counts['solver_error'] else 'COMPARABLE',
                accepted=counts['accepted'], accepted_value=report['accepted_value'],
                not_scheduled=counts['not_scheduled'], solver_error=counts['solver_error'],
                admission_rate=counts['accepted']/len(jobs) if jobs else 0.,
                utilization=utilization.tolist(), window_resource_time=clipped.tolist(),
                tail_resource_time=full.tolist(), mean_placeability=placeability/T,
                cumulative_accepted=cumulative,
                batch_statuses=dict(Counter(b['solver_status'] for b in report['batches'])),
                optimal_batches=sum(b['optimal'] for b in report['batches']),
                solve_wall_ms=report['solve_wall_ms'])


def fixture():
    # Hand ledger: [0,2) empty; [2,4) A; [4,6) C. Window ends at 5.
    from rolling import Job
    jobs = [Job('A', 2, 2, (1, 4, 16)), Job('B', 3, 1, (1, 4, 16)),
            Job('C', 4, 2, (1, 4, 16))]
    decisions = [dict(job_id=j.id, arrival=j.arrival, status=s, node=i,
                      start=j.arrival if i is not None else None,
                      end=j.arrival+j.duration if i is not None else None, solver_status='feasible')
                 for j, s, i in zip(jobs, ['accepted', 'not_scheduled', 'accepted'], [0, None, 0])]
    batches, events = [], []
    for t, active, completed, arriving in [(2, [0], [], 0), (3, [0], [], 1),
                                           (4, [2], ['A'], 2), (6, [], ['C'], None)]:
        bindex = None
        if arriving is not None:
            d = decisions[arriving]
            bindex = len(batches)
            batches.append(dict(time=t, job_ids=[d['job_id']], method='first_fit',
                                solver_status='feasible', reported_optimal=False, optimal=False,
                                certified_value=None, accepted_value=int(d['node'] is not None),
                                placement=[-1 if d['node'] is None else 0], validation_status='valid',
                                solve_wall_ms=0.))
        occupied = [1, 4, 16] if active else [0, 0, 0]
        free = [[1-occupied[0], 4-occupied[1], 16-occupied[2]]]
        events.append(dict(time=t, completed=completed, batch_index=bindex,
                           accepted=[decisions[arriving]['job_id']] if arriving in (0, 2) else [],
                           rejected=['B'] if arriving == 1 else [], solver_errors=[],
                           free=free, running_load=[occupied],
                           free_before_arrivals=[[0, 0, 0]] if t == 3 else [[1, 4, 16]],
                           running=[dict(job_id=jobs[k].id, node=0, end=jobs[k].arrival+jobs[k].duration)
                                    for k in active]))
    report = dict(decisions=decisions, events=events, batches=batches, accepted_count=2,
                  completed_count=2, rejected_count=1, solver_error_count=0, accepted_value=2.,
                  admitted_resources=[2, 8, 32], released_resources=[2, 8, 32],
                  resource_time=[4, 16, 64], final_free=[[1, 4, 16]], solve_wall_ms=0.)
    return ([[1, 4, 16]], ['T4'], jobs, [{'family': 'T4'}]*3,
            np.ones((1, 3), bool), report, [('T4', (1, 4, 16))], 5, 'first_fit')


def test_window_tail_and_time_weighted_placeability():
    m = audit(*fixture())
    assert m['utilization'] == [.6, .6, .6]
    assert m['tail_resource_time'] == [4, 16, 64]
    assert m['mean_placeability'] == .4  # Empty [0,2), not equal event weights.
    assert m['not_scheduled'] == 1 and m['solver_error'] == 0


def test_corrupt_ledgers_and_certificates_rejected():
    import pytest
    mutations = [lambda r: r['resource_time'].__setitem__(0, 3),
                 lambda r: r['events'][2]['completed'].clear(),
                 lambda r: r['events'][0]['running'][0].__setitem__('node', 1),
                 lambda r: r['events'][0]['free'][0].__setitem__(0, 1),
                 lambda r: r['batches'][0].__setitem__('optimal', True),
                 lambda r: r.__setitem__('rejected_count', 0),
                 lambda r: r['decisions'][0].__setitem__('end', 5),
                 lambda r: r['events'][0]['free'][0].__setitem__(0, 1e-12)]
    for mutate in mutations:
        args = list(deepcopy(fixture()))
        mutate(args[5])
        with pytest.raises(AssertionError):
            audit(*args)


def test_solver_error_is_not_rejection():
    args = list(fixture())
    r = args[5]
    r['decisions'][1].update(status='solver_error', solver_status='uncertified')
    r['events'][1].update(rejected=[], solver_errors=['B'])
    r['batches'][1].update(solver_status='uncertified', placement=None,
                           validation_status='not_run', accepted_value=None)
    r.update(rejected_count=0, solver_error_count=1)
    m = audit(*args)
    assert m['comparison'] == 'NOT COMPARABLE' and m['not_scheduled'] == 0


def test_fixed_generators_no_solver_calls():
    from bench_dynamic import generate, CAPACITY, NODE_FAMILIES, OMEGA
    for target in (.3, .7, 1.1):
        for seed in (11, 29, 47):
            jobs, e, manifest, info = generate(seed, target)
            assert info == generate(seed, target)[3]
            assert abs(info['offered_load'][0]-target) <= .1
            assert not info['individually_impossible']
            assert all(0 <= j.arrival < 40 and j.duration in (1, 2, 3, 6, 7, 8, 9, 10)
                       and j.demand[0] in (1, 2, 4) and j.demand[1] > 0 and j.demand[2] > 0
                       for j in jobs)
            assert e.tolist() == [[f == j['family'] for j in manifest['jobs']] for f in NODE_FAMILIES]
    assert all(any(f == q and np.all(CAPACITY[i] >= d) for i, f in enumerate(NODE_FAMILIES))
               for q, d in OMEGA)


def test_eligibility_and_family_are_independent_checks():
    import pytest
    args = list(fixture())
    args[4][0, 0] = False
    with pytest.raises(AssertionError):
        audit(*args)
    args = list(fixture())
    args[3] = [{'family': 'P100'}]*3
    with pytest.raises(AssertionError):
        audit(*args)

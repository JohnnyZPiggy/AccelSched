"""Small independent chronology/occupancy and per-batch integer-oracle tests."""
from dataclasses import FrozenInstanceError, replace
from itertools import product

import numpy as np
import pytest
import rolling
from accelsched import METHODS, SolveResult
from rolling import Job, simulate


def check_accounting(capacity, jobs, eligibility, result):
    """Time-stepped oracle from final decisions, not simulator event updates."""
    cap = np.array(capacity, dtype=float)
    lookup = {j.id: (i, j) for i, j in enumerate(jobs)}
    decisions = {d['job_id']: d for d in result['decisions']}
    accepted = [d for d in decisions.values() if d['status'] == 'accepted']
    assert len(accepted) == result['accepted_count'] == result['completed_count']
    assert result['accepted_count']+result['rejected_count']+result['solver_error_count'] == len(jobs)
    times = sorted({j.arrival for j in jobs} | {d['end'] for d in accepted})
    assert [e['time'] for e in result['events']] == times
    seen_completions = []
    for event in result['events']:
        t = event['time']
        active = [d for d in accepted if d['start'] <= t < d['end']]
        load = np.zeros_like(cap)
        for d in active:
            j, job = lookup[d['job_id']]
            assert eligibility[d['node']][j]
            assert d['start'] == job.arrival and d['end'] == job.arrival+job.duration
            load[d['node']] += job.demand
        np.testing.assert_allclose(event['running_load'], load, rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.array(event['free'])+load, cap, rtol=0, atol=1e-9)
        assert np.all(np.array(event['free']) >= -1e-9)
        assert np.all(np.array(event['free']) <= cap+1e-9)
        expected_completed = {d['job_id'] for d in accepted if d['end'] == t}
        assert set(event['completed']) == expected_completed
        seen_completions.extend(event['completed'])
        assert {r['job_id']: r['node'] for r in event['running']} == {d['job_id']: d['node'] for d in active}
        assert not (set(event['rejected']) | set(event['solver_errors'])) & set(r['job_id'] for r in event['running'])
    assert len(seen_completions) == len(set(seen_completions)) == len(accepted)
    integrated = np.zeros(3)
    for t in range(max(times, default=0)):
        for d in accepted:
            if d['start'] <= t < d['end']:
                integrated += lookup[d['job_id']][1].demand
    np.testing.assert_allclose(result['resource_time'], integrated, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result['admitted_resources'], result['released_resources'], rtol=0, atol=1e-9)
    assert result['final_free'] == cap.tolist()


def test_hand_chronology_release_before_arrival_and_end_only():
    cap = [[1, 2, 4]]
    jobs = [Job('A', 0, 2, (1, 2, 4)), Job('B', 1, 2, (1, 2, 4)), Job('C', 2, 3, (1, 2, 4))]
    e = [[True]*3]
    r = simulate(cap, jobs, e)
    assert [d['status'] for d in r['decisions']] == ['accepted', 'not_scheduled', 'accepted']
    assert r['events'][2]['completed'] == ['A'] and r['events'][2]['accepted'] == ['C']
    assert [b['time'] for b in r['batches']] == [0, 1, 2]
    assert r['events'][-1]['time'] == 5 and r['events'][-1]['batch_index'] is None
    assert r['accepted_value'] == 2 and r['resource_time'] == [5, 10, 20]
    check_accounting(cap, jobs, e, r)


def test_simultaneous_one_call_input_order_residual_shapes_and_no_forecast(monkeypatch):
    calls = []
    original = rolling.solve
    def spy(a, **kw):
        calls.append(a)
        assert not a.weights.any() and not a.future_eligible.any()
        for i in range(len(a.capacity)):
            for k in range(len(a.capacity)):
                assert (a.shapes[i] == a.shapes[k]) == np.array_equal(a.capacity[i], a.capacity[k])
        return original(a, **kw)
    monkeypatch.setattr(rolling, 'solve', spy)
    cap = [[2, 4, 8], [2, 4, 8]]
    jobs = [Job('later', 1, 2, (1, 2, 4)), Job('z', 0, 3, (1, 2, 4)), Job('a', 0, 2, (1, 2, 4))]
    r = simulate(cap, jobs, np.ones((2, 3), bool))
    assert len(calls) == 2 and calls[0].demand.shape == (2, 3)
    assert r['batches'][0]['job_ids'] == ['z', 'a']
    assert r['batches'][0]['shapes'] == [0, 0]
    assert len(set(r['batches'][1]['shapes'])) == 2
    check_accounting(cap, jobs, np.ones((2, 3), bool), r)


@pytest.mark.parametrize('method', METHODS)
def test_six_methods_heterogeneous_no_migration(method):
    cap = [[1, 4, 8], [8, 64, 256]]
    jobs = [Job('A', 0, 3, (1, 2, 4)), Job('B', 1, 1, (1, 2, 4)),
            Job('C', 1, 3, (1, 2, 4)), Job('D', 3, 1, (1, 2, 4))]
    eligibility = [[True]*4, [False]*4]  # Abundant but incompatible node.
    r = simulate(cap, jobs, eligibility, method, time_limit=.5)
    assert [d['status'] for d in r['decisions']] == ['accepted', 'not_scheduled', 'not_scheduled', 'accepted']
    assert all(d['node'] in (0, None) for d in r['decisions'])
    assert all(b['optimal'] == (method == 'highs_mip') for b in r['batches'])
    check_accounting(cap, jobs, eligibility, r)
    again = simulate(cap, jobs, eligibility, method, time_limit=.5)
    assert r['decisions'] == again['decisions']
    assert r['events'] == again['events']


def brute_batch(cap, demands, values, eligible):
    best = 0
    for p in product(range(-1, len(cap)), repeat=len(demands)):
        load = np.zeros_like(cap)
        value = 0
        for j, i in enumerate(p):
            if i >= 0:
                if not eligible[i, j]: break
                load[i] += demands[j]
                value += values[j]
        else:
            if np.all(load <= cap): best = max(best, value)
    return best


def test_exhaustive_tiny_event_oracle_and_per_batch_mip():
    # 32 integer chronologies; exact enumeration independent of solver model.
    cap = [[1, 2, 4], [1, 2, 4]]
    for arrivals in product((0, 1), repeat=3):
        for durations in product((1, 2), repeat=2):
            jobs = [Job('A', arrivals[0], durations[0], (1, 1, 2), 2),
                    Job('B', arrivals[1], durations[1], (1, 1, 2), 3),
                    Job('C', arrivals[2], 1, (1, 1, 2), 4)]
            e = np.array([[True, True, False], [False, True, True]])
            r = simulate(cap, jobs, e, 'highs_mip', time_limit=.5)
            check_accounting(cap, jobs, e, r)
            for event in r['events']:
                if event['batch_index'] is None: continue
                b = r['batches'][event['batch_index']]
                indices = [j for j, job in enumerate(jobs) if job.arrival == event['time']]
                optimum = brute_batch(np.array(event['free_before_arrivals']),
                                     np.array([jobs[j].demand for j in indices]),
                                     [jobs[j].value for j in indices], e[:, indices])
                assert b['optimal'] and b['certified_value'] == optimum


@pytest.mark.parametrize('arrival,duration', [(-1,1), (.5,1), (True,1), (0,0), (0,-1), (0,.5), (0,True), (float('nan'),1)])
def test_invalid_time(arrival, duration):
    with pytest.raises(ValueError): simulate([[1,2,4]], [Job('x',arrival,duration,(1,1,1))], [[True]])


@pytest.mark.parametrize('ident', ['', '  ', 1, None])
def test_invalid_ids(ident):
    with pytest.raises(ValueError): simulate([[1,2,4]], [Job(ident,0,1,(1,1,1))], [[True]])


def test_duplicate_identity_freezing_and_input_copies():
    j = Job('x',0,1,[1,1,1])
    assert j == Job('x',0,1,(1,1,1))
    with pytest.raises(FrozenInstanceError): j.arrival = 2
    with pytest.raises(ValueError): simulate([[1,2,4]], [j,replace(j,arrival=1)], [[True,True]])
    cap = np.array([[1.,2.,4.]]); eligible = np.ones((1,1),bool)
    simulate(cap, [j], eligible)
    assert cap.tolist()==[[1,2,4]] and eligible.tolist()==[[True]]


@pytest.mark.parametrize('capacity,demand,value,eligible', [
    ([[1,2]],(1,1,1),1,[[True]]), ([[1,-1,4]],(1,1,1),1,[[True]]),
    ([[np.inf,2,4]],(1,1,1),1,[[True]]), ([[.5,2,4]],(1,1,1),1,[[True]]),
    ([[1,2,4]],(.5,1,1),1,[[True]]), ([[1,2,4]],(1,1),1,[[True]]),
    ([[1,2,4]],(1,np.nan,1),1,[[True]]), ([[1,2,4]],(1,-1,1),1,[[True]]),
    ([[1,2,4]],(1,1,1),-1,[[True]]), ([[1,2,4]],(1,1,1),np.inf,[[True]]),
    ([[1,2,4]],(1,1,1),1,[[1]]), ([[1,2,4]],(1,1,1),1,[[True,False]])])
def test_resource_contract(capacity,demand,value,eligible):
    with pytest.raises(ValueError): simulate(capacity,[Job('x',0,1,demand,value)],eligible)


def test_empty_zero_and_all_ineligible(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('must not solve empty input')
    original=rolling.solve; monkeypatch.setattr(rolling,'solve',forbidden)
    r=simulate([[0,0,0]],[],np.zeros((1,0),bool))
    assert r['events']==r['batches']==r['decisions']==[] and r['final_free']==[[0,0,0]]
    with pytest.raises(ValueError): simulate([[0,0,0]],[],[[]])
    monkeypatch.setattr(rolling,'solve',original)
    for cap,e in [([[0,0,0]],[[True]]),([[8,64,256]],[[False]])]:
        r=simulate(cap,[Job('x',0,1,(1,1,1))],e)
        assert r['rejected_count']==1 and r['solver_error_count']==0
        assert len(r['events'])==1  # Rejected jobs do not create end events.


@pytest.mark.parametrize('kw', [{'method':'bad'},{'seed':True},{'seed':-1},{'time_limit':0},{'time_limit':True},{'time_limit':np.inf}])
def test_options_even_for_empty_workload(kw):
    with pytest.raises(ValueError): simulate([[0,0,0]],[],np.zeros((1,0),bool),**kw)


@pytest.mark.parametrize('status,placement,value', [
    ('uncertified',None,None), ('failed',None,None), ('invalid',(0,),1),
    ('optimal',(2,),1), ('optimal',(0.5,),1), ('optimal',(True,),1),
    ('optimal',(0,0),1), ('optimal',(0,),99), ('optimal',5,1)])
def test_bad_outcomes_atomic_and_preserve_running(monkeypatch,status,placement,value):
    original=rolling.solve
    def fake(a,**kw):
        if a.demand[0,1]==2: return SolveResult('highs_mip',status,placement,value,None,0.,True,'injected')
        return original(a,**kw)
    monkeypatch.setattr(rolling,'solve',fake)
    jobs=[Job('active',0,3,(1,1,1)),Job('failed',1,1,(1,2,1)),Job('after',3,1,(1,1,1))]
    r=simulate([[2,4,8]],jobs,[[True]*3],'highs_mip')
    assert r['decisions'][1]['status']=='solver_error' and r['solver_error_count']==1 and r['rejected_count']==0
    assert r['batches'][1]['solver_status']==status and not r['batches'][1]['optimal']
    assert r['events'][1]['running']==[{'job_id':'active','node':0,'end':3}]
    check_accounting([[2,4,8]],jobs,[[True]*3],r)


def test_injected_eligibility_capacity_exception_and_valid_incumbent(monkeypatch):
    jobs=[Job('x',0,1,(1,1,1))]
    for p,e in [((0,),[[False]]),((0,),[[True]])]:
        cap=[[0,0,0]] if e==[[True]] else [[1,2,4]]
        monkeypatch.setattr(rolling,'solve',lambda *args,**kw: SolveResult('highs_mip','optimal',p,1.,None,0.,True))
        r=simulate(cap,jobs,e,'highs_mip'); assert r['solver_error_count']==1
    def explode(*args,**kw): raise RuntimeError('injected exception')
    monkeypatch.setattr(rolling,'solve',explode)
    r=simulate([[1,2,4]],jobs,[[True]])
    assert r['batches'][0]['solver_status']=='exception' and r['solver_error_count']==1
    monkeypatch.setattr(rolling,'solve',lambda *args,**kw: SolveResult('highs_mip','feasible_uncertified',(0,),1.,None,0.,False,'deadline'))
    r=simulate([[1,2,4]],jobs,[[True]],'highs_mip')
    assert r['accepted_count']==1 and not r['batches'][0]['optimal'] and r['batches'][0]['certified_value'] is None
    assert r['batches'][0]['solver_status']=='feasible_uncertified'


def test_numerical_tolerance_overdraw_is_exposed_not_clipped():
    jobs=[Job('tiny-overdraw',0,2,(1,1+5e-11,1)),Job('next',1,1,(0,0,0)),Job('released',2,1,(1,1,1))]
    r=simulate([[1,1,1]],jobs,[[True]*3])
    assert r['accepted_count']==2 and r['solver_error_count']==1
    assert r['events'][0]['free'][0][1]<0
    assert r['batches'][1]['solver_status']=='input_invalid'
    assert r['events'][2]['free_before_arrivals']==[[1,1,1]]


def test_bad_simultaneous_placement_is_atomic(monkeypatch):
    jobs=[Job('one',0,1,(1,1,1)),Job('two',0,1,(1,1,1))]
    monkeypatch.setattr(rolling,'solve',lambda *args,**kw:
        SolveResult('highs_mip','optimal',(0,1),2.,None,0.,True))
    r=simulate([[2,4,8],[2,4,8]],jobs,[[True,True],[False,False]],'highs_mip')
    assert r['accepted_count']==0 and r['solver_error_count']==2
    assert r['events'][0]['running']==[] and len(r['events'])==1

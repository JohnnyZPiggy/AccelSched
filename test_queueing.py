"""Independent half-open occupancy/FIFO oracle; tiny bounded CPU fixtures only."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import json
import numpy as np
import pytest

import queueing
from rolling import Job, simulate


def oracle(capacity, jobs, eligible, report):
    """Reconstruct from original requests and final decisions, not solver totals."""
    cap = np.asarray(capacity, float)
    by_id = {j.id: (k, j) for k, j in enumerate(jobs)}
    ds = report['decisions']
    assert [d['job_id'] for d in ds] == [j.id for j in jobs]
    accepted = [d for d in ds if d['status'] == 'accepted']
    for d in ds:
        k, j = by_id[d['job_id']]
        assert d['arrival'] == j.arrival and d['decision_time'] >= j.arrival
        if d['status'] == 'accepted':
            assert d['start'] >= j.arrival and d['end'] == d['start']+j.duration
            assert d['wait'] == d['start']-j.arrival
            assert d['turnaround'] == d['end']-j.arrival
            assert eligible[d['node']][k]
        else:
            assert all(d[f] is None for f in ('start', 'end', 'node', 'wait', 'turnaround'))
    order = sorted(range(len(jobs)), key=lambda k: (jobs[k].arrival, k))
    for pos, k in enumerate(order):
        if ds[k]['status'] != 'accepted':
            continue
        for previous in order[:pos]:
            other = ds[previous]
            assert other['decision_time'] <= ds[k]['start']
            if other['status'] == 'accepted':
                assert other['start'] <= ds[k]['start']
            else:
                assert other['status'] == 'unschedulable'

    def load(t, before=False):
        result = np.zeros_like(cap)
        for d in accepted:
            if (d['start'] < t if before else d['start'] <= t) and t < d['end']:
                result[d['node']] += jobs[by_id[d['job_id']][0]].demand
        return result

    expected_times = sorted({j.arrival for j in jobs} | {d['end'] for d in accepted})
    assert [e['time'] for e in report['events']] == expected_times
    event_area = np.zeros(3)
    previous_time, previous_load = 0, np.zeros_like(cap)
    for event in report['events']:
        t = event['time']
        event_area += (t-previous_time)*previous_load.sum(axis=0)
        current = load(t)
        assert np.all(current <= cap+1e-9)
        np.testing.assert_allclose(event['running_load'], current, atol=1e-9, rtol=0)
        np.testing.assert_allclose(event['free'], cap-current, atol=1e-9, rtol=0)
        np.testing.assert_allclose(event['free_before_arrivals'], cap-load(t, True), atol=1e-9, rtol=0)
        assert event['arrivals'] == [j.id for j in jobs if j.arrival == t]
        assert event['completed'] == [d['job_id'] for d in ds if d['status']=='accepted' and d['end']==t]
        assert event['started'] == [jobs[k].id for k in order if ds[k]['status']=='accepted' and ds[k]['start']==t]
        assert event['waiting'] == [jobs[k].id for k in order if jobs[k].arrival<=t<ds[k]['decision_time']]
        assert event['running'] == [dict(job_id=d['job_id'], node=d['node'], end=d['end'])
                                    for d in ds if d['status']=='accepted' and d['start']<=t<d['end']]
        # If the head is left waiting, no eligible physical node currently fits it.
        if event['waiting']:
            k, j = by_id[event['waiting'][0]]
            free = cap-current
            assert not any(eligible[i][k] and np.all(np.asarray(j.demand)<=free[i]+1e-9)
                           for i in range(len(cap)))
            assert event['blocked_head'] == j.id
        previous_time, previous_load = t, current
    work = np.zeros(3)
    mass = np.zeros(3)
    for d in accepted:
        j = jobs[by_id[d['job_id']][0]]
        mass += j.demand
        work += np.asarray(j.demand)*(d['end']-d['start'])
    # Independent integer-time integration, also covers end-only dispatch events.
    ticks = sum((load(t).sum(axis=0) for t in range(max(expected_times, default=0))), start=np.zeros(3))
    for measured in (report['resource_time'], event_area, ticks):
        np.testing.assert_allclose(measured, work, atol=1e-9, rtol=0)
    for measured in (report['admitted_resources'], report['released_resources']):
        np.testing.assert_allclose(measured, mass, atol=1e-9, rtol=0)
    np.testing.assert_array_equal(report['final_free'], cap)
    assert report['accepted_count'] == report['completed_count'] == len(accepted)
    assert report['accepted_value'] == sum(jobs[by_id[d['job_id']][0]].value for d in accepted)
    assert sum(report['counts'].values()) == len(jobs)
    for status,count in report['counts'].items():
        assert count == sum(d['status']==status for d in ds)
    assert report['wait_mean'] == (sum(d['wait'] for d in accepted)/len(accepted) if accepted else None)
    assert report['wait_max'] == (max(d['wait'] for d in accepted) if accepted else None)
    assert len(report['attempts']) == len(accepted)+report['counts']['solver_error']
    json.dumps(report, allow_nan=False)


def run(capacity, jobs, eligible=None, **kwargs):
    if eligible is None:
        eligible = np.ones((len(capacity),len(jobs)),bool)
    before = deepcopy((capacity,jobs,eligible))
    report = queueing.simulate_fifo(capacity,jobs,eligible,**kwargs)
    assert capacity == before[0] and jobs == before[1]
    np.testing.assert_array_equal(eligible,before[2])
    oracle(capacity,jobs,eligible,report)
    return report


def test_fifo_hand_chronology_and_rolling_unchanged():
    cap = [[1,4,16]]
    jobs = [Job('A',0,2,(1,4,16)),Job('B',1,2,(1,4,16)),Job('C',2,1,(1,4,16))]
    report = run(cap,jobs)
    assert [(d['start'],d['end'],d['wait'],d['turnaround']) for d in report['decisions']] == [(0,2,0,2),(2,4,1,3),(4,5,2,3)]
    assert report['wait_mean']==1 and report['resource_time']==[5,20,80]
    assert next(e for e in report['events'] if e['time']==4)['started']==['C']
    # A single tiny existing immediate-start run; it must still reject B forever.
    old = simulate(cap,jobs,[[True]*3])
    assert [d['status'] for d in old['decisions']]==['accepted','not_scheduled','accepted']
    assert [(d['start'],d['end']) for d in old['decisions']]==[(0,2),(None,None),(2,3)]


def test_no_backfill_for_smaller_job_and_end_only_trigger():
    jobs=[Job('running',0,4,(1,2,4)),Job('large',1,2,(2,4,8)),Job('small',2,1,(1,2,4))]
    report=run([[2,4,8]],jobs)
    assert [d['start'] for d in report['decisions']]==[0,4,6]
    assert next(e for e in report['events'] if e['time']==2)['waiting']==['large','small']
    assert [(a['time'],a['job_id']) for a in report['attempts']]==[(0,'running'),(4,'large'),(6,'small')]


def test_arrival_order_original_index_and_same_timestamp_release():
    jobs=[Job('later',2,1,(1,1,1)),Job('z',0,2,(1,1,1)),Job('a',0,1,(1,1,1))]
    report=run([[1,1,1]],jobs)
    assert [d['start'] for d in report['decisions']]==[3,0,2]
    event=next(e for e in report['events'] if e['time']==2)
    assert event['completed']==['z'] and event['arrivals']==['later'] and event['started']==['a']


@pytest.mark.parametrize('kind',['eligibility','cpu','ram','gpu'])
def test_structural_head_removed_only_when_impossible(kind):
    demand={'eligibility':(1,1,1),'cpu':(1,3,1),'ram':(1,1,3),'gpu':(2,1,1)}[kind]
    jobs=[Job('impossible',0,1,demand),Job('next',0,1,(1,1,1))]
    report=run([[1,2,2]],jobs,[[kind!='eligibility',True]])
    assert report['decisions'][0]['status']=='unschedulable'
    assert report['decisions'][0]['reason']=='no_empty_eligible_node'
    assert report['decisions'][1]['start']==0


def test_empty_zero_resource_and_all_ineligible():
    report=run([[0,0,0]],[],np.empty((1,0),bool))
    assert report['events']==[] and report['wait_mean'] is None
    jobs=[Job('zero',0,1,(0,0,0),0),Job('zero2',0,2,(0,0,0),0)]
    report=run([[0,0,0]],jobs)
    assert report['accepted_count']==2 and report['resource_time']==[0,0,0]
    report=run([[0,0,0]],jobs,[[False,False]])
    assert report['counts']['unschedulable']==2 and len(report['events'])==1 and not report['attempts']


@pytest.mark.parametrize('demand',[(0,1.5,.125),(0,.125,1.5)])
def test_fractional_cpu_ram_bottlenecks(demand):
    report=run([[0,2,2]],[Job('a',0,2,demand),Job('b',1,1,demand)])
    assert [d['start'] for d in report['decisions']]==[0,2]


@pytest.mark.parametrize('failure',['exception','none','empty','bool','invalid_node','nan_value','residual','bad_status'])
def test_solver_failure_aborts_waiting_and_future_but_running_finishes(monkeypatch,failure):
    original=queueing.solve;calls=[]
    def injected(instance,**kwargs):
        calls.append(instance)
        assert instance.demand.shape==(1,3)
        if len(calls)==1:
            return original(instance,**kwargs)
        if failure=='exception':raise RuntimeError('injected solver failure')
        result=SimpleNamespace(status='feasible',optimal=False,error=None,placement=(0,),value=1.,residual=instance.capacity.copy())
        result.residual[0]-=instance.demand[0]
        if failure=='none':result.placement=None
        if failure=='empty':result.placement=(-1,)
        if failure=='bool':result.placement=(True,)
        if failure=='invalid_node':result.placement=(5,)
        if failure=='nan_value':result.value=float('nan')
        if failure=='residual':result.residual=None
        if failure=='bad_status':result.status='failed'
        return result
    monkeypatch.setattr(queueing,'solve',injected)
    jobs=[Job('running',0,5,(1,1,1)),Job('head',1,2,(1,1,1)),Job('waiting',1,1,(1,1,1)),Job('future',2,1,(1,1,1))]
    report=run([[2,2,2]],jobs)
    assert len(calls)==2
    assert [d['status'] for d in report['decisions']]==['accepted','solver_error','blocked_by_error','blocked_by_error']
    assert report['events'][-1]['time']==5 and report['events'][-1]['completed']==['running']
    assert report['counts']['unschedulable']==0 and report['completed_count']==1
    assert report['attempts'][-1]['error'] and not report['attempts'][-1]['certified']


def test_real_mip_singleton_certificate_is_not_horizon_certificate():
    report=run([[1,1,1]],[Job('a',0,2,(1,1,1)),Job('b',1,1,(1,1,1))],method='highs_mip',time_limit=.5)
    assert all(a['certified'] and a['solver_status']=='optimal' for a in report['attempts'])
    assert report['certification_scope']=='singleton attempts only; no FIFO/horizon optimum'


def test_valid_uncertified_placement_can_start(monkeypatch):
    original=queueing.solve
    def uncertified(a,**kwargs):
        return replace(original(a,**kwargs),status='feasible_uncertified',optimal=False)
    monkeypatch.setattr(queueing,'solve',uncertified)
    report=run([[1,1,1]],[Job('a',0,1,(1,1,1))])
    assert report['accepted_count']==1 and not report['attempts'][0]['certified']


@pytest.mark.parametrize('case',['duplicate','blank','arrival_bool','duration_zero','nan','negative','fractional_gpu','bad_mask','no_nodes','method','seed','time_limit','unrepresentable_gpu'])
def test_prevalidation_before_any_solve(monkeypatch,case):
    cap=[[1,1,1]];jobs=[Job('a',0,1,(1,1,1))];mask=[[True]];kw={}
    if case=='duplicate':jobs*=2;mask=[[True,True]]
    if case=='blank':jobs=[Job(' ',0,1,(1,1,1))]
    if case=='arrival_bool':jobs=[Job('a',True,1,(1,1,1))]
    if case=='duration_zero':jobs=[Job('a',0,0,(1,1,1))]
    if case=='nan':jobs=[Job('a',0,1,(1,float('nan'),1))]
    if case=='negative':jobs=[Job('a',0,1,(1,-1,1))]
    if case=='fractional_gpu':jobs=[Job('a',0,1,(.5,1,1))]
    if case=='bad_mask':mask=[[1]]
    if case=='no_nodes':cap=[];mask=[]
    if case=='method':kw['method']='missing'
    if case=='seed':kw['seed']=True
    if case=='time_limit':kw['time_limit']=float('inf')
    if case=='unrepresentable_gpu':cap=[[2**53+1,1,1]]
    def forbidden(*a,**k):raise AssertionError('must validate before solve')
    monkeypatch.setattr(queueing,'solve',forbidden)
    with pytest.raises(ValueError):queueing.simulate_fifo(cap,jobs,mask,**kw)


def test_deterministic_fifo_without_timing_fields():
    jobs=[Job('a',0,2,(1,1,1)),Job('b',0,1,(1,1,1))]
    assert run([[1,1,1]],jobs)==run([[1,1,1]],jobs)


def test_abundant_ineligible_node_is_never_used():
    jobs=[Job('a',0,2,(1,1,1)),Job('b',1,1,(1,1,1))]
    report=run([[8,64,256],[1,1,1]],jobs,[[False,False],[True,True]])
    assert [d['node'] for d in report['decisions']]==[1,1]
    assert [d['start'] for d in report['decisions']]==[0,2]


def test_oracle_tolerance_is_not_silent_capacity_rejection():
    # Validator permits this tiny CPU overload; first_fit's tighter threshold does
    # not place it. A feasible-oracle/solver disagreement is an error, not rejection.
    report=run([[0,0,0]],[Job('tiny',0,1,(0,5e-10,0))])
    assert report['decisions'][0]['status']=='solver_error'
    assert report['counts']['unschedulable']==0

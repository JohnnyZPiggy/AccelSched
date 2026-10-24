"""Independent original-job Oracle; reuse the benchmark's six actual ledgers.

SAE_QUEUE_COMPARE_RESULT may point to that batch's JSON stdout. Without it, one
six-call module fixture is run. Boundary fixtures below are pure fabricated
reports explicitly labelled as mocks; they do not run a solver or simulator.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import bench_queue_compare as bench


@pytest.fixture(scope='module')
def experiment():
    path = os.environ.get('SAE_QUEUE_COMPARE_RESULT')
    result = json.loads(Path(path).read_bytes()) if path else bench.run_benchmark()
    assert result['simulator_invocations'] == 6
    assert len(result['cases']) == 3 and result['source_dirty'] is False
    for case, fixed in zip(result['cases'], bench.fixed_cases()):
        assert case['input'] == json.loads(json.dumps(fixed))
        assert case['input_sha256'] == bench.input_sha256(fixed)
    with patch.object(bench, 'simulate', side_effect=AssertionError('reuse six-run evidence')), \
         patch.object(bench, 'simulate_fifo', side_effect=AssertionError('reuse six-run evidence')):
        yield result


def physical_oracle(manifest, raw, policy):
    """Reconstruct physical load from original demands and half-open decisions."""
    jobs = manifest['jobs']; cap = np.array(manifest['capacity'], float)
    ids = {j['id']: k for k, j in enumerate(jobs)}
    decisions = raw['decisions']; ds = {d['job_id']: d for d in decisions}
    assert len(ds) == len(decisions) == len(jobs) and set(ds) == set(ids)
    accepted = [d for d in decisions if d['status'] == 'accepted']
    allowed = {'accepted', 'not_scheduled', 'solver_error'} if policy == 'immediate' else {
        'accepted', 'unschedulable', 'solver_error', 'blocked_by_error', 'blocked_by_fifo'}
    for d in decisions:
        j = jobs[ids[d['job_id']]]
        assert d['status'] in allowed and d['arrival'] == j['arrival']
        if d['status'] != 'accepted':
            assert d['node'] is d['start'] is d['end'] is None
            continue
        assert manifest['eligible'][d['node']][ids[d['job_id']]]
        assert manifest['node_families'][d['node']] == j['family']
        assert d['start'] >= j['arrival'] and d['end'] == d['start']+j['duration']
        if policy == 'immediate':
            assert d['start'] == j['arrival']
        else:
            assert d['wait'] == d['start']-j['arrival']
            assert d['turnaround'] == d['end']-j['arrival']
    order = sorted(range(len(jobs)), key=lambda k: (jobs[k]['arrival'], k))
    assert [d['job_id'] for d in decisions] == ([j['id'] for j in jobs] if policy == 'fifo' else [jobs[k]['id'] for k in order])

    def load(t, before_arrivals=False):
        values = np.zeros_like(cap)
        for d in accepted:
            if (d['start'] < t if before_arrivals else d['start'] <= t) and t < d['end']:
                values[d['node']] += jobs[ids[d['job_id']]]['demand']
        return values

    times = sorted({j['arrival'] for j in jobs} | {d['end'] for d in accepted})
    assert [e['time'] for e in raw['events']] == times
    mass = sum((np.array(jobs[ids[d['job_id']]]['demand']) for d in accepted), start=np.zeros(3))
    full = sum((np.array(jobs[ids[d['job_id']]]['demand'])*(d['end']-d['start']) for d in accepted), start=np.zeros(3))
    window = sum((np.array(jobs[ids[d['job_id']]]['demand'])*max(0,min(d['end'],manifest['horizon'])-d['start']) for d in accepted), start=np.zeros(3))
    event_window, event_full = np.zeros(3), np.zeros(3)
    previous, previous_load = 0, np.zeros_like(cap)
    for e in raw['events']:
        t = e['time']; current = load(t)
        assert np.all(current >= 0) and np.all(current <= cap)
        # Integer fixture quantities must match exactly; no floating tolerance used.
        np.testing.assert_array_equal(e['running_load'], current)
        np.testing.assert_array_equal(e['free'], cap-current)
        np.testing.assert_array_equal(e['free_before_arrivals'], cap-load(t,True))
        assert e['completed'] == [j['id'] for j in jobs if ds[j['id']]['status']=='accepted' and ds[j['id']]['end']==t]
        start_key = 'accepted' if policy=='immediate' else 'started'
        assert e[start_key] == [jobs[k]['id'] for k in order if ds[jobs[k]['id']]['status']=='accepted' and ds[jobs[k]['id']]['start']==t]
        assert e['running'] == [dict(job_id=j['id'],node=ds[j['id']]['node'],end=ds[j['id']]['end']) for j in jobs if ds[j['id']]['status']=='accepted' and ds[j['id']]['start']<=t<ds[j['id']]['end']]
        event_full += (t-previous)*previous_load.sum(axis=0)
        event_window += max(0,min(t,manifest['horizon'])-min(previous,manifest['horizon']))*previous_load.sum(axis=0)
        if policy=='fifo':
            waiting=[jobs[k]['id'] for k in order if jobs[k]['arrival']<=t<ds[jobs[k]['id']]['decision_time']]
            assert e['waiting']==waiting
            if waiting:
                j=jobs[ids[waiting[0]]]
                assert not any(manifest['eligible'][i][ids[j['id']]] and all(c>=d for c,d in zip(e['free'][i],j['demand'])) for i in range(len(cap)))
        previous,previous_load=t,current
    event_window += max(0,manifest['horizon']-previous)*previous_load.sum(axis=0)
    ticks=sum((load(t).sum(axis=0) for t in range(manifest['horizon'])),start=np.zeros(3))
    for measured in (event_window,ticks): np.testing.assert_array_equal(measured,window)
    for measured in (raw['resource_time'],event_full): np.testing.assert_array_equal(measured,full)
    for measured in (raw['admitted_resources'],raw['released_resources']): np.testing.assert_array_equal(measured,mass)
    np.testing.assert_array_equal(raw['final_free'],cap)
    assert raw['accepted_count']==raw['completed_count']==len(accepted)
    if policy=='fifo':
        for pos,k in enumerate(order):
            d=ds[jobs[k]['id']]
            if d['status']=='accepted':
                assert all(ds[jobs[h]['id']]['decision_time']<=d['start'] for h in order[:pos])
    else:
        # Every permanent refusal is independently capacity/eligibility blocked at
        # its arrival, including accepted earlier peers in the same arrival batch.
        for k in order:
            j=jobs[k];d=ds[j['id']]
            if d['status']=='not_scheduled':
                occupied=load(j['arrival'],True)
                for h in order[:order.index(k)]:
                    other=ds[jobs[h]['id']]
                    if other['status']=='accepted' and other['start']==j['arrival']:
                        occupied[other['node']]+=jobs[h]['demand']
                assert not any(manifest['eligible'][i][k] and np.all(np.asarray(j['demand'])<=cap[i]-occupied[i]) for i in range(len(cap)))
    return accepted, window


@pytest.mark.parametrize('index',[0,1,2])
def test_six_real_runs_physical_time_and_status_oracles(experiment,index):
    case=experiment['cases'][index];m=case['input'];t=m['horizon']
    for policy in ('immediate','fifo'):
        accepted,work=physical_oracle(m,case['raw'][policy],policy)
        metrics=case['policies'][policy]
        lookup={j['id']:j for j in m['jobs']}
        assert metrics['accepted_total']==len(accepted)
        assert metrics['completed_by_T']==sum(d['end']<=t for d in accepted)
        assert metrics['started_by_T']==sum(d['start']<t for d in accepted)
        waits=[d['start']-lookup[d['job_id']]['arrival'] for d in accepted]
        assert metrics['waiting_time_mean']==sum(waits)/len(waits) and metrics['waiting_time_max']==max(waits)
        assert metrics['turnaround_mean']==sum(d['end']-lookup[d['job_id']]['arrival'] for d in accepted)/len(accepted)
        np.testing.assert_array_equal(metrics['window_resource_time'],work)
        assert metrics['utilization']==[work[k]/(t*sum(c[k] for c in m['capacity'])) for k in range(3)]
        for status,count in metrics['status_counts'].items():assert count==sum(d['status']==status for d in case['raw'][policy]['decisions'])
    assert case['comparison_status']=='COMPARABLE'
    for key in ('accepted_total','completed_by_T','started_by_T','waiting_time_mean','waiting_time_max','turnaround_mean'):
        assert case['delta_FIFO_minus_immediate'][key]==case['policies']['fifo'][key]-case['policies']['immediate'][key]


@pytest.mark.parametrize('index,policy,counts,wait,turn,work',[
    (0,'immediate',(2,2,2),0,1.5,[3,12,48]),
    (0,'fifo',(3,2,2),1,8/3,[4,16,64]),
    (1,'immediate',(2,2,2),0,2.5,[5,10,20]),
    (1,'fifo',(3,1,2),7/3,14/3,[6,12,24]),
    (2,'immediate',(3,3,3),0,7/3,[7,11,10]),
    (2,'fifo',(5,2,3),11/5,4,[4,8,6])])
def test_preregistered_hand_table(experiment,index,policy,counts,wait,turn,work):
    r=experiment['cases'][index]['policies'][policy]
    assert (r['accepted_total'],r['completed_by_T'],r['started_by_T'])==counts
    assert r['waiting_time_mean']==wait and r['turnaround_mean']==turn
    assert r['window_resource_time']==work


def test_hol_idle_capacity_censoring_and_exact_T(experiment):
    sequential,hol,family=experiment['cases']
    assert sequential['policies']['fifo']['accepted_total']==3
    assert sequential['policies']['fifo']['started_by_T']==2  # C starts exactly T.
    assert sequential['policies']['fifo']['completed_by_T']==2  # B completes at T.
    assert sequential['raw']['fifo']['resource_time']!=sequential['policies']['fifo']['window_resource_time']
    event=next(e for e in hol['raw']['fifo']['events'] if e['time']==2)
    assert event['waiting']==['large','small'] and event['free']==[[1,2,4]]
    assert hol['delta_FIFO_minus_immediate']['completed_by_T']==-1
    assert all(d['end']>family['input']['horizon'] for d in family['raw']['fifo']['decisions'] if d['job_id'] in ['V-ram','V-wait','T-later'])
    assert all(e['waiting']==[] for e in [c['raw']['fifo']['events'][-1] for c in experiment['cases']])


def mock_case(jobs,capacity,horizon=5):
    return dict(name='MOCK_BOUNDARY',horizon=horizon,capacity=capacity,node_families=['T4'],
                jobs=jobs,eligible=[[True]*len(jobs)],solver=dict(method='first_fit',seed=11,time_limit=.5))


def test_zero_denominator_initial_empty_interval_and_T_after_tail():
    m=mock_case([dict(id='x',family='T4',arrival=1,duration=1,demand=[0,.125,.5],value=1)],[[0,1,1]])
    raw=dict(decisions=[dict(job_id='x',status='accepted',start=1,end=2)],events=[
        dict(time=1,running_load=[[0,.125,.5]]),dict(time=2,running_load=[[0,0,0]])])
    r=bench.summarize_policy(m,raw)
    np.testing.assert_allclose(r['window_resource_time'],[0,.125,.5],atol=1e-9,rtol=0)
    assert r['utilization']==[None,.025,.1] and r['waiting_time_mean']==0
    assert r['completed_by_T']==1 and r['turnaround_mean']==1


def test_mock_empty_and_all_ineligible_and_original_statuses():
    empty=mock_case([],[[0,0,0]])
    raw=dict(decisions=[],events=[])
    result=bench.compare_results(empty,raw,raw)
    assert result['policies']['fifo']['waiting_time_mean'] is None
    assert result['policies']['fifo']['utilization']==[None]*3
    j=dict(id='missing',family='P100',arrival=0,duration=1,demand=[1,1,1],value=1)
    m=mock_case([j],[[1,1,1]]);m['eligible']=[[False]]
    a=dict(decisions=[dict(job_id='missing',status='not_scheduled')],events=[])
    b=dict(decisions=[dict(job_id='missing',status='unschedulable')],events=[])
    result=bench.compare_results(m,a,b)
    assert result['policies']['immediate']['status_counts']=={'not_scheduled':1}
    assert result['policies']['fifo']['status_counts']=={'unschedulable':1}
    assert result['comparison_status']=='COMPARABLE'  # No future event needed for either terminal refusal.


@pytest.mark.parametrize('status',['solver_error','blocked_by_error','blocked_by_fifo'])
def test_mock_error_suppresses_comparison_deltas(experiment,status):
    case=experiment['cases'][0];a=deepcopy(case['raw']['immediate']);b=deepcopy(case['raw']['fifo'])
    # Terminal nonstarted synthetic error, replacing already nonaccepted immediate B.
    a['decisions'][1]['status']=status
    result=bench.compare_results(case['input'],a,b)
    assert result['comparison_status']=='NOT COMPARABLE'
    assert 'no superiority claim' in result['interpretation']
    assert all(v is None for k,v in result['delta_FIFO_minus_immediate'].items() if k!='utilization')
    assert result['delta_FIFO_minus_immediate']['utilization']==[None]*3


def test_pure_comparison_no_extra_simulations_immutability(experiment,monkeypatch):
    case=experiment['cases'][1];before=deepcopy(case)
    def forbidden(*a,**kw):raise AssertionError('no simulation allowed in reporting')
    monkeypatch.setattr(bench,'simulate',forbidden);monkeypatch.setattr(bench,'simulate_fifo',forbidden)
    result=bench.compare_results(case['input'],case['raw']['immediate'],case['raw']['fifo'])
    assert result==case and before==case
    result['input']['jobs'][0]['value']=99
    assert before==case


def test_oracle_detects_capacity_and_event_corruption(experiment):
    case=experiment['cases'][2]
    bad=deepcopy(case['raw']['fifo']);bad['events'][0]['running_load'][0][1]+=1
    with pytest.raises(AssertionError):physical_oracle(case['input'],bad,'fifo')
    bad=deepcopy(case['raw']['fifo']);bad['decisions'][0]['node']=1
    with pytest.raises(AssertionError):physical_oracle(case['input'],bad,'fifo')


def test_actual_batch_provenance_and_singleton_scope(experiment):
    assert len(experiment['source_sha'])==40
    for case in experiment['cases']:
        digest=hashlib.sha256(json.dumps(case['input'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        assert digest==case['input_sha256']
        assert all(not a['certified'] for a in case['raw']['fifo']['attempts'])
        assert all(not b['optimal'] for b in case['raw']['immediate']['batches'])
        assert all(d['status']=='accepted' for d in case['raw']['fifo']['decisions'])
    json.dumps(experiment,allow_nan=False)

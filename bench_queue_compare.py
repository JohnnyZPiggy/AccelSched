"""Three pre-fixed SYNTHETIC cases, six calls; no policy or speedup ranking.

Run ``python -m bench_queue_compare`` for JSON. The output retains the six raw
decision/event ledgers so independent tests can reuse this exact batch without
another simulation. Window metrics use original jobs, not full-tail totals.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from rolling import Job, simulate
from queueing import simulate_fifo


def fixed_cases():
    """Fresh manifests in fixed input order; no random generation or resampling."""
    specs = [
        ('sequential', 4, [(1, 4, 16)], ['T4'],
         [Job('A', 0, 2, (1, 4, 16)), Job('B', 1, 2, (1, 4, 16)),
          Job('C', 2, 1, (1, 4, 16))], ['T4']*3),
        ('head_of_line', 5, [(2, 4, 8)], ['T4'],
         [Job('running', 0, 4, (1, 2, 4)), Job('large', 1, 2, (2, 4, 8)),
          Job('small', 2, 1, (1, 2, 4))], ['T4']*3),
        ('family_cpu_ram', 4, [(1, 2, 8), (1, 8, 2)], ['T4', 'V100'],
         [Job('T-cpu', 0, 3, (1, 2, 1)), Job('T-wait', 1, 1, (0, 1, 1)),
          Job('V-ram', 1, 3, (1, 1, 2)), Job('V-wait', 2, 1, (0, 1, 1)),
          Job('T-later', 3, 1, (1, 2, 1))], ['T4', 'T4', 'V100', 'V100', 'T4'])]
    return [dict(name=name, horizon=t, capacity=[list(c) for c in cap],
                 node_families=families,
                 jobs=[dict(**asdict(j), family=f) for j, f in zip(jobs, requests)],
                 eligible=[[f == request for request in requests] for f in families],
                 solver=dict(method='first_fit', seed=11, time_limit=.5))
            for name, t, cap, families, jobs, requests in specs]


def input_sha256(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def event_work(events, horizon):
    """Integrate post-event loads, including initial empty and clipped final interval."""
    work, load, previous = [0., 0., 0.], [0., 0., 0.], 0
    for e in events:
        t = e['time']
        if t < previous:
            raise ValueError('event times must be chronological')
        width = max(0, min(t, horizon)-min(previous, horizon))
        work = [w+width*v for w, v in zip(work, load)]
        load = [sum(row[k] for row in e['running_load']) for k in range(3)]
        previous = t
    width = max(0, horizon-previous)
    return [w+width*v for w, v in zip(work, load)]


def summarize_policy(manifest, raw):
    """Pure metrics from ID-joined original jobs and accepted half-open decisions."""
    jobs = {j['id']: j for j in manifest['jobs']}
    decisions = raw['decisions']
    if len(decisions) != len(jobs) or {d['job_id'] for d in decisions} != set(jobs):
        raise ValueError('decisions must cover each original job exactly once')
    accepted = [d for d in decisions if d['status'] == 'accepted']
    t = manifest['horizon']
    waits = [d['start']-jobs[d['job_id']]['arrival'] for d in accepted]
    turnarounds = [d['end']-jobs[d['job_id']]['arrival'] for d in accepted]
    work = [sum(jobs[d['job_id']]['demand'][k]*max(0, min(d['end'], t)-max(d['start'], 0))
                for d in accepted) for k in range(3)]
    integrated = event_work(raw['events'], t)
    if not all(math.isclose(a, b, abs_tol=1e-9, rel_tol=0) for a, b in zip(work, integrated)):
        raise ValueError('decision/event window work mismatch')
    total = [t*sum(c[k] for c in manifest['capacity']) for k in range(3)]
    return dict(status_counts=dict(Counter(d['status'] for d in decisions)),
                accepted_total=len(accepted),
                completed_by_T=sum(d['end'] <= t for d in accepted),
                started_by_T=sum(d['start'] < t for d in accepted),
                waiting_time_mean=sum(waits)/len(waits) if waits else None,
                waiting_time_max=max(waits) if waits else None,
                turnaround_mean=sum(turnarounds)/len(turnarounds) if turnarounds else None,
                window_resource_time=work, event_window_resource_time=integrated,
                utilization=[w/c if c else None for w, c in zip(work, total)])


def compare_results(manifest, immediate, fifo):
    policies = dict(immediate=summarize_policy(manifest, immediate),
                    fifo=summarize_policy(manifest, fifo))
    errors = ('solver_error', 'blocked_by_error', 'blocked_by_fifo')
    comparable = not any(r['status_counts'].get(s, 0) for r in policies.values() for s in errors)
    a, b = policies['immediate'], policies['fifo']
    delta = {k: b[k]-a[k] if comparable and a[k] is not None and b[k] is not None else None
             for k in ('accepted_total', 'completed_by_T', 'started_by_T',
                       'waiting_time_mean', 'waiting_time_max', 'turnaround_mean')}
    delta['utilization'] = [y-x if comparable and x is not None and y is not None else None
                            for x, y in zip(a['utilization'], b['utilization'])]
    result = dict(name=manifest['name'], label='SYNTHETIC', input=deepcopy(manifest),
                  input_sha256=input_sha256(manifest), policies=policies,
                  comparison_status='COMPARABLE' if comparable else 'NOT COMPARABLE',
                  delta_FIFO_minus_immediate=delta,
                  interpretation='observed policy differences; no overall winner' if comparable
                  else 'diagnostic outcomes only; no superiority claim',
                  raw=dict(immediate=deepcopy(immediate), fifo=deepcopy(fifo)))
    json.dumps(result, allow_nan=False)
    return result


def run_benchmark():
    results, calls = [], 0
    for manifest in fixed_cases():
        jobs = tuple(Job(**{k: v for k, v in j.items() if k != 'family'}) for j in manifest['jobs'])
        outputs = []
        for simulator in (simulate, simulate_fifo):
            cap = np.array(manifest['capacity'], float)
            mask = np.array(manifest['eligible'], bool)
            before = deepcopy((cap, mask, jobs))
            outputs.append(simulator(cap, jobs, mask, **manifest['solver']))
            calls += 1
            if not (np.array_equal(cap, before[0]) and np.array_equal(mask, before[1]) and jobs == before[2]):
                raise ValueError('simulator mutated the shared workload')
        results.append(compare_results(manifest, *outputs))
    try:
        root = str(Path(__file__).resolve().parent)
        head = subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True, timeout=2).strip()
        dirty = bool(subprocess.check_output(['git', '-C', root, 'status', '--porcelain'], timeout=2))
    except (OSError, subprocess.SubprocessError):
        head, dirty = None, None
    return dict(label='SYNTHETIC PROBE_ONLY', source_sha=head, source_dirty=dirty,
                simulator_invocations=calls, cases=results,
                resource_units=['GPU-count', 'CPU-cores', 'host-RAM-GiB'],
                resource_time_units='resource multiplied by arbitrary integer time',
                limits='no speedup, FIFO superiority, horizon optimality, hardware or ROI claim')


if __name__ == '__main__':
    print(json.dumps(run_benchmark(), allow_nan=False))

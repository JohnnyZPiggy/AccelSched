"""Strict FIFO over immutable rolling.Job requests; no backfill or prediction.

At each real arrival/completion event: integrate [previous,t), release completions,
append arrivals in input order, then dispatch the head only. Structurally impossible
heads terminate as unschedulable; temporarily busy heads wait. A failed/invalid
singleton solve aborts dispatch: the failed head is solver_error, every other
unstarted job (including future arrivals) is blocked_by_error. Running jobs finish.
No artificial horizon, error retry, migration, reservation or preemption is used.

Resource feasibility uses the existing validate tolerance, not exact arithmetic.
Negative residuals within that tolerance are exposed, not clipped; if the builder
cannot represent current free capacity, the feasible head records solver_error.
Certificates cover singleton attempts only, never FIFO/horizon optimality.
"""
from collections import Counter, deque
import heapq

import numpy as np

from accelsched import METHODS, solve
from m0_oracle import validate
from rolling import Job, _integer, _shapes
from workload import build_workload


def simulate_fifo(capacity, jobs, eligible, method='first_fit', *, seed=11, time_limit=2.):
    """Return final input-order decisions and chronological post-event snapshots.

    Units: whole GPU count, CPU cores, host RAM GiB; integer arrival/duration time.
    Empty jobs require explicit boolean eligibility of shape (nodes,0). All inputs
    are validated/copied before dispatch. Wait/turnaround are null for nonstarts.
    blocked_by_fifo/no_future_event is a terminal defensive deadlock outcome, not
    physical rejection. Solver errors never become capacity rejection.
    """
    if method not in METHODS:
        raise ValueError(f'unknown method {method!r}')
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (isinstance(time_limit, (bool, np.bool_))
            or not isinstance(time_limit, (int, float, np.integer, np.floating))
            or not np.isfinite(time_limit) or time_limit <= 0):
        raise ValueError('time_limit must be finite and positive')
    jobs = tuple(jobs)
    ids = set()
    for j in jobs:
        if not isinstance(j, Job) or not isinstance(j.id, str) or not j.id.strip():
            raise ValueError('each job must have a nonblank string id')
        if j.id in ids:
            raise ValueError(f'duplicate job id {j.id!r}')
        ids.add(j.id)
        if not _integer(j.arrival) or j.arrival < 0:
            raise ValueError('arrival must be a nonnegative integer')
        if not _integer(j.duration) or j.duration <= 0:
            raise ValueError('duration must be a positive integer')
    cap, mask = np.asarray(capacity), np.asarray(eligible)
    if cap.ndim != 2 or cap.shape[0] == 0 or cap.shape[1] != 3:
        raise ValueError('capacity must have shape (nonempty nodes,3)')
    s, n = len(cap), len(jobs)
    if mask.shape != (s, n) or mask.dtype != np.dtype(bool):
        raise ValueError('eligible must be a boolean (nodes,jobs) array')
    full = build_workload(cap, [j.demand for j in jobs] if n else [[0, 0, 0]],
                          value=[j.value for j in jobs] if n else [0],
                          eligible=mask if n else np.zeros((s, 1), bool),
                          shapes=np.arange(s))
    # Integer inputs must not silently lose whole GPUs during float64 conversion.
    for original in [row[0] for row in capacity] + [j.demand[0] for j in jobs]:
        if int(original) != int(float(original)):
            raise ValueError('GPU count must be exactly representable as float64')
    arrivals = {}
    for index, j in enumerate(jobs):
        arrivals.setdefault(int(j.arrival), []).append(index)
    times = list(arrivals)
    heapq.heapify(times)
    scheduled = set(times)
    endings, running = {}, {}
    waiting = deque()
    decisions = [None]*n
    events, attempts = [], []
    admitted, released, resource_time = (np.zeros(3) for _ in range(3))
    previous, previous_load = 0, np.zeros_like(full.capacity)
    aborted = False

    def placement_valid(placements):
        p = [-1]*len(full.value)
        for j, i in placements.items():
            p[j] = i
        return validate(full, p)

    def state():
        load = np.zeros_like(full.capacity)
        for j, i in running.items():
            load[i] += full.demand[j]
        return load, full.capacity-load

    def finish(j, t, status, reason, node=None):
        start = t if node is not None else None
        end = t+int(jobs[j].duration) if node is not None else None
        decisions[j] = dict(job_id=jobs[j].id, status=status, reason=reason,
                            arrival=int(jobs[j].arrival), decision_time=t,
                            start=start, end=end, node=node,
                            wait=t-int(jobs[j].arrival) if node is not None else None,
                            turnaround=end-int(jobs[j].arrival) if node is not None else None)

    while times:
        t = heapq.heappop(times)
        resource_time += (t-previous)*previous_load.sum(axis=0)
        completed = sorted(endings.pop(t, []))
        for j in completed:
            running.pop(j)
            released += full.demand[j]
        before = state()[1].tolist()
        group = arrivals.get(t, [])
        waiting.extend(group)
        started, terminal = [], []
        blocked_head = None
        if aborted:
            while waiting:
                j = waiting.popleft()
                finish(j, t, 'blocked_by_error', 'dispatch_aborted')
                terminal.append(jobs[j].id)
        while waiting:
            j = waiting[0]
            empty_fits = [i for i in range(s) if placement_valid({j: i})]
            if not empty_fits:
                waiting.popleft()
                finish(j, t, 'unschedulable', 'no_empty_eligible_node')
                terminal.append(jobs[j].id)
                continue
            if not any(placement_valid(running | {j: i}) for i in empty_fits):
                blocked_head = jobs[j].id
                break
            _, free = state()
            record = dict(time=t, job_id=jobs[j].id, method=method,
                          solver_status='input_invalid', validation_status='not_run',
                          error=None, node=None, reported_optimal=False,
                          certified=False, certification_scope='singleton attempt only')
            node = None
            try:
                batch = build_workload(free, full.demand[[j]], value=full.value[[j]],
                                       eligible=full.eligible[:, [j]], shapes=_shapes(free))
                record['solver_status'] = 'exception'
                result = solve(batch, method=method, seed=seed, time_limit=time_limit)
                record.update(solver_status=result.status, error=result.error,
                              reported_optimal=bool(result.optimal))
                p = tuple(result.placement) if result.placement is not None else ()
                valid = (result.status in ('optimal', 'feasible', 'feasible_uncertified')
                         and len(p) == 1 and _integer(p[0]) and 0 <= p[0] < s)
                if valid:
                    i = int(p[0])
                    expected_free = free.copy()
                    expected_free[i] -= full.demand[j]
                    valid = (validate(batch, p) and placement_valid(running | {j: i})
                             and isinstance(result.value, (int, float, np.integer, np.floating))
                             and not isinstance(result.value, (bool, np.bool_))
                             and np.isfinite(result.value) and result.value == full.value[j]
                             and result.residual is not None
                             and np.shape(result.residual) == expected_free.shape
                             and np.allclose(result.residual, expected_free, atol=1e-9, rtol=0))
                record['validation_status'] = 'valid' if valid else 'invalid'
                if valid:
                    node = i
                    record.update(node=i, certified=bool(method == 'highs_mip'
                                  and result.status == 'optimal' and result.optimal))
                else:
                    record['error'] = 'feasible head has no independently valid placement/value/residual'
            except Exception as exc:
                record['error'] = f'{type(exc).__name__}: {exc}'
                record['validation_status'] = 'invalid'
                if record['solver_status'] != 'input_invalid':
                    record['solver_status'] = 'exception'
            attempts.append(record)
            waiting.popleft()
            if node is None:
                finish(j, t, 'solver_error', 'feasible_head_solver_failure')
                terminal.append(jobs[j].id)
                aborted = True
                while waiting:
                    k = waiting.popleft()
                    finish(k, t, 'blocked_by_error', 'dispatch_aborted')
                    terminal.append(jobs[k].id)
                break
            finish(j, t, 'accepted', 'head_started', node)
            running[j] = node
            end = decisions[j]['end']
            endings.setdefault(end, []).append(j)
            if end not in scheduled:
                heapq.heappush(times, end)
                scheduled.add(end)
            admitted += full.demand[j]
            started.append(jobs[j].id)
        if waiting and not times:
            while waiting:
                j = waiting.popleft()
                finish(j, t, 'blocked_by_fifo', 'no_future_event')
                terminal.append(jobs[j].id)
        assert placement_valid(running), 'FIFO physical invariant violated'
        load, free = state()
        events.append(dict(time=t, arrivals=[jobs[j].id for j in group],
                           completed=[jobs[j].id for j in completed], started=started,
                           terminal=terminal, waiting=[jobs[j].id for j in waiting],
                           blocked_head=blocked_head, dispatch_aborted=aborted,
                           free_before_arrivals=before, free=free.tolist(),
                           running_load=load.tolist(),
                           running=[dict(job_id=jobs[j].id, node=i, end=decisions[j]['end'])
                                    for j, i in sorted(running.items())]))
        previous, previous_load = t, load
    assert not running and not waiting and all(d is not None for d in decisions)
    counts = Counter(d['status'] for d in decisions)
    accepted = [d for d in decisions if d['status'] == 'accepted']
    assert np.allclose(admitted, released, atol=1e-9, rtol=0)
    return dict(semantics='strict FIFO; release then arrivals then head-only dispatch',
                resource_units=['GPU-count', 'CPU-cores', 'host-RAM-GiB'],
                certification_scope='singleton attempts only; no FIFO/horizon optimum',
                decisions=decisions, events=events, attempts=attempts,
                counts={k: counts[k] for k in ('accepted', 'unschedulable', 'solver_error',
                                               'blocked_by_error', 'blocked_by_fifo')},
                accepted_count=len(accepted), completed_count=len(accepted),
                accepted_value=float(sum(full.value[j] for j in range(n)
                                         if decisions[j]['status'] == 'accepted')),
                wait_mean=sum(d['wait'] for d in accepted)/len(accepted) if accepted else None,
                wait_max=max(d['wait'] for d in accepted) if accepted else None,
                admitted_resources=admitted.tolist(), released_resources=released.tolist(),
                resource_time=resource_time.tolist(), final_free=state()[1].tolist())

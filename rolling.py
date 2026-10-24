"""Integer-time, immediate-start rolling batches; no queue, retry or migration.

Job IDs are nonblank, case-sensitive strings (no normalization). Scheduling
identity is ID, although frozen Job equality compares all fields. Arrival ties
retain input order. Eligibility and resource arrays are copied at entry.

Decisions use node=None/start=None/end=None for permanent `not_scheduled` or
`solver_error`; only the former counts as a normal rejection. Original solver
statuses/errors remain in batches. Certificates are per arrival batch, never
for the dynamic horizon. End-only events do not call the static solver.

Resource checks reuse m0_oracle.validate unchanged. Tiny negative residuals
allowed by that validator are exposed, never clipped: if the nonnegative
builder cannot represent them, that arrival batch records an input_invalid
solver_error until releases restore representable capacity. No forecast is
inferred from arrivals/durations; build_workload's neutral future is used.
Determinism concerns event order/decisions, not measured wall times or solver
tie-breaking guarantees beyond the underlying solver's own behavior.
"""
from dataclasses import dataclass
import heapq
from time import perf_counter_ns

import numpy as np
from accelsched import METHODS, solve
from m0_oracle import validate
from workload import build_workload


@dataclass(frozen=True)
class Job:
    id: str
    arrival: int
    duration: int
    demand: tuple[float, float, float]
    value: float = 1.

    def __post_init__(self):
        object.__setattr__(self, 'demand', tuple(self.demand))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _shapes(capacity):
    labels = {v: i for i, v in enumerate(sorted(set(map(tuple, capacity))))}
    return np.array([labels[tuple(v)] for v in capacity], dtype=int)


def simulate(capacity, jobs, eligible, method='first_fit', *, seed=11, time_limit=2.):
    """Simulate immediate-start jobs in arbitrary integer time units.

    Resources are (whole GPU count, CPU cores, host RAM GiB); finite nonnegative
    capacity/demand/value follow build_workload. IDs must be unique; arrivals
    >=0 and durations >0 are integers (bools excluded). Empty jobs require an
    explicit boolean eligibility array of shape (nodes, 0).

    Returns JSON-friendly decisions, post-event snapshots, batch certificates,
    accepted/rejected/error/completed counts and resource accounting. A valid
    feasible_uncertified incumbent may start jobs but has no certificate. Any
    malformed/invalid/no-placement outcome atomically refuses that batch as
    solver_error, preserving already running jobs and completed releases.
    """
    started = perf_counter_ns()
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
    cap = np.asarray(capacity)
    if cap.ndim != 2 or cap.shape[0] == 0 or cap.shape[1] != 3:
        raise ValueError('capacity must have shape (nonempty nodes, 3)')
    n, s = len(jobs), len(cap)
    eligibility = np.asarray(eligible)
    if eligibility.shape != (s, n) or eligibility.dtype != np.dtype(bool):
        raise ValueError('eligible must be a boolean (nodes, jobs) array')
    # The dummy validates an empty workload without ever solving zero jobs.
    full = build_workload(cap, [j.demand for j in jobs] if n else [[0, 0, 0]],
                          value=[j.value for j in jobs] if n else [0],
                          eligible=eligibility if n else np.zeros((s, 1), bool),
                          shapes=np.arange(s))
    full.shapes = _shapes(full.capacity)
    arrivals = {}
    for index, job in enumerate(jobs):
        arrivals.setdefault(int(job.arrival), []).append(index)
    times = list(arrivals)
    heapq.heapify(times)
    queued = set(times)
    endings, running = {}, {}
    decisions, events, batches = [], [], []
    admitted = np.zeros(3)
    released = np.zeros(3)
    resource_time = np.zeros(3)
    previous_time = 0
    previous_load = np.zeros((s, 3))
    accepted_count = rejected_count = error_count = completed_count = 0
    accepted_value = 0.

    def state(placements):
        load = np.zeros_like(full.capacity)
        for j, i in placements.items():
            load[i] += full.demand[j]
        return load, full.capacity - load

    def valid_global(placements):
        p = [-1] * len(full.value)
        for j, i in placements.items():
            p[j] = i
        return validate(full, p)

    while times:
        t = heapq.heappop(times)
        resource_time += (t-previous_time) * previous_load.sum(axis=0)
        completed = sorted(endings.pop(t, []))
        for j in completed:
            running.pop(j)  # One removal per accepted completion; never retry.
            released += full.demand[j]
        completed_count += len(completed)
        _, free = state(running)
        before = free.tolist()
        group = arrivals.get(t, [])
        accepted, rejected, errors = [], [], []
        batch_index = None
        if group:
            batch_index = len(batches)
            record = dict(time=t, job_ids=[jobs[j].id for j in group], method=method,
                          solver_status='input_invalid', validation_status='not_run',
                          error=None, reported_optimal=False, optimal=False,
                          certified_value=None, accepted_value=None, placement=None,
                          shapes=_shapes(free).tolist(), solve_wall_ms=0.)
            placement = None
            try:
                batch = build_workload(free, full.demand[group], value=full.value[group],
                                       eligible=full.eligible[:, group], shapes=record['shapes'])
            except ValueError as exc:
                record['error'] = str(exc)
            else:
                start = perf_counter_ns()
                out = None
                try:
                    try:
                        out = solve(batch, method=method, seed=seed, time_limit=time_limit)
                    finally:
                        record['solve_wall_ms'] = (perf_counter_ns()-start)/1e6
                    record.update(solver_status=out.status, error=out.error,
                                  reported_optimal=bool(out.optimal))
                    if out.status in ('feasible', 'feasible_uncertified', 'optimal') and out.placement is not None:
                        p = tuple(out.placement)
                        candidate = dict(running)
                        well_formed = (len(p) == len(group) and all(
                            _integer(i) and -1 <= i < s for i in p))
                        if well_formed:
                            candidate.update({j: int(i) for j, i in zip(group, p) if i >= 0})
                            value = float(sum(full.value[j] for j, i in zip(group, p) if i >= 0))
                            well_formed = (validate(batch, p) and valid_global(candidate)
                                           and isinstance(out.value, (int, float, np.integer, np.floating))
                                           and not isinstance(out.value, (bool, np.bool_))
                                           and np.isfinite(out.value) and out.value == value)
                        record['validation_status'] = 'valid' if well_formed else 'invalid'
                        if well_formed:
                            placement = tuple(int(i) for i in p)
                            certified = method == 'highs_mip' and out.status == 'optimal' and out.optimal
                            record.update(placement=list(placement), accepted_value=value,
                                          optimal=bool(certified), certified_value=value if certified else None)
                        else:
                            record['error'] = 'independent placement/value validation failed'
                except Exception as exc:
                    record.update(solver_status=getattr(out, 'status', 'exception'), validation_status='invalid',
                                  error=f'{type(exc).__name__}: {exc}')
            batches.append(record)
            for k, j in enumerate(group):
                node = None if placement is None or placement[k] < 0 else placement[k]
                status = 'solver_error' if placement is None else ('accepted' if node is not None else 'not_scheduled')
                end = t + int(jobs[j].duration) if node is not None else None
                decisions.append(dict(job_id=jobs[j].id, arrival=t, start=t if node is not None else None,
                                      end=end, node=node, status=status, solver_status=record['solver_status']))
                if node is not None:
                    running[j] = node
                    endings.setdefault(end, []).append(j)
                    if end not in queued:
                        heapq.heappush(times, end)
                        queued.add(end)
                    admitted += full.demand[j]
                    accepted_value += float(full.value[j])
                    accepted_count += 1
                    accepted.append(jobs[j].id)
                elif placement is None:
                    error_count += 1
                    errors.append(jobs[j].id)
                else:
                    rejected_count += 1
                    rejected.append(jobs[j].id)
        assert valid_global(running), 'rolling resource/eligibility invariant violated'
        load, free = state(running)
        events.append(dict(time=t, completed=[jobs[j].id for j in completed],
                           accepted=accepted, rejected=rejected, solver_errors=errors,
                           free_before_arrivals=before, free=free.tolist(), running_load=load.tolist(),
                           running=[dict(job_id=jobs[j].id, node=i, end=int(jobs[j].arrival)+int(jobs[j].duration))
                                    for j, i in sorted(running.items())], batch_index=batch_index))
        previous_time, previous_load = t, load
    assert not running and completed_count == accepted_count
    return dict(decisions=decisions, events=events, batches=batches,
                accepted_count=accepted_count, accepted_value=accepted_value,
                rejected_count=rejected_count, solver_error_count=error_count,
                completed_count=completed_count, admitted_resources=admitted.tolist(),
                released_resources=released.tolist(), resource_time=resource_time.tolist(),
                final_free=full.capacity.tolist(),
                solve_wall_ms=sum(b['solve_wall_ms'] for b in batches),
                elapsed_ms=(perf_counter_ns()-started)/1e6)

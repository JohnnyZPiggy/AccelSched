"""SYNTHETIC offline planner: python -m examples.time_window_capacity."""
from collections import Counter
from dataclasses import asdict
import json

import numpy as np
from rolling import Job, simulate
from examples.capacity_whatif import BASELINE, SCENARIOS


T = 6
JOBS = (Job('A-T4', 1, 3, (2, 8, 32)),
        Job('B-V100', 1, 5, (4, 16, 64)),
        Job('C-T4', 2, 2, (2, 8, 32)),
        Job('D-V100', 2, 2, (4, 16, 64)),
        Job('P-missing', 3, 1, (1, 4, 16)),
        Job('E-T4', 4, 4, (2, 8, 32)),
        Job('F-V100', 4, 4, (4, 16, 64)),
        Job('H-T4', 5, 1, (2, 8, 32)))
REQUESTS = ('T4', 'V100', 'T4', 'V100', 'P100', 'T4', 'V100', 'T4')
REFERENCES = {'T4': (2, 8, 32), 'V100': (4, 16, 64), 'P100': (1, 4, 16)}


def utilization(work, capacity, horizon):
    total = np.array(capacity).sum(axis=0)*horizon
    return [float(w/c) if c else None for w, c in zip(work, total)]


def pressure(nodes, events, horizon):
    """Fraction of [0,T) with no single compatible node fitting a reference."""
    points = [(0, [n.capacity for n in nodes])]
    points += [(e['time'], e['free']) for e in events if e['time'] < horizon]
    points.append((horizon, None))
    result = {}
    for family, demand in REFERENCES.items():
        intervals = []
        for (start, free), (end, _) in zip(points, points[1:]):
            fits = any(n.family == family and all(c >= d for c, d in zip(free[i], demand))
                       for i, n in enumerate(nodes))
            if end > start and not fits:
                if intervals and intervals[-1][1] == start:
                    intervals[-1][1] = end
                else:
                    intervals.append([start, end])
        result[family] = dict(reference=list(demand), intervals=intervals,
                              fraction=sum(b-a for a, b in intervals)/horizon)
    return result


def summarize(nodes, jobs, requests, raw, horizon):
    if (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0
            or any(not 0 <= j.arrival < horizon for j in jobs)):
        raise ValueError('positive integer horizon; all arrivals must be in [0,T)')
    lookup = {j.id: j for j in jobs}
    accepted = [d for d in raw['decisions'] if d['status'] == 'accepted']
    clipped = np.zeros(3)
    for d in accepted:
        clipped += np.array(lookup[d['job_id']].demand)*max(
            0, min(d['end'], horizon)-max(d['start'], 0))
    ids = lambda status: [d['job_id'] for d in raw['decisions'] if d['status'] == status]
    accepted_ids = ids('accepted')
    return dict(inventory=[n._asdict() for n in nodes], job_ids=[j.id for j in jobs],
                requested_families=list(requests), accepted_ids=accepted_ids,
                accepted_count=len(accepted_ids),
                accepted_value=float(sum(lookup[id].value for id in accepted_ids)),
                rejected_ids=ids('not_scheduled'), rejected_count=len(ids('not_scheduled')),
                solver_error_ids=ids('solver_error'), solver_error_count=len(ids('solver_error')),
                completed_by_T=[d['job_id'] for d in accepted if d['end'] <= horizon],
                running_at_T=[d['job_id'] for d in accepted if d['start'] <= horizon < d['end']],
                full_tail_completed_count=raw['completed_count'],
                full_tail_resource_time=list(raw['resource_time']),
                window_resource_time=clipped.tolist(),
                utilization=utilization(clipped, [n.capacity for n in nodes], horizon),
                pressure=pressure(nodes, raw['events'], horizon),
                batch_status_counts=dict(Counter(b['solver_status'] for b in raw['batches'])),
                certified_batch_count=sum(bool(b['optimal']) for b in raw['batches']),
                batches=[{k: b[k] for k in ('time', 'solver_status', 'optimal', 'error', 'solve_wall_ms')}
                         for b in raw['batches']],
                solve_wall_ms=raw['solve_wall_ms'], simulation_wall_ms=raw['elapsed_ms'])


def compare():
    rows = []
    for name, nodes in SCENARIOS:
        eligible = np.array([[n.family == f for f in REQUESTS] for n in nodes], bool)
        raw = simulate([n.capacity for n in nodes], JOBS, eligible,
                       method='highs_mip', seed=11, time_limit=.5)
        row = summarize(nodes, JOBS, REQUESTS, raw, T)
        row.update(scenario=name, added_nodes=[n.id for n in nodes[len(BASELINE):]])
        rows.append(row)
    comparable = not any(r['solver_error_count'] for r in rows)
    for row in rows:
        row['comparison'] = 'COMPARABLE' if comparable else 'NOT COMPARABLE'
        row['delta_admitted'] = row['accepted_count']-rows[0]['accepted_count'] if comparable else None
        row['delta_value'] = row['accepted_value']-rows[0]['accepted_value'] if comparable else None
        row['difference_kind'] = 'observed simulation difference' if comparable else 'NOT COMPARABLE'
    return dict(label='SYNTHETIC offline time-window capacity', T=T, seed=11,
                time_limit=.5, jobs=[dict(**asdict(j), family=f) for j, f in zip(JOBS, REQUESTS)],
                rows=rows)


def main():
    report = compare()
    print(f"{report['label']} | [0,{report['T']}] | GPU-count / CPU-cores / host-RAM-GiB")
    print('Shared jobs: ' + ', '.join(j['id'] for j in report['jobs']))
    print('Scenario | Admit/reject/error | Complete by T / running at T | Delta admit | Certified batches | Comparison')
    for r in report['rows']:
        print(f"{r['scenario']} | {r['accepted_count']}/{r['rejected_count']}/{r['solver_error_count']} | "
              f"{len(r['completed_by_T'])}/{len(r['running_at_T'])} | {r['delta_admitted']} | "
              f"{r['certified_batch_count']}/{len(r['batches'])} | {r['comparison']}")
        print('  ' + json.dumps({k: r[k] for k in (
            'inventory', 'added_nodes', 'accepted_ids', 'rejected_ids', 'solver_error_ids',
            'accepted_value', 'delta_value', 'completed_by_T', 'running_at_T',
            'window_resource_time', 'utilization', 'pressure', 'batch_status_counts',
            'solve_wall_ms', 'difference_kind')}, allow_nan=False))
    return 0 if all(r['comparison'] == 'COMPARABLE' for r in report['rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())

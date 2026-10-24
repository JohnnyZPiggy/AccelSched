"""Offline nominal-capacity comparison: python -m examples.capacity_whatif."""
import json
from typing import NamedTuple

import numpy as np
from accelsched import solve
from m0_oracle import validate
from workload import build_workload


class Node(NamedTuple):
    id: str
    family: str
    capacity: tuple


class Task(NamedTuple):
    id: str
    family: str
    demand: tuple
    value: float = 1.


# Illustrative GPU-count / CPU-cores / host-RAM-GiB, not vendor specifications.
BASELINE = (Node('t4-base', 'T4', (2, 8, 32)),
            Node('v100-base', 'V100', (4, 16, 64)))
SCENARIOS = (('Baseline', BASELINE),
             ('Add T4', BASELINE + (Node('t4-extra', 'T4', (4, 16, 64)),)),
             ('Add V100', BASELINE + (Node('v100-extra', 'V100', (4, 16, 64)),)))
TASKS = (Task('t-small-1', 'T4', (1, 4, 16)),
         Task('t-small-2', 'T4', (1, 4, 16)),
         Task('t-large-1', 'T4', (2, 8, 32)),
         Task('t-large-2', 'T4', (2, 8, 32)),
         Task('v-large-1', 'V100', (4, 16, 64)),
         Task('v-large-2', 'V100', (4, 16, 64)),
         Task('p100-missing', 'P100', (1, 4, 16)),
         Task('v-oversize', 'V100', (8, 32, 128)))


def build_instance(nodes):
    capacity = [n.capacity for n in nodes]
    labels = {c: i for i, c in enumerate(sorted(set(capacity)))}
    return build_workload(capacity, [j.demand for j in TASKS],
                          value=[j.value for j in TASKS],
                          eligible=np.array([[n.family == j.family for j in TASKS]
                                             for n in nodes], dtype=bool),
                          shapes=[labels[c] for c in capacity])


def describe(a, out, ids):
    """Expose metrics only for a validated, status-feasible public result."""
    row = dict(status=out.status, solver_optimal=bool(out.optimal), error=out.error,
               solve_wall_ms=out.elapsed_ms, valid=False, certified=False,
               comparison='NOT COMPARABLE', placement=None, accepted_ids=None,
               accepted_count=None, accepted_value=None, unplaced_ids=None,
               per_node_residual=None, total_residual=None, utilization=None,
               validation_error=None)
    if out.status not in ('optimal', 'feasible', 'feasible_uncertified') or out.placement is None:
        return row
    p = out.placement
    integer_nodes = (len(p) == len(ids) == len(a.demand) and all(
        isinstance(i, (int, np.integer)) and not isinstance(i, (bool, np.bool_))
        and -1 <= i < len(a.capacity) for i in p))
    if not integer_nodes or not validate(a, p):
        row['validation_error'] = 'invalid placement'
        return row
    load = np.zeros_like(a.capacity)
    for j, i in enumerate(p):
        if i >= 0:
            load[i] += a.demand[j]
    left = a.capacity-load
    value = float(sum(a.value[j] for j, i in enumerate(p) if i >= 0))
    if (out.value != value or out.residual is None
            or np.shape(out.residual) != left.shape
            or not np.allclose(out.residual, left, atol=1e-9, rtol=0)):
        row['validation_error'] = 'value/residual mismatch'
        return row
    certified = out.method == 'highs_mip' and out.status == 'optimal' and bool(out.optimal)
    used, total = load.sum(axis=0), a.capacity.sum(axis=0)
    row.update(valid=True, certified=certified,
               comparison='CERTIFIED' if certified else 'FEASIBLE CANDIDATE',
               placement=[int(i) for i in p],
               accepted_ids=[id for id, i in zip(ids, p) if i >= 0],
               accepted_count=int(sum(i >= 0 for i in p)), accepted_value=value,
               unplaced_ids=[id for id, i in zip(ids, p) if i < 0],
               per_node_residual=left.tolist(), total_residual=left.sum(axis=0).tolist(),
               utilization=[float(u/c) if c else None for u, c in zip(used, total)])
    return row


def compare():
    rows = []
    for name, nodes in SCENARIOS:
        a = build_instance(nodes)
        out = solve(a, method='highs_mip', seed=11, time_limit=.5)
        row = describe(a, out, [j.id for j in TASKS])
        row.update(scenario=name, inventory=[n._asdict() for n in nodes],
                   added_nodes=[n.id for n in nodes[len(BASELINE):]])
        rows.append(row)
    base = rows[0]
    for row in rows:
        comparable = base['valid'] and row['valid']
        row['delta_count'] = row['accepted_count']-base['accepted_count'] if comparable else None
        row['delta_value'] = row['accepted_value']-base['accepted_value'] if comparable else None
        row['difference_kind'] = ('static-model optimal difference' if base['certified'] and row['certified']
                                  else 'observed difference (uncertified)') if comparable else 'NOT COMPARABLE'
    return dict(label='SYNTHETIC offline capacity what-if', seed=11, time_limit=.5,
                tasks=[j._asdict() for j in TASKS], rows=rows)


def main():
    report = compare()
    print(report['label'] + ' | GPU-count / CPU-cores / host-RAM-GiB')
    print('Scenario | Added nodes | Count/value | Delta count/value | Status | Comparison | Solve ms')
    for r in report['rows']:
        print(f"{r['scenario']} | {','.join(r['added_nodes']) or '-'} | "
              f"{r['accepted_count']}/{r['accepted_value']} | {r['delta_count']}/{r['delta_value']} | "
              f"{r['status']} | {r['comparison']} | {r['solve_wall_ms']:.3f}")
        print('  ' + json.dumps({k: r[k] for k in (
            'unplaced_ids', 'per_node_residual', 'total_residual', 'utilization',
            'solver_optimal', 'certified', 'error', 'validation_error', 'difference_kind')}, allow_nan=False))
    return 0 if all(r['valid'] for r in report['rows']) else 1


if __name__ == '__main__':
    raise SystemExit(main())

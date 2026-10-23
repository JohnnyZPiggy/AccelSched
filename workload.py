"""Plain-Python workloads and six-method CPU evaluation.

Columns are always (whole GPU count, CPU cores, host RAM GiB); rows retain input
node/job order. GPU family compatibility must be encoded in eligibility, never
inferred from counts or shapes. Shapes are explicit capacity-aggregation labels.

Example: ``evaluate(heterogeneous_example())`` or ``python -m workload``.
No trace parser, hardware performance model, migration or time simulation.
"""
import json
from time import perf_counter_ns
import numpy as np
from accelsched import METHODS, Instance, _input_error, solve
from m0_oracle import validate

RESOURCE_UNITS = ('GPU-count', 'CPU-cores', 'RAM-GiB')


def build_workload(capacity, demand, *, value, eligible, shapes,
                   future=None, weights=None, future_eligible=None):
    """Copy ordered arrays into a validated Instance, without running a solver.

    Shapes: capacity (s,3), demand (n,3), value (n,), eligible (s,n) boolean,
    shapes (s,) nonnegative integers. Supply future (k,3), weights (k,) and
    future_eligible (s,k) boolean together, or omit all three for a neutral
    zero-demand, zero-weight, all-ineligible future type (no forecast signal).
    GPU columns must be whole counts, including future demands. Existing public
    input validation and capacity tolerances are reused; input order is preserved.
    Numeric resource/value arrays are copied to float64 under those conventions.
    """
    capacity = np.asarray(capacity)
    if future is None:
        if weights is not None or future_eligible is not None:
            raise ValueError('supply all future fields together or omit all')
        future, weights = np.zeros((1, 3)), np.zeros(1)
        future_eligible = np.zeros((capacity.shape[0] if capacity.ndim else 0, 1), bool)
    elif weights is None or future_eligible is None:
        raise ValueError('supply all future fields together or omit all')
    a = Instance(*(np.asarray(x).copy() for x in
                   (capacity, demand, value, eligible, shapes, future, weights, future_eligible)))
    error = _input_error(a, 1., 2026)
    if error:
        raise ValueError(error)
    if a.capacity.shape[1] != 3:
        raise ValueError('resource columns must be GPU-count, CPU-cores, RAM-GiB')
    for name in ('capacity', 'demand', 'future'):
        if np.any(getattr(a, name)[:, 0] != np.floor(getattr(a, name)[:, 0])):
            raise ValueError('GPU counts must be whole numbers')
    for name in ('capacity', 'demand', 'value', 'future', 'weights'):
        setattr(a, name, getattr(a, name).astype(float))
    return a


def heterogeneous_example(nodes=4, jobs=8, seed=2026):
    """Synthetic single-family GPU nodes A/B plus CPU-only nodes, not a trace.

    Repeated inventories: A=(4 GPUs,32 cores,128 GiB), B=(8,64,256),
    CPU-only=(0,64,256). A/B are abstract incompatible GPU families; RAM is host
    RAM, not GPU VRAM. A GPU job requires one family; CPU-only jobs use any node.
    The final job needs absent family C, guaranteeing an unplaced outcome.
    Shape labels group identical capacity vectors, following bench_cpu_v31.
    """
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) or x < 2
           for x in (nodes, jobs)):
        raise ValueError('nodes and jobs must be integers >= 2')
    rng = np.random.default_rng(seed)
    capacity = np.array([[4., 32., 128.], [8., 64., 256.], [0., 64., 256.]])[np.arange(nodes) % 3]
    node_family = np.resize([0, 1, -1], nodes)
    types = np.array([[1., 4., 16.], [2., 8., 32.], [4., 16., 64.],
                      [0., 8., 32.], [8., 32., 128.], [0., 16., 64.]])
    demand = types[rng.integers(len(types), size=jobs)].copy()
    job_family = rng.integers(2, size=jobs)
    demand[-1], job_family[-1] = [1., 4., 16.], 2
    eligible = (demand[None, :, 0] == 0) | (node_family[:, None] == job_family[None, :])
    labels = {v: i for i, v in enumerate(sorted(set(map(tuple, capacity))))}
    shapes = np.array([labels[tuple(v)] for v in capacity])
    future = np.array([[1., 4., 16.], [4., 16., 64.], [0., 8., 32.]])
    future_eligible = (future[None, :, 0] == 0) | (node_family[:, None] == np.array([0, 1, -1]))
    return build_workload(capacity, demand, value=rng.integers(1, 21, jobs),
                          eligible=eligible, shapes=shapes, future=future,
                          weights=[1., 2., 1.], future_eligible=future_eligible)


def evaluate(a, *, seed=2026, time_limit=2.):
    """Run existing public methods on the same Instance and reconstruct metrics.

    Validity uses the unchanged numerical validator, not exact-real feasibility.
    Utilization is used/total capacity per column (None when capacity is zero),
    without clipping. Zero offered value yields a None accepted-value fraction.
    Certified reference is only a validated status-optimal MIP result, within
    solver tolerances. A timeout incumbent is never promoted to that reference.
    Solve wall time includes solver model construction and the full public call;
    evaluation time separately covers independent validation and metric work.
    """
    error = _input_error(a, time_limit, seed)
    if error:
        raise ValueError(error)
    if a.capacity.shape[1] != 3:
        raise ValueError('expected three resource columns')
    total_capacity = a.capacity.astype(float).sum(axis=0)
    offered_value = float(sum(a.value))
    rows = []
    for method in METHODS:
        start = perf_counter_ns()
        out = solve(a, method, seed=seed, time_limit=time_limit)
        solve_ms = (perf_counter_ns()-start)/1e6
        start = perf_counter_ns()
        valid = None if out.placement is None else validate(a, out.placement)
        row = dict(method=method, status=out.status if valid is not False else 'invalid',
                   optimal=bool(out.optimal and valid and out.status == 'optimal'),
                   error=out.error if valid is not False else 'independent validator rejected placement',
                   placement=None if out.placement is None else list(out.placement),
                   independent_valid=valid, admitted_count=None, admitted_rate=None,
                   accepted_value=None, offered_value_fraction=None, used=None, utilization=None,
                   solve_wall_ms=solve_ms)
        if valid:
            p = out.placement
            load = np.zeros_like(a.capacity, dtype=float)
            for j, i in enumerate(p):
                if i >= 0:
                    load[i] += a.demand[j]
            count = sum(i >= 0 for i in p)
            value = float(sum(a.value[j] for j, i in enumerate(p) if i >= 0))
            if value != out.value:
                raise RuntimeError('solver value does not reconstruct from placement')
            used = load.sum(axis=0)
            row.update(admitted_count=count, admitted_rate=count/len(p), accepted_value=value,
                       offered_value_fraction=value/offered_value if offered_value else None,
                       used=used.tolist(), utilization=[float(u/c) if c else None for u, c in zip(used, total_capacity)])
        row['evaluation_ms'] = (perf_counter_ns()-start)/1e6
        rows.append(row)
    mip = rows[-1]
    reference = mip['accepted_value'] if mip['optimal'] else None
    for row in rows:
        row['certified_mip_value'] = reference
        row['value_to_mip_fraction'] = (row['accepted_value']/reference
            if reference and row['accepted_value'] is not None else None)
    return dict(resource_units=RESOURCE_UNITS, nodes=len(a.capacity), jobs=len(a.demand),
                shapes=len(np.unique(a.shapes)), eligibility_density=float(a.eligible.mean()),
                physical_mip_variables=int(a.eligible.size), physical_mip_capacity_rows=int(a.capacity.size),
                physical_mip_job_rows=len(a.demand), shape_lp_variables=len(np.unique(a.shapes))*len(a.demand),
                shape_lp_rows=len(a.demand)+3*len(np.unique(a.shapes)), offered_value=offered_value,
                total_capacity=total_capacity.tolist(), rows=rows)


if __name__ == '__main__':
    started = perf_counter_ns()
    example = heterogeneous_example()
    input_build_ms = (perf_counter_ns()-started)/1e6
    report = evaluate(example)
    report['input_build_ms'] = input_build_ms
    print(json.dumps(report, indent=2, allow_nan=False))

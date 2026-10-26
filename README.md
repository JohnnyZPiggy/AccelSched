# AccelSched

[English](README.md) | [简体中文](README.zh-CN.md)

**Heterogeneous GPU/CPU resource placement and capacity optimization.**

AccelSched is a Python optimization engine for placing indivisible jobs across heterogeneous compute nodes. It combines six scheduling methods, event-driven simulation, LP/MIP optimization, certified solver interfaces, and capacity planning reports.

![MIT](https://img.shields.io/badge/license-MIT-0f766e) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab) ![Version](https://img.shields.io/badge/version-1.0.0-0f766e) ![NumPy/SciPy](https://img.shields.io/badge/core-NumPy%20%2B%20SciPy-334155)

**Explore:** [Algorithms](#scheduling-methods) · [Quick start](#installation-and-quick-start) · [Capacity planning](#capacity-planning) · [Benchmarks](#performance-and-validation) · [Software design](docs/release/software-design.zh-CN.md)

## Highlights

- **Heterogeneous placement:** whole-GPU counts, CPU cores, host RAM, explicit GPU-family eligibility, and per-node physical capacity.
- **Six scheduling methods:** First-Fit, Best-Fit, LP-guided heuristics, future-capacity-aware PGFP, and HiGHS MIP.
- **Dynamic simulation:** arrival batches, job durations, completion-driven resource release, and admission metrics.
- **Capacity planning:** compare baseline and expanded inventories with reproducible static and time-window reports.
- **Optimization backends:** CPU HiGHS and optional certified GPU LP backends, including cuPDLPx and H-PDLP.
- **Auditable decisions:** physical feasibility checks, solver statuses, original-model LP certificates, and reproducible benchmarks.

## Optimization architecture

```text
Cluster nodes + workload + eligibility
                  │
                  ▼
       Validated resource model
                  │
      ┌───────────┴────────────┐
      ▼                        ▼
  Static placement       Rolling simulation
      │                        │
      ├─ Greedy / PGFP          ├─ Arrival batches
      ├─ LP-guided placement    ├─ Completion events
      └─ HiGHS MIP              └─ Resource release
      │                        │
      └───────────┬────────────┘
                  ▼
       Physical feasibility audit
                  ▼
       Placement + utilization
                  ▼
       Capacity decision reports
```

The optional solver portfolio supports certified GPU LP solving with a CPU fallback. The public scheduling interface remains lightweight and CPU-installable; GPU execution is an optional optimization component.

## Installation and quick start

Python 3.10+; NumPy and SciPy are the only required runtime dependencies.

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

Compare all six methods on the same heterogeneous instance:

```python
from workload import heterogeneous_example, evaluate

results = evaluate(heterogeneous_example(seed=11), seed=11, time_limit=2.0)
for row in results["rows"]:
    print(row["method"], row["status"], row["admitted_count"])
```

The lower-level [`Instance` / `solve` API](accelsched.py) returns placement decisions, objective value, residual capacities, solver status and elapsed time.

### Event-driven scheduling

```python
from rolling import Job, simulate

jobs = [
    Job("A", 0, 2, (1, 4, 16)),
    Job("B", 1, 3, (1, 4, 16)),
    Job("C", 2, 2, (1, 4, 16)),
]

report = simulate([[1, 8, 32]], jobs, [[True, True, True]], method="first_fit")
for decision in report["decisions"]:
    print(decision["job_id"], decision["status"], decision["node"])
```

The simulator releases completed jobs before processing arrivals at the same timestamp.

## Scheduling methods

| Method | Approach |
| --- | --- |
| `first_fit` | First eligible node with sufficient capacity |
| `classic_best_fit` | Smallest combined residual resources |
| `lp_order_best_fit` | LP-guided job ordering with global Best-Fit |
| `lp_best_fit` | LP-guided ordering with shape-aware Best-Fit |
| `pgfp` | Shadow-price-guided, future-capacity-aware placement |
| `highs_mip` | Physical-node mixed-integer optimization |

Resource vectors are `[GPU count, CPU cores, RAM GiB]`. Shape labels group equal capacity vectors; eligibility controls hardware-family compatibility. Optimality claims are tied to validated solver certificates and tolerances.

## Capacity planning

Bundled synthetic inputs provide ready-to-run static and event-driven scenarios:

```bash
python -m examples.capacity_whatif --input examples/data/static.json
python -m examples.time_window_capacity --input examples/data/dynamic.json
python -m examples.capacity_decision_report --input examples/data/static.json --mode static --format json
python -m examples.capacity_decision_report --input examples/data/dynamic.json --mode dynamic --format json
```

- **P0 — Static what-if:** compare node inventory expansions against Baseline.
- **P1 — Time-window capacity:** evaluate admissions, completions and resource-time utilization.
- **P2 — Decision reporting:** summarize comparable scenarios in text or JSON.

See the [input schema](docs/examples/planner-input.md), [static guide](docs/examples/capacity-whatif.md), [dynamic guide](docs/examples/time-window-capacity.md) and [report guide](docs/examples/capacity-decision-report.md).

## Performance and validation

The repository includes independently checked placement and numerical experiments.

| Study | Recorded result | Evidence |
| --- | --- | --- |
| Trace-derived static counterfactuals | 60/60 physical outcomes independently validated | [Alibaba PAI study](https://github.com/JohnnyZPiggy/AccelSched/issues/31) |
| Synthetic event-driven scheduling | 54/54 simulations audited; 313/313 MIP arrival batches optimal within solver tolerances | [Dynamic benchmark](docs/research/dynamic-scheduling-benchmark.md) |
| GPU solver + CPU placement | 0.738 s warm end-to-end on a measured 1K L4 workload; faster than tested GPU-resident greedy variants | [P6-G0 study](https://github.com/JohnnyZPiggy/AccelSched/issues/99) |
| Release validation | 264 CPU-focused tests passed in the documented environment | [Validation manifest](docs/release/validation.md) |

The benchmarks identify their workloads and measurement boundaries; the cited L4 result is one same-host experiment, not a universal speedup claim.

## Applications

| Area | AccelSched capability |
| --- | --- |
| AI compute clusters | Compare placements across heterogeneous node inventories |
| Capacity expansion | Evaluate added GPU nodes against a baseline |
| Batch scheduling | Study admission and resource utilization under arrivals/completions |
| Optimization research | Benchmark LP-guided heuristics, MIP and certified solver backends |
| Infrastructure planning | Generate reproducible static and time-window comparison reports |

## Documentation

- [Chinese software design](docs/release/software-design.zh-CN.md)
- [JSON input contract](docs/examples/planner-input.md)
- [Release notes](RELEASE_NOTES.md)
- [Validation and testing](docs/release/validation.md)
- [Chinese registration preparation](docs/release/registration-checklist.zh-CN.md)

Run focused CPU checks:

```bash
python -m pip install pytest
python -m pytest -q test_m0_oracle.py test_workload.py test_rolling.py test_capacity_whatif.py test_time_window_capacity.py test_capacity_decision_report.py
```

## License

[MIT License](LICENSE) · Copyright © 2026 Ru Zhang.

Developed with the [SAE (Scientist–Architect–Engineer) framework](https://github.com/JohnnyZPiggy/SAE-Framework).

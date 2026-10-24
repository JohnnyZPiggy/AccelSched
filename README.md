# AccelSched

[简体中文](README.zh-CN.md)

A lightweight Python toolkit for **GPU/CPU resource placement optimization**. Place indivisible jobs onto eligible servers under multi-resource capacity constraints; compare fast heuristics against a physical-server HiGHS mixed-integer optimization baseline.

## Install

Python 3.10+:

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

Requires NumPy and SciPy. No GPU or external scheduler is needed.

## Quick start

```python
import numpy as np
from accelsched import Instance, solve

jobs = Instance(
    capacity=np.array([[4., 8.], [6., 8.], [8., 12.]]),
    demand=np.array([[2., 2.], [4., 4.], [1., 2.], [3., 4.]]),
    value=np.array([6., 11., 3., 10.]),
    eligible=np.ones((3, 4), dtype=bool),
    shapes=np.array([0, 1, 2]),
    future=np.array([[1., 2.], [2., 4.], [4., 8.]]),
    weights=np.array([1., 2., 4.]),
    future_eligible=np.ones((3, 3), dtype=bool),
)

for method in ("first_fit", "classic_best_fit", "lp_order_best_fit",
               "lp_best_fit", "pgfp", "highs_mip"):
    result = solve(jobs, method=method)
    print(method, result.status, result.value, result.placement)
```

`Instance` stores server resource capacities, job demands/values, job–server eligibility and future job types. The example's resource columns represent GPUs and CPUs. `SolveResult` returns a validated placement, accepted value, residual resources, elapsed solve time, status and optimality flag.

For a synthetic three-resource comparison:

```python
from workload import heterogeneous_example, evaluate

report = evaluate(heterogeneous_example(seed=11), seed=11, time_limit=2.0)
for row in report["rows"]:
    print(row["method"], row["status"], row["admitted_count"])
```

[`workload.py`](workload.py) provides `build_workload` for ordered whole-GPU-count, CPU-core and host-RAM-GiB vectors with explicit eligibility. `evaluate` runs all six methods on the same instance and reconstructs admission and resource utilization. GPU types belong in `eligible`; `shapes` group identical capacity vectors. Trace parsing remains external.

## Rolling batches

[`rolling.py`](rolling.py) provides `Job` and `simulate` for event-driven batches using the same six methods. This synthetic example uses resource vectors ordered as whole GPU count, CPU cores and host RAM GiB:

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

The output is `A accepted 0`, `B not_scheduled None`, then `C accepted 0`. Arrival times and durations use arbitrary integer time units; accepted jobs start immediately. Completions release resources before arrivals at the same timestamp, and all simultaneous arrivals form one batch in input order. Eligibility is fixed per node/job. Rejection is permanent: there is no queue, retry, migration or preemption.

With `method="highs_mip"`, a validated `optimal` result certifies only that arrival batch within solver tolerances, never the full scheduling horizon. This example is synthetic, not an Alibaba historical allocation or production schedule.

## Solvers

| Method | Strategy |
| --- | --- |
| `first_fit` | Input-order first feasible server |
| `classic_best_fit` | Input-order smallest summed residual resources |
| `lp_order_best_fit` | Shape-LP job priority followed by global Best-Fit |
| `lp_best_fit` | Shape-LP priority and shape-aware Best-Fit |
| `pgfp` | Shape-LP priority and future-capacity-aware placement |
| `highs_mip` | HiGHS physical-server mixed-integer optimization |

An `optimal` result means HiGHS certified optimality within solver tolerances; heuristic `feasible` results are **not** optimality certificates. `feasible_uncertified` means a valid MIP incumbent without optimality proof; `uncertified` means no valid incumbent. Other statuses include `input_invalid`, `invalid` and `failed`. `solve(..., time_limit=10.0)` sets the HiGHS time limit.

## Validation and scope

Run the CPU regression from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m0_oracle.py bench_cpu.py bench_cpu_v2.py bench_cpu_v3.py bench_cpu_v31.py accelsched.py
```

AccelSched makes **single-node placement decisions for each job in a batch**. The public three-resource adapter supports synthetic workloads; trace-derived counterfactual research evidence is summarized below. It is an optimization toolkit, not a production Kubernetes scheduler; topology, job lifetime, migrations, live cluster integration and GPU acceleration are outside its scope.

### Trace-derived counterfactual benchmark

An [accepted Issue #31 study](https://github.com/JohnnyZPiggy/AccelSched/issues/31#issuecomment-6377137390) used Alibaba PAI GPU 2020 tasks and nominal-capacity empty-cluster counterfactuals, labeled `TRACE_DERIVED_ASSUMED`. Whole-GPU singleton-job samples covered explicit T4/P100/V100/V100M32 request eligibility, wrong-family nodes and separate `SYNTHETIC_CONTROL` RAM-shortage cases. The Engineer reported **60/60 independently feasible placement results** and **10/10 physical HiGHS MIP runs returning `optimal`**, within solver tolerances. Feasibility includes correctly unplaced negative controls.

Two **ASSUMED** RAM interpretations applied consistently to capacity and demand: raw values already GiB, or decimal GB converted to GiB by `10**9/2**30`. Equal task values (`value=1`) measured admission, not revenue. Deliberately stratified, individually fitting samples introduce selection bias: these static results are not population admission rates, historical Alibaba allocations or production-performance measurements.

The [Alibaba PAI 2020 trace](https://github.com/alibaba/clusterdata/tree/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020) is licensed under [CC BY 4.0](https://github.com/alibaba/clusterdata/blob/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020/LICENSE); attribution: [Weng et al., NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/weng). The source-data license is distinct from AccelSched's [MIT code license](LICENSE).

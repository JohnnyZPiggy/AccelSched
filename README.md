# AccelSched

A lightweight Python toolkit for **GPU/CPU resource placement optimization**. Place indivisible jobs onto eligible servers under multi-resource capacity constraints; compare fast heuristics against an exact HiGHS mixed-integer optimization baseline.

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

`Instance` stores server resource capacities, job demands/values, job–server eligibility and future job types. The example's resource columns represent GPUs and CPUs. `SolveResult` returns a validated placement, accepted value, residual resources, elapsed CPU time, status and optimality flag.

## Solvers

| Method | Strategy |
| --- | --- |
| `first_fit` | Input-order first feasible server |
| `classic_best_fit` | Input-order smallest summed residual resources |
| `lp_order_best_fit` | Shape-LP job priority followed by global Best-Fit |
| `lp_best_fit` | Shape-LP priority and shape-aware Best-Fit |
| `pgfp` | Shape-LP priority and future-capacity-aware placement |
| `highs_mip` | HiGHS exact physical-server mixed-integer optimization |

An `optimal` result means HiGHS certified optimality; heuristic `feasible` results are **not** optimality certificates. `feasible_uncertified` means a valid MIP incumbent without optimality proof; `uncertified` means no valid incumbent. Other statuses include `input_invalid`, `invalid` and `failed`. `solve(..., time_limit=10.0)` sets the HiGHS time limit.

## Validation and scope

Run the CPU regression from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 python -m pytest -q test_m0_oracle.py bench_cpu.py bench_cpu_v2.py bench_cpu_v3.py bench_cpu_v31.py accelsched.py
```

Version 0.1 implements **single-node placement decisions for a batch of jobs**, with synthetic GPU/CPU resource models. It is an optimization toolkit, not a production Kubernetes scheduler; topology, job lifetime, migrations, live cluster integration and GPU acceleration are outside this version's scope.

Licensed under [MIT](LICENSE).

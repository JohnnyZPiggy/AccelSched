# AccelSched

[简体中文](README.zh-CN.md) · [Static placement](#static-placement) · [Rolling batches](#rolling-batches) · [Benchmarks](#benchmarks)

A CPU-only Python toolkit for **heterogeneous GPU/CPU/RAM placement and event-driven scheduling simulation**.

Place each indivisible job on one eligible physical node under multiple resource constraints. Compare six methods on a static batch, or simulate immediate-start rolling batches with job durations and completion events. GPU-family compatibility is explicit; AccelSched is a research toolkit, not a live orchestrator.

## Install

Python 3.10+, NumPy and SciPy; no GPU or external scheduler required.

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

## Static placement

Resource vectors are ordered as **whole GPU count, CPU cores, host RAM GiB**. Run all six methods on one synthetic heterogeneous workload:

```python
from workload import heterogeneous_example, evaluate

report = evaluate(heterogeneous_example(seed=11), seed=11, time_limit=2.0)
for row in report["rows"]:
    print(row["method"], row["status"], row["admitted_count"])
```

[`workload.py`](workload.py) provides `build_workload` for custom capacities, demands, values and fixed node/job `eligible` arrays, and `evaluate` for admission and utilization comparisons. GPU families belong in `eligible`; `shapes` group identical capacity vectors, not GPU families. The lower-level [`Instance` / `solve` API](accelsched.py) returns placement, accepted value, residual resources, time and status. Trace parsing is external.

## Rolling batches

[`rolling.Job` / `simulate`](rolling.py) reuse the same resource units and six methods. This synthetic example uses one node:

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

Output: `A accepted 0`, `B not_scheduled None`, `C accepted 0`. Arrival times are nonnegative integers and durations positive integers in arbitrary time units; accepted jobs start immediately. At each timestamp, completions release resources **before** all simultaneous arrivals are solved as one batch in input order. Eligibility stays fixed; refusal is permanent, with no queue, retry, migration or preemption. `solver_error` is recorded separately from normal `not_scheduled` rejection.

## Six methods

| Method | Strategy |
| --- | --- |
| `first_fit` | Input-order first feasible server |
| `classic_best_fit` | Input-order smallest summed residual resources |
| `lp_order_best_fit` | Shape-LP job priority followed by global Best-Fit |
| `lp_best_fit` | Shape-LP priority and shape-aware Best-Fit |
| `pgfp` | Shape-LP priority and future-capacity-aware placement |
| `highs_mip` | HiGHS physical-server mixed-integer optimization |

`optimal` certifies a validated MIP optimum **within solver tolerances**; `feasible_uncertified` is a valid incumbent without proof, while `uncertified` has no valid incumbent. Heuristic `feasible` is not an optimality certificate. Other statuses include `input_invalid`, `invalid` and `failed`. The `time_limit` argument sets the HiGHS solve time limit.

Rolling uses a **neutral future**, not an inferred forecast, so it does not demonstrate predictive PGFP benefits. A rolling MIP certificate covers only the current arrival batch, never the full scheduling horizon.

## Benchmarks

### Alibaba trace-derived static counterfactual

The [accepted Issue #31 study](https://github.com/JohnnyZPiggy/AccelSched/issues/31#issuecomment-6377137390) used Alibaba PAI GPU 2020 tasks in **ASSUMED empty-cluster counterfactuals** (`TRACE_DERIVED_ASSUMED`). Engineer-reported results: **60/60 independently feasible placement results** and **10/10 physical HiGHS MIP runs returning `optimal`**, within solver tolerances. Whole-GPU singleton-job samples covered explicit T4/P100/V100/V100M32 eligibility, wrong-family nodes and separate `SYNTHETIC_CONTROL` RAM-shortage cases; feasible outcomes include correctly unplaced controls.

Two **ASSUMED** RAM interpretations apply to both capacity and demand: raw GiB, or decimal GB converted by `10**9/2**30`. Unit task values (`value=1`) measure admission, not revenue. Deliberately stratified, individually fitting samples introduce selection bias: these are not population admission rates, historical Alibaba allocations or production-performance results.

### Bounded SYNTHETIC dynamic benchmark

The [Issue #36 research report](docs/research/dynamic-scheduling-benchmark.md) records nine fixed workloads: 8 nominal heterogeneous nodes, T=40, seeds 11/29/47 and offered GPU loads 0.3/0.7/1.1. Six methods × nine workloads yielded **54/54 independently audited simulations**, zero `solver_error`, and **313/313 `highs_mip` arrival batches returning `optimal` within solver tolerances**. Audits cover fixed-window GPU/CPU/RAM resource-time utilization, event ledgers, eligibility and time-weighted placeability.

Low-pressure admission ties are retained; higher pressure sometimes separates cumulative admission. Batch-optimal MIP is not necessarily best across the horizon. This is **SYNTHETIC**, not historical scheduling or evidence of speedup or production performance. Detailed raw JSON remains local, indexed by path and SHA in the report; it is not published on GitHub.

## Validation and scope

With pytest installed, run focused CPU tests from the checkout:

```bash
python -m pytest -q test_rolling.py test_dynamic_bench.py
```

See [rolling tests](test_rolling.py) and the [dynamic metric oracle](test_dynamic_bench.py). Simulation supports job duration and completion events; it does not model multi-node/gang placement, topology, device VRAM, runtime performance or live Kubernetes integration. GPU execution, queueing, migration and preemption are outside scope.

## Attribution and licenses

The [Alibaba PAI 2020 trace](https://github.com/alibaba/clusterdata/tree/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020) is licensed under [CC BY 4.0](https://github.com/alibaba/clusterdata/blob/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-gpu-v2020/LICENSE); attribution: [Weng et al., NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/weng). That source-data license is distinct from AccelSched's [MIT code license](LICENSE).

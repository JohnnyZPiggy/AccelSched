# AccelSched 1.0.0 release candidate

Independent AIDC optimization engine. Chinese filing name: **AccelSched 异构算力资源优化调度软件 V1.0** (display V1.0 maps to package 1.0.0). This checkout prepares a release; no tag, published wheel or software registration is implied. [Release notes](RELEASE_NOTES.md) · [中文设计说明书](docs/release/software-design.zh-CN.md) · [Registration/ownership checklist](docs/release/registration-checklist.zh-CN.md).

[简体中文](README.zh-CN.md) · [Static placement](#static-placement) · [Rolling batches](#rolling-batches) · [P0](#p0-static-what-if) · [P1](#p1-time-window-capacity) · [P2](#p2-capacity-decision-report) · [Six methods](#six-methods) · [Benchmarks](#benchmarks)

A CPU-based Python toolkit for **offline heterogeneous GPU/CPU/host RAM placement, event-driven simulation and capacity decision reporting**.

Place each indivisible job on one eligible physical node under multiple resource constraints. Compare six methods on a static batch, or simulate immediate-start rolling batches with job durations and completion events. GPU-family compatibility is explicit; AccelSched is a research toolkit, not a live orchestrator.

## Install

Python 3.10+, NumPy and SciPy; no GPU or external scheduler required.

```bash
git clone https://github.com/JohnnyZPiggy/AccelSched.git
cd AccelSched
python -m pip install .
```

Build/install a wheel from a reviewed checkout (CPU only; no GPU packages required):

```bash
python -m pip wheel --no-deps --wheel-dir dist .
python -m pip install dist/accelsched-1.0.0-py3-none-any.whl
python -c "from importlib.metadata import version; print(version('accelsched'))"
```

From the checkout root, the bundled **synthetic nominal** inputs give Baseline/Add T4 accepted counts 1/2 (static), 2/3 (dynamic), with missing-family requests legitimately rejected:

```bash
python -m examples.capacity_whatif --input examples/data/static.json
python -m examples.time_window_capacity --input examples/data/dynamic.json
python -m examples.capacity_decision_report --input examples/data/static.json --mode static --format json
python -m examples.capacity_decision_report --input examples/data/dynamic.json --mode dynamic --format json
```

The wheel also installs `examples` and these inputs. Outside a checkout, create a new `demo-inputs` directory (the command refuses an existing directory), then use `--input demo-inputs/static.json` / `--input demo-inputs/dynamic.json`:

```bash
python -c "from importlib.resources import files; from pathlib import Path; d=Path('demo-inputs'); d.mkdir(); [d.joinpath(n+'.json').write_bytes(files('examples').joinpath('data',n+'.json').read_bytes()) for n in ('static','dynamic')]"
```

Family labels are illustrative eligibility labels, not vendor performance or VRAM specifications. CPU default behavior is unchanged; optional certified `hpdlp`/`cupdlpx`/portfolio APIs do not replace default PGFP. GPU research probes and FIFO research files are not installed as V1 CLI features. Python >=3.10 is declared; see the exact tested environment and remaining release gates in the [release validation manifest](docs/release/validation.md).

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

## Capacity planning: P0/P1/P2

From the checkout root, use the bundled `examples/data/static.json` and
`examples/data/dynamic.json`, or prepare your own files. The [Shared JSON Input v1 guide](docs/examples/planner-input.md)
includes an illustrative JSON document and explains how to prepare both modes.

### P0: Static what-if

P0 compares nominal additive static inventories against a named Baseline.

```bash
python -m examples.capacity_whatif --input static.json
```

### P1: Time-window capacity

P1 compares observed finite-window admissions, completions and pressure for explicit references.

```bash
python -m examples.time_window_capacity --input dynamic.json
```

### P2: Capacity decision report

P2 turns P0/P1 outputs into text or JSON reports: its CLI runs the selected planner once per scenario, then summarizes without re-solving.

```bash
python -m examples.capacity_decision_report --input static.json --mode static
python -m examples.capacity_decision_report --input dynamic.json --mode dynamic --format json
```

Baseline-relative deltas are observed unless backed by validated P0 static-model
optimality certificates within solver tolerances; rolling MIP certificates cover
only individual arrival batches, and the descriptive leader means largest accepted
count, not a cost-optimal purchase, forecast, ROI or whole-horizon optimum.
See the [P0 guide](docs/examples/capacity-whatif.md),
[P1 guide](docs/examples/time-window-capacity.md) and
[P2 report guide](docs/examples/capacity-decision-report.md) for interpretation and limits.

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

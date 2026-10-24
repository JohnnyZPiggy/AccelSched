# Time-window GPU capacity planner

Offline **SYNTHETIC** immediate-start simulation on the same immutable task stream, comparing Baseline / Add T4 / Add V100. This extends [P0's static example](capacity-whatif.md) with duration, completion and fixed-window interpretation; it is not a new scheduler or a prediction/cost model.

```bash
python -m examples.time_window_capacity
```

Run from the checkout root in the existing Python/NumPy/SciPy environment. The [source](../../examples/time_window_capacity.py) imports unchanged `rolling.Job/simulate` and P0 inventories. Programmatic `compare()` returns a JSON-serializable report; stdout gives a small comparison table and per-scenario detail. No new dependencies, packaging or path changes.

## Fixed example

Resources are `(whole GPU count, CPU cores, host RAM GiB)`. Baseline has T4 `(2,8,32)` and V100 `(4,16,64)`; alternatives add exactly one T4 `(4,16,64)` or one V100 `(4,16,64)`, preserving both baseline nodes. Capacities are nominal illustrations, not vendor specifications. Fresh eligibility matrices use exact requested-family equality, never family substitution. Rolling derives capacity shapes itself; its neutral future is unchanged.

Window T=6, all arrivals strictly before T, unit values, arbitrary integer time units:

| ID | Family | Arrival | Duration | Demand |
| --- | --- | --- | --- | --- |
| A-T4 | T4 | 1 | 3 | (2,8,32) |
| B-V100 | V100 | 1 | 5 | (4,16,64) |
| C-T4 | T4 | 2 | 2 | (2,8,32) |
| D-V100 | V100 | 2 | 2 | (4,16,64) |
| P-missing | P100 | 3 | 1 | (1,4,16) |
| E-T4 | T4 | 4 | 4 | (2,8,32) |
| F-V100 | V100 | 4 | 4 | (4,16,64) |
| H-T4 | T4 | 5 | 1 | (2,8,32) |

Overlap causes normal rejection in Baseline; A completes at 4 before E arrives, B completes exactly at T, E crosses T, and Add T4 can complete H exactly at T. Rejection is permanent: no queue, retry, migration or preemption. Each execution makes three `simulate` calls using `highs_mip`, seed 11 and time_limit=.5 seconds per arrival batch, with no retries or method sweep.

## Window and pressure semantics

Admissions/rejections/errors refer to arrivals in [0,T). Accepted IDs and values come only from `accepted` decisions; `not_scheduled` and `solver_error` are separate lists/counts. Completed-by-T means `end <= T`; running-at-T means `start <= T < end`. Simulator `completed_count` and `resource_time` run through the last completion and are reported separately as full-tail evidence.

Window work for resource k is `sum(demand[j,k] * max(0,min(end[j],T)-max(start[j],0)))` over accepted decisions. Utilization divides by `T * sum(node_capacity[:,k])`, with `None` on zero total. Tests independently integrate post-event load intervals, including [0,first event), the final interval and cutoff tail; they do not divide raw full-tail area by T.

Pressure uses one fixed reference per family: T4 `(2,8,32)`, V100 `(4,16,64)`, P100 `(1,4,16)`. At each post-event state, a family is under pressure if **no single compatible node** can fit that request in all three residual resources. Report merged half-open intervals within [0,T) and their duration/T. Pooled free GPUs do not establish placeability; missing P100 has pressure 1. This is an observed reference-request diagnostic, not rejected-job causality, a future forecast or a hardware-fragmentation proof.

Only original `batches[].optimal` flags count as arrival-batch certificates within solver tolerances. Even if every batch is certified, admission/value deltas are always **observed simulation differences**, never horizon-global optimal differences. Any solver_error in any scenario marks all comparisons **NOT COMPARABLE** and suppresses all deltas; independently valid accepted work remains diagnostic, not a failure-derived rejection list. Valid uncertified batches retain their actual statuses and are not promoted to certificates.

## Validation and actual evidence

```bash
python -m pytest -q -p no:cacheprovider test_time_window_capacity.py
```

The [focused oracle](../../test_time_window_capacity.py) reconstructs original-job/node occupancy and eligibility independently, verifies event release/conservation/no migration, exact-once completion, clipped/full event integrals, initial/final capacity, T-boundary counts, hand pressure and no pooled-GPU shortcut. Integer fixtures require exact accounting. Crafted error/uncertified outputs add no numerical calls; an error in either Baseline or Add V100 suppresses all comparisons, while a valid uncertified batch loses its certificate but retains its actual admitted/rejected decisions. The oracle reinterprets the same retained reports at T=10 to check an empty final interval, without another simulation.

At exact base `b1ead989583d08e7e518d21393e30b475f701948`, the warmed Windows CPU environment (Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0) ran the focused tests **once: 9 passed in 1.37 s**, then the module smoke **once: exit 0**. Commands used `D:/software/anaconda3/python.exe`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; 30 s outer timeout per subprocess. Outer walls were 2.658449 s and 1.360734 s. Three real scenario calls in the shared test fixture plus three in the smoke, each five arrival batches: **six simulations / 30 MIP batches total**, all batches reported validated `optimal` within solver tolerances. No retries, extra seeds or old benchmark reruns.

| Smoke scenario | Admit / reject / error | Complete by T / running at T | Window GPU/CPU/RAM work | GPU/CPU/RAM utilization | Certified batches |
| --- | --- | --- | --- | --- | --- |
| Baseline | 3 / 5 / 0 | 2 / 1 | (30,120,480) | (0.833333,0.833333,0.833333) | 5/5 |
| Add T4 | 5 / 3 / 0 | 4 / 1 | (36,144,576) | (0.6,0.6,0.6) | 5/5 |
| Add V100 | 5 / 3 / 0 | 3 / 2 | (46,184,736) | (0.766667,0.766667,0.766667) | 5/5 |

Accepted values equal counts, and observed admission/value deltas are 0/+2/+2. All three actual scenarios were comparable with zero solver_error. Full-tail GPU work is 34/40/58, not 30/36/46; all full-tail completions are 3/5/5, not the by-T counts. Cumulative smoke solve walls were 13.7601/10.7313/10.6883 ms, instrumentation only, not a speedup claim.

Pressure intervals and fractions:

| Smoke scenario | T4 pressure | V100 pressure | P100 pressure |
| --- | --- | --- | --- |
| Baseline | [1,6): 5/6 | [1,6): 5/6 | [0,6): 1 |
| Add T4 | none: 0 | [1,6): 5/6 | [0,6): 1 |
| Add V100 | [1,6): 5/6 | [2,6): 4/6 | [0,6): 1 |

Baseline accepts A/B/E; Add T4 also accepts C/H; Add V100 also accepts D/F. Every remaining ID is a normal rejection, including P-missing in all inventories. At T, E remains running in all scenarios, and F also runs in Add V100. Adding capacity can lower utilization while increasing admissions because the denominator also grows; no cost recommendation follows from these nominal inputs.

Raw test/smoke stdout remains local outside the checkout in `D:/Dropbox/study/ms-phd/PHD-EPFL/Doctoral proposal/EPFL/small interesting research/AccelSched/.sae-issue42/`, files `focused-tests.txt` and `module-smoke.txt` (not downloadable from GitHub). The worktree and evidence are retained for independent review; no bytecode/pytest caches, data downloads, GPU/CI, new dependencies or paid resources were created.

No GPU execution, live cluster/Kubernetes, device VRAM, topology/runtime model, throughput/speedup or real-world compatibility claim. Production rolling/PGFP/solvers and existing benchmarks remain unchanged.

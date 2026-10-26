# Offline GPU capacity what-if

Compare one shared static task batch under **Baseline**, **Add T4** and **Add V100** inventories. This thin SYNTHETIC example calls existing `build_workload`, `solve(method='highs_mip')` and `validate`; it changes no scheduler or mathematical model. No prediction or live cluster is involved.

## Run from the checkout root

Use the project's existing Python/NumPy/SciPy environment; no extra package is needed for the example:

```bash
python -m examples.capacity_whatif
```

The module invocation works from the repository root without path modifications or packaging changes. Source: [capacity_whatif.py](../../examples/capacity_whatif.py). Programmatic `compare()` returns a JSON-serializable report; stdout prints a compact table plus per-scenario residuals, utilization and status details.

## Fixed model

Resources are `(whole GPU count, CPU cores, host RAM GiB)`. Capacities are illustrative nominal inputs, **not vendor hardware specifications**.

| Scenario | Inventory delta from Baseline |
| --- | --- |
| Baseline | T4 `(2,8,32)` + V100 `(4,16,64)` |
| Add T4 | Baseline + one T4 `(4,16,64)` |
| Add V100 | Baseline + one V100 `(4,16,64)` |

The immutable ordered eight-task set has unit values: two T4 `(1,4,16)`, two T4 `(2,8,32)`, two V100 `(4,16,64)`, one P100 `(1,4,16)` and one V100 `(8,32,128)`. Exact requested-family equality determines each fresh boolean eligibility matrix. P100 is absent, and the oversized V100 request cannot span nodes, so both remain unplaced under every scenario. Other rejections are capacity-limited; tied solutions may choose different identical task IDs.

`shapes` group equal capacity vectors, not families: the added T4 shares a shape with the baseline V100 but has different eligibility. All scenarios reuse exactly the same ordered demands and values. Omitted future fields produce the existing neutral future; no forecast is inferred.

Each scenario makes one MIP call with seed 11 and `time_limit=0.5` seconds, no retries. This example uses only the primary MIP comparison; optional six-method `evaluate` is not run. Output reports count/value, observed differences from Baseline, unplaced IDs, per-node/total residual resources, utilization and actual solver status/optimal/error/timing. Zero-total resource utilization is `None`.

Only validated placements with feasible solver statuses get admission/rejection metrics. `feasible_uncertified` is a feasible candidate, not a certificate; failures/no valid incumbent are **NOT COMPARABLE**, with no invented unplaced list or admission delta. A static-model optimal difference is labeled only when both MIP results are validated, status `optimal` and flag `optimal=True`, within solver tolerances. Otherwise any valid difference is observed and uncertified.

## Validation

Focused tests independently reconstruct eligibility, integer resource loads/residuals, IDs, values and rejection accounting; check additive inventories, shared inputs, capacity-based shapes, zero-capacity utilization, failure handling and a tiny baseline brute-force optimum. These integer fixtures use exact resource checks; displayed public residual validation uses the unchanged absolute `1e-9` convention.

```bash
python -m pytest -q -p no:cacheprovider test_capacity_whatif.py
```

See [focused tests](../../tests/test_capacity_whatif.py). At baseline `07e225fe282c4a629e8950399f03127dba5853bc`, the existing Windows CPU environment (Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0) ran the focused file **once: 12 passed in 1.94 s**, and the example smoke **once: exit 0**. Commands used `D:/software/anaconda3/python.exe` with `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`; outer process walls were 3.527201 s and 1.405356 s. Three real MIP calls in the shared test fixture plus three in the smoke: six total; no retries or optional six-method comparison.

| Smoke scenario | Accepted count/value | Difference from Baseline | Status / certificate | Solve wall ms |
| --- | --- | --- | --- | --- |
| Baseline | 3 / 3 | 0 / 0 | optimal / true | 6.943 |
| Add T4 | 5 / 5 | +2 / +2 | optimal / true | 23.000 |
| Add V100 | 4 / 4 | +1 / +1 | optimal / true | 28.074 |

Focused tests independently checked each scenario's solver placement with the integer ledger and certificate checks; the smoke reported the same certified counts. Every smoke per-node and total residual vector was `(0,0,0)`, with utilization `(1,1,1)`; this is resource occupancy, not execution throughput. Smoke unplaced IDs:

- Baseline: `t-large-1`, `t-large-2`, `v-large-2`, `p100-missing`, `v-oversize`.
- Add T4: `v-large-2`, `p100-missing`, `v-oversize`.
- Add V100: `t-large-1`, `t-large-2`, `p100-missing`, `v-oversize`.

The baseline brute-force optimum was 3. All six real MIP calls returned validated `optimal` results, within solver tolerances; no failures occurred. Single-shot times are instrumentation, not performance comparisons. Raw command outputs are retained locally outside the checkout in `.sae-issue40/focused-tests.txt` and `.sae-issue40/example-smoke.txt`; they are not published as downloadable artifacts. No caches, downloads, GPU/CI, new dependencies or paid resources were created.

This is offline static-model admission planning, not a cost/ROI estimator, runtime throughput benchmark, real-world compatibility proof or horizon-global scheduling certificate. It does not model GPU VRAM, topology or live Kubernetes, and uses no GPU or new service.

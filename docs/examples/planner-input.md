# Shared JSON Input v1

Use your own nominal inventories and one shared job set with the existing offline planners, from the checkout root:

```bash
python -m examples.capacity_whatif --input static.json
python -m examples.time_window_capacity --input dynamic.json
```

Both entry points use [one parser/validator](../../examples/planner_input.py). Without `--input`, the existing P0/P1 SYNTHETIC demos, defaults and output structure remain unchanged. External inputs are labeled `USER_INPUT`, without inferring their provenance or real-world validity. `--help` describes the input option. No installation, new dependency or solver is required beyond the existing project environment.

## Schema

UTF-8 JSON object, with exact field names and no extra fields at any level. IDs, families and names are nonblank, case-sensitive strings of at most 128 characters; whitespace is not normalized. Duplicate object keys are rejected, including nested keys. NaN, Infinity, overflow, booleans used as numbers and non-integer timing fields are invalid.

| Field | Contract |
| --- | --- |
| `schema_version` | Required integer `1`, not bool or `1.0` |
| `mode` | Required `static` or `dynamic`, matching the selected entry point |
| `nodes` | 1–32 baseline nodes: `{id, family, capacity}`; unique IDs |
| `jobs` | 1–64 ordered jobs: `{id, family, demand, value?}`; unique IDs; value defaults to 1 |
| `scenarios` | 2–8 ordered `{name, add_nodes}` objects with distinct names; exactly one `Baseline` with zero additions, plus at least one alternative |
| `solver` | Optional object; defaults `{method: "highs_mip", seed: 11, time_limit: 0.5}`; omitted members use those defaults |
| `horizon` | Dynamic only, required integer 1–1,000,000 |
| `pressure_references` | Dynamic only, optional 0–16 `{family, demand}` objects, with distinct families |

Node additions have the same schema as baseline nodes. Every scenario is baseline nodes followed by its own additions; the combined inventory must have at most 32 unique IDs. An addition ID may be reused in a different alternative, but cannot collide within an inventory. Baseline may occur anywhere in the scenario list: order is preserved and differences use its name, not its position. Empty additions in alternatives are valid.

Capacity/demand vectors contain exactly three finite, nonnegative numbers: **whole GPU count, CPU cores, host RAM GiB**. Bounds are GPU ≤1024, CPU/RAM ≤1,000,000; GPU `1.0` is a whole count, unlike bool. Zero-capacity resources are valid; utilization is `None` when their total is zero. Demands (including pressure references) must use at least one resource. Job values are finite in [0,1,000,000,000]. These explicit v1 limits and a 1 MiB file limit keep this a bounded offline application, not a generic ingestion system.

Dynamic jobs additionally require integer `arrival` in [0,horizon) and `duration` in [1,1,000,000]. Static jobs must omit timing fields; static inputs must also omit horizon and pressure references. Solver v1 supports only `highs_mip`, integer seed in [0,2³¹−1] and 0 < time_limit ≤0.5 seconds per solver call/arrival batch. Unknown methods, invalid bounds and missing required fields fail before any solve/simulate call.

## Small dynamic example

Save this illustrative document as `dynamic.json`; the smaller release smoke inputs are bundled separately as [`static.json`](../../examples/data/static.json) / [`dynamic.json`](../../examples/data/dynamic.json):

```json
{
  "schema_version": 1,
  "mode": "dynamic",
  "horizon": 6,
  "nodes": [
    {"id": "t0", "family": "T4", "capacity": [2, 8, 32]},
    {"id": "v0", "family": "V100", "capacity": [4, 16, 64]}
  ],
  "jobs": [
    {"id": "early", "family": "T4", "demand": [2, 8, 32], "arrival": 1, "duration": 3},
    {"id": "later", "family": "T4", "demand": [2, 8, 32], "arrival": 4, "duration": 4}
  ],
  "scenarios": [
    {"name": "Baseline", "add_nodes": []},
    {"name": "Add T4", "add_nodes": [{"id": "t1", "family": "T4", "capacity": [4, 16, 64]}]}
  ],
  "pressure_references": [{"family": "T4", "demand": [2, 8, 32]}]
}
```

For a static file, change mode to `static` and remove horizon, pressure_references, and each job's arrival/duration. The remaining inventory/job/scenario structure is shared. For programmatic use, `load_input(path, "static"|"dynamic")` returns explicit validated configuration for the corresponding planner's `compare(config)`. No module-level constants are mutated.

## Semantics and provenance

Exact requested-family equality creates fresh eligibility matrices; missing families are legitimate potentially rejected requests. P0 derives shapes from capacity equality, not family. P1 reuses unchanged rolling completion-before-arrival batching, permanent refusal and neutral future. Missing or empty external pressure_references disables that diagnostic (`{}`); no reference or forecast is invented. Explicit references use P1's per-node residual fit and time-weighted blocked intervals, including absent families. Default no-input P1 still uses its original three references.

Static certificates require validated MIP `optimal` status and flag; valid `feasible_uncertified` results are candidates only. Dynamic certificates remain **per arrival batch**, and all horizon differences remain observed simulation differences. A dynamic solver_error suppresses all comparisons/deltas and is never a normal rejection; static no-incumbent failures have no unplaced list. Schema errors print `INPUT_INVALID: field.path: reason` to stderr and exit 2, distinct from solver statuses and `not_scheduled`.

External output includes schema/mode, exact raw-file SHA-256, normalized solver parameters, horizon, and input order (baseline IDs, job IDs, scenario names and reference families). Added-node order remains visible in each scenario inventory. Git HEAD SHA and dirty flag are recorded when Git is available, otherwise null; HEAD identifies the checkout commit, not an assertion that uncommitted files belong to that commit. Timing and solver tie breaks need not be identical on another run. `USER_INPUT` is not a historical scheduling or hardware compatibility claim.

See [P0](capacity-whatif.md) and [P1](time-window-capacity.md) for existing resource, window, pressure and certificate semantics. No VRAM/topology/runtime model, live cluster, cost/ROI or performance prediction is added.

## Focused validation

```bash
python -m pytest -q -p no:cacheprovider test_planner_input.py
```

[Tests](../../test_planner_input.py) mirror the built-in eight-job P0/P1 inputs as external JSON and independently reconstruct integer ledgers, family compatibility, capacity shapes, clipped/event/full integrals, completions and pressure; also check validation-before-solve, default entry paths, order, provenance, zero-total utilization, missing families and error/certificate boundaries.

At base `e524cb2218d9525f5418d717373c0ff1d6603707`, one focused invocation returned **66 passed in 2.65 s** (exit 0, outer 3.530238 s); one external-input CLI smoke per mode returned **exit 0** (static 1.204221 s, dynamic 1.088871 s outer walls). Existing Windows CPU environment: Python 3.14.6 / NumPy 2.4.6 / SciPy 1.18.0; `D:/software/anaconda3/python.exe`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. Pytest additionally used `--basetemp ../pytest-temp` on D:; each of the three subprocesses had a 30 s timeout. No old test/benchmark sweep or numerical retry.

| External smoke, Baseline / Add T4 / Add V100 | Result |
| --- | --- |
| Static accepted count/value | 3/3, 5/5, 4/4; all three validated MIP `optimal` |
| Dynamic admit / reject / error | 3/5/0, 5/3/0, 5/3/0 |
| Dynamic completed by T / running at T | 2/1, 4/1, 3/2 |
| Dynamic clipped GPU/CPU/RAM work | (30,120,480), (36,144,576), (46,184,736) |
| Dynamic certificates | 5/5 original arrival-batch flags per scenario, within solver tolerances |

The focused fixture runs defaults and equivalent external inputs (six static solves and six dynamic simulations); each CLI adds three scenarios in its mode. All other error/status/default-entry tests use mocks or parser-only calls. Static certificates remain static-model certificates; dynamic differences are observed, not horizon-global. Missing-family jobs remain ordinary valid rejections. Both smokes report Git HEAD=the base above and `repo_dirty=true` because the tested changes were not yet committed.

Retained external fixture raw SHA-256:

- Static: `ea14826ed8b3a3feb4060a020a9598430381c26d8609f348d017ecd29068bef1`.
- Dynamic: `99bbad010a445dbf4388f2e59a2097dbba26d7104552dddd97f913d3497d6a0d`.

Fixtures (`static-input.json`, `dynamic-input.json`), exact stdout (`focused-tests.txt`, `static-smoke.txt`, `dynamic-smoke.txt`) and `commands.json` are retained locally outside the checkout under `D:/Dropbox/study/ms-phd/PHD-EPFL/Doctoral proposal/EPFL/small interesting research/AccelSched/.sae-issue44/`, not published GitHub downloads. D: test scratch (`pytest-temp/`) is also retained. No data downloads, GPU/CI, new dependencies or paid resources.

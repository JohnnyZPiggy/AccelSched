# Offline capacity decision report (P2)

Run from the checkout with an existing [Shared JSON Input v1](planner-input.md):

```bash
python -m examples.capacity_decision_report --input static.json --mode static
python -m examples.capacity_decision_report --input dynamic.json --mode dynamic --format json
```

Default text output compares admission count/value, differences from the named
Baseline and a descriptive leader by accepted count. JSON output (`--format json`)
also includes provenance, scenario evidence and resource diagnostics. Nothing is
written to disk by default. Invalid inputs return exit 2 with `INPUT_INVALID`;
invalid report data or noncomparable results return exit 1. Successful reports
return 0 and serialize with `allow_nan=False`.

The input schema is unchanged: baseline nodes and ordered jobs, one named Baseline
and explicit additive alternatives, `highs_mip`, seed 11 and at most .5 seconds per
solve/arrival batch. Use the existing v1 guide for field bounds and units.
GPU family requests determine eligibility; capacity equality determines shapes.

The CLI parses once and invokes exactly one existing P0 or P1 `compare(config)`;
that planner runs once per scenario. Programmatic
`summarize_report(raw_report, mode)` only transforms an already computed report:
it does not parse, solve, simulate or mutate its input. The returned data is detached
from the original. Missing/malformed scenarios and nonfinite values fail clearly.

`report_version=1`, mode, input provenance, baseline, ordered scenarios and
comparison_status form the report. The original input hash/order/solver and Git
dirty/unknown state are preserved; no device identity is inferred. Counts/value
differences are recomputed against the named Baseline, even in the middle. A
unique largest accepted count produces a descriptive leader; ties list names in
user order. There is no ranking by cost, value, future performance or ROI.

Static results retain validated accepted/unplaced IDs, residuals, utilization and
solver status. Only validated status-optimal MIPs are certified within solver
tolerances. Feasible uncertified incumbents remain observed candidates. Any
invalid scenario suppresses every delta and the leader; failure does not imply
that all jobs were rejected.

Dynamic results retain admitted value/count, rejection and solver-error counts,
completion/running counts at T, clipped resource-time, utilization and batch
evidence. Dynamic comparisons are always observed simulations; batch MIP
certificates never prove horizon optimality. Any solver error suppresses every
delta and the leader. Pressure is the fraction of time no single compatible
physical node fits the explicit reference; empty references yield `{}` and
`pressure_status=UNAVAILABLE`. No forecast is invented.

Resources are whole GPU count / CPU cores / host RAM GiB. Window work has resource
times time units, utilization uses the [0,T] window, and zero-total resource
utilization stays null. No historical utilization, VRAM/topology, hardware
equivalence, throughput, latency, energy, purchasing or live-cluster claim follows.

## Bounded validation

The focused test uses two tiny shared planner comparisons, static enumeration and
an independent dynamic original-job/event ledger. Mocked reports cover errors,
uncertified incumbents, ties, middle Baseline, zero-total utilization, unavailable
references, malformed/nonfinite data, immutability and old CLI boundaries without
additional solves. GPU_NOT_RUN; no new dependencies, CI or dataset download.

Executed once at source `32ff36c8afcc0349b13188e38639918e47e4c15a` on Windows 11,
Python 3.14.6 / NumPy 2.4.6 / SciPy 1.18.0, seed 11 and .5 s per MIP. The final
follow-up changes only this evidence section; tested module/test blobs remain
`c1065756a7b3699ac4d23dd58a56376a20fb8ec1` and
`edc2f5d41c1378c48dd7d74afbae108af4443129`.

From the checkout, with existing `D:/software/anaconda3/python.exe`:

```bash
python -B -m pytest -q -p no:cacheprovider --basetemp ../tmp/pytest test_capacity_decision_report.py
python -B -m examples.capacity_decision_report --input ../static-input.json --mode static --format json
python -B -m examples.capacity_decision_report --input ../dynamic-input.json --mode dynamic --format json
```

Actual subprocess arguments used resolved absolute D: paths for basetemp/inputs.
Batch environment: PYTHONDONTWRITEBYTECODE=1, PYTHONIOENCODING=utf-8,
OMP/OPENBLAS/MKL_NUM_THREADS=1; TEMP/TMP/TMPDIR pointed at task-owned D: scratch.
Outer limits: 60 s for pytest, 30 s per CLI. All three exited 0, with no retries:

| Gate | Actual outcome | Outer wall seconds |
| --- | --- | ---: |
| Focused pytest | 20 passed in 1.18 s | 2.102845 |
| Static CLI | Add Node / Baseline / No change counts 2/1/1, values 5/3/3; all certified; Add Node delta +1/+2 | 1.133772 |
| Dynamic CLI | Counts 3/2/2, values 6/5/5; Add Node delta +1/+1; window work (6,24,96) vs (4,16,64); pressure .5 vs 1 | 1.074082 |

Dynamic results have zero solver errors, four validated optimal arrival batches
per scenario, completed-by-T 2/1/1 and running-at-T 1/1/1. These are observed
simulation differences, not horizon optima. Both fixtures put Baseline in the
middle. The test's `input_document(mode)` defines reproducible fixtures;
UTF-8 JSON with indent 2 and a final newline gives:

| Local evidence | Input bytes | SHA-256 |
| --- | ---: | --- |
| static-input.json | 1248 | `0e45339bf16ba760ecab8fbe2bbcc0d4fa959492a1403d4aefb5eb4ed9fa9d26` |
| dynamic-input.json | 1573 | `1f387d3a54ec7b046c0f0c8279dd8a7cc429c9d263cf80c35064ff345f4fef41` |
| focused-tests.stdout | — | `76d6b7ce495a7fae1f928fdaa1c199f5f2ca7bd57c320a883970d289c1c636ad` |
| static-smoke.stdout | — | `abd5f788354df1505af1f1e35743dbf5ab8c025417bb2929cc9f3d740318c28d` |
| dynamic-smoke.stdout | — | `dc7f17db4b76f0089e1957320ce38f1e429819adfd87b7174f314dbfe0a78437` |

Raw stdout, empty stderr, exact commands/environment and fixtures remain local
under `.sae-issue46/`, outside the checkout; these are not GitHub downloads.
Owned pytest scratch under `tmp/` remains: automatic approval rejected deletion
with only "blocked by policy"; no alternative cleanup was attempted. No other
tests/benchmarks ran. Existing CLI boundaries were checked with mocks; all
existing tracked file blobs are unchanged. No global environment settings changed.

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
additional solves. Commands and actual one-batch outcomes are recorded below after
execution. GPU_NOT_RUN; no new dependencies, CI or dataset download.

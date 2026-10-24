# Bounded synthetic dynamic scheduling benchmark

Issue [#36](https://github.com/JohnnyZPiggy/AccelSched/issues/36); baseline `e05316ed2c9b75ddad77e5689ba4035717a8ab51`. Research-only **SYNTHETIC**, not trace-derived. Production modules are unchanged; `rolling.py` blob is `9d69b564562d210df5f9b10f7133c5c60b6f42da`.

## Protocol

Eight nominal nodes, two per family: T4 `(4,32,128)`, P100 `(4,48,192)`, V100 `(8,64,256)`, V100M32 `(8,96,384)`. Columns are whole GPU count, CPU cores, host RAM GiB; these are synthetic capacities, not hardware specifications. Eligibility is exact requested-family equality, without substitutions. Every job has value 1; GPU demand is 1/2/4 and CPU/RAM demands are positive. No individually impossible jobs occurred; none were repaired or removed.

Generator: NumPy PCG64, restarted at seed 11/29/47 for each target .3/.7/1.1; T=40. Sequential draws in order: arrival uniform integer [0,40), equal-probability short/long class, uniform integer duration 1â€“3 or 6â€“10, uniform GPU choice 1/2/4, uniform family, uniform CPU-per-GPU choice 2/4/8, uniform RAM-per-GPU choice 8/16/32. Keep all draws through the first job reaching target GPU work. Full-duration offered load is sum(demand*duration)/(40*total capacity), including work past T. Maximum GPU overshoot is 40/1920=0.020834, below .10 tolerance. Targets sharing a seed share a stream prefix; this is a limitation, not independent resampling. Canonical input SHA-256 uses sorted compact JSON of capacities, node families, full ordered jobs, eligibility, T, seed, target and Omega; `generate` prints those complete manifests.

One unchanged `rolling.simulate` call per method/workload; `accelsched.METHODS` order, solver seed=11, time_limit=.5 s per batch. Immediate start, completion before same-time arrivals, permanent rejection, no queue/retry/migration/preemption. Neutral PGFP future remains unchanged. Gate 1 is seed 11/.3 across six methods with audit after each; Gate 2 runs remaining eight cases. Stop on first error, no retries/seed searches; reserve 30 s before the 480 s deadline. An external 480 s subprocess timeout additionally bounds the experiment.

## Independent measurements

`test_dynamic_bench.audit` reconstructs occupancy from original demands and final decisions. Fixed-window work is demand * max(0,min(end,40)-max(start,0)); divide each resource by 40 times total capacity. A second integration uses successive post-event running-load snapshots, including the empty initial interval and clipping the tail. Full-tail decision work separately equals simulator `resource_time` and full event integration. Each event checks free+running=capacity, release-before-arrival capacity, fixed family eligibility, unique accepted IDs, immutable node/end, exact-once completions, cumulative acceptance, rejection/error lists, and batch placement/status/certificate consistency. Admitted and released resources equal the sum of accepted demands and final resources equal initial capacity. All experiment resources/integrals are exactly represented integers and require exact equality; fractional CPU/RAM comparisons allow absolute 1e-9, relative 0, without changing production tolerances.

Omega is the fixed 12-type catalogue (each family x GPU 1/2/4, CPU=4, RAM=16), unit weights and same-family eligibility. F(t) counts feasible node/type pairs from post-event residuals; integrate its piecewise constant value on [0,40] and divide by 40. This is a candidate-node-count diagnostic, not simultaneous future admission or a forecast.

## Actual execution

Windows 11; Python 3.14.6, NumPy 2.4.6, SciPy 1.18.0; existing warmed environment. PowerShell commands from the checkout, with `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`:

```powershell
D:/software/anaconda3/python.exe -m pytest -q -p no:cacheprovider test_dynamic_bench.py
D:/software/anaconda3/python.exe bench_dynamic.py
```

The benchmark command was invoked once by `subprocess.run` with stdout directed to the D: evidence file and timeout=480. Gate 1 PASS, Gate 2 PASS, **54/54 calls**, all 54 independent audits PASS, all errors zero, all comparable. Experiment-internal wall 5.044239 s; outer process wall 6.055856 s. Five final focused tests PASS in 0.89 s (initial four tests passed in 0.97 s before experiment). Additional corruption/family tests and cumulative-count auditing were then added; all 54 retained reports were re-audited without new simulator calls. The final input-mutation guard hashes actual arrays/jobs as well as the manifest; it does not alter the generated inputs or solve calls. The final runner was also exercised with all 54 retained reports replayed in place of simulate (zero new algorithm executions), checking exact inputs, method/seed/time-limit and the strengthened mutation guard. No existing regression was necessary because production code was untouched.

All heuristic batch statuses were `feasible`; all **313/313 MIP arrival batches** were validated `optimal`. No `feasible_uncertified`, `uncertified`, `solver_error`, failed gate or omitted seed. These are batch-only certificates within solver tolerances, never horizon-global proofs.

Accepted counts below are also accepted values. Order: FF=first_fit, CBF=classic_best_fit, LPO=lp_order_best_fit, LPBF=lp_best_fit, PGFP, MIP=highs_mip.

| Target | Seed | Jobs | Actual GPU load | FF | CBF | LPO | LPBF | PGFP | MIP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3 | 11 | 54 | 0.306771 | 54 | 54 | 54 | 54 | 54 | 54 |
| 0.3 | 29 | 49 | 0.301042 | 46 | 46 | 46 | 46 | 46 | 46 |
| 0.3 | 47 | 37 | 0.300521 | 35 | 35 | 35 | 35 | 35 | 35 |
| 0.7 | 11 | 120 | 0.702604 | 103 | 104 | 104 | 103 | 103 | 102 |
| 0.7 | 29 | 115 | 0.703125 | 97 | 96 | 96 | 97 | 97 | 96 |
| 0.7 | 47 | 99 | 0.702083 | 82 | 82 | 82 | 82 | 82 | 83 |
| 1.1 | 11 | 184 | 1.107813 | 133 | 133 | 133 | 133 | 133 | 137 |
| 1.1 | 29 | 189 | 1.100521 | 136 | 135 | 135 | 136 | 136 | 141 |
| 1.1 | 47 | 163 | 1.101042 | 111 | 111 | 113 | 112 | 112 | 119 |

Single-shot timings are instrumentation, not speedup evidence. Higher offered load sometimes changes admission and placeability; low-load admission ties are retained. MIP accepts fewer cumulatively than some heuristics for seed 11/.7 and 29/.7 despite every batch being optimal. PGFP and LPBF metrics tie here with neutral future. Exact-family imbalance (for example T4/P100 offered load above one while global load is lower) limits compatibility; there is no family-relaxation causal control. Long jobs consume resources across arrivals and beyond T, but no short/long sensitivity was run, so H3 is not isolated causally. No regret, global optimum, historical Alibaba allocation, device VRAM/topology/runtime, production performance or population-generalization claim.

## Window utilization and placeability

Ranges below are min–max across the six methods for each fixed workload, not confidence intervals. U is resource-time utilization on [0,40], in GPU/CPU/RAM order; F is the time-weighted candidate-node count defined above. Admission by individual method is retained in the preceding table. Values are rounded only for presentation.

| Target / seed | GPU U | CPU U | RAM U | Mean F |
| --- | --- | --- | --- | --- |
| 0.3 / 11 | 0.289583 | 0.133646 | 0.163854 | 18.375000 |
| 0.3 / 29 | 0.263021 | 0.096042 | 0.114583 | 17.425000 |
| 0.3 / 47 | 0.254688 | 0.106979 | 0.140313 | 20.150000–20.350000 |
| 0.7 / 11 | 0.509896–0.518750 | 0.243438–0.256458 | 0.254375–0.256458 | 14.200000–14.375000 |
| 0.7 / 29 | 0.515104–0.518229 | 0.231667–0.234375 | 0.211458–0.218542 | 13.425000–13.700000 |
| 0.7 / 47 | 0.516667–0.518750 | 0.230208–0.232708 | 0.263125–0.271875 | 13.475000–13.700000 |
| 1.1 / 11 | 0.646875–0.669271 | 0.320208–0.332500 | 0.318021–0.335104 | 10.675000–11.450000 |
| 1.1 / 29 | 0.673438–0.681771 | 0.313125–0.320521 | 0.286875–0.306042 | 9.225000–9.925000 |
| 1.1 / 47 | 0.640104–0.653646 | 0.283229–0.300104 | 0.292187–0.309896 | 10.150000–10.875000 |

## Input provenance and local evidence index

All nine input SHA-256 values are retained below. Offered CPU/RAM loads use full durations (including work after T), unlike the clipped utilization above; actual offered GPU loads and job counts are in the admission table. All nine individually impossible counts were zero.

| Target / seed | Offered CPU / RAM | Input SHA-256 |
| --- | --- | --- |
| 0.3 / 11 | 0.144479 / 0.171563 | `10ce7d5c4cd3b39ba28acea587886b8cc0c3c9a02015c71e682db8fb7a687e0b` |
| 0.3 / 29 | 0.118958 / 0.135937 | `610fa4819556f18fbcffbed13a5630375296e88bd7d21ed41d0cbcb4eba06681` |
| 0.3 / 47 | 0.120938 / 0.155729 | `ae88e2150a6627c1bf2961fb194bf0810a32861b025034cd14691eabe1cbd600` |
| 0.7 / 11 | 0.347708 / 0.367188 | `636a3a75c632b16b404392aa1cfd3a5b781dc74625e50b93e673cb46f194974a` |
| 0.7 / 29 | 0.314583 / 0.322292 | `f1021d81dbd93a53ff0ed605a10cef006a51bb2eeaef6856a2c0d6a496831fe5` |
| 0.7 / 47 | 0.319063 / 0.355938 | `8ed13976b0fce81afb42d550a5683cb140ddd69a75bf2e0494f3d3a2f0388b0d` |
| 1.1 / 11 | 0.559583 / 0.530208 | `2bdb05b35b2bfe6ca6b7f8b8ff45b123f562acb546f4af4fd38a217d19a43873` |
| 1.1 / 29 | 0.497292 / 0.499271 | `288fb01e74b223c4d34aa17fe7226abafe3232bb62f943556a316977cead156a` |
| 1.1 / 47 | 0.502812 / 0.537813 | `8cf6d4b7fc70a55c740de6e6ce559d2cdcf1834d2b3eeb4f9399e136439b6b44` |

Execution source head: `a950b6ea4c5a78a46c76021da670207d369f954b`. Unchanged Git blobs: `bench_dynamic.py` = `d6547ba4b50f7eb5392eaf72d3b065a28ba8c1dd`; `test_dynamic_bench.py` = `43e2c45457e18e17625082d92b2b8e6c29614533`.

Raw stdout is retained **locally only**, outside the Git checkout, at `D:/Dropbox/study/ms-phd/PHD-EPFL/Doctoral proposal/EPFL/small interesting research/AccelSched/.sae-issue36/dynamic-evidence.json`. It is not published in this PR and cannot be downloaded from GitHub. SHA-256: `75cdad271ba02a067556c0f34eeb7449001b7b0ca822f9115a65c1d461e49dd9`; bytes: 4150123. The digest identifies the retained evidence; it does not promise identical timings on another run.

Local JSON index (case order: target .3/.7/1.1, then seed 11/29/47; method order as above):

- `cases[k].manifest`: complete ordered jobs, capacities, node families, eligibility and Omega.
- `cases[k].workload`: input hash, full-duration resource/family loads, request/duration counts, arrival histogram and impossible-job list.
- `cases[k].results[m].metrics`: accepted/value/rate, normal rejection and solver_error separately, window/tail resource-time, utilization, cumulative acceptance, mean F and batch-status totals.
- `cases[k].results[m].report`: every decision, post-event resource ledger and batch, including per-batch status/optimal flag and solve_wall_ms; the enclosing result includes total wall_ms and report_sha256.
- Top-level `status`, `runs`, `smoke`, `elapsed_seconds`, `environment` and `post_run_audit`: execution and audit metadata.

No numerical rerun was performed for this documentation revision. The worktree and evidence remain intentionally retained for review; no downloads, GPU/CI, new dependencies or paid resources. The 54/54 audit and 313/313 batch-optimal claims remain Engineer-reported execution evidence awaiting independent Architect acceptance.

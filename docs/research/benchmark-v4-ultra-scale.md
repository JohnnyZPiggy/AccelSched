# Benchmark V4 — Ultra-scale physical scheduling: a reproducibility and evidence report

**Project:** AccelSched · **Status:** research benchmark completed; performance-optimization work stopped by Scientist decision · **Date:** 2026-10-26

> **Scope, not a speedup claim.** This report collates previously performed *synthetic* experiments. The 10-million-job result measures an experimental streaming **First-Fit on host CPU**, not PGFP, GPU-executed placement, or the production default. The million-job optimization result measures an *opt-in* indexed, true future-marginal PGFP with certified-price policies; its LP stage used optional GPU solver(s) while ordering/placement/audit ran on the CPU. All reported numerical performance is Engineer-executed, not independently rerun by the Architect from the large locally retained archives.

## 1. Research questions

1. How many heterogeneous resource jobs can be physically placed and independently checked in a bounded, end-to-end execution?
2. Can **true PGFP** (not merely an LP-price ordering followed by First-Fit) and a named LP-guided best-fit policy complete 100K / 1M-job physical placement?
3. What are the documented numerical-certification failures, quality tradeoffs, runtime and memory boundaries?

There is **no claim of a new optimization algorithm, global integer optimality, commercial cluster performance, or a universal GPU/CPU speedup**.

## 2. Code and evidence provenance

| Milestone | Exact merged PR | Benchmark implementation / evidence |
| --- | --- | --- |
| Streaming physical scale | [#109](https://github.com/JohnnyZPiggy/AccelSched/pull/109), [Issue #108](https://github.com/JohnnyZPiggy/AccelSched/issues/108) | `bench_v4_gpu.py`, `test_bench_v4_gpu.py` |
| True PGFP and LP-guided | [#111](https://github.com/JohnnyZPiggy/AccelSched/pull/111), [Issue #110](https://github.com/JohnnyZPiggy/AccelSched/issues/110) | `bench_ultra_pgfp.py`, `test_bench_ultra_pgfp.py` |
| Optional P1 measured watcher-cache probe | [#112](https://github.com/JohnnyZPiggy/AccelSched/pull/112) | `probes/pgfp_p1.py`, `probes/pgfp_p1_fast.py`, `test_pgfp_p1.py` |

P1 is **experimental and opt-in**. Its limited observed improvements are *not* a default scheduler change. The upstream production `accelsched.solve` and existing defaults remain unchanged across these three PRs.

The SHA of the complete #109 Engineer-local archive is `c7e893a5a6d675db8d6891ee7a2c82870a59978f9411a55a59732af0cc88a250` (19,729,338 bytes). The #111 separately recovered archive is `6e180a93a63672aaa2896ddd379acdea6a766357b915a90bd186e2d4ce9f301f` (237,743,274 bytes). The #112 archive is `17739aba0fb7126055689981ab4008c83148fc1d5575f8886014ffb328102c67` (221,899,540 bytes). **These complete archives were retained in the Engineer's designated local D: workspace and are not provided in this repository; their hashes are integrity identifiers, not a public download or Architect reproduction.** Summary tables and test receipts were disclosed in the linked PRs / Issues. For independent archival reproduction, the raw datasets, execution receipts, and runtime pinning must be made available separately.

## 3. Workloads, paths and validation semantics

Four predeclared *synthetic* workload families: `balanced`, `high_pressure`, `fragmentation`, and `multiresource`, with mixed whole-GPU/CPU/RAM demands, hardware-family and group predicates. This is **not** a production trace. The balanced 10M-job case had only about **0.097% GPU-requiring jobs**; high-pressure was about 60%, fragmentation about 25%, and multiresource about 3%. Consequently, *jobs/s is not GPU-jobs/s*. Capacity/inventory scaling, eligibility constraints and per-case workload composition must be kept with each result; do not interpret the size curves as pure strong scaling.

**Track A — experimental streaming First-Fit:** immutable generator with per-job exact assignment, input-order First-Fit tie-breaking, lazy family/group eligibility, monotone capacity, independent regenerated all-job capacity/eligibility/value/residual audit, lossless placements and input/output SHA-256. Its LP status is **N/A**. At <=100K it also uses bounded dense-original oracle and reference-equivalence checks, which add proof-time overhead relative to the >=1M streaming-only audit.

**Track O — actual optimized placement:** certified sparse Shape-LP prices, stable decreasing task utility order and per-task shape utility order, then physical node placement. For *true PGFP*, each feasible node is scored by the **exact weighted loss of future-job suitability**, choosing maximum marginal score and minimum node index on ties; for named `lp_best_fit`, selection is by smallest remaining resource sum, with stable ties. The indexed approach is **conditionally equal to the original policy given identical LP prices**, including candidate score updates after residual changes. Distinct certified dual prices can lead to distinct order/placements; these are explicitly `POLICY_DIFFERENT` from the legacy HiGHS result. The earlier LP-score First-Fit of #109 Track C is neither true PGFP nor proof of PGFP equivalence.

**Certificates and audit:** LP successes require the existing original-model certificate and an independent KKT check, not merely a native OPTIMAL flag. Only those successes enter physical replay. Independently regenerated tasks and all assignments are checked for per-node CPU/GPU/RAM capacities, eligibility and final residual/value consistency; no sample-only or checkpoint-only approval. `PHYSICAL_E2E_VALID` means validated feasibility, **not** a globally optimal placement.

## 4. Observed ultra-scale First-Fit benchmark (#109)

Same measured Colab **NVIDIA L4 host CPU** (physical placement is CPU-executed); the numbers below are full **warm reconstructed** source-to-audited-and-serialized paths reported by the Engineer, not GPU computation speed.

| Workload size | Nodes | Scenario | Valid / declared | Warm complete wall | Offered jobs/s |
| --- | ---: | --- | ---: | ---: | ---: |
| 100K | 1,024 | balanced | 6 / 6 | 1.245 s | ~80,302 |
| 1M | 4,096 | balanced | 2 / 2 | 4.581 s | ~218,308 |
| 10M | 8,192 | balanced | 2 / 2 | 44.846 s | ~222,983 |
| 10M | 8,192 | high_pressure | 2 / 2 | 39.038 s | ~256,161 |
| 10M | 8,192 | fragmentation | 2 / 2 | 42.756 s | ~233,884 |
| 10M | 8,192 | multiresource | 2 / 2 | 44.502 s | ~224,708 |

Predeclared Track A totals: S0 10K **8/8**, S1 100K **24/24**, S2 1M **8/8**, S3 10M **8/8** physically audited. The ten-million-job outcome is **an experimental First-Fit scaling benchmark only**. Higher load does not mean the same accepted job mix.

## 5. Actual 100K / 1M LP-guided and true PGFP (#111)

The primary solver for this batch was **cuPDLPx** on the observed L4; physical node placement, scoring, and audit were on the host CPU.

| Size | Policy | Physical successes / declared | Interpretation |
| --- | --- | ---: | --- |
| 10K / 256 nodes | true PGFP | 8 / 8 | Certified-price full physical placement |
| 10K / 256 nodes | `lp_best_fit` | 8 / 8 | Certified-price full physical placement |
| 100K / 1,024 nodes | true PGFP | 22 / 24 | Two failed/unavailable numerical results |
| 100K / 1,024 nodes | `lp_best_fit` | 22 / 24 | Two failed/unavailable numerical results |
| 1M / 4,096 nodes | true PGFP | **6 / 8** | Two failed/unavailable numerical results |
| 1M / 4,096 nodes | `lp_best_fit` | **6 / 8** | Two failed/unavailable numerical results |

The original #111 recovered run contains 128 declared records: **116 physically valid, 12 UNCERTIFIED** (including backend time limits and numerical residual rejection); among them are all the First-Fit controls and same-host diagnostic probes. These counts are not 128 independent optimized PGFP results. The first GPU run's large ZIP timed out on download and is separately labeled remote-reported-only; a predeclared replay produced the locally audited recovery archive. Engineer independently reopened 27.44 million job decisions and 76 final LP KKT certificates across that recovered run. Architect has reviewed the committed benchmark/audit code but **has not independently retrieved and rerun the complete raw archive**.

Selected same-host *warm complete E2E* measurements from #111: at 1M fragmentation, **First-Fit 4.387 s, true PGFP 38.764 s, LP best-fit 40.957 s**; 1M balanced true PGFP **45.677 s** versus streaming First-Fit **4.356 s**. Times incorporate LP solve/certification, ranking, physical placement, independent audit and serialization for optimized policies. First-Fit includes additional legacy dense/public reference proof cost in smaller <=100K cases, so cross-policy small-scale speed ratios require special caution. Different policy quality, degeneracy and success denominators prevent universal dominance or GPU-acceleration claims.

As an illustrative 100K high-pressure quality observation, Engineer-reported accepted-value medians were **First-Fit 221,186** versus **PGFP 369,343**; these are descriptive results on selected synthetic seeds with different valid denominators, not statistical or universal guarantees. More accepted value can come with fewer accepted GPU jobs; inspect workload class and full quality tables before concluding.

**Failures must remain visible:** native `TIME_LIMIT`, `NUMERIC_UNCERTIFIED`, and original KKT residual failures are not counted as physical placements. Separate same-host HiGHS / H-PDLP diagnostics exhibit additional failures. Original numerical thresholds were not relaxed to manufacture success. Peak GPU VRAM was **NOT_AVAILABLE**, not zero. CPU RSS is process high-water, not isolated device allocation.

## 6. Optional P1 profiling appendix (#112)

P1 measured the existing 100K fragmentation index `commit()` self time at **1.0291 s / 77,772 calls** under cProfile, then evaluated an **explicitly selected live-watcher-list dispatch** while keeping the original score updates and stable ties. This is an opt-in research probe; no default production integration.

Engineer-reported single paired warm observations:

| Workload | True PGFP before | Optional candidate after | Observed difference |
| --- | ---: | ---: | --- |
| 100K fragmentation | 3.6965 s | 3.4900 s | ~5.6% lower wall |
| 1M fragmentation | 38.4065 s | 36.4342 s | ~5.1% lower wall |
| 1M high_pressure | 39.1703 s | 39.0212 s | ~0.4% lower wall |

57 targeted tests were reported passing; the Engineer reported 44/48 valid matched primary before/after pairs in *each* phase, plus 30 exact same-price full-order/placement/residual shadows. These experiments do **not** establish a population p95, statistical significance or a universal speedup. Tracemalloc indicated a possible retained-memory tradeoff, and some local task cache deletion was blocked by policy; neither was concealed or bypassed. Both Colab sessions for #111 and the one for #112 were reported stopped, with authenticated zero-active-session receipts.

## 7. Reproduction guide

Start with a checkout of the exact source commit rather than mixing benchmark code and result data across milestones. For final current-source reproduction, consult PR #112's merge and use the committed opt-in benchmark files:

```bash
python -m pytest -q test_bench_v4_gpu.py test_bench_ultra_pgfp.py test_pgfp_p1.py
python bench_v4_gpu.py --out /tmp/accelsched-v4-firstfit --budget 1500
python bench_ultra_pgfp.py --out /tmp/accelsched-v4-optimized --budget 2100
```

**Warning:** these latter two commands can trigger large bounded CPU/memory and optional GPU solver work, write substantial raw archives, and require their separately documented environment/dependencies. The second command's default GPU backend `cupdlpx` must be installed to reproduce optimized GPU runs. `--out` must point to a **new** nonexistent output directory. Use isolated scratch, existing authorized hardware only, and stop GPU sessions in a guaranteed cleanup path. The above commands are **instructions, not tests performed in this documentation-only pass**.

Recommended independent audit: preserve exact `main`/branch tree, input configuration/seed/chunk and source SHA, sparse model's CSR digest, solver vectors/prices and original KKT tolerances, complete assignment/remainder arrays, exact files' SHA-256, failure/retry and run clocks, CPU/GPU model/drivers, OS threads, RSS/VRAM availability, and explicit session stop receipts. Check *all* jobs for physical capacity and eligibility—not just a hash. Never count an LP-only success as a complete physical placement.

## 8. Interpretation, limitations and stopping decision

- **Established in the reported synthetic experiments:** physically audited streaming First-Fit to 10M; true indexed PGFP and named LP best-fit complete physical placement on some 100K/1M attempts; numerical failure states preserved; feasible output is independently recomputed in the Engineer's retained run.
- **Not established:** production `accelsched.solve` throughput at millions of tasks; PGFP at 10M; fully GPU-resident physical placement; comparative GPU kernel speedup; real AI training/cluster telemetry; statistical performance confidence intervals; universal improvement of PGFP over First-Fit; independent Architect replay of Engineer-local multi-hundred-MB ZIPs.
- **Decision:** Scientist ended additional P2 micro-optimization. Benchmark V4 closes as a documentation and reproducibility milestone, without new performance experiments or algorithm changes.

The outcome is primarily an **engineering integration and reproducible-evidence study**, not a novel scheduling algorithm claim.

---
MIT-licensed AccelSched · Copyright © 2026 Ru Zhang.

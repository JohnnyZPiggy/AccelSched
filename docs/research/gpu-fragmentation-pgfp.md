# Scientific Mentor Research Note: GPU Fragmentation and Batch Placement

**Project:** AccelSched · **Research track:** Price-Guided Fragmentation-Aware Placement (PGFP)  
**Status:** Mathematical baseline **RESEARCH_READY**; performance, general approximation guarantees, and GPU implementation **UNVERIFIED**  
**Date:** 2026-10-23 · **SAE reference:** V3.3, commit `cc96b803466a5d2848d8eeedd9e6fac37a3016ba`  
**Scope:** mathematical formulation, proofs/counterexamples, algorithmic hypotheses, numerical oracle, and experimental plan. No implementation, device benchmarks, or empirical speedup claims.

## 1. Research question and boundaries

How should a heterogeneous GPU cluster accept a batch of indivisible, single-node tasks while retaining useful contiguous capacity for future tasks? We distinguish:

- **Batch placement:** existing workloads and their allocations are fixed; only new tasks are placed. The initial per-server *residual* capacities are inputs.
- **Repacking/defragmentation:** existing workloads may be migrated; source/destination feasibility, migration budget and disruption costs must be added to the model. This is **not** included in the initial PGFP formulation.
- **Gang / topology-constrained placement:** multi-node jobs, interconnect bandwidth, GPU islands and detailed GPU-device affinity need separate constraints. Counts alone do not model those restrictions.

CvxCluster motivates a two-stage method: relaxed optimization and capacity shadow prices, followed by feasible physical-server greedy placement. Work on fragmentation-aware GPU scheduling (e.g., FGD) motivates measuring *future placeability*. Combining these ideas is a design hypothesis, **not** a claim of algorithmic novelty or a result established by either paper.

## 2. Notation, contracts and assumptions

| Symbol | Shape | Meaning |
|---|---|---|
| \(N\) | \(s\) nodes | Physical servers |
| \(J\) | \(n\) tasks | New indivisible tasks |
| \(K\) | \(r\) types | Resources (GPU count, CPU, memory, optional virtual resources) |
| \(C\) | \(s\times r\) | Current residual capacity, nonnegative |
| \(D\) | \(n\times r\) | Per-task requirements, nonnegative |
| \(p\) | \(n\) | Nonnegative task values/priorities |
| \(E\) | \(s\times n\) | Physical task/server compatibility bits |
| \(x\) | \(s\times n\) | Binary task-to-server assignment |
| \(h(i)\) | -- | Shape of physical node \(i\) |
| \(\Omega\) | finite set | Prospective future task types |
| \(d^\omega,w_\omega\) | \(r\), scalar | Future demand and nonnegative weight |
| \(e_{i\omega}\) | binary | Physical future-type compatibility |
| \(z_{i\omega}\) | binary | Auxiliary future-placeability decision |

For the initial model GPU requirements and capacities are integer counts; CPU/RAM can be nonnegative rational/real resource units. Each accepted task occupies **one** physical node; no partial GPU or split task is deployable. Anti-affinity, where needed, imposes \(\sum_{j\in G}x_{ij}\le \ell_{iG}\). Compatibility \(E_{ij}=0\Rightarrow x_{ij}=0\). Invalid/nonfinite data must be rejected. Shape grouping may collapse hardware-equivalent servers only to the extent the relaxation maintains a valid *outer bound*; physical eligibility must still be checked at placement time.

## 3. Base batch-placement MILP

Let \(y_j=\sum_i x_{ij}\) indicate accepted tasks. Maximize weighted placed value:

\[
V(x)=\sum_{j\in J}p_jy_j.
\]

Subject to:

\[
\begin{aligned}
&\sum_i x_{ij}\le1 &&(\forall j),\\
&\sum_j D_{jk}x_{ij}\le C_{ik} &&(\forall i,k),\\
&0\le x_{ij}\le E_{ij},\quad x_{ij}\in\{0,1\},\\
&\sum_{j\in G}x_{ij}\le\ell_{iG} &&(\text{when anti-affinity is specified}).
\end{aligned}
\]

This is a multidimensional generalized assignment/packing-style 0–1 model. **Objective \(V\) alone is insensitive to resource shape left behind among equal-value feasible schedules.**

## 4. Future-placement capability as a fragmentation proxy

For residual server resource vector \(R_i(x)=C_i-\sum_jD_jx_{ij}\), define

\[
f_{i\omega}(x)=e_{i\omega}{\bf1}\{R_i(x)\succeq d^\omega\},\qquad
F(x)=\sum_i\sum_{\omega\in\Omega}w_\omega f_{i\omega}(x).
\]

\(F\) is the **weighted number of eligible candidate nodes** for separately considered future task types. It is **not** the maximum number of distinct future jobs that can be concurrently packed, nor a calibrated forecast unless \(\Omega,w\) derive from validated workloads.

Single-resource special case with integer free GPU counts \(R_i\), thresholds \(Q\), and weights \(w_q\):

\[
F(R)=\sum_i\sum_{q\in Q}w_q{\bf1}\{R_i\ge q\},\quad
F_q(R)=\sum_i{\bf1}\{R_i\ge q\}.
\]

**Exact local marginal loss.** Placing a feasible \(g_j\)-GPU task on node \(i\) changes only that node, giving

\[
\Delta F_{ij}
=\sum_{q\in Q}w_q[{\bf1}\{R_i-g_j\ge q\}-{\bf1}\{R_i\ge q\}]
=-\sum_{R_i-g_j<q\le R_i}w_q\le0.
\]

With threshold prefix weights \(W(z)=\sum_{q\le z}w_q\),

\[
\Delta F_{ij}=W(R_i-g_j)-W(R_i).
\]

Computational result: naively \(O(|Q|)\) per **candidate node**, \(O(\log|Q|)\) using sorted thresholds/prefix sums, or \(O(1)\) with a bounded integer GPU-capacity lookup table. This does **not** imply constant-time selection across servers or constant-time overall scheduling.

For multidimensional resources,

\[
\Delta F_{ij}=\sum_{\omega}w_\omega e_{i\omega}
\left[{\bf1}\{R_i-D_j\succeq d^\omega\}-{\bf1}\{R_i\succeq d^\omega\}\right],
\]

computable locally in \(O(|\Omega|r)\) by straightforward comparisons, with additional possible indexing/precomputation. A task/node compatibility check remains mandatory before scoring.

### Structural properties, proofs and counterexamples

**Monotonicity (proved).** Under batch-only placement, nonnegative consumption, no release or migration, adding a physically feasible assignment cannot increase any coordinate of \(R_i\). With \(w_\omega\ge0\), every threshold indicator is nonincreasing, hence \(F\) is monotone **nonincreasing** as assignments accumulate. This statement is about nested *fixed physical assignments*, not arbitrary sets with implicit reoptimization or a time-evolving system.

**Not submodular in general (counterexample).** One 8-GPU server, future threshold \(q=8\), two distinct existing candidate tasks \(a,b\) each requiring 4 GPU. Values: \(F(\varnothing)=1\), \(F(\{a\})=F(\{b\})=F(\{a,b\})=0\). For \(A=\varnothing,B=\{a\},e=b\), the marginal differences are \(-1\) and \(0\), violating submodularity's \(-1\ge0\).

**Not supermodular in general (counterexample).** On the same machine, set future threshold \(q=4\). Then \(F(\varnothing)=F(\{a\})=F(\{b\})=1\), \(F(\{a,b\})=0\). Marginals are \(0\) and \(-1\), violating supermodularity's \(0\le-1\). These are two different single-threshold instances; neither universal structure holds.

Consequently standard greedy approximation theorems for monotone submodular maximization cannot be invoked for this \(F\) without *additional assumptions and proof*.

## 5. Lexicographic fragmentation-aware integer optimization

Set

\[
\operatorname{lexmax}_{x}\;(V(x),F(x)).
\]

A linearized second-stage formulation introduces binary \(z_{i\omega}\) and, for all \(i,k,\omega\),

\[
\sum_jD_{jk}x_{ij}+d_k^\omega z_{i\omega}\le C_{ik},\qquad
0\le z_{i\omega}\le e_{i\omega}.
\]

No Big-M is necessary: \(z=1\) certifies that server \(i\) still accommodates future type \(\omega\); \(z=0\) restores regular capacity limits. All types are evaluated **individually**, not simultaneously reserved. For positive weights, maximizing \(\sum w_\omega z_{i\omega}\) sets all feasible indicators to 1; zero-weight types may be left at 0, so the independent validator should recompute \(F\) directly.

**Exact sequential lexicographic solve:**

1. MILP-1: maximize \(V(x)\); obtain *proven* integer optimum \(V^*\).
2. MILP-2: maximize \(\sum w_\omega z_{i\omega}\) with \(V(x)=V^*\) and the original constraints.

Avoid numerically brittle arbitrarily large scalar weights. If MILP-1 stops without proof, fixing its incumbent only optimizes \(F\) *conditional on that incumbent's value*, not a globally lexicographic optimum.

## 6. LP relaxations, valid bounds, and dual prices

Relax binary \(x,z\) to \(0\le x,z\le1\). The LP feasible set contains the MILP feasible set. For **the same scalar linear objective**, maximization over the LP yields a valid upper bound. In particular

\[
V_{\mathrm{integer}}^*\le V_{\mathrm{LP}}^*.
\]

For fixed known integer-optimal value \(V^*\), the LP second-stage optimum with \(V(x)=V^*\) upper-bounds the true lexicographic second-stage optimum. **Do not fix \(V=V_{\mathrm{LP}}^*\)** and pretend this represents the true integer lexicographic problem: the LP value may be unattainable integrally.

**Fractional future-placeability counterexample.** One node has residual 6 GPUs, future demand is 8. Integer \(z\) from \(8z\le6\) must be 0; relaxed \(z\) may be 0.75. This is a valid LP surrogate, **not** 75% of a physically deployable 8-GPU job.

### Shape/global relaxation

Let \(m\) be number of shapes, \(B_{hk}=\sum_{i:h(i)=h}C_{ik}\), and \(X\in\mathbb R_+^{m\times n}\) represent fractional allocations to shape pools. The base relaxation is

\[
\begin{aligned}
\max_{X\ge0}\quad&\sum_{h,j}p_jX_{hj}\\
\text{s.t.}\quad&\sum_hX_{hj}\le1\quad(\forall j),\\
&\sum_jD_{jk}X_{hj}\le B_{hk}\quad(\forall h,k).
\end{aligned}
\]

This model ignores physical bin packing; it is an outer relaxation of the physical integer model when any server allocation maps to its shape and all other physical restrictions have been dropped or only valid necessary conditions retained. Extra shape eligibility restrictions are valid **only if they do not exclude any physically feasible integer placement**. Coarse global pricing collapses all capacity into one pool, producing a weaker-or-equal bound under those conditions.

The dual of the base shape LP, for \(\lambda_{hk}\ge0\) (capacity) and \(\mu_j\ge0\) (at-most-one assignment), is

\[
\begin{aligned}
\min_{\lambda,\mu\ge0}\quad&
\sum_{h,k}B_{hk}\lambda_{hk}+\sum_j\mu_j\\
\text{s.t.}\quad&
\mu_j+\sum_kD_{jk}\lambda_{hk}\ge p_j\quad(\forall h,j).
\end{aligned}
\]

Define task/shape **net utility**

\[
U_{hj}=p_j-\lambda_h^\top D_j,\qquad U\in\mathbb R^{m\times n}.
\]

This is the CvxCluster-style price-guided ranking signal. Capacity shadow prices are dual marginal values under this chosen LP and solver sign convention; they are not hardware sticker prices. Dual solutions may be nonunique. Do not demand elementwise identity between different optimal solvers; check primal feasibility, dual feasibility, complementary slackness and objective gap. If dual prices are missing or untrustworthy, report that and distinguish any fallback heuristic from verified price guidance.

**Important modeling boundary:** the new \(F,z\) formulation introduces additional constraints/duals; simply applying the original \(U=p-\Lambda D^\top\) to the full lexicographic LP is *not* a mathematically derived marginal for its second objective. Hence M0 keeps the **base LP prices** and modifies only physical-server selection.

## 7. PGFP algorithm hypothesis

Compare three procedures that use the **same** base LP instance, prices, task ordering and shape ordering:

- **Random Feasible:** within a feasible shape, uniformly sample an eligible physical server using a controlled seed (CvxCluster-style baseline).
- **Best-Fit:** within a feasible shape, choose a reproducible capacity-packing criterion (define precisely per resource).
- **PGFP / Minimum Fragmentation Loss:** within a feasible shape, choose a node maximizing \(\Delta F_{ij}\) (equivalently minimizing \(-\Delta F_{ij}\)).

Common skeleton:

1. Solve base global/shape LP, verify solver status and admissible dual prices.
2. Compute \(U_{hj}\) for allowed shape/task pairs; order tasks by \(\max_h U_{hj}\) descending with deterministic ties.
3. For each task, traverse candidate shapes by \(U_{hj}\) descending; find **physically feasible** nodes in the shape (GPU, CPU, RAM, model/label, anti-affinity and other hard restrictions).
4. Apply one of the server-selection strategies; update actual residuals. If a shape has no feasible node, try next shape. If none fit, mark task unplaced.
5. Recompute all hard constraints from the original inputs and final placement using an **independent validator**. Record \(V,F\), source/dual evidence, runtime and algorithm identity.

For equal shape \(h\) and task \(j\), \(U_{hj}\) is constant across nodes. Hence adding \(\eta\Delta F_{ij}\) to the within-shape score is equivalent to ranking solely by \(\Delta F_{ij}\) for \(\eta>0\). Do not describe this as a globally lexicographic optimal algorithm. Ties may determine performance; specify deterministic or seeded tie handling.

### Limits and adversarial failure

**Local-best is not global-best.** Greedy can choose a low immediate loss and thereby block valuable later jobs; it is a heuristic, with no established general approximation factor.

**Constructed no-uniform-ratio example for a coarse-shape variant:** Two nodes A and B each have \(n\) GPUs and are grouped in one shape, but \(n\) unit-GPU tasks are compatible *only* with A. One \(n\)-GPU flexible task of value \(1+\epsilon\) fits either node; each unit task has value 1. Suppose coarse shape pricing does not represent physical eligibility and the LP returns an optimal zero capacity-price vector (possible because all tasks fit aggregate capacity), so the flexible task ranks first. With \(F\) based on a prospective \(n\)-GPU task, placement on either initially empty node has equal immediate fragmentation loss. If the permitted tie break selects A, all unit tasks become infeasible, yielding \(1+\epsilon\); optimal discrete placement puts the flexible task on B and unit tasks on A, yielding \(n+1+\epsilon\). Ratio \((1+\epsilon)/(n+1+\epsilon)\to0\). This is a counterexample **to that specified coarse model and tie rule**, not a universal theorem for all compatibility-aware price/placement variants.

Fragmentation-only objectives can also trade off against present job acceptance: compare lexicographic objectives explicitly and report both metrics rather than hiding their trade-off in an arbitrary score.

## 8. Mandatory hand-worked oracle cases

1. **Equal value, different fragmentation:** 2 nodes with 8 GPUs each; two current 4-GPU jobs worth 10 each. Packing both on one node gives \(V=20,F_8=1\); spreading yields \(V=20,F_8=0\). Exact lexicographic optimum packs.
2. **LP integrality gap:** 2 nodes with 4 GPUs each; three 3-GPU jobs worth 5 each. Aggregate LP \(V=40/3\), physical integer optimum \(V=10\), deploy at most 2 jobs. The LP upper bound is not deployable.
3. **Repacking vs batch-only:** two nodes each capacity 8 with a fixed existing 4-GPU job on each. An incoming 8-GPU job cannot be placed without migration. If moving one existing job to the other node is allowed and feasible, the other becomes empty and fits the new job. Migration cost/budget is *not* included in batch-only M0.
4. **Fractional future-placeability:** capacity 6, future type demands 8: binary \(z=0\), LP may yield 0.75.
5. **Submodularity and supermodularity:** two explicit single-node 8-capacity threshold cases in §4.
6. **CPU or memory bottleneck:** plenty of free GPUs but CPU or RAM insufficient: validator must reject.
7. **GPU type / shape compatibility:** never place incompatible tasks, even if aggregate free GPU count suffices.
8. **Anti-affinity and fallback:** same-group replicas cannot co-locate where group limit is 1; next compatible shape/node may still work.
9. **Dual-price degeneracy:** compare KKT residuals/objective, not exact dual vector equality.
10. **Random/tie sensitivity:** fixed seeds and multiple trials when random server selection is used.

## 9. Benchmark and falsification plan (NOT RUN)

**Methods:** independent First-Fit, precisely defined Best-Fit/GPU Packing, small-instance HiGHS MILP (with time limit + certified incumbent/bound), global/shape LP + random greedy, global/shape LP + Best-Fit, PGFP, and later compatible GPU LP adapters. FGD baseline only after verifying its exact task semantics and public implementation/license.

**Controlled factors:** nodes \(2\)–\(10\), \(10\)–\(100\), \(100\)–\(1{,}000\), \(1{,}000+\); pending jobs \(2\)–\(20\), \(20\)–\(500\), \(500\)–\(10{,}000\), \(10{,}000+\); shape count, GPU saturation, workload demand skew, heterogeneity, compatibility patterns, CPU/RAM ratios, future \(\Omega,w\), and anti-affinity. Compare identical instances and initial residual states; distinguish synthetic workloads from trace-derived fields.

**Measures:** current placed value \(V\), number accepted, physically recomputed \(F\), an empirical future-batch acceptance metric (separate from \(F\)), feasibility violations (must be zero), LP/MIP bounds, integrality gap when known, heuristic loss when integer optimum certified, solver and pricing time, greedy and validation time, total end-to-end wall time, peak memory, plus P50/P95/P99 across cases. For unproven MIP optimality, report incumbent and bound, **not** exact optimum.

**CPU oracle:** SciPy/HiGHS float64 LP + independently verified primal/dual feasibility, gap/KKT; small brute force or HiGHS MIP oracle. Suggested **starting** normalized residual/gap target \(10^{-7}\) for float64 (tune by scaling/conditioning), and PyTorch float32 comparisons on normalized inputs with tentative atol/rtol \(10^{-4}\), subject to empirical validation. Distinguish solver-optimal from merely feasible status.

**GPU phase (future, NOT RUN):** test GPU-supported LP software only after reviewing license/version, API, dual sign, solver status and actual hardware capability. Synchronize correctly; account for host-device transfer, presolve, iterative solve, convergence, postsolve, compile/warmup, placement and validation. Device use must be confirmed at runtime, session usage bounded and released. A faster LP **does not** imply a faster full scheduler (Amdahl's law).

**Trace candidates for separate verification:** Alibaba clusterdata GPU traces (2023/2026) and related published simulators, with provenance/license/fields checked before reuse. Never call synthetic/derived GPU demands fully observed real telemetry.

## 10. Mathematical and scientific decision log

| Item | Status | Basis |
|---|---|---|
| Base batch-placement MILP | PROVED model | Explicit binary feasibility constraints |
| Local \(\Delta F\) identity | PROVED | One-node difference/cancellation |
| Monotone nonincreasing \(F\) without migration | PROVED | Nonnegative resource consumption |
| General submodularity of \(F\) | REFUTED | 8-GPU / 8-threshold counterexample |
| General supermodularity of \(F\) | REFUTED | 8-GPU / 4-threshold counterexample |
| Fixed-objective LP upper bound | PROVED | Inclusion of feasible sets |
| Fractional \(z\) = physical future capacity | FALSE | 6-vs-8 counterexample |
| Lexicographic sequential MILP when \(V^*\) certified | PROVED | Constrained second optimization |
| Price-guided PGFP improves realized placement | HYPOTHESIS | Requires fair controlled experiments |
| Price-guided PGFP has general approximation guarantee | NOT ESTABLISHED | Adversarial behavior possible |
| Shape aggregation preserves all bin-level feasibility | FALSE | 2×4 GPUs / 3×3 jobs |
| GPU solver or GPU scoring accelerates end-to-end | UNVERIFIED | No device execution/benchmark |

**Mentor decision:** **RESEARCH_READY for a bounded CPU reference/oracle experiment**, with GPU acceleration, topology extension, task migration, fragmentation-aware LP dual modification and superiority claims retained as open research/engineering hypotheses. No code tests, PR merges, GPU sessions, or benchmark runs are implied by this note.

## 11. Source / reuse map and provenance limits

- CvxCluster: O. Nnorom Jr., S. Boyd, P. Levis, *CvxCluster: Solving Large, Complex, Granular Resource Allocation Problems 100–1000x Faster* (2026), https://arxiv.org/abs/2605.01614 ; owner-supplied PDF `literature/CvxCluster.pdf`. Mathematical inspiration: aggregate LP, dual shadow prices, price-guided feasible greedy placement. The fragmentation objective, its exact local marginal formula, and counterexamples here are **our analysis**, not claims from the paper.
- FGD: *Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent*, USENIX ATC 2023, https://www.usenix.org/conference/atc23/presentation/weng . Motivation for workload-demand-aware fragmentation. Verify actual baseline implementation, semantics and license before reuse.
- Open-source reference MILP/LP: HiGHS, https://highs.dev/ . Package version, exact dependency pin, and empirical compatibility are **TO VERIFY** during implementation.
- GPU LP solvers: investigate the source PDFs already in `literature/` (cuPDLP, H-PDLP, PDNQP), actual implementations, dual support and licensing before suggesting production integration. **No GPU acceleration is claimed here.**

## 12. Next bounded SAE handoff

Engineer should begin with **read-only source/package confirmation**, then implement *only* the tiny CPU oracle and three placement variants in a subsequent authorized development cycle. Keep the study as the single mathematical source of truth, avoid redundant reports/Issues, use exact GitHub SHA and one PR per focused implementation batch; no default GitHub Actions. Architect independently reviews mathematical correctness, actual tests, residuals and clean-install evidence before merging. Any significant change of the mathematical objective updates this study with an explicit proof obligation rather than silently altering the claimed bounds.


## 13. Local exchange criterion (M2 extension, NOT YET BENCHMARKED)

**Purpose.** Improve a *completed* batch-only placement by rearranging **newly admitted** jobs while keeping all pre-existing jobs fixed. Do not conflate this local search with migration/repacking of running jobs. Preserve Random/Best-Fit/PGFP greedy baselines without exchange as controls.

Let \(\pi:J\to N\cup\{-1\}\) denote a complete placement, with \(-1\) for rejected jobs. Define verified objective values \(V(\pi)\) and \(F(\pi)\) on actual physical residual capacities. Let \(\mathcal N(\pi)\) include the following legal moves, all checked against **all** physical resource capacities, GPU models, eligibility, and any anti-affinity constraints:

1. **Relocate (1-opt):** move one *accepted, newly placed* job \(j\) from node \(i\) to a different node \(k\). No change to accepted job set, so \(V\) is invariant.
2. **Exchange (2-opt):** swap destinations of two *accepted, newly placed* jobs \(j,\ell\) assigned to different nodes. Again \(V\) is invariant.
3. **Optional admission exchange:** eject one or more newly admitted jobs and accept one or more previously rejected jobs. This changes \(V\) and must be treated under strict lexicographic dominance. Leave this outside the first M2 exchange implementation to preserve interpretation and bounded runtime.

**Strict lexicographic acceptance:** for any candidate feasible \(\pi'\), accept iff
\[
(V(\pi'),F(\pi')) >_{\rm lex}(V(\pi),F(\pi)).
\]
For fixed accepted-set relocation/pair exchange, this simplifies to
\[
\boxed{\Delta V=0,\qquad \Delta F=F(\pi')-F(\pi)>0.}
\]
The strict inequality prevents zero-gain cycling. Use deterministic move enumeration and choose the first strict improvement (or explicitly specify best improvement); stop at local optimum or a fixed scan/move budget. With integer GPU counts and nonnegative integer future weights, strictly increasing \(F\) bounds the number of accepted improvements; with arbitrary real weights, finiteness still follows from finite placements plus no revisiting under strict increase, but there is **no practical polynomial iteration guarantee**.

**Local delta derivation.** For a relocation of new job \(j\) with resource vector \(D_j\) from source \(i\) to destination \(k\), let current residual vectors be \(R_i,R_k\). Provided destination fits and all other constraints hold,
\[
\Delta F =
  [f_i(R_i+D_j)-f_i(R_i)]
 +[f_k(R_k-D_j)-f_k(R_k)],
\quad f_i(r)=\sum_\omega w_\omega e_{i\omega}{\bf1}\{r\succeq d^\omega\}.
\]
Only two nodes need rescoring. For swapping \(j:i\to k\) and \(\ell:k\to i\),
\[
R_i'=R_i+D_j-D_\ell,\quad R_k'=R_k+D_\ell-D_j,
\]
and \(\Delta F=[f_i(R_i')-f_i(R_i)]+[f_k(R_k')-f_k(R_k)]\). Verify both new residual vectors are componentwise nonnegative, plus all other hard constraints.

**Tiny exact example.** Two 8-GPU nodes; two newly admitted 4-GPU jobs are initially split (residuals \(4,4\)); \(F_8=0\). Relocating either newly placed job onto the other node yields residuals \(8,0\), hence \(F_8=1\) with unchanged admitted value. This is a certified strict lexicographic improvement. If those were *pre-existing* running jobs, the move is **forbidden** in batch-only M2.

**Ablation contract.** Hold instance, price vectors, greedy task/shape order, future batch and random seed fixed. Compare Random, Best-Fit and PGFP **each with and without** identical local-exchange postprocessing; report pre/post \(V,F\), accepted relocation/swaps, feasibility, exchange computation time and future-batch acceptance. Do not claim an improvement in \(F\) necessarily raises actual future-batch acceptance. Local optimality is *neighborhood-relative*, not global MILP optimality. A bounded 1+2-opt neighborhood can still get stuck in a non-global optimum.

**Validation gates:** independent recomputation of residual capacities and compatibility, exact \(\Delta V=0\) on fixed-set moves, strict positive \(\Delta F\) on each accepted move, finite termination under bound, reproducible fixed-seed order, and no mutation of pre-existing allocation. Tests include split 4+4 relocation, swap with nonidentical GPU/CPU requirements, incompatible destination, zero-gain move rejection and CPU-capacity violation.

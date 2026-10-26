# Phase 2 architecture freeze — future Python package migration (DESIGN ONLY)

**SAE V3.3 — Chief Architect + Scientific Mentor** · 2026-10-26  
**Frozen inspected main:** `618837eada32d865722135d7f0645c86ae79182b` · **release remains v1.0.0**  
**Status:** architecture study / compatibility contract, **NOT** an approved code-migration plan or a claim of executed tests.

> The Scientist authorizes **documentation only**. Do not move any production module, modify defaults/API, introduce GPU requirements, allocate GPU compute, publish a release, or create new spend. The Phase 1 source layout and Benchmark V4 results remain unchanged. A later implementation needs a separate explicit approval and an independent SAE T2.

## 1. Current installed layout and dependency graph

`pyproject.toml` uses `setuptools>=77`, `packages=["examples"]` and exactly **ten root `py-modules`**. The current declared runtime dependencies are `numpy>=1.23`, `scipy>=1.9`; Python is `>=3.10`. The wheel includes the ten modules, the `examples` package and two JSON assets (`examples/data/static.json`, `dynamic.json`). It does **not** install `tests/`, `benchmarks/`, `probes/`, `tools/` or source-only `queueing.py`. The Phase 1 PR #115 Engineer evidence reports a 22-entry wheel and matching baseline installed assets on a bounded Windows environment; this architecture-only pass did **not** independently build a wheel or run installation.

The direct **in-repository top-level imports** below are read from the actual 10 files at the frozen main; dynamic imports/callbacks and consumers in noninstalled research files must additionally be traced when implementing a migration.

| Installed module | Direct other production-module imports | Responsibilities / externally importable named entrypoints |
| --- | --- | --- |
| `accelsched.py` | `scipy_backend.solve_mip`; `m0_oracle.Instance,greedy,shape_lp,placed_value,residual,validate` | `solve`, `SolveResult`, `METHODS`, `first_fit`, `pack`, `lp_order`; also re-exports imported names |
| `m0_oracle.py` | — | `Instance`, `validate`, `residual`, `future_score`, `placed_value`, `brute_lex`, `shape_lp`, `mip_value`, `marginal`, `greedy`, `packing_case`, `integrality_case` |
| `scipy_backend.py` | `m0_oracle.shape_lp,validate` | `solve_lp`, `solve_mip` |
| `workload.py` | `accelsched.METHODS,Instance,_input_error,solve`; `m0_oracle.validate` | `build_workload`, `heterogeneous_example`, `evaluate` |
| `rolling.py` | `accelsched.METHODS,solve`; `m0_oracle.validate`; `workload.build_workload` | `Job`, `simulate` |
| `lp_contract.py` | — | `CanonicalLP` |
| `lp_certify.py` | — | `certify` |
| `lp_backends.py` | `lp_contract.CanonicalLP`; `lp_certify.certify` | `RawResult`, `Attempt`, `CertifiedResult`, `solve_lp`; optional backends via deferred `importlib` |
| `certified_pgfp.py` | `m0_oracle` as `oracle`; `lp_contract.CanonicalLP`; `lp_backends.solve_lp` | `canonical_shape`, `shape_lp`, `shadow` |
| `solver_portfolio.py` | `lp_contract.CanonicalLP`; `lp_backends.CertifiedResult,solve_lp` | `PortfolioResult`, `device_hint`, `solve_portfolio` |

The dependency flow is **not all package-independent**:

```text
m0_oracle ──────> scipy_backend ───> accelsched ───> workload ───> rolling
     ╲                                  │                 ╲        ↗
      ╲─────────────────────────────────┘                  └───────┘

lp_contract ──> lp_backends <── lp_certify
      │                 │
      ├──────────> certified_pgfp <── m0_oracle
      └──────────> solver_portfolio <── lp_backends
```

Arrows above mean `dependency → importing consumer`; only the table enumerates the actual direct edges. The source has several import-time re-exports, so a filename-only reorganization may silently change user-visible objects.

## 2. Current API surface and freeze requirements

The **documented primary entrypoint** is `from accelsched import solve`, including `METHODS`, `SolveResult` and `Instance`. Six documented method tokens: `first_fit`, `classic_best_fit`, `lp_order_best_fit`, `lp_best_fit`, `pgfp`, `highs_mip`. The `solve` signature is `solve(instance: Instance, method='pgfp', *, seed=2026, time_limit=10.0)`. Input validation, method defaults, statuses, result fields, timing and random-seed behavior are compatibility-sensitive.

Existing consumer-level imports below are real supported or source-used paths that any subsequent migration must test:

```python
from accelsched import solve, METHODS, SolveResult, Instance
from m0_oracle import Instance as OracleInstance, validate, greedy, shape_lp
from scipy_backend import solve_mip
from workload import build_workload, heterogeneous_example, evaluate
from rolling import Job, simulate
from lp_contract import CanonicalLP
from lp_certify import certify
from lp_backends import RawResult, Attempt, CertifiedResult, solve_lp
from certified_pgfp import canonical_shape, shadow
from solver_portfolio import PortfolioResult, solve_portfolio
```

The ten `py-modules` currently export **more** than these named APIs: direct-imported functions, module attributes, helpers and test-time monkeypatch targets may be observable. Treat the full module namespace as a **compatibility candidate inventory**, not declare every imported/private name an indefinitely supported API. Before implementation, generate a machine-readable complete namespace snapshot (`dir`, `__module__`, signatures, dataclass fields, exception identities, method tokens, defaults) in an isolated baseline installation and classify **documented supported**, **historically used**, and **private / best-effort**. Do not infer `__all__` or a promised API solely from these source declarations.

### Identity and monkeypatch pitfalls

- `accelsched.Instance is m0_oracle.Instance` holds under the current import/re-export wiring. Reimplementing or wrapping the dataclass as a new class breaks `isinstance`, pickle/module path and equality semantics. `Job`, `SolveResult`, `CanonicalLP`, `CertifiedResult`, `PortfolioResult` similarly need identity and inheritance contracts.
- `accelsched.solve` binds imported `greedy` / `solve_mip` names at module load. Tests may patch `accelsched.greedy`, `accelsched.solve_mip`, `lp_backends.importlib.import_module`, `scipy_backend` or a module's original implementation. A wrapper that merely calls a different module's `solve` may change patch interception; test this explicitly.
- Pickle of objects/functions uses `__module__` plus qualified name. Moving the class/function definition changes the serialized locator even if the public import still works. Retain an importable old module location with the **same actual class/function object**, or implement/test a deliberate pickle compatibility mechanism. Avoid duplicate class definitions.
- Import order, monkeypatch/module attributes, star imports, exception types, dataclass field layouts, NumPy array dtypes, LP certificates and backend failure/fallback semantics are migration-sensitive.
- Runtime `from accelsched import solve` and `import accelsched; accelsched.solve` must resolve consistently both from the source checkout and an installed wheel. Do **not** place `accelsched.py` and `accelsched/__init__.py` side by side and assume import precedence. A switch to the `accelsched` package requires a coordinated, tested removal/replacement of the flat module plus explicit facade exports.

## 3. Proposed target design (not created)

```text
src/accelsched/
    __init__.py                # stable public facade: solve, Instance, METHODS, SolveResult
    api.py                     # coordination / public input contracts
    models.py                  # one canonical home for data/result objects
    scheduling/
        greedy.py
        pgfp.py
        lp_guided.py
    optimization/
        contract.py
        certification.py
        backends.py
        portfolio.py
        scipy_mip.py
    simulation/
        rolling.py
        workload.py
    validation/
        physical.py
```

This is a **responsibility map**, not a mandate to split every small file. Preserve a single canonical `Instance` definition. The actual placement of existing attributes and private machinery should follow measured coupling, test/import evidence, and the smallest independently reviewable changes. Production code is ~1.2K lines across ten modules, so deferring this migration is reasonable; the main benefit arises when third-party integrations, multiple interchangeable backends or larger public API boundaries demand it.

### Stable old locations and wheel scope

A later implementation must decide explicitly whether the original top-level modules remain shipped as **compatibility forwarders**, while `accelsched` becomes the new installed package. That usually implies packaging `src/accelsched` **plus** top-level legacy shim `py-modules`, and careful setuptools `package-dir`/wheel-content validation; changing to src layout alone can accidentally drop `examples` and JSON data. Compat forwarders must preserve module-bound patch behavior and object identity, not just function names.

`examples` currently is a real installed Python package and its `python -m examples.capacity_whatif`, `python -m examples.time_window_capacity`, `python -m examples.capacity_decision_report` entrypoints remain an independently required contract. Research scripts under root, `benchmarks/`, `probes/`, `tests/`, `tools/` remain source-only unless explicitly approved otherwise. `queueing.py` is not a v1 wheel member. `tools/source_deposit.py` derives official source scope from declared installed modules; any future scope change requires a separate provenance/ownership review and cannot silently alter release records.

## 4. Migration gates requiring a *separate future Scientist approval*

| Gate | Proposed future change | Required evidence before merge |
| --- | --- | --- |
| P2-A | Freeze installed baseline; no source modification | package path/module namespace/identity snapshots, wheel manifest + SHA, source-deposit manifest, fixture/hash baseline, Python 3.10+ and Windows/Linux import behavior |
| P2-B | Prepare isolated internal package scaffold **only if needed** | no overlapping `accelsched.py`/package ambiguity; import graph and init-order checks; no changed v1 behavior |
| P2-C | Single coordinated public-facade cutover + forwarders | `accelsched.solve`, `m0_oracle.Instance`, monkeypatch interception, qualified names/pickle roundtrip from pre-migration objects, wheel/install from outside checkout, example data |
| P2-D | Move internals in small independent batches | complete production CPU tests, tiny deterministic scheduling/LP certificates, failure-mode equivalence, historical CLI, Benchmark V4 small input/order/placement SHA parity; no new million/10M/GPU experiments |
| P2-E | Optional version/release decision | explicit Scientist authorization, changelog, separately verified wheel/source deposit and license |

**No numeric baseline, solver fallback, tolerance, default, seed, data shape, status label or production behavior can be changed under the guise of relocation.** Preserve the original independent physical oracle. No GitHub Actions/CI requirement; run permitted CPU tests directly in the Engineer environment and report actual versions, exact commands, exceptions, skips, and hashes.

### Required future acceptance commands (illustrative, NOT executed during this documentation pass)

```bash
python -m pytest -q
python -m pytest -q test_m0_oracle.py test_workload.py test_rolling.py
python -m examples.capacity_whatif --input examples/data/static.json
python -m examples.time_window_capacity --input examples/data/dynamic.json
python -m examples.capacity_decision_report --input examples/data/static.json --mode static --format json
python bench_v4_gpu.py --help
python bench_ultra_pgfp.py --help
```

A future install test must additionally create an isolated CPU environment/wheel and test `import accelsched, m0_oracle, workload, rolling, lp_backends` from **outside the checkout**; check exact distribution inventory, examples JSON assets, old API and serialization signatures. A full historical/GPU benchmark must not be launched merely to prove packaging.

## 5. Decision and evidence limits

- **Phase 1 (#115) already achieved the necessary near-term hygiene.** Preserve the current stable modules unless there is a concrete installed-library/integration need.
- This document reflects **actual code imports and definitions** at the frozen main, but it does not assert an exhaustive dynamically evaluated API export list, external plugin compatibility, pickle byte compatibility, or a new wheel test in this pass.
- The longer-term `src/accelsched/` proposal remains **DEFERRED / DESIGN-FROZEN**. There is **no authorization to start P2 code work, publish a release, run Colab or change dependencies**.

Links: [Phase 1 directory guide](repository-layout.md) · [Phase 1 PR #115](https://github.com/JohnnyZPiggy/AccelSched/pull/115) · [Benchmark V4](../research/benchmark-v4-ultra-scale.md).

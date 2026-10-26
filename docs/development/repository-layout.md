# Repository layout and developer guide

Phase 1 changes source locations, not algorithms, installed APIs, datasets or
recorded scientific results. Python 3.10+ and the existing dependencies remain
the runtime contract. No GPU allocation is needed for this maintenance work.

| Location | Canonical source and compatibility |
| --- | --- |
| Root production modules | The ten explicit setuptools `py-modules` remain here, byte unchanged. `queueing.py` also stays here as source-only research code. No `accelsched/` package shadows `accelsched.py`. |
| `examples/` | Installed CLI package and bundled `data/*.json`; unchanged. |
| `tests/` | Canonical implementations of all former root `test_*.py` files. Root files are small import/pytest compatibility adapters, not duplicate test implementations. |
| `benchmarks/historical/` | Canonical V3 and V3.1 CPU scripts. Root `bench_cpu_v3.py` and `bench_cpu_v31.py` retain imports, pytest entrypoints and script execution. |
| Root benchmark hubs | `bench_cpu.py`, `bench_cpu_v2.py`, `bench_m1.py`, `bench_m2.py`, dynamic and V4 workers remain in place. Production/import dependencies and self-subprocess `__file__` paths make a wholesale move unsafe in this phase. |
| `probes/` | Existing opt-in research implementations; unchanged and not installed runtime API. |
| `docs/`, `literature/`, `tools/` | Existing reports/reference papers/source-deposit tool stay. `tools/repository_inventory.py` audits tracked Python syntax and documentation references without importing research modules. |

## Tests and stable entrypoints

From the checkout root, the default discovery target is explicitly `tests/`:

```bash
python -m pytest -q
python -m pytest -q tests/test_m0_oracle.py tests/test_workload.py
```

Legacy paths remain supported, including cross-test imports used by existing
research code:

```bash
python -m pytest -q test_m0_oracle.py test_workload.py
python -m pytest -q bench_cpu_v3.py bench_cpu_v31.py
python bench_v4_gpu.py --help
python bench_ultra_pgfp.py --help
python bench_v4_static.py --help
python bench_v4_dynamic.py --help
```

Do not combine canonical and legacy test paths in the same pytest invocation:
they intentionally expose the same tests. Explicit `pytest .` also visits the
compatibility adapters; use default discovery or `tests/` to avoid duplicates.
The adapters forward functions and fixtures; they are not alternate mathematical
oracles. Three relocated tests resolve repository assets/subprocess cwd from
their new parent directory. The queue comparison fixture normalizes its in-memory
fallback through the same JSON representation as its documented external batch;
its assertions, six actual simulator calls and clean-source requirement remain.
For development in a dirty checkout, generate a genuine queue JSON batch at a
clean commit and supply its path with the existing `SAE_QUEUE_COMPARE_RESULT`
environment variable. Do not fabricate a clean-source flag.

Canonical historical invocations use module mode, so package resolution does not
depend on injecting cwd into `sys.path`:

```bash
python -m benchmarks.historical.bench_cpu_v3
python -m benchmarks.historical.bench_cpu_v31
```

These run the existing CPU studies; they are not `--help` parsers. Old root script
commands still call the same `run()` functions. Direct file execution inside
`benchmarks/historical/` is not the supported new invocation; use `-m` or the root
entrypoint. Benchmark V4 root scripts and all self-worker paths are unchanged.
Benchmark V4 is closed; its original source commits, measurements, limits and
local-only evidence remain the provenance, not newly measured results here.

## Packaging, examples and source deposit

```bash
python -m pip install .
python -m examples.capacity_whatif --input examples/data/static.json
python -m examples.time_window_capacity --input examples/data/dynamic.json
python -m examples.capacity_decision_report --input examples/data/static.json --mode static --format json
```

The wheel still installs only the ten declared modules, `examples` and its two
JSON assets. Tests, probes, historical benchmarks, `queueing.py` and developer
tools are source-checkout utilities, not new wheel packages or dependencies.
`tools/source_deposit.py` continues selecting precisely the declared runtime
modules and example Python files; it does not expand to relocated tests or probes.
No version, release, license or registration/ownership statement is changed.

## Complete syntax inventory and move history

```bash
python tools/repository_inventory.py --repo . --output ../inventory.json
```

The tool examines every tracked Python file using AST, records import/from-import
edges (including relative imports), aliased importlib/subprocess/path calls,
script/path literals, pytest functions/fixtures and packaging declarations, and
lists Markdown links/commands. It performs no dynamic imports or solver calls.
Its graph is syntax-level evidence, not a proof of arbitrary dynamic Python
resolution; subprocess/CLI/install checks complement it. Raw working-file SHA-256
can differ across LF/CRLF checkouts; immutable Git blob IDs and the path map prove
source movement separately.

Phase 1 evidence: [before inventory](inventory-before.json),
[after inventory](inventory-after.json), [original-to-canonical path map](path-map.json).
The dedicated pure-move commit records 51 identical-content renames before any
adapter/path edit. Since legacy paths are then reused by adapters, the aggregate
base-to-head diff may show additions rather than all rename pairs; inspect the
pure-move commit and Git `--follow`, not an invented final R100 classification.
The path map includes unchanged originals and all compatibility paths. No
scientifically meaningful probe, test or historical evidence was deleted.

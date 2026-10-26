# V1.0 release validation protocol

This is a bounded **CPU release** check, not GPU R&D or a new performance benchmark. Actual commands, versions, return codes/stdout/stderr, immutable source/tree/blob, wheel and JSON output hashes are in Issue #101 / its one Draft PR and retained local evidence. No tag/release/application is performed.

## Executed CPU checkpoint (2026-10-26)

At source `086a51ac2cc4f6c616871db33a999394d3ca957f`, a new Windows x86-64 venv (`include-system-site-packages=false`) ran **264 tests PASS in11.52s**. Python3.14.6, NumPy2.5.4, SciPy1.18.1, pytest9.1.1, setuptools84.2.1, pip26.1.2; no optional GPU packages. These are observations, not new dependency pins or a tested Python matrix.

The manifest covered release tests, static/workload/rolling, P0/P1/P2/parser, LP/MIP statuses and optional-backend CPU behavior. Isolated wheel build, install, pip check, `python -I` imports/six methods/rolling/reports outside checkout, all documented CLIs and invalid/mode-mismatch exit2 passed. An initial invocation from the outer workspace found no tests; the corrected source-cwd result is above. Both logs retained.

The subsequent generator path-boundary fix passed **4 focused tests in1.26s**. Final wheel/install and deterministic deposit checks use the final source commit; exact artifacts and command outcomes are in Issue/PR evidence. No scheduler implementation changed, so the core suite is not repeated for a document/deposit-tool-only change.

## Required gates

1. Reproduce base-wheel gap: old0.1.0 wheel lacks examples CLI package.
2. Build1.0.0 candidate wheel from exact committed Git source in D scratch. Fresh venv **without system-site-packages**, no GPU packages, isolated PEP517 build. Record Python, installer/backend/runtime versions and dependency-install report.
3. Install candidate non-editably, pip check and metadata/license/module inventory; `python -I` imports outside checkout must come from venv site-packages. Read packaged assets with importlib.resources.
4. CPU-only import/default/static six methods, original physical validity, rolling completion-before-arrival, P0/P1 and P2 static/dynamic text/JSON, malformed-input exit2. Each subprocess finite; JSON values finite and admissions/eligibility independently checked.
5. From source checkout run focused existing CPU/static/rolling/planner/report/LP-MIP status suites plus release tests; no research GPU/100K suite or unrelated sweep. Tests are developer tooling, not runtime dependencies.
6. Verify bilingual commands/asset paths, local Markdown links, source scope/version/name/ownership consistency, deterministic draft export twice from committed source, unchanged core algorithm/default and no new runtime dependency.

## Known interpretation / remaining gates

Python>=3.10 is the declared minimum, not an executed cross-platform version matrix. A fresh venv on the existing Windows host is an isolated package environment, not a new OS image or hardware machine. Solver ties and timing may vary; output hashes identify real bytes, not a promise of identical runs. Missing family can be a valid rejection; INPUT_INVALID is exit2. MIP static optimality is tolerance-bounded; rolling certificates are per arrival batch, never horizon-global.

The candidate includes ten core modules plus examples; research/FIFO files remain source research, not installed V1 CLI. Optional GPU APIs remain opt-in and untested for this release task. No Colab allocation, package upload or CI action. User-confirmed copyright name is ru zhang; applicant identity spelling and full contribution/third-party rights clearance still require owner review. Technical PASS does not itself authorize a tag, public Release or filing submission.

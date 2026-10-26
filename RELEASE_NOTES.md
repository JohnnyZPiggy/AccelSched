# v1.0.0 — Published GitHub Release (2026-10-26)

发行展示名 / Filing display name: **AccelSched 异构算力资源优化调度软件 V1.0**. Package version: **1.0.0**. The v1.0.0 Git tag and [GitHub Release](https://github.com/JohnnyZPiggy/AccelSched/releases/tag/v1.0.0) were subsequently published on 2026-10-26 (non-draft, non-prerelease). This document preserves the original release preparation scope; it does not claim a PyPI publication or software-registration submission.

## Included existing scope / 既有发行范围

- CPU offline indivisible-job placement with explicit GPU-family eligibility, whole GPU/CPU-core/host-RAM-GiB capacities and six existing methods.
- Immediate-start rolling simulation with completion-before-arrival events; permanent refusal, no queue or migration in this public path.
- P0 static capacity comparison, P1 finite-window comparison, P2 text/JSON capacity report and strict Shared JSON Input v1.
- Original-model certified LP contracts, explicit optional solver/fallback/portfolio APIs. CPU HiGHS works without GPU packages; optional GPU adapters remain opt-in, with no zero-copy or end-to-end GPU acceleration promise.

## Release gaps repaired / 发行缺口修复

- Version metadata 0.1.0 → 1.0.0 candidate; SPDX MIT license files and README/project URLs recorded. Build minimum setuptools77 matches existing SPDX-string metadata support; runtime requirements remain Python>=3.10, NumPy>=1.23, SciPy>=1.9.
- Wheel now includes the existing `examples` CLI package and two small deterministic synthetic JSON inputs. No planner/solver/default-policy code changed.
- Chinese/English quick-start, Chinese design description, validation manifest, ownership checklist and deterministic source-deposit **draft** generation protocol.
- Copyright notice alias changed to **ru zhang**, per the user's 2026-10-26 confirmation; MIT grant text unchanged. User declares no EPFL resources/results involved. This is not a new ownership adjudication or legal transfer.

## Limits / 尚未放行

Research probes, benchmark traces, third-party source/binaries and FIFO research modules are not part of the installed V1 command interface. No live cluster, service, GUI, VRAM/topology model, production trace replay, pricing/ROI or horizon-global optimality claim. A validated static MIP optimum is tolerance-bounded; rolling certificates concern arrival batches only. GPU research is on hold for this Issue.

Release preparation checks and actual versions/hashes are recorded in the Engineer Issue/PR evidence; [validation protocol](docs/release/validation.md) defines the scope. Declaration of Python>=3.10 is not a tested version matrix. Tag/release publication requires Architect T2 and separate owner approval. Filing also requires the applicant to confirm identity spelling, contribution/third-party notices and final deposit formatting; there is no registration number or granted registration.

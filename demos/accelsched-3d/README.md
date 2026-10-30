# AccelSched · Cluster Observatory

A small, static Three.js replay of **real `rolling.simulate` decisions on a labeled
synthetic fixture**. This is not a physical digital twin, trace, hardware performance
model or benchmark. No GPU, Python server, cloud account or external runtime assets
are required in the browser. Production scheduling code is unchanged.

The current presentation uses a bright white-green theme with silver racks in one
front-facing row. All eight nodes and their clickable ID labels fit the default
view, including phone width. Home view clears orbit momentum and restores that
composition; resizing refits the row. The short heading keeps the cluster central.
See [visual revision evidence](evidence/visual-revision.md) for current screenshots.

## Reproduce

Python 3.10+ with the repository's existing NumPy/SciPy dependencies; Node 20.19+
or 22.12+ and npm. From the repository root:

```powershell
python tools/export_3d_replay.py
python -m pytest -q tests/test_3d_replay.py
cd demos/accelsched-3d
npm ci
npm run dev
```

Open **http://127.0.0.1:5174/**. Both dev and preview use strict port 5174; a busy
port fails instead of silently taking another port. Port 5173 and Tailscale routes
are untouched. The server binds only to loopback. A phone preview route is a later
Scientist choice; no network routing or publication is configured here.

```powershell
npm run build
npm run preview
```

Serve `dist/` with any static HTTP server; no Python at runtime. Asset paths support
the domain root. For a subdirectory deployment, set Vite's `--base` at build time.
Do not open `index.html` directly over `file://`.

Browser tests use an installed Microsoft Edge, avoiding a browser download:

```powershell
npm test
```

This Windows Codex environment has npm outside PATH. Its existing fallback is:
`node "$env:LOCALAPPDATA/3D-Playground/tools/package/bin/npm-cli.js"` in place of
`npm`. It is not a new project dependency.

## Data contract / provenance

`public/data/replay.json` is a schema-version-1 bundle, ~46 kB. The exporter calls
the existing `rolling.simulate` once per `first_fit`, `classic_best_fit`, `pgfp`,
with identical ordered nodes/jobs/eligibility, seed **11** and time limit **2.0 s**.
Eight heterogeneous nodes and 32 jobs use abstract A/B GPU families plus CPU-only
nodes; job J32 requires unavailable family C. These labels imply no real GPU model
or speed. Quantities are **whole GPU count, CPU cores, host RAM GiB** in that order.
Time is arbitrary fixture units. Job values/timing/demands are fixture inputs.

Top-level fields: `schema_version`, `synthetic`, `label`, `source_sha`,
`source_dirty`, `seed`, `time_limit`, `resource_units`, `nodes`, `jobs`, `eligible`,
`reports`. Each report is the **unmodified JSON-friendly return** from the engine:
`decisions`, ordered `events`, `batches`, aggregate counts, accounting and measured
wall times. Nulls, errors and solver statuses are preserved. Wall times are actual
export measurements, not predictions; subsequent runs may differ. Source SHA is
the checkout HEAD at export, and `source_dirty` truthfully records working changes.
The evidence document records the final delivery SHA separately.

Example source event (abridged; full report remains in the JSON):

```json
{"time":0,"completed":[],"accepted":["J01","J02","J03"],"rejected":["J04"],"solver_errors":[]}
```

The inspector uses the last event with `time <= playhead`; it never interpolates
resource allocations. Completion releases happen **before** arrivals at the same
timestamp. Rejection is permanent, distinct from `solver_error`. Heuristics never
gain an optimality certificate; batch status is shown in Source & solver status.
Full-horizon counts are labeled separately from current GPU occupancy.

Geometry is cosmetic: each rack represents one node. Eight light strips show
`ceil(8 * max(used / capacity))` over nonzero resource dimensions, **not eight
physical GPUs or measured utilization**. A CPU-only job can light a GPU-capable
rack while occupying zero GPUs. The numeric inspector is authoritative. New-start
and completion colors last 1.5 fixture units after the event; fades only smooth
these state transitions. Rejected/error jobs appear in the event tray, never as
running placements. The event tray shows the latest event's job IDs.

## Interaction / accessibility / performance

Drag to orbit, wheel/pinch to zoom, click/tap a rack or use the eight native node
buttons. Play/pause, reset, range scrub, speed and method controls work with keyboard
and touch. Method changes preserve the playhead for comparison. Playback advances
three fixture units per wall second at 1×, purely for presentation. Home view resets
the camera. Reduced-motion starts paused, disables orbit damping and light fades,
and enables low-power mode. Low power caps pixel ratio at 1 and rendering at 30 Hz;
normal mode caps pixel ratio at 1.5. Hidden tabs stop rendering/playback advancement.
Without WebGL, the real-data inspector and transport remain usable with a visible
explanation. Runtime assets, fonts and geometry are all local/procedural.

## Reuse for another project

Replace `public/data/replay.json` with a schema-1 bundle containing the same three
method keys and report fields; node IDs/capacities and jobs come from that file.
Keep eight nodes for this deliberately fixed rack layout. Adapt the small exporter
to your own real scheduling calls and rerun the independent checks. Change method
labels/keys and layout explicitly if needed; this is not a general replay framework.
Keep provenance and synthetic labels honest. The current fixture does not demonstrate
every possible solver error; an injected-error Python test checks that path.

## Attribution / evidence

Project source and procedural geometry: repository MIT license. Three.js: MIT;
Vite: MIT; TypeScript: Apache-2.0; Playwright: Apache-2.0; dependency notices remain
with installed packages. System fonts only; no downloaded hardware assets or NVIDIA
artwork. See [evidence/verification.md](evidence/verification.md) and screenshots
for the actual check matrix, limitations and measured browser observations.

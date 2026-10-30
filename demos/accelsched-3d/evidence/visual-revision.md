# Architect visual revision

Revision of existing Draft PR #119, after delivery `70e583ea420b2b6f9921d320b634f477a1683a20`.

- Replaced the occluding two-row arrangement with one evenly spaced horizontal row.
- Default camera faces the row, with a slight elevation and aspect-based framing.
  All eight racks are individually visible on initial load. Home clears pending
  orbit momentum; resize refits the row. Orbit and zoom remain available.
- Added readable, selectable ID labels above racks, following their projected
  positions, with green selection matching the rack outline and inspector.
- Replaced dark materials/background/UI with pale green-white, silver/light-gray
  racks, green status highlights and restrained cyan completion accents.
- Reduced the hero heading to a small descriptive title. Kept every control,
  summary counter, inspector field and truthful replay behavior.

Only presentation files, browser checks and accompanying evidence/docs changed.
The exporter, replay state helper, static JSON, Python tests, production scheduling
modules, schema and dependencies have **zero diff** against the prior delivery.
No data regeneration or scheduling experiments were needed for this visual pass.

## Validation

`npm run build`: PASS (TypeScript + Vite). The existing Three.js chunk advisory
remains, approximately 511 kB / 130 kB gzip; no assets or dependencies were added.

`npm test`: PASS, four Edge browser tests against the existing production preview
on **http://127.0.0.1:5174/**. Raw logs: [build](build.txt), [browser](browser-tests.txt).

Both 1440×1080 and 390×844 runs now independently check every rack can be selected
by clicking its actual geometry, without orbiting, and its inspector reports the
expected node. They assert all eight ID labels are on-screen and do not overlap,
both at initial load and after orbit/zoom followed by Home view. Existing method,
scrub, pause/reset, speed/touch, final-release, reduced-motion and WebGL fallback
checks remain. No normal-view console/page errors or external runtime requests.

Visually inspected default, selected-node, playback and phone-width screenshots.
Port 5173 retains its original process; Tailscale settings were not changed.
Physical-phone/Safari/Firefox checks remain NOT_RUN; phone width and touch were
tested through Edge emulation. No merge or public deployment.

![Default view](default.png)
![Selected N08](selected.png)
![Playback](playback.png)
![Phone width](mobile.png)

# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

Fluxel2 is a Tauri desktop port of [Fluxel](https://github.com/TZ387/Fluxel), a browser-based simulator for
light transport in biological tissue (diffusion approximation). Fluxel itself is a static, build-free
HTML/CSS/vanilla-JS app; this project reworks it as a Tauri app (TypeScript frontend + Rust backend),
targeting Linux and Windows as desktop platforms.

The port is underway: two of Fluxel's theoretical models are implemented (Farrell-Patterson-Wilson 1992 and
Kubelka-Munk), plus two added beyond upstream Fluxel — Liemert & Kienle 2010 (N-layer point-source diffusion) and an
N-layer Monte Carlo photon-transport model, which is the default and the reference the other three are checked against
(see README.md's Models section for both) — with a schema-driven parameter UI, JSON save/load of a run's inputs, and a
canvas-based 3-slice volume renderer. See [README.md](README.md)'s Roadmap for what's not built yet — a reasonable
source of next tasks if none is otherwise specified.

- `src/` — frontend (TypeScript, vanilla — no framework): `models.ts` (each model's parameter schema and defaults),
  `ui-params.ts` (renders any model's params generically from that schema and reads it back, including each repeating
  instance's editable name), `settings.ts` (the save/load file format and the checking of a loaded file — pure, so it
  needs no DOM to test), `render.ts` (colormaps, value scales, slice-plane images, the colourbar, and the flat 3-panel
  renderer), `render3d.ts` (the 3-D slice box: an orthographic projection drawn with canvas 2-D affine transforms,
  occlusion ordered by the three planes' BSP — its header comment explains why both of those, and why not WebGL),
  `compute.ts` (the Tauri IPC bridge), `main.ts` (wires it together, and owns the file dialogs). Both renderers take the
  same `SliceScene`, so the layout switch is a choice of function and nothing else; anything about *colour* belongs in
  `render.ts`, which `render3d.ts` imports.
- `src-tauri/src/physics/` — the physics itself, in Rust rather than TypeScript: each model's `derived()`,
  `check_validity()`, and `compute_volume()`, exposed to the frontend as a `<model>_summary`/`<model>_volume`
  Tauri command pair registered in `src-tauri/src/lib.rs`. Lives here rather than in `src/` as JS because the
  per-voxel compute loops are a genuine hot path at the grid sizes this app targets — the same reasoning
  applies to any future compute-heavy addition. `monte_carlo.rs` bends that shape slightly: its
  `compute_volume()` also returns the overlay buffer (recomputing it separately would mean a second
  simulation) and takes a progress callback, wired to a `tauri::ipc::Channel` in lib.rs. It is also the one
  model that runs multi-threaded (`std::thread::scope` over its photon batches, no dependency), which has one
  consequence worth knowing before adding tests: its own tests pin the worker count, because the test harness
  is already parallel and a run per test taking every core makes wall-clock assertions anywhere in the suite
  measure spare capacity rather than code. Keep any timing bound generous for the same reason.
- `tests/` — frontend tests, run with `npm test`. No framework: `tests/run.mjs` bundles each `tests/*.test.ts` with
  esbuild (which is what resolves the extensionless imports `src/` uses, the way Vite does in the app) and runs each
  in its own node process. `harness.ts` is a recording canvas context plus a failure counter — the renderers draw and
  return nothing, so the way to test them is to record what they asked the context to do and check the recording. They
  cover the two renderers plus the settings-file format (`settings.test.ts`, which is pure and installs no `document`
  — the shape of a saved file and the rules for reading a hand-edited or older one are where its decisions are;
  ui-params.ts's own reading and writing of controls is the seam left uncovered). They exist mainly for
  `render3d.ts`'s occlusion order, which is easy to get wrong and hard to see: `render3d.test.ts` rebuilds the camera
  and projection independently from the az/el angles and compares the true depths of both surfaces wherever two drawn
  quads overlap on screen, over the whole camera sphere. That check, and the flat layout's margin arithmetic, each
  caught real bugs that looked fine at the default view. If you change either renderer's geometry or margins, run this
  before trusting it — and if you change the tests, check they can still fail (breaking the quad order or a margin by
  hand should light up the matching assertion).
- `src-tauri/capabilities/default.json` — permission allow-list for what the webview's JS may call natively;
  extend this when adding plugins. `tauri-plugin-dialog` is in it already, for the settings files' native
  save/open pickers; reading and writing the file itself is two `std::fs` commands in `lib.rs` rather than
  `tauri-plugin-fs`, since the path always comes from a picker the user has just used and scope configuration
  is most of what that plugin would add. The same pair is what a CSV/HDF5 export would write through.
- `examples/` is bundled into every installer by `resources` in `tauri.conf.json`, which puts it somewhere
  install-type-specific (`/usr/lib/fluxel2/examples` for the `.deb`, the install folder on Windows, a path
  inside the mount for an AppImage). `examples_dir` in `lib.rs` is what turns that into a path the frontend
  can use, and `main.ts` opens the Load dialog there until the user has picked a directory of their own —
  without which the files ship but nobody finds them. Adding an example is therefore just a file in
  `examples/`; nothing enumerates them by name.

## Environment / commands

- Install deps: `npm install`
- Dev server: `npm run tauri dev`
- Production build: `npm run tauri build`
- Frontend tests: `npm test` (see `tests/` above). `npm run build` typechecks them too, since tsconfig's
  `include` covers `tests` as well as `src`.
- Rust tests: `cargo test --manifest-path src-tauri/Cargo.toml`

Desktop only (Linux + Windows) — no cross-compilation is set up. Producing a Windows installer requires
building on Windows (e.g. via CI with a build matrix), and likewise for Linux.

## Conventions

- Commit messages: by default, the repo owner (Tilen) makes all commits himself after reviewing changes —
  don't run `git add`/`git commit`; propose a one-line commit message suggestion and let him commit it
  instead. He may explicitly authorize committing directly within a given conversation (e.g. "make the
  commits yourself for this session"); treat that as a one-off grant for that conversation, not a standing
  change to this default — go back to proposing messages once it ends.
- Keep changes minimal and behavior-preserving unless asked otherwise; this is a small hobby-scale project —
  avoid speculative abstractions or new dependencies unless asked.
- Don't commit `node_modules/`, `dist/`, or `src-tauri/target/` (already gitignored).

# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

Fluxel2 is a Tauri desktop port of [Fluxel](https://github.com/TZ387/Fluxel), a browser-based simulator for
light transport in biological tissue (diffusion approximation). Fluxel itself is a static, build-free
HTML/CSS/vanilla-JS app; this project reworks it as a Tauri app (TypeScript frontend + Rust backend),
targeting Linux and Windows as desktop platforms.

The port is underway: two of Fluxel's theoretical models are implemented (Farrell-Patterson-Wilson 1992 and
Kubelka-Munk), plus two added beyond upstream Fluxel — Liemert & Kienle 2010 (N-layer point-source diffusion)
and an N-layer Monte Carlo photon-transport model, which is the default and the reference the other three are
checked against (see README.md's Models section for both) — with a schema-driven parameter UI and a
canvas-based 3-slice volume renderer. See [README.md](README.md)'s Roadmap for what's not built yet — a reasonable source of next tasks if
none is otherwise specified.

- `src/` — frontend (TypeScript, vanilla — no framework): `models.ts` (each model's parameter schema and defaults),
  `ui-params.ts` (renders any model's params generically from that schema and reads it back, including each repeating
  instance's editable name), `render.ts` (colormaps, value scales, slice-plane images, the colourbar, and the flat
  3-panel renderer), `render3d.ts` (the 3-D slice box: an orthographic projection drawn with canvas 2-D affine
  transforms, occlusion ordered by the three planes' BSP — its header comment explains why both of those, and why not
  WebGL), `compute.ts` (the Tauri IPC bridge), `main.ts` (wires it together). Both renderers take the same
  `SliceScene`, so the layout switch is a choice of function and nothing else; anything about *colour* belongs in
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
- `tests/` — frontend tests, run with `npm test`. No framework: `tests/run.mjs` bundles each
  `tests/*.test.ts` with esbuild (which is what resolves the extensionless imports `src/` uses, the way Vite
  does in the app) and runs each in its own node process. `harness.ts` is a recording canvas context plus a
  failure counter — the renderers draw and return nothing, so the way to test them is to record what they
  asked the context to do and check the recording. They cover the two renderers, and they exist mainly for
  `render3d.ts`'s occlusion order, which is easy to get wrong and hard to see: `render3d.test.ts` rebuilds the
  camera and projection independently from the az/el angles and compares the true depths of both surfaces
  wherever two drawn quads overlap on screen, over the whole camera sphere. That check, and the flat layout's
  margin arithmetic, each caught real bugs that looked fine at the default view. If you change either
  renderer's geometry or margins, run this before trusting it — and if you change the tests, check they can
  still fail (breaking the quad order or a margin by hand should light up the matching assertion).
- `src-tauri/capabilities/default.json` — permission allow-list for what the webview's JS may call natively;
  extend this when adding plugins (e.g. filesystem access for CSV/HDF5 export).

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

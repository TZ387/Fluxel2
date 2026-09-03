# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

Fluxel2 is a Tauri desktop port of [Fluxel](https://github.com/TZ387/Fluxel), a browser-based simulator for
light transport in biological tissue (diffusion approximation). Fluxel itself is a static, build-free
HTML/CSS/vanilla-JS app; this project reworks it as a Tauri app (TypeScript frontend + Rust backend),
targeting Linux and Windows as desktop platforms.

The port is underway: two of Fluxel's theoretical models are implemented so far (Farrell-Patterson-Wilson
1992 and Kubelka-Munk), with a schema-driven parameter UI and a canvas-based 3-slice volume renderer.

- `src/` — frontend (TypeScript, vanilla — no framework): `models.ts` (each model's parameter schema and
  defaults), `ui-params.ts` (renders any model's params generically from that schema), `render.ts` (the
  3-slice canvas renderer and colormap), `compute.ts` (the Tauri IPC bridge), `main.ts` (wires it together).
- `src-tauri/src/physics/` — the physics itself, in Rust rather than TypeScript: each model's `derived()`,
  `check_validity()`, and `compute_volume()`, exposed to the frontend as a `<model>_summary`/`<model>_volume`
  Tauri command pair registered in `src-tauri/src/lib.rs`. Lives here rather than in `src/` as JS because the
  per-voxel compute loops are a genuine hot path at the grid sizes this app targets — the same reasoning
  applies to any future compute-heavy addition (e.g. the Monte Carlo validation on Fluxel's roadmap).
- `src-tauri/capabilities/default.json` — permission allow-list for what the webview's JS may call natively;
  extend this when adding plugins (e.g. filesystem access for CSV/HDF5 export).

## Environment / commands

- Install deps: `npm install`
- Dev server: `npm run tauri dev`
- Production build: `npm run tauri build`

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

# Fluxel2

A Tauri desktop app (TypeScript frontend + Rust backend) for simulating light transport in biological tissue
(diffusion approximation), targeting Linux and Windows — a schema-driven parameter UI, a 3-slice volume
renderer, and three theoretical models (see Models below). See [AGENTS.md](AGENTS.md) for the current layout.

It started as a port of [Fluxel](https://github.com/TZ387/Fluxel), a static, build-free HTML/CSS/vanilla-JS
browser simulator covering the same physics, but has since grown well beyond it: a third model
(Liemert & Kienle 2010) and the beam-shaping features described below have no Fluxel counterpart.

## AI-assisted development

This project gives AI coding agents a fairly free hand — including letting them commit directly during some
sessions, when explicitly authorized (see the commit convention in [AGENTS.md](AGENTS.md)). Keep that in mind
when reading the code or commit history here.

## Models

Each model is self-contained in Rust under `src-tauri/src/physics/` — its compute, validity checks, and doc
comments with the full derivation notes are the single source of truth (see that directory, not here, for the
math).

- **Farrell, Patterson & Wilson (1992)** — pencil beam, semi-infinite slab. A narrow collimated beam entering a
  homogeneous tissue slab, modelled as a real + image point-source pair below the surface (accounting for the
  refractive-index mismatch at the air-tissue boundary). Has genuine 3-D structure: fluence falls off radially
  from where the beam enters. `src-tauri/src/physics/fpw1992.rs`.
- **Kubelka-Munk** — two-flux, N-layer stack. A 1-D model: the sample is illuminated by a perfectly diffuse flux
  across the whole top face, and two counter-propagating streams (up/down) are tracked through an arbitrary
  stack of homogeneous layers, each with its own absorption, scattering, and thickness. No lateral structure —
  the computed depth profile is broadcast across every (x, y) column. `src-tauri/src/physics/kubelka_munk.rs`.
- **Liemert & Kienle (2010)** — N-layer, point-source diffusion. The combination FPW1992 and Kubelka-Munk
  each stop short of: a point/pencil beam through a stack of homogeneous layers (1 to 8 of them), solved via a
  Fourier-Bessel series (zeros of J0) on a finite cylinder rather than FPW1992's closed-form shortcut, since
  layering breaks the symmetry that shortcut relies on. Each series term reduces to a 1-D problem in depth
  that any number of layers folds into, via a bottom-up reflection-coefficient recursion — the reference
  implementation this was ported from covers only the top and bottom layer, so that recursion (and with it the
  middle-layer Green's function) is derived here, and checked both against the ported two-layer form and
  against a direct numerical solve. Not in upstream Fluxel — added here to fill the gap its own roadmap named.
  `src-tauri/src/physics/liemert_kienle.rs`.

Both point-source models (FPW1992 and Liemert & Kienle) also support widening their beam from an idealised
pencil to a Gaussian or flat-top (disk) profile — the finite-beam convolution shared between them lives in
`src-tauri/src/physics/beam.rs`. Liemert-Kienle folds the beam's profile into its existing Fourier-Bessel series
as a per-mode spectral factor (cheap, exact to the model's own cylinder-radius approximation); FPW1992 has no
such series, so its convolution is a direct 2-D numerical integral over the beam footprint instead.

Both also take a beam *pattern* — a single spot, a line (a scanner's row of pulses), or a square grid (a
fractional handpiece's array) — sharing P0 equally between the spots and superposing their fields, which
diffusion being linear makes exact. The per-spot field is the same function at every spot, just shifted, so it
is evaluated once and reused: a 25-spot grid costs under twice a single spot, not 25 times.

## Roadmap

Adapted from [Fluxel's own roadmap](https://github.com/TZ387/Fluxel#roadmap) — a reasonable source of next
tasks if none is otherwise specified:

- **Monte Carlo validation** — an optional MC reference run to cross-check the diffusion result. Unlike
  upstream Fluxel (which plans this via WebAssembly), this can be plain native Rust here, since Tauri already
  runs a Rust process
- **Export** — download fluence/absorption volumes as CSV or HDF5
- **Isosurface overlay** — 3D isosurface rendering on top of the slice views

## Development

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) is the recommended setup.

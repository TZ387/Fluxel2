# Fluxel2

A Tauri desktop app (TypeScript frontend + Rust backend) for simulating light transport in biological tissue,
targeting Linux and Windows — a schema-driven parameter UI, a 3-slice volume renderer, and four models: a
Monte Carlo reference plus three closed-form approximations (see Models below). See [AGENTS.md](AGENTS.md) for
the current layout.

It started as a port of [Fluxel](https://github.com/TZ387/Fluxel), a static, build-free HTML/CSS/vanilla-JS
browser simulator covering the diffusion-approximation part of this ground, but has since grown well beyond
it: the Monte Carlo model, Liemert & Kienle 2010, and the beam-shaping features described below have no Fluxel
counterpart.

## AI-assisted development

This project gives AI coding agents a fairly free hand — including letting them commit directly during some
sessions, when explicitly authorized (see the commit convention in [AGENTS.md](AGENTS.md)). Keep that in mind
when reading the code or commit history here.

## Models

Each model is self-contained in Rust under `src-tauri/src/physics/` — its compute, validity checks, and doc
comments with the full derivation notes are the single source of truth (see that directory, not here, for the
math).

- **Monte Carlo** — N-layer photon transport; the default, and the reference the other three are checked
  against. Traces photon packets through the layer stack (hop by an exponentially sampled free path, deposit
  weight at each collision, scatter by Henyey-Greenstein, Fresnel reflect/refract at every refractive-index
  step including total internal reflection) rather than approximating the transport equation, so none of the
  restrictions the other models carry apply: a layer thinner than a mean free path, absorption comparable to
  scattering, an index mismatch between layers, or the unscattered first millimetre below the surface are all
  handled as ordinary cases. Written from the published MCML algorithm (Wang, Jacques & Zheng 1995), not
  ported from existing code, so it carries no third-party license obligations.

  Every geometry it accepts is axisymmetric — flat layers, normal incidence, a radially symmetric beam — so it
  scores photons into an (r, z) grid rather than 3-D voxels, which is exactly the axisymmetric kernel the two
  point-source diffusion models already feed to `beam.rs`. One run therefore serves any beam pattern (see
  below), and the beam *profile* costs nothing either: each packet's launch point is drawn from the profile
  instead of the kernel being convolved with it afterwards. That is what makes an in-house Monte Carlo
  affordable here — a full 3-D voxel simulation would need orders of magnitude more photons for the same
  noise. The price is that the answer is statistical: the photon budget is a parameter, error falls as
  1/√photons, and the run reports its own per-voxel standard error as an overlay (batch means over 64 equal
  batches).

  Those batches are also the unit of parallel work, so a run spreads over every core the machine has
  (`std::thread::scope`, no dependency). A batch seeds its own generator from its own index and fills its own
  tally, so nothing writable is shared: no locks, no atomic accumulation, and the answer doesn't depend on how
  many cores ran it — there is a test pinning that down. Measured 4.2x on a 4-core/8-thread laptop, where
  all-core turbo is well below single-core turbo; a desktop should land closer to its thread count.
  `src-tauri/src/physics/monte_carlo.rs`.
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

Both diffusion point-source models (FPW1992 and Liemert & Kienle) also support widening their beam from an idealised
pencil to a Gaussian or flat-top (disk) profile — the finite-beam convolution shared between them lives in
`src-tauri/src/physics/beam.rs`. Liemert-Kienle folds the beam's profile into its existing Fourier-Bessel series
as a per-mode spectral factor (cheap, exact to the model's own cylinder-radius approximation); FPW1992 has no
such series, so its convolution is a direct 2-D numerical integral over the beam footprint instead.

All three point-source models (Monte Carlo included) also take a beam *pattern* — a single spot, a line (a
scanner's row of pulses), or a square grid (a fractional handpiece's array) — sharing P0 equally between the
spots and superposing their fields, which transport being linear makes exact. The per-spot field is the same function at every spot, just shifted, so it
is evaluated once and reused: a 25-spot grid costs under twice a single spot, not 25 times.

## Roadmap

Adapted from [Fluxel's own roadmap](https://github.com/TZ387/Fluxel#roadmap) — a reasonable source of next
tasks if none is otherwise specified:

- **Monte Carlo on the GPU** — `wgpu` + a WGSL compute shader, which reaches Vulkan on Linux and DX12 on
  Windows and so runs on Intel and AMD integrated graphics rather than NVIDIA only, and needs no C++
  toolchain or vendor SDK. The (r, z) tally is small enough to live in a workgroup's shared memory, so a
  workgroup would keep a private tally exactly as a CPU worker does now and only reduce globally at the end —
  the same decomposition, one level down. Two real costs: WGSL is f32-only, so the tally needs fixed-point
  `atomicAdd` on u32 (what MCX does), and it is a genuine port of the inner loop. Worth it on a discrete
  card; on integrated graphics sharing system memory with the CPU it would be a wash at best, since photon
  transport is branch-divergent and tally-heavy — the shape a small iGPU handles worst.

  A cheaper lever first, if runs ever feel slow: tracks average some 500 collisions per photon for typical
  tissue, and a more aggressive roulette threshold trades a little variance for a lot of wall clock.
- **Export** — download fluence/absorption volumes as CSV or HDF5
- **Isosurface overlay** — 3D isosurface rendering on top of the slice views

Cross-checking against a mature external tool ([MCX](https://mcx.space) and its OpenCL variant
[mcxcl](https://github.com/fangq/mcxcl), MMC, mcmatlab) is still worth doing for anything load-bearing — they
are GPU-parallelized, far more general, and far more validated than the model here. What they are not is
bundleable into a Tauri desktop app, which is why this one exists.

## Development

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) is the recommended setup.

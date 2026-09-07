# Fluxel2

A Tauri desktop app (TypeScript frontend + Rust backend) for simulating light transport in biological tissue,
targeting Linux and Windows — a schema-driven parameter UI, a 3-slice volume renderer (see Visualisation
below), and four models: a Monte Carlo reference plus three closed-form approximations (see Models below).
See [AGENTS.md](AGENTS.md) for the current layout.

It started as a port of [Fluxel](https://github.com/TZ387/Fluxel), a static, build-free HTML/CSS/vanilla-JS
browser simulator covering the diffusion-approximation part of this ground, but has since grown well beyond
it: the Monte Carlo model, Liemert & Kienle 2010, and the beam-shaping features described below have no Fluxel
counterpart.

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

## Visualisation

Fluence and absorption are shown side by side, each as three orthogonal cuts through the volume, in either of
two layouts. The default is a **3-D slice box**: the cuts drawn where they actually are inside the tissue, at
true proportions, in a box with cm ticks on its outer edges and the layer interfaces marked on the back walls
— so a 0.3 cm epidermis over a 1.7 cm dermis is drawn as the thin layer it is. Drag either plot to orbit it
and both follow, since the point of the pairing is to read one against the other. The alternative **flat
slices** layout puts the three cuts side by side with their own axes: nothing is foreshortened there, so a
distance on screen is a distance in the tissue, which is the view to read a depth off. The colour scale is
logarithmic by default and switchable to linear, the colormap is inferno by default with the original
blue→red ramp still available, and the slice sliders step one voxel at a time but read out in cm.

The 3-D box is plain canvas 2-D, no WebGL and no dependency, which is possible because an orthographic
projection is linear: an axis-aligned slice rectangle projects to a parallelogram, and the map from image
pixels to that parallelogram is exactly the affine transform canvas 2-D already applies to an image. The
occlusion that a depth buffer would otherwise handle — three full planes intersect, so no fixed draw order
works — comes from splitting each plane into quadrants at the other two and drawing the twelve pieces in the
order the three planes' BSP defines. `src/render3d.ts`'s header comment has the reasoning, including why the
obvious centroid-depth shortcut is wrong.

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
- **Isosurface overlay** — a true isosurface in the 3-D box: the closed shell where the field equals one
  chosen level, most usefully an absorbed-power density corresponding to a damage threshold, which answers
  "how deep and how wide is the region above it" in one shape. Marching cubes belongs in Rust next to the
  physics, returning a mesh over the same raw-bytes IPC the volumes already use; the rendering is the real
  cost, since a translucent shell intersecting the slice planes is where the canvas-2-D approach above runs
  out and WebGL starts paying for itself. Worth most for the multi-spot patterns, where the question is
  whether adjacent spots' fields merge at depth. Cheaper and most of the value: **iso-contour lines** on the
  slices themselves, which marching squares gives for a fraction of the work and which you can read a number
  off.
- **Model-vs-model difference view** — colour bounds come from each volume separately, so two runs are not
  visually comparable today. A locked colour range plus a ratio against a stored previous run, on a diverging
  colormap, would make the README's central claim — Monte Carlo as reference, the other three as
  approximations — visible *spatially*: where diffusion goes wrong, not just that it does.
- **1-D profile plots** — fluence against depth on the beam axis (log y) and against radius at a chosen
  depth, with the Monte Carlo standard error as a band. These are what compare directly to the literature and
  to the Beer-Lambert and diffusion asymptotes the Rust tests already check numerically.

Cross-checking against a mature external tool ([MCX](https://mcx.space) and its OpenCL variant
[mcxcl](https://github.com/fangq/mcxcl), MMC, mcmatlab) is still worth doing for anything load-bearing — they
are GPU-parallelized, far more general, and far more validated than the model here. What they are not is
bundleable into a Tauri desktop app, which is why this one exists.

## Development

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer) is the recommended setup.

//! Monte Carlo photon transport through a layered slab — the reference model
//! here, and the only one that isn't an approximation.
//!
//! The other three models solve an approximate transport equation in closed
//! form (diffusion, or two-flux). This one solves the radiative transfer
//! equation itself by tracing individual photon packets and letting the
//! statistics of many tracks stand in for the answer — no requirement that
//! scattering dominate absorption, no minimum layer thickness, no
//! extrapolated-boundary trick. What it costs instead is time and noise (see
//! the noise overlay below).
//!
//! # Provenance
//!
//! Written from the published algorithm, not ported from existing code: MCML
//! (Wang, Jacques & Zheng, Comput. Methods Programs Biomed. 47(2), 1995) for
//! the transport scheme — launch, exponentially-sampled hops split at layer
//! interfaces, Henyey-Greenstein scattering, roulette on the survivors — and
//! xoshiro256++/splitmix64 (Blackman & Vigna, public domain) for the RNG.
//! Nothing here derives from MCX/µMCX or any GPL-licensed source, so this
//! file imposes no license obligation on the rest of the app.
//!
//! # Why an (r, z) grid rather than 3-D voxels
//!
//! Every geometry this model accepts is axisymmetric about the beam axis, so
//! a single spot's fluence depends only on (r, z) — the same kernel shape
//! beam::sample_axisymmetric_volume already consumes for the diffusion
//! models. That buys three things: a run needs orders of magnitude fewer
//! photons than a voxel-based Monte Carlo for the same noise; a beam
//! *pattern* is free (shift and superpose one kernel per spot, exact because
//! transport is linear); and the beam *profile* is free and exact too, since
//! each photon's launch point is drawn straight from it (launch_radius)
//! rather than convolved in afterwards. The assumption this rests on: tilt
//! the incidence, warp an interface, or embed an inclusion, and the symmetry
//! is gone and a full 3-D grid becomes necessary.
//!
//! This is why the model is parameterized in (N_r, dr, N_z) rather than the
//! diffusion models' (L_x, L_y, N_x, N_y, N_z): the simulation has no x and
//! no y to divide up, and offering three independent lateral knobs for a
//! two-dimensional answer only invited pairs of them that mean nothing. dr
//! is the width of one radial ring — the resolution the answer actually
//! exists at — and N_r is how many of them, so the result reaches
//! R = N_r * dr from the axis.
//!
//! What the viewer draws is still a Cartesian box, since that is the only
//! thing the two renderers know how to draw and the only shape the other
//! three models produce. display_grid is the one place that mapping lives:
//! a square footprint of half-width R, divided into voxels exactly dr
//! across, so a display voxel and a tally ring are the same size and the
//! picture shows the resolution that was computed rather than an
//! interpolated version of it.
//!
//! # Noise, and the overlay
//!
//! A Monte Carlo answer is an estimate with an error bar, so this model
//! reports one: the photon budget splits into equal batches (N_BATCHES),
//! each tallied separately, and the spread between batch totals is the
//! standard "batch means" estimate of variance. That becomes the per-voxel
//! overlay (green/amber/red on REL_SE_GOOD/REL_SE_POOR). Counter-intuitively,
//! the noisiest bins are on the beam axis: a radial bin's volume grows with
//! its index, so the innermost bins are smallest and collect the fewest
//! collisions even though the fluence there is highest.
//!
//! # Parallelism
//!
//! Batches are also the unit of parallel work: each seeds its own generator
//! and fills its own tally, so batches share nothing writable and there is
//! no locking or atomic accumulation — run_mc just hands each worker a
//! contiguous run of them, which is why this scales close to linearly with
//! cores and the answer doesn't depend on how many there are.
//!
//! # Units and normalization
//!
//! A packet launches with weight 1, with specular reflection deducted
//! immediately, so tallies are fractions of incident power P0. Note the
//! diffusion models here launch at full weight and never deduct specular
//! reflection, so their absolute fluence runs a few percent high next to
//! this model's (2.8% for n = 1.4) — real physics, not a discrepancy.

use crate::physics::beam::{self, BeamPattern, BeamProfile, Grid};
use crate::physics::validity::{require, ValidityResult};
use serde::{Deserialize, Serialize};

use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

/// Refractive index of the medium bounding the stack above and below. Air at
/// both ends, matching liemert_kienle.rs's convention so the two models
/// describe the same slab.
const N_OUTSIDE: f64 = 1.0;

/// Below this weight a packet is no longer worth tracing in full, so it
/// enters the roulette: it survives with probability ROULETTE_CHANCE and has
/// its weight scaled up by 1/ROULETTE_CHANCE, and is terminated otherwise.
/// Unbiased (the expected weight is unchanged) and it is what keeps a
/// high-albedo run finite. MCML's own thresholds.
const W_MIN: f64 = 1e-4;
const ROULETTE_CHANCE: f64 = 0.1;

/// Batches the photon budget is split into — doing two jobs at once: the
/// sample size of the batch-means error estimate, and the unit of parallel
/// work (see run_mc). 64 keeps both the estimator's variance and the
/// load-imbalance tail (batches are handed out in contiguous runs, so an
/// unlucky pair costs real time on a busy machine) small up to a 64-core
/// machine, without stranding a worker on odd core counts. Fixed rather
/// than derived from the core count, since it decides which photons get
/// traced — tying it to hardware would make identical parameters give
/// different answers on different machines.
const N_BATCHES: usize = 64;

/// Workers every test in this file runs with — see compute_volume_with for
/// why they don't just take the machine. Two rather than one so that each of
/// them still crosses the parallel path.
#[cfg(test)]
const TEST_WORKERS: usize = 2;

/// How often the run reports progress while the workers are going. Long
/// enough that polling costs nothing, short enough that a status line
/// doesn't look stuck.
const PROGRESS_POLL: Duration = Duration::from_millis(40);

/// Relative standard error boundaries for the noise overlay's three codes.
const REL_SE_GOOD: f64 = 0.05;
const REL_SE_POOR: f64 = 0.20;

/// Ceiling on the (r, z) tally's bin count, so a run can never ask for an
/// unbounded allocation — a wide beam pattern and a small, finely divided
/// grid can push the radial reach up and the bin width down at once, and
/// nothing clamps what's typed into the parameter panel. Hitting this only
/// coarsens the far field; check_validity warns about a reach like that
/// well before it gets here. Costed per worker (each allocates its own
/// tally, see trace_batches): 2M bins is 48 MB, the same order as this
/// app's largest voxel volumes.
const MAX_TALLY_BINS: usize = 2_000_000;

/// Cosine above which a direction counts as exactly along z, where the
/// azimuthal rotation in `scatter` degenerates (0/0).
const COS_ZERO: f64 = 1.0 - 1e-12;

/// Seed the per-batch generators are derived from. Fixed, so that identical
/// parameters give an identical volume — a plot should never change for a
/// reason the user didn't cause. Tests reseed to get genuinely independent
/// runs (see the estimator's own validation).
const SEED: u64 = 0x5EED_0000_0000_0000;

/// Deserialized from one entry of the frontend's layer array, which also
/// carries the editable name that entry is shown under (src/ui-params.ts).
/// Serde ignores fields a struct doesn't declare, which is what lets that
/// name live inside the layer object — and the reason not to add
/// `deny_unknown_fields` here.
#[derive(Deserialize, Clone, Copy)]
pub struct McLayerParams {
    pub mua: f64,
    pub mus: f64,
    pub g: f64,
    pub n: f64,
    pub thickness: f64,
}

#[derive(Deserialize)]
pub struct MonteCarloParams {
    /// Top → bottom; the beam enters the top face of layers[0].
    pub layers: Vec<McLayerParams>,
    pub p0: f64,
    /// "pencil" | "gaussian" | "flattop" — see beam.rs. Unlike the diffusion
    /// models, which convolve their point-source kernel with this profile,
    /// here it is the distribution each photon's launch point is drawn from.
    pub beam_profile: String,
    pub beam_width: f64,
    /// "single" | "cross" | "grid" — see beam.rs.
    pub beam_pattern: String,
    pub pattern_count: usize,
    pub pattern_spacing: f64,
    /// Photon budget in thousands. In thousands because the useful range
    /// spans four decades, which no linear slider handles gracefully in
    /// units of one photon.
    pub photons_k: f64,
    /// Rings in the radial direction. With `dr` this sets how far from the
    /// beam axis the answer is reported: R = n_r * dr (see display_grid).
    pub nr: usize,
    /// Width of one radial ring [cm] — the lateral resolution of both the
    /// tally and the volume built from it.
    pub dr: f64,
    pub nz: usize,
}

#[derive(Serialize, Clone, Copy)]
pub struct McLayerDerived {
    /// Reduced scattering coefficient. Not used by the simulation at all —
    /// this model scatters explicitly through g — but reported because it is
    /// what the other models are parameterized by, so it is what makes a
    /// comparison against them meaningful.
    pub musp: f64,
    /// Single-scattering albedo mu_s/mu_t: the fraction of weight a packet
    /// keeps per collision, and so what sets how long a track runs.
    pub albedo: f64,
    /// Scattering mean free path 1/mu_t — the mean hop length.
    pub mfp: f64,
}

#[derive(Serialize)]
pub struct MonteCarloDerived {
    pub layers: Vec<McLayerDerived>,
    /// Total stack depth, which is also the grid's depth here.
    #[serde(rename = "Lz")]
    pub lz: f64,
    /// How far from the beam axis the answer reaches, N_r * dr [cm].
    /// Reported because the two parameters that set it are a resolution and
    /// a count, so the extent they add up to is worth saying outright.
    pub radius: f64,
    /// How many spots the chosen beam pattern works out to.
    pub spots: usize,
    /// Photons the run will actually trace (the budget rounded to a whole
    /// number of equal batches).
    pub photons: u64,
    /// Fraction of P0 reflected specularly off the top surface, before any
    /// scattering — lost to the tissue no matter how the rest is traced.
    pub specular: f64,
    /// Radial bins the run actually tallies onto, for a sense of what the
    /// photon budget is being spread over. Equal to the `nr` asked for when
    /// the beam is a single spot filling the reported radius, and larger
    /// when the grid's corners or an off-axis beam pattern put a displayed
    /// voxel further from a spot than R (see radial_grid).
    pub n_r: usize,
}

/* ================================================================
   RANDOM NUMBERS
   ================================================================ */

/// xoshiro256++ (Blackman & Vigna), seeded through splitmix64 — both written
/// from their published descriptions, both placed in the public domain by
/// their authors. Fast, 2^256 period, and passes the usual test suites,
/// which is all a photon transport run asks of a generator. Per-batch
/// instances are seeded from a fixed constant plus the batch index, so a run
/// is reproducible: identical parameters give an identical volume, and a
/// plot never changes for reasons the user didn't cause.
struct Rng {
    s: [u64; 4],
}

impl Rng {
    fn seeded(seed: u64) -> Rng {
        let mut z = seed;
        let mut splitmix = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        let a = splitmix();
        let b = splitmix();
        let c = splitmix();
        let d = splitmix();
        Rng { s: [a, b, c, d] }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on [0, 1). 53 bits, the most an f64 mantissa holds.
    #[inline]
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on (0, 1] — for the places a zero would produce an infinity
    /// (the log of the step-length sampling).
    #[inline]
    fn unit_nonzero(&mut self) -> f64 {
        1.0 - self.unit()
    }
}

/* ================================================================
   GEOMETRY
   ================================================================ */

/// One layer, with everything the inner loop needs already divided out.
/// This loop runs some 500 times per photon for typical tissue and so sets
/// the whole model's cost, and a division is several times the price of a
/// multiply — none of these belong inside it.
struct Layer {
    mua: f64,
    mut_: f64,
    /// 1/mu_t: the mean free path, and the collision estimator's path-length
    /// stand-in.
    inv_mut: f64,
    /// mu_a/mu_t: the fraction of its weight a packet leaves at a collision.
    absorb: f64,
    /// Henyey-Greenstein sampling constants, so that drawing a deflection
    /// cosine costs one division rather than five operations on g:
    /// cos(theta) = (hg_c - t^2) * hg_inv2g with t = hg_num/(hg_a + hg_b*u).
    isotropic: bool,
    hg_num: f64,
    hg_a: f64,
    hg_b: f64,
    hg_c: f64,
    hg_inv2g: f64,
    n: f64,
    z_top: f64,
    z_bot: f64,
}

struct Stack {
    layers: Vec<Layer>,
    lz: f64,
}

impl Stack {
    fn new(p: &MonteCarloParams) -> Stack {
        let mut z = 0.0;
        let layers = p
            .layers
            .iter()
            .map(|l| {
                let z_top = z;
                z += l.thickness;
                let mut_ = l.mua + l.mus;
                let isotropic = l.g.abs() <= 1e-6;
                Layer {
                    mua: l.mua,
                    mut_,
                    inv_mut: 1.0 / mut_,
                    absorb: l.mua / mut_,
                    isotropic,
                    hg_num: 1.0 - l.g * l.g,
                    hg_a: 1.0 - l.g,
                    hg_b: 2.0 * l.g,
                    hg_c: 1.0 + l.g * l.g,
                    hg_inv2g: if isotropic { 0.0 } else { 1.0 / (2.0 * l.g) },
                    n: l.n,
                    z_top,
                    z_bot: z,
                }
            })
            .collect();
        Stack { layers, lz: z }
    }

    /// Index of the layer containing depth `z`; the last layer for anything
    /// at or past the stack's floor. Same contract as
    /// liemert_kienle.rs's Stack::layer_at.
    fn layer_at(&self, z: f64) -> usize {
        for (i, l) in self.layers.iter().enumerate() {
            if z <= l.z_bot {
                return i;
            }
        }
        self.layers.len() - 1
    }
}

/// Unpolarized Fresnel reflectance for a ray crossing from index `n1` into
/// `n2` at angle-of-incidence cosine `ca1`, plus the cosine on the far side.
/// The textbook average of the s- and p-polarized coefficients, since an
/// unpolarized packet carries equal parts of each; total internal reflection
/// falls out as sin(theta_t) >= 1.
fn fresnel(n1: f64, n2: f64, ca1: f64) -> (f64, f64) {
    if (n1 - n2).abs() < 1e-12 {
        return (0.0, ca1);
    }
    let sa1 = (1.0 - ca1 * ca1).max(0.0).sqrt();
    let sa2 = n1 * sa1 / n2;
    if sa2 >= 1.0 {
        return (1.0, 0.0);
    }
    let ca2 = (1.0 - sa2 * sa2).max(0.0).sqrt();
    let rs = (n1 * ca1 - n2 * ca2) / (n1 * ca1 + n2 * ca2);
    let rp = (n1 * ca2 - n2 * ca1) / (n1 * ca2 + n2 * ca1);
    (0.5 * (rs * rs + rp * rp), ca2)
}

/// Distance from the beam axis at which to launch a photon, drawn from the
/// beam's own radial power density (beam.rs's radial_weight, normalized to
/// integrate to 1 over the plane) by inverting its cumulative distribution:
/// a 2-D Gaussian of per-axis sigma gives a Rayleigh-distributed radius,
/// sigma*sqrt(-2 ln u); a flat-top disk of radius R gives R*sqrt(u), uniform
/// over its area.
fn launch_radius(beam: &BeamProfile, rng: &mut Rng) -> f64 {
    match *beam {
        BeamProfile::Pencil => 0.0,
        BeamProfile::Gaussian { sigma } => sigma * (-2.0 * rng.unit_nonzero().ln()).sqrt(),
        BeamProfile::FlatTop { radius } => radius * rng.unit().sqrt(),
    }
}

/* ================================================================
   THE (r, z) TALLY GRID
   ================================================================ */

/// The axisymmetric grid photons are scored into. Laid out z-contiguous
/// (`ir * n_z + iz`) to match how it is read back: one lookup per (x, y)
/// column walks a whole column of z at once, the same access pattern
/// beam::sample_axisymmetric_volume's own table uses.
struct TallyGrid {
    n_r: usize,
    n_z: usize,
    dr: f64,
    dz: f64,
    /// Reciprocals of the two above, for the same reason Layer precomputes
    /// its own: finding a collision's bin is the innermost thing this model
    /// does, and a divide costs several times a multiply. Worth about 10% of
    /// the run — measured as the faster variant in four of five paired,
    /// core-pinned A/B rounds, but the spread between repeats on a
    /// thermally-throttled laptop is wider than the effect, so treat the
    /// figure as a ballpark rather than a benchmark.
    inv_dr: f64,
    inv_dz: f64,
    /// Radius each radial bin's value is taken to represent. Not simply the
    /// bin's midpoint: a bin is an annulus, so its area-weighted mean radius
    /// sits slightly outside the midpoint, and near the axis — where the
    /// fluence varies fastest — the difference matters. MCML's correction,
    /// r = (i + 0.5 - 1/(12(i + 0.5))) dr.
    r_c: Vec<f64>,
}

impl TallyGrid {
    fn new(n_r: usize, n_z: usize, dr: f64, dz: f64) -> TallyGrid {
        let r_c = (0..n_r)
            .map(|i| {
                let m = i as f64 + 0.5;
                (m - 1.0 / (12.0 * m)) * dr
            })
            .collect();
        TallyGrid { n_r, n_z, dr, dz, inv_dr: 1.0 / dr, inv_dz: 1.0 / dz, r_c }
    }

    fn len(&self) -> usize {
        self.n_r * self.n_z
    }

    /// Volume of bin (ir, ·): the annulus [ir*dr, (ir+1)*dr] one bin deep.
    fn bin_volume(&self, ir: usize) -> f64 {
        PI * (2 * ir + 1) as f64 * self.dr * self.dr * self.dz
    }

    /// Add `v` to the bin containing (x, y, z). A collision outside the
    /// grid's radial reach is dropped: the kernel is only ever read out to
    /// n_r * dr, so what lies beyond it is not part of the answer.
    #[inline]
    fn add(&self, tally: &mut [f64], x: f64, y: f64, z: f64, v: f64) {
        let ir = ((x * x + y * y).sqrt() * self.inv_dr) as usize;
        if ir >= self.n_r {
            return;
        }
        let iz = ((z * self.inv_dz) as usize).min(self.n_z - 1);
        tally[ir * self.n_z + iz] += v;
    }

    /// The two radial bins bracketing `rho` and the fraction between them,
    /// for linear interpolation over the bins' representative radii. Flat
    /// inside the first radius and past the last, where there is nothing to
    /// interpolate against. Split out from `lookup` because a caller walking
    /// a whole column of z at one radius should pay for the search once, not
    /// once per depth.
    fn radial_weights(&self, rho: f64) -> (usize, usize, f64) {
        if rho <= self.r_c[0] {
            return (0, 0, 0.0);
        }
        let hi = self.r_c.partition_point(|&v| v < rho);
        // hi == 0 can't happen for a finite rho past r_c[0] — the predicate
        // holds at index 0 by definition — but it does for a NaN, where
        // every comparison is false. Caught here rather than left to
        // underflow `hi - 1` below into a panic.
        if hi == 0 || hi >= self.n_r {
            return (self.n_r - 1, self.n_r - 1, 0.0);
        }
        let f = (rho - self.r_c[hi - 1]) / (self.r_c[hi] - self.r_c[hi - 1]);
        (hi - 1, hi, f)
    }

    /// Index of the depth bin containing `z`. No interpolation in z: the
    /// grid's depth bins are the voxel grid's own, so every value asked for
    /// sits at a bin centre (see radial_grid / compute_volume).
    fn depth_bin(&self, z: f64) -> usize {
        ((z * self.inv_dz) as usize).min(self.n_z - 1)
    }

    /// Value of a table on this grid at an arbitrary (rho, z).
    fn lookup(&self, table: &[f64], rho: f64, z: f64) -> f64 {
        let iz = self.depth_bin(z);
        let (lo, hi, f) = self.radial_weights(rho);
        let (a, b) = (table[lo * self.n_z + iz], table[hi * self.n_z + iz]);
        a + (b - a) * f
    }
}

/// The Cartesian voxel box the (r, z) answer is drawn in — the one place
/// the model's own (N_r, dr, N_z) parameterization is turned into the shape
/// both renderers and the other three models speak (see this module's doc
/// comment for why the two differ at all).
///
/// A square footprint of half-width R = N_r * dr, cut into voxels exactly dr
/// across, so N_x = N_y = 2*N_r. The frontend has to size the IPC buffer
/// before it can read it, so it works the same mapping out for itself —
/// `grid` in src/models.ts. Change one and change the other; compute.ts
/// checks the two agreed by measuring the buffer it got back.
fn display_grid(p: &MonteCarloParams, lz: f64) -> Grid {
    let n_r = p.nr.max(1);
    let n_z = p.nz.max(1);
    let width = 2.0 * n_r as f64 * p.dr;
    Grid { lx: width, ly: width, nx: 2 * n_r, ny: 2 * n_r, nz: n_z, dz: lz / n_z as f64 }
}

/// The (r, z) grid a run tallies onto. `dr` is the ring width asked for, and
/// the reach covers every displayed voxel: the box's own corners sit at
/// R*sqrt(2), and an off-axis beam pattern puts them further still from the
/// spot they are measured against, since a voxel in one corner can be far
/// from a spot in the other. So n_r is generally larger than the parameter
/// of the same name — the extra rings are what fills in the corners, not
/// extra resolution. Depth bins are the voxel grid's own, so no depth
/// interpolation is ever needed.
fn radial_grid(p: &MonteCarloParams, pattern: &BeamPattern, lz: f64) -> TallyGrid {
    let grid = display_grid(p, lz);
    let reach = beam::max_kernel_radius(grid.lx, grid.ly, pattern) * 1.0001;
    let dr = p.dr.max(f64::MIN_POSITIVE);
    let n_z = grid.nz;
    // Capped before the cast, not after: a dr of zero makes the quotient
    // infinite, and an infinite (or NaN) float-to-integer cast saturates
    // rather than erroring, so the `+ 1` below would be the thing that
    // finally panicked.
    let cap = (MAX_TALLY_BINS / n_z).max(1);
    let wanted = (reach / dr).ceil().min(cap as f64) as usize + 1;
    TallyGrid::new(wanted.clamp(1, cap), n_z, dr, lz / n_z as f64)
}

/* ================================================================
   THE SIMULATION
   ================================================================ */

/// Where the launched weight ended up. Only the totals — the per-bin story
/// is in the tally — and only used to check energy conservation in the
/// tests, which is the one global identity a transport solver has to satisfy.
#[derive(Default)]
struct RunStats {
    specular: f64,
    absorbed: f64,
    /// Escaped back out of the top surface (diffuse reflectance).
    reflected: f64,
    /// Escaped through the stack's floor.
    transmitted: f64,
}

struct McRun {
    grid: TallyGrid,
    /// Fluence per unit incident power [1/cm^2] on the (r, z) grid.
    phi: Vec<f64>,
    /// Standard error of `phi`, same layout and units.
    se: Vec<f64>,
    /// Where the launched weight ended up, and how much of it there was.
    /// Nothing in the UI shows these — the summary command can't, since it
    /// deliberately doesn't run the simulation — but the energy balance is
    /// the one identity that has to hold exactly, so the tests check it.
    #[cfg_attr(not(test), allow(dead_code))]
    stats: RunStats,
    #[cfg_attr(not(test), allow(dead_code))]
    photons: u64,
}

/// Trace one photon packet from launch to termination, scoring into `tally`.
///
/// The loop is MCML's hop/drop/spin, with the one addition that a step is
/// split at a layer interface rather than being redrawn: the unused part is
/// carried across as remaining optical depth (`sleft`), which is what makes
/// the step-length distribution correct across a boundary — and what makes
/// splitting a homogeneous slab into two identical layers change nothing at
/// all (see tests).
fn trace_photon(
    rng: &mut Rng,
    stack: &Stack,
    beam: &BeamProfile,
    grid: &TallyGrid,
    tally: &mut [f64],
    specular: f64,
    stats: &mut RunStats,
) {
    let r_l = launch_radius(beam, rng);
    let psi = 2.0 * PI * rng.unit();
    let (mut x, mut y, mut z) = (r_l * psi.cos(), r_l * psi.sin(), 0.0);
    let (mut ux, mut uy, mut uz) = (0.0, 0.0, 1.0);

    let mut w = 1.0 - specular;
    let mut li = 0usize;
    // Remaining optical depth of a step interrupted by an interface; 0 means
    // "draw a fresh step".
    let mut sleft = 0.0f64;

    loop {
        let layer = &stack.layers[li];

        // Hop. The step length is exponentially distributed with mean free
        // path 1/mu_t, so s = -ln(u)/mu_t.
        let s = layer.inv_mut
            * if sleft > 0.0 {
                sleft
            } else {
                -rng.unit_nonzero().ln()
            };

        // Would it leave this layer first?
        let db = if uz > 0.0 {
            (layer.z_bot - z) / uz
        } else if uz < 0.0 {
            (layer.z_top - z) / uz
        } else {
            f64::INFINITY
        };

        if db < s {
            x += ux * db;
            y += uy * db;
            z += uz * db;
            sleft = (s - db) * layer.mut_;

            let down = uz > 0.0;
            let next = if down {
                (li + 1 < stack.layers.len()).then(|| li + 1)
            } else {
                (li > 0).then(|| li - 1)
            };
            let n_next = next.map_or(N_OUTSIDE, |i| stack.layers[i].n);

            let (r, ca2) = fresnel(layer.n, n_next, uz.abs());
            // An index-matched interface reflects nothing, so it consumes no
            // random number — without that short-circuit, describing one
            // medium as two identical layers would perturb the random
            // stream and change the answer for no physical reason.
            if r > 0.0 && rng.unit() < r {
                uz = -uz;
            } else {
                match next {
                    Some(i) => {
                        // Snell's law, in direction-cosine form: the
                        // transverse cosines scale by n1/n2 and the axial
                        // one is the refracted cosine Fresnel already found.
                        let ratio = layer.n / n_next;
                        ux *= ratio;
                        uy *= ratio;
                        uz = if down { ca2 } else { -ca2 };
                        li = i;
                    }
                    None => {
                        if down {
                            stats.transmitted += w;
                        } else {
                            stats.reflected += w;
                        }
                        return;
                    }
                }
            }
            continue;
        }

        x += ux * s;
        y += uy * s;
        z += uz * s;
        sleft = 0.0;

        // Drop. The collision estimator scores w/mu_t here: the expected
        // path length a packet of weight w contributes to the bin it
        // collided in. Fluence rather than absorbed weight, so that a depth
        // bin straddling a layer boundary needs no per-layer mu_a to undo
        // (the absorbed-density channel is recovered as mu_a * Phi later,
        // in beam::sample_axisymmetric_volume).
        grid.add(tally, x, y, z, w * layer.inv_mut);

        let dw = w * layer.absorb;
        w -= dw;
        stats.absorbed += dw;

        // Spin. Henyey-Greenstein deflection, uniform
        // azimuth. The inverted cumulative distribution is
        // cos(theta) = (1 + g^2 - ((1-g^2)/(1-g+2gu))^2)/(2g), grouped into
        // the per-layer constants above; g = 0 degenerates to 0/0, and
        // scattering is isotropic there anyway.
        let ct = if layer.isotropic {
            2.0 * rng.unit() - 1.0
        } else {
            let t = layer.hg_num / (layer.hg_a + layer.hg_b * rng.unit());
            ((layer.hg_c - t * t) * layer.hg_inv2g).clamp(-1.0, 1.0)
        };
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        // The azimuth is uniform, but taking its sine and cosine outright is
        // the single most expensive thing in this loop — which runs some 500
        // times per photon for typical tissue, so it sets the model's whole
        // cost. Marsaglia's polar method gets the same pair from a point
        // rejection-sampled out of the unit disc: two uniforms, a rejection
        // rate of 1 - pi/4, and one reciprocal square root. Rejection makes
        // the iteration count vary, but only by 27% on average.
        let (cp, sp) = loop {
            let a = 2.0 * rng.unit() - 1.0;
            let b = 2.0 * rng.unit() - 1.0;
            let r2 = a * a + b * b;
            if r2 > 0.0 && r2 <= 1.0 {
                let inv = 1.0 / r2.sqrt();
                break (a * inv, b * inv);
            }
        };

        if uz.abs() > COS_ZERO {
            // Along the axis the azimuthal frame below degenerates, and the
            // new direction is just (theta, psi) read off directly.
            ux = st * cp;
            uy = st * sp;
            uz = ct * uz.signum();
        } else {
            let d = (1.0 - uz * uz).sqrt();
            let k = st / d;
            let nux = k * (ux * uz * cp - uy * sp) + ux * ct;
            let nuy = k * (uy * uz * cp + ux * sp) + uy * ct;
            let nuz = -st * cp * d + uz * ct;
            ux = nux;
            uy = nuy;
            uz = nuz;
        }

        // Roulette, once the packet is too faint to be worth its remaining
        // steps. Terminated weight is deliberately not booked anywhere: the
        // survivors' boost replaces it in expectation, which is why energy
        // conservation holds statistically rather than exactly.
        if w < W_MIN {
            if rng.unit() < ROULETTE_CHANCE {
                w /= ROULETTE_CHANCE;
            } else {
                return;
            }
        }
    }
}

/// One worker's share of a run, reduced into the same pair of moments the
/// whole run wants so that combining shares is a plain elementwise add.
struct Partial {
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    stats: RunStats,
}

/// Trace the batches `range` names. Everything a worker touches is either
/// read-only (the stack, the beam, the grid's geometry) or freshly allocated
/// here, so there is nothing to lock, nothing to atomically add, and no
/// false sharing between workers — the only shared write is the batch
/// counter that feeds the progress readout, and nothing reads it for the
/// answer.
fn trace_batches(
    range: std::ops::Range<usize>,
    per_batch: u64,
    seed: u64,
    stack: &Stack,
    beam: &BeamProfile,
    grid: &TallyGrid,
    specular: f64,
    done: &AtomicUsize,
) -> Partial {
    let mut sum = vec![0.0f64; grid.len()];
    let mut sum_sq = vec![0.0f64; grid.len()];
    let mut batch = vec![0.0f64; grid.len()];
    let mut stats = RunStats::default();

    for b in range {
        batch.iter_mut().for_each(|v| *v = 0.0);
        // Each batch's generator is seeded from its own index, so which
        // photons a batch traces doesn't depend on which worker picked it up
        // or on what ran before it. That independence is the whole reason
        // this parallelizes without changing the answer.
        let mut rng = Rng::seeded(seed ^ b as u64);
        for _ in 0..per_batch {
            trace_photon(&mut rng, stack, beam, grid, &mut batch, specular, &mut stats);
        }
        // Fold the finished batch into this worker's running moments. Each
        // batch is an independent estimate of the same quantity, so their
        // spread is the variance of the total (see this module's doc
        // comment).
        for (k, &v) in batch.iter().enumerate() {
            sum[k] += v;
            sum_sq[k] += v * v;
        }
        done.fetch_add(1, Ordering::Relaxed);
    }

    Partial { sum, sum_sq, stats }
}

/// Workers a run uses unless told otherwise: one per hardware thread, capped
/// at the number of batches there are to hand out. Split out so that a test
/// can pin it and check the answer doesn't depend on it.
fn default_workers() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get()).min(N_BATCHES)
}

/// Whole photon budget, rounded down to a whole number of equal batches (and
/// at least one photon per batch, so the batch-means estimate always has
/// N_BATCHES samples to work with).
fn photon_budget(p: &MonteCarloParams) -> (u64, u64) {
    let requested = (p.photons_k.max(0.0) * 1000.0).round().max(0.0) as u64;
    let per_batch = (requested / N_BATCHES as u64).max(1);
    (per_batch * N_BATCHES as u64, per_batch)
}

/// Trace the whole budget in parallel and reduce the batches into fluence +
/// standard error. `progress` is called on *this* thread as batches finish
/// (so a caller need not be thread-safe), with however many intermediate
/// reports the poll interval catches, and always with 1.0 at the end.
fn run_mc(
    p: &MonteCarloParams,
    stack: &Stack,
    beam: &BeamProfile,
    pattern: &BeamPattern,
    seed: u64,
    workers: usize,
    mut progress: impl FnMut(f64),
) -> McRun {
    let grid = radial_grid(p, pattern, stack.lz);
    let (photons, per_batch) = photon_budget(p);
    let (specular, _) = fresnel(N_OUTSIDE, stack.layers[0].n, 1.0);

    // Each worker takes a contiguous run of batch indices. Contiguous and
    // fixed rather than claimed on demand, because floating-point addition
    // isn't associative: a fixed split plus a reduction in worker order is
    // what keeps a run reproducible. (Across machines with different core
    // counts the last bits can still differ — orders of magnitude below the
    // statistical error the run reports for itself, and there is a test
    // pinning that down.)
    let workers = workers.clamp(1, N_BATCHES);
    let done = AtomicUsize::new(0);
    let mut partials: Vec<Partial> = Vec::with_capacity(workers);
    // Outside the scope because the last report has to be reconciled after
    // it: a worker bumps the counter before it finishes, so the loop below
    // can legitimately observe every batch done and report 1.0 itself.
    let mut reported = 0usize;

    thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let range = (w * N_BATCHES / workers)..((w + 1) * N_BATCHES / workers);
                let done = &done;
                let grid = &grid;
                scope.spawn(move || {
                    trace_batches(range, per_batch, seed, stack, beam, grid, specular, done)
                })
            })
            .collect();

        // Progress is reported from this thread rather than called back from
        // the workers: it drives a UI channel and stays single-threaded that
        // way, and polling the handles (rather than waiting for the counter
        // to reach N_BATCHES) still terminates if a worker panics.
        while handles.iter().any(|h| !h.is_finished()) {
            let n = done.load(Ordering::Relaxed);
            if n > reported {
                reported = n;
                progress(n as f64 / N_BATCHES as f64);
            }
            thread::sleep(PROGRESS_POLL);
        }

        // Joined in worker order, which is what makes the fold below
        // deterministic.
        partials.extend(
            handles
                .into_iter()
                .map(|h| h.join().expect("a Monte Carlo worker panicked")),
        );
    });
    if reported < N_BATCHES {
        progress(1.0);
    }

    let mut stats = RunStats::default();
    let mut sum = vec![0.0f64; grid.len()];
    let mut sum_sq = vec![0.0f64; grid.len()];
    for part in &partials {
        for k in 0..sum.len() {
            sum[k] += part.sum[k];
            sum_sq[k] += part.sum_sq[k];
        }
        stats.absorbed += part.stats.absorbed;
        stats.reflected += part.stats.reflected;
        stats.transmitted += part.stats.transmitted;
    }
    stats.specular = specular * photons as f64;

    let k = N_BATCHES as f64;
    let mut phi = vec![0.0f64; grid.len()];
    let mut se = vec![0.0f64; grid.len()];
    for ir in 0..grid.n_r {
        // Fluence = (path length * weight) / volume, per launched photon.
        let norm = 1.0 / (photons as f64 * grid.bin_volume(ir));
        for iz in 0..grid.n_z {
            let idx = ir * grid.n_z + iz;
            let total = sum[idx];
            phi[idx] = total * norm;
            // Sample variance of one batch, scaled to the variance of their
            // sum (k independent batches), then to the same units as phi.
            let var_batch = ((sum_sq[idx] - total * total / k) / (k - 1.0)).max(0.0);
            se[idx] = (k * var_batch).sqrt() * norm;
        }
    }

    McRun { grid, phi, se, stats, photons }
}

/* ================================================================
   MODEL INTERFACE
   ================================================================ */

pub fn derived(p: &MonteCarloParams) -> MonteCarloDerived {
    let layers: Vec<McLayerDerived> = p
        .layers
        .iter()
        .map(|l| McLayerDerived {
            musp: l.mus * (1.0 - l.g),
            albedo: l.mus / (l.mua + l.mus),
            mfp: 1.0 / (l.mua + l.mus),
        })
        .collect();
    let lz = p.layers.iter().map(|l| l.thickness).sum::<f64>();
    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    let (specular, _) = fresnel(N_OUTSIDE, p.layers[0].n, 1.0);

    MonteCarloDerived {
        layers,
        lz,
        radius: p.nr.max(1) as f64 * p.dr,
        spots: pattern.len(),
        photons: photon_budget(p).0,
        specular,
        n_r: radial_grid(p, &pattern, lz).n_r,
    }
}

/// Unlike the other three models this one has no approximation to be weakly
/// justified, so nothing here questions the physics: a thin layer, a strong
/// absorber, a bare index step are all traced exactly. What can go wrong is
/// sampling — too few photons for the grid, or a field so wide that its
/// outskirts are reached by almost no photons — plus the non-physical inputs
/// every model has to reject.
pub fn check_validity(p: &MonteCarloParams, derived: &MonteCarloDerived) -> ValidityResult {
    let mut reasons = Vec::new();

    for (i, l) in p.layers.iter().enumerate() {
        let at = |name: &str| format!("layer {}: {}", i + 1, name);
        require(&mut reasons, l.mua > 0.0, &at("μ<sub>a</sub>"), "greater than 0", l.mua);
        require(&mut reasons, l.mus > 0.0, &at("μ<sub>s</sub>"), "greater than 0", l.mus);
        require(&mut reasons, (-1.0..1.0).contains(&l.g), &at("g"), "above -1 and below 1", l.g);
        require(&mut reasons, l.n >= 1.0, &at("n"), "at least 1", l.n);
        require(&mut reasons, l.thickness > 0.0, &at("thickness"), "greater than 0", l.thickness);
    }
    require(&mut reasons, p.p0 > 0.0, "P<sub>0</sub>", "greater than 0", p.p0);
    require(&mut reasons, p.dr > 0.0, "Δr ring width", "greater than 0", p.dr);
    require(&mut reasons, p.nr >= 1, "N<sub>r</sub>", "at least 1", p.nr as f64);
    require(&mut reasons, p.photons_k > 0.0, "photon budget", "greater than 0", p.photons_k);
    if !reasons.is_empty() {
        return ValidityResult { valid: false, reasons };
    }

    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    let box_grid = display_grid(p, derived.lz);
    if let Some(reason) =
        beam::pattern_extent_warning(&pattern, box_grid.lx, box_grid.ly, "raise N<sub>r</sub> or Δr")
    {
        reasons.push(reason);
    }

    // The one thing this model assumes, said out loud when the beam breaks
    // it. Not an error: the slab is laterally uniform, so a pattern really
    // is one shifted copy of the same kernel per spot and the superposition
    // is exact — what a multi-spot pattern costs is that the single kernel
    // now has to cover the whole pattern's reach on the same photon budget,
    // and that the noise overlay treats the spots' errors as independent
    // when they are all read off that one kernel, so it reads a little
    // optimistic where spots overlap.
    if !pattern.is_single() {
        reasons.push(format!(
            "the beam pattern's {} spots are not radially symmetric, and this model is: it traces \
             one axisymmetric (r, z) kernel and superposes a shifted copy of it per spot. The sum \
             itself is exact — the layers are flat and uniform, so every spot really does see the \
             same kernel — but that one kernel now has to reach across the whole pattern on the \
             same photon budget, and the noise overlay adds the spots' errors as if they were \
             independent when they all come off it, so it reads optimistic where spots overlap. \
             Read the overlay, and treat a wide pattern as needing more photons than a single spot \
             would",
            pattern.len()
        ));
    }

    // Every photon's track is spread over the whole (r, z) grid, so there is
    // no clean photons-per-bin figure — but a budget below a few photons per
    // bin cannot produce a usable kernel however the tracks fall.
    let bins = derived.n_r * box_grid.nz;
    let per_bin = derived.photons as f64 / bins as f64;
    if per_bin < 20.0 {
        reasons.push(format!(
            "{} photons over an {} x {} (r, z) kernel grid is only {:.1} per bin — expect a visibly \
             noisy volume. Raise the photon budget, or widen Δr and coarsen N<sub>z</sub>. \
             Switch on the noise overlay to see which voxels are actually affected",
            derived.photons, derived.n_r, box_grid.nz, per_bin
        ));
    }

    // How far the kernel has to reach, in penetration depths of the first
    // layer. Fluence falls off roughly exponentially at that rate, and the
    // photon count with it, so a kernel asked to span many decades of decay
    // is asking for its far field to be sampled by almost nothing.
    let l1 = &p.layers[0];
    let mueff = (3.0 * l1.mua * (l1.mua + l1.mus * (1.0 - l1.g))).sqrt();
    let reach = beam::max_kernel_radius(box_grid.lx, box_grid.ly, &pattern);
    if reach * mueff > 15.0 {
        reasons.push(format!(
            "the kernel has to reach {:.2} cm, about {:.0} penetration depths (1/μ<sub>eff</sub> = \
             {:.3} cm) — fluence out there is some 10<sup>{:.0}</sup> times below the peak, and \
             hardly any photon gets that far, so the volume's outskirts will be noise. Lower \
             N<sub>r</sub>/Δr or the spot spacing, or accept that only the bright \
             region is meaningful",
            reach, reach * mueff, 1.0 / mueff, reach * mueff / 2.303
        ));
    }

    // A packet that keeps essentially all its weight per collision survives
    // thousands of them, and the run time goes with it.
    if let Some((i, worst)) = derived
        .layers
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.albedo.total_cmp(&b.1.albedo))
    {
        if worst.albedo > 0.9995 {
            reasons.push(format!(
                "layer {}'s single-scattering albedo is {:.5} — a packet keeps almost all its \
                 weight per collision, so tracks run to thousands of steps and the run will be \
                 slow. Physically fine, just expensive",
                i + 1,
                worst.albedo
            ));
        }
    }

    // The innermost radial bin is an area average over 0 <= r < dr, and a
    // pencil beam's fluence varies fastest exactly there.
    let mfp_transport = 1.0 / (l1.mua + l1.mus * (1.0 - l1.g));
    if BeamProfile::from_params(&p.beam_profile, p.beam_width).is_pencil() && p.dr > mfp_transport {
        reasons.push(format!(
            "the radial ring width Δr ({:.3} cm) is wider than layer 1's transport mean free path \
             ({:.3} cm), so the on-axis peak of an idealised pencil beam is averaged away over \
             the innermost ring. Narrow Δr (raising N<sub>r</sub> to keep the same reach), or use \
             a beam profile with a real width",
            p.dr, mfp_transport
        ));
    }

    ValidityResult {
        valid: reasons.is_empty(),
        reasons,
    }
}

/// Runs the simulation and builds the voxel volumes from its axisymmetric
/// kernel. Returns the noise overlay alongside phi/abs rather than exposing
/// a separate compute_validity_volume like the other models, because that
/// would mean running the whole simulation a second time.
pub fn compute_volume(
    p: &MonteCarloParams,
    progress: impl FnMut(f64),
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    compute_volume_with(p, default_workers(), progress)
}

/// As compute_volume, with the worker count pinned. Exists so that the tests
/// can keep their own footprint small: the test harness already runs tests
/// concurrently, so a run per test taking every core would oversubscribe the
/// machine several times over and make wall-clock assertions anywhere in the
/// suite (this module's, and other models') measure spare capacity rather
/// than code.
fn compute_volume_with(
    p: &MonteCarloParams,
    workers: usize,
    progress: impl FnMut(f64),
) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
    let beam_profile = BeamProfile::from_params(&p.beam_profile, p.beam_width);
    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    let stack = Stack::new(p);
    let run = run_mc(p, &stack, &beam_profile, &pattern, SEED, workers, progress);

    let grid = display_grid(p, stack.lz);
    let (phi, abs) = beam::sample_axisymmetric_volume(
        &grid,
        &pattern,
        p.p0,
        |z| stack.layers[stack.layer_at(z)].mua,
        |rho, z| run.grid.lookup(&run.phi, rho, z),
    );
    let codes = noise_codes(&grid, p.p0, &pattern, &run, &phi);
    (phi, abs, codes)
}

/// Per-voxel noise code for the overlay (0 poor, 1 marginal, 2 good) — this
/// model's counterpart to compute_validity_volume in the other point-source
/// models: "how much of this voxel is signal" rather than "is diffusion
/// justified here". A voxel's variance is the sum over spots of the
/// kernel's variance at each spot's distance (independent, since each draws
/// on a different part of the kernel), scaled the same way the fluence
/// itself is. Structured like sample_axisymmetric_volume's own loop, so it
/// costs one more pass of the same order, not a multiple.
fn noise_codes(
    grid: &Grid,
    p0: f64,
    pattern: &BeamPattern,
    run: &McRun,
    phi: &[f32],
) -> Vec<u8> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let dx = grid.lx / nx as f64;
    let dy = grid.ly / ny as f64;
    let share = p0 / pattern.len() as f64;

    let mut codes = vec![0u8; nx * ny * nz];
    let mut col = vec![0.0f64; nz];

    for ix in 0..nx {
        let x = (ix as f64 + 0.5) * dx - grid.lx / 2.0;
        for iy in 0..ny {
            let y = (iy as f64 + 0.5) * dy - grid.ly / 2.0;

            col.fill(0.0);
            for &(sx, sy) in pattern.spots() {
                let (lo, hi, f) = run.grid.radial_weights((x - sx).hypot(y - sy));
                let a = &run.se[lo * run.grid.n_z..lo * run.grid.n_z + nz];
                let b = &run.se[hi * run.grid.n_z..hi * run.grid.n_z + nz];
                for iz in 0..nz {
                    let sd = share * (a[iz] + (b[iz] - a[iz]) * f);
                    col[iz] += sd * sd;
                }
            }

            for iz in 0..nz {
                let idx = ix + iy * nx + iz * nx * ny;
                let value = phi[idx] as f64;
                let rel = if value > 0.0 { col[iz].sqrt() / value } else { f64::INFINITY };
                codes[idx] = if !rel.is_finite() || rel > REL_SE_POOR {
                    0
                } else if rel > REL_SE_GOOD {
                    1
                } else {
                    2
                };
            }
        }
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fpw1992;

    fn layer(mua: f64, mus: f64, g: f64, n: f64, thickness: f64) -> McLayerParams {
        McLayerParams { mua, mus, g, n, thickness }
    }

    /// A homogeneous slab with the app's default optical properties, on the
    /// default grid — 20 rings of 0.05 cm, so the volume it is drawn in is
    /// 40 x 40 x 40 voxels over 2 x 2 cm, matching the other models' own
    /// defaults — at whatever photon budget a test can afford.
    fn params(layers: Vec<McLayerParams>, photons_k: f64) -> MonteCarloParams {
        MonteCarloParams {
            layers,
            p0: 1.0,
            beam_profile: "pencil".into(),
            beam_width: 0.05,
            beam_pattern: "single".into(),
            pattern_count: 5,
            pattern_spacing: 0.2,
            photons_k,
            nr: 20,
            dr: 0.05,
            nz: 40,
        }
    }

    /// The voxel box `volume()` returns, worked out the same way
    /// compute_volume does — tests index into that box, not into the
    /// parameters.
    fn box_of(p: &MonteCarloParams) -> Grid {
        display_grid(p, p.layers.iter().map(|l| l.thickness).sum())
    }

    fn run(p: &MonteCarloParams) -> McRun {
        let stack = Stack::new(p);
        let profile = BeamProfile::from_params(&p.beam_profile, p.beam_width);
        let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
        run_mc(p, &stack, &profile, &pattern, SEED, TEST_WORKERS, |_| {})
    }

    fn volume(p: &MonteCarloParams) -> (Vec<f32>, Vec<f32>, Vec<u8>) {
        compute_volume_with(p, TEST_WORKERS, |_| {})
    }

    /* ── the interface physics, which is exact and so testable exactly ── */

    /// At normal incidence the s- and p-coefficients coincide and both
    /// reduce to (n1-n2)/(n1+n2), so the reflectance is its square — the
    /// figure quoted for specular reflection off tissue (2.8% at n = 1.4).
    #[test]
    fn fresnel_at_normal_incidence_matches_the_closed_form() {
        for n2 in [1.33, 1.4, 1.55] {
            let (r, ca2) = fresnel(1.0, n2, 1.0);
            let want = ((1.0 - n2) / (1.0 + n2)).powi(2);
            assert!((r - want).abs() < 1e-12, "n2 = {n2}: R = {r}, want {want}");
            assert!((ca2 - 1.0).abs() < 1e-12, "normal incidence stays normal");
        }
        assert!((fresnel(1.0, 1.4, 1.0).0 - 0.0278).abs() < 1e-3);
    }

    /// Past the critical angle nothing crosses. sin(theta_c) = n2/n1, so for
    /// tissue into air theta_c = asin(1/1.4) = 45.6 degrees.
    #[test]
    fn fresnel_is_total_beyond_the_critical_angle() {
        let n1 = 1.4;
        let crit = (1.0f64 / n1).asin();
        for deg_past in [0.5f64, 5.0, 20.0] {
            let theta = crit + deg_past.to_radians();
            let (r, ca2) = fresnel(n1, 1.0, theta.cos());
            assert_eq!(r, 1.0, "theta = {theta}: expected total internal reflection");
            assert_eq!(ca2, 0.0);
        }
        // And just inside it, some light still gets through.
        let (r, _) = fresnel(n1, 1.0, (crit - 0.05).cos());
        assert!(r < 1.0, "R = {r} just inside the critical angle should be < 1");
    }

    #[test]
    fn index_matched_interface_reflects_nothing() {
        for ca in [1.0, 0.7, 0.05] {
            let (r, ca2) = fresnel(1.4, 1.4, ca);
            assert_eq!(r, 0.0);
            assert_eq!(ca2, ca);
        }
    }

    /// The Henyey-Greenstein phase function is defined by its mean cosine
    /// being g, so sampling it and averaging is a direct check on the
    /// inversion formula. 400k samples put the standard error near 0.001.
    #[test]
    fn henyey_greenstein_mean_cosine_is_g() {
        for g in [0.0f64, 0.5, 0.9, -0.6] {
            let mut rng = Rng::seeded(7);
            let n = 400_000;
            let mut sum = 0.0;
            for _ in 0..n {
                sum += if g.abs() > 1e-6 {
                    let t: f64 = (1.0 - g * g) / (1.0 - g + 2.0 * g * rng.unit());
                    ((1.0 + g * g - t * t) / (2.0 * g)).clamp(-1.0, 1.0)
                } else {
                    2.0 * rng.unit() - 1.0
                };
            }
            let mean = sum / n as f64;
            assert!((mean - g).abs() < 0.005, "g = {g}: sampled <cos> = {mean}");
        }
    }

    /* ── global identities ── */

    /// Every launched unit of weight has to end up somewhere: reflected
    /// specularly, absorbed, escaped back out of the top, or transmitted
    /// through the floor. Exactly true up to the roulette, which preserves
    /// weight only in expectation — hence a tolerance rather than an
    /// equality, though a loose one is already a strong constraint on a
    /// transport solver.
    #[test]
    fn launched_weight_is_accounted_for() {
        for layers in [
            vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)],
            // Strongly absorbing, index-stepped, and thin: the cases the
            // diffusion models cannot describe at all.
            vec![layer(2.0, 20.0, 0.5, 1.5, 0.05), layer(0.05, 80.0, 0.95, 1.33, 0.3)],
        ] {
            let p = params(layers, 50.0);
            let r = run(&p);
            let total = r.stats.specular + r.stats.absorbed + r.stats.reflected + r.stats.transmitted;
            let rel = (total / r.photons as f64 - 1.0).abs();
            assert!(rel < 2e-3, "weight balance off by {:.2e}: {:?} of {} photons",
                rel,
                (r.stats.specular, r.stats.absorbed, r.stats.reflected, r.stats.transmitted),
                r.photons);
        }
    }

    /// Describing one homogeneous medium as two stacked identical layers is
    /// the same physics, and here it is the same *arithmetic*: carrying the
    /// unused step across the interface as remaining optical depth keeps the
    /// track identical, and an index-matched interface draws no random
    /// number, so the two runs agree bit for bit. Nothing weaker would rule
    /// out a step-length bias at layer boundaries, which is the classic way
    /// to get a layered Monte Carlo subtly wrong.
    #[test]
    fn splitting_a_layer_in_half_changes_nothing() {
        let one = run(&params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 20.0));
        let two = run(&params(
            vec![layer(0.1, 100.0, 0.9, 1.4, 0.7), layer(0.1, 100.0, 0.9, 1.4, 1.3)],
            20.0,
        ));
        assert_eq!(one.phi.len(), two.phi.len());
        for (k, (&a, &b)) in one.phi.iter().zip(two.phi.iter()).enumerate() {
            assert_eq!(a, b, "bin {k} differs: {a} vs {b}");
        }
    }

    /// How the work is split must not change the answer — the whole contract
    /// behind the parallelism, and the part that could break silently (a
    /// batch leaking state into its neighbour). Seven workers, not a divisor
    /// of N_BATCHES, so the uneven chunking gets exercised too. Not
    /// bit-identical (float addition isn't associative once the reduction
    /// groups terms differently), so "doesn't change the answer" means
    /// within a few ulp here.
    #[test]
    fn the_answer_does_not_depend_on_how_many_workers_run_it() {
        let p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 32.0);
        let stack = Stack::new(&p);
        let profile = BeamProfile::from_params(&p.beam_profile, p.beam_width);
        let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);

        let one = run_mc(&p, &stack, &profile, &pattern, SEED, 1, |_| {});
        let many = run_mc(&p, &stack, &profile, &pattern, SEED, 7, |_| {});

        for (k, (&a, &b)) in one.phi.iter().zip(many.phi.iter()).enumerate() {
            let tol = 1e-12 * a.abs().max(b.abs());
            assert!((a - b).abs() <= tol, "bin {k}: {a} with 1 worker, {b} with 7");
        }
        for (k, (&a, &b)) in one.se.iter().zip(many.se.iter()).enumerate() {
            let tol = 1e-9 * a.abs().max(b.abs());
            assert!((a - b).abs() <= tol, "error at bin {k}: {a} with 1 worker, {b} with 7");
        }
    }

    /* ── against an exactly known answer ── */

    /// The one test here that checks an *absolute* level against a closed
    /// form, rather than a ratio, a slope, or the model against itself. Take
    /// scattering away and the answer is Beer-Lambert:
    ///
    ///   int_V Phi dV = int exp(-mu_t z) dz over the bin's depth
    ///
    /// so radial bin 0's fluence is [exp(-mu_t z_lo) - exp(-mu_t z_hi)] /
    /// (mu_t * V). With n = 1 there's no specular loss or Fresnel either, so
    /// what's left is exactly the estimator, its normalization and its
    /// units — the parts every other test here could agree with itself
    /// about while still being wrong by a constant factor. Checked against
    /// the run's own reported error bars, which pins down the noise
    /// estimate too; the fixed seed keeps the margin from being flaky.
    #[test]
    fn a_non_scattering_slab_reproduces_beer_lambert() {
        let mua = 1.0;
        let mus = 1e-9; // not zero: mu_t == 0 would be a medium, not a slab
        let mut p = params(vec![layer(mua, mus, 0.0, 1.0, 2.0)], 200.0);
        p.nz = 40;
        let r = run(&p);

        let mu_t = mua + mus;
        let bin_volume = PI * r.grid.dr * r.grid.dr * r.grid.dz;
        let mut worst = 0.0f64;
        for iz in 0..r.grid.n_z {
            let z_lo = iz as f64 * r.grid.dz;
            let z_hi = z_lo + r.grid.dz;
            let exact = ((-mu_t * z_lo).exp() - (-mu_t * z_hi).exp()) / (mu_t * bin_volume);

            let got = r.phi[iz]; // radial bin 0
            let se = r.se[iz];
            assert!(se > 0.0, "no error reported for depth bin {iz}");
            let sigmas = (got - exact).abs() / se;
            worst = worst.max(sigmas);
            assert!(
                sigmas < 5.0,
                "depth bin {iz}: {got:.6e} vs Beer-Lambert's {exact:.6e}, \
                 off by {sigmas:.1} of the reported {se:.2e}"
            );
        }
        assert!(worst > 0.05, "suspiciously exact — is the comparison live? worst {worst:.3}");

        // Two more exactly known numbers from the same run: with no
        // scattering, the fraction absorbed is 1 - exp(-mu_a L) and the rest
        // leaves through the floor unscattered.
        let n = r.photons as f64;
        let absorbed = 1.0 - (-mua * 2.0f64).exp();
        assert!(
            (r.stats.absorbed / n - absorbed).abs() < 0.005,
            "absorbed {:.4} of the beam, Beer-Lambert says {absorbed:.4}",
            r.stats.absorbed / n
        );
        assert!(
            (r.stats.transmitted / n - (1.0 - absorbed)).abs() < 0.005,
            "transmitted {:.4}, expected {:.4}",
            r.stats.transmitted / n,
            1.0 - absorbed
        );
        assert!(r.stats.reflected / n < 1e-6, "nothing should come back out of the top");
    }

    /* ── against the models it exists to check ── */

    /// The headline claim of this model — that the diffusion models
    /// approximate *this* — only means something if the two agree where
    /// diffusion is supposed to be valid. So: a high-albedo homogeneous
    /// slab, compared on-axis against FPW1992 in absolute terms, several
    /// mean free paths below the surface and penetration depths above the
    /// floor. One correction has to be explicit: this model deducts
    /// specular loss at launch and FPW1992 doesn't, so FPW1992's fluence
    /// carries 2.8% more power — dividing that out is the whole systematic
    /// difference, and why this compares absolute levels rather than the
    /// decay rate the test below settles for.
    #[test]
    fn diffusive_regime_matches_fpw1992_in_absolute_terms() {
        let (mua, mus, g, n) = (0.1, 100.0, 0.9, 1.4);
        let mut p = params(vec![layer(mua, mus, g, n, 4.0)], 300.0);
        p.nz = 40; // dz = 0.1 cm
        let r = run(&p);
        let b = box_of(&p);

        let fp = fpw1992::Fpw1992Params {
            mua,
            mus,
            g,
            n,
            p0: 1.0,
            beam_profile: "pencil".into(),
            beam_width: 0.0,
            beam_pattern: "single".into(),
            pattern_count: 1,
            pattern_spacing: 0.0,
            lx: b.lx,
            ly: b.ly,
            lz: 4.0,
            nx: b.nx,
            ny: b.ny,
            nz: b.nz,
        };
        let fd = fpw1992::derived(&fp);
        let (diffusion, _) = fpw1992::compute_volume(&fp, &fd);

        let (specular, _) = fresnel(N_OUTSIDE, n, 1.0);
        let cx = b.nx / 2;
        let mfp_transport = 1.0 / (mua + mus * (1.0 - g));

        let mut compared = 0usize;
        for iz in 0..p.nz {
            let z = (iz as f64 + 0.5) * r.grid.dz;
            // Below the first few transport mean free paths (diffusion has
            // nothing to say above that) and well clear of the floor, which
            // this model has and FPW1992's semi-infinite medium doesn't.
            if z < 4.0 * mfp_transport || z > 1.4 {
                continue;
            }
            let mc = r.grid.lookup(&r.phi, 0.0, z) / (1.0 - specular);
            let dif = diffusion[cx + cx * b.nx + iz * b.nx * b.ny] as f64;
            let ratio = mc / dif;
            assert!(
                (0.92..1.08).contains(&ratio),
                "z = {z:.3} cm: Monte Carlo {mc:.4e} vs diffusion {dif:.4e}, ratio {ratio:.3}"
            );
            compared += 1;
        }
        assert!(compared >= 8, "only {compared} depths fell in the comparison band");

        // And the other half of the claim: within the first transport mean
        // free path diffusion is *wrong*, and wrong in the known direction.
        // It replaces the beam with an isotropic source a mean free path
        // down, which smears away the peak that the real, still-collimated
        // light makes right at the surface.
        let z_shallow = 0.5 * mfp_transport;
        let mc = r.grid.lookup(&r.phi, 0.0, z_shallow) / (1.0 - specular);
        let dif = diffusion[cx + cx * b.nx] as f64;
        assert!(
            mc > 2.0 * dif,
            "at {z_shallow:.3} cm transport should far exceed diffusion, got {mc:.3e} vs {dif:.3e}"
        );
    }

    /// Far from the source in a high-albedo medium, transport and diffusion
    /// have to agree — that is the regime diffusion is derived for. Compared
    /// as a decay rate rather than an absolute level, since the two
    /// normalize differently (the diffusion models ignore the specular loss
    /// this one deducts) and a rate is what the comparison is really about:
    /// on the axis, far from an effective point source, fluence falls as
    /// exp(-mu_eff*z)/z.
    #[test]
    fn deep_fluence_decays_at_the_diffusion_rate() {
        let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 4.0)], 300.0);
        p.nz = 40; // dz = 0.1 cm
        let r = run(&p);

        let musp: f64 = 100.0 * (1.0 - 0.9);
        let mueff: f64 = (3.0 * 0.1 * (0.1 + musp)).sqrt();
        let z0 = 1.0 / (0.1 + musp);

        // Least-squares slope of ln(z' * Phi) against z', with z' the
        // distance from the effective source depth — the depth range where
        // diffusion applies and the stack's floor is still far away.
        let pts: Vec<(f64, f64)> = (0..r.grid.n_z)
            .map(|iz| (iz as f64 + 0.5) * r.grid.dz)
            .filter(|&z| (0.6..=1.8).contains(&z))
            .map(|z| {
                let phi = r.grid.lookup(&r.phi, 0.0, z);
                (z - z0, ((z - z0) * phi).ln())
            })
            .collect();
        assert!(pts.len() > 8, "expected a usable depth range, got {}", pts.len());

        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
        let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        let den: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
        let slope = num / den;

        assert!(
            (-slope / mueff - 1.0).abs() < 0.2,
            "decay rate {:.3} cm^-1 vs diffusion's mu_eff = {:.3} cm^-1",
            -slope,
            mueff
        );
    }

    /* ── the noise estimate ── */

    /// Monte Carlo error falls as 1/sqrt(photons), so a nine-fold budget
    /// should cut the reported standard error by close to three. Checked
    /// well inside the bright region, where both budgets have enough
    /// samples for the estimate itself to be meaningful.
    #[test]
    fn reported_error_falls_with_the_square_root_of_the_budget() {
        let layers = || vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)];
        let lo = run(&params(layers(), 30.0));
        let hi = run(&params(layers(), 270.0));

        let rel = |r: &McRun| {
            let idx = 2 * r.grid.n_z + 5; // r ~ 2 bins out, z ~ 0.28 cm
            r.se[idx] / r.phi[idx]
        };
        let (a, b) = (rel(&lo), rel(&hi));
        assert!(a > 0.0 && b > 0.0, "no error reported: {a}, {b}");
        let ratio = a / b;
        assert!(
            (1.8..4.5).contains(&ratio),
            "error shrank by {ratio:.2}x for 9x the photons (expected ~3x): {a:.4} -> {b:.4}"
        );
    }

    /// The overlay is only worth showing if the error bars are honest, and
    /// the way to find out is to run the thing twice with different randoms
    /// and see whether the two answers differ by about what the error bars
    /// claim. Standardizing each bin's difference by the two reported errors
    /// should give a spread of about 1; an error bar understated by a factor
    /// of three would show up here as an RMS of three.
    #[test]
    fn reported_error_matches_the_scatter_between_independent_runs() {
        let p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 100.0);
        let stack = Stack::new(&p);
        let profile = BeamProfile::from_params(&p.beam_profile, p.beam_width);
        let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
        let a = run_mc(&p, &stack, &profile, &pattern, 0x1111_2222_3333_4444, TEST_WORKERS, |_| {});
        let b = run_mc(&p, &stack, &profile, &pattern, 0xAAAA_BBBB_CCCC_DDDD, TEST_WORKERS, |_| {});

        let mut n = 0usize;
        let mut sum_z2 = 0.0;
        for k in 0..a.phi.len() {
            // Only bins where both runs have enough samples for the error
            // estimate itself to be meaningful — the far outskirts of the
            // kernel are exactly where it isn't, which is the point of
            // flagging them.
            if a.se[k] <= 0.0 || b.se[k] <= 0.0 || a.se[k] / a.phi[k] > REL_SE_GOOD {
                continue;
            }
            let z = (a.phi[k] - b.phi[k]) / (a.se[k] * a.se[k] + b.se[k] * b.se[k]).sqrt();
            sum_z2 += z * z;
            n += 1;
        }
        assert!(n > 200, "only {n} bins were well enough sampled to check");
        let rms = (sum_z2 / n as f64).sqrt();
        assert!(
            (0.5..1.7).contains(&rms),
            "standardized difference RMS = {rms:.2} over {n} bins — the reported error is off by \
             about that factor"
        );
    }

    /* ── the volume the UI actually gets ── */

    #[test]
    fn volume_is_finite_positive_and_mostly_well_sampled() {
        let p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 100.0);
        let g = box_of(&p);
        let (phi, abs, codes) = volume(&p);
        let n = g.nx * g.ny * g.nz;
        assert_eq!((phi.len(), abs.len(), codes.len()), (n, n, n));
        assert!(phi.iter().all(|v| v.is_finite() && *v >= 0.0), "phi has a bad value");
        assert!(abs.iter().all(|v| v.is_finite() && *v >= 0.0), "abs has a bad value");
        assert!(codes.iter().all(|c| *c <= 2), "code out of range");

        // The absorbed-density channel is mu_a * Phi, as for every other
        // model here.
        let mid = g.nx / 2 + (g.ny / 2) * g.nx + 2 * g.nx * g.ny;
        assert!((abs[mid] / phi[mid] - 0.1).abs() < 1e-6, "abs/phi should be mu_a");

        // Near the source, this budget should be comfortably converged.
        let good = codes.iter().filter(|c| **c == 2).count();
        assert!(good > n / 20, "only {good} of {n} voxels reached the good-noise band");
    }

    /// Widening the beam spreads the same power over more area, so the peak
    /// drops — the same check fpw1992.rs and liemert_kienle.rs make of their
    /// convolutions, except here the profile is in the launch distribution
    /// rather than in a convolution.
    #[test]
    fn a_wide_beam_lowers_the_peak() {
        let peak = |profile: &str, width: f64| {
            let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 60.0);
            p.beam_profile = profile.into();
            p.beam_width = width;
            let (phi, _, _) = volume(&p);
            phi.iter().fold(0.0f32, |m, &v| m.max(v))
        };
        let pencil = peak("pencil", 0.0);
        let flat = peak("flattop", 0.3);
        let gauss = peak("gaussian", 0.15);
        assert!(flat < pencil, "flat-top peak {flat} should sit below pencil's {pencil}");
        assert!(gauss < pencil, "Gaussian peak {gauss} should sit below pencil's {pencil}");
    }

    /// A pattern is one kernel shifted and summed, so its symmetry is exact
    /// however noisy the kernel is — a cheap run is enough to check it.
    #[test]
    fn grid_pattern_stays_symmetric() {
        let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 16.0);
        p.beam_pattern = "grid".into();
        p.pattern_count = 3;
        p.pattern_spacing = 0.3;
        let g = box_of(&p);
        let (phi, _, _) = volume(&p);

        let at = |ix: usize, iy: usize, iz: usize| phi[ix + iy * g.nx + iz * g.nx * g.ny];
        for iz in [0, 5, 20] {
            for ix in 0..g.nx {
                for iy in 0..g.ny {
                    let mirrored = at(g.nx - 1 - ix, iy, iz);
                    assert_eq!(at(ix, iy, iz), mirrored, "x-mirror broken at ({ix}, {iy}, {iz})");
                    assert_eq!(at(ix, iy, iz), at(iy, ix, iz), "x/y swap broken at ({ix}, {iy}, {iz})");
                }
            }
        }
    }

    /// Progress has to reach the caller monotonically and finish at 1, since
    /// the UI drives a status line off it.
    #[test]
    fn progress_runs_from_zero_to_one() {
        let p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 16.0);
        let mut seen: Vec<f64> = Vec::new();
        compute_volume_with(&p, TEST_WORKERS, |f| seen.push(f));
        // How many intermediate reports land is a matter of timing — a run
        // this small may finish inside a single poll interval — so what is
        // guaranteed is only that they rise, stay in range, and end at 1.
        assert!(!seen.is_empty(), "no progress was reported at all");
        assert!(seen.iter().all(|f| *f > 0.0 && *f <= 1.0), "out of range: {seen:?}");
        assert!(seen.windows(2).all(|w| w[1] > w[0]), "progress went backwards: {seen:?}");
        assert_eq!(seen.last().copied(), Some(1.0));
    }

    /// The (N_r, dr) → voxel-box mapping, which src/models.ts has to
    /// reproduce exactly for the frontend to be able to size the IPC buffer
    /// (see display_grid).
    #[test]
    fn the_voxel_box_is_one_ring_per_voxel() {
        let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 1.0);
        p.nr = 13;
        p.dr = 0.04;
        let g = box_of(&p);
        assert_eq!((g.nx, g.ny), (26, 26), "the box should be 2*N_r across");
        assert!((g.lx / g.nx as f64 - p.dr).abs() < 1e-15, "a voxel should be exactly Δr wide");
        assert!((g.lx / 2.0 - 13.0 * 0.04).abs() < 1e-15, "the box should reach N_r*Δr from the axis");

        // And the tally covers the box's corners, which sit at R*sqrt(2) —
        // so it holds more rings than were asked for, at the same width.
        let single = BeamPattern::single();
        let tally = radial_grid(&p, &single, 2.0);
        assert!((tally.dr - p.dr).abs() < 1e-15, "the tally ring width is the parameter itself");
        assert!(
            tally.n_r as f64 >= 13.0 * 2f64.sqrt(),
            "{} rings don't reach the box's corners",
            tally.n_r
        );
    }

    /* ── validity reporting ── */

    /// A pattern of more than one spot is exact but no longer axisymmetric,
    /// which this model says out loud rather than leaving to the reader —
    /// see the warning's own text for what it actually costs.
    #[test]
    fn a_multi_spot_pattern_is_flagged_as_not_radially_symmetric() {
        let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 100.0);
        let fires = |p: &MonteCarloParams| {
            check_validity(p, &derived(p))
                .reasons
                .iter()
                .any(|r| r.contains("not radially symmetric"))
        };
        assert!(!fires(&p), "a single spot is radially symmetric");

        p.pattern_spacing = 0.2;
        p.pattern_count = 5;
        for kind in ["cross", "grid"] {
            p.beam_pattern = kind.into();
            assert!(fires(&p), "{kind} should be flagged");
        }
    }

    #[test]
    fn nonphysical_layer_input_is_reported_alone() {
        let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 100.0);
        p.layers[0].thickness = 0.0;
        let v = check_validity(&p, &derived(&p));
        assert!(!v.valid);
        assert_eq!(v.reasons.len(), 1, "expected only the hard error: {:?}", v.reasons);
        assert!(v.reasons[0].contains("thickness"));
    }

    /// The two failures that are genuinely this model's own — an
    /// under-funded run, and a field asked to reach far past where any
    /// photon goes.
    #[test]
    fn sampling_problems_are_flagged() {
        let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 0.5);
        let v = check_validity(&p, &derived(&p));
        assert!(v.reasons.iter().any(|r| r.contains("per bin")), "{:?}", v.reasons);

        p.photons_k = 500.0;
        p.beam_pattern = "grid".into();
        p.pattern_count = 16;
        p.pattern_spacing = 1.0;
        let v = check_validity(&p, &derived(&p));
        assert!(v.reasons.iter().any(|r| r.contains("penetration depths")), "{:?}", v.reasons);
    }

    /// A layer thinner than a transport mean free path, and absorption
    /// stronger than scattering, are hard errors for the diffusion models
    /// and perfectly ordinary here. Worth pinning down: it is the reason
    /// this model exists.
    #[test]
    fn geometry_the_diffusion_models_reject_is_accepted_here() {
        let mut p = params(
            vec![layer(3.0, 5.0, 0.7, 1.5, 0.01), layer(0.1, 100.0, 0.9, 1.4, 1.0)],
            100.0,
        );
        p.nz = 40;
        let v = check_validity(&p, &derived(&p));
        assert!(
            !v.reasons.iter().any(|r| r.contains("mean free path") && r.contains("thickness")),
            "a thin layer should not be an objection here: {:?}",
            v.reasons
        );
        let (phi, _, _) = volume(&p);
        assert!(phi.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(phi.iter().any(|v| *v > 0.0), "no light got in at all");
    }
}

/// Timing, kept apart from the correctness tests: budget checks rather than
/// physics assertions, worth having since the photon budget's right value is
/// purely a time/noise trade. Bounds are loose enough to survive a
/// contended machine — the harness runs tests concurrently, so that's all
/// an in-suite wall-clock assertion can honestly check; what they do catch
/// is an algorithmic blowup. The numbers quoted below come from dedicated
/// runs on an otherwise idle machine instead
/// (`cargo test --release <name> -- --nocapture --test-threads=1`).
#[cfg(test)]
mod perf_and_sanity {
    use super::*;

    fn defaults(photons_k: f64) -> MonteCarloParams {
        MonteCarloParams {
            layers: vec![
                McLayerParams { mua: 0.1, mus: 100.0, g: 0.9, n: 1.4, thickness: 0.3 },
                McLayerParams { mua: 0.1, mus: 50.0, g: 0.9, n: 1.4, thickness: 1.7 },
            ],
            p0: 1.0,
            beam_profile: "pencil".into(),
            beam_width: 0.05,
            beam_pattern: "single".into(),
            pattern_count: 5,
            pattern_spacing: 0.2,
            photons_k,
            nr: 20,
            dr: 0.05,
            nz: 40,
        }
    }

    /// The app's own default parameters and default budget, on the default
    /// grid — the run the user gets for clicking Compute without touching
    /// anything, which is the one that has to feel responsive.
    ///
    /// That default (100k photons, models.ts) was picked off this curve, for
    /// the two-layer tissue below on a 40^3 grid, on a 4-core/8-thread
    /// i5-10310U:
    ///
    ///     10k: 0.09 s, 23% of voxels under 5% error
    ///     25k: 0.21 s, 71%
    ///     50k: 0.37 s, 91%
    ///    100k: 0.69 s, 99%
    ///    200k: 1.29 s, 99.6%
    ///    400k: 2.45 s, 99.9%
    ///
    /// The knee is around 100k: essentially everything converged, still
    /// under a second. The slider spans four decades either side for anyone
    /// who wants a rough draft or a reference run.
    ///
    /// Per photon that is about 6 us, for some 500 collisions each. The
    /// same sweep against worker count, at 200k photons:
    ///
    ///     1 worker  26.3 us/photon   1.00x
    ///     2         13.8             1.90x
    ///     4          8.6             3.05x
    ///     8          6.2             4.21x
    ///
    /// 4.2x from 4 cores is about the ceiling on a 15 W laptop part: the
    /// work parallelizes essentially perfectly (nothing is shared, see
    /// run_mc), but all-core turbo is far below single-core turbo, so the
    /// last of it is paid back in clock. A desktop or workstation should
    /// land much closer to its thread count.
    #[test]
    fn default_run_stays_interactive() {
        let p = defaults(100.0);
        let t0 = std::time::Instant::now();
        let (phi, _, codes) = compute_volume_with(&p, TEST_WORKERS, |_| {});
        let dt = t0.elapsed();
        let photons = photon_budget(&p).0;
        println!(
            "{} photons, 40^3 grid, {} workers: {:?} ({:.1} us/photon); {:.0}% of voxels well sampled",
            photons,
            TEST_WORKERS,
            dt,
            dt.as_secs_f64() * 1e6 / photons as f64,
            100.0 * codes.iter().filter(|c| **c == 2).count() as f64 / codes.len() as f64,
        );
        assert!(phi.iter().any(|v| *v > 0.0));
        assert!(dt.as_secs_f64() < 30.0, "default run too slow: {:?}", dt);
    }

    /// A beam pattern costs one simulation however many spots it has — the
    /// whole point of scoring an axisymmetric kernel. Only the per-voxel
    /// superposition grows, and that is cheap next to tracing photons.
    #[test]
    fn many_spots_cost_almost_nothing_extra() {
        let single = defaults(60.0);
        let t0 = std::time::Instant::now();
        compute_volume_with(&single, TEST_WORKERS, |_| {});
        let dt_single = t0.elapsed();

        let mut grid = defaults(60.0);
        grid.beam_pattern = "grid".into();
        grid.pattern_count = 5; // 25 spots
        let t0 = std::time::Instant::now();
        compute_volume_with(&grid, TEST_WORKERS, |_| {});
        let dt_grid = t0.elapsed();

        println!("1 spot: {:?} | 25 spots: {:?}", dt_single, dt_grid);
        assert!(
            dt_grid.as_secs_f64() < 2.5 * dt_single.as_secs_f64(),
            "25 spots cost {:?} against one spot's {:?}",
            dt_grid,
            dt_single
        );
    }
}

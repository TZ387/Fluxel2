//! Monte Carlo photon transport through a layered slab — the reference model
//! here, and the only one that isn't an approximation.
//!
//! The other three models solve an approximate transport equation in closed
//! form (diffusion, or two-flux). This one solves the radiative transfer
//! equation itself, by tracing individual photon packets and letting the
//! statistics of many tracks stand in for the answer: no requirement that
//! scattering dominate absorption, no minimum layer thickness, no
//! extrapolated-boundary trick. What it costs instead is time and noise —
//! see the noise overlay below, which is this model's analogue of the other
//! models' validity overlay.
//!
//! # Provenance
//!
//! Written from the published algorithm, not ported from any existing code.
//! The scheme is MCML's (Wang, Jacques & Zheng, "MCML — Monte Carlo
//! modelling of light transport in multi-layered tissues", Comput. Methods
//! Programs Biomed. 47(2), 131–146, 1995): launch, hop by an exponentially
//! sampled step, split the step at a layer interface, deposit weight at each
//! collision, scatter by Henyey-Greenstein, and roulette the survivors.
//! Every formula below is standard and cited at its use site; the RNG is
//! xoshiro256++ with splitmix64 seeding (Blackman & Vigna, public domain),
//! also written from its published description. Nothing here derives from
//! MCX/µMCX or from any GPL-licensed Monte Carlo source, so this file
//! imposes no license obligation on the rest of the app.
//!
//! # Why an (r, z) grid rather than 3-D voxels
//!
//! Every geometry this model accepts is axisymmetric about the beam axis:
//! flat parallel layers, normal incidence, and a radially symmetric beam
//! profile. So the fluence a *single* spot produces depends only on (r, z),
//! and it is enough to score photons into an (r, z) grid — Phi(r, z) is
//! exactly the axisymmetric kernel beam::sample_axisymmetric_volume already
//! consumes for the two diffusion models. Consequences:
//!
//! - Every photon contributes to the same 2-D table rather than being spread
//!   over a 3-D one, so a run needs orders of magnitude fewer photons than a
//!   voxel-based Monte Carlo would for the same noise.
//! - A beam *pattern* (a scanner's line, a fractional handpiece's grid) is
//!   free: one run's kernel is shifted and superposed per spot, exactly as
//!   for Liemert-Kienle. Transport is linear, so that is exact — the same
//!   argument beam.rs's doc comment makes.
//! - The beam *profile* is free too, and better than a convolution: instead
//!   of convolving a pencil-beam kernel afterwards, each photon's launch
//!   point is sampled from the profile itself (launch_radius below), which
//!   is exact for any profile and costs nothing.
//!
//! The assumption to keep in mind is the one that buys all of this: tilt the
//! incidence, warp an interface, or embed an inclusion, and the symmetry is
//! gone and a full 3-D grid becomes necessary. Nothing outside this file
//! depends on the choice.
//!
//! # Noise, and the overlay
//!
//! A Monte Carlo answer is an estimate with an error bar, so this model
//! reports the error bar too. The photon budget is split into equal batches
//! (N_BATCHES) and each batch tallied separately; the spread between batch
//! totals is an unbiased estimate of the variance of their sum (the standard
//! "batch means" estimator). That gives a per-bin standard error alongside
//! the fluence, which becomes the per-voxel overlay: green where the
//! relative standard error is under REL_SE_GOOD, amber under REL_SE_POOR,
//! red beyond it. Batching is also exactly the decomposition a parallel run
//! wants — independent batches, private tallies, one reduction at the end —
//! so it is doing double duty.
//!
//! Worth knowing what that overlay actually shows, because it is not what
//! intuition suggests: the noisiest bins are the ones *on the beam axis*.
//! A radial bin's volume grows with its index (it is an annulus), so the
//! innermost bins are by far the smallest and collect the fewest collisions
//! even though the fluence there is the highest — the estimator's error
//! goes with the collision count, not with the signal. The far outskirts
//! are the other weak spot, for the opposite reason. Everything between
//! them converges first.
//!
//! # Units and normalization
//!
//! A photon packet is launched with weight 1 and the specular reflection at
//! the surface is deducted immediately, so the tallies are fractions of the
//! *incident* power P0. The estimator scored at each collision is w/mu_t,
//! the collision estimator's unbiased stand-in for the packet's path length
//! in the bin; dividing the total by (photons * bin volume) turns it into
//! fluence per unit incident power [1/cm^2], which is what
//! sample_axisymmetric_volume expects and multiplies by P0.
//!
//! Note that the diffusion models here launch their source with full weight
//! and never account for specular reflection, so their absolute fluence runs
//! a few percent high next to this model's (2.8% for n = 1.4). The
//! difference is real physics, not a discrepancy.

use crate::physics::beam::{self, BeamPattern, BeamProfile, Grid};
use crate::physics::validity::require;
use serde::{Deserialize, Serialize};

use std::f64::consts::PI;

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

/// Batches the photon budget is split into for the batch-means standard
/// error. Enough that the variance of the batch total is itself estimated
/// with reasonable confidence (~25% relative), few enough that the per-batch
/// tally is cleared rarely.
const N_BATCHES: usize = 16;

/// Relative standard error boundaries for the noise overlay's three codes.
const REL_SE_GOOD: f64 = 0.05;
const REL_SE_POOR: f64 = 0.20;

/// Ceiling on the (r, z) tally's bin count, purely so a run can never ask
/// for an unbounded allocation. Two things can push the radial reach up and
/// the bin width down at once — a wide beam pattern and a small, finely
/// divided grid — and nothing stops a value typed into the parameter panel
/// (which re-ranges its slider rather than clamping) from combining them
/// absurdly, or from being zero. Hitting this coarsens the radial grid,
/// which only affects the far field; check_validity complains about a reach
/// like that long before it gets here.
const MAX_TALLY_BINS: usize = 2_000_000;

/// Cosine above which a direction counts as exactly along z, where the
/// azimuthal rotation in `scatter` degenerates (0/0).
const COS_ZERO: f64 = 1.0 - 1e-12;

/// Seed the per-batch generators are derived from. Fixed, so that identical
/// parameters give an identical volume — a plot should never change for a
/// reason the user didn't cause. Tests reseed to get genuinely independent
/// runs (see the estimator's own validation).
const SEED: u64 = 0x5EED_0000_0000_0000;

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
    /// "single" | "line" | "grid" — see beam.rs.
    pub beam_pattern: String,
    pub pattern_count: usize,
    pub pattern_spacing: f64,
    /// Photon budget in thousands. In thousands because the useful range
    /// spans four decades, which no linear slider handles gracefully in
    /// units of one photon.
    pub photons_k: f64,
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
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
    /// How many spots the chosen beam pattern works out to.
    pub spots: usize,
    /// Photons the run will actually trace (the budget rounded to a whole
    /// number of equal batches).
    pub photons: u64,
    /// Fraction of P0 reflected specularly off the top surface, before any
    /// scattering — lost to the tissue no matter how the rest is traced.
    pub specular: f64,
    /// Radial bins in the (r, z) kernel grid, for a sense of what the photon
    /// budget is being spread over.
    pub n_r: usize,
}

#[derive(Serialize)]
pub struct ValidityResult {
    pub valid: bool,
    pub reasons: Vec<String>,
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
        TallyGrid { n_r, n_z, dr, dz, r_c }
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
        let ir = ((x * x + y * y).sqrt() / self.dr) as usize;
        if ir >= self.n_r {
            return;
        }
        let iz = ((z / self.dz) as usize).min(self.n_z - 1);
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
        if hi >= self.n_r {
            return (self.n_r - 1, self.n_r - 1, 0.0);
        }
        let f = (rho - self.r_c[hi - 1]) / (self.r_c[hi] - self.r_c[hi - 1]);
        (hi - 1, hi, f)
    }

    /// Index of the depth bin containing `z`. No interpolation in z: the
    /// grid's depth bins are the voxel grid's own, so every value asked for
    /// sits at a bin centre (see radial_grid / compute_volume).
    fn depth_bin(&self, z: f64) -> usize {
        ((z / self.dz) as usize).min(self.n_z - 1)
    }

    /// Value of a table on this grid at an arbitrary (rho, z).
    fn lookup(&self, table: &[f64], rho: f64, z: f64) -> f64 {
        let iz = self.depth_bin(z);
        let (lo, hi, f) = self.radial_weights(rho);
        let (a, b) = (table[lo * self.n_z + iz], table[hi * self.n_z + iz]);
        a + (b - a) * f
    }
}

/// The (r, z) grid a run tallies onto, chosen to line up with what
/// beam::sample_axisymmetric_volume will ask for: `dr` matches the radial
/// step that function samples the kernel at, and the reach covers the whole
/// beam pattern, since a voxel in one corner can be far from a spot in the
/// other. Depth bins are the voxel grid's own, so no depth interpolation is
/// ever needed.
fn radial_grid(p: &MonteCarloParams, pattern: &BeamPattern, lz: f64) -> TallyGrid {
    let reach = beam::max_kernel_radius(p.lx, p.ly, pattern) * 1.0001;
    let dr = beam::max_kernel_radius(p.lx, p.ly, &BeamPattern::single()) * 1.0001
        / (p.nx.max(p.ny).max(2) - 1) as f64;
    let n_z = p.nz.max(1);
    let wanted = if dr > 0.0 { (reach / dr).ceil() as usize + 1 } else { 1 };
    let n_r = wanted.clamp(1, (MAX_TALLY_BINS / n_z).max(1));
    TallyGrid::new(n_r, n_z, dr.max(f64::MIN_POSITIVE), lz / n_z as f64)
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

/// Whole photon budget, rounded down to a whole number of equal batches (and
/// at least one photon per batch, so the batch-means estimate always has
/// N_BATCHES samples to work with).
fn photon_budget(p: &MonteCarloParams) -> (u64, u64) {
    let requested = (p.photons_k.max(0.0) * 1000.0).round().max(0.0) as u64;
    let per_batch = (requested / N_BATCHES as u64).max(1);
    (per_batch * N_BATCHES as u64, per_batch)
}

/// Trace the whole budget and reduce the batches into fluence + standard
/// error. `progress` is called with the fraction done after each batch,
/// giving the UI something to show during a run that can take seconds.
fn run_mc(
    p: &MonteCarloParams,
    stack: &Stack,
    beam: &BeamProfile,
    pattern: &BeamPattern,
    seed: u64,
    mut progress: impl FnMut(f64),
) -> McRun {
    let grid = radial_grid(p, pattern, stack.lz);
    let (photons, per_batch) = photon_budget(p);
    let (specular, _) = fresnel(N_OUTSIDE, stack.layers[0].n, 1.0);

    let mut stats = RunStats::default();
    // Running first and second moments of the per-batch tallies. Each batch
    // is an independent estimate of the same quantity, so their spread is
    // the variance of the total (see this module's doc comment).
    let mut sum = vec![0.0f64; grid.len()];
    let mut sum_sq = vec![0.0f64; grid.len()];
    let mut batch = vec![0.0f64; grid.len()];

    for b in 0..N_BATCHES {
        batch.iter_mut().for_each(|v| *v = 0.0);
        // Each batch gets its own generator, seeded from the batch index
        // alone: reproducible, and independent of how many batches ran
        // before it — which is what a parallel run needs.
        let mut rng = Rng::seeded(seed ^ b as u64);
        for _ in 0..per_batch {
            trace_photon(&mut rng, stack, beam, &grid, &mut batch, specular, &mut stats);
        }
        for (k, &v) in batch.iter().enumerate() {
            sum[k] += v;
            sum_sq[k] += v * v;
        }
        progress((b + 1) as f64 / N_BATCHES as f64);
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
    let min_extent = p.lx.min(p.ly);
    require(&mut reasons, min_extent > 0.0, "the smaller of L<sub>x</sub>, L<sub>y</sub>", "greater than 0", min_extent);
    require(&mut reasons, p.photons_k > 0.0, "photon budget", "greater than 0", p.photons_k);
    if !reasons.is_empty() {
        return ValidityResult { valid: false, reasons };
    }

    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    if let Some(reason) = beam::pattern_extent_warning(&pattern, p.lx, p.ly) {
        reasons.push(reason);
    }

    // Every photon's track is spread over the whole (r, z) grid, so there is
    // no clean photons-per-bin figure — but a budget below a few photons per
    // bin cannot produce a usable kernel however the tracks fall.
    let bins = derived.n_r * p.nz;
    let per_bin = derived.photons as f64 / bins as f64;
    if per_bin < 20.0 {
        reasons.push(format!(
            "{} photons over an {} x {} (r, z) kernel grid is only {:.1} per bin — expect a visibly \
             noisy volume. Raise the photon budget, or coarsen N<sub>x</sub>/N<sub>y</sub>/N<sub>z</sub>. \
             Switch on the noise overlay to see which voxels are actually affected",
            derived.photons, derived.n_r, p.nz, per_bin
        ));
    }

    // How far the kernel has to reach, in penetration depths of the first
    // layer. Fluence falls off roughly exponentially at that rate, and the
    // photon count with it, so a kernel asked to span many decades of decay
    // is asking for its far field to be sampled by almost nothing.
    let l1 = &p.layers[0];
    let mueff = (3.0 * l1.mua * (l1.mua + l1.mus * (1.0 - l1.g))).sqrt();
    let reach = beam::max_kernel_radius(p.lx, p.ly, &pattern);
    if reach * mueff > 15.0 {
        reasons.push(format!(
            "the kernel has to reach {:.2} cm, about {:.0} penetration depths (1/μ<sub>eff</sub> = \
             {:.3} cm) — fluence out there is some 10<sup>{:.0}</sup> times below the peak, and \
             hardly any photon gets that far, so the volume's outskirts will be noise. Shrink \
             L<sub>x</sub>/L<sub>y</sub> or the spot spacing, or accept that only the bright \
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
    let grid = radial_grid(p, &pattern, derived.lz);
    let mfp_transport = 1.0 / (l1.mua + l1.mus * (1.0 - l1.g));
    if BeamProfile::from_params(&p.beam_profile, p.beam_width).is_pencil() && grid.dr > mfp_transport {
        reasons.push(format!(
            "the radial bin width ({:.3} cm) is wider than layer 1's transport mean free path \
             ({:.3} cm), so the on-axis peak of an idealised pencil beam is averaged away over \
             the innermost bin. Raise N<sub>x</sub>/N<sub>y</sub>, shrink \
             L<sub>x</sub>/L<sub>y</sub>, or use a beam profile with a real width",
            grid.dr, mfp_transport
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
    let beam_profile = BeamProfile::from_params(&p.beam_profile, p.beam_width);
    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    let stack = Stack::new(p);
    let run = run_mc(p, &stack, &beam_profile, &pattern, SEED, progress);

    let grid = Grid {
        lx: p.lx,
        ly: p.ly,
        nx: p.nx,
        ny: p.ny,
        nz: p.nz,
        dz: stack.lz / p.nz as f64,
    };
    let (phi, abs) = beam::sample_axisymmetric_volume(
        &grid,
        &pattern,
        p.p0,
        |z| stack.layers[stack.layer_at(z)].mua,
        |rho, z| run.grid.lookup(&run.phi, rho, z),
    );
    let codes = noise_codes(p, &pattern, &run, &phi);
    (phi, abs, codes)
}

/// Per-voxel noise code for the overlay (0 poor, 1 marginal, 2 good) — this
/// model's counterpart to the other two point-source models'
/// compute_validity_volume, answering "how much of this voxel is signal"
/// rather than "is diffusion justified here".
///
/// A voxel's value is the sum over spots of the kernel at each spot's
/// distance, so its variance is the sum of theirs (each spot draws on a
/// different part of the kernel, so treating the terms as independent is
/// reasonable), each scaled by the same per-spot share of P0 the fluence
/// itself was. Structured like sample_axisymmetric_volume's own loop —
/// radial index per (x, y) column, contiguous accumulate over z — so it
/// costs one more pass of the same order, not a multiple.
fn noise_codes(
    p: &MonteCarloParams,
    pattern: &BeamPattern,
    run: &McRun,
    phi: &[f32],
) -> Vec<u8> {
    let (nx, ny, nz) = (p.nx, p.ny, p.nz);
    let dx = p.lx / nx as f64;
    let dy = p.ly / ny as f64;
    let share = p.p0 / pattern.len() as f64;

    let mut codes = vec![0u8; nx * ny * nz];
    let mut col = vec![0.0f64; nz];

    for ix in 0..nx {
        let x = (ix as f64 + 0.5) * dx - p.lx / 2.0;
        for iy in 0..ny {
            let y = (iy as f64 + 0.5) * dy - p.ly / 2.0;

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

    fn layer(mua: f64, mus: f64, g: f64, n: f64, thickness: f64) -> McLayerParams {
        McLayerParams { mua, mus, g, n, thickness }
    }

    /// A homogeneous slab with the app's default optical properties, on the
    /// default 40^3 grid, at whatever photon budget a test can afford.
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
            lx: 2.0,
            ly: 2.0,
            nx: 40,
            ny: 40,
            nz: 40,
        }
    }

    fn run(p: &MonteCarloParams) -> McRun {
        let stack = Stack::new(p);
        let profile = BeamProfile::from_params(&p.beam_profile, p.beam_width);
        let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
        run_mc(p, &stack, &profile, &pattern, SEED, |_| {})
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

    /* ── against the models it exists to check ── */

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
        let a = run_mc(&p, &stack, &profile, &pattern, 0x1111_2222_3333_4444, |_| {});
        let b = run_mc(&p, &stack, &profile, &pattern, 0xAAAA_BBBB_CCCC_DDDD, |_| {});

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
        let p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 200.0);
        let (phi, abs, codes) = compute_volume(&p, |_| {});
        let n = p.nx * p.ny * p.nz;
        assert_eq!((phi.len(), abs.len(), codes.len()), (n, n, n));
        assert!(phi.iter().all(|v| v.is_finite() && *v >= 0.0), "phi has a bad value");
        assert!(abs.iter().all(|v| v.is_finite() && *v >= 0.0), "abs has a bad value");
        assert!(codes.iter().all(|c| *c <= 2), "code out of range");

        // The absorbed-density channel is mu_a * Phi, as for every other
        // model here.
        let mid = p.nx / 2 + (p.ny / 2) * p.nx + 2 * p.nx * p.ny;
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
            let mut p = params(vec![layer(0.1, 100.0, 0.9, 1.4, 2.0)], 100.0);
            p.beam_profile = profile.into();
            p.beam_width = width;
            let (phi, _, _) = compute_volume(&p, |_| {});
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
        let (phi, _, _) = compute_volume(&p, |_| {});

        let at = |ix: usize, iy: usize, iz: usize| phi[ix + iy * p.nx + iz * p.nx * p.ny];
        for iz in [0, 5, 20] {
            for ix in 0..p.nx {
                for iy in 0..p.ny {
                    let mirrored = at(p.nx - 1 - ix, iy, iz);
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
        compute_volume(&p, |f| seen.push(f));
        assert_eq!(seen.len(), N_BATCHES);
        assert!(seen.windows(2).all(|w| w[1] > w[0]), "progress went backwards: {seen:?}");
        assert_eq!(seen.last().copied(), Some(1.0));
    }

    /* ── validity reporting ── */

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
        let (phi, _, _) = compute_volume(&p, |_| {});
        assert!(phi.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(phi.iter().any(|v| *v > 0.0), "no light got in at all");
    }
}

/// Timing, kept apart from the correctness tests because these are budget
/// checks rather than assertions about the physics. The photon budget is the
/// one parameter here whose right value is purely a time/noise trade, so it
/// is worth pinning down what a run actually costs — and worth noticing if a
/// change to the inner loop makes it cost noticeably more.
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
            lx: 2.0,
            ly: 2.0,
            nx: 40,
            ny: 40,
            nz: 40,
        }
    }

    /// The app's own default parameters and default budget, on the default
    /// grid — the run the user gets for clicking Compute without touching
    /// anything, which is the one that has to feel responsive.
    ///
    /// That default (50k photons, models.ts) was picked off this curve, for
    /// the two-layer tissue below on a 40^3 grid:
    ///
    ///     10k: 0.25 s, 28% of voxels under 5% error
    ///     25k: 0.59 s, 69%
    ///     50k: 1.13 s, 92%
    ///    100k: 2.36 s, 98%
    ///    200k: 4.56 s, 99.6%
    ///
    /// — about 22 us per photon, which for this tissue is some 500
    /// collisions each. Diminishing returns set in right where the wait
    /// starts to be noticeable, which is a convenient place for a default to
    /// sit; the slider goes four decades either way for anyone who wants a
    /// draft or a reference run.
    #[test]
    fn default_run_stays_interactive() {
        let p = defaults(50.0);
        let t0 = std::time::Instant::now();
        let (phi, _, codes) = compute_volume(&p, |_| {});
        let dt = t0.elapsed();
        let photons = photon_budget(&p).0;
        println!(
            "{} photons, 40^3 grid: {:?} ({:.1} us/photon); {:.0}% of voxels well sampled",
            photons,
            dt,
            dt.as_secs_f64() * 1e6 / photons as f64,
            100.0 * codes.iter().filter(|c| **c == 2).count() as f64 / codes.len() as f64,
        );
        assert!(phi.iter().any(|v| *v > 0.0));
        assert!(dt.as_secs_f64() < 10.0, "default run too slow: {:?}", dt);
    }

    /// A beam pattern costs one simulation however many spots it has — the
    /// whole point of scoring an axisymmetric kernel. Only the per-voxel
    /// superposition grows, and that is cheap next to tracing photons.
    #[test]
    fn many_spots_cost_almost_nothing_extra() {
        let single = defaults(100.0);
        let t0 = std::time::Instant::now();
        compute_volume(&single, |_| {});
        let dt_single = t0.elapsed();

        let mut grid = defaults(100.0);
        grid.beam_pattern = "grid".into();
        grid.pattern_count = 5; // 25 spots
        let t0 = std::time::Instant::now();
        compute_volume(&grid, |_| {});
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

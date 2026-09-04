//! N-layer, point-source diffusion approximation.
//! Reference: André Liemert & Alwin Kienle, "Light diffusion in a turbid
//! cylinder. II. Layered case," Opt. Express 18, 9266-9279 (2010).
//!
//! FPW1992 gives a point source but only one layer; Kubelka-Munk gives many
//! layers but only diffuse illumination. This fills the gap: a pencil beam
//! through a stack of homogeneous layers, still within the diffusion
//! approximation — no Monte Carlo involved.
//!
//! Solved for a *finite cylinder* of radius `a` via a Fourier-Bessel series
//! (zeros of J0), since layering breaks the translational symmetry FPW1992's
//! closed form relies on. The cylinder's axis runs through the beam at the
//! center of the grid's top face; `a` is picked generously larger than the
//! visible grid so its wall stays invisible in the display, but not so large
//! that convergence needs many more terms (see `cylinder_radius`).
//!
//! # How it's solved
//!
//! Each series term is one transverse spatial frequency s = j_{0,n}/a',
//! leaving a 1-D boundary-value problem in z:
//!
//!   D_k (G'' - α_k² G) = -δ(z - z0),   α_k = sqrt(μ_a,k/D_k + s²)
//!
//! with the source in the top layer only, G = 0 on the extrapolated planes
//! above and below the stack, and G/n² and D·G' both continuous at every
//! interface (the index-mismatched matching condition). Substituting
//! g = G/n² and n_eff = D n² makes that a layered transmission line —
//! n_eff (g'' - α² g) = -δ, with g and n_eff·g' continuous — of
//! characteristic admittance Yc_k = n_eff_k · α_k. That's what makes any N
//! tractable: everything below a given depth reaches the layers above it
//! through one number only.
//!
//! So, bottom-up, as the reflection coefficient ρ_k at layer k's floor
//! (q_k = exp(-2 α_k t_k)):
//!
//!   ρ_{N-1} = -exp(-2 α_{N-1} zb)   (the extrapolated plane below the stack
//!                                    is a perfect -1 mirror, zb below it)
//!   Y_k^top = Yc_k (1 - ρ_k q_k)/(1 + ρ_k q_k)
//!   ρ_{k-1} = (Yc_{k-1} - Y_k^top)/(Yc_{k-1} + Y_k^top)
//!
//! |ρ| < 1 strictly and every exponent is ≤ 0, so no step can overflow. The
//! top layer is then source + image plus what bounces between the top plane
//! and the interface below it; layers under it carry no source, so each is
//! fixed by the fluence arriving at its top face:
//!
//!   E = exp(-2 α (t₀ + zb)),  K = ρ₀/(1 + ρ₀ E)   (sum over round trips)
//!   G₀(z) = [ e^(-α|z-z₀|) - e^(-α(z+z₀+2zb))
//!             + 4 K E sinh(α(z₀+zb)) sinh(α(z+zb)) ] / (2 D α)
//!   ψ_k(z) = (e^(-u) + ρ_k e^(u - 2 α_k t_k))/(1 + ρ_k q_k), u = α_k(z - z_k)
//!   G_k(z) = n_k² g_k^top ψ_k(z),   g_(k+1)^top = g_k^top ψ_k(floor)
//!
//! from g₁^top = G₀(t₀)/n₀². ψ_k(top face) = 1, so that chain is just
//! fluence/n² carried down one face at a time.
//!
//! # Provenance
//!
//! N=2 was faithfully ported from the paper's own reference implementation
//! (github.com/heltonmc/LightPropagation.jl, `_green_Nlaycylin_top`/`_bottom`
//! specialized to N=2) — exactly the kind of Bessel-series math where a
//! transcription error silently yields a plausible wrong number. That
//! reference has no middle-layer Green's function, so the recursion above is
//! derived here instead; it is algebraically identical to the ported form at
//! N=2 (whose β/γ is coth(α₂(t₂+zb₂)), one step of the recursion), which the
//! tests check term by term alongside an independent finite-volume solve.
//! One deliberate deviation from the reference: the boundary reflection fit
//! (A, zb below) is reused from fpw1992.rs, so both point-source models treat
//! the air-tissue boundary identically.
//!
//! The stack is bounded by air at *both* ends — the last layer's floor is a
//! zero-fluence boundary too, so make it several penetration depths thick for
//! an effectively semi-infinite substrate.

use crate::physics::beam::{self, BeamPattern, BeamProfile, Grid};
use crate::physics::bessel::{j0, j0_zero, j1};
use serde::{Deserialize, Serialize};

/// Every layer carries its own thickness, so the grid's depth is the stack's
/// total depth — the same arrangement as kubelka_munk.rs, the other layered
/// model here.
#[derive(Deserialize, Clone, Copy)]
pub struct LKLayerParams {
    pub mua: f64,
    pub mus: f64,
    pub g: f64,
    pub n: f64,
    pub thickness: f64,
}

#[derive(Deserialize)]
pub struct LiemertKienleParams {
    /// Top → bottom; the beam enters the top face of layers[0].
    pub layers: Vec<LKLayerParams>,
    pub p0: f64,
    /// "pencil" | "gaussian" | "flattop" — see beam.rs. `beam_width` is
    /// sigma (Gaussian) or radius (flattop) in cm, ignored for "pencil".
    pub beam_profile: String,
    pub beam_width: f64,
    /// "single" | "line" | "grid" — see beam.rs. `pattern_count` is spots
    /// along the line or per side of the grid, `pattern_spacing` the pitch
    /// between neighbours in cm; both ignored for "single".
    pub beam_pattern: String,
    pub pattern_count: usize,
    pub pattern_spacing: f64,
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

#[derive(Serialize, Clone, Copy)]
pub struct LayerDerived {
    pub musp: f64,
    #[serde(rename = "D")]
    pub d: f64,
    pub mueff: f64,
}

#[derive(Serialize)]
pub struct LiemertKienleDerived {
    pub layers: Vec<LayerDerived>,
    pub z0: f64,
    /// Total stack depth, which is also the grid's depth here.
    #[serde(rename = "Lz")]
    pub lz: f64,
    /// How many spots the chosen beam pattern works out to.
    pub spots: usize,
}

#[derive(Serialize)]
pub struct ValidityResult {
    pub valid: bool,
    pub reasons: Vec<String>,
}

/// Per-layer coefficients derived from one layer's parameters alone — shared
/// by derived(), check_validity(), and compute_volume() so it's written once.
#[derive(Clone, Copy)]
struct LayerCoeffs {
    mua: f64,
    musp: f64,
    d: f64,
    /// Extrapolation length (2*A*D) at a boundary with air — only the top and
    /// bottom layers' values are used, but it's free to carry for all.
    zb: f64,
    /// D * n^2 — the weighting that makes fluence/n^2 and D*d(fluence)/dz the
    /// continuous pair across an index-mismatched interface.
    n_eff: f64,
    n_sq: f64,
    thickness: f64,
    /// Depth of this layer's top face below the surface.
    z_top: f64,
}

/// Same reflection-coefficient fit as fpw1992.rs (Groenhuis et al.), reused
/// here so both point-source models treat the air-tissue boundary the same
/// way — see this module's doc comment.
fn extrapolation_length(n: f64, d: f64) -> f64 {
    let reff = -1.44 / (n * n) + 0.71 / n + 0.668 + 0.0636 * n;
    let a = (1.0 + reff) / (1.0 - reff);
    2.0 * a * d
}

fn layer_coeffs(l: &LKLayerParams, z_top: f64) -> LayerCoeffs {
    let musp = l.mus * (1.0 - l.g);
    let d = 1.0 / (3.0 * musp);
    LayerCoeffs {
        mua: l.mua,
        musp,
        d,
        zb: extrapolation_length(l.n, d),
        n_eff: d * l.n * l.n,
        n_sq: l.n * l.n,
        thickness: l.thickness,
        z_top,
    }
}

/// The whole stack, plus the two things fixed for one compute_volume() call:
/// where the effective source sits, and the cylinder the series is solved in.
struct Stack {
    layers: Vec<LayerCoeffs>,
    /// Effective source depth, one transport mean free path into layer 1.
    z0: f64,
    /// Total stack depth.
    lz: f64,
    /// Cylinder radius extended by the top layer's extrapolation length.
    a_prime: f64,
}

impl Stack {
    fn new(p: &LiemertKienleParams, pattern: &BeamPattern) -> Stack {
        let mut layers = Vec::with_capacity(p.layers.len());
        let mut z_top = 0.0;
        for l in &p.layers {
            layers.push(layer_coeffs(l, z_top));
            z_top += l.thickness;
        }
        Stack {
            z0: 1.0 / layers[0].musp,
            lz: z_top,
            a_prime: cylinder_radius(p.lx, p.ly, pattern) + layers[0].zb,
            layers,
        }
    }

    /// Index of the layer containing depth `z` (the last layer catches
    /// anything at or past the stack's floor).
    fn layer_at(&self, z: f64) -> usize {
        for (i, l) in self.layers.iter().enumerate() {
            if z <= l.z_top + l.thickness {
                return i;
            }
        }
        self.layers.len() - 1
    }
}

pub fn derived(p: &LiemertKienleParams) -> LiemertKienleDerived {
    let layers: Vec<LayerDerived> = p
        .layers
        .iter()
        .map(|l| {
            let c = layer_coeffs(l, 0.0); // nothing here depends on z_top
            LayerDerived {
                musp: c.musp,
                d: c.d,
                mueff: (c.mua / c.d).sqrt(),
            }
        })
        .collect();
    LiemertKienleDerived {
        z0: 1.0 / layers[0].musp,
        lz: p.layers.iter().map(|l| l.thickness).sum(),
        spots: BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing).len(),
        layers,
    }
}

pub fn check_validity(p: &LiemertKienleParams, derived: &LiemertKienleDerived) -> ValidityResult {
    let mut reasons = Vec::new();

    for (i, (l, d)) in p.layers.iter().zip(derived.layers.iter()).enumerate() {
        let ratio = d.musp / l.mua;
        if ratio < 10.0 {
            reasons.push(format!(
                "layer {}: μ<sub>s</sub>'/μ<sub>a</sub> = {:.2} (want ≳10) — absorption is too strong \
                 relative to scattering for light to randomize direction before being absorbed",
                i + 1,
                ratio
            ));
        }
    }

    let t1 = p.layers[0].thickness;
    if derived.z0 >= t1 {
        reasons.push(format!(
            "the source depth z<sub>0</sub> = {:.3} cm falls outside layer 1 (thickness {:.3} cm) \
             — this model assumes the beam's effective source sits within the first layer; \
             increase layer 1's thickness or scattering coefficient",
            derived.z0, t1
        ));
    }

    let dx = p.lx / p.nx as f64;
    let dy = p.ly / p.ny as f64;
    let dz = derived.lz / p.nz as f64;
    let max_voxel = dx.max(dy).max(dz);
    if max_voxel > 0.5 * derived.z0 {
        reasons.push(format!(
            "voxel size (up to {:.3} cm) is ≳half the source depth z<sub>0</sub> \
             ({:.3} cm) where fluence peaks and varies fastest — the grid is too coarse \
             to resolve that peak, so results near the source will be smeared out. \
             Increase N<sub>x</sub>/N<sub>y</sub>/N<sub>z</sub> or shrink the domain",
            max_voxel, derived.z0
        ));
    }

    let beam = BeamProfile::from_params(&p.beam_profile, p.beam_width);
    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    if let Some(reason) = beam::pattern_extent_warning(&pattern, p.lx, p.ly) {
        reasons.push(reason);
    }
    if !beam.is_pencil() {
        let footprint = beam.extent();
        let cyl = cylinder_radius(p.lx, p.ly, &pattern);
        if footprint > 0.3 * cyl {
            reasons.push(format!(
                "beam footprint (~{:.3} cm) is a large fraction of the finite cylinder radius \
                 used internally for the series solution ({:.3} cm) — the finite-beam \
                 approximation assumes the beam sits well inside that artificial boundary; \
                 shrink the beam width or enlarge L<sub>x</sub>/L<sub>y</sub>",
                footprint, cyl
            ));
        }
    }

    ValidityResult {
        valid: reasons.is_empty(),
        reasons,
    }
}

const MAX_TERMS: usize = 4000;
const REL_TOL: f64 = 1e-7;

/// The zeros of J0 (roots[k] = j_{0,k+1}) and 1/J1(root)^2 at each, up to
/// MAX_TERMS. Pure mathematical constants, independent of any model
/// parameter — computed once (see root_table below) rather than redoing a
/// Newton refinement per term per point.
struct RootTable {
    roots: Vec<f64>,
    inv_j1_sq: Vec<f64>,
}

fn build_root_table() -> RootTable {
    let mut roots = Vec::with_capacity(MAX_TERMS);
    let mut inv_j1_sq = Vec::with_capacity(MAX_TERMS);
    for k in 1..=MAX_TERMS as u32 {
        let root = j0_zero(k);
        let j1_root = j1(root);
        roots.push(root);
        inv_j1_sq.push(1.0 / (j1_root * j1_root));
    }
    RootTable { roots, inv_j1_sq }
}

/// build_root_table(), cached for the life of the process — the roots don't
/// depend on any model parameter, so redoing MAX_TERMS Newton refinements on
/// every click would be wasted work.
static ROOT_TABLE: std::sync::OnceLock<RootTable> = std::sync::OnceLock::new();

fn root_table() -> &'static RootTable {
    ROOT_TABLE.get_or_init(build_root_table)
}

/// Everything in the series that depends on the mode but not on the field
/// point — the layered recursion above, run once per mode instead of once per
/// mode per point. That's what keeps the per-term cost independent of the
/// layer count, over a (rho, z) table of tens of thousands of points.
///
/// Per-layer arrays are layer-major ([layer * n_terms + k]): a field point
/// only ever touches its own layer's row, and walks it in k order.
struct ModeTable {
    n_terms: usize,
    /// Radial spatial frequency of each mode, s_n = j_{0,n}/a'.
    sn: Vec<f64>,
    /// 1/J1(j_{0,n})² times the beam's spectral factor (see beam.rs).
    weight: Vec<f64>,
    alpha: Vec<f64>,
    /// Reflection coefficient at each layer's floor, |ρ| < 1.
    rho: Vec<f64>,
    two_theta: Vec<f64>,
    /// Prefactor of ψ_k, carrying the fluence handed down from above; layers
    /// ≥ 1 only, layer 0 has the source term instead (c_direct/c_refl).
    pre: Vec<f64>,
    c_direct: Vec<f64>,
    c_refl: Vec<f64>,
}

impl ModeTable {
    /// Mode `k`'s Green's function G_n(z) at depth `z`, which must lie in
    /// layer `li` — the two forms from this module's doc comment.
    fn green(&self, k: usize, li: usize, z: f64, stack: &Stack) -> f64 {
        let i = li * self.n_terms + k;
        let alpha = self.alpha[i];
        // Every exponent below stays ≤ 0 (z and z0 both sit inside the top
        // layer; u ≤ 2*theta). The algebraically equal exp(A)*expm1(B) forms
        // would NaN instead: expm1(B) overflows to +inf for large alpha while
        // exp(A) underflows to 0.
        if li == 0 {
            let l = &stack.layers[0];
            let direct = (-alpha * (z - stack.z0).abs()).exp()
                - (-alpha * (z + stack.z0 + 2.0 * l.zb)).exp();
            let reflected = (alpha * (z + stack.z0 - 2.0 * l.thickness)).exp()
                * (-2.0 * alpha * (z + l.zb)).exp_m1();
            self.c_direct[k] * direct + self.c_refl[k] * reflected
        } else {
            let u = alpha * (z - stack.layers[li].z_top);
            self.pre[i] * ((-u).exp() + self.rho[i] * (u - self.two_theta[i]).exp())
        }
    }
}

fn build_mode_table(stack: &Stack, beam: &BeamProfile, roots: &RootTable) -> ModeTable {
    let nl = stack.layers.len();
    let nt = MAX_TERMS;
    let mut t = ModeTable {
        n_terms: nt,
        sn: vec![0.0; nt],
        weight: vec![0.0; nt],
        alpha: vec![0.0; nl * nt],
        rho: vec![0.0; nl * nt],
        two_theta: vec![0.0; nl * nt],
        pre: vec![0.0; nl * nt],
        c_direct: vec![0.0; nt],
        c_refl: vec![0.0; nt],
    };

    let top = stack.layers[0];
    // Scratch, reused across modes: each layer's characteristic admittance
    // and its one-way decay exp(-alpha*t).
    let mut yc = vec![0.0; nl];
    let mut eth = vec![0.0; nl];

    for k in 0..nt {
        let sn = roots.roots[k] / stack.a_prime;
        t.sn[k] = sn;
        t.weight[k] = roots.inv_j1_sq[k] * beam.spectral_factor(sn);

        for (li, l) in stack.layers.iter().enumerate() {
            let alpha = (l.mua / l.d + sn * sn).sqrt();
            let i = li * nt + k;
            t.alpha[i] = alpha;
            t.two_theta[i] = 2.0 * alpha * l.thickness;
            eth[li] = (-alpha * l.thickness).exp();
            yc[li] = l.n_eff * alpha;
        }

        // Bottom-up: each step turns the admittance seen at one layer's top
        // face into the reflection coefficient at the next layer up's floor.
        let last = nl - 1;
        t.rho[last * nt + k] = -(-2.0 * t.alpha[last * nt + k] * stack.layers[last].zb).exp();
        for li in (1..nl).rev() {
            let i = li * nt + k;
            let q = eth[li] * eth[li];
            let y_top = yc[li] * (1.0 - t.rho[i] * q) / (1.0 + t.rho[i] * q);
            t.rho[(li - 1) * nt + k] = (yc[li - 1] - y_top) / (yc[li - 1] + y_top);
        }

        // Top layer: source + image, plus the round trips between the top
        // plane (reflecting -1) and the interface below it (ρ₀).
        let alpha0 = t.alpha[k];
        let rho0 = t.rho[k];
        let e_round = eth[0] * eth[0] * (-2.0 * alpha0 * top.zb).exp();
        let kk = rho0 / (1.0 + rho0 * e_round);
        let inv = 1.0 / (2.0 * top.d * alpha0);
        t.c_direct[k] = inv;
        t.c_refl[k] = kk * (-2.0 * alpha0 * (stack.z0 + top.zb)).exp_m1() * inv;

        // Downward: hand fluence/n² across each interface in turn. Taking
        // the first from green() rather than a second closed form keeps the
        // two branches continuous by construction.
        let mut g_face = t.green(k, 0, top.thickness, stack) / top.n_sq;
        for li in 1..nl {
            let i = li * nt + k;
            let q = eth[li] * eth[li];
            let pre = stack.layers[li].n_sq * g_face / (1.0 + t.rho[i] * q);
            t.pre[i] = pre;
            // ψ_k at this layer's floor in g = Φ/n² terms: what the next
            // layer down sees at its top face.
            g_face = pre * eth[li] * (1.0 + t.rho[i]) / stack.layers[li].n_sq;
        }
    }

    t
}

/// Consecutive below-tolerance terms required before fluence_kernel's series
/// is considered converged (rather than just one). Terms oscillate while
/// decaying — via J0(sn*rho), and for flat-top also via 2*J1(x)/x — so a
/// single term can land near zero at an oscillation node without the sum
/// having actually converged. A short run rules that out cheaply.
const CONVERGED_RUN: u32 = 3;

/// Fluence at one (rho, z) point (multiply by p0 for actual fluence). Sums
/// the Fourier-Bessel series (zeros of J0) until terms drop below a relative
/// tolerance.
fn fluence_kernel(rho: f64, z: f64, stack: &Stack, modes: &ModeTable) -> f64 {
    let li = stack.layer_at(z);
    let mut sum = 0.0f64;
    let mut converged_run = 0u32;
    for k in 0..modes.n_terms {
        let term = modes.green(k, li, z, stack) * modes.weight[k] * j0(modes.sn[k] * rho);
        sum += term;
        if term.abs() < REL_TOL * sum.abs().max(1e-300) {
            converged_run += 1;
            if converged_run >= CONVERGED_RUN {
                break;
            }
        } else {
            converged_run = 0;
        }
    }
    sum / (std::f64::consts::PI * stack.a_prime * stack.a_prime)
}

/// The finite-cylinder radius: large enough that its wall sits beyond every
/// voxel *as seen from every spot* (so it stays invisible in the display),
/// but no larger — a bigger radius costs more series terms to converge. Each
/// spot's kernel is the field of a source on the axis of a cylinder centred
/// on it, so an off-centre spot needs a correspondingly wider one; for a
/// single centred spot this is 1.5x half the grid's diagonal, as before.
fn cylinder_radius(lx: f64, ly: f64, pattern: &BeamPattern) -> f64 {
    1.5 * beam::max_kernel_radius(lx, ly, pattern)
}

/// The expensive part. Fluence depends only on (rho, z), never azimuth, so
/// rather than summing the series per voxel, it's summed once on a
/// coalesced (rho, z) grid and every voxel bilinearly interpolates into that
/// (beam::sample_axisymmetric_volume, shared with fpw1992.rs's finite-beam
/// path).
pub fn compute_volume(p: &LiemertKienleParams) -> (Vec<f32>, Vec<f32>) {
    let beam = BeamProfile::from_params(&p.beam_profile, p.beam_width);
    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    let stack = Stack::new(p, &pattern);
    let modes = build_mode_table(&stack, &beam, root_table());

    let grid = Grid {
        lx: p.lx,
        ly: p.ly,
        nx: p.nx,
        ny: p.ny,
        nz: p.nz,
        dz: stack.lz / p.nz as f64,
    };
    beam::sample_axisymmetric_volume(
        &grid,
        &pattern,
        p.p0,
        |z| stack.layers[stack.layer_at(z)].mua,
        |rho, z| fluence_kernel(rho, z, &stack, &modes),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fpw1992;

    fn layer(mua: f64, mus: f64, g: f64, n: f64, thickness: f64) -> LKLayerParams {
        LKLayerParams { mua, mus, g, n, thickness }
    }

    fn params(layers: Vec<LKLayerParams>) -> LiemertKienleParams {
        LiemertKienleParams {
            layers,
            p0: 1.0,
            beam_profile: "pencil".to_string(),
            beam_width: 0.0,
            beam_pattern: "single".to_string(),
            pattern_count: 1,
            pattern_spacing: 0.0,
            lx: 2.0,
            ly: 2.0,
            nx: 20,
            ny: 20,
            nz: 20,
        }
    }

    /// A genuine middle layer, with every layer differing in all four optical
    /// properties (refractive index included).
    fn contrasting_three_layer() -> LiemertKienleParams {
        params(vec![
            layer(0.15, 120.0, 0.90, 1.45, 0.30),
            layer(0.40, 60.0, 0.80, 1.33, 0.50),
            layer(0.05, 200.0, 0.95, 1.55, 1.20),
        ])
    }

    fn mode_table(stack: &Stack) -> ModeTable {
        build_mode_table(stack, &BeamProfile::Pencil, root_table())
    }

    /// With identical properties in every layer the interfaces vanish, so
    /// this should reduce to FPW1992's independently-derived closed form —
    /// the strongest cross-check short of the reference paper's own numbers.
    /// Three layers, so the middle-layer branch is exercised too.
    #[test]
    fn homogeneous_limit_matches_fpw1992() {
        let (mua, mus, g, n, p0) = (0.1, 100.0, 0.9, 1.4, 1.0);
        let (lx, ly, lz) = (4.0, 4.0, 4.0);

        let mut p = params(vec![
            layer(mua, mus, g, n, 0.5),
            layer(mua, mus, g, n, 1.0),
            layer(mua, mus, g, n, 2.5),
        ]);
        p.lx = lx;
        p.ly = ly;
        let stack = Stack::new(&p, &BeamPattern::single());
        assert_eq!(stack.lz, lz);
        let modes = mode_table(&stack);

        let fpw_params = fpw1992::Fpw1992Params {
            mua, mus, g, n, p0,
            beam_profile: "pencil".to_string(), beam_width: 0.0,
            beam_pattern: "single".to_string(), pattern_count: 1, pattern_spacing: 0.0,
            lx, ly, lz, nx: 2, ny: 2, nz: 2,
        };
        let fd = fpw1992::derived(&fpw_params);
        let reff = -1.44 / (n * n) + 0.71 / n + 0.668 + 0.0636 * n;
        let a_coeff = (1.0 + reff) / (1.0 - reff);
        let zb = 2.0 * a_coeff * fd.d;
        let mut_ = mua + fd.musp;
        let z0_fpw = 1.0 / mut_;
        let zs_img = -(z0_fpw + 2.0 * zb);
        let four_pi_d = 4.0 * std::f64::consts::PI * fd.d;

        // A small residual is expected, growing with distance (this model's
        // D = 1/(3*musp) vs FPW1992's D = 1/(3*(mua+musp)) — both standard,
        // differing conventions); anything beyond that signals a real bug.
        let mut max_rel_err = 0.0f64;
        for iz in 1..8 {
            for irho in 0..8 {
                let z = iz as f64 * 0.25;
                let rho = irho as f64 * 0.25;
                let got = p0 * fluence_kernel(rho, z, &stack, &modes);

                let r1 = (rho * rho + (z - z0_fpw).powi(2)).sqrt();
                let r2 = (rho * rho + (z - zs_img).powi(2)).sqrt();
                let want = p0
                    * ((-fd.mueff * r1).exp() / (four_pi_d * r1) - (-fd.mueff * r2).exp() / (four_pi_d * r2))
                        .max(0.0);

                let rel_err = (got - want).abs() / want.abs();
                max_rel_err = max_rel_err.max(rel_err);
            }
        }
        assert!(max_rel_err < 0.03, "max_rel_err = {max_rel_err:.4}, want < 0.03");
    }

    /// The two-layer closed form exactly as ported from the reference
    /// implementation (LightPropagation.jl's `_green_Nlaycylin_top` /
    /// `_bottom` at N=2), kept verbatim as an oracle: that port is the
    /// trusted part, and the general-N recursion has to reproduce it term
    /// for term.
    mod ported_two_layer {
        use super::LayerCoeffs;

        fn beta_gamma(alpha2: f64, l2: &LayerCoeffs, t2: f64) -> (f64, f64) {
            let beta = -(-2.0 * alpha2 * (t2 + l2.zb)).exp_m1();
            (beta, 2.0 - beta)
        }

        pub fn green_top(sn: f64, l1: &LayerCoeffs, l2: &LayerCoeffs, z0: f64, t1: f64, t2: f64, z: f64) -> f64 {
            let alpha1 = (l1.mua / l1.d + sn * sn).sqrt();
            let alpha2 = (l2.mua / l2.d + sn * sn).sqrt();
            let (beta, gamma) = beta_gamma(alpha2, l2, t2);

            let x = alpha1 * l1.n_eff * beta;
            let xy = x - alpha2 * l2.n_eff * gamma;
            let t = (-2.0 * alpha1 * (t1 + l1.zb)).exp_m1();

            let top = (-alpha1 * (z - z0).abs()).exp() - (-alpha1 * (z + z0 + 2.0 * l1.zb)).exp();
            let mut reflected = (alpha1 * (z + z0 - 2.0 * t1)).exp()
                * (-2.0 * alpha1 * (z0 + l1.zb)).exp_m1()
                * (-2.0 * alpha1 * (z + l1.zb)).exp_m1();
            reflected *= xy / (t * xy + 2.0 * x);

            (top + reflected) / (2.0 * l1.d * alpha1)
        }

        pub fn green_bottom(sn: f64, l1: &LayerCoeffs, l2: &LayerCoeffs, z0: f64, t1: f64, t2: f64, z: f64) -> f64 {
            let alpha1 = (l1.mua / l1.d + sn * sn).sqrt();
            let alpha2 = (l2.mua / l2.d + sn * sn).sqrt();
            let (beta, gamma) = beta_gamma(alpha2, l2, t2);

            let tmp1 = (-2.0 * alpha1 * (t1 + l1.zb)).exp();

            let mut out = l2.n_eff / l2.d;
            out *= (alpha1 * (z0 - t1) + alpha2 * (t1 - z)).exp();
            out /= alpha1 * l1.n_eff * beta * (1.0 + tmp1) + alpha2 * l2.n_eff * gamma * (1.0 - tmp1);
            out *= (-2.0 * alpha1 * (z0 + l1.zb)).exp_m1()
                * (-2.0 * alpha2 * (t1 + t2 - z + l2.zb)).exp_m1();

            out
        }
    }

    #[test]
    fn general_recursion_reproduces_ported_two_layer_form() {
        let p = params(vec![
            layer(0.15, 120.0, 0.90, 1.45, 0.40),
            layer(0.05, 60.0, 0.80, 1.33, 1.60),
        ]);
        let stack = Stack::new(&p, &BeamPattern::single());
        let modes = mode_table(&stack);
        let (l1, l2) = (stack.layers[0], stack.layers[1]);
        let (t1, t2) = (l1.thickness, l2.thickness);

        let mut max_rel_err = 0.0f64;
        for &k in &[0usize, 1, 3, 10, 50, 200, 1000, 3999] {
            let sn = modes.sn[k];
            for i in 0..40 {
                let z = 0.02 + i as f64 * 0.05;
                let got = modes.green(k, stack.layer_at(z), z, &stack);
                let want = if z <= t1 {
                    ported_two_layer::green_top(sn, &l1, &l2, stack.z0, t1, t2, z)
                } else {
                    ported_two_layer::green_bottom(sn, &l1, &l2, stack.z0, t1, t2, z)
                };
                // Deep, high-order terms underflow to zero in both forms;
                // the floor keeps that from reading as infinite error.
                max_rel_err = max_rel_err.max((got - want).abs() / want.abs().max(1e-200));
            }
        }
        assert!(max_rel_err < 1e-9, "max_rel_err = {max_rel_err:.3e}, want < 1e-9");
    }

    /// Splitting one layer into two identical halves has to change nothing —
    /// done below the source layer (a middle layer appears where there was
    /// none) and across it (what the top-layer branch used to answer now has
    /// to come out of the middle-layer one), which is exactly where a wrong
    /// middle-layer Green's function would show up.
    #[test]
    fn splitting_a_layer_in_half_changes_nothing() {
        let top = layer(0.10, 100.0, 0.9, 1.40, 0.30);
        let bottom = layer(0.30, 50.0, 0.8, 1.33, 1.20);

        // (a) split the bottom layer: 2 layers -> 3, with a middle layer.
        let merged = params(vec![top, bottom]);
        let mut split_bottom = bottom;
        split_bottom.thickness = 0.60;
        let split = params(vec![top, split_bottom, split_bottom]);

        // (b) split the top (source) layer: the part of it below the split
        // is now a middle layer, evaluated by the general branch.
        let mut split_top = top;
        split_top.thickness = 0.15;
        let split2 = params(vec![split_top, split_top, bottom]);

        for other in [split, split2] {
            let (sa, sb) = (Stack::new(&merged, &BeamPattern::single()), Stack::new(&other, &BeamPattern::single()));
            assert_eq!(sa.layers.len() + 1, sb.layers.len());
            let (ma, mb) = (mode_table(&sa), mode_table(&sb));
            let mut max_rel_err = 0.0f64;
            for &rho in &[0.0, 0.05, 0.3, 0.9] {
                for i in 1..30 {
                    let z = i as f64 * 0.05;
                    let a = fluence_kernel(rho, z, &sa, &ma);
                    let b = fluence_kernel(rho, z, &sb, &mb);
                    max_rel_err = max_rel_err.max((a - b).abs() / a.abs().max(1e-200));
                }
            }
            assert!(max_rel_err < 1e-9, "max_rel_err = {max_rel_err:.3e}, want < 1e-9");
        }
    }

    /// The two matching conditions the solution is built on, checked either
    /// side of every interface. Per mode rather than on the summed fluence,
    /// so series truncation doesn't muddy the differences.
    #[test]
    fn interface_matching_conditions_hold() {
        let p = contrasting_three_layer();
        let stack = Stack::new(&p, &BeamPattern::single());
        let modes = mode_table(&stack);
        let h = 1e-4;

        for &k in &[0usize, 5, 30] {
            for li in 0..stack.layers.len() - 1 {
                let zi = stack.layers[li].z_top + stack.layers[li].thickness;
                let g = |dz: f64, layer: usize| modes.green(k, layer, zi + dz, &stack);

                // Quadratic through three samples each side, read at the
                // interface — which branch applies exactly there is the
                // question, so it can't be sampled directly.
                let (a1, a2, a3) = (g(-h, li), g(-2.0 * h, li), g(-3.0 * h, li));
                let (b1, b2, b3) = (g(h, li + 1), g(2.0 * h, li + 1), g(3.0 * h, li + 1));
                let above = 3.0 * a1 - 3.0 * a2 + a3;
                let below = 3.0 * b1 - 3.0 * b2 + b3;
                let d_above = (2.5 * a1 - 4.0 * a2 + 1.5 * a3) / h;
                let d_below = -(2.5 * b1 - 4.0 * b2 + 1.5 * b3) / h;

                let (a, b) = (stack.layers[li], stack.layers[li + 1]);
                let fluence_err =
                    (above / a.n_sq - below / b.n_sq).abs() / (above / a.n_sq).abs();
                let flux_err = (a.d * d_above - b.d * d_below).abs() / (a.d * d_above).abs();
                assert!(
                    fluence_err < 1e-6,
                    "mode {k}, interface {li}: fluence/n^2 mismatch {fluence_err:.2e}"
                );
                assert!(
                    flux_err < 1e-4,
                    "mode {k}, interface {li}: D*dPhi/dz mismatch {flux_err:.2e}"
                );
            }
        }
    }

    /// A fine-grid finite-volume solve of the same per-mode boundary-value
    /// problem — an oracle sharing no algebra with the analytic recursion,
    /// only the physics it has to satisfy. Discretizes n_eff·(g'' - α²g) =
    /// -δ(z - z0) in divergence form (so the natural face condition *is*
    /// continuity of n_eff·g'), over the stack extended by the extrapolated
    /// slabs where fluence is pinned to zero. Cells are laid out per region
    /// so interfaces fall on faces and the source on a centre.
    fn finite_volume_mode(stack: &Stack, sn: f64, h: f64) -> (Vec<f64>, Vec<f64>) {
        let last = stack.layers.len() - 1;
        let (z0, t0) = (stack.z0, stack.layers[0].thickness);
        let mut regions = vec![
            (-stack.layers[0].zb, 0.0, 0),
            (0.0, z0 - h / 2.0, 0),
            (z0 - h / 2.0, z0 + h / 2.0, 0), // the source cell
            (z0 + h / 2.0, t0, 0),
        ];
        for (li, l) in stack.layers.iter().enumerate().skip(1) {
            regions.push((l.z_top, l.z_top + l.thickness, li));
        }
        regions.push((stack.lz, stack.lz + stack.layers[last].zb, last));

        let (mut centre, mut width, mut layer) = (Vec::new(), Vec::new(), Vec::new());
        let mut source_cell = usize::MAX;
        for (i, &(lo, hi, li)) in regions.iter().enumerate() {
            let cells = if i == 2 { 1 } else { (((hi - lo) / h).round() as usize).max(1) };
            let hw = (hi - lo) / cells as f64;
            for j in 0..cells {
                if i == 2 {
                    source_cell = centre.len();
                }
                centre.push(lo + (j as f64 + 0.5) * hw);
                width.push(hw);
                layer.push(li);
            }
        }

        let n = centre.len();
        let n_eff: Vec<f64> = layer.iter().map(|&li| stack.layers[li].n_eff).collect();

        // Face conductances; the two outer faces see the pinned zero half a
        // cell away.
        let mut face = vec![0.0; n + 1];
        face[0] = n_eff[0] / (width[0] / 2.0);
        face[n] = n_eff[n - 1] / (width[n - 1] / 2.0);
        for i in 1..n {
            face[i] = 1.0 / (width[i - 1] / (2.0 * n_eff[i - 1]) + width[i] / (2.0 * n_eff[i]));
        }

        let (mut lower, mut diag, mut upper) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            let l = &stack.layers[layer[i]];
            let alpha_sq = l.mua / l.d + sn * sn;
            lower[i] = face[i];
            upper[i] = face[i + 1];
            diag[i] = -(face[i] + face[i + 1] + alpha_sq * l.n_eff * width[i]);
        }
        lower[0] = 0.0;
        upper[n - 1] = 0.0;
        rhs[source_cell] = -1.0;

        // Thomas algorithm (the matrix is tridiagonal and diagonally dominant).
        let (mut cp, mut dp) = (vec![0.0; n], vec![0.0; n]);
        cp[0] = upper[0] / diag[0];
        dp[0] = rhs[0] / diag[0];
        for i in 1..n {
            let m = diag[i] - lower[i] * cp[i - 1];
            cp[i] = upper[i] / m;
            dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m;
        }
        let mut g = vec![0.0; n];
        g[n - 1] = dp[n - 1];
        for i in (0..n - 1).rev() {
            g[i] = dp[i] - cp[i] * g[i + 1];
        }

        // Back to fluence, keeping only the cells inside the stack itself.
        let mut zs = Vec::new();
        let mut phi = Vec::new();
        for i in 0..n {
            if centre[i] >= 0.0 && centre[i] <= stack.lz {
                zs.push(centre[i]);
                phi.push(stack.layers[layer[i]].n_sq * g[i]);
            }
        }
        (zs, phi)
    }

    #[test]
    fn layered_greens_function_matches_finite_volume_solve() {
        let p = contrasting_three_layer();
        let stack = Stack::new(&p, &BeamPattern::single());
        let modes = mode_table(&stack);

        for &k in &[0usize, 3, 12] {
            let (zs, phi) = finite_volume_mode(&stack, modes.sn[k], 2e-4);
            // Against the profile's peak: fluence is pinned to zero at the
            // bottom boundary, where a pointwise relative error means little.
            let scale = phi.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let mut max_err = 0.0f64;
            for (i, &z) in zs.iter().enumerate() {
                let got = modes.green(k, stack.layer_at(z), z, &stack);
                max_err = max_err.max((got - phi[i]).abs() / scale);
            }
            assert!(max_err < 1e-4, "mode {k}: max error {max_err:.3e} of peak, want < 1e-4");
        }
    }

    /// A vanishingly narrow Gaussian's spectral factor is ~1 at every mode
    /// this series actually sums (see beam.rs's own point-source limit
    /// check), so it should reproduce the plain point-source kernel.
    #[test]
    fn narrow_gaussian_matches_pencil() {
        let p = params(vec![
            layer(0.1, 100.0, 0.9, 1.4, 0.3),
            layer(0.3, 50.0, 0.9, 1.4, 1.7),
        ]);
        let stack = Stack::new(&p, &BeamPattern::single());
        let pencil = mode_table(&stack);
        let narrow = build_mode_table(&stack, &BeamProfile::Gaussian { sigma: 1e-7 }, root_table());

        for &(rho, z) in &[(0.0, 0.1), (0.2, 0.3), (0.1, 0.8)] {
            let want = fluence_kernel(rho, z, &stack, &pencil);
            let got = fluence_kernel(rho, z, &stack, &narrow);
            let rel_err = (got - want).abs() / want.abs();
            assert!(rel_err < 1e-4, "(rho={rho}, z={z}): got {got}, want {want}");
        }
    }

    fn beam_params(beam_profile: &str, beam_width: f64) -> LiemertKienleParams {
        let mut p = params(vec![
            layer(0.1, 100.0, 0.9, 1.4, 0.3),
            layer(0.3, 50.0, 0.9, 1.4, 1.7),
        ]);
        p.beam_profile = beam_profile.to_string();
        p.beam_width = beam_width;
        p
    }

    /// A flat-top beam wider than the ~0.3x-cylinder-radius threshold should
    /// trip check_validity's finite-beam warning (see its own comment for
    /// why); a beam well inside it, or a plain pencil beam, should not.
    #[test]
    fn wide_beam_triggers_footprint_validity_warning() {
        // cylinder_radius(2.0, 2.0) is about 2.12 cm, so a 1 cm flat-top
        // radius (~47% of it) should trip the 30% threshold.
        let wide = beam_params("flattop", 1.0);
        let wide_derived = derived(&wide);
        let wide_result = check_validity(&wide, &wide_derived);
        assert!(!wide_result.valid, "wide flat-top beam should be flagged invalid");
        assert!(
            wide_result.reasons.iter().any(|r| r.contains("beam footprint")),
            "expected a beam-footprint reason, got {:?}", wide_result.reasons
        );

        let narrow = beam_params("flattop", 0.05);
        let narrow_derived = derived(&narrow);
        let narrow_result = check_validity(&narrow, &narrow_derived);
        assert!(
            !narrow_result.reasons.iter().any(|r| r.contains("beam footprint")),
            "narrow flat-top beam should not trip the footprint warning, got {:?}", narrow_result.reasons
        );

        let pencil = beam_params("pencil", 0.0);
        let pencil_derived = derived(&pencil);
        let pencil_result = check_validity(&pencil, &pencil_derived);
        assert!(
            !pencil_result.reasons.iter().any(|r| r.contains("beam footprint")),
            "pencil beam should never trip the footprint warning, got {:?}", pencil_result.reasons
        );
    }

    /// A grid pattern through a layered stack: the artificial cylinder has to
    /// grow with the pattern (so its wall stays invisible), the spots have to
    /// land symmetrically, and the whole volume has to stay finite.
    #[test]
    fn grid_pattern_widens_the_cylinder_and_stays_symmetric() {
        let mut p = params(vec![
            layer(0.1, 100.0, 0.9, 1.4, 0.3),
            layer(0.3, 50.0, 0.9, 1.4, 1.2),
        ]);
        p.nx = 41;
        p.ny = 41;
        p.nz = 20;
        p.beam_pattern = "grid".to_string();
        p.pattern_count = 3;
        p.pattern_spacing = 0.25;

        let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
        assert_eq!(derived(&p).spots, 9);
        // The wall has to sit further out than for one centred spot, or the
        // outer spots would see it.
        let widened = cylinder_radius(p.lx, p.ly, &pattern);
        assert!(widened > cylinder_radius(p.lx, p.ly, &BeamPattern::single()));

        let (phi, abs) = compute_volume(&p);
        for (i, (&v, &a)) in phi.iter().zip(abs.iter()).enumerate() {
            assert!(v.is_finite() && v > 0.0, "phi[{i}] = {v}");
            assert!(a.is_finite() && a > 0.0, "abs[{i}] = {a}");
        }

        // A square grid of spots is symmetric under both mirrors and under
        // swapping the two axes.
        let at = |ix: usize, iy: usize, iz: usize| phi[ix + iy * 41 + iz * 41 * 41] as f64;
        for k in 1..=10 {
            let c = 20usize;
            assert!((at(c - k, c, 1) - at(c + k, c, 1)).abs() < 1e-5 * at(c - k, c, 1));
            assert!((at(c - k, c, 1) - at(c, c - k, 1)).abs() < 1e-5 * at(c - k, c, 1));
        }
    }

    /// Warnings name the layer by index, and Lz follows the stack rather than
    /// a separate grid-depth parameter.
    #[test]
    fn derived_and_validity_cover_every_layer() {
        let p = params(vec![
            layer(0.1, 100.0, 0.9, 1.4, 0.3),
            layer(0.5, 50.0, 0.9, 1.4, 0.5),
            layer(2.0, 20.0, 0.9, 1.4, 0.4), // musp'/mua = 1 — should be flagged
        ]);
        let d = derived(&p);
        assert_eq!(d.layers.len(), 3);
        assert!((d.lz - 1.2).abs() < 1e-12, "Lz = {}", d.lz);
        assert!((d.z0 - 0.1).abs() < 1e-12, "z0 = {}", d.z0);

        let result = check_validity(&p, &d);
        assert!(
            result.reasons.iter().any(|r| r.contains("layer 3")),
            "expected layer 3 to be flagged, got {:?}", result.reasons
        );
    }
}

#[cfg(test)]
mod perf_and_sanity {
    use super::*;

    fn layer(mua: f64, mus: f64, g: f64, n: f64, thickness: f64) -> LKLayerParams {
        LKLayerParams { mua, mus, g, n, thickness }
    }

    #[test]
    fn large_grid_is_fast_and_finite() {
        let params = LiemertKienleParams {
            layers: vec![
                layer(0.1, 100.0, 0.9, 1.4, 0.3),
                layer(0.3, 50.0, 0.9, 1.33, 0.5),
                layer(0.2, 80.0, 0.9, 1.4, 1.2),
            ],
            p0: 1.0,
            beam_profile: "pencil".to_string(),
            beam_width: 0.0,
            beam_pattern: "single".to_string(),
            pattern_count: 1,
            pattern_spacing: 0.0,
            lx: 2.0,
            ly: 2.0,
            nx: 400,
            ny: 400,
            nz: 400,
        };
        let t0 = std::time::Instant::now();
        let (phi, abs) = compute_volume(&params);
        let dt = t0.elapsed();
        println!("400^3 grid, 3 layers: {:?}", dt);

        assert_eq!(phi.len(), 400 * 400 * 400);
        for (i, (&p, &a)) in phi.iter().zip(abs.iter()).enumerate() {
            assert!(p.is_finite() && p >= 0.0, "phi[{i}] = {p}");
            assert!(a.is_finite() && a >= 0.0, "abs[{i}] = {a}");
        }
        assert!(dt.as_secs_f64() < 10.0, "too slow: {:?}", dt);

        // fluence should decay with depth on-axis
        let cx = 200usize;
        let idx = |iz: usize| cx + cx * 400 + iz * 400 * 400;
        let phi_shallow = phi[idx(5)];
        let phi_deep = phi[idx(300)];
        assert!(phi_shallow > phi_deep, "fluence should decay with depth: shallow={phi_shallow} deep={phi_deep}");
    }

    /// A pattern adds table lookups per voxel, not kernel evaluations — the
    /// series is still summed exactly once — so a 25-spot grid must not cost
    /// anything like 25x a single spot.
    #[test]
    fn many_spots_stay_affordable() {
        let mut params = LiemertKienleParams {
            layers: vec![
                layer(0.1, 100.0, 0.9, 1.4, 0.3),
                layer(0.3, 50.0, 0.9, 1.33, 1.7),
            ],
            p0: 1.0,
            beam_profile: "pencil".to_string(),
            beam_width: 0.0,
            beam_pattern: "single".to_string(),
            pattern_count: 1,
            pattern_spacing: 0.0,
            lx: 2.0,
            ly: 2.0,
            nx: 200,
            ny: 200,
            nz: 200,
        };
        let t0 = std::time::Instant::now();
        compute_volume(&params);
        let single = t0.elapsed();

        params.beam_pattern = "grid".to_string();
        params.pattern_count = 5;
        params.pattern_spacing = 0.15;
        let t0 = std::time::Instant::now();
        let (phi, _) = compute_volume(&params);
        let patterned = t0.elapsed();
        println!("200^3: 1 spot {single:?}, 25 spots {patterned:?}");

        assert!(phi.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(
            patterned.as_secs_f64() < 4.0 * single.as_secs_f64().max(0.05),
            "25 spots took {patterned:?} vs {single:?} for one"
        );
    }
}

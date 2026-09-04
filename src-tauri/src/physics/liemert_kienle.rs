//! Two-layer, point-source diffusion approximation.
//! Reference: André Liemert & Alwin Kienle, "Light diffusion in a turbid
//! cylinder. II. Layered case," Opt. Express 18, 9266-9279 (2010).
//!
//! FPW1992 gives a point source but only one homogeneous layer; Kubelka-Munk
//! gives many layers but only diffuse (non-point) illumination. This fills
//! the gap: a point/pencil beam through two stacked homogeneous layers. It's
//! still the diffusion approximation (same family as FPW1992), just solved
//! for a layered geometry — no Monte Carlo involved.
//!
//! The Liemert-Kienle solution is for a *finite cylinder* of radius `a`
//! (solved via a Fourier-Bessel series — zeros of J0 — rather than FPW1992's
//! closed-form real+image-source formula, because layering breaks the
//! translational symmetry in z that makes FPW1992's shortcut possible). This
//! module places that cylinder's axis through the beam, at the center of the
//! grid's top face (matching FPW1992's beam placement), and picks `a`
//! generously larger than the visible grid so the artificial cylinder wall
//! doesn't show up in the display — see `cylinder_radius`. A larger `a`
//! costs more series terms to converge, so it isn't picked larger than that.
//!
//! Faithfully ported from the paper's own reference implementation
//! (github.com/heltonmc/LightPropagation.jl, `_green_Nlaycylin_top`/`_bottom`
//! specialized to N=2, `_get_βγ2`), not re-derived from scratch — this is
//! exactly the kind of series-of-Bessel-functions math where a transcription
//! error would silently produce a plausible-looking wrong number rather than
//! an obvious bug. One deliberate deviation from that reference: the
//! reflection-coefficient fit for the extrapolated boundary condition (A, zb
//! below) is reused from fpw1992.rs rather than the reference's own
//! (different but equally standard) polynomial fit, so both point-source
//! models in this app treat the air-tissue boundary identically.

use crate::physics::beam::{self, BeamProfile};
use crate::physics::bessel::{j0, j0_zero, j1};
use serde::{Deserialize, Serialize};

/// Flat rather than nested (layer1: {...}) so this maps directly onto the
/// generic parameter-panel builder in ui-params.ts, which only knows how to
/// merge a group's fields flat or (for a `repeat` group) collect them into
/// an array of identically-shaped instances — neither fits two layers with
/// different schemas (layer 1 alone carries a thickness).
#[derive(Deserialize)]
pub struct LiemertKienleParams {
    pub mua1: f64,
    pub mus1: f64,
    pub g1: f64,
    pub n1: f64,
    /// Layer 1 thickness [cm]. Layer 2 fills the rest of the grid, [t1, lz].
    pub t1: f64,
    pub mua2: f64,
    pub mus2: f64,
    pub g2: f64,
    pub n2: f64,
    pub p0: f64,
    /// "pencil" | "gaussian" | "flattop" — see beam.rs. `beam_width` is
    /// sigma (Gaussian) or radius (flattop) in cm, ignored for "pencil".
    pub beam_profile: String,
    pub beam_width: f64,
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

#[derive(Serialize, Clone, Copy)]
pub struct LiemertKienleDerived {
    pub musp1: f64,
    #[serde(rename = "D1")]
    pub d1: f64,
    pub mueff1: f64,
    pub musp2: f64,
    #[serde(rename = "D2")]
    pub d2: f64,
    pub mueff2: f64,
    pub z0: f64,
}

#[derive(Serialize)]
pub struct ValidityResult {
    pub valid: bool,
    pub reasons: Vec<String>,
}

/// Per-layer coefficients derived from (mua, musp, n) alone — shared by
/// derived(), check_validity(), and compute_volume() so it's written once.
#[derive(Clone, Copy)]
struct LayerCoeffs {
    mua: f64,
    musp: f64,
    d: f64,
    /// Extrapolation length (2*A*D) at this layer's boundary with air.
    zb: f64,
    /// D * n^2 — the interface-weighting term used throughout the Green's
    /// function (a standard generalization of flux continuity to account for
    /// a refractive-index step between layers).
    n_eff: f64,
}

/// Same reflection-coefficient fit as fpw1992.rs (Groenhuis et al.), reused
/// here so both point-source models treat the air-tissue boundary the same
/// way — see this module's doc comment.
fn extrapolation_length(n: f64, d: f64) -> f64 {
    let reff = -1.44 / (n * n) + 0.71 / n + 0.668 + 0.0636 * n;
    let a = (1.0 + reff) / (1.0 - reff);
    2.0 * a * d
}

fn layer_coeffs(mua: f64, mus: f64, g: f64, n: f64) -> LayerCoeffs {
    let musp = mus * (1.0 - g);
    let d = 1.0 / (3.0 * musp);
    let zb = extrapolation_length(n, d);
    LayerCoeffs {
        mua,
        musp,
        d,
        zb,
        n_eff: d * n * n,
    }
}

pub fn derived(p: &LiemertKienleParams) -> LiemertKienleDerived {
    let l1 = layer_coeffs(p.mua1, p.mus1, p.g1, p.n1);
    let l2 = layer_coeffs(p.mua2, p.mus2, p.g2, p.n2);
    LiemertKienleDerived {
        musp1: l1.musp,
        d1: l1.d,
        mueff1: (l1.mua / l1.d).sqrt(),
        musp2: l2.musp,
        d2: l2.d,
        mueff2: (l2.mua / l2.d).sqrt(),
        z0: 1.0 / l1.musp,
    }
}

pub fn check_validity(p: &LiemertKienleParams, derived: &LiemertKienleDerived) -> ValidityResult {
    let mut reasons = Vec::new();

    let ratio1 = derived.musp1 / p.mua1;
    if ratio1 < 10.0 {
        reasons.push(format!(
            "layer 1: μ<sub>s</sub>'/μ<sub>a</sub> = {:.2} (want ≳10) — absorption is too strong \
             relative to scattering for light to randomize direction before being absorbed",
            ratio1
        ));
    }
    let ratio2 = derived.musp2 / p.mua2;
    if ratio2 < 10.0 {
        reasons.push(format!(
            "layer 2: μ<sub>s</sub>'/μ<sub>a</sub> = {:.2} (want ≳10) — absorption is too strong \
             relative to scattering for light to randomize direction before being absorbed",
            ratio2
        ));
    }
    if derived.z0 >= p.t1 {
        reasons.push(format!(
            "the source depth z<sub>0</sub> = {:.3} cm falls outside layer 1 (thickness {:.3} cm) \
             — this model assumes the beam's effective source sits within the first layer; \
             increase layer 1's thickness or scattering coefficient",
            derived.z0, p.t1
        ));
    }
    let dx = p.lx / p.nx as f64;
    let dy = p.ly / p.ny as f64;
    let dz = p.lz / p.nz as f64;
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
    if !beam.is_pencil() {
        let footprint = beam.extent();
        let cyl = cylinder_radius(p.lx, p.ly);
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

/// β, γ coefficients for the N=2 case (Liemert & Kienle eq. 17-18,
/// specialized to two layers) — how layer 2 (with its own extrapolated
/// bottom boundary zb2, assuming a finite lz; make lz large relative to the
/// optical penetration depth for an effectively semi-infinite layer 2)
/// reflects back into layer 1.
fn beta_gamma(alpha2: f64, l2: &LayerCoeffs, t2: f64) -> (f64, f64) {
    let beta = -(-2.0 * alpha2 * (t2 + l2.zb)).exp_m1();
    (beta, 2.0 - beta)
}

/// Everything the series sum needs that stays fixed across every (rho, z)
/// point in one compute_volume() call — bundled so per-point functions take
/// one reference instead of a long, error-prone run of positional f64s.
struct Geometry {
    l1: LayerCoeffs,
    l2: LayerCoeffs,
    z0: f64,
    t1: f64,
    t2: f64,
    a_prime: f64,
}

/// One term's Green's function contribution, for a field point in layer 1
/// (z <= t1). `sn` is the current radial spatial frequency (a zero of J0,
/// scaled by the cylinder's extrapolated radius).
fn green_top(sn: f64, g: &Geometry, z: f64) -> f64 {
    let (l1, l2) = (&g.l1, &g.l2);
    let alpha1 = (l1.mua / l1.d + sn * sn).sqrt();
    let alpha2 = (l2.mua / l2.d + sn * sn).sqrt();
    let (beta, gamma) = beta_gamma(alpha2, l2, g.t2);

    let x = alpha1 * l1.n_eff * beta;
    let xy = x - alpha2 * l2.n_eff * gamma;
    let t = (-2.0 * alpha1 * (g.t1 + l1.zb)).exp_m1();

    // Algebraically exp(A)*expm1(B) with A = -alpha1*(z+z0+2*zb1) and
    // B = -alpha1*(|z-z0|-(z+z0+2*zb1)); expanding gives exp(A+B) - exp(A)
    // with A+B = -alpha1*|z-z0|. Both A and A+B are always <= 0, so this
    // form is two plain decaying exponentials — computing it as exp(A) times
    // expm1(B) instead overflows expm1(B) to +inf for large alpha1 (deep in
    // the series) while exp(A) underflows to exactly 0, giving 0*inf = NaN.
    let top = (-alpha1 * (z - g.z0).abs()).exp() - (-alpha1 * (z + g.z0 + 2.0 * l1.zb)).exp();
    let mut reflected = (alpha1 * (z + g.z0 - 2.0 * g.t1)).exp()
        * (-2.0 * alpha1 * (g.z0 + l1.zb)).exp_m1()
        * (-2.0 * alpha1 * (z + l1.zb)).exp_m1();
    reflected *= xy / (t * xy + 2.0 * x);

    (top + reflected) / (2.0 * l1.d * alpha1)
}

/// Same, for a field point in layer 2 (z > t1).
fn green_bottom(sn: f64, g: &Geometry, z: f64) -> f64 {
    let (l1, l2) = (&g.l1, &g.l2);
    let alpha1 = (l1.mua / l1.d + sn * sn).sqrt();
    let alpha2 = (l2.mua / l2.d + sn * sn).sqrt();
    let (beta, gamma) = beta_gamma(alpha2, l2, g.t2);

    let tmp1 = (-2.0 * alpha1 * (g.t1 + l1.zb)).exp();

    let mut out = l2.n_eff / l2.d;
    out *= (alpha1 * (g.z0 - g.t1) + alpha2 * (g.t1 - z)).exp();
    out /= alpha1 * l1.n_eff * beta * (1.0 + tmp1) + alpha2 * l2.n_eff * gamma * (1.0 - tmp1);
    out *= (-2.0 * alpha1 * (g.z0 + l1.zb)).exp_m1()
        * (-2.0 * alpha2 * (g.t1 + g.t2 - z + l2.zb)).exp_m1();

    out
}

const MAX_TERMS: usize = 4000;
const REL_TOL: f64 = 1e-7;

/// The zeros of J0 (roots[k] = j_{0,k+1}) and 1/J1(root)^2 at each, up to
/// MAX_TERMS. These are pure mathematical constants — independent of rho, z,
/// or any optical property — so they're computed once per compute_volume()
/// call and shared across every (rho, z) point's series sum, rather than
/// recomputed (each root costing a Newton refinement, itself several Bessel
/// evaluations) on every single term of every single point.
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

/// build_root_table(), computed once for the life of the process and reused
/// by every subsequent compute_volume() call — the roots are pure
/// mathematical constants (zeros of J0), independent of any model parameter,
/// so redoing MAX_TERMS Newton refinements on every click is wasted work.
static ROOT_TABLE: std::sync::OnceLock<RootTable> = std::sync::OnceLock::new();

fn root_table() -> &'static RootTable {
    ROOT_TABLE.get_or_init(build_root_table)
}

/// Consecutive below-tolerance terms required before fluence_kernel's series
/// sum is considered converged (rather than just one). A plain pencil beam's
/// terms already oscillate (via J0(sn*rho)) while decaying, and a flat-top
/// beam's spectral factor 2*J1(x)/x oscillates on top of that — either can
/// land an individual term near zero at a node without the sum having
/// actually converged yet. Requiring a short run rather than a single small
/// term makes that a non-issue at negligible extra cost (a handful of terms,
/// against MAX_TERMS' budget of thousands).
const CONVERGED_RUN: u32 = 3;

/// Fluence at one (rho, z) point, in units where multiplying by p0 gives the
/// actual fluence. Sums the Fourier-Bessel series (zeros of J0) until terms
/// drop below a relative tolerance of the running sum. `beam` multiplies
/// each mode by that beam's spectral factor (1.0 for Pencil, i.e. no-op) —
/// see beam.rs for why this is the correct, exact-to-this-model's-own-
/// approximation way to swap in a finite beam.
fn fluence_kernel(rho: f64, z: f64, g: &Geometry, roots: &RootTable, beam: &BeamProfile) -> f64 {
    let mut sum = 0.0f64;
    let mut converged_run = 0u32;
    for k in 0..MAX_TERMS {
        let sn = roots.roots[k] / g.a_prime;
        let green = if z <= g.t1 { green_top(sn, g, z) } else { green_bottom(sn, g, z) };
        let term = (green * roots.inv_j1_sq[k]) * j0(sn * rho) * beam.spectral_factor(sn);
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
    sum / (std::f64::consts::PI * g.a_prime * g.a_prime)
}

/// The cylinder radius used for the finite-cylinder series solution: large
/// enough that its wall sits comfortably beyond every voxel (so the
/// artificial boundary isn't visible in the displayed volume), but no
/// larger — a bigger radius means more series terms to converge (see the
/// reference implementation's own convergence notes).
fn cylinder_radius(lx: f64, ly: f64) -> f64 {
    1.5 * (0.5 * lx).hypot(0.5 * ly)
}

/// The expensive part. Fluence only depends on (rho, z) — the radial
/// distance from the beam axis and depth — never on the azimuthal angle, so
/// rather than summing the series per voxel (nx*ny*nz times), it's summed
/// once on a coalesced (rho, z) grid and every voxel bilinearly interpolates
/// into that, via beam::sample_axisymmetric_volume (shared with fpw1992.rs's
/// own finite-beam path — both models reduce to exactly this shape once you
/// have a Phi(rho, z) kernel in hand).
pub fn compute_volume(p: &LiemertKienleParams) -> (Vec<f32>, Vec<f32>) {
    let l1 = layer_coeffs(p.mua1, p.mus1, p.g1, p.n1);
    let l2 = layer_coeffs(p.mua2, p.mus2, p.g2, p.n2);
    let z0 = 1.0 / l1.musp;
    let t1 = p.t1;
    let t2 = (p.lz - p.t1).max(1e-9);
    let a_prime = cylinder_radius(p.lx, p.ly) + l1.zb;
    let geom = Geometry { l1, l2, z0, t1, t2, a_prime };
    let roots = root_table();
    let beam = beam::BeamProfile::from_params(&p.beam_profile, p.beam_width);
    let dz = p.lz / p.nz as f64;

    beam::sample_axisymmetric_volume(
        p.lx,
        p.ly,
        p.nx,
        p.ny,
        p.nz,
        dz,
        p.p0,
        |z| if z <= t1 { l1.mua } else { l2.mua },
        |rho, z| fluence_kernel(rho, z, &geom, roots, &beam),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::fpw1992;

    /// With identical optical properties in both layers, the interface
    /// physically vanishes — this model should reduce to FPW1992's
    /// homogeneous closed-form solution (independently derived, already in
    /// production use), which is the strongest cross-check available for
    /// this formula short of comparing to the reference paper's own numbers.
    #[test]
    fn homogeneous_limit_matches_fpw1992() {
        let (mua, mus, g, n, p0) = (0.1, 100.0, 0.9, 1.4, 1.0);
        let (lx, ly, lz) = (4.0, 4.0, 4.0);
        let t1 = 0.5;

        let l1 = layer_coeffs(mua, mus, g, n);
        let l2 = layer_coeffs(mua, mus, g, n);
        let z0 = 1.0 / l1.musp;
        let t2 = lz - t1;
        let a_prime = cylinder_radius(lx, ly) + l1.zb;
        let geom = Geometry { l1, l2, z0, t1, t2, a_prime };
        let roots = build_root_table();

        let fpw_params = fpw1992::Fpw1992Params {
            mua, mus, g, n, p0,
            beam_profile: "pencil".to_string(), beam_width: 0.0,
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

        // A small residual (this model's D = 1/(3*musp) vs FPW1992's own
        // D = 1/(3*(mua+musp)) — both standard, differing conventions,
        // see this file's doc comment) that grows with distance from the
        // source is expected; anything beyond that signals an actual bug.
        let mut max_rel_err = 0.0f64;
        for iz in 1..8 {
            for irho in 0..8 {
                let z = iz as f64 * 0.25;
                let rho = irho as f64 * 0.25;
                let got = p0 * fluence_kernel(rho, z, &geom, &roots, &BeamProfile::Pencil);

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

    /// A vanishingly narrow Gaussian's spectral factor is ~1 at every mode
    /// this series actually sums (see beam.rs's own point-source limit
    /// check), so it should reproduce the plain point-source kernel.
    #[test]
    fn narrow_gaussian_matches_pencil() {
        let (mua, mus, g, n) = (0.1, 100.0, 0.9, 1.4);
        let (lx, ly, lz) = (2.0, 2.0, 2.0);
        let t1 = 0.3;

        let l1 = layer_coeffs(mua, mus, g, n);
        let l2 = layer_coeffs(0.3, 50.0, g, n);
        let z0 = 1.0 / l1.musp;
        let t2 = lz - t1;
        let a_prime = cylinder_radius(lx, ly) + l1.zb;
        let geom = Geometry { l1, l2, z0, t1, t2, a_prime };
        let roots = build_root_table();

        let pencil = BeamProfile::Pencil;
        let narrow_gaussian = BeamProfile::Gaussian { sigma: 1e-7 };

        for &(rho, z) in &[(0.0, 0.1), (0.2, 0.3), (0.1, 0.8)] {
            let want = fluence_kernel(rho, z, &geom, &roots, &pencil);
            let got = fluence_kernel(rho, z, &geom, &roots, &narrow_gaussian);
            let rel_err = (got - want).abs() / want.abs();
            assert!(rel_err < 1e-4, "(rho={rho}, z={z}): got {got}, want {want}");
        }
    }

    fn beam_params(beam_profile: &str, beam_width: f64) -> LiemertKienleParams {
        LiemertKienleParams {
            mua1: 0.1, mus1: 100.0, g1: 0.9, n1: 1.4,
            t1: 0.3,
            mua2: 0.3, mus2: 50.0, g2: 0.9, n2: 1.4,
            p0: 1.0,
            beam_profile: beam_profile.to_string(), beam_width,
            lx: 2.0, ly: 2.0, lz: 2.0,
            nx: 20, ny: 20, nz: 20,
        }
    }

    /// A flat-top beam wider than the ~0.3x-cylinder-radius threshold should
    /// trip check_validity's finite-beam warning (see check_validity's own
    /// comment for why that threshold exists); a beam well inside it, or a
    /// plain pencil beam, should not.
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
}

#[cfg(test)]
mod perf_and_sanity {
    use super::*;
    use std::time::Instant;

    #[test]
    fn large_grid_is_fast_and_finite() {
        let params = LiemertKienleParams {
            mua1: 0.1, mus1: 100.0, g1: 0.9, n1: 1.4,
            t1: 0.3,
            mua2: 0.3, mus2: 50.0, g2: 0.9, n2: 1.4,
            p0: 1.0,
            beam_profile: "pencil".to_string(), beam_width: 0.0,
            lx: 2.0, ly: 2.0, lz: 2.0,
            nx: 400, ny: 400, nz: 400,
        };
        let t0 = Instant::now();
        let (phi, abs) = compute_volume(&params);
        let dt = t0.elapsed();
        println!("400^3 grid: {:?}", dt);

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
}

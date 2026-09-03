//! Finite transverse beam profiles, convolved onto a point-source model.
//!
//! Every point-source model in this crate (fpw1992.rs, liemert_kienle.rs)
//! reduces to a kernel Phi_pt(rho, z) — radial distance from the beam axis
//! and depth, never azimuth, since a point source is trivially axisymmetric.
//! A finite beam replaces that point source with a radially symmetric power
//! density S(rho') spread over the entry surface, normalized so
//! integral(S dA) = 1 (so p0 keeps meaning "total input power" regardless of
//! beam shape). Since the diffusion equation is linear, the resulting
//! fluence is exactly the transverse (2-D) convolution of the point-source
//! kernel with S:
//!
//!   Phi_beam(rho_vec, z) = integral[ S(rho'_vec) * Phi_pt(|rho_vec - rho'_vec|, z) d2rho' ]
//!
//! which (convolution of two radially-symmetric functions) is itself
//! radially symmetric in rho_vec, i.e. still just a function of (rho, z).
//!
//! Two ways to evaluate that convolution are used here, chosen per model to
//! match how each already expresses Phi_pt:
//!
//! - Liemert-Kienle already expands Phi_pt as a sum over discrete transverse
//!   spatial frequencies s_n = (zeros of J0)/a' (a Fourier-Bessel/Dini
//!   series, see liemert_kienle.rs) with coefficients that, for a point
//!   source at the axis, all carry an implicit factor of J0(s_n * 0) = 1 —
//!   a delta function's Fourier-Bessel spectrum is flat. Replacing the point
//!   source with a distributed S(rho') replaces that implicit "1" with
//!   S's own 0-order Hankel transform at s_n (spectral_factor below): the
//!   z-dependent part of each mode (how the layered medium responds at
//!   transverse wavenumber s_n) is unchanged, only how strongly the source
//!   excites that mode changes. Exact in the limit that the beam sits well
//!   inside the artificial cylinder wall — the same approximation the
//!   point-source model already relies on for that wall to stay invisible
//!   (see cylinder_radius in liemert_kienle.rs).
//! - FPW1992 has no such series — it's a closed-form real+image dipole pair
//!   for a translation-invariant infinite plane — so its convolution is
//!   evaluated directly as a 2-D numerical integral over the beam's
//!   footprint (convolve_radial below), generic in the point-source kernel.
//!
//! Gaussian and flat-top's 0-order Hankel transforms (spectral_factor) are
//! standard closed forms:
//!   Gaussian, S(rho') = exp(-rho'^2/(2 sigma^2)) / (2*pi*sigma^2):
//!     Hankel transform = exp(-s^2 sigma^2 / 2)
//!   Flat-top disk of radius R, S(rho') = 1/(pi R^2) for rho' <= R:
//!     Hankel transform = 2*J1(s R) / (s R)
//! Both -> 1 as the beam narrows (sigma, R -> 0), recovering the point
//! source — a useful sanity check on the algebra above.

use crate::physics::bessel::j1;

#[derive(Clone, Copy)]
pub enum BeamProfile {
    Pencil,
    Gaussian { sigma: f64 },
    FlatTop { radius: f64 },
}

impl BeamProfile {
    /// `kind` is the UI's beam-profile selector value; `width` is sigma
    /// (Gaussian) or radius (FlatTop) in cm, ignored for Pencil. Unknown
    /// `kind` values fall back to Pencil rather than erroring, since this
    /// only ever comes from the UI's own <select>, never free-form input.
    pub fn from_params(kind: &str, width: f64) -> BeamProfile {
        match kind {
            "gaussian" => BeamProfile::Gaussian { sigma: width.max(1e-9) },
            "flattop" => BeamProfile::FlatTop { radius: width.max(1e-9) },
            _ => BeamProfile::Pencil,
        }
    }

    pub fn is_pencil(&self) -> bool {
        matches!(self, BeamProfile::Pencil)
    }

    /// This profile's 0-order Hankel transform at transverse spatial
    /// frequency `s` — see this module's doc comment. 1.0 for Pencil at
    /// every s (a point source excites every mode equally).
    pub fn spectral_factor(&self, s: f64) -> f64 {
        match *self {
            BeamProfile::Pencil => 1.0,
            BeamProfile::Gaussian { sigma } => (-0.5 * s * s * sigma * sigma).exp(),
            BeamProfile::FlatTop { radius } => {
                let x = s * radius;
                if x < 1e-6 {
                    // 2*J1(x)/x -> 1 - x^2/8 as x -> 0 (next Taylor term of
                    // J1); avoids the 0/0 at x == 0 from the plain formula.
                    1.0 - x * x / 8.0
                } else {
                    2.0 * j1(x) / x
                }
            }
        }
    }

    /// Real-space power density at transverse offset `rho` from the beam
    /// axis, normalized so its integral over the plane is 1. Undefined for
    /// Pencil (a true delta function) — callers must branch on is_pencil()
    /// before reaching for this; only convolve_radial below does.
    fn radial_weight(&self, rho: f64) -> f64 {
        match *self {
            BeamProfile::Pencil => 0.0,
            BeamProfile::Gaussian { sigma } => {
                (-0.5 * rho * rho / (sigma * sigma)).exp() / (2.0 * std::f64::consts::PI * sigma * sigma)
            }
            BeamProfile::FlatTop { radius } => {
                if rho <= radius { 1.0 / (std::f64::consts::PI * radius * radius) } else { 0.0 }
            }
        }
    }

    /// Radius beyond which radial_weight is negligible (Gaussian) or exactly
    /// zero (flat-top) — the beam's numerical footprint for convolve_radial.
    fn footprint_radius(&self) -> f64 {
        match *self {
            BeamProfile::Pencil => 0.0,
            BeamProfile::Gaussian { sigma } => 6.0 * sigma, // exp(-0.5*6^2) ~ 1.5e-8 of peak
            BeamProfile::FlatTop { radius } => radius,
        }
    }

    /// Public wrapper for validity checks elsewhere (e.g. comparing this
    /// against liemert_kienle.rs's cylinder radius) — the field itself stays
    /// private since only convolve_radial needs the unrounded value.
    pub fn extent(&self) -> f64 {
        self.footprint_radius()
    }
}

const N_RHO_QUAD: usize = 24;
const N_THETA_QUAD: usize = 16;

/// Builds an nx*ny*nz (phi, abs) volume from any axisymmetric fluence
/// kernel `kernel_at(rho, z)` (per unit p0) by evaluating it on a coalesced
/// (rho, z) table and bilinearly interpolating (in rho only — z is exact)
/// out to every voxel, then scaling by p0 and by `mua_at(z)` for the
/// absorption channel. Shared by fpw1992.rs's finite-beam path and every
/// call in liemert_kienle.rs, since both reduce to exactly this shape once
/// you have Phi(rho, z) in hand — whether that kernel is a Bessel series, a
/// beam convolution, or (in principle) anything else axisymmetric.
pub fn sample_axisymmetric_volume(
    lx: f64,
    ly: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    dz: f64,
    p0: f64,
    mua_at: impl Fn(f64) -> f64,
    kernel_at: impl Fn(f64, f64) -> f64,
) -> (Vec<f32>, Vec<f32>) {
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;
    let xs = lx / 2.0;
    let ys = ly / 2.0;

    let n_rho = nx.max(ny).max(2);
    let rho_max = xs.max(lx - xs).hypot(ys.max(ly - ys)) * 1.0001;
    let rho_step = rho_max / (n_rho - 1) as f64;

    let mut table = vec![0f64; n_rho * nz];
    for iz in 0..nz {
        let z = (iz as f64 + 0.5) * dz;
        for ir in 0..n_rho {
            let rho = ir as f64 * rho_step;
            table[ir * nz + iz] = kernel_at(rho, z);
        }
    }

    let n = nx * ny * nz;
    let mut phi = vec![0f32; n];
    let mut abs = vec![0f32; n];

    for ix in 0..nx {
        let x = (ix as f64 + 0.5) * dx;
        let rx2 = (x - xs) * (x - xs);
        for iy in 0..ny {
            let y = (iy as f64 + 0.5) * dy;
            let rho = (rx2 + (y - ys) * (y - ys)).sqrt();

            let rf = (rho / rho_step).min((n_rho - 1) as f64);
            let ir0 = rf.floor() as usize;
            let ir1 = (ir0 + 1).min(n_rho - 1);
            let frac = rf - ir0 as f64;

            for iz in 0..nz {
                let z = (iz as f64 + 0.5) * dz;
                let phi_lo = table[ir0 * nz + iz];
                let phi_hi = table[ir1 * nz + iz];
                let val = p0 * (phi_lo + (phi_hi - phi_lo) * frac);
                let idx = ix + iy * nx + iz * nx * ny;
                phi[idx] = val as f32;
                abs[idx] = (mua_at(z) * val) as f32;
            }
        }
    }

    (phi, abs)
}

/// Convolves a point-source kernel `point_kernel(d, z)` (d = transverse
/// distance from the point source to the field point) with this beam's
/// radial profile, at field point (rho, z) — rho being the field point's own
/// transverse distance from the beam axis. Generic in the kernel so both
/// FPW1992's closed form and (in principle) any other translation-invariant
/// point-source model can reuse it; only Liemert-Kienle skips it in favor of
/// the cheaper, exact spectral_factor approach above.
///
/// Composite midpoint rule in rho' (the integrand is smooth on [0, r_cut] —
/// flat-top's cutoff is the integration limit itself, not a discontinuity
/// inside it) and in theta' (periodic, so a plain evenly-spaced sum is the
/// trapezoidal rule). N_RHO_QUAD/N_THETA_QUAD are a hand-picked prototype
/// accuracy/cost tradeoff, not the output of a convergence study — raise
/// them if a narrow, sharply-peaked beam needs finer resolution.
pub fn convolve_radial(profile: &BeamProfile, rho: f64, z: f64, point_kernel: impl Fn(f64, f64) -> f64) -> f64 {
    let r_cut = profile.footprint_radius();
    let dr = r_cut / N_RHO_QUAD as f64;
    let dtheta = 2.0 * std::f64::consts::PI / N_THETA_QUAD as f64;

    let mut sum = 0.0f64;
    for i in 0..N_RHO_QUAD {
        let rp = (i as f64 + 0.5) * dr;
        let radial_weight = profile.radial_weight(rp) * rp * dr;

        let mut angular_sum = 0.0f64;
        for j in 0..N_THETA_QUAD {
            let theta = (j as f64 + 0.5) * dtheta;
            let d = (rho * rho + rp * rp - 2.0 * rho * rp * theta.cos()).max(0.0).sqrt();
            angular_sum += point_kernel(d, z);
        }
        sum += radial_weight * angular_sum * dtheta;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectral_factor_matches_point_source_as_beam_narrows() {
        for s in [0.5, 2.0, 10.0] {
            let gaussian = BeamProfile::Gaussian { sigma: 1e-6 };
            let flattop = BeamProfile::FlatTop { radius: 1e-6 };
            assert!((gaussian.spectral_factor(s) - 1.0).abs() < 1e-6);
            assert!((flattop.spectral_factor(s) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn flattop_spectral_factor_continuous_at_small_x() {
        let profile = BeamProfile::FlatTop { radius: 1.0 };
        let just_below = profile.spectral_factor(1e-6 - 1e-9);
        let just_above = profile.spectral_factor(1e-6 + 1e-9);
        assert!((just_below - just_above).abs() < 1e-9);
    }

    /// Loose tolerance: this deliberately-coarse composite midpoint
    /// quadrature (see convolve_radial's doc comment) has a few tenths of a
    /// percent of error at these knob settings — this test is a sanity
    /// check on the normalization, not a convergence study.
    #[test]
    fn radial_weight_integrates_to_one() {
        for profile in [BeamProfile::Gaussian { sigma: 0.05 }, BeamProfile::FlatTop { radius: 0.05 }] {
            let total = convolve_radial(&profile, 0.0, 0.0, |_d, _z| 1.0);
            assert!((total - 1.0).abs() < 5e-3, "total = {total}");
        }
    }

    #[test]
    fn convolve_radial_narrow_beam_matches_point_kernel() {
        let point_kernel = |d: f64, _z: f64| (-d).exp();
        let narrow = BeamProfile::Gaussian { sigma: 1e-4 };
        for rho in [0.0, 0.1, 0.5] {
            let got = convolve_radial(&narrow, rho, 0.0, point_kernel);
            let want = point_kernel(rho, 0.0);
            assert!((got - want).abs() < 5e-3, "rho={rho}: got {got}, want {want}");
        }
    }
}

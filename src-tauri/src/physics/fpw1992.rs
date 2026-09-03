//! Diffusion approximation — pencil beam, two-source method.
//! Reference: Farrell, Patterson & Wilson, Med. Phys. 19(4) 1992.
//! Ported from the original TypeScript at src/physics/fpw1992.ts (now removed —
//! this is the single source of truth for the physics).

use crate::physics::beam::{self, BeamProfile};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct Fpw1992Params {
    pub mua: f64,
    pub mus: f64,
    pub g: f64,
    pub n: f64,
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
pub struct Fpw1992Derived {
    pub musp: f64,
    #[serde(rename = "D")]
    pub d: f64,
    pub mueff: f64,
    pub delta: f64, // penetration depth
}

#[derive(Serialize)]
pub struct ValidityResult {
    pub valid: bool,
    pub reasons: Vec<String>,
}

/// Scalar quantities derivable from the optical properties alone (O(1)) —
/// shared by both the summary and volume commands so this is written once.
pub fn derived(p: &Fpw1992Params) -> Fpw1992Derived {
    let musp = p.mus * (1.0 - p.g);
    let mut_ = p.mua + musp;
    let d = 1.0 / (3.0 * mut_);
    let mueff = (3.0 * p.mua * mut_).sqrt();
    Fpw1992Derived {
        musp,
        d,
        mueff,
        delta: 1.0 / mueff,
    }
}

/// The diffusion approximation replaces the full radiative transport equation
/// with its lowest-order (P1) angular expansion — only accurate once light has
/// scattered enough times to become nearly isotropic before it's absorbed. See
/// the original TS file's comments (git history) for the full derivation notes
/// behind each of these three checks.
pub fn check_validity(p: &Fpw1992Params, derived: &Fpw1992Derived) -> ValidityResult {
    let ratio = derived.musp / p.mua;
    let min_dim = p.lx.min(p.ly).min(p.lz);
    let mfp_prime = 1.0 / (p.mua + derived.musp);
    let z0 = mfp_prime;

    let dx = p.lx / p.nx as f64;
    let dy = p.ly / p.ny as f64;
    let dz = p.lz / p.nz as f64;
    let max_voxel = dx.max(dy).max(dz);

    let mut reasons = Vec::new();

    if ratio < 10.0 {
        reasons.push(format!(
            "μ<sub>s</sub>'/μ<sub>a</sub> = {:.2} (want ≳10) — absorption is too strong \
             relative to scattering for light to randomize direction before being absorbed",
            ratio
        ));
    }
    if mfp_prime > 0.5 * min_dim {
        reasons.push(format!(
            "transport mean free path ({:.3} cm) is ≳half the shortest slab edge \
             ({:.3} cm) — the medium isn't large enough for a photon to scatter \
             many times before reaching a boundary, so the diffusion (multiple-scattering) \
             assumption breaks down",
            mfp_prime, min_dim
        ));
    }
    if max_voxel > 0.5 * z0 {
        reasons.push(format!(
            "voxel size (up to {:.3} cm) is ≳half the source depth z<sub>0</sub> \
             ({:.3} cm) where fluence peaks and varies fastest — the grid is too coarse \
             to resolve that peak, so results near the source will be smeared out. \
             Increase N<sub>x</sub>/N<sub>y</sub>/N<sub>z</sub> or shrink the domain",
            max_voxel, z0
        ));
    }

    ValidityResult {
        valid: reasons.is_empty(),
        reasons,
    }
}

/// The expensive part: fills phi/abs for every voxel. Math is done in f64
/// throughout (matches the original precision); only the final write into the
/// output buffers casts down to f32 — plenty for a log-scale color plot, and it
/// halves the IPC payload size versus sending f64 end-to-end.
pub fn compute_volume(p: &Fpw1992Params, d: &Fpw1992Derived) -> (Vec<f32>, Vec<f32>) {
    let reff = -1.44 / (p.n * p.n) + 0.71 / p.n + 0.668 + 0.0636 * p.n;
    let a = (1.0 + reff) / (1.0 - reff);
    let zb = 2.0 * a * d.d;

    let mut_ = p.mua + d.musp;
    let z0 = 1.0 / mut_;
    let zs_real = z0;
    let zs_img = -(z0 + 2.0 * zb);

    let xs = p.lx / 2.0;
    let ys = p.ly / 2.0;

    let dx = p.lx / p.nx as f64;
    let dy = p.ly / p.ny as f64;
    let dz = p.lz / p.nz as f64;

    let n = p.nx * p.ny * p.nz;
    let mut phi = vec![0f32; n];
    let mut abs = vec![0f32; n];

    let eps = 1e-9;
    let four_pi_d = 4.0 * std::f64::consts::PI * d.d;

    let beam = BeamProfile::from_params(&p.beam_profile, p.beam_width);

    if beam.is_pencil() {
        for ix in 0..p.nx {
            let x = (ix as f64 + 0.5) * dx;
            let dx2 = (x - xs) * (x - xs);

            for iy in 0..p.ny {
                let y = (iy as f64 + 0.5) * dy;
                let rxy2 = dx2 + (y - ys) * (y - ys);

                for iz in 0..p.nz {
                    let z = (iz as f64 + 0.5) * dz;

                    let r1 = (rxy2 + (z - zs_real) * (z - zs_real)).sqrt();
                    let r2 = (rxy2 + (z - zs_img) * (z - zs_img)).sqrt();

                    let g1 = if r1 > eps {
                        (-d.mueff * r1).exp() / (four_pi_d * r1)
                    } else {
                        0.0
                    };
                    let g2 = if r2 > eps {
                        (-d.mueff * r2).exp() / (four_pi_d * r2)
                    } else {
                        0.0
                    };

                    let val = p.p0 * (g1 - g2).max(0.0);
                    let idx = ix + iy * p.nx + iz * p.nx * p.ny;
                    phi[idx] = val as f32;
                    abs[idx] = (p.mua * val) as f32;
                }
            }
        }
        return (phi, abs);
    }

    // Finite beam: the same point-source kernel, convolved (beam::convolve_radial)
    // with the beam's transverse profile — see beam.rs for why this is the
    // right way to generalize a point-source model to a finite beam. That
    // convolution is a 2-D numerical integral per (rho, z) point, too
    // expensive to redo per voxel, so it's sampled via
    // beam::sample_axisymmetric_volume — the same coalesced-(rho,z)-table-
    // plus-bilinear-interpolation trick liemert_kienle.rs uses for its own
    // (much more expensive) series sum, factored out so both models share
    // one implementation of the table/interpolation bookkeeping.
    let point_kernel = |rho: f64, z: f64| -> f64 {
        let r1 = (rho * rho + (z - zs_real) * (z - zs_real)).sqrt();
        let r2 = (rho * rho + (z - zs_img) * (z - zs_img)).sqrt();
        let g1 = if r1 > eps { (-d.mueff * r1).exp() / (four_pi_d * r1) } else { 0.0 };
        let g2 = if r2 > eps { (-d.mueff * r2).exp() / (four_pi_d * r2) } else { 0.0 };
        (g1 - g2).max(0.0)
    };

    beam::sample_axisymmetric_volume(p.lx, p.ly, p.nx, p.ny, p.nz, dz, p.p0, |_z| p.mua, |rho, z| {
        beam::convolve_radial(&beam, rho, z, point_kernel)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params(beam_profile: &str, beam_width: f64) -> Fpw1992Params {
        Fpw1992Params {
            mua: 0.1, mus: 100.0, g: 0.9, n: 1.4, p0: 1.0,
            beam_profile: beam_profile.to_string(), beam_width,
            lx: 2.0, ly: 2.0, lz: 2.0,
            nx: 20, ny: 20, nz: 20,
        }
    }

    #[test]
    fn finite_beam_volume_is_finite_and_positive() {
        for (profile, width) in [("gaussian", 0.05), ("flattop", 0.05)] {
            let params = base_params(profile, width);
            let d = derived(&params);
            let (phi, abs) = compute_volume(&params, &d);
            for (i, (&v, &a)) in phi.iter().zip(abs.iter()).enumerate() {
                assert!(v.is_finite() && v >= 0.0, "{profile}: phi[{i}] = {v}");
                assert!(a.is_finite() && a >= 0.0, "{profile}: abs[{i}] = {a}");
            }
        }
    }

    /// A beam far narrower than the grid's voxel size should reproduce the
    /// pencil-beam result closely — the same cross-check idea as
    /// liemert_kienle.rs's narrow_gaussian_matches_pencil.
    #[test]
    fn narrow_gaussian_matches_pencil() {
        let pencil = base_params("pencil", 0.0);
        let narrow = base_params("gaussian", 1e-4);

        let dp = derived(&pencil);
        let dn = derived(&narrow);
        let (phi_pencil, _) = compute_volume(&pencil, &dp);
        let (phi_narrow, _) = compute_volume(&narrow, &dn);

        let mut max_rel_err = 0.0f64;
        for (&a, &b) in phi_pencil.iter().zip(phi_narrow.iter()) {
            if a > 1e-6 {
                max_rel_err = max_rel_err.max(((a - b) as f64 / a as f64).abs());
            }
        }
        assert!(max_rel_err < 0.05, "max_rel_err = {max_rel_err:.4}");
    }
}

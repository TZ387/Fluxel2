//! Diffusion approximation — pencil beam, two-source method.
//! Reference: Farrell, Patterson & Wilson, Med. Phys. 19(4) 1992.
//! Ported from the original TypeScript at src/physics/fpw1992.ts (now removed —
//! this is the single source of truth for the physics).

use crate::physics::beam::{self, BeamPattern, BeamProfile, Grid};
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
    /// "single" | "line" | "grid" — see beam.rs. `pattern_count` is spots
    /// along the line or per side of the grid, `pattern_spacing` the pitch
    /// between neighbours in cm; both ignored for "single".
    pub beam_pattern: String,
    pub pattern_count: usize,
    pub pattern_spacing: f64,
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
    /// How many spots the chosen beam pattern works out to.
    pub spots: usize,
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
        spots: BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing).len(),
    }
}

/// The diffusion approximation is the transport equation's lowest-order (P1)
/// angular expansion — accurate only once light has scattered enough to
/// become nearly isotropic before being absorbed. See the original TS
/// file's comments (git history) for the derivation behind each check.
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

    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);
    if let Some(reason) = beam::pattern_extent_warning(&pattern, p.lx, p.ly) {
        reasons.push(reason);
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
    let pattern = BeamPattern::from_params(&p.beam_pattern, p.pattern_count, p.pattern_spacing);

    // One pencil spot has a closed form per voxel with no radial
    // interpolation at all, so it keeps its own exact path; anything wider or
    // repeated goes through the shared (rho, z) table.
    if beam.is_pencil() && pattern.is_single() {
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

    // Anything else: the same point-source kernel, convolved with the beam's
    // transverse profile where it has one (beam::convolve_radial — see
    // beam.rs for why), sampled onto a coalesced (rho, z) table and summed
    // over the pattern's spots. That table is what makes a finite beam (too
    // expensive to reconvolve per voxel) and a many-spot pattern both cheap.
    let point_kernel = |rho: f64, z: f64| -> f64 {
        let r1 = (rho * rho + (z - zs_real) * (z - zs_real)).sqrt();
        let r2 = (rho * rho + (z - zs_img) * (z - zs_img)).sqrt();
        let g1 = if r1 > eps { (-d.mueff * r1).exp() / (four_pi_d * r1) } else { 0.0 };
        let g2 = if r2 > eps { (-d.mueff * r2).exp() / (four_pi_d * r2) } else { 0.0 };
        (g1 - g2).max(0.0)
    };

    let grid = Grid { lx: p.lx, ly: p.ly, nx: p.nx, ny: p.ny, nz: p.nz, dz };
    beam::sample_axisymmetric_volume(&grid, &pattern, p.p0, |_z| p.mua, |rho, z| {
        if beam.is_pencil() { point_kernel(rho, z) } else { beam::convolve_radial(&beam, rho, z, point_kernel) }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_params(beam_profile: &str, beam_width: f64) -> Fpw1992Params {
        Fpw1992Params {
            mua: 0.1, mus: 100.0, g: 0.9, n: 1.4, p0: 1.0,
            beam_profile: beam_profile.to_string(), beam_width,
            beam_pattern: "single".to_string(), pattern_count: 1, pattern_spacing: 0.0,
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

    /// A line pattern has to land where it says it does: mirror-symmetric
    /// about both axes, and — since it spreads the same P0 along x — wider in
    /// x than in y at the depth where the sources sit.
    #[test]
    fn line_pattern_is_centred_and_oriented() {
        let mut params = base_params("pencil", 0.0);
        params.nx = 41;
        params.ny = 41;
        params.nz = 20;
        params.beam_pattern = "line".to_string();
        params.pattern_count = 5;
        params.pattern_spacing = 0.2;

        let d = derived(&params);
        assert_eq!(d.spots, 5);
        let (phi, _) = compute_volume(&params, &d);
        let at = |ix: usize, iy: usize, iz: usize| phi[ix + iy * 41 + iz * 41 * 41] as f64;

        // Odd counts put a voxel dead centre; the pattern is centred there.
        let (cx, cy, iz) = (20usize, 20usize, 1usize);
        for k in 1..=8 {
            let (xl, xr) = (at(cx - k, cy, iz), at(cx + k, cy, iz));
            let (yl, yr) = (at(cx, cy - k, iz), at(cx, cy + k, iz));
            assert!((xl - xr).abs() < 1e-5 * xl, "x mirror at k={k}: {xl} vs {xr}");
            assert!((yl - yr).abs() < 1e-5 * yl, "y mirror at k={k}: {yl} vs {yr}");
            // The line runs along x, so at equal distance the field is
            // stronger along it than across it.
            assert!(xl > yl, "k={k}: along-line {xl} should exceed across-line {yl}");
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

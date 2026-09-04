//! Finite transverse beam profiles, convolved onto a point-source model.
//!
//! Every point-source model here (fpw1992.rs, liemert_kienle.rs) reduces to
//! a kernel Phi_pt(rho, z) — never azimuth, since a point source is
//! axisymmetric. A finite beam replaces the point source with a radially
//! symmetric power density S(rho'), normalized so integral(S dA) = 1 (so p0
//! keeps meaning "total input power" regardless of beam shape). The
//! diffusion equation is linear, so the resulting fluence is just the 2-D
//! convolution of Phi_pt with S:
//!
//!   Phi_beam(rho_vec, z) = integral[ S(rho'_vec) * Phi_pt(|rho_vec - rho'_vec|, z) d2rho' ]
//!
//! — itself radially symmetric (convolution of two radially-symmetric
//! functions), so still just a function of (rho, z).
//!
//! Two ways to evaluate that convolution, picked per model to match how each
//! already expresses Phi_pt:
//! - Liemert-Kienle expands Phi_pt as a sum over spatial frequencies
//!   s_n = (zeros of J0)/a'. A point source excites every mode with weight 1
//!   (a delta function's spectrum is flat); swapping in S(rho') just
//!   multiplies each mode by S's own 0-order Hankel transform
//!   (spectral_factor below) — exact as long as the beam sits well inside
//!   the artificial cylinder wall (see cylinder_radius in liemert_kienle.rs).
//! - FPW1992 has no such series (closed-form real+image dipole for a
//!   translation-invariant plane), so its convolution is evaluated directly
//!   as a 2-D numerical integral over the beam's footprint (convolve_radial).
//!
//! Gaussian and flat-top's 0-order Hankel transforms are standard closed
//! forms — Gaussian: exp(-s^2 sigma^2/2); flat-top disk of radius R:
//! 2*J1(sR)/(sR) — both -> 1 as the beam narrows, recovering the point
//! source (a useful sanity check on the algebra).
//!
//! Where the beam is *aimed* is separate from its shape: a BeamPattern is a
//! list of spot positions on the top face — one spot, a row laid down by a
//! scanner, or the array a fractional handpiece delivers. Diffusion is
//! linear, so a pattern's fluence is the sum of its spots', each carrying an
//! equal share of P0. Since Phi_pt is the same function for every spot, only
//! shifted, the expensive part (Liemert-Kienle's series, FPW1992's
//! convolution) is still evaluated exactly once — a pattern only costs extra
//! table lookups per voxel, not extra kernel evaluations.

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

/// Where the beam is aimed on the top face: the offsets, from the centre of
/// that face, of every spot the pattern delivers. See this module's doc
/// comment for why a whole pattern costs no more kernel evaluations than one
/// spot does.
#[derive(Clone)]
pub struct BeamPattern {
    spots: Vec<(f64, f64)>,
}

impl BeamPattern {
    /// `kind` is the UI's beam-pattern selector value; `count` is the number
    /// of spots along a line, or per side of a grid (so a grid has count^2),
    /// and `spacing` is the pitch between neighbours in cm. Both are ignored
    /// for "single", and an unknown `kind` falls back to it — same reasoning
    /// as BeamProfile::from_params.
    ///
    /// A line is what a scanner actually lays down: a row of discrete pulses,
    /// which approximates a continuous sweep once the pitch is small next to
    /// the beam width.
    pub fn from_params(kind: &str, count: usize, spacing: f64) -> BeamPattern {
        let n = count.max(1);
        let offset = |i: usize| (i as f64 - (n - 1) as f64 / 2.0) * spacing;
        let spots = match kind {
            "line" => (0..n).map(|i| (offset(i), 0.0)).collect(),
            "grid" => (0..n).flat_map(|iy| (0..n).map(move |ix| (offset(ix), offset(iy)))).collect(),
            _ => vec![(0.0, 0.0)],
        };
        BeamPattern { spots }
    }

    pub fn single() -> BeamPattern {
        BeamPattern { spots: vec![(0.0, 0.0)] }
    }

    pub fn len(&self) -> usize {
        self.spots.len()
    }

    pub fn is_single(&self) -> bool {
        self.spots.len() == 1
    }

    /// Half-widths of the pattern's bounding box, for validity checks.
    pub fn half_extent(&self) -> (f64, f64) {
        self.spots.iter().fold((0.0f64, 0.0f64), |(mx, my), &(x, y)| (mx.max(x.abs()), my.max(y.abs())))
    }
}

/// The largest distance from any spot to any corner of the grid's footprint —
/// how far out the axisymmetric kernel has to be evaluated, and (for
/// liemert_kienle.rs) how far out its artificial cylinder wall has to sit.
/// Reduces to half the grid's diagonal for a single centred spot.
pub fn max_kernel_radius(lx: f64, ly: f64, pattern: &BeamPattern) -> f64 {
    let (hx, hy) = pattern.half_extent();
    (hx + lx / 2.0).hypot(hy + ly / 2.0)
}

/// Shared by both point-source models' check_validity: spots outside the
/// grid's footprint still light it up, but the user can't see them.
pub fn pattern_extent_warning(pattern: &BeamPattern, lx: f64, ly: f64) -> Option<String> {
    let (hx, hy) = pattern.half_extent();
    if hx <= lx / 2.0 && hy <= ly / 2.0 {
        return None;
    }
    Some(format!(
        "the beam pattern reaches {:.3} x {:.3} cm from the centre, past the grid's half-width \
         ({:.3} x {:.3} cm) — the spots that fall outside still deposit light into the volume, \
         but you can't see them; enlarge L<sub>x</sub>/L<sub>y</sub>, or reduce the spot \
         spacing or count",
        hx, hy, lx / 2.0, ly / 2.0
    ))
}

/// The voxel grid a volume is sampled onto — bundled rather than passed as a
/// long run of positional numbers, since both models carry the same fields.
pub struct Grid {
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dz: f64,
}

const N_RHO_QUAD: usize = 24;
const N_THETA_QUAD: usize = 16;

/// Builds an nx*ny*nz (phi, abs) volume from an axisymmetric fluence kernel
/// `kernel_at(rho, z)` (per unit p0): evaluate it on a coalesced (rho, z)
/// table, then, for every voxel, sum one interpolated lookup per spot in the
/// pattern, scaling by each spot's share of p0 and by `mua_at(z)` for the
/// absorption channel. Shared by fpw1992.rs's finite-beam path and
/// liemert_kienle.rs, since both reduce to this shape once Phi(rho, z) is in
/// hand.
///
/// The table is built once no matter how many spots there are — only the
/// lookups repeat — and the per-spot radial index is worked out per (x, y)
/// column rather than per voxel, so the innermost loop is a contiguous
/// accumulate over z that vectorizes.
pub fn sample_axisymmetric_volume(
    grid: &Grid,
    pattern: &BeamPattern,
    p0: f64,
    mua_at: impl Fn(f64) -> f64,
    kernel_at: impl Fn(f64, f64) -> f64,
) -> (Vec<f32>, Vec<f32>) {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let dx = grid.lx / nx as f64;
    let dy = grid.ly / ny as f64;
    let xs = grid.lx / 2.0;
    let ys = grid.ly / 2.0;

    // Radial resolution is set by the grid alone, so a pattern that reaches
    // further only lengthens the table, never coarsens it.
    let rho_max = max_kernel_radius(grid.lx, grid.ly, pattern) * 1.0001;
    let rho_step = max_kernel_radius(grid.lx, grid.ly, &BeamPattern::single()) * 1.0001
        / (nx.max(ny).max(2) - 1) as f64;
    let n_rho = (rho_max / rho_step).ceil() as usize + 1;

    let mut table = vec![0f64; n_rho * nz];
    let mut mua_z = vec![0f64; nz];
    for iz in 0..nz {
        let z = (iz as f64 + 0.5) * grid.dz;
        mua_z[iz] = mua_at(z);
        for ir in 0..n_rho {
            table[ir * nz + iz] = kernel_at(ir as f64 * rho_step, z);
        }
    }

    let n = nx * ny * nz;
    let mut phi = vec![0f32; n];
    let mut abs = vec![0f32; n];

    // p0 stays the pattern's *total* power, so the spots share it out.
    let share = p0 / pattern.len() as f64;
    let mut col = vec![0f64; nz];

    for ix in 0..nx {
        let x = (ix as f64 + 0.5) * dx - xs;
        for iy in 0..ny {
            let y = (iy as f64 + 0.5) * dy - ys;

            col.fill(0.0);
            for &(sx, sy) in &pattern.spots {
                let rho = (x - sx).hypot(y - sy);
                let rf = (rho / rho_step).min((n_rho - 1) as f64);
                let ir0 = rf.floor() as usize;
                let ir1 = (ir0 + 1).min(n_rho - 1);
                let frac = rf - ir0 as f64;

                let lo = &table[ir0 * nz..ir0 * nz + nz];
                let hi = &table[ir1 * nz..ir1 * nz + nz];
                for iz in 0..nz {
                    col[iz] += lo[iz] + (hi[iz] - lo[iz]) * frac;
                }
            }

            for iz in 0..nz {
                let val = share * col[iz];
                let idx = ix + iy * nx + iz * nx * ny;
                phi[idx] = val as f32;
                abs[idx] = (mua_z[iz] * val) as f32;
            }
        }
    }

    (phi, abs)
}

/// Convolves a point-source kernel `point_kernel(d, z)` (d = distance from
/// the point source to the field point) with this beam's radial profile, at
/// field point (rho, z). Generic in the kernel so any translation-invariant
/// point-source model can reuse it — only Liemert-Kienle skips it, in favor
/// of the cheaper, exact spectral_factor approach above.
///
/// Composite midpoint rule in rho' and theta' (the integrand is smooth —
/// flat-top's cutoff is the integration limit itself, not a discontinuity
/// inside it). N_RHO_QUAD/N_THETA_QUAD are hand-picked, not from a
/// convergence study — raise them for a narrow, sharply-peaked beam.
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

    fn test_grid() -> Grid {
        Grid { lx: 2.0, ly: 2.0, nx: 8, ny: 8, nz: 4, dz: 0.25 }
    }

    #[test]
    fn patterns_are_centred_and_counted() {
        assert_eq!(BeamPattern::from_params("single", 5, 0.2).len(), 1);
        assert_eq!(BeamPattern::from_params("line", 5, 0.2).len(), 5);
        assert_eq!(BeamPattern::from_params("grid", 4, 0.2).len(), 16);
        assert_eq!(BeamPattern::from_params("nonsense", 4, 0.2).len(), 1);

        // An even count straddles the centre, an odd one sits on it; either
        // way the pattern is centred, so its spots sum to zero.
        for n in [3usize, 4] {
            let line = BeamPattern::from_params("line", n, 0.2);
            let sum: f64 = line.spots.iter().map(|&(x, _)| x).sum();
            assert!(sum.abs() < 1e-12, "n={n}: spots off-centre by {sum}");
        }

        // (n-1)/2 pitches either side of centre, and a line has no y extent.
        let (hx, hy) = BeamPattern::from_params("line", 4, 0.2).half_extent();
        assert!((hx - 0.3).abs() < 1e-12 && hy == 0.0, "line: {hx}, {hy}");
        assert_eq!(BeamPattern::from_params("grid", 3, 0.5).half_extent(), (0.5, 0.5));
    }

    #[test]
    fn max_kernel_radius_reaches_the_far_corner() {
        // Single centred spot: half the grid's diagonal, as before patterns.
        let single = max_kernel_radius(2.0, 2.0, &BeamPattern::single());
        assert!((single - 2f64.sqrt()).abs() < 1e-12, "got {single}");
        // A spot 0.5 off-centre in x is that much further from the far corner.
        let line = max_kernel_radius(2.0, 2.0, &BeamPattern::from_params("line", 2, 1.0));
        assert!((line - 1.5f64.hypot(1.0)).abs() < 1e-12, "got {line}");
    }

    /// Zero pitch stacks every spot on the axis, so splitting P0 between them
    /// has to add back up to exactly the single-spot answer.
    #[test]
    fn coincident_spots_match_a_single_spot() {
        let kernel = |rho: f64, z: f64| (-2.0 * rho).exp() / (z + 0.1);
        let grid = test_grid();
        let (one, _) = sample_axisymmetric_volume(&grid, &BeamPattern::single(), 1.0, |_z| 0.0, kernel);
        let stacked = BeamPattern::from_params("grid", 4, 0.0);
        assert_eq!(stacked.len(), 16);
        let (many, _) = sample_axisymmetric_volume(&grid, &stacked, 1.0, |_z| 0.0, kernel);
        for (i, (&a, &b)) in one.iter().zip(many.iter()).enumerate() {
            assert!((a - b).abs() <= 1e-6 * a.abs().max(1e-6), "voxel {i}: {a} vs {b}");
        }
    }

    /// The pattern sum itself, against the explicit superposition it stands
    /// for. The kernel is linear in rho so the table's interpolation is exact
    /// and any difference is the summing machinery, not sampling error.
    #[test]
    fn sample_axisymmetric_volume_superposes_spots() {
        let kernel = |rho: f64, z: f64| 1.0 + 0.5 * rho + z;
        let grid = test_grid();
        let pattern = BeamPattern::from_params("line", 3, 0.3);
        let p0 = 2.0;
        let (phi, abs) = sample_axisymmetric_volume(&grid, &pattern, p0, |_z| 0.25, kernel);

        let share = p0 / pattern.len() as f64;
        for ix in 0..grid.nx {
            for iy in 0..grid.ny {
                for iz in 0..grid.nz {
                    let x = (ix as f64 + 0.5) * grid.lx / grid.nx as f64 - grid.lx / 2.0;
                    let y = (iy as f64 + 0.5) * grid.ly / grid.ny as f64 - grid.ly / 2.0;
                    let z = (iz as f64 + 0.5) * grid.dz;
                    let want: f64 = pattern
                        .spots
                        .iter()
                        .map(|&(sx, sy)| share * kernel((x - sx).hypot(y - sy), z))
                        .sum();
                    let idx = ix + iy * grid.nx + iz * grid.nx * grid.ny;
                    assert!((phi[idx] as f64 - want).abs() < 1e-6 * want.abs(), "phi at {idx}");
                    assert!((abs[idx] as f64 - 0.25 * want).abs() < 1e-6 * want.abs(), "abs at {idx}");
                }
            }
        }
    }

    #[test]
    fn pattern_extent_warning_fires_only_outside_the_grid() {
        let inside = BeamPattern::from_params("line", 3, 0.2); // reaches 0.2 cm
        assert!(pattern_extent_warning(&inside, 2.0, 2.0).is_none());
        let outside = BeamPattern::from_params("line", 3, 2.0); // reaches 2.0 cm
        assert!(pattern_extent_warning(&outside, 2.0, 2.0).is_some());
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

//! Checks shared between models' check_validity(), of two kinds — kept
//! apart on purpose, because they say opposite things about the answer.
//!
//! `require` is the hard kind: not "this is approximate" but "there is no
//! answer". A negative absorption coefficient, a g of exactly 1, a zero
//! thickness — each leaves the model's own derived coefficients undefined,
//! and the volume comes back all-NaN, which the viewer renders as a blank
//! plot with no hint of why. So each model runs these first and, if any
//! fires, reports only these: the physics warnings below them would be
//! computed from the same broken numbers and would only add noise.
//!
//! The two `*_warning` functions are the soft kind, and specifically the
//! ones both diffusion models (fpw1992.rs, liemert_kienle.rs) ask in the
//! same words — one of a single medium, the other of every layer in a
//! stack. They live here for the reason boundary.rs's extrapolation length
//! does: two copies of a paragraph drift apart, and these two models are
//! meant to be read against each other. They return an Option rather than
//! pushing, matching beam.rs's pattern_extent_warning. Monte Carlo shares
//! neither — it has no approximation to justify.

use serde::Serialize;

/// What check_validity returns, the same shape for every model: whether the
/// inputs are physically usable at all, and — if not — why. Declared once
/// here rather than per model, since the shape never varies even though what
/// goes into `reasons` does.
#[derive(Serialize)]
pub struct ValidityResult {
    pub valid: bool,
    pub reasons: Vec<String>,
}

/// Push a reason unless `ok`. `label` names the parameter (carrying a layer
/// prefix where a model has more than one layer), and `requirement` completes
/// the sentence "must be ...".
pub fn require(reasons: &mut Vec<String>, ok: bool, label: &str, requirement: &str, value: f64) {
    if ok {
        return;
    }
    reasons.push(format!(
        "{} = {:.3} — must be {}. A non-physical input leaves this model's own coefficients \
         undefined, so the volume below is meaningless (blank, or all-NaN) rather than merely \
         approximate",
        label, value, requirement
    ));
}

/// Least μ_s'/μ_a at which diffusion is worth trusting. A round number, and
/// a soft edge — hence "≳" in the message rather than a hard threshold.
const SCATTERING_DOMINANCE_MIN: f64 = 10.0;

/// Fraction of the source depth a voxel may span before the peak there is
/// being averaged away rather than resolved.
const SOURCE_RESOLUTION_FRACTION: f64 = 0.5;

/// Diffusion's central assumption: light has to scatter many times before
/// being absorbed, or it never becomes the nearly isotropic field the
/// approximation solves for. `at` names what the complaint is about
/// ("layer 2: " for a stack), and is empty for a model with one medium.
pub fn scattering_dominance_warning(at: &str, musp: f64, mua: f64) -> Option<String> {
    let ratio = musp / mua;
    if ratio >= SCATTERING_DOMINANCE_MIN {
        return None;
    }
    Some(format!(
        "{}μ<sub>s</sub>'/μ<sub>a</sub> = {:.2} (want ≳{:.0}) — absorption is too strong \
         relative to scattering for light to randomize direction before being absorbed",
        at, ratio, SCATTERING_DOMINANCE_MIN
    ))
}

/// Whether the grid resolves the depth where fluence peaks and varies
/// fastest. What z0 *is* differs between the two callers — one transport
/// mean free path for FPW1992, one scattering mean free path for
/// Liemert-Kienle — but the complaint, and the grid's side of it, do not.
pub fn source_resolution_warning(dx: f64, dy: f64, dz: f64, z0: f64) -> Option<String> {
    let max_voxel = dx.max(dy).max(dz);
    if max_voxel <= SOURCE_RESOLUTION_FRACTION * z0 {
        return None;
    }
    Some(format!(
        "voxel size (up to {:.3} cm) is ≳half the source depth z<sub>0</sub> \
         ({:.3} cm) where fluence peaks and varies fastest — the grid is too coarse \
         to resolve that peak, so results near the source will be smeared out. \
         Increase N<sub>x</sub>/N<sub>y</sub>/N<sub>z</sub> or shrink the domain",
        max_voxel, z0
    ))
}

/// Overlay code (0 invalid, 1 marginal, 2 valid) from a distance-to-nearest-
/// boundary / mean-free-path ratio — not a check_validity() input check like
/// the rest of this file, but the same 1/2-mfp heuristic fpw1992.rs's and
/// liemert_kienle.rs's compute_validity_volume both use for their per-voxel
/// overlays (see either's doc comment for why those thresholds).
pub fn mfp_ratio_code(ratio: f64) -> u8 {
    if ratio < 1.0 {
        0
    } else if ratio < 2.0 {
        1
    } else {
        2
    }
}

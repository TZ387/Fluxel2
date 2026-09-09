//! Parameter sanity checks shared by every model's check_validity().
//!
//! These are a different kind of warning from the rest. The others say the
//! answer is approximate; this one says there is no answer. A negative
//! absorption coefficient, a g of exactly 1, a zero thickness — each leaves
//! the model's own derived coefficients undefined, and the volume comes back
//! all-NaN, which the viewer renders as a blank plot with no hint of why.
//!
//! So each model runs these first and, if any fires, reports only these: the
//! physics warnings below them would be computed from the same broken numbers
//! and would only add noise.

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

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

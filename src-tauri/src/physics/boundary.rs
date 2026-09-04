//! The air-tissue boundary, shared by both point-source models.
//!
//! At a refractive-index step the surface reflects part of the diffusing light
//! back into the medium. The diffusion approximation accounts for that by
//! moving the zero-fluence plane off the physical surface out to z = -zb,
//! where the extrapolation length zb = 2*A*D and A measures how much comes
//! back.
//!
//! A comes from Groenhuis et al.'s empirical fit in n. fpw1992.rs and
//! liemert_kienle.rs both call this rather than each carrying a copy, so a
//! beam through one homogeneous medium meets the same surface whichever model
//! computes it — which is what makes the two comparable at all, and what
//! liemert_kienle.rs's homogeneous-limit test leans on.

/// Extrapolation length zb = 2*A*D for a medium of refractive index `n`
/// bounded by air. Each model passes its own diffusion coefficient `d`: the
/// two use different, equally standard conventions for it.
pub fn extrapolation_length(n: f64, d: f64) -> f64 {
    let reff = -1.44 / (n * n) + 0.71 / n + 0.668 + 0.0636 * n;
    let a = (1.0 + reff) / (1.0 - reff);
    2.0 * a * d
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With no index step there is nothing to reflect, so A → 1 and zb → 2D
    /// exactly. The fit is empirical and isn't constructed to satisfy that,
    /// so how close it lands is a real check on the constants.
    #[test]
    fn index_matched_boundary_approaches_two_d() {
        let zb = extrapolation_length(1.0, 1.0);
        assert!((zb - 2.0).abs() < 0.01, "zb = {zb}, want ≈ 2.0");
    }

    /// A bigger index step reflects more back in, pushing the zero-fluence
    /// plane further outside the surface.
    #[test]
    fn extrapolation_length_grows_with_index_step() {
        let d = 0.03;
        let mut prev = extrapolation_length(1.0, d);
        for n in [1.2, 1.33, 1.4, 1.55, 1.7] {
            let zb = extrapolation_length(n, d);
            assert!(zb > prev, "n = {n}: zb {zb} should exceed {prev}");
            prev = zb;
        }
    }
}

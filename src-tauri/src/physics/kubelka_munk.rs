//! Kubelka–Munk two-flux model, N homogeneous layers.
//! Reference: Kubelka, 1948.
//! Ported from the original TypeScript at src/physics/kubelkaMunk.ts (now
//! removed — this is the single source of truth for the physics). See that
//! file's git history for the full derivation notes behind each equation.

use crate::physics::validity::require;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Clone, Copy)]
pub struct KMLayerParams {
    pub mua: f64, // K, absorption coefficient
    pub mus: f64, // S, scattering coefficient
    pub thickness: f64,
}

#[derive(Deserialize)]
pub struct KubelkaMunkParams {
    pub lx: f64,
    pub ly: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub p0: f64,
    pub layers: Vec<KMLayerParams>,
}

#[derive(Serialize, Clone, Copy)]
pub struct KubelkaMunkDerived {
    #[serde(rename = "R_total")]
    pub r_total: f64,
    #[serde(rename = "T_total")]
    pub t_total: f64,
    #[serde(rename = "A_total")]
    pub a_total: f64,
    #[serde(rename = "Lz")]
    pub lz: f64,
}

#[derive(Serialize)]
pub struct ValidityResult {
    pub valid: bool,
    pub reasons: Vec<String>,
}

struct LayerAlone {
    a: f64,
    b: f64,
    gamma: f64,
    r: f64,
    t: f64,
}

/// Reflectance/transmittance of a single layer alone, black backing.
fn km_layer_alone(mua: f64, mus: f64, thickness: f64) -> LayerAlone {
    let a = 1.0 + mua / mus;
    let b = (a * a - 1.0).sqrt();
    let gamma = b * mus * thickness;
    let th = gamma.tanh();
    let sech = 1.0 / gamma.cosh(); // → 0 for large γ, never NaN/Inf
    let denom = a * th + b;
    let r = th / denom;
    let t = (b * sech) / denom;
    LayerAlone { a, b, gamma, r, t }
}

/// Evaluate I(ξ), J(ξ) inside one layer, ξ ∈ [0,1] = normalised depth (0 = top
/// of this layer, 1 = bottom), given the downward flux i0 entering the top and
/// the reflectance rg of everything below.
fn km_layer_profile(a: f64, b: f64, gamma: f64, i0: f64, rg: f64) -> impl Fn(f64) -> (f64, f64) {
    // k0 = (P/Q) already carries the e^{-2γ} decay, so every exponential
    // evaluated below has a non-positive argument — stable for any γ.
    let k0 = (rg - (a - b)) / (a + b - rg);
    let term_at = move |xi: f64| k0 * (2.0 * gamma * (xi - 1.0)).exp();
    let q = i0 / (term_at(0.0) + 1.0);
    move |xi: f64| {
        let term = term_at(xi);
        let exp_neg = (-gamma * xi).exp();
        let i = q * exp_neg * (term + 1.0);
        let j = q * exp_neg * ((a + b) * term + (a - b));
        (i, j)
    }
}

struct LayerData {
    a: f64,
    b: f64,
    gamma: f64,
    r: f64,
    t: f64,
    mua: f64,
    thickness: f64,
}

/// One layer's z-extent plus its I(ξ)/J(ξ) profile closure — compute_volume's
/// working set, built once per layer and then sampled at many z per voxel row.
struct LayerRange {
    z0: f64,
    z1: f64,
    mua: f64,
    profile: Box<dyn Fn(f64) -> (f64, f64)>,
}

fn layer_data(layers: &[KMLayerParams]) -> Vec<LayerData> {
    layers
        .iter()
        .map(|l| {
            let LayerAlone { a, b, gamma, r, t } = km_layer_alone(l.mua, l.mus, l.thickness);
            LayerData {
                a,
                b,
                gamma,
                r,
                t,
                mua: l.mua,
                thickness: l.thickness,
            }
        })
        .collect()
}

/// Bottom-up reflectance-of-everything-below-each-layer (O(N_layers), N≤8) —
/// shared by both the summary and volume commands so this is written once.
fn r_below(layers: &[LayerData]) -> (Vec<f64>, f64) {
    let n = layers.len();
    let mut r_below = vec![0.0; n];
    let mut r_cum = 0.0;
    for i in (0..n).rev() {
        r_below[i] = r_cum;
        let LayerData { r, t, .. } = layers[i];
        r_cum = r + (t * t * r_cum) / (1.0 - r * r_cum);
    }
    (r_below, r_cum) // r_cum here is R_total, reflectance of the whole stack
}

pub fn derived(p: &KubelkaMunkParams) -> KubelkaMunkDerived {
    let layers = layer_data(&p.layers);
    let (r_below, r_total) = r_below(&layers);

    let i0_top = p.p0 / (p.lx * p.ly);
    let mut i0 = i0_top;
    let mut z_start = 0.0;

    for (i, l) in layers.iter().enumerate() {
        let profile = km_layer_profile(l.a, l.b, l.gamma, i0, r_below[i]);
        let (i_bottom, _) = profile(1.0);
        i0 = i_bottom;
        z_start += l.thickness;
    }

    let lz = z_start;
    let t_total = i0 / i0_top;
    let a_total = 1.0 - r_total - t_total;

    KubelkaMunkDerived {
        r_total,
        t_total,
        a_total,
        lz,
    }
}

/// Kubelka-Munk collapses the full angular distribution of light down to just
/// two diffuse streams (up/down) — a coarser approximation than even the P1/
/// diffusion model, so it needs a stronger separation between scattering and
/// absorption to hold, plus enough physical thickness to behave diffusely.
pub fn check_validity(p: &KubelkaMunkParams, derived: &KubelkaMunkDerived) -> ValidityResult {
    let mut reasons = Vec::new();

    for (i, l) in p.layers.iter().enumerate() {
        let at = |name: &str| format!("Layer {}: {}", i + 1, name);
        // K must be strictly positive, not merely non-negative: b = sqrt(a²-1)
        // is zero for a non-absorbing layer, which takes R and T to 0/0.
        require(&mut reasons, l.mua > 0.0, &at("K"), "greater than 0", l.mua);
        require(&mut reasons, l.mus > 0.0, &at("S"), "greater than 0", l.mus);
        require(&mut reasons, l.thickness > 0.0, &at("thickness"), "greater than 0", l.thickness);
    }
    require(&mut reasons, p.p0 > 0.0, "P<sub>0</sub>", "greater than 0", p.p0);
    let area = p.lx * p.ly;
    require(&mut reasons, area > 0.0, "the illuminated area L<sub>x</sub>·L<sub>y</sub>", "greater than 0", area);
    if !reasons.is_empty() {
        return ValidityResult { valid: false, reasons };
    }

    // The depth profile is sampled on nz voxels spanning the whole stack, so a
    // layer thinner than one voxel can be stepped straight over. The other two
    // models warn about their own grid resolution; this is the equivalent.
    let dz = derived.lz / p.nz as f64;
    for (i, l) in p.layers.iter().enumerate() {
        if l.thickness < dz {
            reasons.push(format!(
                "Layer {}: thickness {:.3} cm is under one depth voxel ({:.3} cm) — the profile \
                 is sampled too coarsely to show this layer at all, however strongly it absorbs. \
                 Increase N<sub>z</sub>",
                i + 1,
                l.thickness,
                dz
            ));
        }
    }

    for (i, l) in p.layers.iter().enumerate() {
        let ratio = l.mus / l.mua;
        let LayerAlone { gamma, .. } = km_layer_alone(l.mua, l.mus, l.thickness);

        if ratio < 5.0 {
            reasons.push(format!(
                "Layer {}: S/K = {:.2} (want ≳5) — absorption is too strong \
                 relative to scattering for a diffuse two-flux field to build up inside this layer",
                i + 1,
                ratio
            ));
        }
        if gamma < 1.0 {
            reasons.push(format!(
                "Layer {}: optical thickness γ = bSd = {:.3} (want ≳1) — this \
                 layer is optically thin, so light passes through with too little scattering for the \
                 diffuse up/down flux picture to apply; it behaves more like direct transmission",
                i + 1,
                gamma
            ));
        }
    }

    if derived.a_total < 0.0 || derived.t_total < 0.0 || derived.r_total < 0.0 {
        reasons.push(format!(
            "energy balance came out negative (R={:.3}, T={:.3}, A={:.3}) — this indicates a \
             numerical edge case rather than a physical result; try adjusting the layer parameters",
            derived.r_total, derived.t_total, derived.a_total
        ));
    }

    ValidityResult {
        valid: reasons.is_empty(),
        reasons,
    }
}

/// The expensive part: builds the 1-D z-profile of Φ and A, then broadcasts it
/// across every (x,y) voxel column — the physics has no lateral structure.
pub fn compute_volume(p: &KubelkaMunkParams) -> (Vec<f32>, Vec<f32>) {
    let layers = layer_data(&p.layers);
    let (r_below, _) = r_below(&layers);

    let i0_top = p.p0 / (p.lx * p.ly);
    let mut i0 = i0_top;
    let mut z_start = 0.0;
    let mut layer_ranges: Vec<LayerRange> = Vec::with_capacity(layers.len());

    for (i, l) in layers.iter().enumerate() {
        let profile = km_layer_profile(l.a, l.b, l.gamma, i0, r_below[i]);
        let z_end = z_start + l.thickness;
        let (i_bottom, _) = profile(1.0);
        layer_ranges.push(LayerRange {
            z0: z_start,
            z1: z_end,
            mua: l.mua,
            profile: Box::new(profile),
        });
        i0 = i_bottom;
        z_start = z_end;
    }

    let lz = z_start;
    let dz = lz / p.nz as f64;
    let mut phi_z = vec![0f64; p.nz];
    let mut abs_z = vec![0f64; p.nz];

    let mut li = 0usize;
    for iz in 0..p.nz {
        let z = (iz as f64 + 0.5) * dz;
        while li < layer_ranges.len() - 1 && z > layer_ranges[li].z1 {
            li += 1;
        }
        let range = &layer_ranges[li];
        let xi = ((z - range.z0) / (range.z1 - range.z0)).clamp(0.0, 1.0);
        let (i, j) = (range.profile)(xi);
        let phi_val = i + j;
        phi_z[iz] = phi_val;
        abs_z[iz] = range.mua * phi_val;
    }

    let n = p.nx * p.ny * p.nz;
    let mut phi = vec![0f32; n];
    let mut abs = vec![0f32; n];
    for iz in 0..p.nz {
        let pv = phi_z[iz] as f32;
        let av = abs_z[iz] as f32;
        for iy in 0..p.ny {
            let base = iy * p.nx + iz * p.nx * p.ny;
            for ix in 0..p.nx {
                phi[base + ix] = pv;
                abs[base + ix] = av;
            }
        }
    }

    (phi, abs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(mua: f64, mus: f64, thickness: f64) -> KMLayerParams {
        KMLayerParams { mua, mus, thickness }
    }

    fn params(layers: Vec<KMLayerParams>, nz: usize) -> KubelkaMunkParams {
        KubelkaMunkParams { lx: 2.0, ly: 2.0, nx: 8, ny: 8, nz, p0: 1.0, layers }
    }

    /// K = 0 is a perfectly reasonable thing to ask for — a non-absorbing
    /// layer — but it takes b = sqrt(a²-1) to zero and R, T to 0/0, so the
    /// whole profile comes back NaN. It has to be caught, not just survived.
    #[test]
    fn nonphysical_input_is_reported_alone() {
        for bad in [layer(0.0, 50.0, 0.5), layer(-1.0, 50.0, 0.5), layer(0.1, 0.0, 0.5), layer(0.1, 50.0, 0.0)] {
            let p = params(vec![bad], 20);
            let result = check_validity(&p, &derived(&p));
            assert_eq!(result.reasons.len(), 1, "got {:?}", result.reasons);
            assert!(result.reasons[0].starts_with("Layer 1:"), "{}", result.reasons[0]);
        }

        let ok = params(vec![layer(0.1, 50.0, 0.5)], 20);
        let result = check_validity(&ok, &derived(&ok));
        assert!(
            !result.reasons.iter().any(|r| r.contains("must be")),
            "sane params shouldn't trip an input check: {:?}", result.reasons
        );
    }

    /// A layer thinner than one depth voxel is stepped straight over, so it
    /// leaves no trace in the profile no matter how strongly it absorbs.
    #[test]
    fn layer_thinner_than_a_depth_voxel_is_flagged() {
        // Lz = 2.02 over 10 voxels -> dz = 0.202 cm; the middle layer is 0.02.
        let p = params(vec![layer(0.1, 50.0, 1.0), layer(3.0, 50.0, 0.02), layer(0.1, 50.0, 1.0)], 10);
        let result = check_validity(&p, &derived(&p));
        let hits: Vec<_> = result.reasons.iter().filter(|r| r.contains("depth voxel")).collect();
        assert_eq!(hits.len(), 1, "expected layer 2 flagged, got {:?}", result.reasons);
        assert!(hits[0].contains("Layer 2"), "{}", hits[0]);

        // Same stack, enough depth voxels to resolve it.
        let fine = params(vec![layer(0.1, 50.0, 1.0), layer(3.0, 50.0, 0.02), layer(0.1, 50.0, 1.0)], 400);
        let result = check_validity(&fine, &derived(&fine));
        assert!(
            !result.reasons.iter().any(|r| r.contains("depth voxel")),
            "a resolved layer shouldn't be flagged: {:?}", result.reasons
        );
    }
}


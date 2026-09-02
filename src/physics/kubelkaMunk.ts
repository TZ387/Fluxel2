import type { ComputeResult, ValidityResult } from "./fpw1992";

/* ================================================================
   KUBELKA–MUNK TWO-FLUX MODEL — N homogeneous layers
   ================================================================
   Unlike FPW1992 (a point-source pencil beam solved via the P1/
   diffusion approximation), classic Kubelka-Munk is a *1-D* model:
   it assumes the sample is illuminated by a perfectly diffuse
   (Lambertian) flux uniform over the whole top face, and tracks only
   two counter-propagating diffuse flux streams, I(z) (downward) and
   J(z) (upward), as a function of depth z. There is no x/y
   dependence in the physics at all.

   To fit this 1-D result into the app's existing 3-D voxel/slice
   viewer, the computed z-profile is broadcast identically across
   every (x,y) voxel column. Φ and A are therefore uniform in x and
   y and vary only with z — this is the physically correct picture
   for KM's own assumptions (broad-area diffuse illumination), not a
   simplification of a beam that should have lateral structure.

   Per-layer equations (Kubelka, 1948):
     a = 1 + K/S
     b = sqrt(a² - 1)
     γ = b·S·d                      (dimensionless optical thickness)

   Reflectance/transmittance of a single layer alone, over a
   non-reflecting (black) backing:
     R = tanh(γ) / (a·tanh(γ) + b)
     T = b·sech(γ) / (a·tanh(γ) + b)
   (equivalent to the textbook sinh/cosh forms, rewritten so nothing
   overflows for optically thick layers — tanh/sech stay in [0,1]
   however large γ gets.)

   Stacking layers (bottom → top), Kubelka's "adding" formulas combine
   a layer's own (R,T) with the reflectance R_g of everything beneath
   it into the reflectance of the whole stack from that point down:
     R_combined = R + T² R_g / (1 - R R_g)
   The last (bottom-most) layer sits on a non-reflecting backing
   (R_g = 0), i.e. a semi-infinite absorber — there's no explicit
   substrate parameter, so any flux reaching the bottom is treated as
   fully absorbed there rather than reflected back up.

   Internal flux profile within a layer: given the flux I0 entering
   its top and the reflectance R_g of the stack beneath it, the
   two-flux ODEs (dI/dx = -(S+K)I + SJ, dJ/dx = (S+K)J - SI) have the
   closed-form solution used in kmLayerProfile() below, written
   so that the always-decaying combination is what actually gets
   evaluated (no raw e^{+γ} term with γ up to hundreds).

   Absorbed power density (analogous to μ_a·Φ in FPW1992):
     A(z) = K · [I(z) + J(z)]
   ================================================================ */

export interface KMLayerParams {
  mua: number; // K, absorption coefficient
  mus: number; // S, scattering coefficient
  thickness: number;
}

export interface KubelkaMunkParams {
  lx: number;
  ly: number;
  nx: number;
  ny: number;
  nz: number;
  p0: number;
  layers: KMLayerParams[];
}

export interface KubelkaMunkDerived {
  R_total: number;
  T_total: number;
  A_total: number;
  Lz: number;
}

interface KMLayerAlone {
  a: number;
  b: number;
  gamma: number;
  R: number;
  T: number;
}

/** Reflectance/transmittance of a single layer alone, black backing. */
function kmLayerAlone(mua: number, mus: number, thickness: number): KMLayerAlone {
  const a = 1.0 + mua / mus;
  const b = Math.sqrt(a * a - 1.0);
  const gamma = b * mus * thickness;
  const th = Math.tanh(gamma);
  const sech = 1.0 / Math.cosh(gamma); // → 0 for large γ, never NaN/Inf
  const denom = a * th + b;
  const R = th / denom;
  const T = (b * sech) / denom;
  return { a, b, gamma, R, T };
}

/**
 * Evaluate I(ξ), J(ξ) inside one layer, ξ ∈ [0,1] = normalised depth
 * (0 = top of this layer, 1 = bottom), given the downward flux I0
 * entering the top and the reflectance Rg of everything below.
 */
function kmLayerProfile(
  a: number,
  b: number,
  gamma: number,
  I0: number,
  Rg: number
): (xi: number) => { I: number; J: number } {
  // K0 = (P/Q) already carries the e^{-2γ} decay, so every exponential
  // evaluated below has a non-positive argument — stable for any γ.
  const K0 = (Rg - (a - b)) / (a + b - Rg);
  const termAt = (xi: number) => K0 * Math.exp(2.0 * gamma * (xi - 1.0));
  const Q = I0 / (termAt(0) + 1.0);
  return (xi: number) => {
    const term = termAt(xi);
    const expNeg = Math.exp(-gamma * xi);
    const I = Q * expNeg * (term + 1.0);
    const J = Q * expNeg * ((a + b) * term + (a - b));
    return { I, J };
  };
}

export function computeDiffusion_KubelkaMunk(
  p: KubelkaMunkParams
): ComputeResult<KubelkaMunkDerived> {
  const { lx, ly, nx, ny, nz, p0, layers } = p;
  const N = layers.length;

  const layerData = layers.map((L) => {
    const { R, T, a, b, gamma } = kmLayerAlone(L.mua, L.mus, L.thickness);
    return { mua: L.mua, mus: L.mus, thickness: L.thickness, a, b, gamma, R, T };
  });

  /* Bottom-up: reflectance of everything strictly below layer i
     (Rbelow), needed as the boundary condition for that layer's own
     internal flux solution. Last layer sees a black backing (Rg=0). */
  const Rbelow = new Array<number>(N);
  let Rcum = 0.0;
  for (let i = N - 1; i >= 0; i--) {
    Rbelow[i] = Rcum;
    const { R, T } = layerData[i];
    Rcum = R + (T * T * Rcum) / (1.0 - R * Rcum);
  }
  const R_total = Rcum; // reflectance of the whole stack, seen from the top

  /* Top-down: propagate the incident flux through each layer,
     building one z-profile function per layer plus the cumulative
     depth range it occupies. */
  const I0_top = p0 / (lx * ly); // incident diffuse flux density [W/cm²]
  let I0 = I0_top;
  let zStart = 0.0;
  const layerRanges: {
    z0: number;
    z1: number;
    mua: number;
    profile: (xi: number) => { I: number; J: number };
  }[] = [];

  for (let i = 0; i < N; i++) {
    const { a, b, gamma, mua, thickness } = layerData[i];
    const profile = kmLayerProfile(a, b, gamma, I0, Rbelow[i]);
    const zEnd = zStart + thickness;
    layerRanges.push({ z0: zStart, z1: zEnd, mua, profile });

    const { I: I_bottom } = profile(1.0);
    I0 = I_bottom; // flux entering the next layer down
    zStart = zEnd;
  }

  const Lz = zStart;
  const T_total = I0 / I0_top; // flux reaching the (absorbing) bottom
  const A_total = 1.0 - R_total - T_total;

  /* Build the 1-D z-profile of Φ and A once, then broadcast across
     every (x,y) voxel column — the physics has no lateral structure. */
  const dz = Lz / nz;
  const phiZ = new Float64Array(nz);
  const absZ = new Float64Array(nz);

  let li = 0;
  for (let iz = 0; iz < nz; iz++) {
    const z = (iz + 0.5) * dz;
    while (li < N - 1 && z > layerRanges[li].z1) li++;
    const { z0, z1, mua, profile } = layerRanges[li];
    const xi = Math.min(1, Math.max(0, (z - z0) / (z1 - z0)));
    const { I, J } = profile(xi);
    const phiVal = I + J;
    phiZ[iz] = phiVal;
    absZ[iz] = mua * phiVal;
  }

  const phi = new Float64Array(nx * ny * nz);
  const abs = new Float64Array(nx * ny * nz);
  for (let iz = 0; iz < nz; iz++) {
    const pv = phiZ[iz],
      av = absZ[iz];
    for (let iy = 0; iy < ny; iy++) {
      const base = iy * nx + iz * nx * ny;
      for (let ix = 0; ix < nx; ix++) {
        phi[base + ix] = pv;
        abs[base + ix] = av;
      }
    }
  }

  return {
    phi,
    abs,
    derived: { R_total, T_total, A_total, Lz },
  };
}

/* ================================================================
   KUBELKA–MUNK VALIDITY CHECK
   ================================================================
   Kubelka-Munk collapses the full angular distribution of light down
   to just two diffuse streams (up/down). That's a much coarser
   angular approximation than even the P1/diffusion model, so it
   needs a stronger separation between scattering and absorption to
   hold, plus enough physical thickness for the medium to actually
   behave diffusely rather than transmit light directly through.

   Two checks, applied per layer:

   1. Albedo: a layer that absorbs too strongly relative to how much
      it scatters (S/K not large) never builds up the near-isotropic
      internal light field the two-flux picture assumes.

   2. Optical thickness γ = bSd: if a layer is optically thin, light
      barely scatters before reaching its far boundary, so there
      isn't room for the diffuse up/down streams to establish
      themselves — the layer behaves more like it's directly
      transmitting light than diffusing it.
   ================================================================ */
export function checkValidity_KubelkaMunk(
  p: KubelkaMunkParams,
  derived: KubelkaMunkDerived
): ValidityResult {
  const reasons: string[] = [];

  p.layers.forEach((L, i) => {
    const ratio = L.mus / L.mua;
    const { gamma } = kmLayerAlone(L.mua, L.mus, L.thickness);

    if (ratio < 5) {
      reasons.push(
        `Layer ${i + 1}: S/K = ${ratio.toFixed(2)} (want ≳5) — absorption is too strong ` +
          `relative to scattering for a diffuse two-flux field to build up inside this layer`
      );
    }
    if (gamma < 1) {
      reasons.push(
        `Layer ${i + 1}: optical thickness γ = bSd = ${gamma.toFixed(3)} (want ≳1) — this ` +
          `layer is optically thin, so light passes through with too little scattering for the ` +
          `diffuse up/down flux picture to apply; it behaves more like direct transmission`
      );
    }
  });

  if (derived.A_total < 0 || derived.T_total < 0 || derived.R_total < 0) {
    reasons.push(
      `energy balance came out negative (R=${derived.R_total.toFixed(3)}, ` +
        `T=${derived.T_total.toFixed(3)}, A=${derived.A_total.toFixed(3)}) — this indicates a ` +
        `numerical edge case rather than a physical result; try adjusting the layer parameters`
    );
  }

  return { valid: reasons.length === 0, reasons };
}

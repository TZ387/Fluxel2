/* ================================================================
   HELP TAB CONTENT
   ================================================================
   Static prose — general usage plus one section per model (equation,
   use cases, limits, reference). Built once at init since none of it
   depends on simulator state. Model titles come from models.ts (the
   MODELS registry) so they can't drift out of sync with the dropdown;
   everything else here is narrative that doesn't fit that schema.

   Equations are written the same way the rest of the app already
   renders math — HTML entities/sub/sup, no rendering library — for
   consistency with the summary line and validity warnings.
   ================================================================ */

import { MODELS } from "./models";

interface ModelHelp {
  modelId: keyof typeof MODELS;
  description: string;
  equation: string;
  useFor: string;
  limits: string[];
  reference: string;
}

/* Ordered by publication year (1948 → 1992 → 2010) — builds up from the
   simplest model to the most involved, unlike the dropdown's own order
   (newest/most general first, so it's the default selection). */
const MODEL_HELP: ModelHelp[] = [
  {
    modelId: "kubelkaMunk",
    description:
      "A 1-D two-flux model. Rather than tracking a beam's exact direction, it lumps all light into two " +
      "counter-propagating diffuse streams — downward I(z) and upward J(z) — through an arbitrary stack of " +
      "homogeneous layers, each with its own absorption K, scattering S, and thickness d.",
    equation:
      "a = 1 + K/S              b = &radic;(a&sup2; &minus; 1)              &gamma; = b&middot;S&middot;d\n\n" +
      "R = sinh &gamma; / (a&middot;sinh &gamma; + b&middot;cosh &gamma;)      " +
      "T = b / (a&middot;sinh &gamma; + b&middot;cosh &gamma;)\n" +
      "                              (reflectance / transmittance, one layer alone)\n\n" +
      "R<sub>stack</sub> = R + T&sup2;&middot;R<sub>below</sub> / (1 &minus; R&middot;R<sub>below</sub>)   " +
      "(combined bottom-up, one layer at a time)",
    useFor:
      "Predominantly diffuse illumination — not a beam — through a layered coating, paint, textile, paper, or " +
      "film stack: anywhere the light source floods the whole top face evenly, so lateral position doesn't " +
      "matter and only depth does.",
    limits: [
      "No lateral (x, y) structure at all — the depth profile is broadcast identically across every column, " +
        "which is only physically correct for genuinely diffuse illumination, not a beam.",
      "Needs S/K &gtrsim; 5 in each layer for the two-flux picture to hold.",
      "Needs each layer's optical thickness &gamma; &gtrsim; 1 — an optically thin layer behaves more like " +
        "direct transmission than a diffuse field.",
      "Doesn't model a refractive-index mismatch between layers, or at the surface.",
    ],
    reference:
      "P. Kubelka, “New Contributions to the Optics of Intensely Light-Scattering Materials, Part I,” " +
      "J. Opt. Soc. Am. 38(5), 448–457 (1948).",
  },
  {
    modelId: "fpw1992",
    description:
      "The diffusion approximation for a narrow, normally-incident beam entering a semi-infinite homogeneous " +
      "turbid slab. The beam is modelled as an isotropic point source one transport mean free path below the " +
      "surface; a matching image source above the surface enforces the extrapolated (Robin) boundary condition " +
      "that accounts for the refractive-index mismatch at the surface.",
    equation:
      "&Phi;(r) = P&#8320; / (4&pi;D) &middot; [ exp(&minus;&mu;<sub>eff</sub>&middot;r&#8321;)/r&#8321; &minus; " +
      "exp(&minus;&mu;<sub>eff</sub>&middot;r&#8322;)/r&#8322; ]        A(r) = &mu;<sub>a</sub>&middot;&Phi;(r)\n\n" +
      "D = 1 / (3(&mu;<sub>a</sub>+&mu;<sub>s</sub>'))     " +
      "&mu;<sub>eff</sub> = &radic;(3&middot;&mu;<sub>a</sub>&middot;(&mu;<sub>a</sub>+&mu;<sub>s</sub>'))     " +
      "z&#8320; = 1 / (&mu;<sub>a</sub>+&mu;<sub>s</sub>')\n" +
      "                                                                    (r&#8321;, r&#8322; = distance to the real / image source)",
    useFor:
      "A quick, closed-form estimate of fluence and absorption from a laser or LED beam in a single, optically " +
      "homogeneous medium — skin, a bulk material, a phantom. It has genuine 3-D structure (the beam enters at " +
      "a point and spreads radially), which the other two models don't.",
    limits: [
      "One homogeneous layer only.",
      "Needs &mu;<sub>s</sub>' &gtrsim; 10&middot;&mu;<sub>a</sub> — absorption weak relative to scattering, " +
        "so light randomises direction many times before being absorbed.",
      "Needs the medium's smallest dimension to be several transport mean free paths, so a photon can scatter " +
        "many times before reaching a boundary.",
      "Least accurate within about one transport mean free path of the source — a real beam has some finite " +
        "width; this model idealises it as a single point.",
    ],
    reference:
      "T. J. Farrell, M. S. Patterson, B. Wilson, “A diffusion theory model of spatially resolved, " +
      "steady-state diffuse reflectance for the noninvasive determination of tissue optical properties in " +
      "vivo,” Med. Phys. 19(4), 879–888 (1992).",
  },
  {
    modelId: "liemertKienle",
    description:
      "The point-source diffusion equation solved for two stacked homogeneous layers — the combination " +
      "FPW1992 (point source, one layer) and Kubelka-Munk (many layers, diffuse illumination) each stop short " +
      "of. Same governing equation as FPW1992, applied per layer and matched across the interface with " +
      "continuity of fluence and flux, plus an extrapolated boundary condition top and bottom.",
    equation:
      "D&middot;&nabla;&sup2;&Phi; &minus; &mu;<sub>a</sub>&Phi; = &minus;S(r)      " +
      "(per layer; D = 1/(3&mu;<sub>s</sub>'), a different but equally standard convention from FPW1992's D above)\n\n" +
      "Boundary conditions: extrapolated at the top surface (same treatment as FPW1992) &middot; &Phi; and flux " +
      "continuous across the layer 1/2 interface &middot; extrapolated (or effectively semi-infinite, if layer " +
      "2 is thick) at the bottom.\n\n" +
      "Layering breaks the spherical symmetry that gives FPW1992 its short closed form, so the solution is a " +
      "Fourier–Bessel series (a sum over zeros of J&#8320;) rather than a one-line formula — reproducing it " +
      "in full doesn't fit here; see the reference paper, or this app's own " +
      "src-tauri/src/physics/liemert_kienle.rs for the complete, commented derivation.",
    useFor:
      "A beam or point source through a two-layer medium where you need real lateral (not just depth) " +
      "structure — skin's epidermis and dermis, a coating on a bulk substrate, a thin film on a different " +
      "material below it.",
    limits: [
      "Exactly two layers in this implementation, not arbitrary N — the reference implementation this is " +
        "ported from only provides the Green's functions for the top and bottom layer, not a middle layer for " +
        "N&ge;3, so this doesn't claim more generality than what's actually been verified.",
      "The beam's effective source point must fall within layer 1 — the app warns if layer 1 is too thin " +
        "(or scatters too weakly) for that.",
      "Same &mu;<sub>s</sub>'/&mu;<sub>a</sub> &gtrsim; 10-per-layer requirement as FPW1992, for the same reason.",
    ],
    reference:
      "A. Liemert, A. Kienle, “Light diffusion in a turbid cylinder. II. Layered case,” " +
      "Opt. Express 18(9), 9266–9279 (2010).",
  },
];

function modelSection(h: ModelHelp): string {
  const label = MODELS[h.modelId].label;
  return `
    <div class="panel help-model">
      <div class="panel-title">${label}</div>
      <p>${h.description}</p>
      <pre class="help-eqn">${h.equation}</pre>
      <p><strong>Use it for:</strong> ${h.useFor}</p>
      <p><strong>Limits:</strong></p>
      <ul>${h.limits.map((l) => `<li>${l}</li>`).join("")}</ul>
      <p class="help-ref">${h.reference}</p>
    </div>`;
}

export function buildHelp(containerId: string): void {
  const root = document.getElementById(containerId)!;
  root.innerHTML = `
    <div class="panel">
      <div class="panel-title">Using this app</div>
      <ol class="help-steps">
        <li>Pick a model from the dropdown on the Simulator tab. Its parameter panel is generated entirely
          from that model's own schema, so different models show different fields.</li>
        <li>Adjust a slider, or type directly into any of the three number boxes next to it (min, max, or the
          current value) — the slider's range extends automatically if you type outside it.</li>
        <li>Click <strong>Compute &amp; visualise</strong>. A warning appears below the result if the diffusion
          approximation is weakly justified for the parameters you've chosen (see each model's Limits below) —
          the result is still shown, but treat it with appropriate skepticism.</li>
        <li>Drag the x / y / z sliders beneath each plot to move the three slice planes through the volume.</li>
        <li>The colour scale is logarithmic, to show the full dynamic range from near the source to far from
          it — read the colourbar's numeric labels, not just its colour, when comparing two runs.</li>
      </ol>
    </div>
    ${MODEL_HELP.map(modelSection).join("")}
  `;
}

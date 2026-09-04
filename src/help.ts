/* ================================================================
   HELP TAB CONTENT
   ================================================================
   Static prose — general usage plus one section per model (equation,
   use cases, limits, reference). Built once at init. Model titles
   come from models.ts's MODELS registry so they can't drift from the
   dropdown; everything else is narrative outside that schema.

   Equations use HTML <sub>/<sup> exclusively, never unicode
   subscript/superscript — mixing the two renders inconsistently,
   since <sub>/<sup> are baseline-shifted and resized by the browser
   while unicode glyphs sit at a fixed small size. One formula per
   line; an explanatory aside gets its own paragraph rather than being
   crammed onto the same line.
   ================================================================ */

import { MODELS } from "./models";

interface ModelHelp {
  modelId: keyof typeof MODELS;
  description: string[]; // one or more paragraphs
  equation: string;
  eqnNote?: string;
  useFor: string;
  limits: string[];
  reference: string;
}

/* Ordered to match the dropdown — newest/most general (and default)
   model first — rather than publication year. */
const MODEL_HELP: ModelHelp[] = [
  {
    modelId: "liemertKienle",
    description: [
      "The point-source diffusion equation solved for a stack of homogeneous layers — the combination " +
        "FPW1992 (point source, one layer) and Kubelka-Munk (many layers, diffuse illumination) each stop " +
        "short of. The same governing equation as FPW1992 below, applied per layer with " +
        "D = 1/(3&mu;<sub>s</sub>') here (a different but equally standard convention from FPW1992's D), and " +
        "matched across each interface with continuity of &Phi;/n&sup2; and of D&middot;d&Phi;/dz, plus an " +
        "extrapolated boundary condition above the stack and below it.",
      "Layering breaks the spherical symmetry that gives FPW1992 its short closed-form solution below, so " +
        "this one is instead a Fourier–Bessel series (a sum over zeros of J<sub>0</sub>) — too long to " +
        "reproduce here in full; see the reference paper, or this app's own " +
        "src-tauri/src/physics/liemert_kienle.rs for the complete, commented derivation.",
      "Each term of that series reduces to a 1-D problem in depth that any number of layers folds into: " +
        "everything below a given depth reaches the layers above it only through a single reflection " +
        "coefficient, accumulated bottom-up one layer at a time. So the stack can be 1 layer or 8 at " +
        "essentially the same cost per voxel.",
      "The beam profile can be widened from an idealised pencil to a Gaussian or flat-top (disk) spot. Since " +
        "each series term above is tied to one transverse spatial frequency, widening the beam only multiplies " +
        "each term by that profile's own spectral factor (1 for a point source, decaying for a wider spot) — " +
        "no change to the layered-medium part of the solution. See src-tauri/src/physics/beam.rs.",
    ],
    equation: "D&middot;&nabla;&sup2;&Phi; &minus; &mu;<sub>a</sub>&middot;&Phi; = &minus;S(r)",
    useFor:
      "A beam or point source through a layered medium where you need real lateral (not just depth) " +
      "structure — skin's epidermis, dermis and subcutis, a coating on a bulk substrate, a thin film on a " +
      "different material below it.",
    limits: [
      "The beam's effective source point must fall within layer 1 — the app warns if layer 1 is too thin " +
        "(or scatters too weakly) for that.",
      "The stack is bounded by air at both ends, so the last layer's bottom is a zero-fluence boundary just " +
        "like the top surface. Make it several penetration depths thick if you mean it as a semi-infinite " +
        "substrate rather than a finite slab.",
      "Same &mu;<sub>s</sub>'/&mu;<sub>a</sub> &gtrsim; 10-per-layer requirement as FPW1992, for the same reason.",
      "A Gaussian or flat-top beam wider than roughly a third of the internal finite-cylinder radius stops " +
        "being accurately convolved — the app warns when the beam footprint gets that large.",
      "The reference implementation this is ported from covers only the top and bottom layer, so the " +
        "middle-layer Green's function is derived here rather than ported — it reproduces the ported " +
        "two-layer form exactly, and is cross-checked against a direct numerical solve.",
    ],
    reference:
      "A. Liemert, A. Kienle, “Light diffusion in a turbid cylinder. II. Layered case,” " +
      "Opt. Express 18(9), 9266–9279 (2010).",
  },
  {
    modelId: "fpw1992",
    description: [
      "The diffusion approximation for a narrow, normally-incident beam entering a semi-infinite homogeneous " +
        "turbid slab. The beam is modelled as an isotropic point source one transport mean free path below " +
        "the surface; a matching image source above the surface enforces the extrapolated (Robin) boundary " +
        "condition that accounts for the refractive-index mismatch at the surface.",
      "The beam profile can be widened from that idealised pencil to a Gaussian or flat-top (disk) spot — " +
        "evaluated as a direct 2-D numerical convolution of the point-source formula below with the chosen " +
        "profile, since (unlike Liemert-Kienle) this model has no existing spatial-frequency series to fold " +
        "the profile into. See src-tauri/src/physics/beam.rs.",
    ],
    equation:
      "&Phi;(r) = P<sub>0</sub> / (4&pi;D) &middot; [ exp(&minus;&mu;<sub>eff</sub>&middot;r<sub>1</sub>)/r<sub>1</sub> " +
      "&minus; exp(&minus;&mu;<sub>eff</sub>&middot;r<sub>2</sub>)/r<sub>2</sub> ]\n" +
      "A(r) = &mu;<sub>a</sub>&middot;&Phi;(r)\n\n" +
      "D = 1 / (3(&mu;<sub>a</sub> + &mu;<sub>s</sub>'))\n" +
      "&mu;<sub>eff</sub> = &radic;(3&middot;&mu;<sub>a</sub>&middot;(&mu;<sub>a</sub> + &mu;<sub>s</sub>'))\n" +
      "z<sub>0</sub> = 1 / (&mu;<sub>a</sub> + &mu;<sub>s</sub>')",
    eqnNote:
      "r<sub>1</sub>, r<sub>2</sub> are the field point's distance to the real source (z<sub>0</sub> below the " +
      "surface) and its mirror image (above the surface).",
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
      "Least accurate within about one transport mean free path of the source for the idealised pencil beam " +
        "— switch to the Gaussian or flat-top profile if the real beam's width is comparable to that distance.",
    ],
    reference:
      "T. J. Farrell, M. S. Patterson, B. Wilson, “A diffusion theory model of spatially resolved, " +
      "steady-state diffuse reflectance for the noninvasive determination of tissue optical properties in " +
      "vivo,” Med. Phys. 19(4), 879–888 (1992).",
  },
  {
    modelId: "kubelkaMunk",
    description: [
      "A 1-D two-flux model. Rather than tracking a beam's exact direction, it lumps all light into two " +
        "counter-propagating diffuse streams — downward I(z) and upward J(z) — through an arbitrary stack of " +
        "homogeneous layers, each with its own absorption K, scattering S, and thickness d.",
    ],
    equation:
      "a = 1 + K/S\n" +
      "b = &radic;(a<sup>2</sup> &minus; 1)\n" +
      "&gamma; = b&middot;S&middot;d\n\n" +
      "R = sinh(&gamma;) / (a&middot;sinh(&gamma;) + b&middot;cosh(&gamma;))\n" +
      "T = b / (a&middot;sinh(&gamma;) + b&middot;cosh(&gamma;))\n\n" +
      "R<sub>stack</sub> = R + T<sup>2</sup>&middot;R<sub>below</sub> / (1 &minus; R&middot;R<sub>below</sub>)",
    eqnNote:
      "R, T above are one layer's reflectance/transmittance in isolation; the last line combines them " +
      "bottom-up, one layer at a time.",
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
];

function modelSection(h: ModelHelp): string {
  const label = MODELS[h.modelId].label;
  const note = h.eqnNote ? `<p class="help-eqn-note">${h.eqnNote}</p>` : "";
  return `
    <div class="panel help-model">
      <div class="panel-title">${label}</div>
      ${h.description.map((p) => `<p>${p}</p>`).join("")}
      <pre class="help-eqn">${h.equation}</pre>
      ${note}
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

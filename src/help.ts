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

/* All three point-source models take the same beam pattern, described the
   same way, so the paragraph is written once. */
const PATTERN_PARAGRAPH =
  "The beam can also be aimed at more than one spot: a cross (two scanned rows at right angles — each a row " +
  "of discrete pulses, which approaches a continuous sweep once the pitch is small next to the beam width) " +
  "or a square grid (a fractional handpiece's array of microbeams). P<sub>0</sub> stays the pattern's total " +
  "power, so the spots share it equally, and light transport being linear means the result is simply " +
  "their fluences added up. The per-spot field is computed once and reused at every spot, so a 25-spot grid " +
  "costs barely more than a single spot rather than 25 times as much. Every model here builds a pattern out " +
  "of one radially symmetric per-spot field, which is why the scan is offered as a cross rather than a " +
  "single line: the pattern as a whole is still not radially symmetric — nothing but one spot is — but a " +
  "cross at least looks the same after a quarter turn.";

/* Ordered to match the dropdown — the reference model (and default) first,
   then the closed-form ones from most to least general — rather than by
   publication year. */
const MODEL_HELP: ModelHelp[] = [
  {
    modelId: "monteCarlo",
    description: [
      "The only model here that isn't an approximation. Rather than solving a simplified transport equation " +
        "in closed form, it traces individual photon packets through the layer stack and lets the statistics " +
        "of many tracks stand in for the answer. Nothing is assumed about scattering dominating absorption, " +
        "about layers being thick enough to diffuse across, or about the boundary being far away — so where " +
        "the other three models warn that they are outside their range, this one simply keeps working. It is " +
        "the model to check the others against, and the one to reach for unless a closed-form answer is " +
        "specifically wanted.",
      "Each packet is launched at the surface with unit weight, less the specular reflection off it. It then " +
        "repeats three steps until it is absorbed or leaves: hop a distance drawn from the exponential " +
        "distribution of free paths; deposit part of its weight where it lands; and scatter into a new " +
        "direction drawn from the Henyey-Greenstein phase function. At every refractive-index step — the " +
        "surface, each internal interface, the floor — it either reflects or refracts according to the " +
        "Fresnel coefficients, total internal reflection included, which is why an index mismatch between " +
        "layers matters here and not in Kubelka-Munk. A step interrupted by an interface is not redrawn but " +
        "continued on the other side, carrying its unused optical depth across.",
      "Every geometry this model accepts is symmetric about the beam axis — flat parallel layers, normal " +
        "incidence, a radially symmetric beam — so photons are scored into an (r, z) grid rather than a 3-D " +
        "one. That is what makes it affordable: every photon contributes to the same 2-D table, so a run " +
        "needs far fewer of them than a voxel-based Monte Carlo would for the same noise. The beam profile " +
        "comes free with it, and more exactly than a convolution would give: each packet's launch point is " +
        "drawn from the profile itself.",
      "Its grid is that 2-D one, and its parameters say so: &Delta;r is the width of one radial ring — the " +
        "lateral resolution the answer actually exists at — and N<sub>r</sub> is how many rings, so the run " +
        "reaches R = N<sub>r</sub>&middot;&Delta;r from the beam axis (reported on the status line). " +
        "N<sub>z</sub> divides the stack's depth as in every other model. What you are shown is still a box, " +
        "because that is what the two viewers draw: a square of half-width R, one voxel per ring across, so " +
        "the picture is at the resolution that was computed rather than an interpolation of it. There is no " +
        "L<sub>x</sub>, no L<sub>y</sub> and no N<sub>x</sub>/N<sub>y</sub> here — the simulation has no x " +
        "and no y to divide up.",
      "The price is that the answer is an estimate. Its error falls as 1/&radic;photons, so each halving of " +
        "the error bar costs four times the wait — which makes the photon budget a real choice rather than a " +
        "detail, and worth sweeping. To keep that honest the run is split into equal batches and their spread " +
        "used to estimate the error per bin, which is what the <em>Show Monte Carlo noise</em> overlay " +
        "displays: green where the relative error is under 5%, amber to 20%, red beyond. A given set of " +
        "parameters always produces the same volume, so a plot never changes for a reason you didn't cause.",
      PATTERN_PARAGRAPH,
    ],
    equation:
      "s = &minus;ln(&xi;) / &mu;<sub>t</sub>\n" +
      "cos&theta; = [ 1 + g&sup2; &minus; ((1&minus;g&sup2;)/(1&minus;g+2g&xi;))&sup2; ] / 2g\n" +
      "&Delta;w = w &middot; &mu;<sub>a</sub>/&mu;<sub>t</sub>\n\n" +
      "R = &frac12;[ ((n<sub>1</sub>cos&theta;<sub>i</sub> &minus; n<sub>2</sub>cos&theta;<sub>t</sub>) / " +
      "(n<sub>1</sub>cos&theta;<sub>i</sub> + n<sub>2</sub>cos&theta;<sub>t</sub>))&sup2; + " +
      "((n<sub>1</sub>cos&theta;<sub>t</sub> &minus; n<sub>2</sub>cos&theta;<sub>i</sub>) / " +
      "(n<sub>1</sub>cos&theta;<sub>t</sub> + n<sub>2</sub>cos&theta;<sub>i</sub>))&sup2; ]\n\n" +
      "&Phi;(r, z) = &Sigma; (w / &mu;<sub>t</sub>) / (N &middot; &Delta;V)\n" +
      "&mu;<sub>t</sub> = &mu;<sub>a</sub> + &mu;<sub>s</sub>",
    eqnNote:
      "Sampling rules rather than a solution: &xi; is a fresh uniform random number on (0, 1], and the last " +
      "line is the estimator — the summed weight-per-collision in each (r, z) bin, over N launched photons " +
      "and the bin's volume. Note that &mu;<sub>s</sub> and g enter separately here; the diffusion models " +
      "only ever see the combination &mu;<sub>s</sub>' = &mu;<sub>s</sub>(1&minus;g).",
    useFor:
      "Ground truth — checking any of the three closed-form models on a case you care about, especially one " +
      "they warn about. And as a model in its own right wherever they can't go: layers thinner than a mean " +
      "free path, absorption comparable to scattering, refractive-index steps inside the stack, or the first " +
      "millimetre below the surface where light hasn't scattered enough to diffuse yet.",
    limits: [
      "The answer carries statistical noise, falling as 1/&radic;photons. Switch on the noise overlay to see " +
        "where it lands, because it is rarely where you would guess: the innermost radial bins enclose the " +
        "least volume, so they collect the fewest photon collisions despite sitting in the brightest part of " +
        "the field, and the beam axis is usually the first thing to go amber. The far outskirts are the other " +
        "weak spot, for the opposite reason — hardly any photon gets that far.",
      "Cost scales with the photon budget and with how long each track runs, which is set by the tissue: " +
        "weakly absorbing, strongly scattering layers between two index steps trap light and make for long " +
        "tracks. The run spreads itself over every core the machine has, and takes well under a second for " +
        "the default budget on the default grid. Splitting the work differently doesn't change the answer, " +
        "so the number of cores affects only how long you wait.",
      "The geometry has to stay symmetric about the beam axis: flat parallel layers, normal incidence, a " +
        "radially symmetric beam. Tilted incidence, a warped interface, or an inclusion inside a layer would " +
        "all need a full 3-D grid instead, and are not supported.",
      "A beam <em>pattern</em> of more than one spot is not radially symmetric either, and the app says so " +
        "when you pick one. It is not an error: the layers are flat and uniform, so every spot really does " +
        "see the same kernel shifted, and adding them up is exact. What it costs is that the one kernel now " +
        "has to reach across the whole pattern on the same photon budget, and that the noise overlay adds " +
        "the spots' errors as though they were independent when they are all read off that single kernel — " +
        "so it reads slightly optimistic wherever spots overlap. Give a wide pattern more photons than a " +
        "single spot would need.",
      "The innermost radial bin is an area average over 0 &le; r &lt; &Delta;r, so an idealised pencil " +
        "beam's on-axis peak gets smoothed over that bin. The app warns when the bin is wide next to a " +
        "transport mean free path; a beam profile with a real width has no such issue.",
      "The specular reflection off the surface is deducted here and ignored by the diffusion models, so this " +
        "model's absolute fluence sits a few percent below theirs (2.8% at n = 1.4) even where they agree " +
        "perfectly otherwise. That difference is real physics, not a discrepancy between them.",
      "Steady state, unpolarized, elastic scattering: no time-of-flight gating, no polarization, no " +
        "fluorescence, and a single refractive index per layer. A beam pattern is simultaneous superposition, " +
        "the same as for the other models.",
    ],
    reference:
      "L. Wang, S. L. Jacques, L. Zheng, “MCML — Monte Carlo modeling of light transport in multi-layered " +
      "tissues,” Comput. Methods Programs Biomed. 47(2), 131–146 (1995); phase function from L. G. Henyey, " +
      "J. L. Greenstein, “Diffuse radiation in the Galaxy,” Astrophys. J. 93, 70–83 (1941). Implemented here " +
      "from those published algorithms — see src-tauri/src/physics/monte_carlo.rs.",
  },
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
      PATTERN_PARAGRAPH,
    ],
    equation: "D&middot;&nabla;&sup2;&Phi; &minus; &mu;<sub>a</sub>&middot;&Phi; = &minus;S(r)",
    useFor:
      "A beam or point source through a layered medium where you need real lateral (not just depth) " +
      "structure — skin's epidermis, dermis and subcutis, a coating on a bulk substrate, a thin film on a " +
      "different material below it.",
    limits: [
      "The beam's effective source point must fall within layer 1 — the app warns if layer 1 is too thin " +
        "(or scatters too weakly) for that.",
      "Every layer has to be at least a transport mean free path (1/&mu;<sub>s</sub>') thick. Light crossing " +
        "a thinner layer doesn't scatter even once inside it, so diffusion says nothing about it — the app " +
        "warns when a layer is that thin.",
      "The stack is bounded by air at both ends, so the last layer's bottom is a zero-fluence boundary just " +
        "like the top surface. Make it several penetration depths thick if you mean it as a semi-infinite " +
        "substrate rather than a finite slab.",
      "Same &mu;<sub>s</sub>'/&mu;<sub>a</sub> &gtrsim; 10-per-layer requirement as FPW1992, for the same reason.",
      "A Gaussian or flat-top beam wider than roughly a third of the internal finite-cylinder radius stops " +
        "being accurately convolved — the app warns when the beam footprint gets that large.",
      "A beam pattern is steady-state superposition, not a time sequence: every spot is on at once. That is " +
        "the right picture for a scan much faster than the tissue's thermal and optical response, not for " +
        "pulses far enough apart to be treated separately.",
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
      PATTERN_PARAGRAPH,
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
      "A beam pattern is steady-state superposition, not a time sequence: every spot is on at once. That is " +
        "the right picture for a scan much faster than the tissue's thermal and optical response, not for " +
        "pulses far enough apart to be treated separately.",
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
      "Read its two plots as its own quantities, not the other three models'. The field here is I + J, the sum " +
        "of the two fluxes, which for a hemispherically isotropic field is about <em>half</em> the fluence " +
        "rate &Phi; the others report; and K is the Kubelka-Munk absorption coefficient, about twice " +
        "&mu;<sub>a</sub>, not &mu;<sub>a</sub> itself. Those two factors of two cancel, so the absorbed " +
        "density A = K(I + J) is exactly right — integrating it through the stack gives back the power the " +
        "R/T/A balance says was absorbed — but I + J and &Phi; are not the same number and shouldn't be read " +
        "off against each other. Both panels are labelled accordingly.",
      "L<sub>x</sub> and L<sub>y</sub> do something different here too. In the other three they only frame " +
        "how much of the answer you see; here the incident power is spread evenly over the illuminated face, " +
        "so the irradiance entering the top is P<sub>0</sub>/(L<sub>x</sub>&middot;L<sub>y</sub>) and " +
        "widening the face dims the whole field in proportion. N<sub>x</sub> and N<sub>y</sub>, by contrast, " +
        "are pure display: the profile is one-dimensional and simply copied across every column, so raising " +
        "them enlarges the volume without adding anything to it.",
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
      "Its K and S are not the &mu;<sub>a</sub> and &mu;<sub>s</sub>' the other three models take, and this " +
        "app converts neither way — so a stack entered here and the same tissue entered in another model are " +
        "not the same optical properties, however similar the numbers look.",
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
        <li>A model with a layer stack opens with two layers and takes up to eight — <strong>+ Add
          layer</strong> and <strong>Remove</strong> change the count, and the rest of the stack keeps its
          values when you do. Click a layer's name to rename it: "Layer 1" until you say otherwise, and
          better as "Epidermis" or "Subcutis" once a stack has more than two. Names are labels only, never
          seen by the physics, but they are saved with the settings and they follow their layer.</li>
        <li>Click <strong>Compute &amp; visualise</strong>. A warning appears below the result if the model
          has something to say about the parameters you chose — the diffusion approximation being weakly
          justified, or, for Monte Carlo, too few photons for the grid you asked for (see each model's Limits
          below). The result is still shown, but treat it with appropriate skepticism.</li>
        <li>Drag the x / y / z sliders beneath each plot to move the three slice planes through the volume.
          They step one voxel at a time and read out in cm, measured the way the plots are labelled: x and y
          from the beam axis, z downward from the tissue surface.</li>
        <li>Both plots open as a <strong>3-D slice box</strong> — the same three cuts placed where they
          actually are inside the volume, at true proportions, with the layer interfaces marked on the back
          walls. Drag either plot to orbit it (both follow, so fluence and absorption stay comparable), and
          double-click to return to the default angle. The <strong>Flat slices</strong> layout puts the three
          cuts side by side instead: nothing is foreshortened there, so a distance on screen is a distance in
          the tissue — the view to use when reading a depth off the plot.</li>
        <li>Hover a plot to read the value under the cursor: the position in cm and the fluence or absorption
          there, plus which overlay band the voxel falls in when the overlay is on. In the 3-D box this
          follows whichever cut is nearest the camera at that point, so it reads the surface you can actually
          see. The blue crosshairs mark where the other two cuts pass through each plane, which is what ties
          the three views together.</li>
        <li>Tick the overlay checkbox under a plot to recolour those slices by how much to trust them
          voxel-by-voxel: for the diffusion models, how far each voxel is from breaking the approximation;
          for Monte Carlo, how converged its estimate is there. Kubelka-Munk has no such overlay.</li>
        <li>The colour scale defaults to logarithmic, which shows the full dynamic range from near the source
          to far from it. Switch it to linear when the question is about a threshold — how far the region
          above some fluence reaches — since a log ramp gives the top decade, where all of that happens, only
          a sliver of the colour range. Either way, read the colourbar's numbers, not just its colour, when
          comparing two runs.</li>
        <li><strong>Save settings&hellip;</strong> writes the whole parameter panel — layer names included —
          together with the view controls the two plots share, as a JSON file;
          <strong>Load settings&hellip;</strong> reads one back, switching the model first if the file
          belongs to a different one. The plots are put away on load, since they were computed from the
          parameters that have just been replaced — click Compute to bring them back. The file is meant to be
          readable and editable by hand, so loading is deliberately forgiving: anything in it that can't be
          used as written falls back to the model's default, and every such substitution is listed under the
          status line rather than left to be noticed.</li>
        <li>Each plot has its own <strong>Export plot&hellip;</strong> and <strong>Export data&hellip;</strong>
          buttons, under that panel's slice sliders. Export plot writes a PNG exactly as shown — whichever
          layout and colour scale are current — with the colourbar included, so the file says what the colour
          means on its own. Export data writes the underlying voxel grid instead of a picture, as a
          <code>.npy</code> array plus a <code>.json</code> file of the grid it was evaluated on; see below for
          how to read the pair back in Python or Julia.</li>
      </ol>
    </div>
    <div class="panel">
      <div class="panel-title">Reading exported data in Python or Julia</div>
      <p><strong>Export data&hellip;</strong> (above) writes two files per click: an array —
        <code>&lt;model&gt;-phi.npy</code> or <code>&lt;model&gt;-abs.npy</code> — and a
        <code>&lt;model&gt;-phi.json</code> / <code>&lt;model&gt;-abs.json</code> beside it, holding what a
        bare array can't carry: the grid the field was evaluated on (<code>nx</code>/<code>ny</code>/<code>nz</code>
        voxels, <code>lx</code>/<code>ly</code>/<code>lz</code> in cm, the layer interface depths), which field
        it is and its units, and the model that produced it. The array's shape is
        <code>(nz, ny, nx)</code> — z slowest, x fastest, matching the metadata's own
        <code>"axes": ["z", "y", "x"]</code> — chosen over CSV or plain JSON for the numbers themselves because
        the largest grid this app allows is 400&sup3; = 64 million voxels, where a text encoding runs to
        hundreds of megabytes and <code>.npy</code> is the raw bytes plus a short header. Both languages read
        it with one call:</p>
      <pre class="help-code">import json, numpy as np

phi = np.load("monteCarlo-phi.npy")       # shape (nz, ny, nx)
meta = json.load(open("monteCarlo-phi.json"))

# the x-y plane nearest z = 0.5 cm
z = (np.arange(meta["nz"]) + 0.5) * meta["lz"] / meta["nz"]  # voxel centres
iz = int(np.abs(z - 0.5).argmin())
plane = phi[iz]                           # shape (ny, nx)</pre>
      <pre class="help-code">using NPZ, JSON

phi = npzread("monteCarlo-phi.npy")       # size (nz, ny, nx), 1-indexed
meta = JSON.parsefile("monteCarlo-phi.json")

# the x-y plane nearest z = 0.5 cm
z = [(i - 0.5) * meta["lz"] / meta["nz"] for i in 1:meta["nz"]]  # voxel centres
iz = argmin(abs.(z .- 0.5))
plane = phi[iz, :, :]                     # size (ny, nx)</pre>
      <p class="help-eqn-note">NPZ.jl corrects for the row-major/column-major difference itself, so
        <code>phi[iz, iy, ix]</code> in Julia (1-indexed) and <code>phi[iz-1, iy-1, ix-1]</code> in NumPy
        (0-indexed) are the same voxel. x and y run from &minus;lx/2 / &minus;ly/2 to +lx/2 / +ly/2, centred on
        the beam axis; z runs from 0 at the surface to lz. Every voxel's coordinate is its <em>centre</em>,
        which the <code>+ 0.5</code> / <code>i - 0.5</code> above account for — the same convention the
        sliders' own cm readout uses (<code>axisPosition</code> in <code>src/main.ts</code>).</p>
    </div>
    ${MODEL_HELP.map(modelSection).join("")}
  `;
}

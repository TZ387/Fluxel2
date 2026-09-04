/* ================================================================
   MODEL REGISTRY
   ================================================================
   Each entry describes one theoretical model in the "Model" dropdown.
   The physics (compute + validity checks) lives in Rust, under
   src-tauri/src/physics/ — this file owns only the UI-facing bits:
   the parameter schema, and which Tauri command pair to invoke (see
   src/compute.ts).

   Parameters live here per-model, not in a shared list, because
   models need genuinely different inputs, not just different
   defaults (FPW1992 takes one set of optical properties;
   Kubelka-Munk needs a *repeatable* set, one per layer). Keeping each
   schema with its model lets ui-params.ts stay generic.

   paramGroups: array of groups, each its own panel.
     Plain group: { id, title, params: [{id,label,min,max,step,def,fmt}, ...] }
       → merges flat into getParams()'s result (p.mua, p.lx, ...).
     Repeating group: { id, title, params: [...], repeat: {min, max, def} }
       → rendered as `def` instances with add/remove buttons (bounded
         by min/max); comes back as an array, e.g. p.layers = [{...}, {...}].
         `repeat.defs` optionally gives the first instances their own
         starting values, so the stack can open non-uniform.

   To add a model: add a Rust module under src-tauri/src/physics/ with
   derived(), check_validity(), compute_volume() (see fpw1992.rs /
   kubelka_munk.rs), register its `<name>_summary`/`<name>_volume`
   commands in lib.rs, then add one entry below (label, command,
   summaryLine, paramGroups). The dropdown, param panel, run handler,
   and warning display all pick it up automatically.
   ================================================================ */

export interface SliderParamDef {
  kind?: "slider";
  id: string;
  label: string;
  min: number;
  max: number;
  step: number;
  def: number;
  fmt: (v: number) => string;
  /** Hide this row unless the named sibling select's value is one of `oneOf`. */
  showIf?: { id: string; oneOf: string[] };
}

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectParamDef {
  kind: "select";
  id: string;
  label: string;
  options: SelectOption[];
  def: string;
}

export type ParamDef = SliderParamDef | SelectParamDef;

export interface RepeatSpec {
  min: number;
  max: number;
  def: number;
  /** Starting values for the first instances, keyed by param id, so a model
      can open with a *contrasting* stack rather than N identical layers.
      Later instances fall back to each param's own `def`; values must lie
      inside that param's [min, max]. */
  defs?: Record<string, number | string>[];
}

export interface ParamGroup {
  id: string;
  title: string;
  params: ParamDef[];
  repeat?: RepeatSpec;
}

export interface ModelDef<D = any> {
  label: string;
  /** Rust command prefix — invokes `<command>_summary` and `<command>_volume`. */
  command: string;
  summaryLine: (derived: D, dt: string) => string;
  paramGroups: ParamGroup[];
}

/* Shared formatter helpers, purely to avoid repeating the same
   arrow function body — models are still free to define their own. */
const fmt3 = (v: number) => v.toFixed(3);
const fmt0 = (v: number) => v.toFixed(0);

/* Only worth saying when the beam pattern is more than one spot. */
const spotsSuffix = (spots: number) => (spots > 1 ? ` | ${spots} spots` : "");

/* Shared by every point-source model's beam group (FPW1992, Liemert-Kienle)
   — same three profile choices and width meaning everywhere, so defined
   once rather than copy-pasted per model. See src-tauri/src/physics/beam.rs
   for what each profile means physically. */
const BEAM_PROFILE_PARAM: SelectParamDef = {
  id: "beam_profile", label: "Beam profile", kind: "select", def: "pencil",
  options: [
    { value: "pencil", label: "Pencil (point source)" },
    { value: "gaussian", label: "Gaussian" },
    { value: "flattop", label: "Flat-top (disk)" },
  ],
};
const BEAM_WIDTH_PARAM: SliderParamDef = {
  id: "beam_width", label: "Beam width — σ (Gaussian) or R (flat-top) [cm]",
  min: 0.001, max: 0.5, step: 0.001, def: 0.05, fmt: fmt3,
  showIf: { id: "beam_profile", oneOf: ["gaussian", "flattop"] },
};

/* Where the beam is aimed, as opposed to its shape above — one spot, a row
   from a scanner, or a fractional handpiece's array. P0 stays the pattern's
   total power, so the spots share it out. */
const BEAM_PATTERN_PARAM: SelectParamDef = {
  id: "beam_pattern", label: "Beam pattern", kind: "select", def: "single",
  options: [
    { value: "single", label: "Single spot" },
    { value: "line", label: "Line scan" },
    { value: "grid", label: "Grid (fractional array)" },
  ],
};
const PATTERN_COUNT_PARAM: SliderParamDef = {
  id: "pattern_count", label: "Spots — along the line, or per side of the grid",
  min: 2, max: 16, step: 1, def: 5, fmt: fmt0,
  showIf: { id: "beam_pattern", oneOf: ["line", "grid"] },
};
const PATTERN_SPACING_PARAM: SliderParamDef = {
  id: "pattern_spacing", label: "Spot spacing (pitch) [cm]",
  min: 0.01, max: 1, step: 0.001, def: 0.2, fmt: fmt3,
  showIf: { id: "beam_pattern", oneOf: ["line", "grid"] },
};

export interface Fpw1992Derived {
  musp: number;
  D: number;
  mueff: number;
  delta: number;
  spots: number;
}

export interface KubelkaMunkDerived {
  R_total: number;
  T_total: number;
  A_total: number;
  Lz: number;
}

export interface LiemertKienleLayerDerived {
  musp: number;
  D: number;
  mueff: number;
}

export interface LiemertKienleDerived {
  layers: LiemertKienleLayerDerived[];
  z0: number;
  Lz: number;
  spots: number;
}

export const MODELS: Record<string, ModelDef> = {
  liemertKienle: {
    label: "Liemert & Kienle (2010) — N-layer, point-source diffusion",
    command: "liemert_kienle",
    summaryLine: (derived: LiemertKienleDerived, dt: string) =>
      `Done in ${dt} ms — ${derived.layers.length} layer${derived.layers.length === 1 ? "" : "s"} | ` +
      `z<sub>0</sub> = ${derived.z0.toFixed(3)} cm | L<sub>z</sub> = ${derived.Lz.toFixed(3)} cm | ` +
      `μ<sub>s</sub>' = ${derived.layers.map((l) => l.musp.toFixed(2)).join(", ")} cm⁻¹ | ` +
      `μ<sub>eff</sub> = ${derived.layers.map((l) => l.mueff.toFixed(3)).join(", ")} cm⁻¹` +
      spotsSuffix(derived.spots),

    /* Point source (pencil beam) through a stack of homogeneous layers —
       the combination FPW1992 (point source, one layer) and Kubelka-Munk
       (many layers, diffuse illumination) each stop short of. Every layer
       has its own thickness, so the grid's depth is the stack's total depth
       (reported as L_z), same as Kubelka-Munk. */
    paramGroups: [
      {
        id: "layers",
        title: "Layers (top → bottom)",
        /* Opens on a thin, strongly scattering top layer over a
           weaker-scattering bulk — the same default this model had when it
           was two-layer-only. Layers added by hand start from the sliders'
           own defs. */
        repeat: {
          min: 1,
          max: 8,
          def: 2,
          defs: [
            { mua: 0.1, mus: 100, g: 0.9, n: 1.4, thickness: 0.3 },
            { mua: 0.1, mus: 50, g: 0.9, n: 1.4, thickness: 1.7 },
          ],
        },
        params: [
          { id: "mua", label: "μ<sub>a</sub> absorption [cm⁻¹]", min: 0.01, max: 5, step: 0.001, def: 0.1, fmt: fmt3 },
          { id: "mus", label: "μ<sub>s</sub> scattering [cm⁻¹]", min: 1, max: 300, step: 0.001, def: 100, fmt: fmt3 },
          { id: "g", label: "g anisotropy factor", min: 0, max: 0.99, step: 0.001, def: 0.9, fmt: fmt3 },
          { id: "n", label: "n refractive index", min: 1.0, max: 1.7, step: 0.001, def: 1.4, fmt: fmt3 },
          { id: "thickness", label: "thickness [cm]", min: 0.01, max: 3, step: 0.001, def: 0.5, fmt: fmt3 },
        ],
      },
      {
        id: "beam",
        title: "Beam & grid",
        params: [
          { id: "p0", label: "P<sub>0</sub> input power [W]", min: 0.01, max: 10, step: 0.001, def: 1.0, fmt: fmt3 },
          BEAM_PROFILE_PARAM,
          BEAM_WIDTH_PARAM,
          BEAM_PATTERN_PARAM,
          PATTERN_COUNT_PARAM,
          PATTERN_SPACING_PARAM,
          { id: "lx", label: "L<sub>x</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "ly", label: "L<sub>y</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "nx", label: "N<sub>x</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
          { id: "ny", label: "N<sub>y</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
          { id: "nz", label: "N<sub>z</sub> voxels (through depth)", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
        ],
      },
    ],
  } as ModelDef<LiemertKienleDerived>,

  fpw1992: {
    label: "Farrell, Patterson & Wilson (1992) — pencil beam, semi-infinite slab",
    command: "fpw1992",
    summaryLine: (derived: Fpw1992Derived, dt: string) =>
      `Done in ${dt} ms — μ<sub>s</sub>' = ${derived.musp.toFixed(3)} cm⁻¹ | ` +
      `D = ${derived.D.toFixed(4)} cm | μ<sub>eff</sub> = ${derived.mueff.toFixed(4)} cm⁻¹ | ` +
      `δ = ${derived.delta.toFixed(3)} cm` + spotsSuffix(derived.spots),

    paramGroups: [
      {
        id: "optical",
        title: "Optical properties",
        params: [
          { id: "mua", label: "μ<sub>a</sub> absorption [cm⁻¹]", min: 0.01, max: 5, step: 0.001, def: 0.1, fmt: fmt3 },
          { id: "mus", label: "μ<sub>s</sub> scattering [cm⁻¹]", min: 1, max: 300, step: 0.001, def: 100, fmt: fmt3 },
          { id: "g", label: "g anisotropy factor", min: 0, max: 0.99, step: 0.001, def: 0.9, fmt: fmt3 },
          { id: "n", label: "n refractive index", min: 1.0, max: 1.7, step: 0.001, def: 1.4, fmt: fmt3 },
        ],
      },
      {
        id: "beam",
        title: "Beam & grid",
        params: [
          { id: "p0", label: "P<sub>0</sub> input power [W]", min: 0.01, max: 10, step: 0.001, def: 1.0, fmt: fmt3 },
          BEAM_PROFILE_PARAM,
          BEAM_WIDTH_PARAM,
          BEAM_PATTERN_PARAM,
          PATTERN_COUNT_PARAM,
          PATTERN_SPACING_PARAM,
          { id: "lx", label: "L<sub>x</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "ly", label: "L<sub>y</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "lz", label: "L<sub>z</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "nx", label: "N<sub>x</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
          { id: "ny", label: "N<sub>y</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
          { id: "nz", label: "N<sub>z</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
        ],
      },
    ],
  } as ModelDef<Fpw1992Derived>,

  kubelkaMunk: {
    label: "Kubelka–Munk — two-flux, N-layer stack (diffuse illumination)",
    command: "kubelka_munk",
    summaryLine: (derived: KubelkaMunkDerived, dt: string) =>
      `Done in ${dt} ms — R = ${derived.R_total.toFixed(4)} | ` +
      `T = ${derived.T_total.toFixed(4)} | absorbed = ${derived.A_total.toFixed(4)} | ` +
      `L<sub>z</sub> = ${derived.Lz.toFixed(3)} cm`,

    /* Unlike FPW1992's pencil beam, KM assumes broad diffuse
       illumination uniform over the top face, so there's no lateral
       beam position — only per-layer optical properties/thickness
       plus the lateral extent (for normalising incident power to a
       flux density) and voxel counts for the 3-D viewer. */
    paramGroups: [
      {
        id: "layers",
        title: "Layers (top → bottom)",
        repeat: { min: 1, max: 8, def: 2 },
        params: [
          { id: "mua", label: "K absorption coeff. [cm⁻¹]", min: 0.001, max: 5, step: 0.001, def: 0.1, fmt: fmt3 },
          { id: "mus", label: "S scattering coeff. [cm⁻¹]", min: 0.01, max: 300, step: 0.001, def: 50, fmt: fmt3 },
          { id: "thickness", label: "thickness [cm]", min: 0.01, max: 2, step: 0.001, def: 0.2, fmt: fmt3 },
        ],
      },
      {
        id: "beam",
        title: "Illumination & lateral grid",
        params: [
          { id: "p0", label: "P<sub>0</sub> incident diffuse power [W]", min: 0.01, max: 10, step: 0.001, def: 1.0, fmt: fmt3 },
          { id: "lx", label: "L<sub>x</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "ly", label: "L<sub>y</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "nx", label: "N<sub>x</sub> voxels", min: 4, max: 400, step: 1, def: 20, fmt: fmt0 },
          { id: "ny", label: "N<sub>y</sub> voxels", min: 4, max: 400, step: 1, def: 20, fmt: fmt0 },
          { id: "nz", label: "N<sub>z</sub> voxels (through depth)", min: 10, max: 400, step: 1, def: 60, fmt: fmt0 },
        ],
      },
    ],
  } as ModelDef<KubelkaMunkDerived>,
};

export function buildModelSelect(): void {
  const sel = document.getElementById("model-select") as HTMLSelectElement;
  sel.innerHTML = "";
  Object.entries(MODELS).forEach(([id, m]) => {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = m.label;
    sel.appendChild(opt);
  });
}

/* ================================================================
   MODEL REGISTRY
   ================================================================
   Each entry describes one theoretical model available in the
   "Model" dropdown. The actual physics (compute + validity checks)
   lives in Rust, under src-tauri/src/physics/ — this file only owns
   the UI-facing bits: the parameter schema, and which Tauri command
   pair to invoke for this model (see src/compute.ts).

   Why parameters live here (per model) rather than in a shared
   global list: different models need different inputs entirely —
   not just different defaults. A single-layer model like FPW1992
   takes one set of optical properties; Kubelka-Munk needs a
   *repeatable* set (one per layer) plus its own grid. Keeping the
   schema next to the model it belongs to means ui-params.ts can stay
   generic.

   paramGroups: array of groups, each rendered as its own panel.
     Plain group:
       { id, title, params: [ {id,label,min,max,step,def,fmt}, ... ] }
       → values merge flat into getParams()'s result, keyed by
         each param's own `id` (e.g. p.mua, p.lx).

     Repeating group (e.g. per-layer parameters):
       { id, title, params: [...], repeat: {min, max, def} }
       → rendered as `def` instances initially, with add/remove-
         instance buttons (bounded by min/max). Values come back as
         an array under result[group.id], one object per instance,
         e.g. p.layers = [ {mua:..., mus:...}, {mua:..., mus:...} ].

   To add a new model in future:
     1. Add a Rust module under src-tauri/src/physics/ with
        derived(), check_validity(), and compute_volume() (see
        fpw1992.rs / kubelka_munk.rs), and register its
        `<name>_summary`/`<name>_volume` commands in lib.rs.
     2. Add one entry below: label, command (matching the Rust
        command prefix), summaryLine(derived, dt), and this model's
        own paramGroups (with its own defaults/ranges — nothing is
        shared with other models unless you explicitly reuse the
        same param objects).

   The dropdown, param panel, run handler, and warning display all
   pick up new entries automatically — no other code needs to change.
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

export interface Fpw1992Derived {
  musp: number;
  D: number;
  mueff: number;
  delta: number;
}

export interface KubelkaMunkDerived {
  R_total: number;
  T_total: number;
  A_total: number;
  Lz: number;
}

export interface LiemertKienleDerived {
  musp1: number;
  D1: number;
  mueff1: number;
  musp2: number;
  D2: number;
  mueff2: number;
  z0: number;
}

export const MODELS: Record<string, ModelDef> = {
  liemertKienle: {
    label: "Liemert & Kienle (2010) — two-layer, point-source diffusion",
    command: "liemert_kienle",
    summaryLine: (derived: LiemertKienleDerived, dt: string) =>
      `Done in ${dt} ms — μ<sub>s1</sub>' = ${derived.musp1.toFixed(3)} cm⁻¹ | ` +
      `D<sub>1</sub> = ${derived.D1.toFixed(4)} cm | μ<sub>eff,1</sub> = ${derived.mueff1.toFixed(4)} cm⁻¹ | ` +
      `z<sub>0</sub> = ${derived.z0.toFixed(3)} cm | μ<sub>eff,2</sub> = ${derived.mueff2.toFixed(4)} cm⁻¹`,

    /* Point source (pencil beam) through two stacked homogeneous layers —
       the combination FPW1992 (point source, one layer) and Kubelka-Munk
       (many layers, diffuse illumination) each stop short of. Layer 1's
       thickness is its own parameter; layer 2 implicitly fills the rest of
       the grid, [t1, Lz], same as how Kubelka-Munk's last layer just runs
       to whatever depth the stack's total thickness works out to. */
    paramGroups: [
      {
        id: "layer1",
        title: "Layer 1 (top)",
        params: [
          { id: "mua1", label: "μ<sub>a1</sub> absorption [cm⁻¹]", min: 0.01, max: 5, step: 0.001, def: 0.1, fmt: fmt3 },
          { id: "mus1", label: "μ<sub>s1</sub> scattering [cm⁻¹]", min: 1, max: 300, step: 0.001, def: 100, fmt: fmt3 },
          { id: "g1", label: "g<sub>1</sub> anisotropy factor", min: 0, max: 0.99, step: 0.001, def: 0.9, fmt: fmt3 },
          { id: "n1", label: "n<sub>1</sub> refractive index", min: 1.0, max: 1.7, step: 0.001, def: 1.4, fmt: fmt3 },
          { id: "t1", label: "thickness [cm]", min: 0.01, max: 2, step: 0.001, def: 0.3, fmt: fmt3 },
        ],
      },
      {
        id: "layer2",
        title: "Layer 2 (bottom, fills the rest of the grid)",
        params: [
          { id: "mua2", label: "μ<sub>a2</sub> absorption [cm⁻¹]", min: 0.01, max: 5, step: 0.001, def: 0.1, fmt: fmt3 },
          { id: "mus2", label: "μ<sub>s2</sub> scattering [cm⁻¹]", min: 1, max: 300, step: 0.001, def: 50, fmt: fmt3 },
          { id: "g2", label: "g<sub>2</sub> anisotropy factor", min: 0, max: 0.99, step: 0.001, def: 0.9, fmt: fmt3 },
          { id: "n2", label: "n<sub>2</sub> refractive index", min: 1.0, max: 1.7, step: 0.001, def: 1.4, fmt: fmt3 },
        ],
      },
      {
        id: "beam",
        title: "Beam & grid",
        params: [
          { id: "p0", label: "P<sub>0</sub> input power [W]", min: 0.01, max: 10, step: 0.001, def: 1.0, fmt: fmt3 },
          BEAM_PROFILE_PARAM,
          BEAM_WIDTH_PARAM,
          { id: "lx", label: "L<sub>x</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "ly", label: "L<sub>y</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "lz", label: "L<sub>z</sub> [cm]", min: 0.5, max: 6, step: 0.001, def: 2, fmt: fmt3 },
          { id: "nx", label: "N<sub>x</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
          { id: "ny", label: "N<sub>y</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
          { id: "nz", label: "N<sub>z</sub> voxels", min: 10, max: 400, step: 1, def: 40, fmt: fmt0 },
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
      `δ = ${derived.delta.toFixed(3)} cm`,

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

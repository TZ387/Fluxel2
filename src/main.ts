import "./styles.css";
import { MODELS, buildModelSelect, type OverlaySpec } from "./models";
import { buildModelParams, getParams } from "./ui-params";
import {
  COLORMAPS,
  colormapLut,
  createPlaneCache,
  drawColorbar,
  drawSlices,
  drawValidityLegend,
  makeScale,
  type PlaneCache,
  type ScaleKind,
  type SliceScene,
  type VolumeView,
} from "./render";
import { runModel } from "./compute";
import { buildHelp } from "./help";

/* ================================================================
   SIMULATION STATE
   ================================================================
   Holds the most recent computed volumes plus the grid dimensions
   they were computed on, as an object rather than loose globals.

   Only the min/max is precomputed per volume — one pass, no
   allocation. The log10 the colour ramp works in is applied per
   drawn pixel instead (render.ts). Precomputing log10 for every
   voxel into a second array is the better trade at Fluxel's
   original ≤80³ grids and the wrong one at the sizes Rust made
   practical: of a 400³ volume's 64M voxels a redraw reads about
   half a million, so that pass did over 100x more work than needed
   (~1.8 s per volume) and doubled peak memory (an extra 512 MB
   across the two). Per-pixel it costs ~25 ms a redraw instead.
   ================================================================ */
type VolumeKind = "phi" | "abs";

interface VolumeCache {
  /** The volume as computed — a view onto the IPC buffer, never copied. */
  data: Float32Array;
  /** Raw bounds. Everything the colour ramp and its bar need follows from
      these through render.ts's makeScale, so switching between the log and
      linear scales is a redraw, not a recompute. */
  vmin: number;
  vmax: number;
}

function buildVolumeCache(vol: Float32Array): VolumeCache {
  let vmin = Infinity,
    vmax = -Infinity;
  for (let i = 0; i < vol.length; i++) {
    if (vol[i] > vmax) vmax = vol[i];
    if (vol[i] < vmin) vmin = vol[i];
  }
  return { data: vol, vmin, vmax };
}

const Simulation = {
  nx: 40,
  ny: 40,
  nz: 40,
  /** Grid extents [cm], so the plots can be labelled in the units every
      parameter is given in: x and y are centred on the beam axis, z runs
      from 0 at the surface down to lz (see beam.rs). */
  lx: 2,
  ly: 2,
  lz: 2,
  /** Depths [cm] of the internal layer interfaces — cumulative thicknesses,
      the bottom of the stack excluded. Empty for a homogeneous model. */
  interfaces: [] as number[],
  /** Bumped per stored result, so a redraw can tell "same volume, different
      camera" from "new volume" — see the plane-image stamp in buildScene. */
  runId: 0,
  phi: null as VolumeCache | null,
  abs: null as VolumeCache | null,
  /** Per-voxel overlay codes for the current run, shared by both plots
      (they grade the run itself, not the field, so they don't differ
      between the phi and abs volumes). Null for models that don't compute
      one. */
  validity: null as Uint8Array | null,
  /** What those codes mean, carried alongside them rather than read back off
      the dropdown at draw time: the two can disagree, since switching model
      rebuilds the panel without discarding the volume already computed. */
  overlay: null as OverlaySpec | null,

  /** Store a freshly computed result, and the grid and geometry it used. */
  set(r: {
    nx: number;
    ny: number;
    nz: number;
    lx: number;
    ly: number;
    lz: number;
    interfaces: number[];
    phi: Float32Array;
    abs: Float32Array;
    validity: Uint8Array | null;
    overlay: OverlaySpec | null;
  }) {
    this.nx = r.nx;
    this.ny = r.ny;
    this.nz = r.nz;
    this.lx = r.lx;
    this.ly = r.ly;
    this.lz = r.lz;
    this.interfaces = r.interfaces;
    this.runId++;
    this.phi = buildVolumeCache(r.phi);
    this.abs = buildVolumeCache(r.abs);
    this.validity = r.validity;
    this.overlay = r.overlay;
  },

  /** 'phi' | 'abs' → the matching cache, or null if not yet computed. */
  volume(suffix: VolumeKind): VolumeCache | null {
    return suffix === "phi" ? this.phi : this.abs;
  },

  hasData(): boolean {
    return this.phi !== null;
  },
};

/* ================================================================
   AXIS SLIDERS FOR EACH PLOT
   ================================================================
   The slider still steps in voxels — that is the resolution the
   answer exists at — but reads out in cm, which is what the plot's
   own axes and every parameter in the panel are quoted in. A voxel
   index only meant anything when the plot labelled itself with one.
   ================================================================ */

/** Where a slice plane sits, in cm: voxel *centres*, since that is where
    the field was evaluated (beam.rs's sample_axisymmetric_volume). */
function axisPosition(ax: "x" | "y" | "z", i: number): number {
  if (ax === "x") return ((i + 0.5) * Simulation.lx) / Simulation.nx - Simulation.lx / 2;
  if (ax === "y") return ((i + 0.5) * Simulation.ly) / Simulation.ny - Simulation.ly / 2;
  return ((i + 0.5) * Simulation.lz) / Simulation.nz;
}

const fmtPos = (v: number) => `${v.toFixed(3)} cm`;

function buildAxisSliders(containerId: string, suffix: VolumeKind): void {
  const container = document.getElementById(containerId)!;
  container.innerHTML = "";
  (["x", "y", "z"] as const).forEach((ax) => {
    const dim = ax === "x" ? Simulation.nx : ax === "y" ? Simulation.ny : Simulation.nz;
    const defV = Math.floor(dim / 2);
    const row = document.createElement("div");
    row.className = "axis-row";
    row.innerHTML = `
      <span class="axis-lbl">${ax}</span>
      <input type="range" id="s${ax}-${suffix}" min="0" max="${dim - 1}" step="1" value="${defV}">
      <span class="axis-val" id="s${ax}-${suffix}-v">${fmtPos(axisPosition(ax, defV))}</span>`;
    container.appendChild(row);

    const el = row.querySelector("input") as HTMLInputElement;
    const out = row.querySelector(".axis-val") as HTMLElement;
    el.addEventListener("input", () => {
      out.textContent = fmtPos(axisPosition(ax, +el.value));
      redraw(suffix);
    });
  });
}

/** The ModelDef for whatever's currently picked in the model dropdown. */
function selectedModel() {
  return MODELS[(document.getElementById("model-select") as HTMLSelectElement).value];
}

function getSlice(suffix: VolumeKind): { ix: number; iy: number; iz: number } {
  return {
    ix: +(document.getElementById(`sx-${suffix}`) as HTMLInputElement).value,
    iy: +(document.getElementById(`sy-${suffix}`) as HTMLInputElement).value,
    iz: +(document.getElementById(`sz-${suffix}`) as HTMLInputElement).value,
  };
}

/** Reads straight from the checkbox rather than tracking a copy of its
    state — same pattern getSlice above uses for the axis sliders. */
function getShowValidity(suffix: VolumeKind): boolean {
  return (document.getElementById(`vchk-${suffix}`) as HTMLInputElement).checked;
}

/* ================================================================
   VIEW CONTROLS
   ================================================================
   The colour scale and the colormap are shared by both plots rather
   than duplicated per panel: fluence and absorption are read against
   each other, and they can only be if they are drawn the same way. The
   per-panel controls stay per-panel — the slice planes and the overlay
   toggle are questions you ask of one field at a time.

   Read straight off the controls at draw time, the same way getSlice
   reads the sliders, so there is no second copy of the state to keep
   in step.
   ================================================================ */
/** One plane-image cache per panel, so the two plots don't evict each
    other's slices on every redraw. */
const planeCaches: Record<VolumeKind, PlaneCache> = {
  phi: createPlaneCache(),
  abs: createPlaneCache(),
};

function getScaleKind(): ScaleKind {
  return (document.getElementById("view-scale") as HTMLSelectElement).value as ScaleKind;
}

function getColormapId(): string {
  return (document.getElementById("view-cmap") as HTMLSelectElement).value;
}

/** Everything the renderer needs for one panel, in physical units. */
function buildScene(suffix: VolumeKind): SliceScene | null {
  const cache = Simulation.volume(suffix);
  if (!cache) return null;
  const { nx, ny, nz } = Simulation;
  const clamp = (v: number, n: number) => Math.max(0, Math.min(n - 1, v));
  const { ix, iy, iz } = getSlice(suffix);
  const showValidity = getShowValidity(suffix) && Simulation.validity !== null && Simulation.overlay !== null;
  const view: VolumeView = {
    data: cache.data,
    nx,
    ny,
    nz,
    scale: makeScale(getScaleKind(), cache.vmin, cache.vmax),
    lut: colormapLut(getColormapId()),
    validity: Simulation.validity,
    showValidity,
    /* Everything that decides a voxel's colour, and nothing that doesn't —
       notably not the camera, which is what makes an orbit cheap. */
    stamp: `${Simulation.runId}:${suffix}:${getScaleKind()}:${getColormapId()}:${showValidity ? 1 : 0}`,
  };
  return {
    view,
    lx: Simulation.lx,
    ly: Simulation.ly,
    lz: Simulation.lz,
    ix: clamp(ix, nx),
    iy: clamp(iy, ny),
    iz: clamp(iz, nz),
    interfaces: Simulation.interfaces,
  };
}

function redraw(suffix: VolumeKind): void {
  const scene = buildScene(suffix);
  if (!scene) return;
  drawSlices(`cv-${suffix}`, scene, planeCaches[suffix]);

  if (scene.view.showValidity) drawValidityLegend(`cbar-${suffix}`, Simulation.overlay!.legend);
  else drawColorbar(`cbar-${suffix}`, scene.view.scale, scene.view.lut);
}

function redrawAll(): void {
  if (!Simulation.hasData()) return;
  redraw("phi");
  redraw("abs");
}

/* ================================================================
   RESPONSIVE CANVAS RESIZE
   ================================================================
   Canvases are sized in JS (cv.width/height) to match their rendered
   CSS pixel size, so they stay crisp at any zoom. A ResizeObserver on
   the canvases themselves (rather than a window 'resize' listener)
   catches every reason their box can change size, and already
   coalesces to one callback per frame — no manual debounce needed.
   ================================================================ */
function syncCanvasSizes(): void {
  ["cv-phi", "cv-abs"].forEach((id) => {
    const cv = document.getElementById(id) as HTMLCanvasElement;
    cv.width = cv.offsetWidth || 400;
    cv.height = cv.offsetHeight || 400;
  });
}

const canvasResizeObserver = new ResizeObserver(() => {
  if (!Simulation.hasData()) return;
  syncCanvasSizes();
  redrawAll();
});
["cv-phi", "cv-abs"].forEach((id) => canvasResizeObserver.observe(document.getElementById(id)!));

/* ================================================================
   MAIN RUN HANDLER
   ================================================================
   The compute can fail for reasons the parameter panel can't rule
   out on its own — a grid too large to allocate, or a value the
   backend won't accept — so the click handler is only the shell:
   it disables the button, hands off to runAndRender(), and makes
   sure that whatever happens the button comes back and the status
   line stops saying "Computing…". Without that, one failed run
   leaves the UI stuck until reload.
   ================================================================ */
async function runAndRender(): Promise<void> {
  const p = getParams(); // reads whatever controls the current model's paramGroups produced
  const st = document.getElementById("status")!;
  const model = selectedModel();

  /* Monte Carlo runs for seconds, not milliseconds, so it reports how far
     along it is (models.ts's `progress`). The run is off on a worker thread
     either way — this only gives the wait something to show. */
  const onProgress = model.progress
    ? (fraction: number) => {
        st.textContent = `Computing… ${Math.round(fraction * 100)}%`;
      }
    : undefined;

  const t0 = performance.now();
  const { phi, abs, validity, derived, valid, reasons } = await runModel(model.command, p, onProgress);
  const dt = (performance.now() - t0).toFixed(1);

  /* The grid's depth is an input for the one homogeneous model and a
     derived quantity for the three layered ones, where it is the stack's
     total thickness — so take it from `derived` when the model reports it
     (models.ts's `Lz`) and fall back to the parameter otherwise. The
     interfaces are the partial sums of the same thicknesses. */
  const lz = Number.isFinite(derived?.Lz) ? (derived.Lz as number) : (p.lz as number);
  const interfaces: number[] = [];
  if (Array.isArray(p.layers)) {
    let d = 0;
    for (let i = 0; i < p.layers.length - 1; i++) {
      d += p.layers[i].thickness;
      interfaces.push(d);
    }
  }

  Simulation.set({
    nx: p.nx,
    ny: p.ny,
    nz: p.nz,
    lx: p.lx,
    ly: p.ly,
    lz,
    interfaces,
    phi,
    abs,
    validity,
    overlay: model.overlay ?? null,
  });

  /* Show plots section */
  (document.getElementById("plots") as HTMLElement).style.display = "";

  /* The toggle only makes sense for a model that computed an overlay buffer
     — hide it, and reset it unchecked, for the rest rather than leaving a
     control that does nothing. Its wording is the model's (models.ts's
     `overlay`), since what the overlay grades differs between them. */
  const overlay = validity === null ? null : model.overlay ?? null;
  (["phi", "abs"] as const).forEach((suffix) => {
    const row = document.getElementById(`vtoggle-${suffix}`) as HTMLElement;
    row.hidden = overlay === null;
    if (overlay === null) (document.getElementById(`vchk-${suffix}`) as HTMLInputElement).checked = false;
    else document.getElementById(`vlbl-${suffix}`)!.textContent = overlay.toggle;
  });

  /* Rebuild sliders with correct max values */
  buildAxisSliders("sl-phi", "phi");
  buildAxisSliders("sl-abs", "abs");

  /* Resize canvases to match their rendered pixel width */
  syncCanvasSizes();

  redrawAll();

  st.innerHTML = model.summaryLine(derived, dt);

  if (!valid) {
    const warn = document.createElement("div");
    warn.className = "status-warn";
    const intro = document.createElement("p");
    intro.textContent = `⚠ ${model.warningIntro}`;
    warn.appendChild(intro);
    reasons.forEach((reason) => {
      /* innerHTML, not textContent: these carry <sub> markup, and they're
         authored in the model's own Rust source, not user input. */
      const para = document.createElement("p");
      para.innerHTML = reason + ".";
      warn.appendChild(para);
    });
    st.appendChild(warn);
  }
}

document.getElementById("run-btn")!.addEventListener("click", async () => {
  const btn = document.getElementById("run-btn") as HTMLButtonElement;
  const st = document.getElementById("status")!;

  btn.disabled = true;
  st.textContent = "Computing…";

  try {
    await runAndRender();
  } catch (err) {
    console.error(err);
    st.textContent = "";
    const box = document.createElement("div");
    box.className = "status-error";
    /* textContent, not innerHTML: unlike the validity reasons above, this
       string is whatever the backend threw, so it isn't trusted markup. */
    box.textContent = `✖ Compute failed — ${err instanceof Error ? err.message : String(err)}`;
    st.appendChild(box);
    /* Any plots on screen are from the previous successful run, so they're
       left alone rather than cleared — the message says this one failed. */
  } finally {
    btn.disabled = false;
  }
});

/* ================================================================
   MODEL SWITCHING
   ================================================================
   Each model owns its own paramGroups (see models.ts), so switching
   means tearing down and rebuilding the whole parameter panel, not
   just resetting values. Plots from the previous model are hidden
   since they'd no longer match the current inputs.
   ================================================================ */
function onModelChange(): void {
  const model = selectedModel();
  buildModelParams(model, "param-panels");
  (document.getElementById("plots") as HTMLElement).style.display = "none";
  document.getElementById("status")!.textContent = "Adjust parameters and click Compute.";
}

/* ================================================================
   TABS
   ================================================================ */
function switchTab(tab: "simulator" | "help"): void {
  (document.getElementById("tab-simulator") as HTMLElement).style.display = tab === "simulator" ? "" : "none";
  (document.getElementById("tab-help") as HTMLElement).style.display = tab === "help" ? "" : "none";
  document.getElementById("tab-btn-simulator")!.classList.toggle("active", tab === "simulator");
  document.getElementById("tab-btn-help")!.classList.toggle("active", tab === "help");
}
document.getElementById("tab-btn-simulator")!.addEventListener("click", () => switchTab("simulator"));
document.getElementById("tab-btn-help")!.addEventListener("click", () => switchTab("help"));

/* ================================================================
   INIT
   ================================================================ */
function buildViewControls(): void {
  const cmap = document.getElementById("view-cmap") as HTMLSelectElement;
  cmap.innerHTML = COLORMAPS.map((c) => `<option value="${c.id}">${c.label}</option>`).join("");
  ["view-scale", "view-cmap"].forEach((id) =>
    document.getElementById(id)!.addEventListener("change", redrawAll)
  );
}

buildViewControls();
buildModelSelect();
document.getElementById("model-select")!.addEventListener("change", onModelChange);
onModelChange();
buildHelp("tab-help");

(["phi", "abs"] as const).forEach((suffix) => {
  document.getElementById(`vchk-${suffix}`)!.addEventListener("change", () => redraw(suffix));
});

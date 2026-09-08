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
  pickFlat,
  type PlaneCache,
  type Probe,
  type ScaleKind,
  type SliceScene,
  type VolumeView,
} from "./render";
import { DEFAULT_CAMERA, EL_LIMIT, drawBox3D, pick3D, type Camera } from "./render3d";
import { runModel } from "./compute";
import { buildHelp } from "./help";
import { parseSettings, serializeSettings, type ViewMode, type ViewSettings } from "./settings";
import { invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";

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

/** Units for each field, for the hover readout — the plot titles carry them
    for the eye, but the readout quotes a number and has to say what of. */
const UNITS: Record<VolumeKind, string> = { phi: "W/cm\u00B2", abs: "W/cm\u00B3" };
const SYMBOL: Record<VolumeKind, string> = { phi: "\u03A6", abs: "A" };

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
   Layout, colour scale, colormap and the camera are shared by both
   plots rather than duplicated per panel: fluence and absorption are
   read against each other, and they can only be if they are drawn the
   same way and from the same angle. The per-panel controls stay
   per-panel — the slice planes and the overlay toggle are questions
   you ask of one field at a time.

   Read straight off the controls at draw time, the same way getSlice
   reads the sliders, so there is no second copy of the state to keep
   in step.
   ================================================================ */
const camera: Camera = { ...DEFAULT_CAMERA };

/** One plane-image cache per panel, so the two plots don't evict each
    other's slices on every redraw. */
const planeCaches: Record<VolumeKind, PlaneCache> = {
  phi: createPlaneCache(),
  abs: createPlaneCache(),
};

function getViewMode(): ViewMode {
  return (document.getElementById("view-mode") as HTMLSelectElement).value as ViewMode;
}

function getScaleKind(): ScaleKind {
  return (document.getElementById("view-scale") as HTMLSelectElement).value as ScaleKind;
}

function getColormapId(): string {
  return (document.getElementById("view-cmap") as HTMLSelectElement).value;
}

/** Everything a renderer needs for one panel, in physical units. Both
    renderers take the same scene, so the layout switch is a choice of
    function and nothing else. */
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
  const cv = document.getElementById(`cv-${suffix}`) as HTMLCanvasElement;
  const box3d = getViewMode() === "box3d";
  /* Both layouts are hoverable, so neither gets the default arrow: grab says
     the box can be turned, crosshair says the flat panels can be read. */
  cv.style.cursor = box3d ? "grab" : "crosshair";
  if (box3d) drawBox3D(`cv-${suffix}`, scene, planeCaches[suffix], camera);
  else drawSlices(`cv-${suffix}`, scene, planeCaches[suffix]);

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
   STATUS LINE
   ================================================================
   Shared by the run and by the two settings-file actions, so the
   three don't each grow their own idea of what a message looks like.
   ================================================================ */

/** Replace the status line with `text`, plus a block of advisory notes —
    what a loaded file had substituted or clamped, say. */
function showStatus(text: string, notes: string[] = []): void {
  const st = document.getElementById("status")!;
  st.textContent = text;
  if (notes.length === 0) return;
  const box = document.createElement("div");
  box.className = "status-warn";
  notes.forEach((note) => {
    const para = document.createElement("p");
    /* textContent, not innerHTML: unlike a model's own validity reasons,
       these quote values out of a file this app didn't write. */
    para.textContent = note;
    box.appendChild(para);
  });
  st.appendChild(box);
}

/** A failed action, as opposed to a completed one with something to say. */
function showStatusError(text: string): void {
  const st = document.getElementById("status")!;
  st.textContent = "";
  const box = document.createElement("div");
  box.className = "status-error";
  box.textContent = text;
  st.appendChild(box);
}

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
    /* Whatever the backend threw is not trusted markup, which is why
       showStatusError sets it as text. Any plots on screen are from the
       previous successful run, so they're left alone rather than cleared —
       the message says this one failed. */
    showStatusError(`✖ Compute failed — ${err instanceof Error ? err.message : String(err)}`);
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

/** Rebuild the parameter panel for whichever model is selected, optionally
    seeded with a loaded file's values (see settings.ts), and put the plots
    away — they were computed from the parameters that have just been
    replaced, so leaving them up would invite reading them as this model's. */
function rebuildPanel(seed?: Record<string, any>): void {
  buildModelParams(selectedModel(), "param-panels", seed);
  (document.getElementById("plots") as HTMLElement).style.display = "none";
}

function onModelChange(): void {
  rebuildPanel();
  showStatus("Adjust parameters and click Compute.");
}

/* ================================================================
   SETTINGS FILES
   ================================================================
   Save the parameter panel to a JSON file and load it back.
   settings.ts owns the file's shape and the checking of one that
   comes back; src-tauri/src/lib.rs owns the two commands that touch
   the disk; the dialogs are the plugin's. So all that's here is the
   wiring — and, notably, a load path that goes through the very same
   buildModelParams the model dropdown uses, so a file can only put
   the panel into a state it could have been put into by hand.

   What a file remembers is the parameters, the layer names, and the
   shared view controls. What it doesn't is anything that belongs to
   a *result*: the slice-plane positions are indices into whatever
   grid the run happened to use, and the volumes themselves are the
   Export item's business, not this one's.
   ================================================================ */
const SETTINGS_FILTERS = [{ name: "Fluxel settings", extensions: ["json"] }];

function getViewSettings(): ViewSettings {
  return {
    mode: getViewMode(),
    scale: getScaleKind(),
    cmap: getColormapId(),
    /* Copied, not referenced: an orbit mutates the live camera in place. */
    camera: { ...camera },
  };
}

function applyViewSettings(view: ViewSettings): void {
  (document.getElementById("view-mode") as HTMLSelectElement).value = view.mode;
  (document.getElementById("view-scale") as HTMLSelectElement).value = view.scale;
  (document.getElementById("view-cmap") as HTMLSelectElement).value = view.cmap;
  Object.assign(camera, view.camera);
  /* Setting a select's value fires no change event, so the hint that the
     'change' handler would have updated is updated here. */
  document.getElementById("view-hint")!.hidden = view.mode !== "box3d";
}

async function saveSettings(): Promise<void> {
  const model = (document.getElementById("model-select") as HTMLSelectElement).value;
  const path = await save({
    title: "Save settings",
    defaultPath: `${model}-settings.json`,
    filters: SETTINGS_FILTERS,
  });
  if (path === null) return; // dialog cancelled
  const contents = serializeSettings(model, getParams(), getViewSettings());
  await invoke("write_text_file", { path, contents });
  showStatus(`Saved to ${path}`);
}

async function loadSettings(): Promise<void> {
  const path = await open({
    title: "Load settings",
    multiple: false,
    directory: false,
    filters: SETTINGS_FILTERS,
  });
  if (typeof path !== "string") return; // dialog cancelled
  const loaded = parseSettings(await invoke<string>("read_text_file", { path }));

  (document.getElementById("model-select") as HTMLSelectElement).value = loaded.model;
  rebuildPanel(loaded.params);
  if (loaded.view) applyViewSettings(loaded.view);

  /* The warnings say what the file couldn't be taken at its word about — a
     parameter this build doesn't have, a layer count the model won't allow.
     The panel holds a usable set of values either way, which is why this is
     a note under a loaded file rather than a failure. */
  const name = path.split(/[/\\]/).pop() || path;
  showStatus(`Loaded ${name} — click Compute & visualise.`, loaded.warnings);
}

/* ================================================================
   PLOT EXPORT
   ================================================================
   One PNG per panel, composed on an offscreen canvas from the plot's
   own two canvases — the slices/box (cv-<suffix>) and its colorbar
   (cbar-<suffix>, render.ts) — so the file matches whatever is on
   screen (either renderer, either colour scale) and always says what
   the colour means, which neither canvas alone does.

   write_binary_file (lib.rs) is write_text_file's sibling for bytes
   that aren't valid UTF-8; the dialog is the same plugin the settings
   files use.
   ================================================================ */
const EXPORT_FILTERS = [{ name: "PNG image", extensions: ["png"] }];

/** Panel background/text/font, matched by hand from styles.css's `--bg2`,
    `--text` and `--mono` — a canvas can't read a CSS custom property. */
function composePlotPng(suffix: VolumeKind): Promise<Blob> {
  const mainCv = document.getElementById(`cv-${suffix}`) as HTMLCanvasElement;
  const barCv = document.getElementById(`cbar-${suffix}`) as HTMLCanvasElement;
  const title = mainCv.closest(".plot-panel")!.querySelector(".plot-title")!.textContent ?? "";

  const pad = 16,
    gap = 16,
    titleH = 24;
  const out = document.createElement("canvas");
  out.width = pad * 2 + mainCv.width + gap + barCv.width;
  out.height = pad * 2 + titleH + Math.max(mainCv.height, barCv.height);
  const ctx = out.getContext("2d")!;

  ctx.fillStyle = "#161b22";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.fillStyle = "#e6edf3";
  ctx.font = "14px 'Cascadia Code', 'Fira Mono', 'Consolas', monospace";
  ctx.textBaseline = "top";
  ctx.fillText(title, pad, pad);

  ctx.drawImage(mainCv, pad, pad + titleH);
  ctx.drawImage(barCv, pad + mainCv.width + gap, pad + titleH + (mainCv.height - barCv.height) / 2);

  return new Promise((resolve, reject) =>
    out.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("could not encode PNG"))), "image/png")
  );
}

async function exportPlot(suffix: VolumeKind): Promise<void> {
  const model = (document.getElementById("model-select") as HTMLSelectElement).value;
  const path = await save({
    title: "Export plot",
    defaultPath: `${model}-${suffix}.png`,
    filters: EXPORT_FILTERS,
  });
  if (path === null) return; // dialog cancelled
  const blob = await composePlotPng(suffix);
  const bytes = Array.from(new Uint8Array(await blob.arrayBuffer()));
  await invoke("write_binary_file", { path, contents: bytes });
  showStatus(`Exported to ${path}`);
}

/* ================================================================
   DATA EXPORT
   ================================================================
   One field's volume per button, as a .npy file plus a small .json
   sidecar of what a bare array doesn't carry (the grid it was
   evaluated on, in cm, and which field and model it is). Chosen over
   CSV or JSON for the array itself because the grid this app allows
   goes up to 400^3 = 64M voxels — a plain-text encoding of that is
   hundreds of megabytes; .npy is the raw float bytes plus a short
   header, readable with one call in Python (`numpy.load`) and Julia
   (NPZ.jl), and needs no dependency here to write.

   The array already sits in memory exactly as .npy wants it: voxel
   (ix,iy,iz) lives at data[ix + iy*nx + iz*nx*ny] (render.ts's
   probeAt uses the same arithmetic), which is C order for shape
   (nz, ny, nx) — so the flat Float32Array goes into the file as-is,
   no reordering.
   ================================================================ */
const DATA_FILTERS = [{ name: "NumPy array", extensions: ["npy"] }];

/** A .npy v1.0 file's bytes for `data`, declared as `shape`. The data itself
    is a view onto the same buffer, not a copy. */
function encodeNpy(data: Float32Array, shape: readonly number[]): Uint8Array {
  const shapeStr = `(${shape.join(", ")}${shape.length === 1 ? "," : ""})`;
  const MAGIC_AND_VERSION = 8; // "\x93NUMPY" + 2 version bytes
  const LEN_FIELD = 2;
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': ${shapeStr}, }`;
  /* Padded with spaces so the magic+version+len-field+header is a multiple
     of 64 bytes — not required for correctness, but every real writer does
     it, so a reader that assumes it (some do) still works. */
  const pad = (64 - ((MAGIC_AND_VERSION + LEN_FIELD + header.length + 1) % 64)) % 64;
  header += " ".repeat(pad) + "\n";
  const headerBytes = new TextEncoder().encode(header);

  const out = new Uint8Array(MAGIC_AND_VERSION + LEN_FIELD + headerBytes.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0); // \x93NUMPY, version 1.0
  new DataView(out.buffer).setUint16(MAGIC_AND_VERSION, headerBytes.length, true);
  out.set(headerBytes, MAGIC_AND_VERSION + LEN_FIELD);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), MAGIC_AND_VERSION + LEN_FIELD + headerBytes.length);
  return out;
}

/** Base64, via the browser's own encoder rather than a hand-written loop —
    at a volume export's largest size (hundreds of MB) that is both faster
    and safer than building one JS string a chunk at a time. */
function bytesToBase64(bytes: Uint8Array): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve((reader.result as string).split(",", 2)[1]);
    reader.onerror = () => reject(reader.error ?? new Error("could not encode base64"));
    reader.readAsDataURL(new Blob([bytes]));
  });
}

async function exportData(suffix: VolumeKind): Promise<void> {
  const cache = Simulation.volume(suffix);
  if (!cache) return; // button is only reachable once a run has produced one

  const model = (document.getElementById("model-select") as HTMLSelectElement).value;
  const path = await save({
    title: "Export data",
    defaultPath: `${model}-${suffix}.npy`,
    filters: DATA_FILTERS,
  });
  if (path === null) return; // dialog cancelled

  const { nx, ny, nz, lx, ly, lz, interfaces } = Simulation;
  const base64 = await bytesToBase64(encodeNpy(cache.data, [nz, ny, nx]));
  await invoke("write_base64_file", { path, base64 });

  /* Alongside the array, not merged into one file: keeps the .npy a plain
     array any reader can open with nothing but numpy, and the metadata
     inspectable without one. */
  const metaPath = `${path.replace(/\.npy$/i, "")}.json`;
  const meta = {
    field: suffix,
    units: UNITS[suffix],
    shape: [nz, ny, nx],
    axes: ["z", "y", "x"],
    nx,
    ny,
    nz,
    lx,
    ly,
    lz,
    interfaces,
    vmin: cache.vmin,
    vmax: cache.vmax,
    model,
  };
  await invoke("write_text_file", { path: metaPath, contents: JSON.stringify(meta, null, 2) + "\n" });

  showStatus(`Exported to ${path} (+ ${metaPath.split(/[/\\]/).pop()})`);
}

/** Both file buttons behave the same way: a dialog that may be cancelled
    (in which case nothing happens at all), followed by work that can fail on
    something outside the app's control — an unreadable file, or one that
    isn't a settings file. That belongs in the status line, not the console,
    and the button has to come back either way. */
function bindFileButton(id: string, action: () => Promise<void>, failure: string): void {
  const btn = document.getElementById(id) as HTMLButtonElement;
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    try {
      await action();
    } catch (err) {
      console.error(err);
      showStatusError(`✖ ${failure} — ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      btn.disabled = false;
    }
  });
}

bindFileButton("save-btn", saveSettings, "Could not save the settings");
bindFileButton("load-btn", loadSettings, "Could not load the settings");
bindFileButton("export-phi-btn", () => exportPlot("phi"), "Could not export the plot");
bindFileButton("export-abs-btn", () => exportPlot("abs"), "Could not export the plot");
bindFileButton("export-phi-data-btn", () => exportData("phi"), "Could not export the data");
bindFileButton("export-abs-data-btn", () => exportData("abs"), "Could not export the data");

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
   HOVER READOUT
   ================================================================
   The plots could be read but not interrogated: you could see that a
   region was bright without being able to put a number on it, which
   for a simulator is most of the point. Hovering now names the
   position in cm and the value there — and the overlay code too when
   the overlay is on, which is the only way to see *which* band a
   particular voxel fell in rather than just its colour.

   A DOM element rather than something drawn into the canvas, so
   following the cursor costs no redraw at all: the plot itself only
   changes when the slices or the camera do.
   ================================================================ */
function formatProbe(suffix: VolumeKind, probe: Probe): string {
  const lines = [
    `x ${probe.x.toFixed(3)}  y ${probe.y.toFixed(3)}  z ${probe.z.toFixed(3)} cm`,
    `${SYMBOL[suffix]} = ${probe.value.toPrecision(4)} ${UNITS[suffix]}`,
  ];
  /* Only meaningful while the overlay is the thing being drawn — the codes
     grade the run, and quoting one next to a field value the user is not
     looking at would just be noise. */
  if (probe.code !== null && getShowValidity(suffix) && Simulation.overlay) {
    lines.push(Simulation.overlay.legend[probe.code] ?? "");
  }
  return lines.join("\n");
}

/** Canvas-pixel coordinates for a pointer event. The canvas is sized in CSS
    pixels, but a browser zoom or a fractional layout makes the two differ, so
    the ratio is taken from the element rather than assumed to be 1. */
function canvasPoint(cv: HTMLCanvasElement, e: PointerEvent): { px: number; py: number } {
  const r = cv.getBoundingClientRect();
  return {
    px: ((e.clientX - r.left) * cv.width) / (r.width || 1),
    py: ((e.clientY - r.top) * cv.height) / (r.height || 1),
  };
}

function updateReadout(suffix: VolumeKind, e: PointerEvent | null): void {
  const box = document.getElementById(`hov-${suffix}`) as HTMLElement;
  const cv = document.getElementById(`cv-${suffix}`) as HTMLCanvasElement;
  const scene = e && Simulation.hasData() ? buildScene(suffix) : null;
  if (!scene || !e) {
    box.hidden = true;
    return;
  }
  const { px, py } = canvasPoint(cv, e);
  const probe =
    getViewMode() === "box3d"
      ? pick3D(scene, cv.width, cv.height, camera, px, py)
      : pickFlat(scene, cv.width, cv.height, px, py);
  if (!probe) {
    box.hidden = true;
    return;
  }
  box.textContent = formatProbe(suffix, probe);
  box.hidden = false;

  /* Offset from the cursor, flipped near an edge so the readout stays inside
     the plot instead of forcing the panel to scroll. */
  const pad = 14;
  const wrapW = cv.offsetWidth || cv.width;
  const wrapH = cv.offsetHeight || cv.height;
  const cx = ((px / cv.width) * wrapW) | 0;
  const cy = ((py / cv.height) * wrapH) | 0;
  const flipX = cx + pad + box.offsetWidth > wrapW;
  const flipY = cy + pad + box.offsetHeight > wrapH;
  box.style.left = `${Math.max(0, flipX ? cx - pad - box.offsetWidth : cx + pad)}px`;
  box.style.top = `${Math.max(0, flipY ? cy - pad - box.offsetHeight : cy + pad)}px`;
}

/* ================================================================
   POINTER: ORBIT AND HOVER
   ================================================================
   One set of handlers for both, since they are the same gestures on
   the same canvas: a move with no button down is a hover, a drag is an
   orbit, and the two must not happen at once.

   Orbit rather than a fixed viewpoint, because the box is drawn at
   equal aspect: a thin stack seen from the default angle is nearly
   edge-on and needs tilting to be read at all, and no single angle
   suits both a 2 cm cube and a 0.3 mm film. "Grab the object"
   convention — drag right and the near face follows the pointer — to
   match the cursor the canvas shows. There is no zoom; see the note on
   Camera in render3d.ts for why.
   ================================================================ */
function bindPointer(suffix: VolumeKind): void {
  const cv = document.getElementById(`cv-${suffix}`) as HTMLCanvasElement;
  let lastX = 0,
    lastY = 0,
    dragging = false;

  cv.addEventListener("pointerdown", (e) => {
    if (getViewMode() !== "box3d" || !Simulation.hasData()) return;
    /* The readout would otherwise sit in the middle of the drag, describing a
       point the cursor has already left. */
    updateReadout(suffix, null);
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    cv.setPointerCapture(e.pointerId);
    cv.style.cursor = "grabbing";
  });

  cv.addEventListener("pointermove", (e) => {
    if (!dragging) {
      updateReadout(suffix, e);
      return;
    }
    camera.az -= (e.clientX - lastX) * 0.01;
    camera.el = Math.max(-EL_LIMIT, Math.min(EL_LIMIT, camera.el + (e.clientY - lastY) * 0.01));
    lastX = e.clientX;
    lastY = e.clientY;
    redrawAll();
  });

  const stop = (e: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId);
    cv.style.cursor = "grab";
  };
  cv.addEventListener("pointerup", stop);
  cv.addEventListener("pointercancel", stop);
  cv.addEventListener("pointerleave", () => updateReadout(suffix, null));

  cv.addEventListener("dblclick", () => {
    if (getViewMode() !== "box3d") return;
    Object.assign(camera, DEFAULT_CAMERA);
    redrawAll();
  });
}

/* ================================================================
   INIT
   ================================================================ */
function buildViewControls(): void {
  const cmap = document.getElementById("view-cmap") as HTMLSelectElement;
  cmap.innerHTML = COLORMAPS.map((c) => `<option value="${c.id}">${c.label}</option>`).join("");
  ["view-mode", "view-scale", "view-cmap"].forEach((id) =>
    document.getElementById(id)!.addEventListener("change", () => {
      /* The hint only describes what the 3-D box does with a drag. */
      document.getElementById("view-hint")!.hidden = getViewMode() !== "box3d";
      redrawAll();
    })
  );
  (["phi", "abs"] as const).forEach(bindPointer);
}

buildViewControls();
buildModelSelect();
document.getElementById("model-select")!.addEventListener("change", onModelChange);
onModelChange();
buildHelp("tab-help");

(["phi", "abs"] as const).forEach((suffix) => {
  document.getElementById(`vchk-${suffix}`)!.addEventListener("change", () => redraw(suffix));
});

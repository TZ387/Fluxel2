import "./styles.css";
import { MODELS, buildModelSelect } from "./models";
import { buildModelParams, getParams } from "./ui-params";
import { drawSlices, drawColorbar } from "./render";
import { runModel } from "./compute";

/* ================================================================
   SIMULATION STATE
   ================================================================
   Holds the most recent computed volumes plus the grid dimensions
   they were computed on. Wrapped in an object (rather than loose
   globals) so the fields it owns and the ways it can be mutated
   are explicit and in one place.

   Each volume's log-scale transform and min/max are precomputed once
   here (an O(nx*ny*nz) pass) rather than in redraw() — redraw() runs
   on every axis-slider drag, which only changes which 2D slice is
   shown, not the underlying data, so redoing that full-volume pass
   per drag was wasted work (invisible at Fluxel's original ≤80³
   grids, very much not at the higher resolutions Rust now makes
   practical — e.g. a slider drag at 400³ was re-running ~64M
   Math.log10 calls before this).
   ================================================================ */
type VolumeKind = "phi" | "abs";

interface VolumeCache {
  logData: Float32Array;
  vmin: number;
  vmax: number;
  logMin: number;
  logMax: number;
}

function buildVolumeCache(vol: Float32Array): VolumeCache {
  let vmin = Infinity,
    vmax = -Infinity;
  for (let i = 0; i < vol.length; i++) {
    if (vol[i] > vmax) vmax = vol[i];
    if (vol[i] < vmin) vmin = vol[i];
  }

  /* Use log scale for colormap normalisation — better shows dynamic range */
  const logMin = vmin > 0 ? Math.log10(vmin) : Math.log10(Math.max(vmax * 1e-6, 1e-30));
  const logMax = vmax > 0 ? Math.log10(vmax) : 0;
  const logData = new Float32Array(vol.length);
  for (let i = 0; i < vol.length; i++) {
    logData[i] = vol[i] > 0 ? Math.log10(vol[i]) : logMin;
  }

  return { logData, vmin, vmax, logMin, logMax };
}

const Simulation = {
  nx: 40,
  ny: 40,
  nz: 40,
  phi: null as VolumeCache | null,
  abs: null as VolumeCache | null,

  /** Store a freshly computed result and remember the grid it used. */
  set(nx: number, ny: number, nz: number, phi: Float32Array, abs: Float32Array) {
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    this.phi = buildVolumeCache(phi);
    this.abs = buildVolumeCache(abs);
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
   ================================================================ */
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
      <span class="axis-val" id="s${ax}-${suffix}-v">${defV}</span>`;
    container.appendChild(row);

    const el = row.querySelector("input") as HTMLInputElement;
    const out = row.querySelector(".axis-val") as HTMLElement;
    el.addEventListener("input", () => {
      out.textContent = el.value;
      redraw(suffix);
    });
  });
}

function getSlice(suffix: VolumeKind): { ix: number; iy: number; iz: number } {
  return {
    ix: +(document.getElementById(`sx-${suffix}`) as HTMLInputElement).value,
    iy: +(document.getElementById(`sy-${suffix}`) as HTMLInputElement).value,
    iz: +(document.getElementById(`sz-${suffix}`) as HTMLInputElement).value,
  };
}

function redraw(suffix: VolumeKind): void {
  const cache = Simulation.volume(suffix);
  if (!cache) return;
  const { ix, iy, iz } = getSlice(suffix);

  drawSlices(
    `cv-${suffix}`,
    cache.logData,
    Simulation.nx,
    Simulation.ny,
    Simulation.nz,
    ix,
    iy,
    iz,
    cache.logMin,
    cache.logMax
  );
  drawColorbar(`cbar-${suffix}`, `clbl-${suffix}-hi`, `clbl-${suffix}-mid`, `clbl-${suffix}-lo`, cache.vmin, cache.vmax);
}

/* ================================================================
   RESPONSIVE CANVAS RESIZE
   ================================================================
   The plot canvases are sized in JS (cv.width/height) to match their
   rendered CSS pixel size, so they stay crisp at any zoom level. A
   ResizeObserver on the canvases themselves — rather than a window
   'resize' listener — catches every reason their box can change size
   (window resize, browser zoom, the plots-row collapsing to a single
   column) and already coalesces to at most one callback per frame, so
   no manual debounce is needed; redraws stay live during a drag
   instead of snapping in after the fact.
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
  redraw("phi");
  redraw("abs");
});
["cv-phi", "cv-abs"].forEach((id) => canvasResizeObserver.observe(document.getElementById(id)!));

/* ================================================================
   MAIN RUN HANDLER
   ================================================================ */
document.getElementById("run-btn")!.addEventListener("click", async () => {
  const p = getParams(); // reads whatever controls the current model's paramGroups produced
  const btn = document.getElementById("run-btn") as HTMLButtonElement;
  const st = document.getElementById("status")!;

  btn.disabled = true;
  st.textContent = "Computing…";

  const model = MODELS[(document.getElementById("model-select") as HTMLSelectElement).value];

  const t0 = performance.now();
  const { phi, abs, derived, valid, reasons } = await runModel(model.command, p);
  const dt = (performance.now() - t0).toFixed(1);

  Simulation.set(p.nx, p.ny, p.nz, phi, abs);

  /* Show plots section */
  (document.getElementById("plots") as HTMLElement).style.display = "";

  /* Rebuild sliders with correct max values */
  buildAxisSliders("sl-phi", "phi");
  buildAxisSliders("sl-abs", "abs");

  /* Resize canvases to match their rendered pixel width */
  syncCanvasSizes();

  redraw("phi");
  redraw("abs");

  const summary = model.summaryLine(derived, dt);

  st.textContent = "";
  st.innerHTML = summary;

  if (!valid) {
    const warn = document.createElement("div");
    warn.className = "status-warn";
    const intro = document.createElement("p");
    intro.textContent = "⚠ Results may not be accurate — diffusion approximation is weakly justified here:";
    warn.appendChild(intro);
    reasons.forEach((reason) => {
      const para = document.createElement("p");
      para.innerHTML = reason + ".";
      warn.appendChild(para);
    });
    st.appendChild(warn);
  }

  btn.disabled = false;
});

/* ================================================================
   MODEL SWITCHING
   ================================================================
   Each model owns its own paramGroups (schema + defaults — see
   models.ts), so switching models means tearing down and rebuilding
   the whole parameter panel, not just resetting values. Any plots
   from a previous model are hidden since they'd correspond to a
   different (or no longer valid) set of inputs.
   ================================================================ */
function onModelChange(): void {
  const model = MODELS[(document.getElementById("model-select") as HTMLSelectElement).value];
  buildModelParams(model, "param-panels");
  (document.getElementById("plots") as HTMLElement).style.display = "none";
  document.getElementById("status")!.textContent = "Adjust parameters and click Compute.";
}

/* ================================================================
   INIT
   ================================================================ */
buildModelSelect();
document.getElementById("model-select")!.addEventListener("change", onModelChange);
onModelChange();

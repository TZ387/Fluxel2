import "./styles.css";
import { MODELS, buildModelSelect } from "./models";
import { buildModelParams, getParams } from "./ui-params";
import { drawSlices, drawColorbar } from "./render";

/* ================================================================
   SIMULATION STATE
   ================================================================
   Holds the most recent computed volumes plus the grid dimensions
   they were computed on. Wrapped in an object (rather than loose
   globals) so the fields it owns and the ways it can be mutated
   are explicit and in one place.
   ================================================================ */
type VolumeKind = "phi" | "abs";

const Simulation = {
  nx: 40,
  ny: 40,
  nz: 40,
  phi: null as Float64Array | null,
  abs: null as Float64Array | null,

  /** Store a freshly computed result and remember the grid it used. */
  set(nx: number, ny: number, nz: number, phi: Float64Array, abs: Float64Array) {
    this.nx = nx;
    this.ny = ny;
    this.nz = nz;
    this.phi = phi;
    this.abs = abs;
  },

  /** 'phi' | 'abs' → the matching Float64Array, or null if not yet computed. */
  volume(suffix: VolumeKind): Float64Array | null {
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
  const vol = Simulation.volume(suffix);
  if (!vol) return;
  const { ix, iy, iz } = getSlice(suffix);

  let vmin = Infinity,
    vmax = -Infinity;
  for (let i = 0; i < vol.length; i++) {
    if (vol[i] > vmax) vmax = vol[i];
    if (vol[i] < vmin) vmin = vol[i];
  }

  /* Use log scale for colormap normalisation — better shows dynamic range */
  const logVol = new Float64Array(vol.length);
  const logMin = vmin > 0 ? Math.log10(vmin) : Math.log10(Math.max(vmax * 1e-6, 1e-30));
  const logMax = vmax > 0 ? Math.log10(vmax) : 0;
  for (let i = 0; i < vol.length; i++) {
    logVol[i] = vol[i] > 0 ? Math.log10(vol[i]) : logMin;
  }

  drawSlices(`cv-${suffix}`, logVol, Simulation.nx, Simulation.ny, Simulation.nz, ix, iy, iz, logMin, logMax);
  drawColorbar(`cbar-${suffix}`, `clbl-${suffix}-hi`, `clbl-${suffix}-mid`, `clbl-${suffix}-lo`, vmin, vmax);
}

/* ================================================================
   RESPONSIVE CANVAS RESIZE
   ================================================================
   The plot canvases are sized in JS (cv.width/height) to match their
   rendered CSS pixel size, so they stay crisp at any zoom level. That
   sizing previously only ran right after a compute or while dragging
   an axis slider — so browser zoom (which fires a 'resize' event but
   touches neither of those) left the canvases stale: blurry, wrongly
   proportioned, or misaligned with their container. This listener
   re-applies that sizing and redraws whenever the viewport changes,
   which covers window resizing as well as zooming in/out.
   ================================================================ */
let resizeTimer: ReturnType<typeof setTimeout> | null = null;
window.addEventListener("resize", () => {
  if (!Simulation.hasData()) return;
  if (resizeTimer) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    ["cv-phi", "cv-abs"].forEach((id) => {
      const cv = document.getElementById(id) as HTMLCanvasElement;
      cv.width = cv.offsetWidth || 400;
      cv.height = cv.offsetHeight || 400;
    });
    redraw("phi");
    redraw("abs");
  }, 120);
});

/* ================================================================
   MAIN RUN HANDLER
   ================================================================ */
document.getElementById("run-btn")!.addEventListener("click", () => {
  const p = getParams(); // reads whatever controls the current model's paramGroups produced
  const btn = document.getElementById("run-btn") as HTMLButtonElement;
  const st = document.getElementById("status")!;

  btn.disabled = true;
  st.textContent = "Computing…";

  /* Yield to browser for status paint, then compute */
  setTimeout(() => {
    const model = MODELS[(document.getElementById("model-select") as HTMLSelectElement).value];

    const t0 = performance.now();
    const { phi, abs, derived } = model.compute(p);
    const dt = (performance.now() - t0).toFixed(1);

    Simulation.set(p.nx, p.ny, p.nz, phi, abs);

    /* Show plots section */
    (document.getElementById("plots") as HTMLElement).style.display = "";

    /* Rebuild sliders with correct max values */
    buildAxisSliders("sl-phi", "phi");
    buildAxisSliders("sl-abs", "abs");

    /* Resize canvases to match their rendered pixel width */
    ["cv-phi", "cv-abs"].forEach((id) => {
      const cv = document.getElementById(id) as HTMLCanvasElement;
      cv.width = cv.offsetWidth || 400;
      cv.height = cv.offsetHeight || 400;
    });

    redraw("phi");
    redraw("abs");

    const summary = model.summaryLine(derived, dt);

    const validity = model.checkValidity ? model.checkValidity(p, derived) : { valid: true, reasons: [] };

    st.textContent = "";
    st.innerHTML = summary;

    if (!validity.valid) {
      const warn = document.createElement("div");
      warn.className = "status-warn";
      const intro = document.createElement("p");
      intro.textContent = "⚠ Results may not be accurate — diffusion approximation is weakly justified here:";
      warn.appendChild(intro);
      validity.reasons.forEach((reason) => {
        const para = document.createElement("p");
        para.innerHTML = reason + ".";
        warn.appendChild(para);
      });
      st.appendChild(warn);
    }

    btn.disabled = false;
  }, 20);
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

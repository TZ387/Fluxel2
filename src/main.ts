import "./styles.css";
import { MODELS, buildModelSelect } from "./models";
import { buildModelParams, getParams } from "./ui-params";
import { drawSlices, drawColorbar } from "./render";
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
  /** Raw bounds, for the colourbar's labels. */
  vmin: number;
  vmax: number;
  /** The same bounds on the log scale the colour ramp works in. */
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

  /* Log scale for the colormap — it shows the full dynamic range from
     near the source to far from it. A volume that never goes positive has
     no log range to speak of, so fall back to six decades below the peak. */
  const logMin = vmin > 0 ? Math.log10(vmin) : Math.log10(Math.max(vmax * 1e-6, 1e-30));
  const logMax = vmax > 0 ? Math.log10(vmax) : 0;

  return { data: vol, vmin, vmax, logMin, logMax };
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

function redraw(suffix: VolumeKind): void {
  const cache = Simulation.volume(suffix);
  if (!cache) return;
  const { ix, iy, iz } = getSlice(suffix);

  drawSlices(
    `cv-${suffix}`,
    cache.data,
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
  redraw("phi");
  redraw("abs");
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

  st.innerHTML = model.summaryLine(derived, dt);

  if (!valid) {
    const warn = document.createElement("div");
    warn.className = "status-warn";
    const intro = document.createElement("p");
    intro.textContent = "⚠ Results may not be accurate — diffusion approximation is weakly justified here:";
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
buildModelSelect();
document.getElementById("model-select")!.addEventListener("change", onModelChange);
onModelChange();
buildHelp("tab-help");

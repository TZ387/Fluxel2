/* ================================================================
   COLORMAP  (deep-blue → cyan → green → yellow → red)
   ================================================================
   Sampled once into a lookup table. drawSlices needs a colour per
   pixel — hundreds of thousands per redraw, and a redraw runs on
   every slider drag — where walking the stop list and allocating a
   triple to hold the answer cost about ten times what a table read
   does. 1024 steps keeps the largest departure from the continuous
   ramp at 1/255: the rounding floor of the 8-bit channels it feeds.
   ================================================================ */
type RGB = [number, number, number];
type Stop = [number, RGB];

const CMAP_STOPS: Stop[] = [
  [0.0, [10, 10, 35]],
  [0.15, [20, 40, 160]],
  [0.35, [10, 160, 200]],
  [0.55, [20, 200, 80]],
  [0.72, [230, 220, 20]],
  [0.88, [240, 100, 10]],
  [1.0, [180, 10, 10]],
];

const CMAP_LEVELS = 1024;

/** Interpolate the stops directly — only used to build the table below. */
function colormapExact(t: number): RGB {
  t = Math.max(0.0, Math.min(1.0, t));
  let lo = CMAP_STOPS[0],
    hi = CMAP_STOPS[CMAP_STOPS.length - 1];
  for (let i = 0; i < CMAP_STOPS.length - 1; i++) {
    if (t >= CMAP_STOPS[i][0] && t <= CMAP_STOPS[i + 1][0]) {
      lo = CMAP_STOPS[i];
      hi = CMAP_STOPS[i + 1];
      break;
    }
  }
  const f = (t - lo[0]) / (hi[0] - lo[0] + 1e-15);
  return lo[1].map((v, i) => Math.round(v + (hi[1][i] - v) * f)) as RGB;
}

const CMAP_LUT: Uint8Array = (() => {
  const lut = new Uint8Array(CMAP_LEVELS * 3);
  for (let i = 0; i < CMAP_LEVELS; i++) {
    const [r, g, b] = colormapExact(i / (CMAP_LEVELS - 1));
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
})();

/** Where t's colour starts in CMAP_LUT. Clamping t before scaling keeps the
    index in range for anything callers pass, NaN included. */
function cmapOffset(t: number): number {
  const c = t > 0 ? (t < 1 ? t : 1) : 0;
  return ((c * (CMAP_LEVELS - 1) + 0.5) | 0) * 3;
}

/* ================================================================
   COLORBAR
   ================================================================ */
export function drawColorbar(
  cvId: string,
  hiId: string,
  midId: string,
  loId: string,
  vmin: number,
  vmax: number
): void {
  const cv = document.getElementById(cvId) as HTMLCanvasElement;
  const w = 20,
    h = 140;
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d")!;
  for (let i = 0; i < h; i++) {
    const k = cmapOffset(1 - i / h);
    ctx.fillStyle = `rgb(${CMAP_LUT[k]},${CMAP_LUT[k + 1]},${CMAP_LUT[k + 2]})`;
    ctx.fillRect(0, i, w, 1);
  }
  const fmtSci = (v: number) => {
    if (v === 0 || !isFinite(v)) return "0";
    const e = Math.floor(Math.log10(Math.abs(v)));
    const m = v / Math.pow(10, e);
    return `${m.toFixed(1)}e${e >= 0 ? "+" : ""}${e}`;
  };
  document.getElementById(hiId)!.textContent = fmtSci(vmax);
  document.getElementById(midId)!.textContent = fmtSci((vmax + vmin) / 2);
  document.getElementById(loId)!.textContent = fmtSci(vmin);
}

/* ================================================================
   3-SLICE RENDERER
   ================================================================
   Draws three orthogonal cross-sections onto a single square canvas:
     top-left:  YZ plane at voxel index ix  (axes: y horizontal, z vertical)
     top-right: XZ plane at voxel index iy  (axes: x horizontal, z vertical)
     bottom-left: XY plane at voxel index iz (axes: x horizontal, y vertical)

   `vol` arrives as raw values and is log10-scaled at sample time, against
   the [logMin, logMax] bounds main.ts derived from the whole volume. Only
   the pixels actually drawn get transformed, which is a tiny fraction of a
   large volume — see the note on Simulation in main.ts.
   ================================================================ */
export function drawSlices(
  cvId: string,
  vol: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  ix: number,
  iy: number,
  iz: number,
  logMin: number,
  logMax: number
): void {
  const cv = document.getElementById(cvId) as HTMLCanvasElement;
  const W = cv.width || cv.offsetWidth || 400;
  const H = cv.height || cv.offsetHeight || 400;
  cv.width = W;
  cv.height = H;

  const ctx = cv.getContext("2d")!;
  ctx.fillStyle = "#080c14";
  ctx.fillRect(0, 0, W, H);

  const PAD = 6;
  const HALF = Math.floor((W - 3 * PAD) / 2);
  const VHALF = Math.floor((H - 3 * PAD) / 2);

  /* Clamp slice indices */
  ix = Math.max(0, Math.min(nx - 1, ix));
  iy = Math.max(0, Math.min(ny - 1, iy));
  iz = Math.max(0, Math.min(nz - 1, iz));

  const range = logMax - logMin + 1e-30;

  /* One voxel, already on the log scale the colour ramp works in.
     Non-positive values (and anything out of bounds) sit at the bottom. */
  function sampleLog(x: number, y: number, z: number): number {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return logMin;
    const v = vol[x + y * nx + z * nx * ny];
    return v > 0 ? Math.log10(v) : logMin;
  }

  /**
   * fillRect2D — rasterise one 2D slice into an ImageData region.
   * ox, oy  — canvas offset of top-left corner
   * rw, rh  — pixel dimensions on canvas
   * sampleFn  — (col, row) → log-scaled voxel, col∈[0,cols-1], row∈[0,rows-1]
   * cols, rows  — voxel dimensions of this slice
   */
  function fillRect2D(
    ox: number,
    oy: number,
    rw: number,
    rh: number,
    sampleFn: (col: number, row: number) => number,
    cols: number,
    rows: number
  ): void {
    const img = ctx.createImageData(rw, rh);
    const d = img.data;
    for (let py = 0; py < rh; py++) {
      const vc = Math.min(rows - 1, Math.floor((py * rows) / rh));
      for (let px = 0; px < rw; px++) {
        const uc = Math.min(cols - 1, Math.floor((px * cols) / rw));
        const k = cmapOffset((sampleFn(uc, vc) - logMin) / range);
        const i = (py * rw + px) * 4;
        d[i] = CMAP_LUT[k];
        d[i + 1] = CMAP_LUT[k + 1];
        d[i + 2] = CMAP_LUT[k + 2];
        d[i + 3] = 255;
      }
    }
    ctx.putImageData(img, ox, oy);

    /* thin border */
    ctx.strokeStyle = "rgba(100,130,160,0.35)";
    ctx.lineWidth = 0.75;
    ctx.strokeRect(ox + 0.5, oy + 0.5, rw - 1, rh - 1);
  }

  /* Top-left: YZ slice at ix — horizontal=y, vertical=z */
  const tlX = PAD,
    tlY = PAD;
  fillRect2D(tlX, tlY, HALF, VHALF, (c, r) => sampleLog(ix, c, r), ny, nz);

  /* Top-right: XZ slice at iy — horizontal=x, vertical=z */
  const trX = PAD * 2 + HALF,
    trY = PAD;
  fillRect2D(trX, trY, HALF, VHALF, (c, r) => sampleLog(c, iy, r), nx, nz);

  /* Bottom-left: XY slice at iz — horizontal=x, vertical=y */
  const blX = PAD,
    blY = PAD * 2 + VHALF;
  fillRect2D(blX, blY, HALF, VHALF, (c, r) => sampleLog(c, r, iz), nx, ny);

  /* Slice labels — a light halo (stroke) behind the dark fill keeps them
     legible against the colormap's own near-black low end, where plain
     fillText disappeared. Baseline is 16px below each rect's top edge
     (not 12px) since the halo's width otherwise pushed it above the
     rect, into the background gutter. */
  ctx.font = "bold 17px monospace";
  ctx.lineJoin = "round";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.fillStyle = "rgba(12,13,15,0.9)";
  const label = (text: string, x: number, y: number) => {
    ctx.strokeText(text, x, y);
    ctx.fillText(text, x, y);
  };
  label(`YZ  x = ${ix}`, tlX + 4, tlY + 16);
  label(`XZ  y = ${iy}`, trX + 4, trY + 16);
  label(`XY  z = ${iz}`, blX + 4, blY + 16);
}

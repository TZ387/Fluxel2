/* ================================================================
   COLORMAP  (deep-blue → cyan → green → yellow → red)
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

function colormap(t: number): RGB {
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
    const [r, g, b] = colormap(1 - i / h);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
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
  vmin: number,
  vmax: number
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

  const range = vmax - vmin + 1e-30;

  function getVoxel(x: number, y: number, z: number): number {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return vmin;
    return vol[x + y * nx + z * nx * ny];
  }

  /**
   * fillRect2D — rasterise one 2D slice into an ImageData region.
   * ox, oy  — canvas offset of top-left corner
   * rw, rh  — pixel dimensions on canvas
   * sampleFn  — (col, row) → voxel value, col∈[0,cols-1], row∈[0,rows-1]
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
        const t = (sampleFn(uc, vc) - vmin) / range;
        const [r, g, b] = colormap(t);
        const i = (py * rw + px) * 4;
        d[i] = r;
        d[i + 1] = g;
        d[i + 2] = b;
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
  fillRect2D(tlX, tlY, HALF, VHALF, (c, r) => getVoxel(ix, c, r), ny, nz);

  /* Top-right: XZ slice at iy — horizontal=x, vertical=z */
  const trX = PAD * 2 + HALF,
    trY = PAD;
  fillRect2D(trX, trY, HALF, VHALF, (c, r) => getVoxel(c, iy, r), nx, nz);

  /* Bottom-left: XY slice at iz — horizontal=x, vertical=y */
  const blX = PAD,
    blY = PAD * 2 + VHALF;
  fillRect2D(blX, blY, HALF, VHALF, (c, r) => getVoxel(c, r, iz), nx, ny);

  /* Slice labels — a light halo (stroke) behind the dark fill keeps them
     legible regardless of what's underneath: dark fill alone disappears
     against the colormap's own near-black low end (e.g. a deep slice far
     from the source), which plain fillText hit in practice. Baseline is
     16px below each rect's top edge rather than 12px — the halo's own
     width pushes its visible top above a 17px font's ascent, which at
     +12 poked above the rect into the background gutter above it. */
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

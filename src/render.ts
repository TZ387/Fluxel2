/* ================================================================
   COLORMAPS
   ================================================================
   Sampled once into a lookup table per map. A slice image needs a
   colour per voxel — up to 160 000 per plane, rebuilt on every
   slider drag — where walking the stop list and allocating a triple
   to hold the answer cost about ten times what a table read does.
   1024 steps keeps the largest departure from the continuous ramp at
   1/255: the rounding floor of the 8-bit channels it feeds.

   Two maps. Inferno is the default: it is perceptually uniform (equal
   steps in value look like equal steps in brightness, which the older
   ramp does not manage — it invents banding at its cyan and yellow
   turns), it survives being printed in greyscale, and it reads
   correctly for the common forms of colour-vision deficiency. Its
   near-black low end also does real work in the 3-D view, where it
   lets weak fluence fade into the box interior instead of drawing a
   hard edge around the slice planes. The original blue→red ramp is
   kept selectable, since every screenshot taken before this existed
   used it.
   ================================================================ */
export type RGB = [number, number, number];
type Stop = [number, RGB];

const CMAP_LEVELS = 1024;

/** Interpolate a stop list — only used to build the tables below. */
function colormapExact(stops: Stop[], t: number): RGB {
  t = Math.max(0.0, Math.min(1.0, t));
  let lo = stops[0],
    hi = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      lo = stops[i];
      hi = stops[i + 1];
      break;
    }
  }
  const f = (t - lo[0]) / (hi[0] - lo[0] + 1e-15);
  return lo[1].map((v, i) => Math.round(v + (hi[1][i] - v) * f)) as RGB;
}

function buildLut(stops: Stop[]): Uint8Array {
  const lut = new Uint8Array(CMAP_LEVELS * 3);
  for (let i = 0; i < CMAP_LEVELS; i++) {
    const [r, g, b] = colormapExact(stops, i / (CMAP_LEVELS - 1));
    lut[i * 3] = r;
    lut[i * 3 + 1] = g;
    lut[i * 3 + 2] = b;
  }
  return lut;
}

const INFERNO_STOPS: Stop[] = [
  [0.0, [0, 0, 4]],
  [0.1, [22, 11, 57]],
  [0.2, [66, 10, 104]],
  [0.3, [106, 23, 110]],
  [0.4, [147, 38, 103]],
  [0.5, [188, 55, 84]],
  [0.6, [221, 81, 58]],
  [0.7, [243, 120, 25]],
  [0.8, [252, 165, 10]],
  [0.9, [246, 215, 70]],
  [1.0, [252, 255, 164]],
];

const SPECTRAL_STOPS: Stop[] = [
  [0.0, [10, 10, 35]],
  [0.15, [20, 40, 160]],
  [0.35, [10, 160, 200]],
  [0.55, [20, 200, 80]],
  [0.72, [230, 220, 20]],
  [0.88, [240, 100, 10]],
  [1.0, [180, 10, 10]],
];

export interface ColormapDef {
  id: string;
  label: string;
  lut: Uint8Array;
}

/** First entry is the default the "Colormap" dropdown opens on. */
export const COLORMAPS: ColormapDef[] = [
  { id: "inferno", label: "Inferno — black → orange → yellow", lut: buildLut(INFERNO_STOPS) },
  { id: "spectral", label: "Spectral — blue → green → red", lut: buildLut(SPECTRAL_STOPS) },
];

export function colormapLut(id: string): Uint8Array {
  return (COLORMAPS.find((c) => c.id === id) ?? COLORMAPS[0]).lut;
}

/** Where t's colour starts in a LUT. Clamping t before scaling keeps the
    index in range for anything callers pass, NaN included. */
function cmapOffset(t: number): number {
  const c = t > 0 ? (t < 1 ? t : 1) : 0;
  return ((c * (CMAP_LEVELS - 1) + 0.5) | 0) * 3;
}

/** The map's own zero colour, as a CSS string. The 3-D view paints the box
    interior with it so that the bottom of the ramp meets the background
    seamlessly — which is what makes the bright core read as a glow rather
    than as three lit rectangles. */
export function colormapFloor(lut: Uint8Array): string {
  return `rgb(${lut[0]},${lut[1]},${lut[2]})`;
}

/* ================================================================
   VALUE SCALE
   ================================================================
   Log was the only option before, and it is still the default: it
   shows the full dynamic range from just under the source to the far
   corner of the grid, which for these fields spans several decades.
   Linear is the better read when the question is about a *threshold*
   — how far the region above some fluence reaches — since on a log
   ramp the top decade, where all of that happens, gets a sliver of
   the colour range.

   The pair (norm, raw) has to be mutual inverses: norm colours a
   voxel, raw puts the numbers on the colourbar's ticks, and if they
   disagree the labels lie about the picture.
   ================================================================ */
export type ScaleKind = "log" | "linear";

export interface Scale {
  kind: ScaleKind;
  /** Bounds in scale space — log10(value) when log, raw value when linear. */
  lo: number;
  hi: number;
  /** Raw value → position on the colour ramp. Not clamped; cmapOffset is. */
  norm(v: number): number;
  /** Position on the ramp → the raw value it stands for. */
  raw(t: number): number;
}

export function makeScale(kind: ScaleKind, vmin: number, vmax: number): Scale {
  if (kind === "linear") {
    const lo = Number.isFinite(vmin) ? vmin : 0;
    let hi = Number.isFinite(vmax) ? vmax : 1;
    if (!(hi > lo)) hi = lo + 1;
    const span = hi - lo;
    return {
      kind,
      lo,
      hi,
      norm: (v) => (v - lo) / span,
      raw: (t) => lo + t * span,
    };
  }
  /* A volume that never goes positive has no log range to speak of, so fall
     back to six decades below the peak. */
  const lo = vmin > 0 ? Math.log10(vmin) : Math.log10(Math.max(vmax * 1e-6, 1e-30));
  let hi = vmax > 0 ? Math.log10(vmax) : 0;
  if (!(hi > lo)) hi = lo + 1;
  const span = hi - lo;
  return {
    kind,
    lo,
    hi,
    /* Non-positive values sit at the bottom of the ramp rather than at -inf. */
    norm: (v) => (v > 0 ? (Math.log10(v) - lo) / span : 0),
    raw: (t) => Math.pow(10, lo + t * span),
  };
}

/* ================================================================
   TICKS
   ================================================================
   One nice-number generator, shared by both renderers' axes and by
   the colourbar, so a 0.05 cm step looks the same everywhere.
   ================================================================ */
export function niceTicks(lo: number, hi: number, target = 5): number[] {
  if (!(hi > lo) || !Number.isFinite(lo) || !Number.isFinite(hi)) return [lo];
  const rough = (hi - lo) / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out: number[] = [];
  const eps = step * 1e-9;
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + eps; v += step) {
    out.push(Math.abs(v) < eps ? 0 : v);
  }
  return out;
}

/** A whole tick set's numbers, formatted alike. The decimal count comes from
    the *spacing*, not from each value: a 0.5 cm step wants one decimal, and
    deciding per value instead gives an axis reading "-1.00, -0.500, 0" — three
    different precisions, and the longest of them setting how wide a margin the
    axis needs. Falls back to per-value scientific where the magnitudes are too
    extreme for a decimal to be short. */
export function tickLabels(values: number[]): string[] {
  const step = values.length > 1 ? Math.abs(values[1] - values[0]) : Math.abs(values[0]) || 1;
  const peak = Math.max(...values.map(Math.abs), 0);
  if (!(step > 0) || step < 1e-4 || peak >= 1e5) return values.map(fmtTick);
  const decimals = Math.min(4, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
  return values.map((v) => (Math.abs(v) < step * 1e-9 ? "0" : v.toFixed(decimals)));
}

/** A single tick's number — used where there is no set to be consistent with,
    namely the colourbar's decades. Plain decimal in the range that reads well
    as one, scientific outside it. */
export function fmtTick(v: number): string {
  if (v === 0) return "0";
  if (!Number.isFinite(v)) return "";
  const a = Math.abs(v);
  if (a >= 1e-3 && a < 1e5) {
    return v.toFixed(a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3);
  }
  const e = Math.floor(Math.log10(a));
  const m = v / Math.pow(10, e);
  const mant = Math.abs(m - Math.round(m)) < 1e-9 ? `${Math.round(m)}` : m.toFixed(1);
  return `${mant}e${e >= 0 ? "+" : ""}${e}`;
}

/* ================================================================
   VALIDITY OVERLAY
   ================================================================
   An alternate, discrete colouring for the same slices: instead of the
   computed field, a per-voxel code (0 worst, 2 best) that each model
   defines for itself — how far the voxel sits from breaking the diffusion
   approximation for the two diffusion models, how converged the estimate
   is for Monte Carlo (see each model's compute_validity_volume, and
   models.ts's `overlay` for the words). This file only knows how to
   colour them, the same way whichever model produced them, so the legend
   takes its labels from the caller. Colours mirror the app's own warning
   palette (styles.css --danger/--warn/--accent2) so the bad end reads the
   same as everywhere else in the UI.
   ================================================================ */
const VALIDITY_COLORS: RGB[] = [
  [248, 81, 73], // 0 invalid  — --danger
  [210, 153, 34], // 1 marginal — --warn
  [63, 185, 80], // 2 valid    — --accent2
];

/* Gutters, and anything outside the data. */
const PANEL_BG = "#080c14";
const AXIS_INK = "rgba(190,205,225,0.75)";
const GRID_INK = "rgba(120,150,185,0.35)";
const TICK_FONT = "11px monospace";
const TITLE_FONT = "bold 12px monospace";

/* ================================================================
   SLICE PLANE IMAGES
   ================================================================
   A slice is built at *voxel* resolution and then scaled to wherever
   it belongs on screen, rather than sampled once per screen pixel as
   this file used to do. Both renderers want that, but for different
   reasons:

     — the 3-D view needs an image it can hand to an affine transform
       (see render3d.ts), which is only possible if the image is the
       plane rather than a picture of it already in place;
     — and point-sampling per screen pixel was wrong in both
       directions anyway. Below the canvas resolution it showed a 20³
       grid as visible blocks; above it, a 400³ volume was sampled at
       roughly every other voxel, so a thin high-fluence feature could
       be missed entirely. Scaling a voxel-resolution image interpolates
       going up and averages coming down, and every voxel in the plane
       contributes either way.

   The three axes differ only in where the plane starts in the volume
   and how far one step along each image direction moves — so one
   base/du/dv triple covers all three.
   ================================================================ */

/** 0: x = const (a YZ plane), 1: y = const (XZ), 2: z = const (XY). */
export type SliceAxis = 0 | 1 | 2;

export interface VolumeView {
  data: Float32Array;
  nx: number;
  ny: number;
  nz: number;
  scale: Scale;
  lut: Uint8Array;
  validity: Uint8Array | null;
  showValidity: boolean;
  /** Identifies everything about this view that decides a voxel's colour —
      the run, the scale, the colormap, the overlay toggle. planeCache keys
      its images on it, so orbiting the 3-D box (which changes none of those)
      reuses the images it already has instead of recolouring half a million
      voxels per frame. It must change whenever any of those change; see
      main.ts's buildScene for the one place it is built. */
  stamp: string;
}

/** Image dimensions for a plane: [cols, rows]. Column runs along the first
    world axis the plane spans, row along the second — y,z for a YZ plane,
    x,z for XZ, x,y for XY. */
export function sliceDims(v: VolumeView, axis: SliceAxis): [number, number] {
  return axis === 0 ? [v.ny, v.nz] : axis === 1 ? [v.nx, v.nz] : [v.nx, v.ny];
}

/** Where the plane starts in `data`, and the strides one image column and
    one image row cost. */
function sliceStrides(v: VolumeView, axis: SliceAxis, index: number): [number, number, number] {
  const layer = v.nx * v.ny;
  if (axis === 0) return [index, v.nx, layer];
  if (axis === 1) return [index * v.nx, 1, layer];
  return [index * layer, 1, v.nx];
}

/** Three reusable canvases, one per axis. Rebuilding them per redraw would
    churn a canvas per frame during a slider drag. */
export interface PlaneCache {
  canvases: (HTMLCanvasElement | null)[];
  keys: (string | null)[];
}

export function createPlaneCache(): PlaneCache {
  return { canvases: [null, null, null], keys: [null, null, null] };
}

/** The plane at `index`, drawn one canvas pixel per voxel. Returns the
    cached canvas untouched when it already holds this exact plane. */
export function planeCanvas(
  cache: PlaneCache,
  v: VolumeView,
  axis: SliceAxis,
  index: number
): HTMLCanvasElement {
  const [cols, rows] = sliceDims(v, axis);
  let cv = cache.canvases[axis];
  if (!cv) {
    cv = document.createElement("canvas");
    cache.canvases[axis] = cv;
  }
  const key = `${v.stamp}|${index}`;
  const sized = cv.width === cols && cv.height === rows;
  if (sized && cache.keys[axis] === key) return cv;
  if (!sized) {
    cv.width = cols;
    cv.height = rows;
  }
  cache.keys[axis] = key;
  const ctx = cv.getContext("2d")!;
  const img = ctx.createImageData(cols, rows);
  const d = img.data;
  const [base, du, dv] = sliceStrides(v, axis, index);
  const { data, lut, scale, validity, showValidity } = v;

  let i = 0;
  for (let r = 0; r < rows; r++) {
    let o = base + r * dv;
    for (let c = 0; c < cols; c++, o += du, i += 4) {
      if (showValidity) {
        const code = validity ? validity[o] : 0;
        const rgb = VALIDITY_COLORS[code] ?? VALIDITY_COLORS[0];
        d[i] = rgb[0];
        d[i + 1] = rgb[1];
        d[i + 2] = rgb[2];
      } else {
        const k = cmapOffset(scale.norm(data[o]));
        d[i] = lut[k];
        d[i + 1] = lut[k + 1];
        d[i + 2] = lut[k + 2];
      }
      d[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

/** Discrete codes must not be blended into colours that mean nothing, so
    smoothing is for the continuous field only. Callers set this on the
    *destination* context, before drawing a plane into it. */
export function setSmoothing(ctx: CanvasRenderingContext2D, v: VolumeView): void {
  ctx.imageSmoothingEnabled = !v.showValidity;
  ctx.imageSmoothingQuality = "high";
}

/* ================================================================
   THE SCENE
   ================================================================
   What both renderers draw, in physical units. x and y are centred on
   the beam axis and z runs from 0 at the tissue surface *downward*,
   which is the convention the volumes are built in (see
   beam.rs's sample_axisymmetric_volume) and the one every depth in the
   parameter panel is quoted in.
   ================================================================ */
export interface SliceScene {
  view: VolumeView;
  /** Grid extents [cm]: x ∈ [-lx/2, lx/2], y ∈ [-ly/2, ly/2], z ∈ [0, lz]. */
  lx: number;
  ly: number;
  lz: number;
  /** Slice plane positions, as voxel indices. */
  ix: number;
  iy: number;
  iz: number;
  /** Depths [cm] of the internal layer interfaces — the cumulative layer
      thicknesses, less the bottom of the stack. Empty for a homogeneous
      model. Drawn because the layering is the whole point of the N-layer
      models, and it was previously only inferrable from where the colour
      kinked. */
  interfaces: readonly number[];
}

/** The physical position of each slice plane — voxel *centres*, since that
    is where the values in the volume were evaluated. */
export function sliceCenters(s: SliceScene): { x: number; y: number; z: number } {
  return {
    x: ((s.ix + 0.5) * s.lx) / s.view.nx - s.lx / 2,
    y: ((s.iy + 0.5) * s.ly) / s.view.ny - s.ly / 2,
    z: ((s.iz + 0.5) * s.lz) / s.view.nz,
  };
}

/* ================================================================
   COLORBAR
   ================================================================
   Ticks are generated from the scale, so they follow it: decades when
   the ramp is logarithmic and spans more than one, nice numbers
   otherwise. They are drawn into the canvas rather than laid out as
   three fixed HTML labels, which is what lets there be a variable
   number of them in the right places.
   ================================================================ */
const CBAR_W = 92;
const CBAR_H = 200;
const CBAR_BAR = 18;

function sizeBar(cvId: string): CanvasRenderingContext2D {
  const cv = document.getElementById(cvId) as HTMLCanvasElement;
  cv.width = CBAR_W;
  cv.height = CBAR_H;
  const ctx = cv.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, CBAR_W, CBAR_H);
  return ctx;
}

/** Ramp positions to label, with their numbers. */
function scaleTicks(scale: Scale): { t: number; label: string }[] {
  const span = scale.hi - scale.lo;
  if (scale.kind === "log" && span >= 1) {
    /* Whole decades, thinned out so a twelve-decade range does not stack
       twelve labels into 200 px. */
    const step = Math.max(1, Math.ceil(span / 6));
    const out: { t: number; label: string }[] = [];
    for (let e = Math.ceil(scale.lo); e <= scale.hi + 1e-9; e += step) {
      out.push({ t: (e - scale.lo) / span, label: fmtTick(Math.pow(10, e)) });
    }
    return out;
  }
  /* Linear, or a log range too narrow for decades: nice numbers in scale
     space, labelled with the values they stand for. */
  return niceTicks(scale.lo, scale.hi, 5)
    .filter((v) => v >= scale.lo - 1e-12 && v <= scale.hi + 1e-12)
    .map((v) => ({ t: (v - scale.lo) / span, label: fmtTick(scale.kind === "log" ? Math.pow(10, v) : v) }));
}

export function drawColorbar(cvId: string, scale: Scale, lut: Uint8Array): void {
  const ctx = sizeBar(cvId);
  for (let i = 0; i < CBAR_H; i++) {
    const k = cmapOffset(1 - i / (CBAR_H - 1));
    ctx.fillStyle = `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
    ctx.fillRect(0, i, CBAR_BAR, 1);
  }
  ctx.strokeStyle = GRID_INK;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, CBAR_BAR - 1, CBAR_H - 1);

  ctx.font = TICK_FONT;
  ctx.fillStyle = AXIS_INK;
  ctx.strokeStyle = AXIS_INK;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  scaleTicks(scale).forEach(({ t, label }) => {
    /* Clamped so the topmost and bottommost labels stay inside the canvas
       instead of being cut in half by its edge. */
    const y = Math.max(6, Math.min(CBAR_H - 6, (1 - t) * (CBAR_H - 1)));
    ctx.beginPath();
    ctx.moveTo(CBAR_BAR, y);
    ctx.lineTo(CBAR_BAR + 4, y);
    ctx.stroke();
    ctx.fillText(label, CBAR_BAR + 7, y);
  });
}

/** Same slot as drawColorbar, for when the overlay toggle is on: three
    discrete bands instead of a continuous gradient, with a word instead of
    a number at each. Valid sits at the top to match drawColorbar's
    convention of the "good" end (there, the max) being on top. */
export function drawValidityLegend(cvId: string, labels: readonly [string, string, string]): void {
  const ctx = sizeBar(cvId);
  const bandH = CBAR_H / VALIDITY_COLORS.length;
  ctx.font = TICK_FONT;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  VALIDITY_COLORS.forEach(([r, g, b], code) => {
    const i = VALIDITY_COLORS.length - 1 - code; // band 0 at the top is the best code
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(0, i * bandH, CBAR_BAR, bandH + 1); // +1 covers the rounding gap between bands
    ctx.fillStyle = AXIS_INK;
    ctx.fillText(labels[code], CBAR_BAR + 7, (i + 0.5) * bandH);
  });
  ctx.strokeStyle = GRID_INK;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, CBAR_BAR - 1, CBAR_H - 1);
}

/* ================================================================
   FLAT 3-SLICE RENDERER
   ================================================================
   Three orthogonal cross-sections on one square canvas:
     top-left:    YZ plane at ix (axes: y horizontal, z vertical)
     top-right:   XZ plane at iy (axes: x horizontal, z vertical)
     bottom-left: XY plane at iz (axes: x horizontal, y vertical)

   The alternative to render3d.ts's box, and still the view to reach
   for when the question is quantitative: nothing is foreshortened, so
   a distance on screen is a distance in the tissue.

   Margins carry cm ticks. There is room for them because the fourth
   quadrant was always empty, and they are shared where the panels
   share an axis: the two top panels have the same depth axis, so z is
   labelled once, on the left.
   ================================================================ */
/* Margins, each sized by what actually has to fit in it:
     ML — a right-aligned tick label (up to ~5 characters) *and* the rotated
          axis title inboard of it;
     MB — a row of ticks (11 px) with the axis title (12 px) below them;
     MT — half a tick label, since the topmost one is centred on the panel's
          top edge;
     MR — half a tick label, likewise for the rightmost one.
   Getting MT and MR wrong is invisible until a tick lands exactly at a corner,
   which is every time: the first and last tick of an axis sit on its ends. */
const ML = 60;
const MB = 30;
const MT = 8;
const MR = 18;
const GAP = 10;

export function drawSlices(cvId: string, scene: SliceScene, cache: PlaneCache): void {
  const cv = document.getElementById(cvId) as HTMLCanvasElement;
  const W = cv.width || cv.offsetWidth || 400;
  const H = cv.height || cv.offsetHeight || 400;
  cv.width = W;
  cv.height = H;

  const ctx = cv.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(0, 0, W, H);

  const { view, lx, ly, lz } = scene;
  const half = Math.floor((W - ML - MR - GAP) / 2);
  const vhalf = Math.floor((H - MT - 2 * MB - GAP) / 2);
  /* Below this the panels are narrower than their own axis labels, so there is
     nothing to read; an empty canvas says that more honestly than three
     unreadable slivers would. */
  if (half < 40 || vhalf < 40) return;

  const c = sliceCenters(scene);
  setSmoothing(ctx, view);

  /* One panel: the plane image scaled into its rect, a border, ticks on the
     axes it owns, layer interfaces if its vertical axis is depth, and a
     label saying which plane it is and where. */
  function panel(
    ox: number,
    oy: number,
    axis: SliceAxis,
    index: number,
    hRange: [number, number],
    vRange: [number, number],
    hName: string,
    vName: string,
    withVAxis: boolean,
    title: string
  ): void {
    ctx.drawImage(planeCanvas(cache, view, axis, index), ox, oy, half, vhalf);

    ctx.strokeStyle = GRID_INK;
    ctx.lineWidth = 1;
    ctx.strokeRect(ox + 0.5, oy + 0.5, half - 1, vhalf - 1);

    ctx.font = TICK_FONT;
    ctx.fillStyle = AXIS_INK;
    ctx.strokeStyle = AXIS_INK;

    /* Horizontal axis, along the bottom. */
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const hTicks = niceTicks(hRange[0], hRange[1], 4);
    const hText = tickLabels(hTicks);
    hTicks.forEach((v, i) => {
      const px = ox + ((v - hRange[0]) / (hRange[1] - hRange[0])) * half;
      ctx.beginPath();
      ctx.moveTo(px, oy + vhalf);
      ctx.lineTo(px, oy + vhalf + 3);
      ctx.stroke();
      ctx.fillText(hText[i], px, oy + vhalf + 5);
    });
    ctx.font = TITLE_FONT;
    ctx.fillText(`${hName} [cm]`, ox + half / 2, oy + vhalf + 16);

    /* Vertical axis, down the left — only on the panels that own one. */
    if (withVAxis) {
      ctx.font = TICK_FONT;
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      const vTicks = niceTicks(vRange[0], vRange[1], 4);
      const vText = tickLabels(vTicks);
      vTicks.forEach((v, i) => {
        const py = oy + ((v - vRange[0]) / (vRange[1] - vRange[0])) * vhalf;
        ctx.beginPath();
        ctx.moveTo(ox - 3, py);
        ctx.lineTo(ox, py);
        ctx.stroke();
        ctx.fillText(vText[i], ox - 5, py);
      });
      ctx.save();
      ctx.font = TITLE_FONT;
      ctx.translate(ox - ML + 9, oy + vhalf / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${vName} [cm]`, 0, 0);
      ctx.restore();
    }

    /* Layer interfaces, where the vertical axis is depth. */
    if (vName === "z" && scene.interfaces.length) {
      ctx.save();
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      scene.interfaces.forEach((d) => {
        if (d <= 0 || d >= lz) return;
        const py = Math.round(oy + (d / lz) * vhalf) + 0.5;
        ctx.beginPath();
        ctx.moveTo(ox, py);
        ctx.lineTo(ox + half, py);
        ctx.stroke();
      });
      ctx.restore();
    }

    /* A light halo behind the dark fill keeps the label legible against the
       colormap's own near-black low end, where plain fillText disappeared.
       Dropped rather than run past the panel edge when the panel is too narrow
       for it — it sits *on* the data, so a clipped one is worse than none. */
    ctx.font = "bold 13px monospace";
    ctx.lineJoin = "round";
    ctx.lineWidth = 3;
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.fillStyle = "rgba(12,13,15,0.92)";
    if (ctx.measureText(title).width + 8 <= half) {
      ctx.strokeText(title, ox + 4, oy + 14);
      ctx.fillText(title, ox + 4, oy + 14);
    }
    ctx.fillStyle = AXIS_INK;
  }

  const xR: [number, number] = [-lx / 2, lx / 2];
  const yR: [number, number] = [-ly / 2, ly / 2];
  const zR: [number, number] = [0, lz];
  const row2 = MT + vhalf + MB + GAP;

  panel(ML, MT, 0, scene.ix, yR, zR, "y", "z", true, `YZ x=${c.x.toFixed(3)}`);
  panel(ML + half + GAP, MT, 1, scene.iy, xR, zR, "x", "z", false, `XZ y=${c.y.toFixed(3)}`);
  panel(ML, row2, 2, scene.iz, xR, yR, "x", "y", true, `XY z=${c.z.toFixed(3)}`);
}

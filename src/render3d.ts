/* ================================================================
   3-D SLICE BOX
   ================================================================
   The same three orthogonal planes render.ts lays out side by side,
   placed where they actually are: three intersecting cuts inside the
   tissue volume, in a box with cm ticks on its outer edges — the view
   that shows how the planes relate (that the bright core on the XZ cut
   is the same core the XY cut passes through), which the flat
   triptych can only imply.

   Two decisions carry the whole file.

   ORTHOGRAPHIC, NOT PERSPECTIVE. An orthographic projection is linear,
   so an axis-aligned rectangle projects to a parallelogram and the map
   from image pixels to it is exactly an affine transform, which canvas
   2-D applies natively (ctx.setTransform + drawImage) — so a slice
   plane stamps into 3-D space pixel-exactly with no shaders, no
   WebGL, and no dependency. It's also what MATLAB's own 3-D axes do
   by default, matching the published figures this is modelled on.
   Perspective would need each quad subdivided per scanline for no
   gain here.

   OCCLUSION WITHOUT A DEPTH BUFFER. Three full planes intersect, so no
   fixed draw order is right. Splitting each plane into its four
   quadrants at the other two planes' positions gives 12 quads that no
   longer interpenetrate, and those three planes are then exactly a
   BSP tree whose back-to-front traversal is the order — which
   collapses to a sort key of three base-3 digits, one per plane (see
   orderKey), and being a BSP traversal is exact rather than a
   heuristic that usually works.

   That precision matters: sorting quads by centroid depth fails as
   soon as the box or slice positions are lopsided — a 0.05 cm film
   under a 6 cm window puts a large fragment's centroid nearer than a
   small one's while the surfaces themselves order the other way.
   That is exactly what a painter's algorithm cannot be built on.

   Coordinates: this file works in a display frame [X, Y, Zup] where
   Zup = -z, so screen-up is up and the tissue surface is the top of
   the box. Everything crossing the API stays in the scene's own
   convention (depth positive downward from the surface).
   ================================================================ */

import {
  type PlaneCache,
  type Probe,
  type SliceAxis,
  type SliceScene,
  CROSSHAIR_INK,
  PANEL_BG,
  TICK_FONT,
  TITLE_FONT,
  colormapFloor,
  niceTicks,
  planeCanvas,
  probeAt,
  setSmoothing,
  sliceCenters,
  sliceDims,
  tickLabels,
} from "./render";

export interface Camera {
  /** Rotation about the depth axis [rad]. */
  az: number;
  /** Angle above the horizon [rad], kept short of ±90° so "up" stays defined. */
  el: number;
}

/* No zoom, deliberately. The box is fitted to the canvas at equal aspect, so
   the fit is always tight in one dimension already — magnifying past it just
   carries the box edges, and with them every tick, off the canvas, leaving a
   view with no axes at all. Magnification here would have to draw the axes on
   the canvas frame rather than on the box, which is a different picture; the
   controls that actually help are the orbit and the grid extents. */

/** A 3/4 view from above: +x to the lower right, +y to the upper right,
    the surface on top. The same angle MATLAB's `view(3)` settles on, which
    is the angle the reference figures for this were drawn at. */
export const DEFAULT_CAMERA: Camera = { az: -Math.PI / 3, el: Math.PI / 6 };

export const EL_LIMIT = (85 * Math.PI) / 180;

type V3 = [number, number, number];

const EDGE_INK = "rgba(120,150,185,0.45)";
const SILHOUETTE_INK = "rgba(150,180,215,0.8)";
const FACE_TINT = ["rgba(255,255,255,0.02)", "rgba(255,255,255,0.05)", "rgba(255,255,255,0.035)"];
const AXIS_INK = "rgba(190,205,225,0.8)";
const INTERFACE_INK = "rgba(255,255,255,0.4)";

/* Room outside the box for the ticks and the axis titles. The ticks sit 8 px
   out from an edge and the titles hang off that edge's outer end, so what has
   to fit is roughly the longest tick label plus that offset — and labelAt
   drops anything that still would not, so a cramped canvas loses a label
   rather than showing half of one. */
const M_SIDE = 58;
const M_TOP = 24;
const M_BOT = 46;

/* ================================================================
   PROJECTION
   ================================================================ */
interface Projector {
  right: V3;
  up: V3;
  /** Toward the camera — larger p·dir is nearer. */
  dir: V3;
  s: number;
  ox: number;
  oy: number;
}

function basis(cam: Camera): { right: V3; up: V3; dir: V3 } {
  const ca = Math.cos(cam.az),
    sa = Math.sin(cam.az),
    ce = Math.cos(cam.el),
    se = Math.sin(cam.el);
  return {
    dir: [ce * ca, ce * sa, se],
    right: [-sa, ca, 0],
    up: [-se * ca, -se * sa, ce],
  };
}

const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function project(pr: Projector, p: V3): [number, number] {
  return [pr.ox + pr.s * dot(p, pr.right), pr.oy - pr.s * dot(p, pr.up)];
}

/** The box in the display frame: x and y centred on the beam axis, and depth
    negated so that screen-up is up and the tissue surface is the top face. */
function bounds(scene: SliceScene): { lo: V3; hi: V3 } | null {
  const { lx, ly, lz } = scene;
  if (!(lx > 0) || !(ly > 0) || !(lz > 0)) return null;
  return { lo: [-lx / 2, -ly / 2, -lz], hi: [lx / 2, ly / 2, 0] };
}

/** The slice planes' meeting point, in the display frame. */
function displayCenter(scene: SliceScene): V3 {
  const c = sliceCenters(scene);
  return [c.x, c.y, -c.z];
}

/** Fit the box to the canvas at equal aspect — so a 0.3 cm epidermis over a
    1.7 cm dermis is drawn as the thin layer it is, rather than stretched to
    fill the box. Returns null when there is no room to draw in. */
function projector(lo: V3, hi: V3, cam: Camera, W: number, H: number): Projector | null {
  const b = basis(cam);
  const fit: Projector = { ...b, s: 1, ox: 0, oy: 0 };
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (let c = 0; c < 8; c++) {
    const [px, py] = project(fit, corner(lo, hi, c));
    minX = Math.min(minX, px);
    maxX = Math.max(maxX, px);
    minY = Math.min(minY, py);
    maxY = Math.max(maxY, py);
  }
  const availW = W - 2 * M_SIDE;
  const availH = H - M_TOP - M_BOT;
  if (availW < 40 || availH < 40) return null;
  const s = Math.min(availW / (maxX - minX), availH / (maxY - minY));
  return {
    ...b,
    s,
    ox: M_SIDE + availW / 2 - ((minX + maxX) / 2) * s,
    oy: M_TOP + availH / 2 - ((minY + maxY) / 2) * s,
  };
}

/* ================================================================
   BOX TOPOLOGY
   ================================================================
   Corners are bit patterns: bit i set means "at the high end of axis
   i". An edge joins two corners differing in exactly one bit, and is
   parallel to that axis. Under an orthographic projection exactly one
   corner is nearest and one farthest, and every edge either touches
   the nearest corner (in front of everything inside the box), touches
   the farthest (behind everything), or does neither — in which case
   it is on the silhouette and nothing inside can cover it. That is
   the whole visibility question for a box, answered by two dot
   products.
   ================================================================ */
interface Edge {
  a: number;
  b: number;
  axis: number;
  kind: "front" | "back" | "silhouette";
}

function corner(lo: V3, hi: V3, bits: number): V3 {
  return [bits & 1 ? hi[0] : lo[0], bits & 2 ? hi[1] : lo[1], bits & 4 ? hi[2] : lo[2]];
}

function classifyEdges(lo: V3, hi: V3, pr: Projector): { edges: Edge[]; near: number; far: number } {
  let near = 0,
    far = 0,
    dNear = -Infinity,
    dFar = Infinity;
  for (let c = 0; c < 8; c++) {
    const d = dot(corner(lo, hi, c), pr.dir);
    if (d > dNear) {
      dNear = d;
      near = c;
    }
    if (d < dFar) {
      dFar = d;
      far = c;
    }
  }
  const edges: Edge[] = [];
  for (let c = 0; c < 8; c++) {
    for (let axis = 0; axis < 3; axis++) {
      const bit = 1 << axis;
      if (c & bit) continue; // enumerate each edge once, from its low end
      const b = c | bit;
      const kind =
        c === near || b === near ? "front" : c === far || b === far ? "back" : "silhouette";
      edges.push({ a: c, b, axis, kind });
    }
  }
  return { edges, near, far };
}

/* ================================================================
   SLICE QUADS
   ================================================================ */
/** For a slice plane, which display axes its image spans and in which
    direction. Image rows run *down* into the tissue on the two depth
    planes, which is -1 along Zup. */
const PLANE_AXES: Record<SliceAxis, { u: number; uDir: 1; v: number; vDir: 1 | -1 }> = {
  0: { u: 1, uDir: 1, v: 2, vDir: -1 },
  1: { u: 0, uDir: 1, v: 2, vDir: -1 },
  2: { u: 0, uDir: 1, v: 1, vDir: 1 },
};

interface Quad {
  axis: SliceAxis;
  /** Projected outline, for clipping. */
  poly: [number, number][];
  /** Position in the planes' BSP traversal — see orderKey. */
  order: number;
}

/** Back-to-front position of one quad, as three base-3 digits, most
    significant first: one per plane, taken in axis order. At the plane's own
    level the digit is 1; above it, 0 if this quad lies on the far side of
    that plane and 2 if on the near side. So a plane's fragments sort between
    everything behind it and everything in front of it, at every level — the
    BSP traversal, written as a number. Digits below the quad's own level are
    0: those levels split planes this quad is not on, and its coplanar
    siblings all share the value, which is fine because coplanar quads cannot
    overlap. */
function orderKey(axis: number, sides: number[], near: number[]): number {
  let k = 0;
  for (let level = 0; level < 3; level++) {
    const digit = level < axis ? (sides[level] === near[level] ? 2 : 0) : level === axis ? 1 : 0;
    k = k * 3 + digit;
  }
  return k;
}

/** Inflate a convex polygon about its centroid by `pad` device pixels, so
    that adjacent quads overlap by a hairline instead of leaving one — a
    shared clip edge is antialiased on both sides, which lets a quarter of the
    background through as a visible seam.

    Capped at a quarter of each vertex's own distance from the centroid, for
    the quads a fixed half-pixel is not small compared to: a slice sitting
    close to a face leaves a sliver of a quadrant a few pixels across, and
    there half a pixel would spill it past where it belongs and over a
    neighbour that is genuinely in front of it. */
function inflate(poly: [number, number][], pad: number): [number, number][] {
  let cx = 0,
    cy = 0;
  poly.forEach(([x, y]) => {
    cx += x;
    cy += y;
  });
  cx /= poly.length;
  cy /= poly.length;
  return poly.map(([x, y]) => {
    const dx = x - cx,
      dy = y - cy;
    const r = Math.hypot(dx, dy);
    if (r < 1e-9) return [x, y];
    const k = Math.min(pad, r * 0.25) / r;
    return [x + dx * k, y + dy * k];
  });
}

function polyArea(poly: [number, number][]): number {
  let a = 0;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    a += poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1];
  }
  return Math.abs(a) / 2;
}

/** The 12 quads, ordered back to front. */
function buildQuads(lo: V3, hi: V3, ctr: V3, pr: Projector): Quad[] {
  const quads: Quad[] = [];
  /** For each axis, which of its two halves the camera is on. */
  const near = [0, 1, 2].map((i) => (pr.dir[i] > 0 ? 1 : 0));
  ([0, 1, 2] as SliceAxis[]).forEach((axis) => {
    const { u, v } = PLANE_AXES[axis];
    for (const uSide of [0, 1]) {
      for (const vSide of [0, 1]) {
        const u0 = uSide ? ctr[u] : lo[u];
        const u1 = uSide ? hi[u] : ctr[u];
        const v0 = vSide ? ctr[v] : lo[v];
        const v1 = vSide ? hi[v] : ctr[v];
        if (u1 - u0 <= 0 || v1 - v0 <= 0) continue; // slice sits on a face

        const at = (uu: number, vv: number): V3 => {
          const p: V3 = [0, 0, 0];
          p[axis] = ctr[axis];
          p[u] = uu;
          p[v] = vv;
          return p;
        };
        const poly: [number, number][] = [
          project(pr, at(u0, v0)),
          project(pr, at(u1, v0)),
          project(pr, at(u1, v1)),
          project(pr, at(u0, v1)),
        ];
        if (polyArea(poly) < 0.5) continue; // edge-on

        const sides = [0, 0, 0];
        sides[u] = uSide;
        sides[v] = vSide;
        quads.push({ axis, poly, order: orderKey(axis, sides, near) });
      }
    }
  });
  quads.sort((a, b) => a.order - b.order);
  return quads;
}

/* ================================================================
   AXES
   ================================================================ */
/** Which edge to hang an axis's ticks on: the one furthest from the box's
    centre on screen, so the ticks end up outside the drawing, and never a
    front edge, so they never sit on top of the data. */
function tickEdge(edges: Edge[], axis: number, lo: V3, hi: V3, pr: Projector, ctrProj: [number, number]): Edge | null {
  let best: Edge | null = null;
  let bestD = -Infinity;
  edges.forEach((e) => {
    if (e.axis !== axis || e.kind === "front") return;
    const pa = corner(lo, hi, e.a),
      pb = corner(lo, hi, e.b);
    const mid: V3 = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
    const [mx, my] = project(pr, mid);
    const d = Math.hypot(mx - ctrProj[0], my - ctrProj[1]);
    if (d > bestD) {
      bestD = d;
      best = e;
    }
  });
  return best;
}

function alignFor(ctx: CanvasRenderingContext2D, ox: number, oy: number): void {
  ctx.textAlign = ox > 0.3 ? "left" : ox < -0.3 ? "right" : "center";
  ctx.textBaseline = oy > 0.3 ? "top" : oy < -0.3 ? "bottom" : "middle";
}

/** Draw a label only if all of it lands on the canvas — measured, not
    guessed from the anchor, since the anchor is an edge of the text for every
    alignment but "center". A tick clipped in half is worse than a tick
    missing, so this decides rather than the canvas edge does. */
function labelAt(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  W: number,
  H: number
): void {
  const w = ctx.measureText(text).width;
  const h = parseInt(ctx.font, 10) || 11;
  const x0 = ctx.textAlign === "left" ? x : ctx.textAlign === "right" ? x - w : x - w / 2;
  const y0 = ctx.textBaseline === "top" ? y : ctx.textBaseline === "bottom" ? y - h : y - h / 2;
  if (x0 < 1 || x0 + w > W - 1 || y0 < 1 || y0 + h > H - 1) return;
  ctx.fillText(text, x, y);
}

/* ================================================================
   THE RENDERER
   ================================================================ */
export function drawBox3D(
  cvId: string,
  scene: SliceScene,
  cache: PlaneCache,
  cam: Camera
): void {
  const cv = document.getElementById(cvId) as HTMLCanvasElement;
  const W = cv.width || cv.offsetWidth || 400;
  const H = cv.height || cv.offsetHeight || 400;
  cv.width = W;
  cv.height = H;

  const ctx = cv.getContext("2d")!;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = PANEL_BG;
  ctx.fillRect(0, 0, W, H);

  const { view, lz } = scene;
  const box = bounds(scene);
  if (!box) return;
  const { lo, hi } = box;
  const pr = projector(lo, hi, cam, W, H);
  if (!pr) return;

  const ctr = displayCenter(scene);
  const { edges } = classifyEdges(lo, hi, pr);
  const ctrProj = project(pr, [0, 0, -lz / 2]);

  const poly = (pts: V3[]) => {
    ctx.beginPath();
    pts.forEach((p, i) => {
      const [px, py] = project(pr, p);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
  };

  /* ── the three back walls ─────────────────────────────────────
     Filled with the colormap's own zero colour, so the weak end of the
     field meets the background without a visible edge. A faint white
     tint on top separates the walls from each other. */
  const floor = colormapFloor(view.lut);
  const walls: { axis: number; side: number }[] = [];
  for (let axis = 0; axis < 3; axis++) {
    const side = pr.dir[axis] > 0 ? 0 : 1; // the face the camera is *not* on
    walls.push({ axis, side });
    const bit = 1 << axis;
    const base = side ? bit : 0;
    const [o1, o2] = [0, 1, 2].filter((i) => i !== axis).map((i) => 1 << i);
    poly([
      corner(lo, hi, base),
      corner(lo, hi, base | o1),
      corner(lo, hi, base | o1 | o2),
      corner(lo, hi, base | o2),
    ]);
    ctx.fillStyle = floor;
    ctx.fill();
    /* A different tint per wall, so the three read as three planes rather
       than as one silhouette. */
    ctx.fillStyle = FACE_TINT[axis];
    ctx.fill();
  }

  /* Everything but the three edges meeting at the near corner — those
     would be drawn across the data, and MATLAB's own boxed 3-D axes leave
     them out for the same reason. */
  ctx.strokeStyle = EDGE_INK;
  ctx.lineWidth = 1;
  edges.forEach((e) => {
    if (e.kind === "front") return;
    const [ax, ay] = project(pr, corner(lo, hi, e.a));
    const [bx, by] = project(pr, corner(lo, hi, e.b));
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  });

  /* ── layer interfaces, on the two vertical back walls ─────────
     On the walls rather than on the slice planes: they are then never
     occluded, and they read as depths of the whole stack rather than as
     features of one cut. */
  if (scene.interfaces.length) {
    ctx.save();
    ctx.strokeStyle = INTERFACE_INK;
    ctx.setLineDash([4, 3]);
    scene.interfaces.forEach((d) => {
      if (!(d > 0) || d >= lz) return;
      walls
        .filter((w) => w.axis !== 2)
        .forEach(({ axis, side }) => {
          const fixed = side ? hi[axis] : lo[axis];
          const other = axis === 0 ? 1 : 0;
          const p0: V3 = [0, 0, -d];
          const p1: V3 = [0, 0, -d];
          p0[axis] = fixed;
          p1[axis] = fixed;
          p0[other] = lo[other];
          p1[other] = hi[other];
          const [x0, y0] = project(pr, p0);
          const [x1, y1] = project(pr, p1);
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          ctx.stroke();
        });
    });
    ctx.restore();
  }

  /* ── the slices ───────────────────────────────────────────────
     Each quad clips to its own outline, then the *whole* plane image is
     stamped with the plane's affine transform. Clipping rather than
     slicing the image into quarters keeps the pixel grid continuous
     across the split — there is no seam to line up, because it is one
     draw of one image, four times over. */
  setSmoothing(ctx, view);
  /* Once per plane, not once per quad: the four quads of a plane are four
     clipped draws of the same image. */
  const images = ([0, 1, 2] as SliceAxis[]).map((axis) =>
    planeCanvas(cache, view, axis, axis === 0 ? scene.ix : axis === 1 ? scene.iy : scene.iz)
  );
  buildQuads(lo, hi, ctr, pr).forEach((q) => {
    const img = images[q.axis];
    const [iw, ih] = sliceDims(view, q.axis);
    const { u, v, vDir } = PLANE_AXES[q.axis];

    const origin: V3 = [0, 0, 0];
    origin[q.axis] = ctr[q.axis];
    origin[u] = lo[u];
    origin[v] = vDir > 0 ? lo[v] : hi[v];

    const uEnd: V3 = [...origin];
    uEnd[u] = hi[u];
    const vEnd: V3 = [...origin];
    vEnd[v] = vDir > 0 ? hi[v] : lo[v];

    const O = project(pr, origin);
    const U = project(pr, uEnd);
    const V = project(pr, vEnd);

    ctx.save();
    const p = inflate(q.poly, 0.5);
    ctx.beginPath();
    p.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.closePath();
    ctx.clip();
    ctx.setTransform((U[0] - O[0]) / iw, (U[1] - O[1]) / iw, (V[0] - O[0]) / ih, (V[1] - O[1]) / ih, O[0], O[1]);
    ctx.drawImage(img, 0, 0);
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.restore();
  });

  /* Crosshairs: the three lines where the planes cut each other. The seams
     between quads are invisible — the same field continues across them — so
     without these there is nothing to say where the cuts actually are, and the
     arrangement reads as one shape rather than three planes. Same accent
     colour the flat layout uses for the same purpose. */
  ctx.strokeStyle = CROSSHAIR_INK;
  ctx.lineWidth = 1;
  for (let axis = 0; axis < 3; axis++) {
    /* The line along `axis` through the meeting point: the two planes that do
       not span it intersect here. */
    const a: V3 = [...ctr];
    const b2: V3 = [...ctr];
    a[axis] = lo[axis];
    b2[axis] = hi[axis];
    const [ax, ay] = project(pr, a);
    const [bx, by] = project(pr, b2);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  /* The silhouette re-stroked over the slices, so the box keeps a crisp
     outline where the planes run out to meet it. */
  ctx.strokeStyle = SILHOUETTE_INK;
  ctx.lineWidth = 1;
  edges.forEach((e) => {
    if (e.kind !== "silhouette") return;
    const [ax, ay] = project(pr, corner(lo, hi, e.a));
    const [bx, by] = project(pr, corner(lo, hi, e.b));
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  });

  /* ── ticks ────────────────────────────────────────────────────
     x and y are measured as they are stored, from the beam axis. The
     depth axis is labelled 0 at the surface increasing downward — the
     convention every thickness in the parameter panel uses — even though
     it runs the other way in the display frame. */
  const AXIS_NAMES = ["x", "y", "z"];
  const ranges: [number, number][] = [
    [lo[0], hi[0]],
    [lo[1], hi[1]],
    [0, lz],
  ];
  ctx.font = TICK_FONT;
  ctx.fillStyle = AXIS_INK;
  ctx.strokeStyle = AXIS_INK;

  for (let axis = 0; axis < 3; axis++) {
    const e = tickEdge(edges, axis, lo, hi, pr, ctrProj);
    if (!e) continue;
    const pa = corner(lo, hi, e.a);
    const mid: V3 = [...pa];
    const pb = corner(lo, hi, e.b);
    mid[axis] = (pa[axis] + pb[axis]) / 2;
    const [mx, my] = project(pr, mid);
    let ox = mx - ctrProj[0],
      oy = my - ctrProj[1];
    const on = Math.hypot(ox, oy) || 1;
    ox /= on;
    oy /= on;

    const [r0, r1] = ranges[axis];
    const vals = niceTicks(r0, r1, 4).filter((v) => v >= r0 - 1e-12 && v <= r1 + 1e-12);
    const text = tickLabels(vals);
    ctx.font = TICK_FONT;
    alignFor(ctx, ox, oy);
    vals.forEach((val, i) => {
      const p: V3 = [...pa];
      /* Depth grows downward, the display axis grows upward. */
      p[axis] = axis === 2 ? -val : val;
      const [px, py] = project(pr, p);
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + ox * 5, py + oy * 5);
      ctx.stroke();
      labelAt(ctx, text[i], px + ox * 8, py + oy * 8, W, H);
    });

    /* Interfaces get a longer mark on the depth axis, tying the dashed
       lines on the walls to a readable depth. */
    if (axis === 2 && scene.interfaces.length) {
      ctx.save();
      ctx.strokeStyle = INTERFACE_INK;
      ctx.lineWidth = 2;
      scene.interfaces.forEach((d) => {
        if (!(d > 0) || d >= lz) return;
        const p: V3 = [...pa];
        p[2] = -d;
        const [px, py] = project(pr, p);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(px + ox * 5, py + oy * 5);
        ctx.stroke();
      });
      ctx.restore();
    }

    /* At the far end of the axis rather than out from its middle: the tick
       labels already occupy the band from 8 px outward to their own width, so
       a title placed perpendicular to the edge lands on top of them. Hanging
       it off whichever end of the edge is further from the box's centre puts
       it clear of them and reads as naming the axis it sits at the end of. */
    ctx.font = TITLE_FONT;
    const ends: V3[] = [pa, pb];
    const outer = ends.reduce((best, p) => {
      const [bx, by] = project(pr, best);
      const [qx, qy] = project(pr, p);
      return Math.hypot(qx - ctrProj[0], qy - ctrProj[1]) > Math.hypot(bx - ctrProj[0], by - ctrProj[1])
        ? p
        : best;
    });
    const [ex, ey] = project(pr, outer);
    let tx = ex - ctrProj[0],
      ty = ey - ctrProj[1];
    const tn = Math.hypot(tx, ty) || 1;
    tx /= tn;
    ty /= tn;
    alignFor(ctx, tx, ty);
    labelAt(ctx, `${AXIS_NAMES[axis]} [cm]`, ex + tx * 12, ey + ty * 12, W, H);
  }
}

/* ================================================================
   PROBING
   ================================================================ */
/** What is under (px, py) on a 3-D box of W x H, or null if the cursor is off
    the slices. Inverts the same projection the drawing used.

    The visible surface at a screen point is whichever plane's intersection
    with the view ray is nearest the camera and still inside the box, which is
    the whole of the hit test: the planes are full planes, so if one lies in
    front of another there, it is the one you can see. */
export function pick3D(
  scene: SliceScene,
  W: number,
  H: number,
  cam: Camera,
  px: number,
  py: number
): Probe | null {
  const box = bounds(scene);
  if (!box) return null;
  const { lo, hi } = box;
  const pr = projector(lo, hi, cam, W, H);
  if (!pr) return null;
  const ctr = displayCenter(scene);

  /* The basis is orthonormal, so a screen point fixes two of the three
     coordinates of every point on its view ray: P = ar·right + au·up + t·dir.
     Requiring P[axis] = ctr[axis] then gives t outright — and t, being the
     component along dir, *is* the depth. */
  const ar = (px - pr.ox) / pr.s;
  const au = -(py - pr.oy) / pr.s;

  let best: { t: number; p: V3 } | null = null;
  for (let axis = 0; axis < 3; axis++) {
    if (Math.abs(pr.dir[axis]) < 1e-9) continue; // edge-on: nothing to hit
    const t = (ctr[axis] - ar * pr.right[axis] - au * pr.up[axis]) / pr.dir[axis];
    const p: V3 = [0, 0, 0];
    let inside = true;
    for (let i = 0; i < 3; i++) {
      p[i] = ar * pr.right[i] + au * pr.up[i] + t * pr.dir[i];
      /* A hair of tolerance so the box's own faces count as hits rather than
         a one-pixel dead border. */
      const eps = (hi[i] - lo[i]) * 1e-9;
      if (p[i] < lo[i] - eps || p[i] > hi[i] + eps) inside = false;
    }
    if (!inside) continue;
    if (!best || t > best.t) best = { t, p };
  }
  if (!best) return null;
  /* Back to the scene's convention, where depth is positive downward. */
  return probeAt(scene, best.p[0], best.p[1], -best.p[2]);
}

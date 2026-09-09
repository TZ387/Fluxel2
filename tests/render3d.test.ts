/* ================================================================
   3-D SLICE BOX
   ================================================================
   Checks render3d.ts against an oracle that derives the same geometry
   independently — its own camera basis from the az/el angles, its own
   projection, its own plane positions — so a mistake in the renderer
   can't cancel out against the test. Three questions:

     1. does each plane image land on that plane's projected rectangle;
     2. wherever two drawn quads overlap on screen, is the one drawn
        later genuinely the nearer one;
     3. does every tick and title actually get drawn (the renderer
        drops a label that wouldn't fit, so a margin too tight shows up
        as a missing label, not a clipped one).

   Question 2 is the point of the file: occlusion order is the one
   piece that's easy to get wrong and hard to see. The first version
   sorted by centroid depth, which looks right at the default angle
   and is wrong the moment the extents are lopsided — so this checks
   over the whole camera sphere, since "correct from this angle" isn't
   the claim being made.
   ================================================================ */
import { drawBox3D, pick3D, DEFAULT_CAMERA, type Camera } from "../src/render3d";
import { makeScale, colormapLut, createPlaneCache, type VolumeView } from "../src/render";
import {
  apply,
  createStubCanvas,
  expect,
  fail,
  finish,
  inCase,
  installStubDocument,
  type ImageOp,
  type Pt,
} from "./harness";

const CANVAS = 520;
/* Must match render3d.ts's own margins — the oracle reproduces its fit rule,
   and a disagreement in either direction is a failure worth seeing. */
const M_SIDE = 58,
  M_TOP = 24,
  M_BOT = 46;

const canvas = createStubCanvas(CANVAS, CANVAS);
installStubDocument(canvas);

type V3 = [number, number, number];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Convex polygon inset radially by `pad` px, or null if it has no interior
    left. The renderer inflates its clip paths by half a pixel so adjacent
    quads overlap rather than leave an antialiased seam, and where two quads
    meet along a shared edge the true depth difference goes to zero — so
    comparing depths inside that sliver measures the inflation, not the
    ordering. Insetting both sides keeps the comparison honest. */
function inset(poly: Pt[], pad: number): Pt[] | null {
  const cx = poly.reduce((t, p) => t + p.x, 0) / poly.length;
  const cy = poly.reduce((t, p) => t + p.y, 0) / poly.length;
  const out: Pt[] = [];
  for (const p of poly) {
    const dx = p.x - cx,
      dy = p.y - cy,
      r = Math.hypot(dx, dy);
    if (r <= pad * 1.5) return null;
    out.push({ x: cx + dx * (1 - pad / r), y: cy + dy * (1 - pad / r) });
  }
  return out;
}

function inPoly(poly: Pt[], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i],
      b = poly[j];
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

/** Which world axes a plane's image spans, and in which direction — image rows
    run *down* into the tissue on the two depth planes, which is -1 along the
    display frame's upward axis. Mirrors render3d.ts's PLANE_AXES. */
const PLANE = [
  { u: 1, v: 2, vDir: -1 },
  { u: 0, v: 2, vDir: -1 },
  { u: 0, v: 1, vDir: 1 },
];

let cases = 0,
  richCases = 0,
  pairs = 0,
  samples = 0,
  probes = 0,
  probeMisses = 0;

function runCase(
  nx: number,
  ny: number,
  nz: number,
  lx: number,
  ly: number,
  lz: number,
  ix: number,
  iy: number,
  iz: number,
  cam: Camera,
  label: string
): void {
  inCase(label);
  canvas.reset();

  const data = new Float32Array(nx * ny * nz).fill(1);
  const view: VolumeView = {
    data,
    nx,
    ny,
    nz,
    scale: makeScale("log", 1, 100),
    lut: colormapLut("inferno"),
    validity: null,
    showValidity: false,
    stamp: label,
  };
  const scene = { view, lx, ly, lz, ix, iy, iz, interfaces: [lz / 3] };
  drawBox3D("cv", scene, createPlaneCache(), cam);

  /* ── the oracle ── */
  const ca = Math.cos(cam.az),
    sa = Math.sin(cam.az),
    ce = Math.cos(cam.el),
    se = Math.sin(cam.el);
  const dir: V3 = [ce * ca, ce * sa, se];
  const right: V3 = [-sa, ca, 0];
  const up: V3 = [-se * ca, -se * sa, ce];
  const lo: V3 = [-lx / 2, -ly / 2, -lz];
  const hi: V3 = [lx / 2, ly / 2, 0];
  const ctr: V3 = [
    ((ix + 0.5) * lx) / nx - lx / 2,
    ((iy + 0.5) * ly) / ny - ly / 2,
    -(((iz + 0.5) * lz) / nz),
  ];

  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (let c = 0; c < 8; c++) {
    const p: V3 = [c & 1 ? hi[0] : lo[0], c & 2 ? hi[1] : lo[1], c & 4 ? hi[2] : lo[2]];
    const px = dot(p, right),
      py = -dot(p, up);
    minX = Math.min(minX, px);
    maxX = Math.max(maxX, px);
    minY = Math.min(minY, py);
    maxY = Math.max(maxY, py);
  }
  const availW = CANVAS - 2 * M_SIDE,
    availH = CANVAS - M_TOP - M_BOT;
  const s = Math.min(availW / (maxX - minX), availH / (maxY - minY));
  const ox = M_SIDE + availW / 2 - ((minX + maxX) / 2) * s;
  const oy = M_TOP + availH / 2 - ((minY + maxY) / 2) * s;
  const proj = (p: V3): Pt => ({ x: ox + s * dot(p, right), y: oy - s * dot(p, up) });

  /* A plane's image dimensions are what say which plane it is, so the three
     have to be distinguishable for the oracle to work at all. */
  const byImage = new Map<string, number>([
    [`${ny},${nz}`, 0],
    [`${nx},${nz}`, 1],
    [`${nx},${ny}`, 2],
  ]);
  if (byImage.size !== 3) {
    fail(`grid ${nx}x${ny}x${nz} gives two planes the same shape; pick distinct dimensions`);
    return;
  }
  const axisOf = (op: ImageOp) => byImage.get(`${op.sw},${op.sh}`);

  const images = canvas.images();
  cases++;

  /* Two ways the quad count drops below 12, both correct: a slice sitting on a
     face has no quads on that side, and a plane seen exactly edge-on
     (dir[axis] == 0, so az a multiple of 90 or el == 0) projects to a line and
     loses all four of its own. */
  const interior = ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1 && iz > 0 && iz < nz - 1;
  const edgeOn = dir.filter((d) => Math.abs(d) < 1e-9).length;
  if (interior) {
    expect(images.length === 4 * (3 - edgeOn), `expected ${4 * (3 - edgeOn)} quads, got ${images.length}`);
  }

  /* ── 1. transforms ── */
  images.forEach((op, k) => {
    const axis = axisOf(op);
    if (axis === undefined) return fail(`draw ${k}: unrecognised image ${op.sw}x${op.sh}`);
    const { u, v, vDir } = PLANE[axis];
    const origin: V3 = [0, 0, 0];
    origin[axis] = ctr[axis];
    origin[u] = lo[u];
    origin[v] = vDir > 0 ? lo[v] : hi[v];
    const uEnd: V3 = [...origin];
    uEnd[u] = hi[u];
    const vEnd: V3 = [...origin];
    vEnd[v] = vDir > 0 ? hi[v] : lo[v];
    const O = proj(origin),
      U = proj(uEnd),
      V = proj(vEnd);
    /* The image's own corners, put through the transform the renderer used. */
    const gotO = apply(op.matrix, 0, 0);
    const gotU = apply(op.matrix, op.sw, 0);
    const gotV = apply(op.matrix, 0, op.sh);
    const err = Math.max(
      Math.hypot(gotO.x - O.x, gotO.y - O.y),
      Math.hypot(gotU.x - U.x, gotU.y - U.y),
      Math.hypot(gotV.x - V.x, gotV.y - V.y)
    );
    expect(err < 1e-6, `draw ${k} (axis ${axis}): image corners off by ${err.toExponential(2)} px`);
  });

  /* ── 2. occlusion ── */
  /* Depth of the point on plane `axis` seen at (px, py): the view basis is
     orthonormal, so P = ar·right + au·up + t·dir, and fixing P[axis] gives t
     directly — which *is* the depth along the view direction. */
  const depthAt = (axis: number, px: number, py: number): number | null => {
    if (Math.abs(dir[axis]) < 1e-9) return null;
    const ar = (px - ox) / s,
      au = -(py - oy) / s;
    return (ctr[axis] - ar * right[axis] - au * up[axis]) / dir[axis];
  };

  let overlaps = 0;
  for (let a = 0; a < images.length; a++) {
    for (let b = a + 1; b < images.length; b++) {
      const A = images[a],
        B = images[b];
      const axA = axisOf(A),
        axB = axisOf(B);
      if (axA === undefined || axB === undefined || axA === axB) continue; // coplanar quads cannot overlap
      if (!A.clip || !B.clip) return fail(`draw ${a} or ${b} was not clipped to its quad`);
      const inA = inset(A.clip, 1.5),
        inB = inset(B.clip, 1.5);
      if (!inA || !inB) continue;
      let hit = false;
      for (let i = 1; i < 8; i++) {
        for (let j = 1; j < 8; j++) {
          /* Bilinear sample of B's (convex, four-cornered) outline. */
          const fu = i / 8,
            fv = j / 8;
          const [p0, p1, p2, p3] = inB;
          const x =
            (1 - fu) * (1 - fv) * p0.x + fu * (1 - fv) * p1.x + fu * fv * p2.x + (1 - fu) * fv * p3.x;
          const y =
            (1 - fu) * (1 - fv) * p0.y + fu * (1 - fv) * p1.y + fu * fv * p2.y + (1 - fu) * fv * p3.y;
          if (!inPoly(inA, x, y)) continue;
          const dA = depthAt(axA, x, y),
            dB = depthAt(axB, x, y);
          if (dA === null || dB === null) continue;
          hit = true;
          samples++;
          /* B was drawn after A, so B must be the nearer surface here. */
          expect(
            dB >= dA - 1e-9,
            `quad ${b} (axis ${axB}) drawn over ${a} (axis ${axA}) but sits ${(dA - dB).toExponential(2)} cm behind it at (${x.toFixed(1)}, ${y.toFixed(1)})`
          );
        }
      }
      if (hit) overlaps++;
    }
  }
  pairs += overlaps;
  if (overlaps >= 6) richCases++;

  /* ── 3. picking ──
     pick3D inverts this same projection, so the test of it is a round trip:
     probe a screen point, project the world position that comes back, and it
     has to land where the probe started. Then the harder half — that it picked
     the *visible* plane, which is the nearest of the three whose intersection
     with the view ray is still inside the box. A hit test that quietly returns
     a plane hidden behind another would read out a number from a voxel the
     user cannot see. */
  const planeHits = (px: number, py: number): { axis: number; t: number; p: V3 }[] => {
    const ar = (px - ox) / s,
      au = -(py - oy) / s;
    const hits: { axis: number; t: number; p: V3 }[] = [];
    for (let axis = 0; axis < 3; axis++) {
      if (Math.abs(dir[axis]) < 1e-9) continue;
      const t = (ctr[axis] - ar * right[axis] - au * up[axis]) / dir[axis];
      const p: V3 = [0, 0, 0];
      let inside = true;
      for (let i = 0; i < 3; i++) {
        p[i] = ar * right[i] + au * up[i] + t * dir[i];
        const eps = (hi[i] - lo[i]) * 1e-9;
        if (p[i] < lo[i] - eps || p[i] > hi[i] + eps) inside = false;
      }
      if (inside) hits.push({ axis, t, p });
    }
    return hits;
  };

  for (let a = 1; a < 8; a++) {
    for (let b = 1; b < 8; b++) {
      const px = (CANVAS * a) / 8,
        py = (CANVAS * b) / 8;
      const probe = pick3D(scene, CANVAS, CANVAS, cam, px, py);
      const hits = planeHits(px, py);
      if (hits.length === 0) {
        expect(probe === null, `probed (${px}, ${py}) where no plane is visible`);
        probeMisses++;
        continue;
      }
      if (!probe) {
        /* The oracle sees a hit but the renderer returned nothing — allowed
           only when that hit lands outside the voxel grid, which the box's
           own far faces do by a rounding hair. */
        const nearest = hits.reduce((m, h) => (h.t > m.t ? h : m));
        const onEdge = [nearest.p[0], nearest.p[1], -nearest.p[2]].some((v, i) => {
          const bound = [lx, ly, lz][i];
          const at = i === 2 ? v : v + bound / 2;
          return at <= 1e-9 || at >= bound - 1e-9;
        });
        expect(onEdge, `no probe at (${px}, ${py}) though plane ${nearest.axis} is visible there`);
        probeMisses++;
        continue;
      }
      probes++;
      /* Round trip: the position that came back must project to where we asked. */
      const back = proj([probe.x, probe.y, -probe.z]);
      const err = Math.hypot(back.x - px, back.y - py);
      expect(err < 1e-6, `probe at (${px}, ${py}) reprojects ${err.toExponential(2)} px away`);
      /* It must be the nearest visible plane, not merely one of them. */
      const nearest = hits.reduce((m, h) => (h.t > m.t ? h : m));
      const gotDepth = dot([probe.x, probe.y, -probe.z], dir);
      expect(
        gotDepth >= nearest.t - 1e-9,
        `probe at (${px}, ${py}) returned a plane ${(nearest.t - gotDepth).toExponential(2)} cm behind the visible one`
      );
      /* And the value must be the one the volume actually holds there. */
      expect(
        probe.value === data[probe.ix + probe.iy * nx + probe.iz * nx * ny],
        `probe value ${probe.value} is not what the volume holds at ${probe.ix},${probe.iy},${probe.iz}`
      );
    }
  }

  /* ── 4. labels ── */
  const texts = canvas.texts().map((t) => t.text);
  ["x [cm]", "y [cm]", "z [cm]"].forEach((t) =>
    expect(texts.includes(t), `missing axis title ${t}`)
  );
  const ticks = texts.filter((t) => !t.endsWith("[cm]"));
  const wantTicks = 3 * (3 - edgeOn);
  expect(
    ticks.length >= wantTicks,
    `only ${ticks.length} tick labels, expected at least ${wantTicks} — margins too tight?`
  );
  canvas.texts().forEach(({ text, box }) =>
    expect(
      box.x >= 0 && box.y >= 0 && box.x + box.w <= CANVAS && box.y + box.h <= CANVAS,
      `label "${text}" is drawn partly off the canvas`
    )
  );
}

const D = Math.PI / 180;
const GRID = { nx: 40, ny: 50, nz: 60, lx: 2.0, ly: 1.4, lz: 0.9 };
const g = GRID;

/* Every 30 degrees of azimuth against elevations above, level with and below
   the surface. The axis-aligned angles are included on purpose: those are
   where a plane goes exactly edge-on. */
for (let az = -180; az < 180; az += 30) {
  for (const el of [-85, -80, -45, -5, 0, 5, 30, 60, 85]) {
    runCase(g.nx, g.ny, g.nz, g.lx, g.ly, g.lz, 13, 31, 22, { az: az * D, el: el * D }, `az=${az} el=${el}`);
  }
}

/* Slices pinned to the faces, extreme aspect ratios in both directions, and an
   angle that is not a multiple of anything. */
const view3 = DEFAULT_CAMERA;
runCase(g.nx, g.ny, g.nz, g.lx, g.ly, g.lz, 0, 0, 0, view3, "slices at min faces");
runCase(g.nx, g.ny, g.nz, g.lx, g.ly, g.lz, 39, 49, 59, view3, "slices at max faces");
runCase(g.nx, g.ny, g.nz, g.lx, g.ly, g.lz, 0, 31, 59, view3, "slices on mixed faces");
runCase(10, 12, 400, 6.0, 6.0, 0.05, 5, 6, 200, view3, "thin film");
runCase(10, 12, 400, 6.0, 6.0, 0.05, 5, 6, 200, { az: -60 * D, el: 70 * D }, "thin film, steep");
runCase(400, 380, 10, 0.5, 0.5, 3.0, 200, 190, 5, view3, "deep column");
runCase(g.nx, g.ny, g.nz, g.lx, g.ly, g.lz, 13, 31, 22, { az: 12 * D, el: 41 * D }, "off-grid angle");

inCase("");
/* Coverage is asserted over the sweep, not per case: a nearly top-down view,
   or a box a couple of pixels deep, foreshortens the depth planes so far that
   insetting away from their shared edges leaves nothing comparable. But if
   most cases stopped comparing anything, this file has stopped testing the
   thing it exists for, and that must fail rather than pass quietly. */
expect(richCases >= 40, `only ${richCases} of ${cases} cases compared 6+ overlapping quads`);
expect(samples >= 10000, `only ${samples} overlapping points compared`);
expect(probes >= 2000, `only ${probes} successful probes — the hit test is barely exercised`);

finish(
  `cases=${cases} rich=${richCases} overlapping pairs=${pairs} points compared=${samples} probes=${probes} (+${probeMisses} off-slice)`
);

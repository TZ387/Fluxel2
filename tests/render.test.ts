/* ================================================================
   FLAT SLICES, COLOURBAR, PLANE CACHE
   ================================================================
   The flat layout has fixed margins and no drop-it-if-it-does-not-fit
   guard, so its margin arithmetic has to be right for every grid the
   parameter panel can produce and every canvas size the window can
   give it: tick label length follows the extents (a 0.02 cm stack
   wants three decimals, a 6 cm window wants one), panel size follows
   the canvas. Two of these were wrong when first written, both
   guaranteed rather than unlucky — the first and last tick of an axis
   sit on its ends, so a margin that can't hold half a label clips
   every time.

   Also checked: that the colourbar's numbers agree with the scale
   they came from (a bar whose labels disagree with its colours is
   worse than none), and that the plane-image cache never serves a
   stale plane, which is what makes orbiting cheap and would otherwise
   be a silent wrong-picture bug.
   ================================================================ */
import {
  drawSlices,
  drawColorbar,
  drawValidityLegend,
  makeScale,
  colormapLut,
  createPlaneCache,
  pickFlat,
  planeCanvas,
  type ScaleKind,
  type SliceScene,
  type VolumeView,
} from "../src/render";
import { createStubCanvas, expect, fail, finish, inCase, installStubDocument, type StubCanvas } from "./harness";

function volume(nx: number, ny: number, nz: number, kind: ScaleKind = "log"): VolumeView {
  const data = new Float32Array(nx * ny * nz);
  for (let i = 0; i < data.length; i++) data[i] = 1 + (i % 313);
  return {
    data,
    nx,
    ny,
    nz,
    scale: makeScale(kind, 1, 313),
    lut: colormapLut("inferno"),
    validity: null,
    showValidity: false,
    stamp: `v-${nx}-${ny}-${nz}-${kind}`,
  };
}

function labelsFit(cv: StubCanvas, W: number, H: number): void {
  cv.texts().forEach(({ text, box }) =>
    expect(
      box.x >= 0 && box.y >= 0 && box.x + box.w <= W && box.y + box.h <= H,
      `label "${text}" box (${box.x.toFixed(1)}, ${box.y.toFixed(1)}) +${box.w.toFixed(1)}x${box.h.toFixed(1)} leaves the ${W}x${H} canvas`
    )
  );
}

/* ================================================================
   1. THE FLAT LAYOUT, IN DETAIL
   ================================================================ */
inCase("flat layout");
{
  const W = 470;
  const cv = createStubCanvas(W, W);
  installStubDocument(cv);
  const view = volume(40, 50, 60);
  const scene: SliceScene = { view, lx: 2.0, ly: 1.4, lz: 0.9, ix: 13, iy: 31, iz: 22, interfaces: [0.3, 0.55] };
  drawSlices("cv", scene, createPlaneCache());

  const images = cv.images();
  expect(images.length === 3, `expected 3 slice images, got ${images.length}`);
  /* Each panel gets the image of its own plane, at voxel resolution: y-z for
     the YZ panel, x-z for XZ, x-y for XY. */
  const want = [
    [view.ny, view.nz],
    [view.nx, view.nz],
    [view.nx, view.ny],
  ];
  images.forEach((op, i) => {
    expect(
      op.sw === want[i][0] && op.sh === want[i][1],
      `panel ${i}: image ${op.sw}x${op.sh}, expected ${want[i][0]}x${want[i][1]}`
    );
    expect(op.dest !== null && op.dest.w > 20 && op.dest.h > 20, `panel ${i}: drawn too small to read`);
  });
  /* Identical on-screen size for all three, so a length reads the same in
     each — the whole reason to keep a flat view alongside the 3-D one. */
  expect(
    new Set(images.map((o) => `${o.dest?.w}x${o.dest?.h}`)).size === 1,
    "panels drawn at differing sizes"
  );

  const texts = cv.texts().map((t) => t.text);
  ["x [cm]", "y [cm]", "z [cm]"].forEach((t) => expect(texts.includes(t), `missing axis title ${t}`));
  /* Panel labels must quote a position in cm, not a voxel index. */
  ["YZ x=", "XZ y=", "XY z="].forEach((p) =>
    expect(
      texts.some((t) => t.startsWith(p) && /-?\d+\.\d{3}$/.test(t)),
      `no "${p}" label carrying a cm position`
    )
  );
  /* Every tick of a set formatted alike — decimals follow the tick spacing,
     not each value, or an axis reads "-1.00, -0.500, 0". */
  const decimals = new Set(
    texts.filter((t) => /^-?\d+\.\d+$/.test(t)).map((t) => t.split(".")[1].length)
  );
  expect(decimals.size <= 2, `tick labels mix ${decimals.size} different precisions: ${[...decimals]}`);

  /* Two interfaces, dashed, on the two panels whose vertical axis is depth. */
  const dashed = cv.strokes().filter((s) => s.dashed);
  expect(dashed.length === 4, `expected 4 dashed interface lines (2 depths x 2 panels), got ${dashed.length}`);

  labelsFit(cv, W, W);
}

/* ================================================================
   2. THE FLAT LAYOUT, SWEPT
   ================================================================ */
let sweptCases = 0,
  boxesChecked = 0;
{
  /* Extents spanning what the sliders allow: lx/ly 0.5 to 6 cm, stack depth
     from one 0.01 cm layer to eight 3 cm ones. */
  const extents: [number, number, number][] = [
    [0.5, 0.5, 0.01],
    [0.5, 0.5, 0.05],
    [0.5, 0.5, 2],
    [2, 2, 0.3],
    [2, 2, 2],
    [6, 6, 0.02],
    [6, 6, 24],
    [0.5, 6, 1],
    [6, 0.5, 1],
    [1.234, 2.345, 0.789],
  ];
  /* Canvas widths from a narrow window to an ultrawide one. Height tracks
     width, since canvas.main is aspect-ratio 1. */
  for (const [lx, ly, lz] of extents) {
    for (const S of [140, 200, 320, 470, 700, 1100]) {
      inCase(`${lx}x${ly}x${lz} @${S}px`);
      const cv = createStubCanvas(S, S);
      installStubDocument(cv);
      const view = volume(40, 50, 60);
      drawSlices("cv", { view, lx, ly, lz, ix: 20, iy: 25, iz: 30, interfaces: [lz / 3] }, createPlaneCache());
      sweptCases++;
      /* Below a readable panel size the renderer draws nothing at all, which
         is a legitimate outcome and has nothing to check. */
      if (cv.images().length === 0) continue;
      expect(cv.images().length === 3, `drew ${cv.images().length} panels`);
      boxesChecked += cv.texts().length;
      labelsFit(cv, S, S);
    }
  }
  /* Tiny and maximal grids: the image sizes change, the layout does not. */
  for (const [nx, ny, nz, label] of [
    [4, 4, 10, "tiny grid"],
    [400, 400, 400, "max grid"],
  ] as [number, number, number, string][]) {
    inCase(label);
    const cv = createStubCanvas(470, 470);
    installStubDocument(cv);
    drawSlices(
      "cv",
      { view: volume(nx, ny, nz), lx: 2, ly: 2, lz: 2, ix: nx >> 1, iy: ny >> 1, iz: nz >> 1, interfaces: [0.3] },
      createPlaneCache()
    );
    sweptCases++;
    expect(cv.images().length === 3, `drew ${cv.images().length} panels`);
    boxesChecked += cv.texts().length;
    labelsFit(cv, 470, 470);
  }
}

/* ================================================================
   3. COLOURBAR AND OVERLAY LEGEND
   ================================================================ */
for (const kind of ["log", "linear"] as ScaleKind[]) {
  inCase(`${kind} colourbar`);
  const cv = createStubCanvas(92, 200);
  installStubDocument(cv);
  const scale = makeScale(kind, 1, 313);
  drawColorbar("cbar", scale, colormapLut("inferno"));

  const ticks = cv.texts().map((t) => ({ v: parseFloat(t.text.replace("e", "E")), y: t.box.y, text: t.text }));
  expect(ticks.length >= 3, `only ${ticks.length} ticks`);
  expect(
    ticks.every((t) => Number.isFinite(t.v)),
    `unparseable tick label among ${ticks.map((t) => t.text).join(", ")}`
  );
  /* The bar puts the maximum on top, so values must fall down the canvas. */
  const byY = [...ticks].sort((a, b) => a.y - b.y);
  for (let i = 1; i < byY.length; i++) {
    expect(byY[i].v <= byY[i - 1].v, `tick ${byY[i].text} sits below the larger ${byY[i - 1].text}`);
  }
  /* And they must describe the data rather than invent range outside it. */
  const vs = ticks.map((t) => t.v);
  expect(
    Math.min(...vs) >= 0.5 && Math.max(...vs) <= 1100,
    `ticks ${Math.min(...vs)}..${Math.max(...vs)} stray outside the data's 1..313`
  );
  labelsFit(cv, 92, 200);
}

inCase("overlay legend");
{
  const cv = createStubCanvas(92, 200);
  installStubDocument(cv);
  drawValidityLegend("cbar", ["bad", "middling", "good"]);
  const order = [...cv.texts()].sort((a, b) => a.box.y - b.box.y).map((t) => t.text);
  /* Best code on top, matching drawColorbar's convention that the good end is
     the top end. */
  expect(order.join(",") === "good,middling,bad", `legend reads top-to-bottom as ${order.join(",")}`);
  labelsFit(cv, 92, 200);
}

/* ================================================================
   4. PLANE-IMAGE CACHE
   ================================================================
   Reusing an image when nothing about it changed is what keeps an orbit
   from recolouring half a million voxels a frame. Serving one when
   something *did* change would draw the wrong picture silently, so both
   directions are checked.
   ================================================================ */
inCase("plane cache");
{
  const cv = createStubCanvas(470, 470);
  const { scratch } = installStubDocument(cv);
  const cache = createPlaneCache();
  const view = volume(40, 50, 60);

  const first = planeCanvas(cache, view, 0, 5);
  const built = () => scratch.reduce((n, c) => n + c.pixels().length, 0);
  expect(built() === 1, `first call built ${built()} images, expected 1`);

  expect(planeCanvas(cache, view, 0, 5) === first, "cache handed back a different canvas for the same plane");
  expect(built() === 1, "cache rebuilt a plane that had not changed");

  planeCanvas(cache, view, 0, 6); // a different slice
  expect(built() === 2, "cache served a stale image after the slice moved");

  planeCanvas(cache, { ...view, stamp: "recoloured" }, 0, 6); // same slice, new colouring
  expect(built() === 3, "cache served a stale image after the colouring changed");

  /* A different axis has its own slot, so the two do not evict each other. */
  planeCanvas(cache, view, 1, 6);
  planeCanvas(cache, view, 0, 6);
  expect(built() === 5, `axes are evicting each other: ${built()} builds, expected 5`);
}

/* ================================================================
   5. PROBING THE FLAT LAYOUT
   ================================================================
   The readout has to agree with the picture, so the panel rectangles
   the test probes inside are taken from where the renderer actually
   drew its images — not from a second copy of the layout arithmetic.
   If picking and drawing ever disagree about where a panel is, that is
   exactly what this catches.
   ================================================================ */
inCase("flat probing");
let probes = 0;
{
  const W = 470;
  const cv = createStubCanvas(W, W);
  installStubDocument(cv);
  const view = volume(40, 50, 60);
  const lx = 2.0,
    ly = 1.4,
    lz = 0.9;
  const scene: SliceScene = { view, lx, ly, lz, ix: 13, iy: 31, iz: 22, interfaces: [0.3] };
  drawSlices("cv", scene, createPlaneCache());
  const rects = cv.images().map((o) => o.dest!);

  /* Panel order is YZ, XZ, XY: which world axis each holds fixed, which runs
     across it, and which runs down it. */
  const panels = [
    { fixed: 0, index: 13, h: 1, v: 2 },
    { fixed: 1, index: 31, h: 0, v: 2 },
    { fixed: 2, index: 22, h: 0, v: 1 },
  ];
  const dims = [view.nx, view.ny, view.nz];
  const indexOf = (p: { ix: number; iy: number; iz: number }, axis: number) =>
    axis === 0 ? p.ix : axis === 1 ? p.iy : p.iz;

  panels.forEach((panel, i) => {
    const r = rects[i];
    let prevH = -Infinity,
      prevV = -Infinity;
    for (let a = 1; a < 10; a++) {
      for (let b = 1; b < 10; b++) {
        const px = r.x + (r.w * a) / 10;
        const py = r.y + (r.h * b) / 10;
        const probe = pickFlat(scene, W, W, px, py);
        if (!probe) {
          fail(`panel ${i}: no probe at (${px.toFixed(1)}, ${py.toFixed(1)}), inside the drawn image`);
          continue;
        }
        probes++;
        /* The panel holds one axis fixed at its own slice. */
        expect(
          indexOf(probe, panel.fixed) === panel.index,
          `panel ${i}: fixed axis reads ${indexOf(probe, panel.fixed)}, expected ${panel.index}`
        );
        /* Indices in range, and the value the one actually stored there. */
        const ix = probe.ix,
          iy = probe.iy,
          iz = probe.iz;
        expect(
          ix >= 0 && ix < dims[0] && iy >= 0 && iy < dims[1] && iz >= 0 && iz < dims[2],
          `panel ${i}: indices ${ix},${iy},${iz} out of a ${dims.join("x")} grid`
        );
        expect(
          probe.value === view.data[ix + iy * view.nx + iz * view.nx * view.ny],
          `panel ${i}: value ${probe.value} is not what the volume holds at ${ix},${iy},${iz}`
        );
        /* Screen right and screen down must move the world position one way
           only — a sign flip here would read the plot mirrored. */
        const hv = [probe.x, probe.y, probe.z][panel.h];
        const vv = [probe.x, probe.y, probe.z][panel.v];
        if (b === 1) {
          expect(hv > prevH, `panel ${i}: moving right did not increase its horizontal axis`);
          prevH = hv;
        }
        if (a === 1) {
          expect(vv > prevV, `panel ${i}: moving down did not increase its vertical axis`);
          prevV = vv;
        }
      }
    }
    /* The image's own corners map to the grid's first and last voxel. */
    const first = pickFlat(scene, W, W, r.x + 0.5, r.y + 0.5);
    const last = pickFlat(scene, W, W, r.x + r.w - 0.5, r.y + r.h - 0.5);
    expect(first !== null && last !== null, `panel ${i}: corners of the drawn image do not probe`);
    if (first && last) {
      expect(
        indexOf(first, panel.h) === 0 && indexOf(first, panel.v) === 0,
        `panel ${i}: top-left corner probes ${indexOf(first, panel.h)},${indexOf(first, panel.v)} not 0,0`
      );
      expect(
        indexOf(last, panel.h) === dims[panel.h] - 1 && indexOf(last, panel.v) === dims[panel.v] - 1,
        `panel ${i}: bottom-right corner probes the wrong last voxel`
      );
    }
  });

  /* The gutters, the margins and the empty fourth quadrant are not the plot. */
  const outside: [number, number][] = [
    [2, 2],
    [W - 2, W - 2],
    [rects[0].x - 4, rects[0].y + 10], // left margin
    [rects[0].x + rects[0].w + 4, rects[0].y + 10], // gutter between the top panels
    [rects[2].x + rects[2].w + 30, rects[2].y + 30], // the empty fourth quadrant
  ];
  outside.forEach(([px, py]) =>
    expect(pickFlat(scene, W, W, px, py) === null, `(${px.toFixed(0)}, ${py.toFixed(0)}) probed, but is not on a panel`)
  );

  /* A canvas too small to draw on is also too small to probe. */
  expect(pickFlat(scene, 100, 100, 50, 50) === null, "probed a canvas too small to have panels");
}

inCase("");
finish(`flat cases=${sweptCases} label boxes checked=${boxesChecked} probes=${probes}`);

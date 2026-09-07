/* ================================================================
   TEST HARNESS
   ================================================================
   The renderers under test draw; they return nothing. So the way to
   test them is to hand them a canvas context that records what it was
   asked to do, run them, and check the recording — which is enough to
   answer the questions that actually matter here: did each slice plane
   land where the geometry says it should, was the occlusion order
   right, does every label fit on the canvas.

   One recorder, shared by both test files. Not three near-copies:
   these stubs are easy to get subtly wrong, and a wrong one invents
   failures. The first version of this had `save`/`restore` as no-ops,
   so a dash set inside a save/restore pair leaked and every later
   stroke counted as dashed — a "bug" in the renderer that was a bug in
   the stub. Anything the renderers rely on is modelled properly here:
   the full transform matrix (so translate/rotate compose the way canvas
   composes them), the save/restore stack, dash state, and text metrics.
   ================================================================ */

/** Canvas's 2x3 transform, in its own [a, b, c, d, e, f] order: maps
    (x, y) to (a·x + c·y + e, b·x + d·y + f). */
export type Matrix = readonly [number, number, number, number, number, number];

export interface Pt {
  x: number;
  y: number;
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

const IDENTITY: Matrix = [1, 0, 0, 1, 0, 0];

export function apply(m: Matrix, x: number, y: number): Pt {
  return { x: m[0] * x + m[2] * y + m[4], y: m[1] * x + m[3] * y + m[5] };
}

/** An image drawn: which image, under what transform, clipped to what. */
export interface ImageOp {
  kind: "image";
  /** Source dimensions — what identifies *which* slice plane this is. */
  sw: number;
  sh: number;
  /** Destination rect, for the five-argument form; null for the three-argument
      one, where the transform alone places the image. */
  dest: Box | null;
  matrix: Matrix;
  /** The clip path in force, in device pixels. */
  clip: Pt[] | null;
}

/** Text drawn, with the box it actually occupies — the anchor alone does not
    say, since it is an edge of the text for every alignment but "center", and
    the vertical axis titles are drawn rotated. */
export interface TextOp {
  kind: "text";
  text: string;
  box: Box;
  font: string;
}

export interface StrokeOp {
  kind: "stroke";
  path: Pt[];
  dashed: boolean;
}

/** One putImageData — i.e. a slice plane's pixels being (re)built. */
export interface PixelsOp {
  kind: "pixels";
  w: number;
  h: number;
}

export type Op = ImageOp | TextOp | StrokeOp | PixelsOp;

/** Advance width of one character, as a fraction of the font size. The
    renderers draw their axes in monospace, where every glyph is this wide;
    0.6 em is what the common monospace faces use (DejaVu Sans Mono is 0.602).
    It matters because the renderers measure their own labels to decide whether
    one fits — too small a value here and a test would pass a layout that
    clips in the app. */
const CHAR_EM = 0.6;

function fontSize(font: string): number {
  const m = /(\d+(?:\.\d+)?)px/.exec(font);
  return m ? parseFloat(m[1]) : 11;
}

export interface StubCanvas {
  /** Pass this where the code expects an HTMLCanvasElement. */
  el: any;
  ops: Op[];
  images(): ImageOp[];
  texts(): TextOp[];
  strokes(): StrokeOp[];
  pixels(): PixelsOp[];
  reset(): void;
}

export function createStubCanvas(w: number, h: number): StubCanvas {
  const ops: Op[] = [];
  let matrix: Matrix = IDENTITY;
  let dash: number[] = [];
  let path: Pt[] = [];
  let clip: Pt[] | null = null;
  const stack: { matrix: Matrix; dash: number[]; clip: Pt[] | null }[] = [];

  const ctx: any = {
    /* Set by the renderers; read back when recording text. */
    font: "11px monospace",
    textAlign: "start",
    textBaseline: "alphabetic",

    setTransform(a: number, b: number, c: number, d: number, e: number, f: number) {
      matrix = [a, b, c, d, e, f];
    },
    /* Canvas post-multiplies for translate/rotate: CTM' = CTM x T. */
    translate(tx: number, ty: number) {
      const [a, b, c, d, e, f] = matrix;
      matrix = [a, b, c, d, a * tx + c * ty + e, b * tx + d * ty + f];
    },
    rotate(ang: number) {
      const [a, b, c, d, e, f] = matrix;
      const cs = Math.cos(ang),
        sn = Math.sin(ang);
      matrix = [a * cs + c * sn, b * cs + d * sn, -a * sn + c * cs, -b * sn + d * cs, e, f];
    },
    save() {
      stack.push({ matrix, dash: [...dash], clip: clip && [...clip] });
    },
    restore() {
      const s = stack.pop();
      if (s) {
        matrix = s.matrix;
        dash = s.dash;
        clip = s.clip;
      }
    },

    beginPath() {
      path = [];
    },
    moveTo(x: number, y: number) {
      path.push(apply(matrix, x, y));
    },
    lineTo(x: number, y: number) {
      path.push(apply(matrix, x, y));
    },
    closePath() {},
    clip() {
      clip = [...path];
    },
    stroke() {
      if (path.length > 1) ops.push({ kind: "stroke", path: [...path], dashed: dash.length > 0 });
    },
    fill() {},
    fillRect() {},
    strokeRect() {},
    clearRect() {},
    setLineDash(d: number[]) {
      dash = [...d];
    },

    measureText(text: string) {
      return { width: text.length * CHAR_EM * fontSize(ctx.font) };
    },

    drawImage(img: any, dx: number, dy: number, dw?: number, dh?: number) {
      ops.push({
        kind: "image",
        sw: img.width,
        sh: img.height,
        dest: dw === undefined || dh === undefined ? null : { x: dx, y: dy, w: dw, h: dh },
        matrix,
        clip: clip && [...clip],
      });
    },

    fillText(text: string, x: number, y: number) {
      const size = fontSize(ctx.font);
      const width = text.length * CHAR_EM * size;
      /* Local box, before the transform: the anchor's meaning depends on the
         alignment, and "alphabetic" (the canvas default) sits about 0.8 em
         above the baseline. */
      const lx =
        ctx.textAlign === "left" || ctx.textAlign === "start"
          ? x
          : ctx.textAlign === "right" || ctx.textAlign === "end"
            ? x - width
            : x - width / 2;
      const ly =
        ctx.textBaseline === "top"
          ? y
          : ctx.textBaseline === "middle"
            ? y - size / 2
            : ctx.textBaseline === "bottom"
              ? y - size
              : y - size * 0.8;
      /* Transformed corners, then their axis-aligned bounds — which is what a
         rotated label really occupies. */
      const corners = [
        apply(matrix, lx, ly),
        apply(matrix, lx + width, ly),
        apply(matrix, lx + width, ly + size),
        apply(matrix, lx, ly + size),
      ];
      const xs = corners.map((p) => p.x);
      const ys = corners.map((p) => p.y);
      const x0 = Math.min(...xs),
        y0 = Math.min(...ys);
      ops.push({
        kind: "text",
        text,
        box: { x: x0, y: y0, w: Math.max(...xs) - x0, h: Math.max(...ys) - y0 },
        font: ctx.font,
      });
    },
    strokeText() {},

    createImageData(cw: number, ch: number) {
      return { data: new Uint8ClampedArray(cw * ch * 4), width: cw, height: ch };
    },
    putImageData(img: any) {
      ops.push({ kind: "pixels", w: img.width, h: img.height });
    },
  };

  const el: any = { width: w, height: h, offsetWidth: w, offsetHeight: h, style: {}, getContext: () => ctx };
  return {
    el,
    ops,
    images: () => ops.filter((o): o is ImageOp => o.kind === "image"),
    texts: () => ops.filter((o): o is TextOp => o.kind === "text"),
    strokes: () => ops.filter((o): o is StrokeOp => o.kind === "stroke"),
    pixels: () => ops.filter((o): o is PixelsOp => o.kind === "pixels"),
    reset() {
      ops.length = 0;
      matrix = IDENTITY;
      dash = [];
      clip = null;
      stack.length = 0;
    },
  };
}

/** Point every getElementById at `main`, and hand out recording canvases to
    createElement — which is what render.ts's plane cache asks for, so the
    scratch list is how a test sees a slice plane being rebuilt. */
export function installStubDocument(main: StubCanvas): { scratch: StubCanvas[] } {
  const scratch: StubCanvas[] = [];
  (globalThis as any).document = {
    getElementById: () => main.el,
    createElement: () => {
      const c = createStubCanvas(1, 1);
      scratch.push(c);
      return c.el;
    },
  };
  return { scratch };
}

/* ================================================================
   ASSERTIONS
   ================================================================
   No framework: a counter, a message, and a non-zero exit. The tests
   here are a handful of files run by tests/run.mjs, and a dependency
   would buy nothing they need.
   ================================================================ */
/* Declared rather than pulled in from @types/node: this is the only Node
   global the tests touch, and a types package for one symbol is not worth a
   dependency. Typed as returning `never` so `finish` below can be too. */
declare const process: { exit(code: number): never };

let failures = 0;
let context = "";

/** Prefix every later failure with `label`, so a sweep says which case. */
export function inCase(label: string): void {
  context = label;
}

export function fail(message: string): void {
  console.log(`FAIL ${context ? `[${context}] ` : ""}${message}`);
  failures++;
}

export function expect(condition: boolean, message: string): void {
  if (!condition) fail(message);
}

/** Report and exit. Called at the end of every test file. */
export function finish(summary: string): never {
  console.log(summary);
  console.log(failures === 0 ? "ALL CHECKS PASSED" : `${failures} FAILURE(S)`);
  process.exit(failures === 0 ? 0 : 1);
}

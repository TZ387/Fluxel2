/* ================================================================
   SETTINGS FILE FORMAT
   ================================================================
   What a saved file has to guarantee is that opening it puts the
   panel back where it was, and that a file which isn't quite right
   — hand-edited, or written by a different build — still opens with
   its parameters intact rather than being rejected wholesale. Both
   of those live in settings.ts's parseSettings, which is pure: it
   takes text and a model schema and returns values. So this file
   needs no DOM stub, unlike the renderer tests next to it.

   The round trip below builds its input the way ui-params.ts's
   getParams() does (plain groups flat, repeating groups as an array
   of named instances) rather than driving the real panel, since a
   panel needs a document. The seam that isn't covered here is
   therefore ui-params.ts's own reading and writing of controls.
   ================================================================ */

import { MODELS, type ModelDef } from "../src/models";
import { COLORMAPS } from "../src/render";
import { EL_LIMIT } from "../src/render3d";
import { SETTINGS_FORMAT, parseSettings, serializeSettings, type ViewSettings } from "../src/settings";
import { expect, fail, finish, inCase } from "./harness";

/* Plausible layer names, cycled — nothing depends on the words, only on
   their surviving the trip. */
const NAMES = ["Epidermis", "Dermis", "Subcutis", "Muscle", "Bone", "Fat", "Gel", "Substrate"];

/** A params object in the shape getParams() produces for `model`, every
    control on its default and every repeating instance named. */
function defaultParams(model: ModelDef): Record<string, any> {
  const r: Record<string, any> = {};
  model.paramGroups.forEach((group) => {
    if (group.repeat) {
      r[group.id] = Array.from({ length: group.repeat.def }, (_, i) => {
        const inst: Record<string, any> = { name: NAMES[i % NAMES.length] };
        group.params.forEach((p) => {
          inst[p.id] = group.repeat!.defs?.[i]?.[p.id] ?? p.def;
        });
        return inst;
      });
    } else {
      group.params.forEach((p) => {
        r[p.id] = p.def;
      });
    }
  });
  return r;
}

const VIEW: ViewSettings = { mode: "flat", scale: "linear", cmap: COLORMAPS[1].id, camera: { az: 0.75, el: -0.5 } };

/** Structural comparison, key order ignored — the file is JSON, and nothing
    about it should depend on the order two objects happen to list fields in.
    Returns a path to the first difference, or "" when they match. */
function diff(a: any, b: any, path = ""): string {
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) return `${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
    if (a.length !== b.length) return `${path}: length ${a.length} vs ${b.length}`;
    for (let i = 0; i < a.length; i++) {
      const d = diff(a[i], b[i], `${path}[${i}]`);
      if (d) return d;
    }
    return "";
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) {
      const d = diff(a[k], b[k], path ? `${path}.${k}` : k);
      if (d) return d;
    }
    return "";
  }
  return a === b ? "" : `${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
}

/* ── 1. round trip, every model ───────────────────────────────── */
Object.entries(MODELS).forEach(([id, model]) => {
  inCase(`round trip ${id}`);
  const params = defaultParams(model);
  const loaded = parseSettings(serializeSettings(id, params, VIEW));

  expect(loaded.model === id, `model came back as ${loaded.model}`);
  const d = diff(params, loaded.params);
  expect(d === "", `params changed in the round trip — ${d}`);
  expect(loaded.warnings.length === 0, `clean file warned: ${loaded.warnings.join("; ")}`);
  const vd = diff(VIEW, loaded.view);
  expect(vd === "", `view changed in the round trip — ${vd}`);
});

/* The serialised form is what a person may open in an editor or commit next
   to their notes, so it is indented, newline-terminated, and self-describing
   enough to tell what wrote it. */
inCase("serialised form");
{
  const text = serializeSettings("fpw1992", defaultParams(MODELS.fpw1992), VIEW);
  expect(text.endsWith("\n"), "no trailing newline");
  expect(text.includes("\n  "), "not indented");
  const raw = JSON.parse(text);
  expect(raw.format === SETTINGS_FORMAT, `format field is ${raw.format}`);
  expect(typeof raw.app === "string" && raw.app.length > 0, "no app marker");
  expect(raw.model === "fpw1992", "model field missing from the file");
}

/* ── 2. files that can't be read at all ───────────────────────── */
function expectThrows(label: string, text: string, expectIn?: string): void {
  inCase(label);
  try {
    parseSettings(text);
    fail("parsed a file that should have been rejected");
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    if (expectIn && !message.includes(expectIn)) fail(`message "${message}" doesn't mention ${expectIn}`);
  }
}

expectThrows("not JSON", "{ not json", "JSON");
expectThrows("not an object", "[1, 2, 3]", "object");
expectThrows("some other JSON file", JSON.stringify({ hello: "world" }), "model");
expectThrows(
  "unknown model",
  JSON.stringify({ format: 1, model: "diffusionTheoryOfEverything", params: {} }),
  "unknown model"
);
expectThrows(
  "newer format",
  JSON.stringify({ format: SETTINGS_FORMAT + 1, model: "fpw1992", params: {} }),
  "newer version"
);

/* ── 3. files that are read, with substitutions ───────────────── */
/** Load `params` as an fpw1992 file (one plain group pair, no layers). */
function loadFpw(params: Record<string, any>) {
  return parseSettings(JSON.stringify({ app: "fluxel2-settings", format: SETTINGS_FORMAT, model: "fpw1992", params }));
}

const FPW = defaultParams(MODELS.fpw1992);

inCase("missing parameter");
{
  const { mua, ...rest } = FPW;
  const loaded = loadFpw(rest);
  expect(loaded.params.mua === 0.1, `mua came back as ${loaded.params.mua}, not its default`);
  expect(
    loaded.warnings.some((w) => w.includes("mua") && w.includes("missing")),
    `no warning naming the missing parameter: ${loaded.warnings.join("; ")}`
  );
}

inCase("unusable parameter");
{
  const loaded = loadFpw({ ...FPW, mus: "quite a lot" });
  expect(loaded.params.mus === 100, `mus came back as ${loaded.params.mus}, not its default`);
  expect(loaded.warnings.some((w) => w.includes("mus")), "no warning naming it");
}

/* A hand-written file may well quote its numbers; reading "0.25" as 0.25 is
   friendlier than replacing it with a default. */
inCase("quoted number");
{
  const loaded = loadFpw({ ...FPW, mua: "0.25" });
  expect(loaded.params.mua === 0.25, `mua came back as ${loaded.params.mua}`);
  expect(loaded.warnings.length === 0, `quoted number warned: ${loaded.warnings.join("; ")}`);
}

/* The panel widens a slider to fit whatever it is given (ui-params.ts), the
   same as typing the value in would — so a value past the schema's range is
   kept, not clamped. Clamping here would silently change someone's run. */
inCase("value outside the slider range");
{
  const loaded = loadFpw({ ...FPW, mus: 900, mua: 1e-4 });
  expect(loaded.params.mus === 900, `mus clamped to ${loaded.params.mus}`);
  expect(loaded.params.mua === 1e-4, `mua clamped to ${loaded.params.mua}`);
  expect(loaded.warnings.length === 0, `in-range-only warning: ${loaded.warnings.join("; ")}`);
}

inCase("select parameter off its options");
{
  const loaded = loadFpw({ ...FPW, beam_profile: "sombrero" });
  expect(loaded.params.beam_profile === "pencil", `beam_profile came back as ${loaded.params.beam_profile}`);
  expect(loaded.warnings.some((w) => w.includes("beam_profile")), "no warning naming it");
}

/* A file from a later build that added a parameter still has to open here. */
inCase("unknown keys");
{
  const loaded = loadFpw({ ...FPW, wavelength_nm: 1064, layers: [{ mua: 1 }] });
  expect(!("wavelength_nm" in loaded.params), "an unknown key was carried into the params");
  expect(!("layers" in loaded.params), "a group this model doesn't have was carried in");
  expect(loaded.warnings.length === 0, `unknown keys warned: ${loaded.warnings.join("; ")}`);
}

/* ── 4. layer stacks ─────────────────────────────────────────── */
function loadLayers(layers: any) {
  const params = { ...defaultParams(MODELS.liemertKienle), layers };
  return parseSettings(JSON.stringify({ format: SETTINGS_FORMAT, model: "liemertKienle", params }));
}

const LAYER = { name: "Dermis", mua: 0.2, mus: 120, g: 0.85, n: 1.37, thickness: 0.4 };
const SPEC = MODELS.liemertKienle.paramGroups[0].repeat!;

inCase("layer stack longer than the model allows");
{
  const loaded = loadLayers(Array.from({ length: SPEC.max + 3 }, () => ({ ...LAYER })));
  expect(loaded.params.layers.length === SPEC.max, `${loaded.params.layers.length} layers survived, not ${SPEC.max}`);
  expect(loaded.warnings.some((w) => w.includes("layers")), "clamping the stack said nothing");
}

inCase("empty layer stack");
{
  const loaded = loadLayers([]);
  expect(loaded.params.layers.length === SPEC.def, `${loaded.params.layers.length} layers, not the default ${SPEC.def}`);
  expect(loaded.warnings.some((w) => w.includes("layers")), "an empty stack said nothing");
}

inCase("no layer stack at all");
{
  const loaded = loadLayers(undefined);
  expect(loaded.params.layers.length === SPEC.def, `${loaded.params.layers.length} layers, not the default ${SPEC.def}`);
  expect(loaded.warnings.some((w) => w.includes("layers")), "a missing stack said nothing");
}

inCase("layer values and names");
{
  const loaded = loadLayers([{ ...LAYER }, { ...LAYER, name: "  Subcutis  " }, { mua: 0.3 }, { name: "" }, 42]);
  const [a, b, c, d, e] = loaded.params.layers;
  expect(a.name === "Dermis" && a.mus === 120, `first layer came back as ${JSON.stringify(a)}`);
  expect(b.name === "Subcutis", `a padded name came back as ${JSON.stringify(b.name)}`);
  /* No name is left absent rather than invented here: what an unnamed layer
     is called depends on its position, which is ui-params.ts's business. */
  expect(!("name" in c), `a nameless layer was given the name ${JSON.stringify(c.name)}`);
  expect(c.mua === 0.3 && c.mus === 100, `partial layer not filled from defaults: ${JSON.stringify(c)}`);
  expect(!("name" in d), "a blank name was kept");
  expect(e && e.mua === 0.1, `a layer that isn't an object came back as ${JSON.stringify(e)}`);
  expect(loaded.warnings.some((w) => w.includes("layers[4]")), "a non-object layer said nothing");
  expect(loaded.warnings.some((w) => w.includes("layers[2].mus")), "a missing layer parameter said nothing");
}

/* ── 5. the view block ───────────────────────────────────────── */
inCase("view block absent");
{
  const loaded = loadFpw(FPW);
  expect(loaded.view === null, "a file with no view block came back with one");
  expect(loaded.warnings.length === 0, `its absence warned: ${loaded.warnings.join("; ")}`);
}

inCase("view block stale");
{
  const loaded = parseSettings(
    JSON.stringify({
      format: SETTINGS_FORMAT,
      model: "fpw1992",
      params: FPW,
      view: { mode: "hologram", scale: "log", cmap: "chartreuse", camera: { az: 1, el: 3 } },
    })
  );
  expect(loaded.view !== null, "a stale view block was dropped entirely");
  expect(loaded.view!.mode === "box3d", `mode came back as ${loaded.view!.mode}`);
  expect(loaded.view!.scale === "log", `a good field was lost alongside the bad ones`);
  expect(loaded.view!.cmap === COLORMAPS[0].id, `cmap came back as ${loaded.view!.cmap}`);
  /* Elevation is capped the way dragging caps it, so a file can't ask for an
     angle the box can't be drawn at. */
  expect(loaded.view!.camera.el === EL_LIMIT, `el came back as ${loaded.view!.camera.el}, not the limit`);
  expect(loaded.view!.camera.az === 1, `az came back as ${loaded.view!.camera.az}`);
  expect(loaded.warnings.some((w) => w.includes("view.mode")), "a bad layout said nothing");
  expect(loaded.warnings.some((w) => w.includes("view.cmap")), "a bad colormap said nothing");
  /* The parameters are the point of the file; a stale view can't cost them. */
  expect(diff(FPW, loaded.params) === "", "a stale view block disturbed the parameters");
}

inCase("view camera unusable");
{
  const loaded = parseSettings(
    JSON.stringify({ format: SETTINGS_FORMAT, model: "fpw1992", params: FPW, view: { mode: "flat", camera: "yes" } })
  );
  expect(loaded.view!.camera.az === 0 && loaded.view!.camera.el === 0, "no fallback camera angles");
  expect(loaded.warnings.some((w) => w.includes("camera")), "an unusable camera said nothing");
}

finish(`settings: ${Object.keys(MODELS).length} models round-tripped, plus malformed-file handling`);

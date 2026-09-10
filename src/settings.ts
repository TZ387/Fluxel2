/* ================================================================
   SETTINGS FILES (SAVE / LOAD)
   ================================================================
   One run's inputs, as JSON on disk. Deliberately not a new format:
   `params` is verbatim the object ui-params.ts's getParams() produces
   and compute.ts hands to Rust, so a saved file is literally what was
   computed, with no second serialisation to keep in step with the
   four models' schemas. Costs one field the physics doesn't read
   (each layer's `name`, which serde drops), buys a file that's
   readable, diffable, and editable by hand.

   The I/O itself is elsewhere: main.ts owns the file dialogs and the
   two Tauri commands that read/write the text (lib.rs). This module
   is pure — a shape, a serialiser, and a parser that checks a file
   against the selected model's schema — which is what makes the part
   with the actual decisions testable (tests/settings.test.ts) without
   a DOM.

   Loading rules, all chosen so a hand-edited or older file still opens:
     - the model must be one this build knows; anything else is an
       error, since its parameters would mean nothing here.
     - a missing or unusable parameter falls back to the schema's
       default, with a warning naming it.
     - a value outside its slider's range is *kept* — the panel
       widens the slider to fit it, the same as typing it in would.
     - the layer count is clamped to what the model allows.
     - unknown keys are ignored, so a file from a later version that
       added a parameter still loads here.
     - a dropdown value that has since been renamed is followed to
       whichever option now claims it (models.ts's `aliases`), so a
       rename doesn't silently reset the control to its default.
     - `view` is optional: a file without it loads, and a file with a
       stale colormap or layout keeps the rest of it.
   ================================================================ */

import { MODELS, type ParamDef, type ParamGroup } from "./models";
import { COLORMAPS, type ScaleKind } from "./render";
import { EL_LIMIT, type Camera } from "./render3d";

/** Bumped only for a change that older files can't be read through — the
    rules above absorb an added or removed parameter without one. */
export const SETTINGS_FORMAT = 1;

/** Marker so a file picked by mistake fails with something better than a
    parameter-by-parameter list of complaints. */
const SETTINGS_APP = "fluxel2-settings";

/** Which of the two layouts the plots are drawn in — main.ts reads it off
    the View dropdown; it lives here because the file records it. */
export type ViewMode = "box3d" | "flat";

/** The view controls that are shared by both plots, and so describe the
    picture as a whole rather than one panel. The per-panel slice planes are
    left out on purpose: they are voxel indices into whatever grid the run
    happened to use, and a file is not tied to a grid. */
export interface ViewSettings {
  mode: ViewMode;
  scale: ScaleKind;
  cmap: string;
  camera: Camera;
}

export interface SettingsFile {
  app: string;
  format: number;
  /** Key into MODELS — which schema `params` belongs to. */
  model: string;
  params: Record<string, any>;
  view: ViewSettings;
}

export interface LoadedSettings {
  model: string;
  /** Checked against that model's schema, ready to seed the panel with
      (ui-params.ts's buildModelParams). */
  params: Record<string, any>;
  /** Null when the file didn't carry a usable view block. */
  view: ViewSettings | null;
  /** What had to be substituted or clamped, in plain text, for the status
      line. Empty for a file that loaded exactly as written. */
  warnings: string[];
}

export function serializeSettings(model: string, params: Record<string, any>, view: ViewSettings): string {
  const file: SettingsFile = { app: SETTINGS_APP, format: SETTINGS_FORMAT, model, params, view };
  /* Indented, and newline-terminated: this is a file a person may well open
     in an editor, and one they might keep in a git repository next to their
     notes. */
  return JSON.stringify(file, null, 2) + "\n";
}

/* ── checking a file against a model's schema ─────────────────── */

/** Every param in `params`, taken from `raw` where it is usable and from the
    schema's default where it isn't. `where` prefixes any warning with the
    param's place in the file ("layers[1]."), so a complaint points at a line
    rather than just a name. */
function normalizeGrid(
  params: ParamDef[],
  raw: Record<string, any>,
  where: string,
  warn: (message: string) => void
): Record<string, any> {
  const out: Record<string, any> = {};
  params.forEach((p) => {
    const given = raw[p.id];
    if (p.kind === "select") {
      const match = typeof given === "string"
        ? p.options.find((o) => o.value === given || o.aliases?.includes(given))
        : undefined;
      if (match) {
        out[p.id] = match.value;
        /* Silent when the file already used the current name; worth a line
           when it didn't, since the option it lands on is not the one the
           file names. */
        if (match.value !== given) warn(`${where}${p.id}: "${String(given)}" is now called "${match.value}" — using that`);
        return;
      }
      warn(`${where}${p.id}: ${given === undefined ? "missing" : `"${String(given)}" is not one of its choices`} — using "${p.def}"`);
      out[p.id] = p.def;
      return;
    }
    /* Number(), not typeof — a hand-edited file may quote a number, and
       reading "0.1" as 0.1 is friendlier than replacing it with the default.
       Anything genuinely unusable (null, a word, an empty string) lands on
       NaN and is reported. Note the range is *not* enforced: the panel widens
       a slider to fit whatever it is given. */
    const v = Number(given);
    if (Number.isFinite(v)) {
      out[p.id] = v;
      return;
    }
    warn(`${where}${p.id}: ${given === undefined ? "missing" : `"${String(given)}" is not a number`} — using ${p.def}`);
    out[p.id] = p.def;
  });
  return out;
}

/** One repeating group's array — the layer stack, in every model that has
    one. Name kept as written when it is a non-empty string; ui-params.ts
    supplies the default otherwise, since what an unnamed instance is called
    depends on its position. */
function normalizeRepeatGroup(
  group: ParamGroup,
  raw: any,
  warn: (message: string) => void
): Record<string, any>[] {
  const spec = group.repeat!;
  const given: any[] = Array.isArray(raw) ? raw : [];
  if (!Array.isArray(raw)) warn(`${group.id}: missing — using ${spec.def} default ${spec.def === 1 ? "instance" : "instances"}`);

  const count = Math.max(spec.min, Math.min(spec.max, given.length || spec.def));
  if (Array.isArray(raw) && count !== given.length) {
    warn(`${group.id}: ${given.length} in the file, but this model takes ${spec.min}–${spec.max} — using ${count}`);
  }

  return Array.from({ length: count }, (_, i) => {
    const item = given[i];
    const source = item && typeof item === "object" && !Array.isArray(item) ? item : {};
    if (given.length > i && source !== item) warn(`${group.id}[${i}]: not an object — using defaults`);
    const values = normalizeGrid(group.params, source, `${group.id}[${i}].`, warn);
    const name = source.name;
    return typeof name === "string" && name.trim() ? { name: name.trim(), ...values } : values;
  });
}

/** Parse and check the text of a settings file. Throws with a message meant
    for the status line when the file isn't one, or is one this build can't
    read; anything it can work around comes back in `warnings` instead. */
export function parseSettings(text: string): LoadedSettings {
  let file: any;
  try {
    file = JSON.parse(text);
  } catch (err) {
    throw new Error(`not valid JSON (${err instanceof Error ? err.message : String(err)})`);
  }
  if (!file || typeof file !== "object" || Array.isArray(file)) throw new Error("not a settings file — expected a JSON object");

  const format = Number(file.format);
  if (!Number.isFinite(format) || typeof file.model !== "string") {
    throw new Error(`not a ${SETTINGS_APP} file — no "model" and "format" in it`);
  }
  if (format > SETTINGS_FORMAT) {
    throw new Error(`written by a newer version of this app (format ${format}, this build reads ${SETTINGS_FORMAT})`);
  }

  const model = MODELS[file.model];
  if (!model) {
    throw new Error(`unknown model "${file.model}" — this build has ${Object.keys(MODELS).join(", ")}`);
  }

  const warnings: string[] = [];
  const warn = (message: string) => warnings.push(message);
  const raw = file.params && typeof file.params === "object" && !Array.isArray(file.params) ? file.params : {};
  if (raw !== file.params) warn("params: missing — every value is the model's default");

  const params: Record<string, any> = {};
  model.paramGroups.forEach((group) => {
    if (group.repeat) params[group.id] = normalizeRepeatGroup(group, raw[group.id], warn);
    else Object.assign(params, normalizeGrid(group.params, raw, "", warn));
  });

  return { model: file.model, params, view: normalizeView(file.view, warn), warnings };
}

/** The view block, or null if there isn't one. Each field is checked on its
    own and dropped to nothing more than a warning if stale, so an old file's
    parameters aren't lost over a colormap that has since been renamed. */
function normalizeView(raw: any, warn: (message: string) => void): ViewSettings | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;

  const mode: ViewMode = raw.mode === "box3d" || raw.mode === "flat" ? raw.mode : "box3d";
  if (raw.mode !== undefined && raw.mode !== mode) warn(`view.mode: "${String(raw.mode)}" is not a layout — using "${mode}"`);

  const scale: ScaleKind = raw.scale === "log" || raw.scale === "linear" ? raw.scale : "log";
  if (raw.scale !== undefined && raw.scale !== scale) warn(`view.scale: "${String(raw.scale)}" is not a scale — using "${scale}"`);

  const known = COLORMAPS.some((c) => c.id === raw.cmap);
  const cmap = known ? (raw.cmap as string) : COLORMAPS[0].id;
  if (raw.cmap !== undefined && !known) warn(`view.cmap: "${String(raw.cmap)}" is not a colormap — using "${cmap}"`);

  /* Azimuth wraps, so any finite angle is a valid one; elevation is capped
     the same way dragging caps it (render3d.ts's EL_LIMIT), since the box's
     projection degenerates at the poles. */
  const cam = raw.camera && typeof raw.camera === "object" ? raw.camera : {};
  const az = Number(cam.az);
  const el = Number(cam.el);
  const camera: Camera = {
    az: Number.isFinite(az) ? az : 0,
    el: Number.isFinite(el) ? Math.max(-EL_LIMIT, Math.min(EL_LIMIT, el)) : 0,
  };
  if (!Number.isFinite(az) || !Number.isFinite(el)) warn("view.camera: no usable angles — looking straight on");

  return { mode, scale, cmap, camera };
}

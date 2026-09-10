/* ================================================================
   GENERIC PARAMETER PANEL BUILDER
   ================================================================
   Renders whatever `paramGroups` the selected model declares (models.ts).
   Knows nothing about any specific model's parameters — only how to turn a
   group's `params` array into rows, and how to repeat that for `repeat`
   groups (one block per tissue layer, with add/remove buttons and an
   editable name per instance, e.g. "Dermis" rather than "Layer 2").

   Each row renders: [min box] — slider — [max box] — [value box], all
   directly editable: the value box moves the slider (clamping to
   [min,max], or extending it if typed outside); a bound box re-ranges the
   slider, clamping the value if needed; the slider updates the value box.

   The controls are the state — getParams() reads them back off the DOM at
   run time rather than keeping a copy in step. The one thing that can't
   live in the DOM is a repeat group's instance list, since rebuilding it
   is what add/remove *do*; that's `repeatState` below, refreshed from the
   controls before any rebuild so an edit or rename survives one.

   buildModelParams() takes an optional seed of starting values — how a
   saved settings file is restored (settings.ts), the same path the
   schema's own defaults take, so there is no second way to write a value
   into this panel.
   ================================================================ */

import type { ModelDef, ParamDef, ParamGroup } from "./models";

/* Model whose panel is currently on screen. Reset each time
   buildModelParams() runs, i.e. on init, on model switch, and on load. */
let currentGroups: ParamGroup[] = [];

/** One rendered instance of a repeat group: its editable name, and the
    values its controls were last seeded with (refreshed from those controls
    by syncRepeatState). */
interface RepeatInstance {
  name: string;
  values: Record<string, any>;
}

let repeatState: Record<string, RepeatInstance[]> = {};

/* ── one param-grid's worth of rows ────────────────────────────
   `prefix` namespaces element ids so repeated instances (layer 0,
   layer 1, ...) don't collide, e.g. "layers0-mua", "layers1-mua".
   `seed` gives this grid's starting values, keyed by param id: the
   schema's own defaults (RepeatSpec.defs), the values carried across
   a rebuild, or a loaded settings file. Anything it doesn't mention
   falls back to the param's own `def`. */
function buildParamGrid(
  params: ParamDef[],
  container: HTMLElement,
  prefix = "",
  seed: Record<string, any> = {}
): void {
  const rows: { p: ParamDef; row: HTMLElement }[] = [];

  params.forEach((p) => {
    const uid = prefix + p.id;
    const row = document.createElement("div");
    row.className = "param-row";
    rows.push({ p, row });

    if (p.kind === "select") {
      const def = String(seed[p.id] ?? p.def);
      row.innerHTML = `
        <div class="param-label">${p.label}</div>
        <div class="param-ctrl">
          <select id="${uid}" class="p-select">
            ${p.options.map((o) => `<option value="${o.value}"${o.value === def ? " selected" : ""}>${o.label}</option>`).join("")}
          </select>
        </div>`;
      container.appendChild(row);
      return;
    }

    const def = Number(seed[p.id] ?? p.def);
    /* A value can legitimately sit outside the slider's declared range: the
       value box takes whatever is typed and re-ranges the slider rather than
       clamping (see below), and both a carried-over edit and a saved file
       can hold the result. So the rendered range opens wide enough for the
       value it is given — widening here is what keeps that value intact
       through a rebuild, where clamping would quietly discard it. */
    const lo = Number.isFinite(def) ? Math.min(p.min, def) : p.min;
    const hi = Number.isFinite(def) ? Math.max(p.max, def) : p.max;
    row.innerHTML = `
      <div class="param-label">${p.label}</div>
      <div class="param-ctrl">
        <input type="number" class="p-bound" id="${uid}-min" value="${lo}" step="${p.step}" title="Slider minimum">
        <input type="range"  id="${uid}"     min="${lo}" max="${hi}" step="${p.step}" value="${def}">
        <input type="number" class="p-bound" id="${uid}-max" value="${hi}" step="${p.step}" title="Slider maximum">
        <input type="number" class="p-val"   id="${uid}-v"   value="${p.fmt(def)}" step="${p.step}" title="Current value" readonly>
      </div>`;
    container.appendChild(row);

    const slider = row.querySelector(`#${CSS.escape(uid)}`) as HTMLInputElement;
    const minBox = row.querySelector(`#${CSS.escape(uid)}-min`) as HTMLInputElement;
    const maxBox = row.querySelector(`#${CSS.escape(uid)}-max`) as HTMLInputElement;
    const valBox = row.querySelector(`#${CSS.escape(uid)}-v`) as HTMLInputElement;

    /* slider → value box */
    slider.addEventListener("input", () => {
      valBox.value = p.fmt(+slider.value);
    });

    /* value box → slider (extend range if needed) */
    valBox.addEventListener("change", () => {
      const v = +valBox.value;
      if (!isFinite(v)) {
        valBox.value = p.fmt(+slider.value);
        return;
      }
      if (v < +minBox.value) {
        minBox.value = p.fmt(v);
        slider.min = String(v);
      }
      if (v > +maxBox.value) {
        maxBox.value = p.fmt(v);
        slider.max = String(v);
      }
      slider.value = String(v);
      valBox.value = p.fmt(v);
    });

    /* min box → slider range (clamp current value if needed) */
    minBox.addEventListener("change", () => {
      const lo = +minBox.value;
      slider.min = String(lo);
      if (+slider.value < lo) {
        slider.value = String(lo);
        valBox.value = p.fmt(lo);
      }
    });

    /* max box → slider range (clamp current value if needed) */
    maxBox.addEventListener("change", () => {
      const hi = +maxBox.value;
      slider.max = String(hi);
      if (+slider.value > hi) {
        slider.value = String(hi);
        valBox.value = p.fmt(hi);
      }
    });
  });

  /* Conditional visibility (showIf): wired after every row exists, since a
     param can declare showIf on a sibling that's built later in the array. */
  rows.forEach(({ p, row }) => {
    if (p.kind === "select" || !p.showIf) return;
    const controller = document.getElementById(prefix + p.showIf.id) as HTMLSelectElement | null;
    if (!controller) return;
    const sync = () => {
      row.hidden = !p.showIf!.oneOf.includes(controller.value);
    };
    controller.addEventListener("change", sync);
    sync();
  });
}

function readParamGrid(params: ParamDef[], prefix = ""): Record<string, any> {
  const r: Record<string, any> = {};
  params.forEach((p) => {
    if (p.kind === "select") {
      r[p.id] = (document.getElementById(`${prefix}${p.id}`) as HTMLSelectElement).value;
      return;
    }
    const vbox = document.getElementById(`${prefix}${p.id}-v`) as HTMLInputElement | null;
    const slider = document.getElementById(`${prefix}${p.id}`) as HTMLInputElement;
    r[p.id] = vbox ? +vbox.value : +slider.value;
  });
  return r;
}

/* ── repeating groups: instances, names, add/remove ─────────────
   An instance is identified by its position, which is what the id
   prefix encodes ("layers1-mua") — so removing the middle of three
   renumbers the ones below it. That means the values and names can't
   be left in the DOM across the rebuild: they are read out first
   (syncRepeatState), the list is edited, and the rebuild seeds each
   instance from it. Before this, add/remove rebuilt every instance
   from the schema's defaults, so adding a fourth layer silently
   reset the three already set up. */

/** What an instance is called until it is renamed: its position in the
    stack. An un-renamed instance is *held* as an empty name, so that the two
    below the top of three become "Layer 1" and "Layer 2" by themselves — but
    it is shown with this text in the box rather than as a placeholder, since
    placeholder text can't be selected or typed over, which makes a name look
    broken rather than editable. syncRepeatState turns it back into an empty
    name by recognising it, so a box left alone keeps renumbering. */
function instanceName(group: ParamGroup, i: number): string {
  return `${group.repeat!.itemLabel ?? group.title} ${i + 1}`;
}

/** Refresh a repeat group's list from the controls on screen, and return it
    for editing. A no-op for a group that isn't rendered yet (nothing to read),
    which is only the case between a reset and its first render. */
function syncRepeatState(group: ParamGroup): RepeatInstance[] {
  const state = repeatState[group.id] ?? [];
  state.forEach((inst, i) => {
    const nameBox = document.getElementById(`${group.id}${i}-name`) as HTMLInputElement | null;
    if (!nameBox) return;
    /* Still the position it was shown with (or empty) means it was never
       renamed. Typing that same text by hand is indistinguishable from
       leaving it, which costs nothing: it means the same thing. */
    const typed = nameBox.value.trim();
    inst.name = typed === instanceName(group, i) ? "" : typed;
    inst.values = readParamGrid(group.params, `${group.id}${i}-`);
  });
  return state;
}

/** A repeat group's starting list: one instance per element of `seed` when
    given an array (a loaded settings file — see settings.ts), otherwise
    `repeat.def` instances seeded from the schema's own `repeat.defs`. The
    instance count is the schema's business either way, so a file asking for
    more layers than the model allows is clamped rather than honoured. */
function repeatStateFor(group: ParamGroup, seed: unknown): RepeatInstance[] {
  const spec = group.repeat!;
  const raw = Array.isArray(seed) ? (seed as Record<string, any>[]) : null;
  const count = raw ? Math.max(spec.min, Math.min(spec.max, raw.length)) : spec.def;
  return Array.from({ length: count }, (_, i) => {
    const given = raw?.[i];
    if (!given || typeof given !== "object") {
      return { name: "", values: { ...(spec.defs?.[i] ?? {}) } };
    }
    /* The name travels inside the instance (see getParams below), so it is
       taken out here rather than being fed to the controls as a value. */
    const { name, ...values } = given;
    return { name: typeof name === "string" ? name.trim() : "", values };
  });
}

function renderRepeatGroup(group: ParamGroup, container: HTMLElement): void {
  const spec = group.repeat!;
  const state = repeatState[group.id];

  container.innerHTML = "";

  state.forEach((inst, i) => {
    const el = document.createElement("div");
    el.className = "repeat-instance";
    el.innerHTML = `
      <div class="repeat-instance-hdr">
        <input type="text" class="repeat-name" id="${group.id}${i}-name" maxlength="40" spellcheck="false"
               title="Name this ${(spec.itemLabel ?? group.title).toLowerCase()} — saved with the settings">
        <button type="button" class="repeat-remove-btn" ${state.length <= spec.min ? "disabled" : ""}>&times; Remove</button>
      </div>`;
    const grid = document.createElement("div");
    grid.className = "param-grid";
    el.appendChild(grid);
    buildParamGrid(group.params, grid, `${group.id}${i}-`, inst.values);
    container.appendChild(el);

    /* Assigned rather than written into the markup above: a name can come
       from a settings file, and an attribute built by string concatenation
       would let one carry markup into the header. The placeholder repeats the
       default only for the case of a box cleared by hand. */
    const nameBox = el.querySelector(".repeat-name") as HTMLInputElement;
    nameBox.placeholder = instanceName(group, i);
    nameBox.value = inst.name || instanceName(group, i);

    el.querySelector(".repeat-remove-btn")!.addEventListener("click", () => {
      const live = syncRepeatState(group);
      if (live.length <= spec.min) return;
      live.splice(i, 1);
      renderRepeatGroup(group, container);
    });
  });

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "repeat-add-btn";
  addBtn.textContent = `+ Add ${(spec.itemLabel ?? group.title).toLowerCase()}`;
  addBtn.disabled = state.length >= spec.max;
  addBtn.addEventListener("click", () => {
    const live = syncRepeatState(group);
    if (live.length >= spec.max) return;
    /* Unnamed, and with empty values: a hand-added instance starts from the
       sliders' own defaults, `repeat.defs` describing the stack a model
       *opens* with rather than what a new instance should be. */
    live.push({ name: "", values: {} });
    renderRepeatGroup(group, container);
  });
  container.appendChild(addBtn);
}

/* ── top-level: (re)build the whole param panel for a model ──────
   `seed` restores a saved set of values (settings.ts's parseSettings
   output, already checked against this model's schema): plain groups
   read their params off it by id, repeat groups take an array under
   their own group id. Omitted, every control opens on its default. */
export function buildModelParams(model: ModelDef, containerId: string, seed?: Record<string, any>): void {
  const root = document.getElementById(containerId)!;
  root.innerHTML = "";
  currentGroups = model.paramGroups;
  repeatState = {};

  model.paramGroups.forEach((group) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<div class="panel-title">${group.title}</div>`;
    root.appendChild(panel);

    if (group.repeat) {
      const wrap = document.createElement("div");
      wrap.className = "repeat-group";
      panel.appendChild(wrap);
      repeatState[group.id] = repeatStateFor(group, seed?.[group.id]);
      renderRepeatGroup(group, wrap);
    } else {
      const grid = document.createElement("div");
      grid.className = "param-grid";
      panel.appendChild(grid);
      buildParamGrid(group.params, grid, "", seed ?? {});
    }
  });
}

/* ── read every current control back into a plain params object ─
   Plain groups merge flat (p.mua, p.lx, ...). Repeat groups come
   back as an array under their own group id (p.layers = [...]). */
export function getParams(): Record<string, any> {
  const r: Record<string, any> = {};
  currentGroups.forEach((group) => {
    if (group.repeat) {
      r[group.id] = syncRepeatState(group).map((inst, i) => ({
        /* Carried inside the instance, alongside its params, so that this
           object — the very thing that goes to Rust (compute.ts) — is also
           the whole of what a settings file has to remember. Rust's params
           structs ignore it, serde dropping fields they don't declare. It
           does reserve `name` as an instance key, so a param with that id
           would be shadowed by it; none has one, and the label to rename is
           the group's `itemLabel`. */
        name: inst.name.trim() || instanceName(group, i),
        ...inst.values,
      }));
    } else {
      Object.assign(r, readParamGrid(group.params));
    }
  });
  /* Counts reach Rust as usize, so they have to be whole and at least 1. A
     value box will take 5.5, or a negative — it re-ranges the slider rather
     than clamping — and either would be rejected by the backend. Math.trunc
     rather than `| 0`, which wraps anything past 2^31 to a bogus count. */
  const count = (v: number) => (Number.isFinite(v) ? Math.max(1, Math.trunc(v)) : 1);
  ["nx", "ny", "nz", "nr", "pattern_count"].forEach((id) => {
    if (id in r) r[id] = count(r[id]);
  });
  return r;
}

/* ================================================================
   GENERIC PARAMETER PANEL BUILDER
   ================================================================
   Renders whatever `paramGroups` the selected model declares (see
   models.ts). Knows nothing about any specific model's parameters —
   only how to turn a group's `params` array into rows, and how to
   repeat that for `repeat` groups (e.g. one block per tissue layer,
   with add/remove buttons since the layer count varies).

   Each row renders: [min box] — slider — [max box] — [value box].
   All three are directly editable: the value box moves the slider
   (clamping to [min,max], or extending it if typed outside); a bound
   box re-ranges the slider, clamping the value if needed; the slider
   updates the value box.
   ================================================================ */

import type { ModelDef, ParamDef, ParamGroup } from "./models";

/* Model whose panel is currently on screen, and (for repeat groups)
   how many instances of each are currently rendered. Both are reset
   each time buildModelParams() runs, i.e. on init and on model switch. */
let currentGroups: ParamGroup[] = [];
let repeatCounts: Record<string, number> = {};

/* ── one param-grid's worth of rows ────────────────────────────
   `prefix` namespaces element ids so repeated instances (layer 0,
   layer 1, ...) don't collide, e.g. "layers0-mua", "layers1-mua".
   `defs` overrides this instance's starting values (see RepeatSpec.defs
   in models.ts). */
function buildParamGrid(
  params: ParamDef[],
  container: HTMLElement,
  prefix = "",
  defs: Record<string, number | string> = {}
): void {
  const rows: { p: ParamDef; row: HTMLElement }[] = [];

  params.forEach((p) => {
    const uid = prefix + p.id;
    const row = document.createElement("div");
    row.className = "param-row";
    rows.push({ p, row });

    if (p.kind === "select") {
      const def = String(defs[p.id] ?? p.def);
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

    const def = Number(defs[p.id] ?? p.def);
    row.innerHTML = `
      <div class="param-label">${p.label}</div>
      <div class="param-ctrl">
        <input type="number" class="p-bound" id="${uid}-min" value="${p.min}" step="${p.step}" title="Slider minimum">
        <input type="range"  id="${uid}"     min="${p.min}" max="${p.max}" step="${p.step}" value="${def}">
        <input type="number" class="p-bound" id="${uid}-max" value="${p.max}" step="${p.step}" title="Slider maximum">
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

/* ── repeating groups: N instances + add/remove-instance buttons ─ */
function renderRepeatGroup(group: ParamGroup, container: HTMLElement): void {
  const spec = group.repeat!;
  const count = repeatCounts[group.id] ?? spec.def;
  repeatCounts[group.id] = count;

  container.innerHTML = "";

  for (let i = 0; i < count; i++) {
    const inst = document.createElement("div");
    inst.className = "repeat-instance";
    inst.innerHTML = `
      <div class="repeat-instance-hdr">
        <span>${group.title} ${i + 1}</span>
        <button type="button" class="repeat-remove-btn" ${count <= spec.min ? "disabled" : ""}>&times; Remove</button>
      </div>`;
    const grid = document.createElement("div");
    grid.className = "param-grid";
    inst.appendChild(grid);
    buildParamGrid(group.params, grid, `${group.id}${i}-`, spec.defs?.[i]);
    container.appendChild(inst);

    inst.querySelector(".repeat-remove-btn")!.addEventListener("click", () => {
      if (repeatCounts[group.id] <= spec.min) return;
      repeatCounts[group.id]--;
      renderRepeatGroup(group, container);
    });
  }

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "repeat-add-btn";
  addBtn.textContent = `+ Add ${group.title.toLowerCase()}`;
  addBtn.disabled = count >= spec.max;
  addBtn.addEventListener("click", () => {
    if (repeatCounts[group.id] >= spec.max) return;
    repeatCounts[group.id]++;
    renderRepeatGroup(group, container);
  });
  container.appendChild(addBtn);
}

/* ── top-level: (re)build the whole param panel for a model ────── */
export function buildModelParams(model: ModelDef, containerId: string): void {
  const root = document.getElementById(containerId)!;
  root.innerHTML = "";
  currentGroups = model.paramGroups;
  repeatCounts = {};

  model.paramGroups.forEach((group) => {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<div class="panel-title">${group.title}</div>`;
    root.appendChild(panel);

    if (group.repeat) {
      const wrap = document.createElement("div");
      wrap.className = "repeat-group";
      panel.appendChild(wrap);
      renderRepeatGroup(group, wrap);
    } else {
      const grid = document.createElement("div");
      grid.className = "param-grid";
      panel.appendChild(grid);
      buildParamGrid(group.params, grid);
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
      const n = repeatCounts[group.id] ?? group.repeat.def;
      const instances = [];
      for (let i = 0; i < n; i++) instances.push(readParamGrid(group.params, `${group.id}${i}-`));
      r[group.id] = instances;
    } else {
      Object.assign(r, readParamGrid(group.params));
    }
  });
  /* Counts reach Rust as usize, so they have to be whole and at least 1. A
     value box will take 5.5, or a negative — it re-ranges the slider rather
     than clamping — and either would be rejected by the backend. Math.trunc
     rather than `| 0`, which wraps anything past 2^31 to a bogus count. */
  const count = (v: number) => (Number.isFinite(v) ? Math.max(1, Math.trunc(v)) : 1);
  if ("nx" in r) r.nx = count(r.nx);
  if ("ny" in r) r.ny = count(r.ny);
  if ("nz" in r) r.nz = count(r.nz);
  if ("pattern_count" in r) r.pattern_count = count(r.pattern_count);
  return r;
}

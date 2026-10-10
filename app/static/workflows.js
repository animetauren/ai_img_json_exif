"use strict";
// Workflows tab: edit, save and run chains of vision-model and LLM steps.
// Shares globals with app.js: el, $, imageDataUrl, exifData, exifPromise, visionModels,
// visionDefaults, defaults, currentModel, fillSelect, flattenExif, setStatus, drawDetections, clearOutputs,
// exportImage, runButtonLabel. batch.js uses computeWorkflow / showWorkflowRecord.

const WF_UI_KEY = "llm-classifier-workflow-ui";

const wf = {
  list: [],       // saved workflows from the server
  current: null,  // working copy being edited (may have unsaved changes)
  dirty: false,
  jsonMode: false,
  loaded: false,
};

const w = {
  select: $("wfSelect"),
  newBtn: $("wfNew"),
  duplicate: $("wfDuplicate"),
  remove: $("wfDelete"),
  save: $("wfSave"),
  name: $("wfName"),
  steps: $("wfSteps"),
  json: $("wfJson"),
  jsonToggle: $("wfJsonToggle"),
  addRow: document.querySelector(".wf-add"),
  addVision: $("wfAddVision"),
  addLlm: $("wfAddLlm"),
  dirtyLabel: $("wfDirty"),
  run: $("wfRun"),
  meta: $("wfMeta"),
  result: $("wfResult"),
  copy: $("wfCopy"),
  exportBtn: $("wfExport"),
};

// ---------- small DOM helper ----------
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false || k === "value" || k === "checked") continue;
    if (k === "class") e.className = v;
    else if (k === "dataset") Object.assign(e.dataset, v);
    else if (k in e && typeof v !== "string") e[k] = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  e.append(...kids.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false));
  // Set after children so <select> can pick one of its options
  if (attrs.value !== undefined && attrs.value !== null) e.value = attrs.value;
  if (attrs.checked !== undefined) e.checked = !!attrs.checked;
  return e;
}

// ---------- UI state ----------
const wfUi = (() => {
  try {
    return JSON.parse(localStorage.getItem(WF_UI_KEY)) || {};
  } catch {
    return {};
  }
})();

function saveWfUi() {
  try {
    localStorage.setItem(WF_UI_KEY, JSON.stringify({ mode: document.body.dataset.mode, wfId: wf.current?.id }));
  } catch { /* storage unavailable */ }
}

function setMode(mode) {
  document.body.dataset.mode = mode;
  document.querySelectorAll(".tab[data-mode]").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  clearOutputs();
  saveWfUi();
  syncKind();
  if (typeof onModeChanged === "function") onModeChanged(); // batch.js
}

// Which settings and panels show: a single model or a workflow. The Batch tab, and the video
// side's Tag scenes, run either one, as their Run with menu says.
function syncKind() {
  const body = document.body.dataset;
  const runWith = body.mode === "batch" || (body.app === "video" && body.vmode === "tag");
  body.kind = runWith ? $("batchKind").value : body.mode;
  if (body.kind === "workflow" && !wf.loaded) loadWorkflows(wfUi.wfId);
}

function setDirty(d) {
  wf.dirty = d;
  w.dirtyLabel.textContent = d ? "• unsaved changes" : "";
}

function confirmDiscard() {
  return !wf.dirty || confirm("Discard unsaved changes to this workflow?");
}

function slug(s) {
  return (s || "workflow").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 30) || "workflow";
}

function newId(name) {
  return `${slug(name)}-${Date.now().toString(36).slice(-5)}`;
}

// ---------- server ----------
async function loadWorkflows(selectId) {
  try {
    const data = await (await fetch("/api/workflows")).json();
    wf.list = data.workflows || [];
    wf.loaded = true;
  } catch (e) {
    setStatus("Could not load workflows: " + e.message, "error");
    return;
  }
  renderSelect(selectId);
}

async function saveDef(def) {
  const r = await fetch(`/api/workflows/${encodeURIComponent(def.id)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(def),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

function renderSelect(wantId) {
  fillSelect(w.select, wf.list.map((x) => ({ value: x.id, text: x.name })), wantId);
  openWorkflow(wf.list.find((x) => x.id === w.select.value) || null);
}

function openWorkflow(def) {
  if (def?.id !== wf.current?.id) clearOutputs(); // results belong to the previous workflow
  wf.current = def ? structuredClone(def) : null;
  setDirty(false);
  renderEditor();
  saveWfUi();
  if (typeof onConfigChanged === "function") onConfigChanged(); // batch.js
}

// ---------- editor ----------
function renderEditor() {
  const c = wf.current;
  w.name.value = c?.name ?? "";
  w.name.disabled = !c;
  [w.duplicate, w.remove, w.save, w.addVision, w.addLlm].forEach((b) => (b.disabled = !c));
  if (wf.jsonMode) {
    w.json.value = c ? JSON.stringify(c, null, 2) : "";
    return;
  }
  if (!c) {
    w.steps.replaceChildren(h("span", { class: "empty note" }, "No workflows yet. Press New to create one."));
  } else if (!c.steps.length) {
    w.steps.replaceChildren(h("span", { class: "empty note" }, "No steps yet. Add a vision or LLM step below."));
  } else {
    w.steps.replaceChildren(...c.steps.map((s, i) => stepCard(s, i)));
    updateRefWarnings();
  }
}

function field(label, path, kind, value, extra = {}) {
  const dataset = { field: path, kind };
  if (kind === "bool") {
    return h("label", { class: "check" }, h("input", { type: "checkbox", dataset, checked: !!value }), label);
  }
  let input;
  if (kind === "lines" || extra.textarea) {
    input = h("textarea", {
      dataset, rows: extra.rows || 4, spellcheck: "false", placeholder: extra.placeholder,
      value: kind === "lines" ? (value || []).join("\n") : value ?? "",
    });
  } else if (kind === "select") {
    input = h("select", { dataset, value: value ?? "" }, extra.options.map(([v, t]) => h("option", { value: v }, t)));
  } else {
    input = h("input", {
      type: kind === "number" ? "number" : "text", dataset, step: extra.step, min: extra.min, max: extra.max,
      placeholder: extra.placeholder, spellcheck: "false",
      value: kind === "csv" ? (value || []).join(", ") : value ?? "",
    });
  }
  return h("label", { class: extra.wide ? "wide" : null }, label, input);
}

function readField(input) {
  const kind = input.dataset.kind;
  if (kind === "bool") return input.checked;
  const v = input.value;
  if (kind === "number") return v === "" ? undefined : Number(v);
  if (kind === "select") return v === "" ? undefined : isNaN(v) ? v : Number(v);
  if (kind === "lines" || kind === "csv") {
    const items = v.split(kind === "lines" ? "\n" : ",").map((s) => s.trim()).filter(Boolean);
    return items.length ? items : undefined;
  }
  return v;
}

function setPath(obj, path, value) {
  const parts = path.split(".");
  let o = obj;
  for (const p of parts.slice(0, -1)) o = o[p] ??= {};
  const last = parts[parts.length - 1];
  if (value === undefined) delete o[last];
  else o[last] = value;
}

function visionFields(spec, p) {
  const kind = spec?.kind;
  return [
    field("Threshold", "params.threshold", "number", p.threshold, {
      step: 0.01, min: 0, max: 1, placeholder: spec ? `auto (${spec.default_threshold})` : "auto",
    }),
    field("Top-k", "params.top_k", "number", p.top_k, { min: 1, max: 50, placeholder: "5" }),
    kind === "detect" && field("Only classes", "params.classes", "csv", p.classes, { placeholder: "all (e.g. person, dog)" }),
    kind === "zero-shot" && field("Labels (one per line; blank = defaults)", "params.labels", "lines", p.labels, { wide: true, rows: 4 }),
    kind === "zero-shot" && field("Label template", "params.label_template", "text", p.label_template, {
      wide: true, placeholder: visionDefaults.default_template,
    }),
  ];
}

function llmFields(p) {
  return [
    field("System prompt", "params.system_prompt", "text", p.system_prompt, { textarea: true, wide: true, rows: 4 }),
    field("User prompt (template)", "params.user_prompt", "text", p.user_prompt, { textarea: true, wide: true, rows: 7 }),
    field("Temp", "params.temperature", "number", p.temperature, { step: 0.1, min: 0, max: 2, placeholder: "0.2" }),
    field("Max tokens", "params.max_tokens", "number", p.max_tokens, { min: 16, step: 16, placeholder: "1024" }),
    field("Send image", "params.include_image", "bool", p.include_image !== false),
    field("Force JSON", "params.json_mode", "bool", !!p.json_mode),
    h("div", { class: "warn wide ref-warn", hidden: true }),
  ];
}

// Flag {{step.…}} placeholders that don't point at an earlier step, before the user runs
function updateRefWarnings() {
  if (!wf.current || wf.jsonMode) return;
  wf.current.steps.forEach((step, i) => {
    const box = w.steps.querySelector(`.wf-step[data-index="${i}"] .ref-warn`);
    if (!box) return;
    const earlier = new Set(wf.current.steps.slice(0, i).map((s) => s.id));
    const text = `${step.params?.system_prompt || ""}\n${step.params?.user_prompt || ""}`;
    const bad = new Set();
    for (const [, root] of text.matchAll(/\{\{\s*([A-Za-z0-9_]+)[.\s}]/g)) {
      if (root !== "exif" && !earlier.has(root)) bad.add(root);
    }
    box.hidden = !bad.size;
    box.textContent = bad.size
      ? `⚠ No earlier step named ${[...bad].map((b) => `'${b}'`).join(", ")}. ` +
        `Earlier steps: ${[...earlier].join(", ") || "none"}. These placeholders will be sent as 'unknown'.`
      : "";
  });
}

function stepCard(step, i) {
  const n = wf.current.steps.length;
  const vision = step.type === "vision";
  const spec = vision ? visionModels.find((m) => m.id === step.model) : null;
  const p = (step.params ??= {});
  const model = vision
    ? h("select", { class: "wf-model", dataset: { field: "model", kind: "text" }, title: spec?.description, value: step.model },
        visionModels.map((m) => h("option", { value: m.id }, `${m.id} (${m.kind})`)),
        !spec && step.model ? h("option", { value: step.model }, `${step.model} (unavailable)`) : null)
    : h("input", {
        class: "wf-model", type: "text", list: "llmModelList", spellcheck: "false",
        placeholder: "model (blank = Single tab's model)", dataset: { field: "model", kind: "text" }, value: step.model || "",
      });
  const head = h("div", { class: "wf-step-head" },
    h("span", { class: "wf-num" }, String(i + 1)),
    h("input", {
      class: "wf-id", type: "text", spellcheck: "false", dataset: { field: "id", kind: "text" }, value: step.id,
      title: `Step id. Later prompts can use {{${step.id}.…}}`,
    }),
    h("span", { class: `badge ${step.type}` }, step.type),
    model,
    h("label", { class: "check", title: "Run this step" },
      h("input", { type: "checkbox", dataset: { field: "enabled", kind: "bool" }, checked: step.enabled !== false })),
    h("button", { type: "button", dataset: { act: "up" }, disabled: i === 0, title: "Move up" }, "↑"),
    h("button", { type: "button", dataset: { act: "down" }, disabled: i === n - 1, title: "Move down" }, "↓"),
    h("button", { type: "button", dataset: { act: "remove" }, title: "Remove step" }, "✕"));
  const body = h("div", { class: "wf-step-body" }, vision ? visionFields(spec, p) : llmFields(p));
  return h("div", { class: "wf-step" + (step.enabled === false ? " disabled" : ""), dataset: { index: i } }, head, body);
}

function onStepEdit(e) {
  const input = e.target.closest("[data-field]");
  const card = e.target.closest(".wf-step");
  if (!input || !card || !wf.current) return;
  const step = wf.current.steps[Number(card.dataset.index)];
  const path = input.dataset.field;
  setPath(step, path, readField(input));
  setDirty(true);
  if (path === "enabled") card.classList.toggle("disabled", !step.enabled);
  if (path === "id" || path.endsWith("_prompt")) updateRefWarnings();
  // A different vision model can need different fields (labels vs classes)
  if (path === "model" && step.type === "vision" && e.type === "change") {
    step.params = {};
    renderEditor();
  }
}

// When a step is renamed, point later prompts' {{old_id.…}} placeholders at the new id
function renameRefs(oldId, newId) {
  if (!oldId || oldId === newId || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(newId)) return;
  if (wf.current.steps.filter((s) => s.id === newId).length > 1) return; // clashes; leave prompts alone
  const pattern = new RegExp(`\\{\\{(\\s*)${oldId}\\.`, "g");
  let changed = 0;
  for (const step of wf.current.steps) {
    for (const key of ["system_prompt", "user_prompt"]) {
      const text = step.params?.[key];
      if (typeof text !== "string") continue;
      step.params[key] = text.replace(pattern, (_, space) => {
        changed++;
        return `{{${space}${newId}.`;
      });
    }
  }
  if (changed) {
    renderEditor();
    setStatus(`Updated ${changed} placeholder${changed > 1 ? "s" : ""} from {{${oldId}.…}} to {{${newId}.…}}.`);
  }
}

w.steps.addEventListener("input", onStepEdit);
w.steps.addEventListener("focusin", (e) => {
  if (e.target.classList.contains("wf-id")) e.target.dataset.prevId = e.target.value;
});
w.steps.addEventListener("change", (e) => {
  if (e.target.tagName === "SELECT" || e.target.type === "checkbox") onStepEdit(e);
  // "change" on a text input fires when it loses focus, so the rename is complete
  if (e.target.classList.contains("wf-id")) {
    renameRefs(e.target.dataset.prevId, e.target.value.trim());
    e.target.dataset.prevId = e.target.value.trim();
  }
});
w.steps.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-act]");
  if (!btn) return;
  const i = Number(btn.closest(".wf-step").dataset.index);
  const steps = wf.current.steps;
  if (btn.dataset.act === "remove") steps.splice(i, 1);
  else {
    const j = btn.dataset.act === "up" ? i - 1 : i + 1;
    [steps[i], steps[j]] = [steps[j], steps[i]];
  }
  setDirty(true);
  renderEditor();
});

w.name.addEventListener("input", () => {
  if (!wf.current) return;
  wf.current.name = w.name.value;
  setDirty(true);
});

function addStep(type) {
  if (!wf.current) return;
  const steps = wf.current.steps;
  const ids = new Set(steps.map((s) => s.id));
  const detector = visionModels.find((m) => m.kind === "detect");
  const base = type === "llm" ? "interpret" : detector && !ids.has("objects") ? "objects" : "vision";
  let id = base;
  for (let k = 2; ids.has(id); k++) id = `${base}${k}`;

  let step;
  if (type === "llm") {
    // Pre-fill the prompt with every earlier vision step's summary
    const refs = steps
      .filter((s) => s.type === "vision")
      .map((s) => `- ${s.id} (${s.model}): {{${s.id}.summary}}`)
      .join("\n");
    step = {
      id, type, model: "", enabled: true,
      params: {
        system_prompt: defaults.system_prompt || "",
        user_prompt: (refs ? `Results from specialised vision models:\n${refs}\n\n` : "") + "Classify this image.",
        include_image: true, json_mode: true, temperature: 0.2, max_tokens: 1024,
      },
    };
  } else {
    const model = (base === "objects" ? detector : null) || visionModels[0];
    step = { id, type, model: model?.id || "", enabled: true, params: {} };
  }
  steps.push(step);
  setDirty(true);
  renderEditor();
  w.steps.lastElementChild?.scrollIntoView({ block: "nearest" });
}

w.addVision.addEventListener("click", () => addStep("vision"));
w.addLlm.addEventListener("click", () => addStep("llm"));

// ---------- JSON editing ----------
function syncFromJson() {
  if (!wf.jsonMode || !wf.current) return true;
  try {
    const parsed = JSON.parse(w.json.value);
    parsed.id = wf.current.id;
    parsed.steps ??= [];
    wf.current = parsed;
    return true;
  } catch (e) {
    setStatus("Workflow JSON is invalid: " + e.message, "error");
    return false;
  }
}

function setJsonMode(on) {
  if (!on && !syncFromJson()) return;
  wf.jsonMode = on;
  w.json.hidden = !on;
  w.steps.hidden = on;
  w.addRow.hidden = on;
  w.name.hidden = on;
  w.jsonToggle.textContent = on ? "edit as form" : "edit as JSON";
  renderEditor();
}

w.jsonToggle.addEventListener("click", () => setJsonMode(!wf.jsonMode));
w.json.addEventListener("input", () => setDirty(true));

// ---------- workflow list actions ----------
w.select.addEventListener("change", () => {
  if (!confirmDiscard()) {
    w.select.value = wf.current?.id ?? "";
    return;
  }
  openWorkflow(wf.list.find((x) => x.id === w.select.value) || null);
});

w.newBtn.addEventListener("click", async () => {
  if (!confirmDiscard()) return;
  const def = { id: newId("workflow"), name: "New workflow", steps: [] };
  try {
    wf.list.push(await saveDef(def));
    renderSelect(def.id);
    w.name.focus();
    w.name.select();
  } catch (e) {
    setStatus("Could not create workflow: " + e.message, "error");
  }
});

w.duplicate.addEventListener("click", async () => {
  if (!wf.current || !syncFromJson()) return;
  const def = structuredClone(wf.current);
  def.id = newId(def.name);
  def.name = `${def.name} (copy)`;
  try {
    wf.list.push(await saveDef(def));
    renderSelect(def.id);
    setStatus("Duplicated. Unsaved edits went into the copy; the original is unchanged.");
  } catch (e) {
    setStatus("Could not duplicate: " + e.message, "error");
  }
});

w.remove.addEventListener("click", async () => {
  if (!wf.current || !confirm(`Delete workflow "${wf.current.name}"?`)) return;
  try {
    const r = await fetch(`/api/workflows/${encodeURIComponent(wf.current.id)}`, { method: "DELETE" });
    if (!r.ok && r.status !== 404) throw new Error(`HTTP ${r.status}`);
    wf.list = wf.list.filter((x) => x.id !== wf.current.id);
    setDirty(false);
    renderSelect();
  } catch (e) {
    setStatus("Could not delete: " + e.message, "error");
  }
});

w.save.addEventListener("click", async () => {
  if (!wf.current || !syncFromJson()) return;
  try {
    const saved = await saveDef(wf.current);
    wf.list = wf.list.map((x) => (x.id === saved.id ? saved : x));
    renderSelect(saved.id);
    setStatus("Workflow saved.");
  } catch (e) {
    setStatus("Could not save: " + e.message, "error");
  }
});

w.copy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(w.result.textContent);
    setStatus("Result copied.");
  } catch {
    setStatus("Clipboard not available.", "error");
  }
});

w.exportBtn.addEventListener("click", () => {
  exportImage(w.result.textContent, `workflow: ${wf.current?.name || "unnamed"}`);
});

window.addEventListener("beforeunload", (e) => {
  if (wf.dirty) e.preventDefault();
});

// Called by app.js once the vision model list arrives, so model dropdowns fill in
function onVisionModelsLoaded() {
  if (wf.current && !wf.jsonMode) renderEditor();
}

// ---------- running ----------
async function* sseEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const line = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (line.startsWith("data:")) yield JSON.parse(line.slice(5));
    }
  }
}

function formatMs(ms) {
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function runCard(step, i) {
  const skipped = step.enabled === false;
  const state = h("span", { class: "state" }, skipped ? "skipped" : "pending");
  const body = h("div", { class: "run-step-body" });
  const modelName = step.model || (step.type === "llm" ? currentModel() || "default model" : "");
  const root = h("div", { class: "run-step" + (skipped ? " skipped" : "") },
    h("div", { class: "run-step-head" },
      h("span", { class: "wf-num" }, String(i + 1)),
      h("span", { class: `badge ${step.type}` }, step.type),
      h("strong", {}, step.id),
      h("span", { class: "meta" }, modelName),
      state),
    body);
  let text = null;
  let reasoning = null;
  let usage = null;
  const textPre = () => (text ??= body.appendChild(h("pre")));
  return {
    root,
    setState(cls, label) {
      root.className = `run-step ${cls}`;
      state.textContent = label;
    },
    note(msg, cls) {
      body.appendChild(h("div", { class: cls }, msg));
    },
    details(title, content) {
      body.appendChild(h("details", {}, h("summary", {}, title), h("pre", {}, content)));
    },
    reasoning(t) {
      if (!reasoning) {
        reasoning = h("span", { class: "reasoning" });
        textPre().prepend(reasoning);
      }
      reasoning.textContent += t;
    },
    token(t) {
      textPre().append(t);
      text.scrollTop = text.scrollHeight;
    },
    usage(u) {
      usage = u;
    },
    done(out, ms) {
      const extra = usage ? ` · ${usage.prompt_tokens ?? "?"}→${usage.completion_tokens ?? "?"} tok` : "";
      this.setState("done", `✓ ${formatMs(ms)}${extra}`);
      if (step.type === "vision") body.appendChild(h("div", { class: "summary" }, out.summary || ""));
      else if (out.json_error) this.note("Output was not valid JSON: " + out.json_error, "warn");
      this.details("Full output", JSON.stringify(out, null, 2));
    },
    fail(msg) {
      this.setState("failed", "✗ failed");
      this.note(msg, "warn");
    },
  };
}

// Builds the step cards for a run and returns a function that applies one run event to them.
// Used for live runs and for replaying a saved run (replay = true). out says where: the Workflows
// tab's panels by default; video-tag.js passes its own, with no result panel or boxes.
function workflowView(steps, out = { run: w.run, result: w.result, meta: w.meta, boxes: true }) {
  const cards = {};
  out.run.replaceChildren(...steps.map((s, i) => (cards[s.id] = runCard(s, i)).root));
  if (out.result) out.result.textContent = "";
  out.meta.textContent = "";
  let drewBoxes = false;
  return (ev, replay = false) => {
    const card = cards[ev.id];
    switch (ev.type) {
      case "step_start": card.setState("running", "running…"); break;
      case "step_skipped": card.setState("skipped", "skipped"); break;
      case "warning": card.note(ev.message, "warn"); break;
      case "prompt": card.details("Prompt sent", ev.user); break;
      case "reasoning": card.reasoning(ev.text); break;
      case "token": card.token(ev.text); break;
      case "usage": card.usage(ev.usage); break;
      case "step_done":
        // Saved runs keep the final output rather than every token
        if (replay && ev.output?.text) card.token(ev.output.text);
        card.done(ev.output, ev.ms);
        // Draw the first detector's boxes on the image
        if (out.boxes && !drewBoxes && ev.output.detections?.length) {
          drawDetections(ev.output.detections, ev.output.image_size);
          drewBoxes = true;
        }
        break;
      case "step_error": card.fail(ev.message); break;
      case "done":
        if (out.result) out.result.textContent = JSON.stringify(ev.result, null, 2);
        out.meta.textContent = `${formatMs(ev.ms)} total`;
        break;
    }
  };
}

// Run a workflow on one image. onEvent sees each run event as it streams (see workflowView).
async function computeWorkflow(signal, { def, dataUrl, exif }, onEvent = null) {
  const record = {
    kind: "workflow", workflow: { id: def.id, name: def.name }, steps: structuredClone(def.steps),
    events: [], result: null, error: null, aborted: false,
  };
  try {
    const r = await fetch("/api/workflows/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        workflow: def,
        image_data_url: dataUrl,
        exif,
        base_url: el.baseUrl.value.trim(),
        api_key: el.apiKey.value,
        model: currentModel(),
      }),
    });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
    }
    for await (const ev of sseEvents(r)) {
      if (ev.type === "error") throw new Error(ev.message);
      if (ev.type !== "token" && ev.type !== "reasoning") record.events.push(ev);
      if (ev.type === "done") record.result = ev.result;
      try {
        onEvent?.(ev);
      } catch { /* display problems must not fail the run */ }
    }
  } catch (e) {
    if (e.name === "AbortError") record.aborted = true;
    else record.error = e.message;
  }
  return record;
}

async function runWorkflow(signal) {
  if (!syncFromJson()) return null;
  const def = wf.current;
  if (!def || !def.steps.length) {
    setStatus("This workflow has no steps yet.", "error");
    return null;
  }
  // Only draw while the photo panel still shows this image: the user may move on mid-run
  const file = currentFile;
  const dataUrl = imageDataUrl;
  // Templates can use {{exif.*}}, so wait for the original file's metadata
  if (exifPromise) await exifPromise;
  if (currentFile !== file) return null; // moved on; exifData now belongs to another photo
  const apply = workflowView(def.steps);
  const record = await computeWorkflow(signal, { def, dataUrl, exif: flattenExif(exifData) },
    (ev) => currentFile === file && apply(ev));
  if (currentFile !== file) return record;
  const failed = record.events.filter((e) => e.type === "step_error").length;
  if (record.aborted) setStatus("Stopped.");
  else if (record.error) setStatus("Workflow failed: " + record.error, "error");
  else setStatus(failed ? `Done, ${failed} step${failed > 1 ? "s" : ""} failed.` : "Done.", failed ? "error" : "ok");
  return record;
}

function showWorkflowRecord(record) {
  const apply = workflowView(record.steps);
  record.events.forEach((ev) => apply(ev, true));
  if (record.error) w.meta.textContent = "failed: " + record.error;
}

// ---------- init ----------
document.querySelectorAll(".tab[data-mode]").forEach((t) => t.addEventListener("click", () => setMode(t.dataset.mode)));
// The saved tab is restored by batch.js, which loads last and defines the Batch tab's hooks

"use strict";

const MAX_IMAGE_EDGE = 1536;
const STORAGE_KEY = "llm-classifier-settings";
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);
const el = {
  backend: $("backend"),
  baseUrl: $("baseUrl"),
  apiKey: $("apiKey"),
  modelSelect: $("modelSelect"),
  modelText: $("modelText"),
  refreshModels: $("refreshModels"),
  temperature: $("temperature"),
  maxTokens: $("maxTokens"),
  jsonMode: $("jsonMode"),
  withYolo: $("withYolo"),
  detectStrip: $("detectStrip"),
  visionModel: $("visionModel"),
  refreshVision: $("refreshVision"),
  topK: $("topK"),
  threshold: $("threshold"),
  autoRun: $("autoRun"),
  runBtn: $("runBtn"),
  status: $("status"),
  dropZone: $("dropZone"),
  dropHint: $("dropHint"),
  preview: $("preview"),
  overlay: $("overlay"),
  imageInfo: $("imageInfo"),
  fileInput: $("fileInput"),
  systemPrompt: $("systemPrompt"),
  userPrompt: $("userPrompt"),
  resetPrompt: $("resetPrompt"),
  labels: $("labels"),
  labelTemplate: $("labelTemplate"),
  labelsNote: $("labelsNote"),
  resetLabels: $("resetLabels"),
  rawOutput: $("rawOutput"),
  visionOutput: $("visionOutput"),
  outputMeta: $("outputMeta"),
  outputPanel: $("outputPanel"),
  jsonOutput: $("jsonOutput"),
  copyJson: $("copyJson"),
  exportImg: $("exportImg"),
  exportSettings: $("exportSettings"),
  exportOverwrite: document.querySelectorAll(".export-overwrite"), // one per tab, kept in sync
  exifPanel: $("exifPanel"),
  exifCount: $("exifCount"),
  exifFilter: $("exifFilter"),
  copyExif: $("copyExif"),
  toggleExif: $("toggleExif"),
  exifSummary: $("exifSummary"),
  exifTable: $("exifTable"),
};

let defaults = {};
let visionDefaults = { default_labels: [], default_template: "" };
let visionModels = [];
let imageDataUrl = null;
let currentFile = null; // the original upload, for EXIF and export
let abortController = null;
let exifData = null;
let exifSeq = 0;
let exifPromise = null; // settles when the current file's metadata has loaded (or failed)

// ---------- settings persistence ----------
function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch {
    return {};
  }
}

function saveSettings() {
  const s = {
    backend: el.backend.value,
    baseUrl: el.baseUrl.value,
    apiKey: el.apiKey.value,
    model: currentModel(),
    temperature: el.temperature.value,
    maxTokens: el.maxTokens.value,
    jsonMode: el.jsonMode.checked,
    withYolo: el.withYolo.checked,
    visionModel: el.visionModel.value,
    topK: el.topK.value,
    threshold: el.threshold.value,
    autoRun: el.autoRun.checked,
    exportOverwrite: el.exportOverwrite[0].checked,
    exifCollapsed: el.exifPanel.classList.contains("collapsed"),
    systemPrompt: el.systemPrompt.value,
    userPrompt: el.userPrompt.value,
    labels: el.labels.value,
    labelTemplate: el.labelTemplate.value,
  };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch { /* storage unavailable */ }
}

function setStatus(msg, kind = "ok") {
  // Tag scenes shares the model and workflow settings; their messages show on its status line
  if (document.body.dataset.app === "video" && document.body.dataset.vmode === "tag" && typeof vStatus === "function") {
    vStatus(msg, kind); // video.js
    return;
  }
  el.status.textContent = msg || "";
  el.status.className = "status " + kind;
}

function isVision() {
  return el.backend.value === "vision";
}

function setBackend(value) {
  el.backend.value = value;
  document.body.dataset.backend = value;
  clearOutputs();
  if (value === "vision") refreshVisionModels();
  else refreshModels();
}

// ---------- LLM models ----------
function currentModel() {
  return el.modelText.hidden ? el.modelSelect.value : el.modelText.value.trim();
}

async function refreshModels(preferred) {
  const want = preferred ?? currentModel();
  const params = new URLSearchParams({ base_url: el.baseUrl.value.trim() });
  if (el.apiKey.value) params.set("api_key", el.apiKey.value);
  setStatus("Loading models…");
  try {
    const r = await fetch("/api/models?" + params);
    const data = await r.json();
    if (data.error || !data.models.length) {
      el.modelSelect.hidden = true;
      el.modelText.hidden = false;
      if (want) el.modelText.value = want;
      setStatus(data.error ? `Could not list models (${data.error}) — enter a model name manually.` : "Endpoint returned no models.", data.error ? "error" : "ok");
      return;
    }
    fillSelect(el.modelSelect, data.models.map((id) => ({ value: id, text: id })), want);
    // Workflow LLM steps offer the same models as suggestions
    $("llmModelList").replaceChildren(...data.models.map((id) => Object.assign(document.createElement("option"), { value: id })));
    el.modelSelect.hidden = false;
    el.modelText.hidden = true;
    setStatus(`${data.models.length} model${data.models.length === 1 ? "" : "s"} available.`);
  } catch (e) {
    setStatus("Model list request failed: " + e.message, "error");
  } finally {
    if (typeof onModelListChanged === "function") onModelListChanged(); // video-tag.js
  }
}

function fillSelect(select, options, want) {
  select.innerHTML = "";
  for (const o of options) {
    const opt = document.createElement("option");
    opt.value = o.value;
    opt.textContent = o.text;
    if (o.title) opt.title = o.title;
    select.appendChild(opt);
  }
  if (want && options.some((o) => o.value === want)) select.value = want;
}

// ---------- vision models ----------
function currentVisionSpec() {
  return visionModels.find((m) => m.id === el.visionModel.value);
}

async function refreshVisionModels(preferred) {
  const want = preferred ?? el.visionModel.value;
  setStatus("Loading vision models…");
  try {
    const data = await (await fetch("/api/vision/models")).json();
    if (data.error) {
      visionModels = [];
      el.visionModel.innerHTML = "";
      setStatus(data.error, "error");
      return;
    }
    visionModels = data.models;
    visionDefaults = { default_labels: data.default_labels, default_template: data.default_template };
    if (!el.labels.value.trim()) el.labels.value = visionDefaults.default_labels.join("\n");
    el.labelTemplate.placeholder = visionDefaults.default_template;
    fillSelect(
      el.visionModel,
      visionModels.map((m) => ({ value: m.id, text: `${m.id} (${m.kind})`, title: m.description })),
      want
    );
    updateVisionUi();
    if (typeof onVisionModelsLoaded === "function") onVisionModelsLoaded(); // workflows.js
    if (typeof onModelListChanged === "function") onModelListChanged(); // video-tag.js
    setStatus(`${visionModels.length} vision models available. First use of a model downloads its weights.`);
  } catch (e) {
    setStatus("Vision model list request failed: " + e.message, "error");
  }
}

function updateVisionUi() {
  const spec = currentVisionSpec();
  if (!spec) return;
  const custom = spec.kind === "zero-shot";
  el.labels.disabled = !custom;
  el.labelTemplate.disabled = !custom;
  el.labelsNote.hidden = custom;
  el.labelsNote.textContent = custom ? "" : `${spec.id} uses its own fixed label set (${spec.labels}); this list is ignored.`;
  el.threshold.placeholder = `auto (${spec.default_threshold})`;
}

// ---------- image handling ----------
function resetPreview() {
  imageDataUrl = null;
  el.preview.hidden = true;
  el.preview.removeAttribute("src");
  el.dropHint.hidden = false;
  el.dropZone.classList.remove("has-image");
  el.imageInfo.hidden = true;
  el.runBtn.disabled = true;
  clearOutputs();
}

// Decode a file and scale it down for the models. Rejects if the browser can't display it.
function prepareImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(img.width, img.height));
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      canvas.getContext("2d").drawImage(img, 0, 0, w, h);
      resolve({ dataUrl: canvas.toDataURL("image/jpeg", 0.9), width: img.width, height: img.height, w, h });
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`The browser can't display ${file.name}.`));
    };
    img.src = url;
  });
}

// Show a file in the photo panel and make it the image to run. Resolves to false if it can't be shown.
async function loadImage(file, { autoRun = el.autoRun.checked } = {}) {
  if (!file) return false;
  currentFile = file;
  // Metadata comes from the original file: the resized canvas copy we send to models has none
  readExif(file);
  let img;
  try {
    img = await prepareImage(file);
  } catch {
    if (currentFile !== file) return false;
    resetPreview();
    setStatus(`The browser can't display ${file.name}, so it can't be sent to a model. Metadata is shown below.`, "error");
    return false;
  }
  if (currentFile !== file) return false; // a newer file was chosen meanwhile
  imageDataUrl = img.dataUrl;
  el.preview.src = imageDataUrl;
  el.preview.hidden = false;
  el.overlay.setAttribute("viewBox", `0 0 ${img.w} ${img.h}`);
  el.dropHint.hidden = true;
  el.dropZone.classList.add("has-image");
  el.imageInfo.hidden = false;
  el.imageInfo.textContent = `${file.name} · ${img.width}×${img.height}` + (img.w < img.width ? ` → ${img.w}×${img.h}` : "");
  el.runBtn.disabled = false;
  clearOutputs();
  if (autoRun) run();
  return true;
}

// In the Batch tab the photo panel shows the focused photo, so it doesn't take drops or clicks
const singlePhotoInput = () => document.body.dataset.mode !== "batch";
el.dropZone.addEventListener("click", () => singlePhotoInput() && el.fileInput.click());
el.dropZone.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && singlePhotoInput()) el.fileInput.click();
});
el.fileInput.addEventListener("change", () => {
  loadImage(el.fileInput.files[0]);
  el.fileInput.value = "";
});
["dragenter", "dragover"].forEach((t) =>
  el.dropZone.addEventListener(t, (e) => {
    e.preventDefault();
    if (singlePhotoInput()) el.dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((t) =>
  el.dropZone.addEventListener(t, (e) => {
    e.preventDefault();
    el.dropZone.classList.remove("dragover");
  })
);
el.dropZone.addEventListener("drop", (e) => singlePhotoInput() && loadImage(e.dataTransfer.files[0]));
// Avoid the browser navigating to the file if it's dropped outside the zone
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => e.preventDefault());

// ---------- EXIF ----------
const EXIF_SUMMARY = [
  ["Camera", (t) => [t.Make, t.Model].filter(Boolean).join(" ")],
  ["Lens", (t) => t.LensModel || t.LensID || t.Lens],
  ["Taken", (t) => t.DateTimeOriginal || t.CreateDate],
  ["Exposure", (t) => [t.FNumber && `f/${t.FNumber}`, t.ExposureTime && `${t.ExposureTime}s`, t.ISO && `ISO ${t.ISO}`, t.FocalLength]
    .filter(Boolean).join(" · ")],
  ["Size", (t) => t.ImageSize],
  ["Software", (t) => t.Software],
];

function readExif(file) {
  exifPromise = loadExif(file);
  return exifPromise;
}

async function loadExif(file) {
  const seq = ++exifSeq;
  exifData = null;
  el.exifCount.textContent = "";
  el.exifSummary.replaceChildren();
  el.exifTable.innerHTML = '<span class="empty">Reading metadata…</span>';
  try {
    const data = await fetchExif(file);
    if (seq !== exifSeq) return; // a newer file was dropped meanwhile
    exifData = data;
    renderExifSummary(data);
    renderExifTable();
  } catch (e) {
    if (seq !== exifSeq) return;
    el.exifTable.innerHTML = "";
    const err = document.createElement("span");
    err.className = "empty";
    err.textContent = "Could not read metadata: " + e.message;
    el.exifTable.appendChild(err);
  }
}

// All metadata exiftool finds in the original file, grouped (see /api/exif)
async function fetchExif(file) {
  const form = new FormData();
  form.append("file", file, file.name);
  const r = await fetch("/api/exif", { method: "POST", body: form });
  const data = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

// First value of each tag name, whichever group it came from
function flattenExif(data) {
  const flat = {};
  for (const tags of Object.values(data?.groups || {})) {
    for (const [k, v] of Object.entries(tags)) if (!(k in flat)) flat[k] = v;
  }
  return flat;
}

function renderExifSummary(data) {
  const flat = flattenExif(data);
  const chips = [];
  for (const [name, pick] of EXIF_SUMMARY) {
    const v = pick(flat);
    if (v) chips.push(chip(`${name}: ${v}`));
  }
  if (data.location) {
    const { lat, lon } = data.location;
    const c = chip("GPS: ");
    const a = document.createElement("a");
    a.href = `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=15/${lat}/${lon}`;
    a.target = "_blank";
    a.rel = "noopener noreferrer";
    a.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
    c.appendChild(a);
    chips.push(c);
  }
  el.exifSummary.replaceChildren(...chips);
}

function chip(text) {
  const c = document.createElement("span");
  c.className = "chip";
  c.textContent = text;
  return c;
}

function formatExifValue(v) {
  if (v === null || v === undefined) return "";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
}

function renderExifTable() {
  if (!exifData) return;
  const q = el.exifFilter.value.trim().toLowerCase();
  const table = document.createElement("table");
  let shown = 0;
  for (const [group, tags] of Object.entries(exifData.groups)) {
    const rows = Object.entries(tags).filter(([k, v]) =>
      !q || group.toLowerCase().includes(q) || k.toLowerCase().includes(q) || formatExifValue(v).toLowerCase().includes(q)
    );
    if (!rows.length) continue;
    const head = table.insertRow();
    const th = document.createElement("th");
    th.className = "group";
    th.colSpan = 2;
    th.textContent = `${group} (${rows.length})`;
    head.appendChild(th);
    for (const [k, v] of rows) {
      const tr = table.insertRow();
      const tag = tr.insertCell();
      tag.className = "tag";
      tag.textContent = k;
      const val = tr.insertCell();
      val.className = "val";
      val.textContent = formatExifValue(v);
      shown++;
    }
  }
  el.exifCount.textContent = q ? `${shown} of ${exifData.count} tags` : `${exifData.count} tags`;
  if (shown) {
    el.exifTable.replaceChildren(table);
  } else {
    el.exifTable.innerHTML = `<span class="empty">${q ? "No tags match the filter." : "No metadata found."}</span>`;
  }
}

// ---------- export ----------
async function exportImage(jsonText, source) {
  if (!currentFile) {
    setStatus("Drop an image first.", "error");
    return;
  }
  let result = null;
  try {
    result = JSON.parse(jsonText);
  } catch { /* reported below */ }
  if (!result || typeof result !== "object") {
    setStatus("Nothing to export yet: run the image and wait for a JSON result.", "error");
    return;
  }
  const form = new FormData();
  form.append("file", currentFile, currentFile.name);
  form.append("result", JSON.stringify(result));
  form.append("overwrite", el.exportOverwrite[0].checked);
  form.append("source", source);
  setStatus("Writing metadata…");
  try {
    const r = await fetch("/api/export", { method: "POST", body: form });
    if (!r.ok) {
      const d = await r.json().catch(() => ({}));
      throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
    }
    const report = JSON.parse(decodeURIComponent(r.headers.get("X-Export-Report") || "{}"));
    const url = URL.createObjectURL(await r.blob());
    const a = Object.assign(document.createElement("a"), { href: url, download: report.filename || currentFile.name });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    setStatus(exportSummary(report), report.written?.length ? "ok" : "error");
  } catch (e) {
    setStatus("Export failed: " + e.message, "error");
  }
}

function exportSummary(report) {
  const written = report.written || [];
  const skipped = report.skipped || [];
  const parts = [`Exported ${report.filename}: wrote ${written.length} tag${written.length === 1 ? "" : "s"}`];
  if (skipped.length) {
    parts.push(`kept ${skipped.length} existing (${skipped.map((x) => x.tag).join(", ")}). Tick "Overwrite existing metadata" to replace them`);
  }
  return parts.join(", ") + "." + (report.warnings || []).map((w) => " " + w).join("");
}

// ---------- settings file for the CLI ----------
// Everything the CLI (app/cli.py) needs to reproduce a run, except the API key.
function settingsSnapshot() {
  const kind = document.body.dataset.kind === "workflow" ? "workflow" : isVision() ? "vision" : "llm";
  const settings = {
    format: "photo-organizer/settings",
    version: 1,
    exported: new Date().toISOString(),
    kind,
    // Workflow LLM steps use this endpoint and, when they name none, this model
    llm: {
      base_url: el.baseUrl.value.trim(),
      model: currentModel(),
      system_prompt: el.systemPrompt.value,
      user_prompt: el.userPrompt.value,
      temperature: parseFloat(el.temperature.value) || 0,
      max_tokens: parseInt(el.maxTokens.value, 10) || 1024,
      json_mode: el.jsonMode.checked,
    },
    vision: {
      model: el.visionModel.value,
      labels: el.labels.value.split("\n").map((l) => l.trim()).filter(Boolean),
      label_template: el.labelTemplate.value.trim() || null,
      top_k: parseInt(el.topK.value, 10) || 5,
      threshold: el.threshold.value === "" ? null : parseFloat(el.threshold.value),
    },
  };
  if (kind === "workflow") {
    if (!syncFromJson()) return null; // workflows.js: the JSON editor has an error
    if (!wf.current) {
      setStatus("Choose a workflow first.", "error");
      return null;
    }
    settings.workflow = structuredClone(wf.current); // includes unsaved edits
  }
  return settings;
}

el.exportSettings.addEventListener("click", () => {
  const settings = settingsSnapshot();
  if (!settings) return;
  const name = settings.kind === "workflow" ? settings.workflow.name : settings.kind === "vision" ? settings.vision.model : settings.llm.model;
  const filename = `${slug(name || settings.kind)}.settings.json`; // slug: workflows.js
  const url = URL.createObjectURL(new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  setStatus(`Saved ${filename} (no API key). Run it with: python -m app.cli ${filename} <photos or folder>`);
});

// ---------- JSON extraction ----------
function extractJson(text) {
  // Drop <think>...</think> blocks some reasoning models emit inline
  let t = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  const fence = t.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) t = fence[1].trim();
  try {
    return JSON.parse(t);
  } catch { /* fall through to balanced-block scan */ }

  const start = t.search(/[{[]/);
  if (start === -1) throw new Error("No JSON object found in output.");
  const open = t[start];
  const close = open === "{" ? "}" : "]";
  let depth = 0, inStr = false, esc = false;
  for (let i = start; i < t.length; i++) {
    const c = t[i];
    if (inStr) {
      if (esc) esc = false;
      else if (c === "\\") esc = true;
      else if (c === '"') inStr = false;
      continue;
    }
    if (c === '"') inStr = true;
    else if (c === open) depth++;
    else if (c === close && --depth === 0) return JSON.parse(t.slice(start, i + 1));
  }
  throw new Error("JSON object in output is incomplete.");
}

// ---------- run ----------
function showOutputView(view) {
  el.outputPanel.dataset.view = view;
  el.outputPanel.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
}

function clearOutputs() {
  el.rawOutput.textContent = "";
  el.rawOutput.classList.remove("error");
  el.visionOutput.textContent = "";
  el.visionOutput.classList.remove("error");
  el.jsonOutput.textContent = "";
  el.jsonOutput.classList.remove("error");
  el.outputMeta.textContent = "";
  el.overlay.replaceChildren();
  el.detectStrip.hidden = true;
  el.detectStrip.replaceChildren();
  $("wfRun").replaceChildren();
  $("wfResult").textContent = "";
  $("wfMeta").textContent = "";
}

async function run() {
  if (abortController) {
    abortController.abort();
    return;
  }
  saveSettings();
  if (document.body.dataset.mode === "batch") {
    await withRun(runBatch); // batch.js
    refreshTiles(); // re-enables Export all and Run, which stay disabled while a run is going
  } else if (imageDataUrl) {
    await withRun(runActive);
  }
}

// Run fn(signal) with the Run button acting as Stop
async function withRun(fn) {
  abortController = new AbortController();
  el.runBtn.textContent = "Stop";
  setStatus("Running…");
  try {
    return await fn(abortController.signal);
  } finally {
    abortController = null;
    el.runBtn.textContent = runButtonLabel();
  }
}

function runButtonLabel() {
  if (document.body.dataset.mode !== "batch") return "Run";
  if (document.body.dataset.focus) return "Run photo";
  return view.filtered ? "Run shown" : "Run all"; // batch-view.js
}

// Run the photo shown in the photo panel through the selected model or workflow, streaming
// into the output panels. Returns a record of the outputs that showRecord() can display later.
async function runActive(signal) {
  clearOutputs();
  if (document.body.dataset.kind === "workflow") return runWorkflow(signal); // workflows.js
  if (isVision()) return runVision(signal);
  // The object detector runs in parallel with the LLM; it only draws boxes and the summary strip
  const [record] = await Promise.all([runLlm(signal), el.withYolo.checked ? runYoloOverlay(signal) : null]);
  return record;
}

// Same as runActive but for any image, without touching the photo or output panels (batch runs)
async function computeActive(signal, { dataUrl, exif = {} }) {
  if (document.body.dataset.kind === "workflow") return computeWorkflow(signal, { def: wf.current, dataUrl, exif }); // workflows.js
  if (isVision()) return computeVision(signal, dataUrl);
  return computeLlm(signal, dataUrl);
}

function showRecord(record) {
  if (record.kind === "llm") showLlmRecord(record);
  else if (record.kind === "vision") showVisionRecord(record);
  else showWorkflowRecord(record); // workflows.js
}

// Classify one image with the LLM. onEvent sees each streamed event (tokens, reasoning, errors).
async function computeLlm(signal, dataUrl, onEvent = null) {
  const started = performance.now();
  const record = { kind: "llm", text: "", reasoning: "", error: null, aborted: false, meta: "", json: null, jsonError: null };
  let usage = null;
  let timings = null;
  try {
    const r = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        base_url: el.baseUrl.value.trim(),
        api_key: el.apiKey.value,
        model: currentModel(),
        system_prompt: el.systemPrompt.value,
        user_prompt: el.userPrompt.value,
        image_data_url: dataUrl,
        temperature: parseFloat(el.temperature.value) || 0,
        max_tokens: parseInt(el.maxTokens.value, 10) || 1024,
        json_mode: el.jsonMode.checked,
      }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
    for await (const ev of sseEvents(r)) { // workflows.js
      if (ev.type === "token") record.text += ev.text;
      else if (ev.type === "reasoning") record.reasoning += ev.text;
      else if (ev.type === "usage") usage = ev.usage;
      else if (ev.type === "timings") timings = ev.timings;
      else if (ev.type === "error") record.error = ev.message;
      try {
        onEvent?.(ev);
      } catch { /* display problems must not fail the run */ }
    }
  } catch (e) {
    if (e.name === "AbortError") record.aborted = true;
    else record.error = e.message;
  }
  const meta = [`${((performance.now() - started) / 1000).toFixed(1)}s`];
  if (usage) meta.push(`${usage.prompt_tokens ?? "?"} in / ${usage.completion_tokens ?? "?"} out`);
  if (timings?.predicted_per_second) meta.push(`${timings.predicted_per_second.toFixed(1)} tok/s`);
  record.meta = meta.join(" · ");
  if (record.text) {
    try {
      record.json = extractJson(record.text);
    } catch (e) {
      record.jsonError = e.message;
    }
  }
  return record;
}

async function runLlm(signal) {
  // Only draw while the photo panel still shows this image: the user may move on mid-run
  const file = currentFile;
  const onScreen = () => currentFile === file;
  const textNode = document.createTextNode("");
  let reasoningNode = null;
  el.rawOutput.appendChild(textNode);
  const record = await computeLlm(signal, imageDataUrl, (ev) => {
    if (!onScreen() || !textNode.isConnected) return;
    if (ev.type === "token") {
      textNode.data += ev.text;
    } else if (ev.type === "reasoning") {
      if (!reasoningNode) {
        reasoningNode = document.createElement("span");
        reasoningNode.className = "reasoning";
        el.rawOutput.insertBefore(reasoningNode, textNode);
      }
      reasoningNode.textContent += ev.text;
    }
    el.rawOutput.scrollTop = el.rawOutput.scrollHeight;
  });
  if (!onScreen()) return record;
  showLlmRecord(record);
  if (record.aborted) setStatus("Stopped.");
  else if (record.error) setStatus("Request failed.", "error");
  else if (!record.text) setStatus("Model returned no content.", "error");
  else if (record.jsonError) setStatus("Done (output was not valid JSON).", "error");
  else setStatus("Done.");
  return record;
}

function showLlmRecord(record) {
  el.rawOutput.replaceChildren();
  if (record.reasoning) {
    const span = document.createElement("span");
    span.className = "reasoning";
    span.textContent = record.reasoning;
    el.rawOutput.append(span);
  }
  el.rawOutput.append(record.text + (record.error ? (record.text ? "\n\n" : "") + record.error : ""));
  el.rawOutput.classList.toggle("error", !!record.error);
  el.outputMeta.textContent = record.meta || "";
  if (record.json != null) {
    el.jsonOutput.textContent = JSON.stringify(record.json, null, 2);
  } else if (record.jsonError) {
    el.jsonOutput.classList.add("error");
    el.jsonOutput.textContent = "Could not parse JSON: " + record.jsonError;
  }
  showOutputView(record.error ? "raw" : "json");
}

async function computeVision(signal, dataUrl) {
  const spec = currentVisionSpec();
  if (!spec) {
    return { kind: "vision", data: null, error: "No vision model selected — is the vision service running?", aborted: false };
  }
  const started = performance.now();
  const threshold = el.threshold.value === "" ? null : parseFloat(el.threshold.value);
  try {
    const r = await fetch("/api/vision/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({
        model: spec.id,
        image_data_url: dataUrl,
        labels: el.labels.value.split("\n").map((l) => l.trim()).filter(Boolean),
        label_template: el.labelTemplate.value.trim() || null,
        top_k: parseInt(el.topK.value, 10) || 5,
        threshold,
      }),
    });
    const data = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) {
      const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? data);
      throw new Error(`HTTP ${r.status}: ${detail}`);
    }
    spec.loaded = true;
    const t = data.timing_ms || {};
    const meta = [`${((performance.now() - started) / 1000).toFixed(2)}s total`, `${t.inference} ms inference`];
    if (t.load) meta.push(`${(t.load / 1000).toFixed(1)}s load`);
    return { kind: "vision", data, meta: meta.join(" · "), error: null, aborted: false };
  } catch (e) {
    const aborted = e.name === "AbortError";
    return { kind: "vision", data: null, error: aborted ? null : e.message, aborted };
  }
}

async function runVision(signal) {
  const file = currentFile;
  const spec = currentVisionSpec();
  if (spec && !spec.loaded) setStatus(`Running… (loading ${spec.id}; first use downloads weights and can take a minute)`);
  const record = await computeVision(signal, imageDataUrl);
  if (currentFile !== file) return record; // the user moved on to another photo
  if (record.aborted) {
    setStatus("Stopped.");
    return record;
  }
  showVisionRecord(record);
  if (record.error) setStatus(spec ? "Request failed." : record.error, "error");
  else setStatus("Done.");
  return record;
}

function showVisionRecord(record) {
  if (record.data) {
    renderVisionResult(record.data);
    el.jsonOutput.textContent = JSON.stringify(record.data, null, 2);
    el.outputMeta.textContent = record.meta || "";
    showOutputView("json");
  } else {
    el.visionOutput.classList.add("error");
    el.visionOutput.textContent = record.error || "No result.";
    showOutputView("raw");
  }
}

function renderVisionResult(data) {
  const out = el.visionOutput;
  out.replaceChildren();
  if (!data.predictions.length) {
    const p = document.createElement("div");
    p.className = "empty";
    p.textContent = data.kind === "detect" ? `No objects detected above threshold ${data.threshold}.` : "No predictions.";
    out.appendChild(p);
  }
  const max = Math.max(...data.predictions.map((p) => p.score), 1e-9);
  for (const p of data.predictions) {
    const row = document.createElement("div");
    row.className = "bar-row" + (p.score < data.threshold ? " below" : "");
    const label = document.createElement("span");
    label.className = "bar-label";
    label.textContent = p.count > 1 ? `${p.label} ×${p.count}` : p.label;
    label.title = p.label;
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("span");
    // SigLIP scores are independent sigmoids and often small, so scale bars to the top score
    fill.style.width = `${(100 * p.score) / (data.kind === "zero-shot" ? max : 1)}%`;
    bar.appendChild(fill);
    const score = document.createElement("span");
    score.className = "bar-score";
    score.textContent = p.score.toFixed(3);
    row.append(label, bar, score);
    out.appendChild(row);
  }
  drawDetections(data.detections || [], data.image_size);
}

async function runYoloOverlay(signal) {
  const spec = visionModels.find((m) => m.kind === "detect");
  const strip = el.detectStrip;
  strip.hidden = false;
  strip.classList.remove("error");
  if (!spec) {
    strip.classList.add("error");
    strip.textContent = "Object detector: vision service unavailable";
    return;
  }
  strip.textContent = spec.loaded ? `${spec.id}: detecting…` : `${spec.id}: loading model…`;
  const file = currentFile;
  try {
    const r = await fetch("/api/vision/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      body: JSON.stringify({ model: spec.id, image_data_url: imageDataUrl, top_k: 20 }),
    });
    const data = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
    if (!r.ok) throw new Error(typeof data.detail === "string" ? data.detail : `HTTP ${r.status}`);
    spec.loaded = true;
    if (currentFile !== file) return;
    drawDetections(data.detections || [], data.image_size);
    const name = document.createElement("strong");
    name.textContent = spec.id;
    const found = data.predictions.length
      ? data.predictions.map((p) => `${p.label}${p.count > 1 ? ` ×${p.count}` : ""} ${p.score.toFixed(2)}`).join(" · ")
      : "no objects";
    strip.replaceChildren(name, `: ${found} · ${data.timing_ms.inference} ms`);
  } catch (e) {
    if (e.name === "AbortError") {
      strip.hidden = true;
      return;
    }
    strip.classList.add("error");
    strip.textContent = `Object detector: ${e.message}`;
  }
}

function drawDetections(detections, size) {
  el.overlay.replaceChildren();
  if (!detections.length || !size) return;
  const [w, h] = size;
  const stroke = Math.max(2, w / 300);
  const font = Math.max(12, w / 45);
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.box;
    const g = document.createElementNS(SVG_NS, "g");
    const rect = document.createElementNS(SVG_NS, "rect");
    Object.entries({ x: x1, y: y1, width: x2 - x1, height: y2 - y1, "stroke-width": stroke })
      .forEach(([k, v]) => rect.setAttribute(k, v));
    const caption = `${d.label} ${d.score.toFixed(2)}`;
    const tagH = font * 1.35;
    const tagY = y1 - tagH < 0 ? y1 : y1 - tagH;
    const tag = document.createElementNS(SVG_NS, "rect");
    tag.setAttribute("class", "tag");
    Object.entries({ x: x1, y: tagY, width: caption.length * font * 0.58 + font * 0.6, height: tagH })
      .forEach(([k, v]) => tag.setAttribute(k, v));
    const text = document.createElementNS(SVG_NS, "text");
    Object.entries({ x: x1 + font * 0.3, y: tagY + font * 1.05, "font-size": font })
      .forEach(([k, v]) => text.setAttribute(k, v));
    text.textContent = caption;
    g.append(rect, tag, text);
    el.overlay.appendChild(g);
  }
}

// ---------- events ----------
el.runBtn.addEventListener("click", run);
el.backend.addEventListener("change", () => { setBackend(el.backend.value); saveSettings(); });
el.refreshModels.addEventListener("click", () => refreshModels());
el.refreshVision.addEventListener("click", () => refreshVisionModels());
el.visionModel.addEventListener("change", () => { updateVisionUi(); clearOutputs(); });
el.baseUrl.addEventListener("change", () => { saveSettings(); refreshModels(); });
el.apiKey.addEventListener("change", () => { saveSettings(); refreshModels(); });
[el.modelSelect, el.modelText, el.temperature, el.maxTokens, el.jsonMode, el.withYolo, el.visionModel, el.topK,
  el.threshold, el.autoRun, el.systemPrompt, el.userPrompt, el.labels, el.labelTemplate]
  .forEach((x) => x.addEventListener("change", saveSettings));
el.resetPrompt.addEventListener("click", () => {
  el.systemPrompt.value = defaults.system_prompt || "";
  el.userPrompt.value = defaults.user_prompt || "";
  saveSettings();
});
el.resetLabels.addEventListener("click", () => {
  el.labels.value = visionDefaults.default_labels.join("\n");
  el.labelTemplate.value = "";
  saveSettings();
});
el.withYolo.addEventListener("change", () => {
  if (!el.withYolo.checked) {
    el.overlay.replaceChildren();
    el.detectStrip.hidden = true;
  }
});
el.exifFilter.addEventListener("input", renderExifTable);
el.toggleExif.addEventListener("click", () => {
  const collapsed = el.exifPanel.classList.toggle("collapsed");
  el.toggleExif.textContent = collapsed ? "show" : "hide";
  saveSettings();
});
el.copyExif.addEventListener("click", async () => {
  if (!exifData) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(exifData.groups, null, 2));
    setStatus("Metadata copied as JSON.");
  } catch {
    setStatus("Clipboard not available.", "error");
  }
});
el.outputPanel.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showOutputView(t.dataset.view)));
el.exportImg.addEventListener("click", () => {
  const source = isVision() ? `vision: ${el.visionModel.value}` : `llm: ${currentModel() || "default model"}`;
  exportImage(el.jsonOutput.textContent, source);
});
el.exportOverwrite.forEach((box) => box.addEventListener("change", () => {
  el.exportOverwrite.forEach((b) => { b.checked = box.checked; });
  saveSettings();
}));
el.copyJson.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(el.jsonOutput.textContent);
    setStatus("JSON copied.");
  } catch {
    setStatus("Clipboard not available.", "error");
  }
});
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    if (document.body.dataset.app === "video") videoRun(); // video.js
    else run();
  }
});

// ---------- init ----------
// Resolves once saved settings are applied, before the model lists load (tour.js waits for it)
let settingsApplied;
const settingsReady = new Promise((resolve) => (settingsApplied = resolve));

(async function init() {
  try {
    defaults = await (await fetch("/api/config")).json();
  } catch {
    defaults = {};
  }
  const s = loadSettings();
  el.baseUrl.value = s.baseUrl || defaults.base_url || "";
  el.apiKey.value = s.apiKey || "";
  el.temperature.value = s.temperature ?? 0.2;
  el.maxTokens.value = s.maxTokens ?? 1024;
  el.jsonMode.checked = !!s.jsonMode;
  el.withYolo.checked = !!s.withYolo;
  el.topK.value = s.topK ?? 5;
  el.threshold.value = s.threshold ?? "";
  el.autoRun.checked = !!s.autoRun;
  el.exportOverwrite.forEach((b) => { b.checked = !!s.exportOverwrite; });
  if (s.exifCollapsed) {
    el.exifPanel.classList.add("collapsed");
    el.toggleExif.textContent = "show";
  }
  el.systemPrompt.value = s.systemPrompt ?? defaults.system_prompt ?? "";
  el.userPrompt.value = s.userPrompt ?? defaults.user_prompt ?? "";
  el.labels.value = s.labels ?? "";
  el.labelTemplate.value = s.labelTemplate ?? "";

  const backend = s.backend === "vision" ? "vision" : "llm";
  el.backend.value = backend;
  document.body.dataset.backend = backend;
  settingsApplied();
  // Load both lists so switching backends is instant; the active one sets the status line last
  if (backend === "vision") {
    await refreshModels(s.model);
    await refreshVisionModels(s.visionModel);
  } else {
    await refreshVisionModels(s.visionModel);
    await refreshModels(s.model);
  }
})();

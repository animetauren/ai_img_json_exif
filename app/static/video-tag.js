"use strict";
// Tag scenes: the video side's Batch tab. Runs a model or workflow on a frame of every scene of the
// result shown on the timeline and saves each scene's JSON on the server beside the scenes (tags in
// video/server.py), so the timeline, the Scenes list and Export scenes carry it.
// It uses the Batch tab's own controls rather than copies: Run with and the model or workflow
// settings in the header, and the prompt or workflow editor and output panels on the right
// (the "tag-shared" parts of index.html). The scene grid takes the photo grid's place.
// Shares globals with app.js (el, $, isVision, currentModel, computeLlm, computeVision, showRecord,
// showOutputView, clearOutputs), workflows.js (h, wf, syncKind, syncFromJson, computeWorkflow,
// workflowView), batch-view.js (parseQuery, searchEntry, entryMatches, sortIndices, groupIndices,
// fillFieldSelect, leaves), batch.js (b, configKey, configLabel, recordResult, recordOk,
// transientError, TRANSIENT_RE, RETRY_DELAYS, abortableSleep, batchEta, formatDuration,
// onConfigChanged) and video.js (vid, ve, vFetch, vStatus, fmtTime, fmtDur, thumbTime, activeResult,
// sceneAt...). Loads after video.js.

const vt = {
  tabs: document.querySelectorAll(".tab[data-vmode]"),
  rerun: $("vtRerun"),
  run: $("vtRun"),
  progress: $("vtProgress"),
  progressBar: $("vtProgressBar"),
  progressText: $("vtProgressText"),
  resultLabel: $("vtResultLabel"),
  summary: $("vtSummary"),
  back: $("vtBack"),
  prev: $("vtPrev"),
  next: $("vtNext"),
  focusName: $("vtFocusName"),
  play: $("vtPlay"),
  timeline: $("vtTimeline"),
  search: $("vtSearch"),
  helpBtn: $("vtSearchHelp"),
  help: $("vtHelp"),
  status: $("vtStatus"),
  sort: $("vtSort"),
  dir: $("vtSortDir"),
  group: $("vtGroup"),
  empty: $("vtEmpty"),
  tiles: $("vtTiles"),
  frame: $("vtFrame"),
};

const tag = {
  at: null,           // { videoId, key, config } the entries belong to
  for: "",            // the same as one string, to compare runs against
  entries: new Map(), // scene index -> saved entry { ok, result, error, source, at }
  loadSeq: 0,
  resultFor: "",      // "<video>/<result key>" shown
  tiles: [],          // one tile per scene of the shown result, built on first view
  tilesFor: "",
  clones: new Map(),  // scene index -> extra tiles, when grouping puts a scene in several groups
  order: [],          // scene indices the grid shows, each once, in display order
  fieldsStale: true,  // the sort and group menus may not list every result field yet
  focus: -1,          // scene open in the focus view; -1 shows the grid
  run: null,          // the tagging run: { for, ctrl, current, waiting, lastError }
  pending: null,      // scenes left when a run stopped, so the next one resumes: { for, rerun, items }
  hover: -1,
  panelsUsed: false,  // the shared output panels show scenes, so the image side must redraw its own
};

const tagKey = (videoId, key, config) => `${videoId}\n${key}\n${config}`;
const tagUrl = (at, scene = null) =>
  `/api/video/${at.videoId}/scenes/${at.key}/tags${scene === null ? "" : `/${scene}`}?${new URLSearchParams({ config: at.config })}`;

// What tags are saved under: the Batch tab's keys and labels, for Run with's choice
const tagConfig = () => configKey(b.kind.value);
const tagLabel = () => configLabel(b.kind.value);
const panelsShown = () => document.body.dataset.app === "video" && document.body.dataset.vmode === "tag";

// ---------- Scenes / Tag scenes ----------
function setVideoMode(mode) {
  document.body.dataset.vmode = mode;
  vt.tabs.forEach((t) => t.classList.toggle("active", t.dataset.vmode === mode));
  vid.ui.mode = mode;
  saveVideoUi(); // video.js
  syncKind(); // workflows.js: Tag scenes shows the settings Run with picks
  if (mode === "tag") {
    ve.player.pause();
    showTagPanels();
  } else {
    requestAnimationFrame(drawAll); // video.js: the canvases had no size while hidden
  }
  updateTagButtons();
}

vt.tabs.forEach((t) => t.addEventListener("click", () => setVideoMode(t.dataset.vmode)));

// The output panels are the image side's: take them over for the scene grid
function showTagPanels() {
  tag.panelsUsed = true;
  clearOutputs(); // app.js
  renderTagView();
  if (tag.focus >= 0) showStoredTag(tag.focus);
}

// Called by setApp (video.js) when the video side is shown again: the settings may have changed
function onTagShown() {
  syncKind();
  if (tagConfig() !== tag.at?.config) loadTags();
  if (panelsShown()) showTagPanels();
}

// Called by setApp (video.js) when the image side is shown: give it back its kind and panels
function onImageShown() {
  syncKind();
  if (!tag.panelsUsed) return;
  tag.panelsUsed = false;
  clearOutputs();
  onConfigChanged(); // batch.js: the photo grid catches up with the settings, and a focused photo shows its result again
}

// Called by batch.js when Run with, the model or the workflow changes on the video side, and by
// app.js when the model lists arrive
function onTagConfigChanged() {
  if (!vid.loaded || tagConfig() === tag.at?.config) return;
  if (panelsShown() && !tag.run) clearOutputs();
  loadTags();
}

function onModelListChanged() {
  onTagConfigChanged();
}

// ---------- saved tags ----------
// Called by video.js whenever the result shown changes (or none is shown)
function onTagResult() {
  const r = activeResult();
  const id = r ? `${vid.id}/${r.key}` : "";
  if (id === tag.resultFor) return;
  tag.resultFor = id;
  setTagFocus(-1);
  loadTags();
}

async function loadTags() {
  const r = activeResult();
  const config = tagConfig();
  const seq = ++tag.loadSeq;
  const at = r && config ? { videoId: vid.id, key: r.key, config } : null;
  const forKey = at ? tagKey(at.videoId, at.key, at.config) : "";
  if (forKey !== tag.for) {
    // Show nothing rather than another result's or model's tags while these load
    tag.at = at;
    tag.for = forKey;
    tag.entries = new Map();
    tagsChanged();
  }
  if (!at) return;
  let data;
  try {
    data = await vFetch(tagUrl(at));
  } catch (e) {
    if (seq === tag.loadSeq) vStatus(`Could not load tags: ${e.message}`, "error");
    return;
  }
  if (seq !== tag.loadSeq) return;
  tag.entries = new Map(Object.entries(data.scenes || {}).map(([i, e]) => [Number(i), e]));
  tagsChanged();
  if (tag.focus >= 0) showStoredTag(tag.focus);
}

// Everything that shows tags: the grid, both timelines and the Scenes list
function tagsChanged() {
  tag.fieldsStale = true;
  if (document.body.dataset.vmode === "tag") renderTagView();
  refreshSceneTags();
  drawTimeline(); // video.js
}

function sceneState(i) {
  const run = tag.run;
  if (run && run.for === tag.for) {
    if (run.current === i) return "running";
    if (run.waiting === i) return "waiting";
  }
  const e = tag.entries.get(i);
  return !e ? "new" : e.ok ? "done" : "failed";
}

// One line for a scene's tags: its description (the last one, as a workflow's final LLM step
// comes last), else its category and tags: the keys export maps for photos (app/metadata.py)
function tagSummary(json) {
  let text = "";
  const words = [];
  const walk = (node) => {
    if (Array.isArray(node)) node.forEach(walk);
    else if (node && typeof node === "object") {
      for (const [k, v] of Object.entries(node)) {
        const key = k.toLowerCase();
        if (["description", "caption", "summary"].includes(key) && typeof v === "string" && v.trim()) text = v.trim();
        else if (["category", "tags", "keywords", "labels"].includes(key) && (typeof v === "string" || Array.isArray(v))) {
          for (const x of [v].flat()) if (typeof x === "string" || typeof x === "number") words.push(String(x));
        } else walk(v);
      }
    }
  };
  walk(json);
  return text || [...new Set(words)].slice(0, 10).join(", ");
}

function entryText(e) {
  if (!e) return "";
  return e.ok ? tagSummary(e.result) || "tagged" : `failed: ${e.error || "no result"}`;
}

// ---------- the Scenes view: tags under each scene and on the timeline ----------
const tagsShown = () => !!tag.at && tag.at.videoId === vid.id && tag.at.key === vid.active;

function sceneTagLine(i) {
  const e = tagsShown() ? tag.entries.get(i) : null;
  if (!e) return null;
  return h("span", {
    class: "v-scene-tag meta" + (e.ok ? "" : " failed"),
    title: e.ok ? JSON.stringify(e.result, null, 2) : e.error,
  }, entryText(e));
}

function refreshSceneTags(only = null) {
  for (const row of ve.scenes.querySelectorAll(".v-scene")) {
    const i = Number(row.dataset.i);
    if (only !== null && i !== only) continue;
    const text = row.querySelector(".v-scene-text");
    text.querySelector(".v-scene-tag")?.remove();
    const line = sceneTagLine(i);
    if (line) text.append(line);
  }
}

// A thin strip under the main timeline's scene blocks: green tagged, red failed
function drawTagMarks(ctx, scenes, dur, w, y) {
  if (!tagsShown() || !tag.entries.size) return;
  const ok = cssVar("--ok");
  const bad = cssVar("--error");
  for (const [i, e] of tag.entries) {
    const s = scenes[i];
    if (!s) continue;
    ctx.fillStyle = e.ok ? ok : bad;
    ctx.fillRect((s.start / dur) * w, y, Math.max(((s.end - s.start) / dur) * w, 0.5), 3);
  }
}

function tagHoverText(i) {
  const e = tagsShown() ? tag.entries.get(i) : null;
  if (!e) return "";
  const text = entryText(e);
  return ` · ${text.length > 90 ? text.slice(0, 89) + "…" : text}`;
}

// The tags Export scenes includes: Run with's model or workflow, when it has tagged this result
function tagExportConfig() {
  return tagsShown() && [...tag.entries.values()].some((e) => e.ok) ? tag.at.config : "";
}

// ---------- the grid ----------
function renderTagView() {
  const r = activeResult();
  vt.resultLabel.textContent = r ? `· ${r.label} · ${paramsText(r.detector, r.params, true)}` : ""; // video.js
  vt.empty.hidden = !!r;
  if (!r) {
    tag.tiles = [];
    tag.tilesFor = "";
    tag.order = [];
    tag.clones = new Map();
    vt.tiles.replaceChildren();
    vt.empty.replaceChildren(
      h("strong", {}, vid.id ? "No scenes to tag yet" : "No video chosen"), h("br"),
      vid.id ? "Detect scenes in the Scenes tab; the result shown there is the one tagged here."
        : "Choose a video and detect its scenes, then tag them here.");
    updateTagSummary();
    drawTagTimeline();
    updateTagButtons();
    return;
  }
  if (tag.tilesFor !== `${vid.id}/${r.key}`) buildTiles(r);
  tag.tiles.forEach((_, i) => updateTile(i));
  applyTagView();
}

function buildTiles(r) {
  tag.tilesFor = `${vid.id}/${r.key}`;
  tag.tiles = r.scenes.map((s, i) => h("button", { type: "button", class: "tile", dataset: { i } },
    h("img", { loading: "lazy", decoding: "async", alt: "", src: `/api/video/${vid.id}/thumb?t=${thumbTime(s).toFixed(3)}` }),
    h("span", { class: "tile-name" }, `${i + 1} · ${fmtTime(s.start)} · ${fmtDur(s.end - s.start)}`),
    h("span", { class: "vt-tile-text" })));
}

const tilesOf = (i) => [tag.tiles[i], ...(tag.clones.get(i) || [])].filter(Boolean);

function updateTile(i) {
  const state = sceneState(i);
  const text = entryText(tag.entries.get(i));
  for (const t of tilesOf(i)) {
    t.dataset.state = state;
    t.lastChild.textContent = text;
    t.title = `Scene ${i + 1}${text ? `: ${text}` : ""}`;
  }
}

function refreshTileStates() {
  tag.tiles.forEach((_, i) => tilesOf(i).forEach((t) => (t.dataset.state = sceneState(i))));
  drawTagTimeline();
}

// Search, sort and group fields of a scene: its result JSON (as photo results are searched in
// batch-view.js), plus scene, start, end, duration, error and status
function sceneSearch(i) {
  const e = tag.entries.get(i);
  if (e?.search) return e.search;
  const s = activeResult().scenes[i];
  const entry = searchEntry(e?.result ?? null, (add) => {
    add("scene", i + 1);
    add("start", Math.round(s.start * 100) / 100);
    add("end", Math.round(s.end * 100) / 100);
    add("duration", Math.round((s.end - s.start) * 100) / 100);
    if (e?.error) add("error", e.error);
  });
  if (e) e.search = entry;
  return entry;
}

const sceneValues = (i, field) => (field === "status" ? [sceneState(i)] : sceneSearch(i).fields.get(field));

function rebuildTagFields() {
  tag.fieldsStale = false;
  const paths = new Set();
  for (const e of tag.entries.values()) for (const [path] of leaves(e.result)) paths.add(path);
  const fixed = [["status", "Status"], ["duration", "Duration"]];
  fillFieldSelect(vt.sort, "Scene order", fixed, paths); // batch-view.js
  fillFieldSelect(vt.group, "No grouping", fixed, paths);
}

// The Batch tab's search, status filter, sort and group, over the scenes (applyView in batch-view.js)
function applyTagView() {
  const r = activeResult();
  if (!r || !tag.tiles.length) return;
  if (tag.fieldsStale) rebuildTagFields();
  const terms = parseQuery(vt.search.value);
  const status = vt.status.value;
  let shown = r.scenes.map((_, i) => i)
    .filter((i) => (!status || sceneState(i) === status) && terms.every((t) => entryMatches(sceneSearch(i), (f) => sceneValues(i, f), t) !== t.neg));
  shown = sortIndices(shown, vt.sort.value, vt.dir.dataset.dir, sceneValues);
  renderSceneTiles(groupIndices(shown, vt.group.value, sceneValues), r.scenes.length);
  updateTagSummary();
  drawTagTimeline();
  updateTagButtons();
}

// Like renderTiles in batch-view.js: group headings, and a copy of the tile for each extra group
function renderSceneTiles(groups, total) {
  const nodes = [];
  const used = new Set();
  tag.order = [];
  tag.clones = new Map();
  for (const g of groups) {
    if (g.label !== null) {
      nodes.push(h("div", { class: "tile-group" }, h("strong", {}, g.label), h("span", { class: "meta" }, ` ${g.indices.length}`)));
    }
    for (const i of g.indices) {
      if (!used.has(i)) {
        used.add(i);
        tag.order.push(i);
        nodes.push(tag.tiles[i]);
      } else {
        const clone = tag.tiles[i].cloneNode(true); // clicks are handled on the grid, by data-i
        if (!tag.clones.has(i)) tag.clones.set(i, []);
        tag.clones.get(i).push(clone);
        nodes.push(clone);
      }
    }
  }
  if (!tag.order.length && total) nodes.push(h("div", { class: "tile-group empty" }, "No scenes match."));
  vt.tiles.replaceChildren(...nodes);
}

function updateTagSummary() {
  const r = activeResult();
  if (!r) {
    vt.summary.textContent = "";
    return;
  }
  let done = 0;
  let failed = 0;
  for (const e of tag.entries.values()) e.ok ? done++ : failed++;
  vt.summary.textContent = `${r.scenes.length} scenes · ${done} done${failed ? ` · ${failed} failed` : ""}` +
    ` with ${tagLabel()}` + (tag.order.length < r.scenes.length ? ` · showing ${tag.order.length}` : "");
}

function setTagSortDir(dir) {
  vt.dir.dataset.dir = dir;
  vt.dir.textContent = dir === "desc" ? "↓" : "↑";
  vt.dir.title = dir === "desc" ? "Descending (click for ascending)" : "Ascending (click for descending)";
}

let tagSearchTimer = null;
vt.search.addEventListener("input", () => {
  clearTimeout(tagSearchTimer);
  tagSearchTimer = setTimeout(() => {
    applyTagView();
    saveTagUi();
  }, 150);
});
[vt.status, vt.sort, vt.group].forEach((x) => x.addEventListener("change", () => {
  applyTagView();
  saveTagUi();
}));
vt.dir.addEventListener("click", () => {
  setTagSortDir(vt.dir.dataset.dir === "desc" ? "asc" : "desc");
  applyTagView();
  saveTagUi();
});
vt.helpBtn.addEventListener("click", () => (vt.help.hidden = !vt.help.hidden));
vt.rerun.addEventListener("change", saveTagUi);
vt.tiles.addEventListener("click", (e) => {
  const tile = e.target.closest(".tile");
  if (tile) setTagFocus(Number(tile.dataset.i));
});

function saveTagUi() {
  Object.assign(vid.ui, {
    tagRerun: vt.rerun.checked, tagSearch: vt.search.value, tagStatus: vt.status.value,
    tagSort: vt.sort.value, tagDir: vt.dir.dataset.dir, tagGroup: vt.group.value,
  });
  saveVideoUi();
}

// ---------- the scene strip ----------
// Every scene of the result across the whole video, coloured as the tiles are; click one to open it
function drawTagTimeline() {
  if (!panelsShown()) return;
  const { ctx, w, h: hgt } = sizeCanvas(vt.timeline); // video.js
  const r = activeResult();
  const dur = r?.duration;
  if (!r || !w || !dur) return;
  const colors = {
    done: cssVar("--ok"), failed: cssVar("--error"), running: cssVar("--accent"), waiting: "#f59e0b", new: cssVar("--tile-new"),
  };
  const shown = new Set(tag.order);
  r.scenes.forEach((s, i) => {
    const x0 = (s.start / dur) * w;
    const width = ((s.end - s.start) / dur) * w;
    ctx.fillStyle = colors[sceneState(i)];
    // Alternate shades so neighbouring scenes stay apart; filtered-out scenes fade
    ctx.globalAlpha = (shown.has(i) ? 1 : 0.3) * (i % 2 ? 0.9 : 0.6);
    ctx.fillRect(x0, 0, Math.max(width > 3 ? width - 1 : width, 0.5), hgt);
  });
  ctx.globalAlpha = 1;
  for (const [i, color] of [[tag.hover, cssVar("--muted")], [tag.focus, cssVar("--text")]]) {
    const s = r.scenes[i];
    if (!s) continue;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect((s.start / dur) * w + 1, 1, Math.max(((s.end - s.start) / dur) * w - 2, 2), hgt - 2);
  }
}

function stripScene(e) {
  const rect = vt.timeline.getBoundingClientRect();
  const t = ((e.clientX - rect.left) / rect.width) * videoDuration(); // video.js
  return sceneAt(t);
}

vt.timeline.addEventListener("pointermove", (e) => {
  const i = stripScene(e);
  if (i === tag.hover) return;
  tag.hover = i;
  const s = activeResult()?.scenes[i];
  if (s) vt.summary.textContent = `Scene ${i + 1} · ${fmtTime(s.start)}${tagHoverText(i)}`;
  drawTagTimeline();
});
vt.timeline.addEventListener("pointerleave", () => {
  tag.hover = -1;
  updateTagSummary();
  drawTagTimeline();
});
vt.timeline.addEventListener("click", (e) => {
  const i = stripScene(e);
  if (i >= 0) setTagFocus(i);
});
new ResizeObserver(() => requestAnimationFrame(drawTagTimeline)).observe(vt.timeline);

// ---------- one scene ----------
function setTagFocus(i) {
  const left = tag.focus;
  tag.focus = i;
  const r = activeResult();
  const s = r?.scenes[i];
  if (s) document.body.dataset.vtfocus = "1";
  else {
    tag.focus = -1;
    delete document.body.dataset.vtfocus;
  }
  updateTagButtons();
  drawTagTimeline();
  if (!s) {
    vt.frame.removeAttribute("src");
    if (panelsShown()) clearOutputs();
    // The grid lost its scroll position while hidden: bring back the scene just left
    if (left >= 0 && tag.tiles[left]?.isConnected) tag.tiles[left].scrollIntoView({ block: "nearest" });
    return;
  }
  const pos = tag.order.indexOf(i);
  vt.focusName.textContent = `Scene ${i + 1} of ${r.scenes.length} · ${fmtTime(s.start)} – ${fmtTime(s.end)} · ${fmtDur(s.end - s.start)}`;
  vt.prev.disabled = pos <= 0;
  vt.next.disabled = pos < 0 || pos === tag.order.length - 1;
  vt.frame.src = frameUrl(vid.id, s); // the picture the model sees
  showStoredTag(i);
}

function stepTagFocus(delta) {
  const next = tag.order[tag.order.indexOf(tag.focus) + delta];
  if (next !== undefined) setTagFocus(next);
}

vt.back.addEventListener("click", () => setTagFocus(-1));
vt.prev.addEventListener("click", () => stepTagFocus(-1));
vt.next.addEventListener("click", () => stepTagFocus(1));
vt.play.addEventListener("click", () => {
  const i = tag.focus;
  setVideoMode("scenes");
  scrollToScene(i); // video.js
  if (vid.loop?.i !== i) playScene(i);
  else ve.player.play().catch(() => {});
});
document.addEventListener("keydown", (e) => {
  if (!panelsShown() || tag.focus < 0) return;
  if (e.target instanceof Element && e.target.closest("input, textarea, select")) return;
  if (e.key === "Escape") setTagFocus(-1);
  else if (e.key === "ArrowLeft") stepTagFocus(-1);
  else if (e.key === "ArrowRight") stepTagFocus(1);
});

// The open scene's saved result in the output panels, as the Batch tab shows a focused photo's
async function showStoredTag(i) {
  if (!panelsShown()) return;
  const at = tag.at;
  if (tag.run?.for === tag.for && tag.run.current === i) return; // its run is showing, or will be
  clearOutputs();
  if (!tag.entries.has(i)) {
    vStatus(`Scene ${i + 1} isn't tagged with ${tagLabel()} yet. Press Tag scene to try these settings on it.`);
    return;
  }
  let data;
  try {
    data = await vFetch(tagUrl(at, i));
  } catch (e) {
    if (tag.focus === i && tag.at === at) vStatus(`Could not load scene ${i + 1}'s result: ${e.message}`, "error");
    return;
  }
  if (tag.focus !== i || tag.at !== at || !panelsShown()) return;
  showTag(data, data.record);
  vStatus(`Scene ${i + 1}: saved result from ${new Date(data.at * 1000).toLocaleString()} (${data.source}).`, data.ok ? "ok" : "error");
}

function showTag(entry, record) {
  if (!panelsShown()) return;
  clearOutputs();
  if (record?.kind) showRecord(record); // app.js
  else el.jsonOutput.textContent = JSON.stringify(entry.result, null, 2);
}

// ---------- running ----------
const frameUrl = (videoId, s) => `/api/video/${videoId}/frame?t=${thumbTime(s).toFixed(3)}`;

// The scene's frame from the original video, as a data URL for the models
async function frameDataUrl(videoId, s, signal) {
  const r = await fetch(frameUrl(videoId, s), { signal });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    const detail = typeof d.detail === "string" ? d.detail : "";
    // Gateway errors keep their status, so they count as the server being away (TRANSIENT_RE)
    throw Object.assign(new Error(r.status > 500 || !detail ? `HTTP ${r.status}${detail ? `: ${detail}` : ""}` : detail), { status: r.status });
  }
  const blob = await r.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

// What workflow prompts get as {{exif.*}} for a scene: the video's name, its recording date moved
// on to the frame's time (in EXIF's format), where the scene sits in the video, and with a
// transcript shown, what is said in it
function sceneContext(ctx, i) {
  const s = ctx.result.scenes[i];
  const t = thumbTime(s);
  const out = {
    FileName: ctx.name, Scene: i + 1, SceneCount: ctx.result.scenes.length,
    SceneStart: fmtTime(s.start), SceneEnd: fmtTime(s.end), SceneDuration: Math.round((s.end - s.start) * 100) / 100,
    FrameTime: fmtTime(t),
  };
  if (ctx.speech) out.Speech = ctx.speech[i] || "(no speech)";
  const recorded = Date.parse(ctx.recorded || "");
  if (isFinite(recorded)) {
    const d = new Date(recorded + t * 1000).toISOString();
    out.DateTimeOriginal = out.CreateDate = d.slice(0, 19).replace("T", " ").replace(/-/g, ":");
  }
  return out;
}

// The settings to run right now. Read per scene, as the Batch tab does per photo, so a setting
// changed mid-run is saved under the right model or workflow.
function runConfig() {
  const config = tagConfig();
  if (!config) return { error: "Choose a workflow first." };
  const cfg = { kind: b.kind.value === "workflow" ? "workflow" : isVision() ? "vision" : "llm", config, label: tagLabel(), def: null };
  if (cfg.kind === "workflow") {
    if (!syncFromJson()) return { error: "The workflow's JSON has an error; fix it in the editor." };
    if (!wf.current.steps.length) return { error: "This workflow has no steps yet." };
    cfg.def = structuredClone(wf.current); // with any unsaved edits, as the Batch tab runs it
  }
  return cfg;
}

function computeTag(cfg, signal, dataUrl, exif, onEvent) {
  if (cfg.kind === "workflow") return computeWorkflow(signal, { def: cfg.def, dataUrl, exif }, onEvent); // workflows.js
  if (cfg.kind === "vision") return computeVision(signal, dataUrl); // app.js
  return computeLlm(signal, dataUrl, onEvent);
}

// A run that never reached the model, shaped like a real one so the panels can show it
function failedRecord(cfg, error) {
  if (cfg.kind === "workflow") {
    return { kind: "workflow", workflow: { id: cfg.def.id, name: cfg.def.name }, steps: cfg.def.steps, events: [], result: null, error, aborted: false };
  }
  if (cfg.kind === "vision") return { kind: "vision", data: null, error, aborted: false };
  return { kind: "llm", text: "", reasoning: "", error, aborted: false, meta: "", json: null, jsonError: null };
}

// Why a run counts as failed, for the entry and the tile
function recordError(record) {
  if (recordOk(record)) return null;
  const steps = (record.events || []).filter((e) => e.type === "step_error").map((e) => `${e.id}: ${e.message}`);
  return record.error || steps.join("; ") || (record.jsonError && `Not valid JSON: ${record.jsonError}`) || "No result.";
}

// Streams a run into the output panels while its scene stays open (as runActive does for a photo)
function liveView(cfg, forKey, i) {
  const onScreen = () => tag.focus === i && tag.for === forKey && panelsShown();
  if (!onScreen()) return null;
  clearOutputs();
  if (cfg.kind === "workflow") {
    const apply = workflowView(cfg.def.steps); // workflows.js: the Steps and Result panels
    return (ev) => onScreen() && apply(ev);
  }
  if (cfg.kind !== "llm") return null;
  showOutputView("raw"); // app.js
  const text = el.rawOutput.appendChild(document.createTextNode(""));
  let reasoning = null;
  return (ev) => {
    if (!onScreen() || !text.isConnected) return;
    if (ev.type === "token") text.data += ev.text;
    else if (ev.type === "reasoning") (reasoning ??= el.rawOutput.insertBefore(h("span", { class: "reasoning" }), text)).append(ev.text);
    el.rawOutput.scrollTop = el.rawOutput.scrollHeight;
  };
}

// Tags one scene and saves it. Returns true/false for a usable result, null if stopped, "transient"
// if a server was unreachable (nothing saved), or "gone" if the video or its scenes were deleted or
// the settings can't run.
async function tagScene(ctx, i, signal, { focused = false } = {}) {
  const run = tag.run;
  const cfg = runConfig();
  if (cfg.error) {
    run.lastError = cfg.error;
    return "gone";
  }
  const forKey = tagKey(ctx.videoId, ctx.result.key, cfg.config);
  const s = ctx.result.scenes[i];
  run.for = forKey;
  run.current = i;
  refreshTileStates();
  let record = null;
  try {
    let dataUrl = null;
    try {
      dataUrl = await frameDataUrl(ctx.videoId, s, signal);
    } catch (e) {
      if (e.name === "AbortError") return null;
      if (e.status === 404 && /Unknown or expired video/.test(e.message)) {
        run.lastError = "the video was deleted";
        return "gone";
      }
      if (TRANSIENT_RE.test(e.message)) {
        run.lastError = e.message;
        return "transient";
      }
      record = failedRecord(cfg, `Could not read the scene's frame: ${e.message}`);
    }
    if (!record) {
      if (signal.aborted) return null;
      record = await computeTag(cfg, signal, dataUrl, sceneContext(ctx, i), focused ? liveView(cfg, forKey, i) : null);
    }
  } finally {
    run.current = -1;
    refreshTileStates();
  }
  if (!record || record.aborted) return null;
  const transient = transientError(record); // batch.js
  if (transient) {
    run.lastError = transient;
    return "transient";
  }
  record.at = Date.now();
  record.source = cfg.label;
  const ok = recordOk(record);
  let entry;
  try {
    entry = await vFetch(tagUrl({ videoId: ctx.videoId, key: ctx.result.key, config: cfg.config }, i), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: cfg.label, ok, result: recordResult(record), error: recordError(record), record }),
    });
  } catch (e) {
    run.lastError = e.status === 404 ? e.message : `could not save (${e.message})`;
    return e.status === 404 ? "gone" : "transient";
  }
  if (tag.for === forKey) {
    tag.entries.set(i, entry);
    tag.fieldsStale = true;
    updateTile(i);
    refreshSceneTags(i);
    scheduleDraw(); // video.js: the main timeline's strip
    updateTagSummary();
    // The open scene shows its own result; on the grid, the panels show the latest one finished
    if (tag.focus === i || tag.focus < 0) showTag(entry, record);
  }
  return ok;
}

async function runTagging() {
  if (tag.run) {
    tag.run.ctrl.abort();
    return;
  }
  if (vt.run.disabled) return; // Cmd/Ctrl+Enter comes here too
  const r = activeResult();
  if (!r) return;
  const check = runConfig();
  if (check.error) {
    vStatus(check.error, "error");
    return;
  }
  const ctx = {
    videoId: vid.id, result: r, name: vid.status?.name, recorded: vid.status?.info?.created,
    speech: speechForScenes(r.scenes), // video-speech.js
  };
  const ctrl = new AbortController();
  tag.run = { for: tagKey(vid.id, r.key, check.config), ctrl, current: -1, waiting: -1, lastError: null };
  saveSettings(); // app.js: as Run does on the image side
  updateTagButtons();
  try {
    if (tag.focus >= 0) await tagFocused(ctx, tag.focus, ctrl.signal);
    else await tagAll(ctx, ctrl.signal);
  } finally {
    tag.run = null;
    refreshTileStates();
    updateTagButtons();
    if (panelsShown()) applyTagView(); // new results can change what the filter, sort and groups show
  }
}

async function tagFocused(ctx, i, signal) {
  vStatus(`Tagging scene ${i + 1} with ${tagLabel()}…`);
  const ok = await tagScene(ctx, i, signal, { focused: true });
  if (ok === null) vStatus("Stopped.");
  else if (ok === "transient") vStatus(`${tagLabel()} is not responding (${tag.run.lastError}). Nothing was saved for this scene.`, "error");
  else if (ok === "gone") vStatus(`Stopped: ${tag.run.lastError}`, "error");
  else if (ok) vStatus(`Scene ${i + 1} done.`);
  else vStatus(`Scene ${i + 1} failed: ${tag.entries.get(i)?.error || "no result"}`, "error");
}

// Every scene shown (all of them with no filter), skipping done ones unless Re-run done is on.
// Like runBatch in batch.js: unreachable servers are retried, then the run pauses and resumes there.
async function tagAll(ctx, signal) {
  const shown = new Set(tag.order);
  const p = tag.pending;
  const resuming = p?.for === tag.run.for && p.rerun === vt.rerun.checked && p.items.some((i) => shown.has(i));
  const todo = resuming
    ? p.items.filter((i) => shown.has(i))
    : tag.order.filter((i) => vt.rerun.checked || tag.entries.get(i)?.ok !== true);
  const startedFor = tag.run.for;
  tag.pending = null;
  if (!todo.length) {
    vStatus(`Every scene ${shown.size < ctx.result.scenes.length ? "shown " : ""}already has a result for ${tagLabel()}. ` +
      'Tick "Re-run done" to run them again.');
    return;
  }
  const started = performance.now();
  const eta = batchEta(todo.length, vt); // batch.js
  let ran = 0;
  let failed = 0;
  let stoppedBy = null; // "transient" or "gone"
  try {
    for (const i of todo) {
      if (signal.aborted) break;
      vStatus(`${resuming && !ran ? "Resuming: " : ""}scene ${i + 1} (${ran + 1} of ${todo.length}) with ${tagLabel()}`);
      eta.start(ran);
      let ok = await tagScene(ctx, i, signal);
      const retried = ok === "transient";
      for (let attempt = 0; ok === "transient" && attempt < RETRY_DELAYS.length && !signal.aborted; attempt++) {
        vStatus(`Scene ${i + 1}: ${tag.run.lastError}. Retrying in ${RETRY_DELAYS[attempt] / 1000}s (${attempt + 1} of ${RETRY_DELAYS.length})…`, "error");
        tag.run.waiting = i;
        refreshTileStates();
        await abortableSleep(RETRY_DELAYS[attempt], signal);
        tag.run.waiting = -1;
        if (signal.aborted) break;
        ok = await tagScene(ctx, i, signal);
      }
      if (ok === "transient" || ok === "gone") {
        stoppedBy = ok;
        break;
      }
      if (ok === null) break;
      if (!retried) eta.finish(); // waiting time would skew the average
      ran++;
      if (!ok) failed++;
    }
  } finally {
    eta.stop();
  }
  if (ran < todo.length && stoppedBy !== "gone") tag.pending = { for: startedFor, rerun: vt.rerun.checked, items: todo.slice(ran) };
  const avg = eta.average();
  const summary = `${ran} of ${todo.length} scenes in ${formatDuration(performance.now() - started)}` +
    (avg ? ` (avg ${formatDuration(avg)} per scene)` : "") + (failed ? `, ${failed} failed (red)` : "") + ".";
  const button = tag.order.length < ctx.result.scenes.length ? "Tag shown" : "Tag all";
  if (stoppedBy === "gone") {
    vStatus(`Stopped: ${tag.run.lastError}. Ran ${summary}`, "error");
  } else if (stoppedBy) {
    vStatus(`Paused: ${tagLabel()} is not responding (${tag.run.lastError}). Finished scenes are saved; ` +
      `${todo.length - ran} remain. Fix the server, then press ${button} to resume. Ran ${summary}`, "error");
  } else if (signal.aborted) {
    vStatus(`Stopped after ${summary} Press ${button} to continue with the ${todo.length - ran} left.`, failed ? "error" : "ok");
  } else {
    vStatus(`Tagging done: ${summary}`, failed ? "error" : "ok");
  }
}

vt.run.addEventListener("click", runTagging);

function updateTagButtons() {
  if (tag.run) {
    vt.run.textContent = "Stop";
    vt.run.disabled = false;
    vt.run.title = "Stop tagging; finished scenes stay saved";
    return;
  }
  const r = activeResult();
  vt.run.textContent = tag.focus >= 0 ? "Tag scene" : r && tag.order.length < r.scenes.length ? "Tag shown" : "Tag all";
  vt.run.disabled = !r || vid.status?.state !== "ready" || (tag.focus < 0 && !tag.order.length);
  vt.run.title = "Run the model or workflow on each scene's frame and save its JSON with the scene (Cmd/Ctrl+Enter)";
}

// ---------- init ----------
// Called by initVideo (video.js), before the first video opens
function initTagging() {
  vt.tabs.forEach((t) => t.classList.toggle("active", t.dataset.vmode === document.body.dataset.vmode));
  const ui = vid.ui;
  vt.rerun.checked = !!ui.tagRerun;
  vt.search.value = ui.tagSearch || "";
  vt.status.value = ui.tagStatus || "";
  setTagSortDir(ui.tagDir === "desc" ? "desc" : "asc");
  rebuildTagFields();
  for (const [select, value] of [[vt.sort, ui.tagSort], [vt.group, ui.tagGroup]]) {
    if (!value) continue;
    if (![...select.options].some((o) => o.value === value)) select.append(h("option", { value }, value));
    select.value = value;
  }
  syncKind(); // workflows.js: also loads the workflows when Run with is Workflow
  if (panelsShown()) tag.panelsUsed = true;
}

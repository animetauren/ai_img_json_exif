"use strict";
// Video Classifier: uploads a video in chunks, has the video service prepare it, splits it into
// scenes with a choice of detectors, and exports the scene list with thumbnails. The video
// itself is never cut. The header's app switch
// flips between this and the image side. Shares globals with app.js ($, fillSelect),
// workflows.js (h) and batch.js (formatDuration). video-tag.js adds the Tag scenes view, and
// shows its tags in this one (onTagResult, sceneTagLine, drawTagMarks, ...). video-speech.js adds
// the Transcript panel and its rows under the timeline (onSpeechStatus, speechTracks, ...).

const APP_KEY = "llm-classifier-app";
const VIDEO_UI_KEY = "llm-classifier-video-ui";
const UPLOAD_RETRY_DELAYS = [1000, 2000, 5000, 10000, 20000, 30000, 60000];
const SCENE_PAGE = 150; // scene rows added at a time as the list scrolls
const POLL_MS = 1000;
const RESULT_CACHE = 20;

const ve = {
  appKind: $("appKind"),
  pick: $("vPick"),
  file: $("vFile"),
  recent: $("vRecent"),
  refresh: $("vRefresh"),
  detector: $("vDetector"),
  params: $("vParams"),
  exportBtn: $("vExport"),
  detect: $("vDetect"),
  status: $("vStatus"),
  stage: $("vStage"),
  hint: $("vHint"),
  player: $("vPlayer"),
  noPreview: $("vNoPreview"),
  busy: $("vBusy"),
  busyText: $("vBusyText"),
  busyBar: $("vBusyBar"),
  busyEta: $("vBusyEta"),
  cancel: $("vCancel"),
  timeline: $("vTimeline"),
  timelineMeta: $("vTimelineMeta"),
  trackLabel: $("vTrackLabel"),
  hover: $("vHover"),
  compare: $("vCompare"),
  scenes: $("vScenes"),
  sceneCount: $("vSceneCount"),
  saveCsv: $("vSaveCsv"),
  saveJson: $("vSaveJson"),
  info: $("vInfo"),
  del: $("vDelete"),
};

const vid = {
  loaded: false,
  detectors: [],
  ui: { detector: "transnetv2", params: {}, videoId: null },
  id: null, // the open video
  status: null, // its latest status from the server
  results: new Map(), // "<video>/<key>" -> scene result
  active: null, // key of the result shown
  upload: null, // { id, file, xhr, sent, size, startSent, started, cancelled }
  local: null, // { id, url, failed }: the chosen file itself, played until the proxy is ready
  pollTimer: null,
  loop: null, // { i, start, end }: the scene being looped
  rendered: 0, // scene rows in the list so far
  playing: -1,
  hoverX: null,
  shownErrors: new Set(),
  busyFrame: 0,
  scores: new Map(), // detector id -> Promise<Float32Array|null>, per-frame scores of the open video
  drag: null, // { result, detector, max, value, moved }: the threshold line being dragged
  preview: null, // { threshold, scenes }: cuts at the dragged threshold, before the server saves them
  dragEnded: false, // swallows the click that follows a drag
  curveGeom: null, // where drawCurve put the curve: { y, h, max, thresholdY }
  drawFrame: 0,
};

function vStatus(msg, kind = "ok") {
  ve.status.textContent = msg || "";
  ve.status.className = "status video-app " + kind;
}

// ---------- formatting ----------
function fmtTime(sec) {
  if (sec == null || !isFinite(sec)) return "–";
  const neg = sec < 0;
  sec = Math.abs(sec);
  const h = Math.floor(sec / 3600);
  const m = Math.floor(sec / 60) % 60;
  const s = (sec % 60).toFixed(2).padStart(5, "0");
  return (neg ? "-" : "") + (h ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`);
}

function fmtDur(sec) {
  return sec < 60 ? `${sec.toFixed(sec < 10 ? 2 : 1)} s` : formatDuration(sec * 1000); // batch.js
}

function fmtBytes(n) {
  if (n == null) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1000 && i < units.length - 1) {
    n /= 1000;
    i++;
  }
  return `${n.toFixed(n < 10 && i ? 1 : 0)} ${units[i]}`;
}

// ---------- requests ----------
async function vFetch(url, opts = {}) {
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const detail = typeof data.detail === "string" ? data.detail : `HTTP ${r.status}`;
    throw Object.assign(new Error(detail), { status: r.status, data });
  }
  return data;
}

const vPost = (url, body) => vFetch(url, {
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}),
});

function download(url, filename = "") {
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------- app switch ----------
function setApp(kind) {
  document.body.dataset.app = kind;
  ve.appKind.value = kind;
  document.title = kind === "video" ? "Video Classifier" : "LLM Image Classifier";
  try {
    localStorage.setItem(APP_KEY, kind);
  } catch { /* storage unavailable */ }
  if (kind === "video") {
    if (!vid.loaded) initVideo();
    else {
      requestAnimationFrame(drawAll);
      onTagShown(); // video-tag.js
    }
  } else {
    ve.player.pause();
    onImageShown(); // video-tag.js: the shared settings and panels go back to the image side
  }
}

ve.appKind.addEventListener("change", () => setApp(ve.appKind.value));

// ---------- settings ----------
function saveVideoUi() {
  try {
    localStorage.setItem(VIDEO_UI_KEY, JSON.stringify(vid.ui));
  } catch { /* storage unavailable */ }
}

function loadVideoUi() {
  try {
    Object.assign(vid.ui, JSON.parse(localStorage.getItem(VIDEO_UI_KEY)) || {});
  } catch { /* storage unavailable */ }
  vid.ui.params ||= {};
}

// ---------- detectors ----------
function currentDetector() {
  return vid.detectors.find((d) => d.id === ve.detector.value);
}

async function loadDetectors() {
  const data = await vFetch("/api/video/detectors");
  vid.detectors = data.detectors;
  ve.detector.replaceChildren(...vid.detectors.map((d) => h("option", {
    value: d.id, disabled: !d.available, title: d.note || d.description,
  }, d.available ? d.label : `${d.label} (not available yet)`)));
  const want = vid.detectors.find((d) => d.id === vid.ui.detector && d.available) || vid.detectors.find((d) => d.available);
  if (want) ve.detector.value = want.id;
  renderParams();
}

function renderParams() {
  const d = currentDetector();
  if (!d) return;
  const saved = vid.ui.params[d.id] || {};
  ve.params.replaceChildren(...d.params.map((p) => h("label", { title: p.help },
    p.label,
    h("input", {
      type: "number", min: String(p.min), max: String(p.max), step: String(p.step),
      value: saved[p.name] ?? p.default, dataset: { name: p.name },
    }))));
  ve.detector.title = d.description;
}

// Same clamping and rounding as Detector.normalize on the server, so results match by value
function currentParams(d = currentDetector()) {
  const out = {};
  for (const p of d?.params || []) {
    const input = ve.params.querySelector(`input[data-name="${p.name}"]`);
    let value = parseFloat(input?.value);
    if (!isFinite(value)) value = p.default;
    out[p.name] = Math.round(Math.min(Math.max(value, p.min), p.max) * 1e4) / 1e4;
  }
  return out;
}

function sameParams(a, b) {
  const keys = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  return [...keys].every((k) => Math.abs((a?.[k] ?? NaN) - (b?.[k] ?? NaN)) < 1e-6);
}

// The finished result for the detector and settings chosen in the header, if any
function selectedDetection() {
  const d = currentDetector();
  const params = currentParams();
  return (vid.status?.detections || []).find((x) => x.detector === d?.id && sameParams(x.params, params));
}

function onSelectionChanged() {
  vid.ui.detector = ve.detector.value;
  vid.ui.params[ve.detector.value] = currentParams();
  saveVideoUi();
  const det = selectedDetection();
  if (det) setActive(det.key);
  else showNoResult();
  updateButtons();
}

ve.detector.addEventListener("change", () => {
  renderParams();
  onSelectionChanged();
});
ve.params.addEventListener("change", onSelectionChanged);

// "Threshold 0.5 · Min scene 0.5 s", or with short, "thr 0.5 · min 0.5 s" for the narrow track labels
const SHORT_PARAM = { threshold: "thr", adaptive_threshold: "thr", min_scene: "min", min_content_val: "min Δ", fade_bias: "bias" };

function paramsText(detectorId, params, short = false) {
  const d = vid.detectors.find((x) => x.id === detectorId);
  return (d?.params || []).map((p) => {
    const name = short ? SHORT_PARAM[p.name] || p.name : p.label.replace(/ \(s\)$/, "");
    return `${name} ${params?.[p.name]}${p.name === "min_scene" ? " s" : ""}`;
  }).join(" · ");
}

// ---------- videos on the server ----------
async function refreshVideos(want = vid.id) {
  try {
    const { videos } = await vFetch("/api/video");
    const label = (x) => {
      const state = {
        uploading: `uploading ${x.size ? Math.floor((100 * x.received) / x.size) : 0}%`,
        processing: "preparing", error: "failed", cancelled: "stopped",
      }[x.state];
      return `${x.name}${x.duration ? ` · ${fmtTime(x.duration).replace(/\.\d+$/, "")}` : ""}${state ? ` (${state})` : ""}`;
    };
    const options = videos.map((x) => ({ value: x.id, text: label(x) }));
    if (!options.some((o) => o.value === want)) options.unshift({ value: "", text: videos.length ? "Choose a video…" : "No videos yet" });
    fillSelect(ve.recent, options, want || ""); // app.js
    return videos;
  } catch (e) {
    vStatus(`Could not list videos: ${e.message}`, "error");
    return [];
  }
}

ve.recent.addEventListener("change", () => {
  if (ve.recent.value) openVideo(ve.recent.value);
});
ve.refresh.addEventListener("click", async () => {
  if (!vid.detectors.length) await loadDetectors().catch((e) => vStatus(e.message, "error"));
  if (!se.model.options.length) await initSpeech(); // video-speech.js
  await refreshVideos();
  if (vid.id) pollNow();
});

async function openVideo(id, { file = null } = {}) {
  if (vid.id !== id) {
    vid.id = id;
    vid.status = null;
    vid.active = null;
    vid.loop = null;
    vid.playing = -1;
    vid.shownErrors.clear();
    vid.scores.clear();
    vid.preview = null;
    vid.drag = null;
    showNoResult();
    ve.info.replaceChildren(h("span", { class: "empty" }, "Loading…"));
    ve.compare.replaceChildren();
    resetSpeech(); // video-speech.js
    if (vid.local && vid.local.id !== id) {
      URL.revokeObjectURL(vid.local.url);
      vid.local = null;
    }
    setPlayerSource(null);
  }
  if (file) {
    // Play the chosen file straight from disk while it uploads, if the browser can
    if (vid.local) URL.revokeObjectURL(vid.local.url);
    vid.local = { id, url: URL.createObjectURL(file), failed: false };
    setPlayerSource(vid.local.url);
  }
  vid.ui.videoId = id;
  saveVideoUi();
  if (![...ve.recent.options].some((o) => o.value === id)) await refreshVideos(id);
  else ve.recent.value = id;
  await pollNow();
}

function closeVideo() {
  vid.id = null;
  vid.status = null;
  vid.active = null;
  vid.ui.videoId = null;
  saveVideoUi();
  if (vid.local) URL.revokeObjectURL(vid.local.url);
  vid.local = null;
  setPlayerSource(null);
  showNoResult();
  ve.info.replaceChildren(h("span", { class: "empty" }, "No video chosen."));
  ve.compare.replaceChildren();
  resetSpeech(); // video-speech.js
  renderBusy();
  updateButtons();
}

// ---------- status polling ----------
function pollNow() {
  clearTimeout(vid.pollTimer);
  vid.pollTimer = null;
  return poll();
}

async function poll() {
  const id = vid.id;
  if (!id) return;
  let s;
  try {
    s = await vFetch(`/api/video/${id}`);
  } catch (e) {
    if (id !== vid.id) return;
    if (e.status === 404) {
      vStatus("That video is no longer on the server.", "error");
      closeVideo();
      refreshVideos();
      return;
    }
    vStatus(e.message, "error");
    vid.pollTimer = setTimeout(poll, 5000);
    return;
  }
  if (id !== vid.id) return;
  applyStatus(s);
  if (s.state === "processing" || s.jobs.length || (s.state === "uploading" && !vid.upload)) {
    clearTimeout(vid.pollTimer);
    vid.pollTimer = setTimeout(poll, s.state === "uploading" ? 5000 : POLL_MS);
  }
}

function applyStatus(s) {
  const prev = vid.status;
  vid.status = s;
  if (s.state === "ready" && (!vid.local || vid.local.failed || vid.local.id !== s.id)) {
    setPlayerSource(`/api/video/${s.id}/proxy.mp4`);
  }
  for (const job of s.recent || []) {
    if (job.state === "error" && !vid.shownErrors.has(job.id)) {
      vid.shownErrors.add(job.id);
      vStatus(`${job.label} failed: ${job.error}`, "error");
    }
  }
  if (s.state === "error" && s.error && !vid.shownErrors.has(s.error)) {
    vid.shownErrors.add(s.error);
    vStatus(`Could not prepare the video: ${s.error}`, "error");
  }

  const before = new Set((prev?.id === s.id ? prev.detections : []).map((d) => d.key));
  const added = s.detections.filter((d) => !before.has(d.key));
  const selected = selectedDetection();
  if (!vid.active || !s.detections.some((d) => d.key === vid.active)) {
    // First look at this video: show the chosen detector's result, or else the latest one
    const pick = selected || (prev?.id === s.id ? null : s.detections[s.detections.length - 1]);
    if (pick) {
      if (!selected) selectDetection(pick);
      else setActive(pick.key);
    } else if (vid.active) {
      showNoResult();
    }
  } else if (selected && added.some((d) => d.key === selected.key)) {
    setActive(selected.key);
  }
  if (prev?.id === s.id) {
    for (const d of added) {
      // Elapsed 0: a live detector's saved scores cut at a new threshold, with no re-run
      const took = d.elapsed ? ` in ${formatDuration(d.elapsed * 1000)}` : "";
      vStatus(`${d.label} (${paramsText(d.detector, d.params)}): ${d.count} scene${d.count === 1 ? "" : "s"}${took}.`);
    }
    if (prev.state !== "ready" && s.state === "ready") refreshVideos();
  }
  onSpeechStatus(s, prev); // video-speech.js
  renderInfo();
  renderCompare();
  renderBusy();
  updateButtons();
}

// ---------- player ----------
function setPlayerSource(url) {
  const p = ve.player;
  if (url) {
    if (p.getAttribute("src") !== url) {
      // Keep the position when switching from the local file to the proxy
      const t = p.getAttribute("src") ? p.currentTime : 0;
      p.src = url;
      if (t) p.currentTime = t;
    }
    p.hidden = false;
  } else if (p.getAttribute("src")) {
    p.removeAttribute("src");
    p.load();
  }
  p.hidden = !url;
  // With a video open but nothing to play yet, the stage stays dark under the progress bar
  ve.noPreview.hidden = !!url || !vid.local?.failed;
  ve.hint.hidden = !!vid.id;
  ve.stage.classList.toggle("has-video", !!vid.id);
}

ve.player.addEventListener("error", () => {
  if (!ve.player.getAttribute("src")) return;
  if (vid.local && ve.player.getAttribute("src") === vid.local.url) {
    vid.local.failed = true;
    if (vid.status?.state === "ready") setPlayerSource(`/api/video/${vid.id}/proxy.mp4`);
    else setPlayerSource(null);
  } else {
    vStatus("The preview could not be played.", "error");
  }
});

function scenesNow() {
  return activeResult()?.scenes || [];
}

// Index of the scene playing at time t (binary search over scene starts)
function sceneAt(t) {
  const scenes = scenesNow();
  let lo = 0;
  let hi = scenes.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (scenes[mid].start <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

ve.player.addEventListener("timeupdate", () => {
  const t = ve.player.currentTime;
  if (vid.loop && (t >= vid.loop.end - 0.03 || t < vid.loop.start - 0.5)) ve.player.currentTime = vid.loop.start;
  const i = sceneAt(t);
  if (i !== vid.playing) {
    ve.scenes.querySelector(".v-scene.playing")?.classList.remove("playing");
    ve.scenes.querySelector(`.v-scene[data-i="${i}"]`)?.classList.add("playing");
    vid.playing = i;
  }
  drawTimeline();
});
ve.player.addEventListener("seeked", () => {
  const t = ve.player.currentTime;
  if (vid.loop && (t < vid.loop.start - 0.1 || t > vid.loop.end + 0.1)) setLoop(null);
  drawTimeline();
});

function setLoop(loop) {
  vid.loop = loop;
  ve.scenes.querySelector(".v-scene.looping")?.classList.remove("looping");
  if (loop) ve.scenes.querySelector(`.v-scene[data-i="${loop.i}"]`)?.classList.add("looping");
  drawTimeline();
}

function playScene(i) {
  const s = scenesNow()[i];
  if (!s || ve.player.hidden) return;
  if (vid.loop?.i === i) {
    setLoop(null);
    return;
  }
  setLoop({ i, start: s.start, end: s.end });
  ve.player.currentTime = s.start;
  ve.player.play().catch(() => { /* autoplay blocked: the user can press play */ });
}

// ---------- results ----------
function activeResult() {
  return vid.active ? vid.results.get(`${vid.id}/${vid.active}`) : null;
}

async function loadResult(key) {
  const cacheKey = `${vid.id}/${key}`;
  if (!vid.results.has(cacheKey)) {
    const data = await vFetch(`/api/video/${vid.id}/scenes/${key}`);
    vid.results.set(cacheKey, data);
    while (vid.results.size > RESULT_CACHE) vid.results.delete(vid.results.keys().next().value);
  }
  return vid.results.get(cacheKey);
}

async function setActive(key) {
  const id = vid.id;
  vid.active = key;
  let result;
  try {
    result = await loadResult(key);
  } catch (e) {
    vStatus(`Could not load scenes: ${e.message}`, "error");
    return;
  }
  if (id !== vid.id || vid.active !== key) return;
  vid.preview = null;
  setLoop(null);
  vid.playing = -1;
  renderScenes(result);
  renderCompare();
  drawTimeline();
  updateButtons();
  onTagResult(); // video-tag.js: this result's tags
}

// Show a finished result and set the header to the detector and settings that made it
function selectDetection(det) {
  if (ve.detector.value !== det.detector) {
    ve.detector.value = det.detector;
    vid.ui.detector = det.detector;
  }
  vid.ui.params[det.detector] = { ...det.params };
  saveVideoUi();
  renderParams();
  setActive(det.key);
  updateButtons();
}

function showNoResult() {
  vid.active = null;
  setLoop(null);
  vid.preview = null;
  ve.sceneCount.textContent = "";
  renderTrackInfo();
  const msg = !vid.id ? "Choose a video, then press Detect scenes."
    : vid.status?.state === "ready" ? "No scenes for this detector and settings yet. Press Detect scenes."
      : "Scenes appear here once detection has run.";
  ve.scenes.replaceChildren(h("span", { class: "empty" }, msg));
  drawTimeline();
  renderCompare();
  updateButtons();
  onTagResult(); // video-tag.js
}

function renderScenes(result) {
  vid.rendered = 0;
  ve.scenes.replaceChildren();
  ve.scenes.scrollTop = 0;
  const n = result.scenes.length;
  ve.sceneCount.textContent = `${n} · ${result.label}`;
  renderTrackInfo();
  appendScenes(SCENE_PAGE);
}

const moreObserver = new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting)) appendScenes(vid.rendered + SCENE_PAGE);
}, { root: ve.scenes, rootMargin: "400px" });

function appendScenes(upto) {
  const result = activeResult();
  if (!result) return;
  const scenes = result.scenes;
  upto = Math.min(upto, scenes.length);
  ve.scenes.querySelector(".v-more")?.remove();
  const rows = [];
  for (let i = vid.rendered; i < upto; i++) rows.push(sceneRow(scenes[i], i));
  ve.scenes.append(...rows);
  vid.rendered = upto;
  if (upto < scenes.length) {
    const more = h("div", { class: "v-more meta" }, `${scenes.length - upto} more…`);
    ve.scenes.append(more);
    moreObserver.observe(more);
  }
  if (vid.playing >= 0) ve.scenes.querySelector(`.v-scene[data-i="${vid.playing}"]`)?.classList.add("playing");
  if (vid.loop) ve.scenes.querySelector(`.v-scene[data-i="${vid.loop.i}"]`)?.classList.add("looping");
}

// A little into the scene, past any transition (same as thumb_time in video/server.py). Tagging
// sends the original's frame at this time to the models.
function thumbTime(s) {
  return s.start + Math.min((s.end - s.start) / 3, 2);
}

function sceneRow(s, i) {
  return h("div", { class: "v-scene", dataset: { i } },
    h("img", { loading: "lazy", decoding: "async", alt: "", src: `/api/video/${vid.id}/thumb?t=${thumbTime(s).toFixed(3)}` }),
    h("div", { class: "v-scene-text" },
      h("strong", {}, `Scene ${i + 1}`),
      h("span", { class: "v-scene-time" }, `${fmtTime(s.start)} – ${fmtTime(s.end)}`),
      h("span", { class: "meta" }, fmtDur(s.end - s.start)),
      sceneTagLine(i))); // video-tag.js
}

function scrollToScene(i) {
  if (i >= vid.rendered) appendScenes(i + SCENE_PAGE);
  ve.scenes.querySelector(`.v-scene[data-i="${i}"]`)?.scrollIntoView({ block: "nearest" });
}

ve.scenes.addEventListener("click", (e) => {
  const row = e.target.closest(".v-scene");
  if (!row) return;
  playScene(Number(row.dataset.i));
});

// ---------- timeline ----------
function videoDuration() {
  return activeResult()?.duration || vid.status?.info?.duration || ve.player.duration || 0;
}

function cssVar(name) {
  return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function sizeCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  const hgt = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(hgt * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(hgt * dpr);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, hgt);
  return { ctx, w, h: hgt };
}

function drawTimeline() {
  if (document.body.dataset.app !== "video") return;
  const { ctx, w, h: hgt } = sizeCanvas(ve.timeline);
  const dur = videoDuration();
  if (!w || !dur) return;
  const accent = cssVar("--accent");
  const muted = cssVar("--muted");
  const result = activeResult();
  const band = Math.min(40, Math.max(24, Math.round(hgt * 0.15))); // scene blocks along the top; the score curve fills the rest
  vid.curveGeom = null;
  if (result) {
    const scenes = vid.preview?.scenes || result.scenes;
    for (let i = 0; i < scenes.length; i++) {
      const x0 = (scenes[i].start / dur) * w;
      const x1 = (scenes[i].end / dur) * w;
      ctx.fillStyle = accent;
      ctx.globalAlpha = i % 2 ? 0.55 : 0.3;
      ctx.fillRect(x0, 0, Math.max(x1 - x0, 0.5), band);
    }
    ctx.globalAlpha = 1;
    if (!vid.preview) drawTagMarks(ctx, scenes, dur, w, band + 1); // video-tag.js
    if (vid.loop) {
      ctx.strokeStyle = cssVar("--text");
      ctx.lineWidth = 2;
      const x0 = (vid.loop.start / dur) * w;
      ctx.strokeRect(x0 + 1, 1, Math.max(((vid.loop.end - vid.loop.start) / dur) * w - 2, 1), band - 2);
    }
    vid.curveGeom = drawCurve(ctx, result.curve, 0, band + 4, w, hgt - band - 6, muted, accent,
      vid.drag?.value ?? result.curve.threshold, vid.drag?.max);
    if (vid.curveGeom) drawThresholdHandle(ctx, w, vid.curveGeom.thresholdY, vid.drag?.value ?? result.curve.threshold, accent);
  }
  if (vid.hoverX != null) {
    ctx.fillStyle = muted;
    ctx.globalAlpha = 0.6;
    ctx.fillRect(vid.hoverX, 0, 1, hgt);
    ctx.globalAlpha = 1;
  }
  if (ve.player.getAttribute("src")) {
    const x = (ve.player.currentTime / dur) * w;
    ctx.fillStyle = cssVar("--box");
    ctx.fillRect(Math.round(x) - 1, 0, 2, hgt);
  }
}

// The detector's score per bucket of frames, with its threshold as a dashed line. max is kept
// fixed while the line is dragged, so the scale doesn't move under the pointer.
function drawCurve(ctx, curve, x, y, w, hgt, color, thresholdColor, threshold = curve?.threshold, max = null) {
  const values = curve?.values || [];
  if (!values.length || hgt < 8) return null;
  if (max == null) {
    max = curve.threshold * 1.25;
    for (const v of values) if (v != null && v > max) max = v;
  }
  if (!(max > 0)) return null;
  const toY = (v) => y + hgt - (Math.min(Math.max(v, 0), max) / max) * hgt;
  ctx.beginPath();
  let pen = false;
  values.forEach((v, i) => {
    if (v == null) {
      pen = false;
      return;
    }
    const px = x + ((i + 0.5) / values.length) * w;
    if (pen) ctx.lineTo(px, toY(v));
    else ctx.moveTo(px, toY(v));
    pen = true;
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.stroke();
  const ty = toY(threshold);
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(x, ty);
  ctx.lineTo(x + w, ty);
  ctx.strokeStyle = thresholdColor;
  ctx.lineWidth = vid.drag ? 2 : 1;
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 1;
  return { y, h: hgt, max, thresholdY: ty };
}

// A tab at the right end of the threshold line, showing its value and that it can be dragged
function drawThresholdHandle(ctx, w, y, value, color) {
  const text = `⇕ ${fmtParam(value)}`;
  ctx.font = "11px system-ui, sans-serif";
  const tw = ctx.measureText(text).width + 10;
  const top = Math.min(Math.max(y - 8, 0), ve.timeline.clientHeight - 16);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(w - tw - 2, top, tw, 16, 4);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.fillText(text, w - tw + 3, top + 8);
}

function fmtParam(v) {
  return String(Math.round(v * 1000) / 1000);
}

function scheduleDraw() {
  if (vid.drawFrame) return;
  vid.drawFrame = requestAnimationFrame(() => {
    vid.drawFrame = 0;
    drawTimeline();
    renderTrackInfo();
  });
}

function timelineTime(e) {
  const r = ve.timeline.getBoundingClientRect();
  const x = Math.min(Math.max(e.clientX - r.left, 0), r.width);
  return { x, t: (x / r.width) * videoDuration() };
}

function onThresholdLine(e) {
  const g = vid.curveGeom;
  if (!g || !activeResult()) return false;
  const r = ve.timeline.getBoundingClientRect();
  return Math.abs(e.clientY - r.top - g.thresholdY) <= 6;
}

// ---------- threshold dragging ----------
// Live detectors (TransNetV2, FFmpeg) save a score per frame, so the browser cuts them at the
// dragged threshold as it moves: the same rule as cuts_at_peaks / cuts_above in video/detectors.py.
// On release the server saves that result instantly. Other detectors re-run on release.
function loadScores(detectorId) {
  if (!vid.scores.has(detectorId)) {
    vid.scores.set(detectorId, fetch(`/api/video/${vid.id}/scores/${detectorId}`)
      .then((r) => (r.ok ? r.arrayBuffer() : null))
      .then((buf) => buf && new Float32Array(buf))
      .catch(() => null));
  }
  return vid.scores.get(detectorId);
}

function cutsFromScores(rule, scores, threshold, minFrames) {
  // The server compares float32 scores against a float32 threshold
  const thr = Math.fround(threshold);
  const n = scores.length;
  const cuts = [];
  if (rule === "peaks") {
    for (let i = 0; i < n;) {
      if (!(scores[i] > thr)) {
        i++;
        continue;
      }
      let best = i;
      let j = i;
      for (; j < n && scores[j] > thr; j++) if (scores[j] > scores[best]) best = j;
      if (best + 1 < n) cuts.push(best + 1);
      i = j;
    }
  } else {
    for (let i = 1; i < n; i++) if (scores[i] > thr) cuts.push(i);
  }
  const out = [];
  let prev = 0;
  for (const c of cuts) {
    if (c - prev >= minFrames) {
      out.push(c);
      prev = c;
    }
  }
  return out;
}

function scenesFromCuts(cuts, fps, duration) {
  const starts = [0];
  for (const c of cuts) if (c / fps < duration) starts.push(Math.round((c / fps) * 1000) / 1000);
  return starts.map((start, i) => {
    const end = i + 1 < starts.length ? starts[i + 1] : Math.round(duration * 1000) / 1000;
    return { start, end, start_frame: Math.round(start * fps), end_frame: Math.round(end * fps) };
  });
}

function startDrag(e) {
  const result = activeResult();
  const detector = vid.detectors.find((d) => d.id === result.detector);
  if (!detector || !vid.curveGeom) return;
  vid.drag = { result, detector, max: vid.curveGeom.max, value: result.curve.threshold, moved: false, scores: null };
  if (detector.live) {
    loadScores(detector.id).then((scores) => {
      if (vid.drag?.result !== result) return;
      vid.drag.scores = scores;
      previewDrag();
    });
  }
  ve.timeline.setPointerCapture(e.pointerId);
  scheduleDraw();
}

function moveDrag(e) {
  const d = vid.drag;
  const g = vid.curveGeom;
  if (!g) return;
  const r = ve.timeline.getBoundingClientRect();
  const param = d.detector.params.find((p) => p.name === d.detector.curve_param);
  let value = ((g.y + g.h - (e.clientY - r.top)) / g.h) * d.max;
  value = Math.min(Math.max(value, param.min), param.max);
  // One decimal finer than the input's step: 0.15 for TransNetV2, 45.9 for PySceneDetect
  const scale = 10 ** (Math.max(0, Math.ceil(-Math.log10(param.step))) + 1);
  d.value = Math.round(value * scale) / scale;
  d.moved = true;
  previewDrag();
  scheduleDraw();
}

function previewDrag() {
  const d = vid.drag;
  if (!d?.scores || !d.moved) return;
  const r = d.result;
  const minFrames = Math.max(1, Math.floor(r.params.min_scene * r.fps + 0.5));
  const cuts = cutsFromScores(d.detector.rule, d.scores, d.value, minFrames);
  vid.preview = { threshold: d.value, scenes: scenesFromCuts(cuts, r.fps, r.duration) };
  scheduleDraw();
}

function endDrag() {
  const d = vid.drag;
  vid.drag = null;
  if (!d?.moved) {
    scheduleDraw();
    return;
  }
  vid.dragEnded = true;
  // Make the dragged value the header setting, then detect: instant for live detectors with
  // saved scores (the preview stays up until the saved result replaces it), a re-run otherwise
  if (ve.detector.value !== d.detector.id) {
    ve.detector.value = d.detector.id;
    vid.ui.detector = d.detector.id;
  }
  vid.ui.params[d.detector.id] = { ...d.result.params, [d.detector.curve_param]: d.value };
  saveVideoUi();
  renderParams();
  if (!(d.detector.live && d.scores)) {
    vid.preview = null;
    onSelectionChanged();
  }
  scheduleDraw();
  detectScenes();
}

ve.timeline.addEventListener("pointerdown", (e) => {
  if (e.button !== 0 || !onThresholdLine(e)) return;
  e.preventDefault();
  startDrag(e);
});
ve.timeline.addEventListener("pointermove", (e) => {
  if (vid.drag) {
    moveDrag(e);
    return;
  }
  const { x, t } = timelineTime(e);
  vid.hoverX = x;
  const i = sceneAt(t);
  ve.hover.textContent = fmtTime(t) + (i >= 0 ? ` · scene ${i + 1}${tagHoverText(i)}` : "") + speechHoverText(t); // video-tag.js, video-speech.js
  ve.timeline.style.cursor = onThresholdLine(e) ? "ns-resize" : "";
  scheduleDraw();
});
ve.timeline.addEventListener("pointerup", () => vid.drag && endDrag());
ve.timeline.addEventListener("pointercancel", () => vid.drag && endDrag());
ve.timeline.addEventListener("pointerleave", () => {
  if (vid.drag) return;
  vid.hoverX = null;
  ve.hover.textContent = "";
  scheduleDraw();
});
ve.timeline.addEventListener("click", (e) => {
  if (vid.dragEnded) {
    vid.dragEnded = false;
    return;
  }
  const { t } = timelineTime(e);
  if (!ve.player.getAttribute("src")) return;
  setLoop(null);
  ve.player.currentTime = t;
  const i = sceneAt(t);
  if (i >= 0) scrollToScene(i);
});

// What the timeline is showing: detector, settings and scene count, beside the track
function renderTrackInfo() {
  const r = activeResult();
  if (!r) {
    ve.trackLabel.replaceChildren(h("span", { class: "meta" }, vid.id ? "No result shown" : ""));
    ve.timelineMeta.textContent = "";
    return;
  }
  const d = vid.drag;
  const params = d ? { ...r.params, [d.detector.curve_param]: d.value } : r.params;
  let count = `${r.scenes.length} scenes`;
  if (d && vid.preview) count = `${vid.preview.scenes.length} scenes · release to keep`;
  else if (d) count = "release to re-run";
  ve.trackLabel.replaceChildren(
    h("strong", {}, r.label),
    h("span", { class: "meta", title: paramsText(r.detector, params) }, paramsText(r.detector, params, true)),
    h("span", { class: d ? "v-track-count dragging" : "v-track-count" }, count));
  const det = vid.detectors.find((x) => x.id === r.detector);
  ve.timelineMeta.textContent = det?.live
    ? "· drag the dashed line to change the threshold"
    : "· drag the dashed line, release to re-run";
  ve.timelineMeta.title = det?.live
    ? "Scenes update as you drag: this detector saved a score for every frame."
    : "This detector re-runs with the new threshold when you let go; long videos take a few minutes.";
}

// One thin row per result, so detectors can be compared at a glance; click one to show it.
// Transcripts add a row each showing where there is speech.
function renderCompare() {
  const s = vid.status;
  const dur = videoDuration();
  const rows = [];
  // Grouped by detector (in menu order), then by threshold, so runs of one detector sit together
  const order = (det) => vid.detectors.findIndex((d) => d.id === det.detector);
  const thr = (det) => det.params?.[vid.detectors.find((d) => d.id === det.detector)?.curve_param] ?? 0;
  const detections = [...(s?.detections || [])].sort((a, b) => order(a) - order(b) || thr(a) - thr(b));
  for (const det of detections) {
    const canvas = h("canvas", {});
    const del = h("button", { type: "button", title: "Delete this result" }, "×");
    const row = h("div", {
      class: "v-track" + (det.key === vid.active ? " active" : ""),
      title: `${det.label}: ${paramsText(det.detector, det.params)}, ${det.count} scenes. Click to show.`,
    },
    h("span", { class: "v-track-label" },
      h("strong", {}, det.label),
      h("span", { class: "meta" }, `${paramsText(det.detector, det.params, true)} · ${det.count} scenes`)),
    canvas, del);
    row.addEventListener("click", () => selectDetection(det));
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteResult(det);
    });
    rows.push({ row, canvas, cuts: det.cuts, active: det.key === vid.active });
  }
  if (s) rows.push(...speechTracks()); // video-speech.js
  for (const job of s?.jobs || []) {
    if (job.kind === "prepare") continue;
    const canvas = h("canvas", {});
    const state = job.state === "queued" ? "queued" : job.stage || `running · ${Math.floor(job.progress * 100)}%`;
    const row = h("div", { class: "v-track running" },
      h("span", { class: "v-track-label" }, h("strong", {}, job.kind === "transcribe" ? "Speech" : job.label),
        h("span", { class: "meta" }, job.kind === "transcribe" ? `${job.label.replace(/^Whisper /, "")} · ${state}` : state)),
      canvas, h("span"));
    rows.push({ row, canvas, progress: job.progress });
  }
  ve.compare.replaceChildren(...rows.map((r) => r.row));
  const accent = cssVar("--accent");
  const muted = cssVar("--muted");
  for (const r of rows) {
    const { ctx, w, h: hgt } = sizeCanvas(r.canvas);
    ctx.fillStyle = cssVar("--border");
    ctx.fillRect(0, hgt / 2, w, 1);
    if (r.progress != null) {
      ctx.fillStyle = accent;
      ctx.globalAlpha = 0.35;
      ctx.fillRect(0, 2, w * r.progress, hgt - 4);
      ctx.globalAlpha = 1;
      continue;
    }
    if (!dur) continue;
    ctx.fillStyle = r.active ? accent : muted;
    if (r.spans) {
      ctx.globalAlpha = r.active ? 0.6 : 0.45;
      for (const [a, b] of r.spans) ctx.fillRect((a / dur) * w, 4, Math.max(((b - a) / dur) * w, 1), hgt - 8);
      ctx.globalAlpha = 1;
      continue;
    }
    for (const c of r.cuts) ctx.fillRect(Math.round((c / dur) * w), 1, 1, hgt - 2);
  }
}

async function deleteResult(det) {
  try {
    await vFetch(`/api/video/${vid.id}/scenes/${det.key}`, { method: "DELETE" });
  } catch (e) {
    vStatus(`Could not delete: ${e.message}`, "error");
    return;
  }
  vid.results.delete(`${vid.id}/${det.key}`);
  if (vid.active === det.key) showNoResult();
  await pollNow();
}

function drawAll() {
  drawTimeline();
  renderCompare();
}

new ResizeObserver(() => requestAnimationFrame(drawAll)).observe(ve.timeline);

// ---------- info ----------
function renderInfo() {
  const s = vid.status;
  if (!s) return;
  const i = s.info;
  const rows = [["File", s.name], ["Size", fmtBytes(s.size)]];
  if (s.state === "uploading") rows.push(["Uploaded", `${fmtBytes(s.received)} of ${fmtBytes(s.size)}`]);
  if (i) {
    const v = i.video;
    rows.push(
      ["Duration", fmtTime(i.duration)],
      ["Container", i.format],
      ["Video", [v.codec, v.profile].filter(Boolean).join(" · ")],
      ["Resolution", `${v.width}×${v.height}${v.rotation ? ` (rotated ${v.rotation}°)` : ""}`],
      ["Frame rate", `${Math.round(v.fps * 1000) / 1000} fps`],
      ["Pixel format", v.pix_fmt],
      ["Bit rate", i.bit_rate ? `${(i.bit_rate / 1e6).toFixed(1)} Mb/s` : null],
      ["Audio", i.audio.length
        ? i.audio.map((a) => [a.codec, a.channels && `${a.channels} ch`, a.language !== "und" && a.language].filter(Boolean).join(" ")).join(", ")
        : "none"],
      ["Recorded", i.created],
    );
  }
  rows.push(["Uploaded on", s.created ? new Date(s.created * 1000).toLocaleString() : null]);
  const table = h("table", {}, rows.filter(([, val]) => val != null && val !== "").map(([k, val]) =>
    h("tr", {}, h("td", {}, k), h("td", {}, String(val)))));
  ve.info.replaceChildren(table);
}

// ---------- progress ----------
function renderBusy() {
  if (vid.busyFrame) return;
  vid.busyFrame = requestAnimationFrame(() => {
    vid.busyFrame = 0;
    drawBusy();
  });
}

function drawBusy() {
  const up = vid.upload?.id === vid.id ? vid.upload : null;
  const s = vid.status?.id === vid.id ? vid.status : null;
  let text = "";
  let progress = null;
  let eta = "";
  let error = false;
  if (up) {
    const elapsed = (performance.now() - up.started) / 1000;
    const rate = elapsed > 1 ? (up.sent - up.startSent) / elapsed : 0;
    text = `Uploading ${fmtBytes(up.sent)} of ${fmtBytes(up.size)}` + (rate ? ` · ${fmtBytes(rate)}/s` : "");
    progress = up.sent / up.size;
    if (rate) eta = `${formatDuration(((up.size - up.sent) / rate) * 1000)} left`;
  } else if (s?.state === "uploading") {
    text = `Upload incomplete (${fmtBytes(s.received)} of ${fmtBytes(s.size)}). Choose the same file again to resume.`;
    progress = s.received / s.size;
  } else if (s?.state === "error" || s?.state === "cancelled") {
    text = s.state === "error" ? `Could not prepare the video: ${s.error}` : "Preparing was stopped. Press Detect scenes to continue.";
    error = s.state === "error";
  } else if (s?.jobs?.length) {
    const running = s.jobs.find((j) => j.state === "running") || s.jobs[0];
    const queued = s.jobs.length - 1;
    text = running.kind === "prepare" ? "Preparing a 360p copy for the detectors"
      : running.kind === "transcribe" ? `Transcribing · ${running.label}${running.stage ? ` · ${running.stage}` : ""}`
        : `Detecting scenes · ${running.label}`;
    if (running.state === "queued") text += " (waiting for other videos)";
    if (queued > 0) text += ` · ${queued} more queued`;
    progress = running.progress;
    if (running.eta != null) eta = `${formatDuration(running.eta * 1000)} left`;
  }
  ve.busy.hidden = !text;
  ve.busy.classList.toggle("error", error);
  ve.busyText.textContent = text;
  ve.busyEta.textContent = eta;
  if (progress != null) ve.busyBar.value = progress;
  ve.busyBar.hidden = progress == null;
  ve.cancel.hidden = !(up || s?.jobs?.length);
  ve.cancel.textContent = up ? "Pause" : "Cancel";
}

ve.cancel.addEventListener("click", async () => {
  if (vid.upload?.id === vid.id) {
    stopUpload();
    vStatus("Upload paused. Choose the same file again to resume it.");
    return;
  }
  try {
    await vPost(`/api/video/${vid.id}/cancel`);
    vStatus("Stopped.");
  } catch (e) {
    vStatus(e.message, "error");
  }
  pollNow();
});

// ---------- buttons ----------
function updateButtons() {
  const s = vid.status;
  const d = currentDetector();
  const canDetect = !!s && s.state !== "uploading" && !!d?.available;
  ve.detect.disabled = !canDetect;
  const det = selectedDetection();
  ve.detect.textContent = "Detect scenes";
  ve.detect.title = det ? "Already detected with these settings; shown below." : "Split the video into scenes with this detector (Cmd/Ctrl+Enter)";
  ve.exportBtn.disabled = !activeResult() || s?.state !== "ready";
  ve.saveCsv.disabled = ve.saveJson.disabled = !activeResult();
  ve.del.disabled = !vid.id;
  updateTagButtons(); // video-tag.js
}

// ---------- detection ----------
async function detectScenes() {
  if (!vid.id || ve.detect.disabled) return;
  const d = currentDetector();
  const params = currentParams();
  try {
    const r = await vPost(`/api/video/${vid.id}/detect`, { detector: d.id, params });
    if (r.cached) {
      await pollNow();
      setActive(r.key);
      return;
    }
    vStatus(`${d.label} queued.`);
  } catch (e) {
    vStatus(e.message, "error");
  }
  pollNow();
}

ve.detect.addEventListener("click", detectScenes);

// ---------- upload ----------
async function chooseFile(file) {
  if (!file) return;
  if (vid.upload && vid.upload.file !== file) {
    if (!confirm(`${vid.upload.file.name} is still uploading. Pause it and upload ${file.name} instead?`)) return;
    stopUpload();
  }
  let state;
  try {
    state = await vPost("/api/video/uploads", { name: file.name, size: file.size, last_modified: file.lastModified });
  } catch (e) {
    vStatus(e.message, "error");
    return;
  }
  await openVideo(state.id, { file });
  if (state.complete) {
    vStatus(`${file.name} is already on the server.`);
    return;
  }
  vStatus(state.received ? `Resuming the upload of ${file.name}.` : `Uploading ${file.name}.`);
  await uploadFile(file, state);
}

function putChunk(up, offset, blob) {
  return new Promise((resolve, reject) => {
    const xhr = (up.xhr = new XMLHttpRequest());
    xhr.open("PUT", `/api/video/uploads/${up.id}?offset=${offset}`);
    xhr.upload.onprogress = (e) => {
      up.sent = offset + e.loaded;
      renderBusy();
    };
    xhr.onload = () => {
      let data = null;
      try {
        data = JSON.parse(xhr.responseText);
      } catch { /* not JSON */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else reject(Object.assign(new Error(data?.detail || `HTTP ${xhr.status}`), { status: xhr.status, data }));
    };
    xhr.onerror = () => reject(new Error("network error"));
    xhr.onabort = () => reject(new Error("aborted"));
    xhr.send(blob);
  });
}

// Send the file in chunks: the browser reads one slice at a time, so a 40 GB file never sits in memory.
// Failures retry with backoff and resume from what the server actually has.
async function uploadFile(file, state) {
  const up = (vid.upload = {
    id: state.id, file, xhr: null, size: state.size, sent: state.received, startSent: state.received,
    started: performance.now(), cancelled: false,
  });
  renderBusy();
  let offset = state.received;
  let failures = 0;
  while (offset < up.size && !up.cancelled) {
    const end = Math.min(offset + state.chunk_size, up.size);
    try {
      offset = (await putChunk(up, offset, file.slice(offset, end))).received;
      failures = 0;
    } catch (e) {
      if (up.cancelled) break;
      if (e.status === 409 && e.data?.received != null) {
        offset = e.data.received;
        continue;
      }
      if (e.status && e.status < 500) {
        vStatus(`Upload failed: ${e.message}`, "error");
        break;
      }
      const delay = UPLOAD_RETRY_DELAYS[failures++];
      if (delay === undefined) {
        vStatus(`Upload stopped (${e.message}). Choose the same file again to resume it.`, "error");
        break;
      }
      vStatus(`Upload interrupted (${e.message}); retrying in ${delay / 1000}s…`, "error");
      await new Promise((r) => setTimeout(r, delay));
      try {
        offset = (await vFetch(`/api/video/uploads/${up.id}`)).received;
      } catch { /* try the same offset again */ }
    }
    up.sent = offset;
    renderBusy();
  }
  const done = offset >= up.size && !up.cancelled;
  if (vid.upload === up) vid.upload = null;
  renderBusy();
  if (!done) return;
  vStatus(`Uploaded ${file.name}. Preparing it for the detectors…`);
  if (vid.id === up.id) {
    await pollNow();
    detectScenes(); // queues behind the preparation
  }
  refreshVideos();
}

function stopUpload() {
  const up = vid.upload;
  if (!up) return;
  up.cancelled = true;
  up.xhr?.abort();
  vid.upload = null;
  renderBusy();
  pollNow();
}

window.addEventListener("beforeunload", (e) => {
  if (vid.upload) {
    e.preventDefault();
    e.returnValue = "";
  }
});

ve.pick.addEventListener("click", () => ve.file.click());
ve.file.addEventListener("change", () => {
  chooseFile(ve.file.files[0]);
  ve.file.value = "";
});
ve.stage.addEventListener("click", (e) => {
  if (ve.player.hidden && e.target.closest("#vBusy") === null) ve.file.click();
});
ve.stage.addEventListener("keydown", (e) => {
  if ((e.key === "Enter" || e.key === " ") && e.target === ve.stage && ve.player.hidden) ve.file.click();
});
["dragenter", "dragover"].forEach((t) => ve.stage.addEventListener(t, (e) => {
  e.preventDefault();
  ve.stage.classList.add("dragover");
}));
["dragleave", "drop"].forEach((t) => ve.stage.addEventListener(t, (e) => {
  e.preventDefault();
  ve.stage.classList.remove("dragover");
}));
ve.stage.addEventListener("drop", (e) => chooseFile(e.dataTransfer.files[0]));

// ---------- export and delete ----------
// The server builds every format (see export in video/server.py), so they always match. Scenes
// tagged with the model or workflow chosen in Tag scenes carry their JSON, and with a transcript
// shown, each scene carries the words spoken in it.
function exportScenes(format) {
  const r = activeResult();
  if (!r) return;
  const config = tagExportConfig(); // video-tag.js
  const transcript = speechExportKey(); // video-speech.js
  download(`/api/video/${vid.id}/export?${new URLSearchParams({ key: r.key, format, ...(config && { config }), ...(transcript && { transcript }) })}`);
  const tags = (config ? ` with tags from ${tagLabel()}` : "") + (transcript ? `${config ? " and" : " with"} speech` : ""); // video-tag.js
  if (format === "zip") {
    vStatus(`Exporting ${r.scenes.length} scenes${tags}. The zip downloads as the thumbnails are made; ` +
      "a long video can take a minute the first time.");
  } else if (tags) {
    vStatus(`Exporting ${r.scenes.length} scenes${tags}.`);
  }
}

ve.exportBtn.addEventListener("click", () => exportScenes("zip"));
ve.saveJson.addEventListener("click", () => exportScenes("json"));
ve.saveCsv.addEventListener("click", () => exportScenes("csv"));

ve.del.addEventListener("click", async () => {
  const s = vid.status;
  if (!vid.id || !confirm(`Delete ${s?.name || "this video"} and its scene results from the server?`)) return;
  if (vid.upload?.id === vid.id) stopUpload();
  try {
    await vFetch(`/api/video/${vid.id}`, { method: "DELETE" });
  } catch (e) {
    vStatus(`Could not delete: ${e.message}`, "error");
    return;
  }
  vStatus(`Deleted ${s?.name || "the video"}.`);
  closeVideo();
  refreshVideos();
});

// Cmd/Ctrl+Enter (app.js) runs detection, or tagging in Tag scenes, when the video side is open
function videoRun() {
  if (document.body.dataset.vmode === "tag") runTagging(); // video-tag.js
  else detectScenes();
}

// ---------- init ----------
async function initVideo() {
  vid.loaded = true;
  loadVideoUi();
  initTagging(); // video-tag.js
  updateButtons();
  setPlayerSource(null);
  try {
    await loadDetectors();
  } catch (e) {
    vStatus(`Video service unavailable: ${e.message}. Press ↻ to try again.`, "error");
  }
  await initSpeech(); // video-speech.js
  const videos = await refreshVideos(vid.ui.videoId);
  if (vid.ui.videoId && videos.some((x) => x.id === vid.ui.videoId)) await openVideo(vid.ui.videoId);
  else if (!vid.id) {
    vid.ui.videoId = null;
    showNoResult();
  }
}

// Once every script has loaded, video-tag.js included
document.addEventListener("DOMContentLoaded", () => setApp(document.body.dataset.app === "video" ? "video" : "image"));

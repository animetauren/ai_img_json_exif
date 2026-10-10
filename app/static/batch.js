"use strict";
// Batch tab: pick a folder, see every photo in a grid, run the selected model or workflow on all
// of them (or on one, in focused mode), and download tagged copies of them all as one zip.
// Shares globals with app.js and workflows.js; loads last.

const BATCH_UI_KEY = "llm-classifier-batch-ui";
const PENDING_EXPORTS_KEY = "llm-classifier-pending-exports";
// A run without a manifest that changed more recently than this is taken to be still exporting in
// another tab; one quieter than this was left behind when its page closed
const EXPORT_IDLE_SECONDS = 120;
const PHOTO_RE = /\.(jpe?g|png|webp|gif|avif|heic|heif|tiff?|bmp)$/i;
const RUN_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$/;
const THUMB_EDGE = 320;

const batch = {
  items: [],         // {file, path, rel, fp, tile, img, running, unsupported}
  folder: "",
  focus: -1,         // index of the focused photo; -1 shows the grid
  status: new Map(), // result key -> true (usable result) | false (failed)
  // Photos left when a Run all was paused or stopped, so the next one picks up there even when
  // "Re-run done" would otherwise start over: {key, rerun, items}
  pending: null,
};

const b = {
  kind: $("batchKind"),
  pick: $("batchPick"),
  input: $("batchInput"),
  rerun: $("batchRerun"),
  exportBtn: $("batchExport"),
  summary: $("batchSummary"),
  empty: $("batchEmpty"),
  tiles: $("batchTiles"),
  focusName: $("focusName"),
  back: $("focusBack"),
  prev: $("focusPrev"),
  next: $("focusNext"),
  dialog: $("exportDialog"),
  form: $("exportForm"),
  runName: $("exportRunName"),
  exportInfo: $("exportInfo"),
  exportError: $("exportError"),
  cancel: $("exportCancel"),
  pending: $("pendingExports"),
  progress: $("batchProgress"),
  progressBar: $("batchProgressBar"),
  progressText: $("batchProgressText"),
};

// ---------- saved results (IndexedDB) ----------
// "results" holds full records, "status" a small ok flag per key so the grid can colour every
// tile without loading every record, and "exif" each photo's flat EXIF (keyed by photo only).
const resultStore = (() => {
  const memory = new Map(); // used when IndexedDB is unavailable, e.g. some private windows
  const memoryExif = new Map();
  let dbPromise = null;
  const open = () => (dbPromise ??= new Promise((resolve) => {
    try {
      const req = indexedDB.open("photo-organizer", 2);
      req.onupgradeneeded = () => {
        for (const name of ["results", "status", "exif"]) {
          if (!req.result.objectStoreNames.contains(name)) req.result.createObjectStore(name);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  }));
  const done = (req) => new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return {
    async get(key) {
      const db = await open();
      return db ? done(db.transaction("results").objectStore("results").get(key)) : memory.get(key)?.record;
    },
    async put(key, record, ok) {
      const db = await open();
      if (!db) {
        memory.set(key, { record, ok });
        return;
      }
      const tx = db.transaction(["results", "status"], "readwrite");
      tx.objectStore("results").put(record, key);
      tx.objectStore("status").put(ok, key);
      await new Promise((resolve, reject) => {
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error);
      });
    },
    async getMany(keys) {
      const db = await open();
      if (!db) return keys.map((k) => memory.get(k)?.record);
      const store = db.transaction("results").objectStore("results");
      return Promise.all(keys.map((k) => done(store.get(k))));
    },
    async getExif(fps) {
      const db = await open();
      if (!db) return new Map(fps.filter((fp) => memoryExif.has(fp)).map((fp) => [fp, memoryExif.get(fp)]));
      const store = db.transaction("exif").objectStore("exif");
      const values = await Promise.all(fps.map((fp) => done(store.get(fp))));
      return new Map(fps.map((fp, i) => [fp, values[i]]).filter(([, v]) => v));
    },
    async putExif(fp, flat) {
      const db = await open();
      if (!db) memoryExif.set(fp, flat);
      else await done(db.transaction("exif", "readwrite").objectStore("exif").put(flat, fp));
    },
    async statuses() {
      const db = await open();
      if (!db) return new Map([...memory].map(([k, v]) => [k, v.ok]));
      const store = db.transaction("status").objectStore("status");
      const [keys, values] = await Promise.all([done(store.getAllKeys()), done(store.getAll())]);
      return new Map(keys.map((k, i) => [k, values[i]]));
    },
  };
})();

// ---------- which model or workflow results belong to ----------
// kind is single or workflow; video-tag.js passes Run with's value, whichever tab is open
function configKey(kind = document.body.dataset.kind) {
  if (kind === "workflow") return wf.current ? `workflow:${wf.current.id}` : null;
  return isVision() ? `vision:${el.visionModel.value}` : `llm:${currentModel() || "default"}`;
}

function configLabel(kind = document.body.dataset.kind) {
  if (kind === "workflow") return `workflow: ${wf.current?.name || "none selected"}`;
  return isVision() ? `vision: ${el.visionModel.value}` : `llm: ${currentModel() || "default model"}`;
}

const resultKey = (item, key = configKey()) => `${key}\n${item.fp}`;

// The JSON that export writes, or null when the run produced nothing usable
function recordResult(record) {
  if (!record || record.error || record.aborted) return null;
  if (record.kind === "llm") return record.json ?? null;
  if (record.kind === "vision") return record.data ?? null;
  return record.result ?? null;
}

// Errors that mean the model server is unreachable or overloaded rather than that this photo got
// a bad answer. These are retried and never saved, so an outage can't turn a folder red.
const TRANSIENT_RE = /ConnectError|ConnectTimeout|ReadTimeout|WriteTimeout|PoolTimeout|TimeoutException|ReadError|WriteError|RemoteProtocolError|All connection attempts failed|Connection refused|Failed to fetch|NetworkError|HTTP (408|429|5\d\d)\b/i;
const RETRY_DELAYS = [5000, 15000, 30000];

function transientError(record) {
  if (!record || record.aborted) return null;
  const messages = [record.error, ...(record.events || []).filter((e) => e.type === "step_error").map((e) => e.message)];
  return messages.find((m) => m && TRANSIENT_RE.test(m)) || null;
}

function recordOk(record) {
  if (recordResult(record) == null) return false;
  return record.kind !== "workflow" || !record.events.some((e) => e.type === "step_error");
}

// ---------- folder + grid ----------
b.pick.addEventListener("click", () => b.input.click());
b.empty.addEventListener("click", () => b.input.click());
b.empty.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") b.input.click();
});
b.input.addEventListener("change", async () => {
  const files = [...b.input.files].filter((f) => PHOTO_RE.test(f.name) && !f.name.startsWith("."));
  b.input.value = "";
  if (!files.length) {
    setStatus("No photos found in that folder.", "error");
    return;
  }
  files.sort((x, y) => x.webkitRelativePath.localeCompare(y.webkitRelativePath, undefined, { numeric: true }));
  batch.items.forEach((item) => item.thumbUrl && URL.revokeObjectURL(item.thumbUrl));
  batch.folder = files[0].webkitRelativePath.split("/")[0];
  batch.items = files.map((file) => ({
    file,
    path: file.webkitRelativePath,
    // Path inside the chosen folder: exports keep it, so names and subfolders are preserved
    rel: file.webkitRelativePath.split("/").slice(1).join("/") || file.name,
    // Same file, same key: results come back when the folder is picked again
    fp: `${file.webkitRelativePath}|${file.size}|${file.lastModified}`,
  }));
  batch.status = await resultStore.statuses();
  batch.pending = null;
  renderGrid();
  setFocus(-1);
  loadViewData(); // batch-view.js: results and cached EXIF for search, sort and group
  setStatus(`${files.length} photos in ${batch.folder}.`);
});

function renderGrid() {
  thumbObserver.disconnect();
  thumbQueue.length = 0;
  batch.items.forEach((item, i) => {
    item.img = h("img", { alt: "", draggable: "false" });
    item.tile = h("button", { type: "button", class: "tile", title: item.path, dataset: { index: String(i) } },
      item.img, h("span", { class: "tile-name" }, item.file.name));
    item.tile.addEventListener("click", () => setFocus(i));
    item.clones = []; // extra copies when grouping puts a photo in several groups
    thumbObserver.observe(item.tile);
  });
  b.empty.hidden = batch.items.length > 0;
  b.tiles.hidden = !batch.items.length;
  f.bar.hidden = !batch.items.length;
  applyView(); // batch-view.js: filters, sorts and groups the tiles into the grid
}

// The photos the grid shows, in display order (all of them with no filter)
const shownItems = () => view.order.map((i) => batch.items[i]);

function tileState(item) {
  if (item.running) return "running";
  if (item.waiting) return "waiting";
  const ok = batch.status.get(resultKey(item));
  return ok === true ? "done" : ok === false ? "failed" : "new";
}

function refreshTiles() {
  let done = 0;
  let failed = 0;
  for (const item of batch.items) {
    const state = tileState(item);
    if (item.tile) item.tile.dataset.state = state;
    item.clones?.forEach((c) => (c.dataset.state = state));
    if (state === "done") done++;
    else if (state === "failed") failed++;
  }
  b.summary.textContent = batch.items.length
    ? `${batch.folder} · ${batch.items.length} photos · ${done} done${failed ? ` · ${failed} failed` : ""} with ${configLabel()}` +
      (view.filtered ? ` · showing ${view.order.length}` : "")
    : "";
  // With a filter on, Export works on the photos shown
  b.exportBtn.textContent = view.filtered ? "Export shown…" : "Export all…";
  b.exportBtn.disabled = !shownItems().some((item) => tileState(item) === "done") || !!abortController;
  updateRunButton();
}

function updateRunButton() {
  if (document.body.dataset.mode !== "batch" || abortController) return;
  el.runBtn.textContent = runButtonLabel();
  el.runBtn.disabled = batch.focus >= 0 ? !imageDataUrl : !view.order.length;
}

// ---------- thumbnails: decoded lazily as tiles scroll into view ----------
const thumbQueue = [];
let thumbsActive = 0;
const thumbObserver = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    thumbObserver.unobserve(entry.target);
    thumbQueue.push(batch.items[Number(entry.target.dataset.index)]);
  }
  pumpThumbs();
}, { root: b.tiles, rootMargin: "400px" });

function pumpThumbs() {
  while (thumbsActive < 3 && thumbQueue.length) {
    const item = thumbQueue.shift();
    if (!item?.tile) continue;
    if (!item.tile.isConnected) {
      thumbObserver.observe(item.tile); // filtered out for now; load when it scrolls into view again
      continue;
    }
    thumbsActive++;
    makeThumb(item.file)
      .then((url) => {
        item.thumbUrl = url;
        item.img.src = url;
        item.clones.forEach((c) => (c.querySelector("img").src = url));
      })
      .catch(() => markUnsupported(item))
      .finally(() => {
        thumbsActive--;
        pumpThumbs();
      });
  }
}

async function makeThumb(file) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, THUMB_EDGE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
  return URL.createObjectURL(blob);
}

// The browser can't decode it (e.g. HEIC outside Safari), so it can't be sent to a model either
function markUnsupported(item) {
  item.unsupported = true;
  item.tile?.classList.add("no-preview");
  item.clones?.forEach((c) => c.classList.add("no-preview"));
}

// ---------- focused photo ----------
async function setFocus(index) {
  batch.focus = index;
  const item = batch.items[index];
  if (item) document.body.dataset.focus = "1";
  else delete document.body.dataset.focus;
  updateRunButton();
  if (!item) {
    clearOutputs();
    applyView(); // statuses may have changed while focused
    return;
  }
  const pos = view.order.indexOf(index);
  b.focusName.textContent = pos >= 0 ? `${item.path} (${pos + 1} of ${view.order.length})` : item.path;
  b.prev.disabled = pos <= 0;
  b.next.disabled = pos < 0 || pos === view.order.length - 1;
  item.tile?.scrollIntoView({ block: "nearest" });
  const shown = await loadImage(item.file, { autoRun: false });
  if (batch.focus !== index) return;
  if (!shown) markUnsupported(item);
  exifPromise?.then(() => {
    if (exifData && currentFile === item.file) rememberExif(item, exifData);
  });
  await showStored(item);
  updateRunButton();
}

async function showStored(item) {
  const key = resultKey(item);
  const record = batch.status.has(key) ? await resultStore.get(key) : null;
  if (batch.items[batch.focus] !== item) return;
  clearOutputs();
  if (item.unsupported) return; // loadImage already explained why
  if (!record) {
    setStatus(`Not run with ${configLabel()} yet. Press Run photo to try these settings on it.`);
    return;
  }
  showRecord(record);
  setStatus(`Saved result from ${new Date(record.at).toLocaleString()} (${record.source}).`, recordOk(record) ? "ok" : "error");
}

function stepFocus(delta) {
  const pos = view.order.indexOf(batch.focus);
  const next = view.order[pos + delta];
  if (pos >= 0 && next !== undefined) setFocus(next);
}

b.back.addEventListener("click", () => setFocus(-1));
b.prev.addEventListener("click", () => stepFocus(-1));
b.next.addEventListener("click", () => stepFocus(1));
document.addEventListener("keydown", (e) => {
  if (document.body.dataset.mode !== "batch" || document.body.dataset.app === "video" || batch.focus < 0 || b.dialog.open) return;
  if (e.target instanceof Element && e.target.closest("input, textarea, select")) return;
  if (e.key === "Escape") setFocus(-1);
  else if (e.key === "ArrowLeft") stepFocus(-1);
  else if (e.key === "ArrowRight") stepFocus(1);
});

// ---------- running ----------
// Called by run() in app.js (through withRun) when the Batch tab is active
async function runBatch(signal) {
  if (!configKey()) {
    setStatus("Choose a workflow first.", "error");
    return;
  }
  if (batch.focus >= 0) {
    const item = batch.items[batch.focus];
    if (await runItem(item, signal, { focused: true }) === "transient") {
      setStatus(`${configLabel()} is not responding (${item.lastError}). Nothing was saved for this photo.`, "error");
    }
    return;
  }
  if (document.body.dataset.kind === "workflow" && (!syncFromJson() || !wf.current.steps.length)) {
    setStatus("This workflow has no steps yet.", "error");
    return;
  }
  const shown = new Set(shownItems());
  const resuming = batch.pending?.key === configKey() && batch.pending.rerun === b.rerun.checked &&
    batch.pending.items.some((item) => shown.has(item));
  const todo = resuming
    ? batch.pending.items.filter((item) => shown.has(item))
    : shownItems().filter((item) => !item.unsupported && (b.rerun.checked || batch.status.get(resultKey(item)) !== true));
  batch.pending = null;
  if (!todo.length) {
    setStatus(`Every photo already has a result for ${configLabel()}. Tick "Re-run done" to run them again.`);
    return;
  }
  b.pick.disabled = true;
  b.exportBtn.disabled = true;
  const started = performance.now();
  if (resuming) setStatus(`Resuming: ${todo.length} photos left from the last run.`);
  const eta = batchEta(todo.length);
  let ran = 0;
  let failed = 0;
  let pausedOn = null; // the photo the batch gave up on when the server stayed unreachable
  try {
    for (const item of todo) {
      if (signal.aborted) break;
      setStatus(`Batch: photo ${ran + 1} of ${todo.length} · ${item.path}`);
      eta.start(ran);
      let ok = await runItem(item, signal);
      const retried = ok === "transient";
      for (let attempt = 0; ok === "transient" && attempt < RETRY_DELAYS.length && !signal.aborted; attempt++) {
        const secs = RETRY_DELAYS[attempt] / 1000;
        setStatus(`${item.path}: ${item.lastError}. Retrying in ${secs}s (${attempt + 1} of ${RETRY_DELAYS.length})…`, "error");
        item.waiting = true;
        refreshTiles();
        await abortableSleep(RETRY_DELAYS[attempt], signal);
        item.waiting = false;
        if (signal.aborted) break;
        setStatus(`Batch: photo ${ran + 1} of ${todo.length} · ${item.path} (retry ${attempt + 1})`);
        ok = await runItem(item, signal);
      }
      refreshTiles();
      if (ok === "transient") {
        pausedOn = item;
        break;
      }
      if (ok === null) break; // stopped, or nothing could run (e.g. an empty workflow)
      // Skipped photos (instant) and retried ones (waiting time) would skew the average
      if (!item.unsupported && !retried) eta.finish();
      ran++;
      if (!ok) failed++;
    }
  } finally {
    eta.stop();
    b.pick.disabled = false;
    applyView(); // new results can change what the filter, sort and groups show
  }
  if (ran < todo.length) {
    batch.pending = { key: configKey(), rerun: b.rerun.checked, items: todo.slice(ran) };
  }
  const avg = eta.average();
  const summary = `${ran} of ${todo.length} photos in ${formatDuration(performance.now() - started)}` +
    (avg ? ` (avg ${formatDuration(avg)} per photo)` : "") + (failed ? `, ${failed} failed (red)` : "") + ".";
  if (pausedOn) {
    setStatus(`Paused: ${configLabel()} is not responding (${pausedOn.lastError}). Finished photos are saved; ` +
      `${todo.length - ran} remain. Fix the server, then press ${runButtonLabel()} to resume. Ran ${summary}`, "error");
    return;
  }
  if (signal.aborted) {
    setStatus(`Stopped after ${summary} Press ${runButtonLabel()} to continue with the ${todo.length - ran} left.`, failed ? "error" : "ok");
    return;
  }
  setStatus(`Batch done: ${summary}`, failed ? "error" : "ok");
}

function abortableSleep(ms, signal) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}

// Progress and ETA for Run all, shown next to the Run button so it stays visible while browsing.
// The average uses the last 20 photos, leaving out the first once there are others: it often
// includes loading the model. ui holds the progress elements (video-tag.js has its own).
function batchEta(total, ui = b) {
  const times = [];
  let done = 0;
  let itemStarted = performance.now();
  const average = () => {
    const recent = (times.length > 1 ? times.slice(1) : times).slice(-20);
    return recent.length ? recent.reduce((a, t) => a + t, 0) / recent.length : 0;
  };
  const render = () => {
    ui.progressBar.max = total;
    ui.progressBar.value = done;
    const parts = [`${Math.min(done + 1, total)} of ${total}`];
    const avg = average();
    if (avg) {
      // Time already spent on the current photo comes off the estimate
      const left = Math.max(0, avg * (total - done) - (performance.now() - itemStarted));
      const finish = new Date(Date.now() + left).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      parts.push(`avg ${formatDuration(avg)}`, `~${formatDuration(left)} left`, `done ≈ ${finish}`);
    } else {
      parts.push("estimating…");
    }
    ui.progressText.textContent = parts.join(" · ");
  };
  ui.progress.hidden = false;
  const timer = setInterval(render, 1000);
  return {
    start(n) {
      done = n;
      itemStarted = performance.now();
      render();
    },
    finish() {
      times.push(performance.now() - itemStarted);
    },
    stop() {
      clearInterval(timer);
      ui.progress.hidden = true;
    },
    average,
  };
}

function formatDuration(ms) {
  const s = Math.round(ms / 1000);
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 10000) return `${(ms / 1000).toFixed(1)}s`;
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(s / 3600)}h ${String(Math.floor(s / 60) % 60).padStart(2, "0")}m`;
}

// Runs one photo and saves the result. Returns true/false for a usable result, null if nothing was
// saved (stopped), or "transient" if the model server was unreachable (also not saved).
// The focused photo runs through the photo panel, streaming into the outputs (runActive). Batch photos
// are computed from their own image data and never touch the panel, so browsing during a batch is safe.
async function runItem(item, signal, { focused = false } = {}) {
  // The key is taken per photo, so a setting changed mid-batch is recorded against the right model
  const key = resultKey(item);
  item.running = true;
  refreshTiles();
  let record = null;
  try {
    if (focused) {
      record = await runActive(signal);
    } else {
      let dataUrl;
      try {
        dataUrl = (await prepareImage(item.file)).dataUrl;
      } catch {
        markUnsupported(item);
        return false;
      }
      let exif = {};
      // Workflow prompts can use {{exif.*}}; single models don't need the upload
      if (document.body.dataset.kind === "workflow") {
        try {
          const data = await fetchExif(item.file);
          exif = flattenExif(data);
          rememberExif(item, data);
        } catch { /* run without EXIF */ }
      }
      if (signal.aborted) return null;
      record = await computeActive(signal, { dataUrl, exif });
    }
  } finally {
    item.running = false;
    refreshTiles();
  }
  if (!record || record.aborted) return null;
  const transient = transientError(record);
  if (transient) {
    item.lastError = transient;
    return "transient"; // nothing saved: the photo keeps whatever result it had
  }
  record.at = Date.now();
  record.source = configLabel();
  const ok = recordOk(record);
  await resultStore.put(key, record, ok);
  batch.status.set(key, ok);
  if (key === resultKey(item)) noteResult(item, recordResult(record));
  if (focused && exifData && currentFile === item.file) rememberExif(item, exifData);
  // The output panels belong to Tag scenes while the video side is open
  if (!focused && document.body.dataset.app !== "video") {
    // On the grid, the outputs show the latest finished photo; a focused photo keeps its own
    if (batch.focus < 0) {
      clearOutputs();
      showRecord(record);
    } else if (batch.items[batch.focus] === item) {
      showStored(item);
    }
  }
  refreshTiles();
  return ok;
}

// ---------- bulk export ----------
function defaultRunName() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `run-${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
}

b.exportBtn.addEventListener("click", () => {
  const n = shownItems().filter((item) => batch.status.get(resultKey(item)) === true).length;
  b.runName.value = defaultRunName();
  b.exportInfo.textContent = `${n} photo${n === 1 ? "" : "s"} with results for ${configLabel()} will be downloaded as ` +
    `<run name>.zip, keeping their file names and subfolders. Photos without a result (grey or red) are skipped.`;
  b.exportError.textContent = "";
  b.dialog.showModal();
  b.runName.select();
});
b.cancel.addEventListener("click", () => b.dialog.close());
b.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const name = b.runName.value.trim();
  if (!RUN_NAME_RE.test(name)) {
    b.exportError.textContent = "Use letters, digits, spaces, dots, dashes or underscores (starting with a letter or digit).";
    return;
  }
  let runId;
  try {
    const r = await fetch("/api/exports", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
    runId = d.id;
    pendingExports.add(runId);
  } catch (err) {
    b.exportError.textContent = "Could not start the export: " + err.message;
    return;
  }
  b.dialog.close();
  await withRun((signal) => bulkExport(name, runId, signal));
  refreshTiles();
});

// Tag each photo on the server, then download them all as <run>.zip
async function bulkExport(run, runId, signal) {
  const items = shownItems().filter((item) => batch.status.get(resultKey(item)) === true);
  const overwrite = el.exportOverwrite[0].checked;
  const source = configLabel();
  b.exportBtn.disabled = true;
  b.pick.disabled = true;
  const files = [];
  try {
    for (const item of items) {
      if (signal.aborted) break;
      setStatus(`Tagging ${files.length + 1} of ${items.length} for ${run}.zip · ${item.rel}`);
      const entry = { path: item.rel };
      try {
        const record = await resultStore.get(resultKey(item));
        const form = new FormData();
        form.append("file", item.file, item.file.name);
        form.append("result", JSON.stringify(recordResult(record)));
        form.append("overwrite", overwrite);
        form.append("source", record.source || source);
        form.append("save_to", runId);
        form.append("rel_path", item.rel);
        const r = await fetch("/api/export", { method: "POST", body: form, signal });
        const d = await r.json().catch(() => ({ detail: `HTTP ${r.status}` }));
        if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : `HTTP ${r.status}`);
        Object.assign(entry, { written: d.written, skipped: d.skipped.map((x) => x.tag), warnings: d.warnings });
      } catch (err) {
        if (err.name === "AbortError") break;
        entry.error = err.message;
      }
      files.push(entry);
    }
  } finally {
    b.pick.disabled = false;
  }
  const failed = files.filter((f) => f.error);
  // The manifest records the run even when it was stopped part-way
  const manifest = {
    run, folder: batch.folder, source, overwrite, exported: new Date().toISOString(),
    complete: !signal.aborted && files.length === items.length, files,
  };
  try {
    await fetch(`/api/exports/${runId}/manifest`, {
      method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(manifest),
    });
  } catch { /* the photos are tagged; only the summary file is missing */ }
  const done = files.length - failed.length;
  // The run stays remembered after the download starts: if the zip never arrives, the next visit offers it again
  if (done) downloadRun(runId, run);
  else discardRun(runId).catch(() => { /* offered again on the next visit, then expires */ });
  const kept = files.filter((f) => f.skipped?.length).length;
  let msg = `${signal.aborted ? "Stopped: downloading" : "Downloading"} ${run}.zip with ${done} of ${items.length} photos.`;
  if (!done) msg = `Nothing was exported${signal.aborted ? " (stopped)" : ""}.`;
  if (kept) msg += ` ${kept} kept some existing metadata; tick "Overwrite existing metadata" to replace it.`;
  if (failed.length) msg += ` ${failed.length} failed, e.g. ${failed[0].path}: ${failed[0].error}`;
  setStatus(msg, failed.length || signal.aborted || !done ? "error" : "ok");
}

// A plain link download, so the browser streams the zip to Downloads instead of holding it in memory
function downloadRun(runId, run) {
  const a = Object.assign(document.createElement("a"), { href: `/api/exports/${runId}/zip`, download: `${run}.zip` });
  document.body.appendChild(a);
  a.click();
  a.remove();
}

async function discardRun(runId) {
  const r = await fetch(`/api/exports/${runId}`, { method: "DELETE" });
  if (!r.ok && r.status !== 404) throw new Error(`HTTP ${r.status}`);
  pendingExports.remove(runId);
}

// ---------- exports left behind ----------
// Run ids stay here until the server no longer has the run (zip sent, discarded or expired), so a
// run whose page closed before its zip arrived can still be downloaded. Only this browser knows
// the ids: the server never lists runs, since an id is all it takes to download one.
const pendingExports = {
  list() {
    try {
      return JSON.parse(localStorage.getItem(PENDING_EXPORTS_KEY)) || [];
    } catch {
      return [];
    }
  },
  save(ids) {
    try {
      localStorage.setItem(PENDING_EXPORTS_KEY, JSON.stringify(ids));
    } catch { /* storage unavailable: such runs just expire on the server */ }
  },
  add(id) {
    this.save([...this.list().filter((x) => x !== id), id]);
  },
  remove(id) {
    this.save(this.list().filter((x) => x !== id));
  },
};

async function offerPendingExports() {
  for (const id of pendingExports.list()) {
    let info;
    try {
      const r = await fetch(`/api/exports/${id}`);
      if (r.status === 404) {
        pendingExports.remove(id);
        continue;
      }
      if (!r.ok) continue;
      info = await r.json();
    } catch {
      continue; // server unreachable: ask again on the next visit
    }
    if (!info.manifest && info.idle < EXPORT_IDLE_SECONDS) continue;
    if (!info.photos) {
      discardRun(id).catch(() => {});
      continue;
    }
    const photos = `${info.photos} photo${info.photos === 1 ? "" : "s"}`;
    const text = info.complete
      ? `Export “${info.name}” (${photos}) was not downloaded.`
      : `Export “${info.name}” stopped after tagging ${photos}.`;
    const row = h("div", { class: "pending-export" }, h("span", {}, text));
    const remove = () => {
      row.remove();
      b.pending.hidden = !b.pending.children.length;
    };
    const download = h("button", { type: "button", class: "primary" }, "Download zip");
    download.addEventListener("click", () => {
      downloadRun(id, info.name);
      remove();
    });
    const discard = h("button", { type: "button", title: "Delete the tagged copies from the server" }, "Discard");
    discard.addEventListener("click", async () => {
      try {
        await discardRun(id);
        remove();
      } catch (err) {
        setStatus(`Could not discard ${info.name}: ${err.message}`, "error");
      }
    });
    row.append(download, discard);
    b.pending.append(row);
    b.pending.hidden = false;
  }
  // The offer lives on the Batch tab
  if (!b.pending.hidden && document.body.dataset.mode !== "batch") {
    setStatus("An export was not downloaded. Open the Batch tab to download or discard it.");
  }
}

// ---------- settings changes ----------
function saveBatchUi() {
  try {
    localStorage.setItem(BATCH_UI_KEY, JSON.stringify({ kind: b.kind.value, rerun: b.rerun.checked }));
  } catch { /* storage unavailable */ }
}

// Called by workflows.js after a tab switch
function onModeChanged() {
  if (document.body.dataset.mode === "batch") {
    refreshTiles();
    const item = batch.items[batch.focus];
    if (item) showStored(item);
  } else if (!abortController) {
    el.runBtn.textContent = runButtonLabel();
    el.runBtn.disabled = !imageDataUrl;
  }
}

// Called when the model or workflow changes: tiles and the focused photo show that config's results.
// On the video side the same settings drive Tag scenes; the Batch tab catches up when shown again.
function onConfigChanged() {
  if (document.body.dataset.app === "video") {
    onTagConfigChanged(); // video-tag.js
    return;
  }
  if (document.body.dataset.mode !== "batch") return;
  refreshTiles();
  loadViewData();
  const item = batch.items[batch.focus];
  if (item && !abortController) showStored(item);
}

b.kind.addEventListener("change", () => {
  saveBatchUi();
  // In Tag scenes, Run with changes what runs without switching the image side's tab
  if (document.body.dataset.app === "video") syncKind(); // workflows.js
  else setMode("batch");
  onConfigChanged();
});
b.rerun.addEventListener("change", saveBatchUi);
[el.backend, el.modelSelect, el.modelText, el.visionModel].forEach((x) => x.addEventListener("change", onConfigChanged));

// ---------- init ----------
(() => {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(BATCH_UI_KEY)) || {};
  } catch { /* storage unavailable */ }
  b.kind.value = saved.kind === "workflow" ? "workflow" : "single";
  b.rerun.checked = !!saved.rerun;
  if (wfUi.mode === "workflow" || wfUi.mode === "batch") setMode(wfUi.mode);
  offerPendingExports();
})();

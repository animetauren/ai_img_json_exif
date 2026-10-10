"use strict";
// Batch grid search, filter, sort and group, by result JSON or EXIF.
// Loads before batch.js, which owns the grid; everything here runs only once both have loaded.

const VIEW_UI_KEY = "llm-classifier-batch-view";

const view = {
  json: new Map(),  // fp -> result JSON for the selected model/workflow
  exif: new Map(),  // fp -> flat EXIF tags; cached in IndexedDB and shared by every model/workflow
  order: [],        // item indices in display order, each once (for prev/next in focused mode)
  filtered: false,  // true when the search or status filter hides some photos
  loadSeq: 0,
  exifReading: false,
  fieldsStale: false, // a new result may have fields the sort and group menus don't list yet
};

const f = {
  bar: $("filterBar"),
  search: $("batchSearch"),
  status: $("batchStatus"),
  sort: $("batchSort"),
  dir: $("batchSortDir"),
  group: $("batchGroup"),
  exifBtn: $("batchExif"),
  helpBtn: $("batchSearchHelp"),
  help: $("searchHelp"),
};

// ---------- per-photo searchable fields ----------
// Every leaf of the result JSON, keyed by its full path and by each shorter suffix, so
// "category" finds classification.category. Array items share their array's path.
function leaves(value, path = "", out = []) {
  if (value === null || value === undefined) return out;
  if (Array.isArray(value)) value.forEach((v) => leaves(v, path, out));
  else if (typeof value === "object") {
    for (const [k, v] of Object.entries(value)) leaves(v, path ? `${path}.${k}` : k, out);
  } else out.push([path, value]);
  return out;
}

// A result JSON as search fields, { fields: lowercase field -> values, text }; more(add) adds
// fields of its own. Also used for video scenes' tags (video-tag.js).
function searchEntry(json, more = null) {
  const fields = new Map();
  const add = (key, value) => {
    const k = key.toLowerCase();
    if (!fields.has(k)) fields.set(k, []);
    fields.get(k).push(value);
  };
  for (const [path, value] of leaves(json)) {
    const parts = path.split(".");
    for (let i = 0; i < parts.length; i++) add(parts.slice(i).join("."), value);
    add(`json.${path}`, value);
  }
  more?.(add);
  const text = [...fields.values()].flat().map((v) => String(v).toLowerCase()).join("\n");
  return { fields, text };
}

function entryOf(item) {
  if (item.entry) return item.entry;
  return (item.entry = searchEntry(view.json.get(item.fp), (add) => {
    for (const [tag, value] of Object.entries(view.exif.get(item.fp) || {})) {
      for (const [, v] of leaves(value)) {
        add(`exif.${tag}`, v);
        add(tag, v);
      }
    }
    add("name", item.file.name);
    add("path", item.path);
  }));
}

// "status" changes as runs finish, so it is read live rather than cached in the entry
function fieldValues(item, field) {
  return field === "status" ? [tileState(item)] : entryOf(item).fields.get(field);
}

function invalidateEntries() {
  batch.items.forEach((item) => (item.entry = null));
}

// ---------- query ----------
// Terms are ANDed: free text, field:value (contains), field=value (exact), field>n, field<n,
// field>=n, field<=n, has:field, and a leading "-" negates. Quote values with spaces.
const TERM_RE = /(-?)(?:([A-Za-z_][\w.-]*)(>=|<=|:|=|>|<))?(?:"([^"]*)"|(\S+))/g;

function parseQuery(q) {
  const terms = [];
  for (const m of q.matchAll(TERM_RE)) {
    const value = (m[4] ?? m[5] ?? "").toLowerCase();
    if (!value && !m[2]) continue;
    terms.push({ neg: m[1] === "-", field: m[2]?.toLowerCase() || null, op: m[3] || null, value });
  }
  return terms;
}

const toNum = (v) => (typeof v === "number" ? v : /^-?\d+(\.\d+)?$/.test(String(v).trim()) ? Number(v) : null);

function compareValues(a, b) {
  const x = toNum(a);
  const y = toNum(b);
  if (x !== null && y !== null) return x - y;
  // EXIF dates (2024:05:01 10:00:00) sort correctly as text
  return String(a).localeCompare(String(b), undefined, { numeric: true, sensitivity: "base" });
}

function termMatches(item, t) {
  return entryMatches(entryOf(item), (field) => fieldValues(item, field), t);
}

// values(field) lists a field's values, including live ones such as status
function entryMatches(entry, values, t) {
  if (!t.field) return entry.text.includes(t.value);
  if (t.field === "has") return !!values(t.value);
  return (values(t.field) || []).some((v) => {
    const s = String(v).toLowerCase();
    if (t.op === ":") return s.includes(t.value);
    if (t.op === "=") return s === t.value;
    const c = compareValues(v, t.value);
    return t.op === ">" ? c > 0 : t.op === "<" ? c < 0 : t.op === ">=" ? c >= 0 : c <= 0;
  });
}

// ---------- applying the view ----------
function applyView() {
  if (view.fieldsStale) rebuildFieldOptions();
  const terms = parseQuery(f.search.value);
  const status = f.status.value;
  let shown = batch.items.map((_, i) => i).filter((i) => {
    const item = batch.items[i];
    if (status && tileState(item) !== status) return false;
    return terms.every((t) => termMatches(item, t) !== t.neg);
  });
  view.filtered = shown.length < batch.items.length;

  const values = (i, field) => fieldValues(batch.items[i], field);
  shown = sortIndices(shown, f.sort.value, f.dir.dataset.dir, values);
  renderTiles(groupIndices(shown, f.group.value, values));
  saveViewUi();
  refreshTiles();
}

// Sorting and grouping take values(i, field), a field's values for item i, so the video side's
// scene grid (video-tag.js) uses them too.
function sortIndices(indices, field, dir, values) {
  if (!field) return indices;
  const sign = dir === "desc" ? -1 : 1;
  const key = (i) => values(i, field)?.[0];
  return indices.slice().sort((a, b) => {
    const ka = key(a);
    const kb = key(b);
    if (ka === undefined || kb === undefined) return (ka === undefined) - (kb === undefined); // missing last
    return sign * compareValues(ka, kb) || a - b;
  });
}

// [{label, indices}]; with no grouping, one unlabelled group
function groupIndices(indices, field, valuesOf) {
  if (!field) return [{ label: null, indices }];
  const groups = new Map();
  const none = [];
  for (const i of indices) {
    const values = valuesOf(i, field);
    if (!values?.length) {
      none.push(i);
      continue;
    }
    // An item appears once in each of its values' groups (e.g. once per tag)
    for (const v of new Set(values.map((x) => String(x)))) {
      const k = v.toLowerCase();
      if (!groups.has(k)) groups.set(k, { label: v, indices: [] });
      groups.get(k).indices.push(i);
    }
  }
  const list = [...groups.values()];
  const numeric = list.every((g) => toNum(g.label) !== null);
  list.sort(numeric
    ? (a, b) => toNum(a.label) - toNum(b.label)
    : (a, b) => b.indices.length - a.indices.length || compareValues(a.label, b.label));
  if (none.length) list.push({ label: "(no value)", indices: none });
  return list;
}

function renderTiles(groups) {
  const nodes = [];
  const used = new Set();
  view.order = [];
  batch.items.forEach((item) => (item.clones = []));
  for (const g of groups) {
    if (g.label !== null) {
      nodes.push(h("div", { class: "tile-group" }, h("strong", {}, g.label), h("span", { class: "meta" }, ` ${g.indices.length}`)));
    }
    for (const i of g.indices) {
      const item = batch.items[i];
      if (!used.has(i)) {
        used.add(i);
        view.order.push(i);
        nodes.push(item.tile);
      } else {
        const clone = item.tile.cloneNode(true);
        clone.addEventListener("click", () => setFocus(i));
        item.clones.push(clone);
        nodes.push(clone);
      }
    }
  }
  if (!view.order.length && batch.items.length) nodes.push(h("div", { class: "tile-group empty" }, "No photos match."));
  b.tiles.replaceChildren(...nodes);
}

// ---------- data: results for the selected model/workflow, cached EXIF ----------
async function loadViewData() {
  const seq = ++view.loadSeq;
  const items = batch.items;
  const key = configKey();
  const withResult = key ? items.filter((item) => batch.status.has(resultKey(item, key))) : [];
  const [records, exif] = await Promise.all([
    resultStore.getMany(withResult.map((item) => resultKey(item, key))),
    resultStore.getExif(items.map((item) => item.fp)),
  ]);
  if (seq !== view.loadSeq || items !== batch.items) return;
  view.json = new Map(withResult.map((item, n) => [item.fp, recordResult(records[n])]));
  view.exif = exif;
  invalidateEntries();
  rebuildFieldOptions();
  updateExifButton();
  applyView();
}

// Called by batch.js when a run saves a new result
function noteResult(item, json) {
  view.json.set(item.fp, json);
  item.entry = null;
  view.fieldsStale = true;
}

async function rememberExif(item, data) {
  const flat = flattenExif(data);
  if (data.location) {
    flat.GPSLatitudeDecimal = data.location.lat;
    flat.GPSLongitudeDecimal = data.location.lon;
  }
  const isNew = !view.exif.has(item.fp);
  view.exif.set(item.fp, flat);
  item.entry = null;
  await resultStore.putExif(item.fp, flat);
  if (isNew && !view.exifReading) {
    rebuildFieldOptions();
    updateExifButton();
  }
}

function updateExifButton() {
  const missing = batch.items.filter((item) => !view.exif.has(item.fp)).length;
  f.exifBtn.hidden = !missing && !view.exifReading;
  if (!view.exifReading) f.exifBtn.textContent = `read EXIF (${missing})`;
  f.exifBtn.title = "Upload each photo once to read its EXIF so you can search, sort and group by it. Cached in this browser.";
}

f.exifBtn.addEventListener("click", async () => {
  if (view.exifReading) return;
  const items = batch.items;
  const missing = items.filter((item) => !view.exif.has(item.fp));
  view.exifReading = true;
  let done = 0;
  const worker = async () => {
    while (missing.length && items === batch.items) {
      const item = missing.shift();
      try {
        await rememberExif(item, await fetchExif(item.file));
      } catch { /* skip unreadable files */ }
      f.exifBtn.textContent = `reading EXIF ${++done}…`;
    }
  };
  try {
    await Promise.all([worker(), worker()]);
  } finally {
    view.exifReading = false;
  }
  rebuildFieldOptions();
  updateExifButton();
  applyView();
});

// Sort and group offer every field seen in this folder's results and EXIF
function rebuildFieldOptions() {
  view.fieldsStale = false;
  const jsonPaths = new Set();
  for (const json of view.json.values()) for (const [path] of leaves(json)) jsonPaths.add(path);
  const exifTags = new Set();
  for (const flat of view.exif.values()) Object.keys(flat).forEach((t) => exifTags.add(t));
  const fixed = [["name", "File name"], ["status", "Status"]];
  fillFieldSelect(f.sort, "Folder order", fixed, jsonPaths, exifTags);
  fillFieldSelect(f.group, "No grouping", fixed, jsonPaths, exifTags);
}

// A sort or group menu: none (first), fixed [value, text] fields, then result paths and EXIF tags
function fillFieldSelect(select, first, fixed, jsonPaths, exifTags = new Set()) {
  const keep = select.value;
  const opt = (value, text) => h("option", { value }, text);
  const sorted = (set) => [...set].sort((a, b) => a.localeCompare(b));
  select.replaceChildren(
    opt("", first),
    ...fixed.map(([value, text]) => opt(value, text)),
    jsonPaths.size ? h("optgroup", { label: "Result" }, sorted(jsonPaths).map((p) => opt(p.toLowerCase(), p))) : null,
    exifTags.size ? h("optgroup", { label: "EXIF" }, sorted(exifTags).map((t) => opt(`exif.${t.toLowerCase()}`, t))) : null,
  );
  // Keep a saved choice even before its field has data (e.g. EXIF not read yet)
  if (keep && ![...select.options].some((o) => o.value === keep)) select.append(opt(keep, keep));
  select.value = keep;
}

// ---------- UI ----------
function saveViewUi() {
  try {
    localStorage.setItem(VIEW_UI_KEY, JSON.stringify({
      search: f.search.value, status: f.status.value, sort: f.sort.value, dir: f.dir.dataset.dir, group: f.group.value,
    }));
  } catch { /* storage unavailable */ }
}

function setSortDir(dir) {
  f.dir.dataset.dir = dir;
  f.dir.textContent = dir === "desc" ? "↓" : "↑";
  f.dir.title = dir === "desc" ? "Descending (click for ascending)" : "Ascending (click for descending)";
}

let searchTimer = null;
f.search.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(applyView, 150);
});
[f.status, f.sort, f.group].forEach((x) => x.addEventListener("change", applyView));
f.dir.addEventListener("click", () => {
  setSortDir(f.dir.dataset.dir === "desc" ? "asc" : "desc");
  applyView();
});
f.helpBtn.addEventListener("click", () => (f.help.hidden = !f.help.hidden));

(() => {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(VIEW_UI_KEY)) || {};
  } catch { /* storage unavailable */ }
  f.search.value = saved.search || "";
  f.status.value = saved.status || "";
  setSortDir(saved.dir === "desc" ? "desc" : "asc");
  rebuildFieldOptions();
  for (const [select, value] of [[f.sort, saved.sort], [f.group, saved.group]]) {
    if (!value) continue;
    if (![...select.options].some((o) => o.value === value)) select.append(h("option", { value }, value));
    select.value = value;
  }
})();

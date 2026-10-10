"use strict";
// Transcript panel on the video side's Scenes view: runs Whisper on the open video's speech
// (video/transcribe.py), lists the timed lines, follows the player and downloads subtitles.
// video.js calls onSpeechStatus with each status, adds speechTracks to its comparison rows and
// speechHoverText to the timeline's hover; the scene export (speechExportKey) and Tag scenes'
// workflow prompts (speechForScenes) get each scene's words from the transcript shown here.

const SPEECH_PAGE = 300; // lines added at a time as the list scrolls
const FOLLOW_PAUSE_MS = 4000; // after the list is scrolled by hand, it stops following the player

const se = {
  model: $("vsModel"),
  language: $("vsLanguage"),
  track: $("vsTrack"),
  run: $("vsRun"),
  search: $("vsSearch"),
  list: $("vsList"),
  count: $("vsCount"),
  exports: document.querySelectorAll("[data-vs-export]"),
};

const sp = {
  models: [],
  active: null, // key of the transcript shown
  data: new Map(), // "<video>/<key>" -> transcript
  lines: [], // indices of the shown transcript's segments that match the search
  rendered: 0,
  playing: -1, // segment index under the playhead
  followed: -1, // segment last scrolled into view while playing
  scrolledAt: -Infinity, // when the list was last scrolled by hand
  autoScroll: false,
};

const languageNames = (() => {
  try {
    const names = new Intl.DisplayNames([navigator.language || "en"], { type: "language" });
    return (code) => {
      try {
        return names.of(code) || code;
      } catch {
        return code;
      }
    };
  } catch {
    return (code) => code;
  }
})();

// ---------- settings ----------
async function loadSpeechModels() {
  const data = await vFetch("/api/video/speech-models");
  sp.models = data.models;
  const saved = vid.ui.speech || {};
  fillSelect(se.model, data.models.map((m) => ({
    value: m.id,
    text: `${m.label.replace(/^Whisper /, "")} · ${fmtBytes(m.size_mb * 1e6)}`,
    title: `${m.label}: ${m.note}${m.downloaded ? "" : " Downloads on first use."}`,
  })), saved.model || data.default);
  const langs = data.languages.map((c) => ({ value: c, text: languageNames(c) }))
    .sort((a, b) => a.text.localeCompare(b.text));
  fillSelect(se.language, [{ value: "", text: "Detect language" }, ...langs], saved.language || "");
}

function speechParams() {
  return { model: se.model.value, language: se.language.value, track: Number(se.track.value) || 0 };
}

const sameSpeech = (a, b) => a?.model === b?.model && (a?.language || "") === (b?.language || "") && (a?.track || 0) === (b?.track || 0);

// The finished transcript for the model, language and track chosen in the panel, if any
function selectedTranscript() {
  const params = speechParams();
  return (vid.status?.transcripts || []).find((t) => sameSpeech(t.params, params));
}

function onSpeechSettings() {
  vid.ui.speech = speechParams();
  saveVideoUi(); // video.js
  const t = selectedTranscript();
  if (t) showTranscript(t.key);
  else showNoTranscript();
  updateSpeechButtons();
}

[se.model, se.language, se.track].forEach((el) => el.addEventListener("change", onSpeechSettings));

// One option per audio track, shown only when there is more than one
function renderTracks(info) {
  const tracks = info?.audio || [];
  se.track.hidden = tracks.length < 2;
  if (tracks.length < 2) {
    if (se.track.options.length) se.track.replaceChildren();
    return;
  }
  const want = String(vid.ui.speech?.track ?? se.track.value ?? 0);
  fillSelect(se.track, tracks.map((a, i) => ({
    value: String(i),
    text: `Track ${i + 1}${a.language && a.language !== "und" ? ` · ${languageNames(a.language)}` : ""}`,
    title: [a.codec, a.channels && `${a.channels} channels`].filter(Boolean).join(", "),
  })), want);
}

// ---------- status ----------
// Called by applyStatus in video.js after every poll
function onSpeechStatus(s, prev) {
  renderTracks(s.info);
  const same = prev?.id === s.id;
  const before = new Set((same ? prev.transcripts : []).map((t) => t.key));
  const added = (s.transcripts || []).filter((t) => !before.has(t.key));
  const selected = selectedTranscript();
  if (!sp.active || !s.transcripts.some((t) => t.key === sp.active)) {
    const pick = selected || (same ? null : s.transcripts[s.transcripts.length - 1]);
    if (pick) selectTranscript(pick);
    else showNoTranscript();
  } else if (selected && added.some((t) => t.key === selected.key)) {
    showTranscript(selected.key);
  }
  if (same) {
    for (const t of added) {
      const lang = t.language ? ` (${languageNames(t.language)})` : "";
      vStatus(`${t.label}: ${t.count} line${t.count === 1 ? "" : "s"}, ${t.words} words${lang} in ${formatDuration(t.elapsed * 1000)}.`);
    }
  }
  updateSpeechButtons();
}

function speechJob() {
  const key = selectedKeyGuess();
  return (vid.status?.jobs || []).find((j) => j.kind === "transcribe" && j.key === key);
}

// The key the server gives these settings (speech.key in video/transcribe.py)
function selectedKeyGuess() {
  const p = speechParams();
  return `whisper-${p.model}-${p.language || "auto"}${p.track ? `-a${p.track}` : ""}`;
}

function updateSpeechButtons() {
  const s = vid.status;
  const noAudio = s?.info && !s.info.audio.length;
  const job = speechJob();
  se.run.disabled = !s || s.state === "uploading" || noAudio || !se.model.value || !!job || !!selectedTranscript();
  se.run.textContent = job ? (job.state === "queued" ? "Queued…" : "Transcribing…") : "Transcribe";
  se.run.title = noAudio ? "This video has no audio track."
    : selectedTranscript() ? "Already transcribed with these settings; shown below."
      : "Transcribe the video's speech with timestamps";
  se.exports.forEach((b) => { b.disabled = !activeTranscript(); });
}

// ---------- transcripts ----------
function activeTranscript() {
  return sp.active ? sp.data.get(`${vid.id}/${sp.active}`) : null;
}

async function loadTranscript(key) {
  const cacheKey = `${vid.id}/${key}`;
  if (!sp.data.has(cacheKey)) {
    sp.data.set(cacheKey, await vFetch(`/api/video/${vid.id}/transcripts/${key}`));
    while (sp.data.size > 4) sp.data.delete(sp.data.keys().next().value); // a long one holds every word
  }
  return sp.data.get(cacheKey);
}

async function showTranscript(key) {
  const id = vid.id;
  sp.active = key;
  let t;
  try {
    t = await loadTranscript(key);
  } catch (e) {
    vStatus(`Could not load the transcript: ${e.message}`, "error");
    return;
  }
  if (id !== vid.id || sp.active !== key) return;
  sp.playing = -1;
  sp.followed = -1;
  renderLines();
  renderCompare(); // video.js
  updateSpeechButtons();
}

// Show a finished transcript and set the panel to the settings that made it
function selectTranscript(t) {
  se.model.value = t.params.model;
  se.language.value = t.params.language || "";
  if (!se.track.hidden) se.track.value = String(t.params.track || 0);
  vid.ui.speech = speechParams();
  saveVideoUi();
  showTranscript(t.key);
}

function showNoTranscript() {
  sp.active = null;
  sp.lines = [];
  sp.playing = -1;
  sp.followed = -1;
  se.count.textContent = "";
  const s = vid.status;
  const msg = !vid.id ? "Choose a video, then press Transcribe."
    : s?.info && !s.info.audio.length ? "This video has no audio track."
      : speechJob() ? "The transcript appears here when it's done."
        : "No transcript with these settings yet. Press Transcribe.";
  se.list.replaceChildren(h("span", { class: "empty" }, msg));
  updateSpeechButtons();
}

// Called by openVideo and closeVideo in video.js
function resetSpeech() {
  sp.active = null;
  showNoTranscript();
}

// ---------- the list ----------
function searchTerms() {
  return se.search.value.toLowerCase().split(/\s+/).filter(Boolean);
}

function renderLines() {
  const t = activeTranscript();
  if (!t) return;
  const terms = searchTerms();
  sp.lines = [];
  t.segments.forEach((seg, i) => {
    const text = seg.text.toLowerCase();
    if (terms.every((w) => text.includes(w))) sp.lines.push(i);
  });
  const lang = t.language ? ` · ${languageNames(t.language)}` : "";
  se.count.textContent = terms.length
    ? `${sp.lines.length} of ${t.segments.length} lines`
    : `${t.segments.length} lines · ${t.label.replace(/^Whisper /, "")}${lang}`;
  sp.rendered = 0;
  sp.followed = -1;
  se.list.replaceChildren();
  se.list.scrollTop = 0;
  if (!sp.lines.length) {
    se.list.append(h("span", { class: "empty" }, terms.length ? "No lines match." : "No speech found."));
    return;
  }
  appendLines(SPEECH_PAGE);
}

const moreLines = new IntersectionObserver((entries) => {
  if (entries.some((e) => e.isIntersecting)) appendLines(sp.rendered + SPEECH_PAGE);
}, { root: se.list, rootMargin: "400px" });

function appendLines(upto) {
  const t = activeTranscript();
  if (!t) return;
  upto = Math.min(upto, sp.lines.length);
  se.list.querySelector(".v-more")?.remove();
  const terms = searchTerms();
  const rows = [];
  for (let n = sp.rendered; n < upto; n++) {
    const i = sp.lines[n];
    const seg = t.segments[i];
    rows.push(h("div", { class: "v-line" + (i === sp.playing ? " playing" : ""), dataset: { i }, title: `${fmtTime(seg.start)} – ${fmtTime(seg.end)}` },
      h("span", { class: "v-line-time" }, fmtTime(seg.start).replace(/\.\d+$/, "")),
      h("span", { class: "v-line-text" }, highlight(seg.text, terms))));
  }
  se.list.append(...rows);
  sp.rendered = upto;
  if (upto < sp.lines.length) {
    const more = h("div", { class: "v-more meta" }, `${sp.lines.length - upto} more…`);
    se.list.append(more);
    moreLines.observe(more);
  }
}

// The text with each search word marked
function highlight(text, terms) {
  if (!terms.length) return text;
  const re = new RegExp(`(${terms.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  return text.split(re).map((part, i) => (i % 2 ? h("mark", {}, part) : part));
}

let speechSearchTimer = 0;
se.search.addEventListener("input", () => {
  clearTimeout(speechSearchTimer);
  speechSearchTimer = setTimeout(renderLines, 150);
});

se.list.addEventListener("click", (e) => {
  const row = e.target.closest(".v-line");
  const seg = row && activeTranscript()?.segments[Number(row.dataset.i)];
  if (!seg || ve.player.hidden) return;
  setLoop(null); // video.js
  ve.player.currentTime = seg.start;
  ve.player.play().catch(() => { /* autoplay blocked: the user can press play */ });
});

se.list.addEventListener("scroll", () => {
  if (sp.autoScroll) sp.autoScroll = false;
  else sp.scrolledAt = performance.now();
}, { passive: true });

// ---------- following the player ----------
// Index of the segment spoken at time t, or -1 between segments
function segmentAt(segments, t) {
  let lo = 0;
  let hi = segments.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (segments[mid].start <= t) lo = mid;
    else hi = mid - 1;
  }
  const seg = segments[lo];
  return seg && seg.start <= t && t < seg.end ? lo : -1;
}

ve.player.addEventListener("timeupdate", () => {
  const t = activeTranscript();
  if (!t) return;
  const i = segmentAt(t.segments, ve.player.currentTime);
  if (i !== sp.playing) {
    se.list.querySelector(".v-line.playing")?.classList.remove("playing");
    se.list.querySelector(`.v-line[data-i="${i}"]`)?.classList.add("playing");
    sp.playing = i;
  }
  // Scrolled to once per line while playing, so a line highlighted after a seek while paused is
  // brought into view when playback starts
  if (i < 0 || i === sp.followed || ve.player.paused || performance.now() - sp.scrolledAt < FOLLOW_PAUSE_MS) return;
  sp.followed = i;
  let row = se.list.querySelector(`.v-line[data-i="${i}"]`);
  if (!row) {
    const n = sp.lines.indexOf(i);
    if (n < sp.rendered) return; // filtered out by the search
    appendLines(n + SPEECH_PAGE);
    row = se.list.querySelector(`.v-line[data-i="${i}"]`); // appendLines marks it playing
  }
  if (!row) return;
  const r = row.getBoundingClientRect();
  const box = se.list.getBoundingClientRect();
  if (r.top < box.top || r.bottom > box.bottom) {
    sp.autoScroll = true;
    row.scrollIntoView({ block: "center" });
  }
});
ve.player.addEventListener("pause", () => { sp.followed = -1; });

// ---------- run, delete, export ----------
async function runTranscribe() {
  if (!vid.id || se.run.disabled) return;
  const params = speechParams();
  try {
    const r = await vPost(`/api/video/${vid.id}/transcribe`, params);
    if (r.cached) {
      await pollNow(); // video.js
      showTranscript(r.key);
      return;
    }
    const m = sp.models.find((x) => x.id === params.model);
    vStatus(`${m?.label || "Whisper"} queued.${m && !m.downloaded ? ` It downloads first (${fmtBytes(m.size_mb * 1e6)}).` : ""}`);
    if (m) m.downloaded = true; // only the first run downloads
  } catch (e) {
    vStatus(e.message, "error");
  }
  await pollNow();
  if (!sp.active) showNoTranscript();
}

se.run.addEventListener("click", runTranscribe);

async function deleteTranscript(t) {
  try {
    await vFetch(`/api/video/${vid.id}/transcripts/${t.key}`, { method: "DELETE" });
  } catch (e) {
    vStatus(`Could not delete: ${e.message}`, "error");
    return;
  }
  sp.data.delete(`${vid.id}/${t.key}`);
  if (sp.active === t.key) showNoTranscript();
  await pollNow();
}

se.exports.forEach((b) => b.addEventListener("click", () => {
  if (sp.active) download(`/api/video/${vid.id}/transcripts/${sp.active}/export?format=${b.dataset.vsExport}`); // video.js
}));

// ---------- shared with video.js and video-tag.js ----------
// One row per transcript under the timeline, beside the detectors' rows: where there is speech
function speechTracks() {
  const rows = [];
  for (const t of vid.status?.transcripts || []) {
    const lang = t.language ? ` · ${t.language}` : "";
    const del = h("button", { type: "button", title: "Delete this transcript" }, "×");
    const row = h("div", {
      class: "v-track v-track-speech" + (t.key === sp.active ? " active" : ""),
      title: `${t.label}: ${t.count} lines, ${t.words} words. Click to show.`,
    },
    h("span", { class: "v-track-label" },
      h("strong", {}, "Speech"),
      h("span", { class: "meta" }, `${t.label.replace(/^Whisper /, "")}${lang}${t.params.track ? ` · track ${t.params.track + 1}` : ""} · ${t.count} lines`)),
    h("canvas", {}), del);
    row.addEventListener("click", () => selectTranscript(t));
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteTranscript(t);
    });
    rows.push({ row, canvas: row.querySelector("canvas"), spans: t.spans, active: t.key === sp.active });
  }
  return rows;
}

function speechHoverText(t) {
  const tr = activeTranscript();
  const i = tr ? segmentAt(tr.segments, t) : -1;
  if (i < 0) return "";
  const text = tr.segments[i].text;
  return ` · “${text.length > 70 ? `${text.slice(0, 68)}…` : text}”`;
}

function speechExportKey() {
  return activeTranscript() ? sp.active : "";
}

// The words spoken in each scene, each word in the scene its middle falls in (speech_by_scene
// in video/transcribe.py), or null with no transcript shown
function speechForScenes(scenes) {
  const t = activeTranscript();
  if (!t) return null;
  const out = scenes.map(() => []);
  for (const seg of t.segments) {
    const words = seg.words?.length ? seg.words : [{ start: seg.start, end: seg.end, word: ` ${seg.text}` }];
    for (const w of words) {
      const mid = (w.start + w.end) / 2;
      let lo = 0;
      let hi = scenes.length - 1;
      while (lo < hi) {
        const m = (lo + hi + 1) >> 1;
        if (scenes[m].start <= mid) lo = m;
        else hi = m - 1;
      }
      if (scenes[lo] && scenes[lo].start <= mid) out[lo].push(w.word);
    }
  }
  return out.map((ws) => ws.join("").trim());
}

async function initSpeech() {
  try {
    await loadSpeechModels();
  } catch (e) {
    se.list.replaceChildren(h("span", { class: "empty" }, `Transcription is unavailable: ${e.message}`));
  }
  updateSpeechButtons();
}

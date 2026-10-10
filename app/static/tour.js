"use strict";
// Walkthrough: on first load, dims the page and highlights each control in turn with a note on
// what to pick and when to change it. Visits all three tabs and the video side, then puts the
// tab, backend and side back.
// The ? button in the header replays it. Shares globals with app.js, workflows.js and batch.js; loads last.

const TOUR_KEY = "llm-classifier-tour";
const TOUR_VERSION = "2"; // bump to show the tour again after big UI changes

// target: selector(s) to highlight; each grows to its <label> so the caption lights up too.
// mode / backend: the tab and backend the step needs (left as they are when absent).
// app: "video" for steps on the video side; steps with a mode are on the image side.
// vmode: the video side's view a step needs (Scenes or Tag scenes).
// text is trusted HTML written here, never user data.
const TOUR_STEPS = [
  {
    mode: "single", backend: "llm",
    title: "Welcome",
    text: "This app classifies photos with local models and can write the results into the photos' metadata. " +
      "This tour shows what each setting does and when to change it, across all three tabs and the video side. " +
      "It takes about two minutes, and <strong>Skip tour</strong> ends it at any time.",
  },
  {
    target: ".settings .tabs", mode: "single",
    title: "Three ways to work",
    text: "<strong>Single model</strong>: try one model on one photo while you tune the settings.<br>" +
      "<strong>Workflows</strong>: chain several models, so each does what it's best at.<br>" +
      "<strong>Batch</strong>: run a whole folder once the results look right.",
  },
  {
    target: "#backend", mode: "single",
    title: "Backend",
    text: "<strong>LLM</strong> sends the photo to a vision LLM with your own prompt: the most flexible, free-form answers, but slower. " +
      "<strong>Vision model</strong> uses dedicated image models: fast and repeatable labels or object boxes, with no LLM server needed.",
  },
  {
    target: ["#baseUrl", "#apiKey"], mode: "single", backend: "llm",
    title: "Endpoint and API key",
    text: "Where your OpenAI-compatible server (such as llama-server) is listening. " +
      "Change it if the server runs on another machine or port. " +
      "Leave the API key empty unless your server requires one.",
  },
  {
    target: "#modelSelect", mode: "single", backend: "llm",
    title: "Model",
    text: "Pick a model that can see images. llama-server needs the model's <code>mmproj</code> file for that. " +
      "Press ↻ after loading new models on the server to refresh the list.",
  },
  {
    target: ["#temperature", "#maxTokens"], mode: "single", backend: "llm",
    title: "Temp and Max tokens",
    text: "Keep the temperature low (0–0.3) so the same photo gets the same labels; raise it only for more varied descriptions. " +
      "Raise <strong>Max tokens</strong> if answers get cut off. Models that think before answering need much more.",
  },
  {
    target: "#jsonMode", mode: "single", backend: "llm",
    title: "Force JSON",
    text: "Makes the server return valid JSON only. Turn it on if the JSON panel can't parse the model's answer, " +
      "for example when the model wraps it in prose.",
  },
  {
    target: "#withYolo", mode: "single", backend: "llm",
    title: "Object overlay",
    text: "Also runs the object detector and draws its boxes on the photo. " +
      "Useful for checking what's in the picture, but each run takes longer.",
  },
  {
    target: [".panels.single-only .panel-head.llm-only", "#systemPrompt", "#userPrompt"], mode: "single", backend: "llm",
    title: "Prompts",
    text: "The system prompt decides what you get back: edit it to add or remove JSON fields, or to change the categories. " +
      "Keep the <code>category</code>, <code>tags</code> and <code>confidence</code> fields so results compare with the vision models. " +
      "<strong>reset</strong> restores the default.",
  },
  {
    target: ["#visionModel", "#topK", "#threshold"], mode: "single", backend: "vision",
    title: "Vision model",
    text: "<strong>SigLIP 2</strong> scores your own labels (next step). " +
      "<strong>EfficientNet</strong> and <strong>ResNet</strong> pick from 1,000 fixed ImageNet classes. " +
      "<strong>RT-DETR</strong> finds and counts objects. " +
      "<strong>Top-k</strong> and <strong>Threshold</strong> trim the list; leave Threshold empty for the model's default.",
  },
  {
    target: [".panels.single-only .panel-head.vision-only", "#labels", "#labelTemplate"], mode: "single", backend: "vision",
    title: "Labels",
    text: "SigLIP compares the photo with each line here. Replace them with your own categories, one per line. " +
      "The template under the labels turns each one into a sentence, which usually scores better than a bare word.",
  },
  {
    target: "#dropZone", mode: "single", backend: "llm",
    title: "Your photo",
    text: "Drop a photo here or click to choose one. It stays on your computer: models get a copy scaled to at most 1536 px.",
  },
  {
    target: "#exifPanel", mode: "single",
    title: "EXIF / metadata",
    text: "Every metadata tag, read from the original file. Filter it, copy it as JSON, " +
      "or follow the map link when the photo has GPS data.",
  },
  {
    target: ["#autoRun", "#runBtn"], mode: "single",
    title: "Run",
    text: "<strong>Run</strong> (or Cmd/Ctrl+Enter) sends the photo to the model; it becomes <strong>Stop</strong> while running. " +
      "Tick <strong>Auto-run on drop</strong> to run every photo as soon as you drop it.",
  },
  {
    target: "#outputPanel", mode: "single",
    title: "Results",
    text: "<strong>JSON output</strong> is the parsed result; the other tab shows the model's raw answer. " +
      "<strong>Export image</strong> downloads a copy of the photo with the result written into its metadata. " +
      "Tick <strong>Overwrite</strong> only if you want to replace tags the photo already has.",
  },
  {
    target: "#exportSettings", mode: "single",
    title: "Export settings",
    text: "Saves these settings as a file for the command-line tool, so a large folder can run without the browser open.",
  },
  {
    target: "#wfSelect", mode: "workflow",
    title: "Workflows",
    text: "Pick a workflow; three starters are included. " +
      "<strong>Duplicate</strong> one before experimenting so the original stays as it is, or press <strong>New</strong> to start empty.",
  },
  {
    target: ".wf-panels > .panel:first-child", mode: "workflow",
    title: "Steps",
    text: "Steps run in order. Add vision steps for facts (objects, scene) and an LLM step to interpret them: " +
      "<code>{{step_id.field}}</code> in its prompt inserts an earlier result. " +
      "LLM steps use the endpoint set on the Single model tab.",
  },
  {
    target: "#wfSave", mode: "workflow",
    title: "Save",
    text: "<strong>Run</strong> always uses your current edits, so you can try changes first. " +
      "<strong>Save</strong> keeps them on the server.",
  },
  {
    target: "#batchKind", mode: "batch",
    title: "Run with",
    text: "<strong>Single model</strong> uses the settings from the Single model tab; <strong>Workflow</strong> uses the selected workflow. " +
      "Get one photo right first, then run the folder.",
  },
  {
    target: "#batchPick", mode: "batch",
    title: "Choose folder",
    text: "Shows every photo in a folder and its subfolders. " +
      "Photos stay on your computer; each one is sent, scaled down, only when it runs.",
  },
  {
    target: ".batch-grid", mode: "batch",
    title: "The grid",
    text: "Each border shows the state for the selected model or workflow: grey not run, green done, red failed. " +
      "Results are saved in this browser, so picking the same folder again brings them back. Click a photo to see its result.",
  },
  {
    target: "#batchRerun", mode: "batch",
    title: "Re-run done",
    text: "<strong>Run all</strong> skips green photos. Tick this after changing the prompt or settings to run those again too.",
  },
  {
    target: "#batchExport", mode: "batch",
    title: "Export all",
    text: "Downloads tagged copies of every green photo as one zip, keeping names and subfolders. " +
      "If the page closes before the zip arrives, it's offered again next time.",
  },
  {
    target: "#appKind", app: "video", vmode: "scenes",
    title: "Video Classifier",
    text: "Switch here between photos and videos. The video side splits a video into scenes and exports the scene list with thumbnails. " +
      "Each side keeps its own settings.",
  },
  {
    target: ["#vPick", "#vRecent"], app: "video",
    title: "Choose a video",
    text: "Any length or size: it uploads in 64 MB chunks, so a 40 GB file never has to fit in memory. " +
      "If the upload stops, choose the same file again to resume it. " +
      "Videos stay on the server for a week after you last open them; pick them again from the <strong>Video</strong> list.",
  },
  {
    target: ["#vDetector", "#vParams"], app: "video", vmode: "scenes",
    title: "Detector",
    text: "<strong>TransNetV2</strong> is the most accurate, and catches dissolves and fades. " +
      "<strong>PySceneDetect</strong> is quick and easy to tune: <em>content</em> for hard cuts, <em>adaptive</em> for fast motion, " +
      "<em>fades</em> for fades to black. <strong>FFmpeg</strong> gives a rough split. " +
      "Every result is kept, so switching back to one is instant.",
  },
  {
    target: ".v-timeline-panel", app: "video", vmode: "scenes",
    title: "Timeline",
    text: "Scenes as blocks, with the detector's score below and its threshold as a dashed line. " +
      "<strong>Drag the line</strong> to get more or fewer scenes: TransNetV2 and FFmpeg update as you drag, " +
      "PySceneDetect re-runs when you let go. One row per result lets you compare runs; click a row to show it. " +
      "Click a scene in the list to loop it.",
  },
  {
    target: ".v-speech-panel", app: "video", vmode: "scenes",
    title: "Transcript",
    text: "<strong>Transcribe</strong> turns the speech into timed lines with Whisper, running on the server. " +
      "<em>small</em> is a good start; larger models are more accurate and slower, and each downloads on first use. " +
      "The line being spoken follows the video; click one to jump there. SRT and VTT are subtitles, TXT has a time on each line. " +
      "Each scene's words go into <strong>Export scenes</strong>, and into workflow prompts as <code>{{exif.Speech}}</code>.",
  },
  {
    target: "#vExport", app: "video", vmode: "scenes",
    title: "Export",
    text: "Downloads the scene list as a zip: <code>scenes.json</code> and <code>scenes.csv</code> with each scene's times, " +
      "timecodes and frames, and a thumbnail of every scene. The video itself is never cut. " +
      "The CSV and JSON links above the scene list download the list alone.",
  },
  {
    target: ["#batchKind", "#vtRun"], app: "video", vmode: "tag",
    title: "Tag scenes",
    text: "Once the scenes look right, tag them here. It works like the Batch tab, with the same settings: " +
      "a model or workflow describes a frame from each scene. The JSON is saved with the scene on the server, shows on the " +
      "timeline and in the scene list, and goes into <strong>Export scenes</strong>. Click a scene to see its result or re-run it.",
  },
  {
    target: "#tourBtn",
    title: "That's it",
    text: "Press <strong>?</strong> any time to see this tour again.",
  },
];

const tour = { index: -1, root: null, dim: null, rings: null, card: null, inerted: [], start: null };

function tourSeen() {
  try {
    return localStorage.getItem(TOUR_KEY) === TOUR_VERSION;
  } catch {
    return false;
  }
}

function startTour() {
  if (tour.root || abortController) return;
  tour.start = {
    mode: document.body.dataset.mode, backend: el.backend.value, app: document.body.dataset.app, vmode: document.body.dataset.vmode,
  };
  tour.dim = h("div", { class: "tour-dim" });
  tour.rings = h("div");
  tour.card = h("div", { class: "tour-card", role: "dialog", "aria-modal": "true", "aria-labelledby": "tourTitle" });
  tour.root = h("div", { class: "tour" }, tour.dim, tour.rings, tour.card);
  // The page can't be clicked or tabbed into while the tour is open
  tour.inerted = [...document.body.children].filter((x) => !x.inert && x.tagName !== "SCRIPT");
  tour.inerted.forEach((x) => (x.inert = true));
  document.body.append(tour.root);
  window.addEventListener("keydown", tourKeys, true);
  window.addEventListener("resize", placeTour);
  window.addEventListener("scroll", placeTour, true);
  showTourStep(0);
}

function endTour() {
  if (!tour.root) return;
  try {
    localStorage.setItem(TOUR_KEY, TOUR_VERSION);
  } catch { /* storage unavailable: the tour shows again next time */ }
  window.removeEventListener("keydown", tourKeys, true);
  window.removeEventListener("resize", placeTour);
  window.removeEventListener("scroll", placeTour, true);
  tour.inerted.forEach((x) => (x.inert = false));
  tour.root.remove();
  tour.root = null;
  tour.index = -1;
  if (el.backend.value !== tour.start.backend) setBackend(tour.start.backend);
  if (document.body.dataset.mode !== tour.start.mode) setMode(tour.start.mode);
  if (document.body.dataset.vmode !== tour.start.vmode) setVideoMode(tour.start.vmode); // video-tag.js
  if (document.body.dataset.app !== tour.start.app) setApp(tour.start.app); // video.js
  $("tourBtn").focus();
}

function showTourStep(i) {
  tour.index = i;
  const step = TOUR_STEPS[i];
  const app = step.app || (step.mode ? "image" : null);
  if (app && document.body.dataset.app !== app) setApp(app);
  if (step.vmode && document.body.dataset.vmode !== step.vmode) setVideoMode(step.vmode);
  if (step.mode && document.body.dataset.mode !== step.mode) setMode(step.mode);
  if (step.backend && el.backend.value !== step.backend) setBackend(step.backend);

  const last = i === TOUR_STEPS.length - 1;
  const body = h("p", { class: "tour-text" });
  body.innerHTML = step.text;
  const back = h("button", { type: "button", disabled: i === 0 }, "Back");
  const next = h("button", { type: "button", class: "primary" }, last ? "Done" : "Next");
  const skip = h("button", { type: "button", class: "link" }, "Skip tour");
  back.addEventListener("click", () => showTourStep(i - 1));
  next.addEventListener("click", () => (last ? endTour() : showTourStep(i + 1)));
  skip.addEventListener("click", endTour);
  tour.card.replaceChildren(
    h("div", { class: "tour-head" },
      h("h2", { id: "tourTitle" }, step.title),
      h("span", { class: "meta" }, `${i + 1} of ${TOUR_STEPS.length}`)),
    body,
    h("div", { class: "tour-actions" }, last ? null : skip, h("span", { class: "spacer" }), back, next),
  );
  tourTargets(step)[0]?.scrollIntoView({ block: "nearest" });
  placeTour();
  next.focus();
}

// The visible elements a step highlights, each grown to its <label>
function tourTargets(step) {
  return [step.target || []].flat()
    .map((sel) => document.querySelector(sel))
    .map((x) => x && (x.closest(".settings label") || x))
    .filter((x) => x && x.getClientRects().length);
}

// Highlighted areas: each target padded and kept on screen, with neighbours merged into one.
// Targets that wrapped onto different rows stay separate instead of one box spanning the header.
function tourRects(step) {
  const pad = 6;
  const near = 8; // areas closer than this merge, e.g. side-by-side settings
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  let rects = tourTargets(step).map((x) => {
    const r = x.getBoundingClientRect();
    return {
      left: Math.max(2, r.left - pad), top: Math.max(2, r.top - pad),
      right: Math.min(vw - 2, r.right + pad), bottom: Math.min(vh - 2, r.bottom + pad),
    };
  });
  for (let merged = true; merged;) {
    merged = false;
    outer: for (let i = 0; i < rects.length; i++) {
      for (let j = i + 1; j < rects.length; j++) {
        const [a, b] = [rects[i], rects[j]];
        if (a.left <= b.right + near && b.left <= a.right + near && a.top <= b.bottom + near && b.top <= a.bottom + near) {
          rects[i] = {
            left: Math.min(a.left, b.left), top: Math.min(a.top, b.top),
            right: Math.max(a.right, b.right), bottom: Math.max(a.bottom, b.bottom),
          };
          rects.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return rects.filter((r) => r.right > r.left && r.bottom > r.top);
}

// A rounded rectangle as an SVG path, for the holes cut into the dimmed layer
function roundRectPath({ left: l, top: t, right: r, bottom: b }, rad = 8) {
  rad = Math.min(rad, (r - l) / 2, (b - t) / 2);
  return `M${l + rad} ${t}H${r - rad}A${rad} ${rad} 0 0 1 ${r} ${t + rad}V${b - rad}` +
    `A${rad} ${rad} 0 0 1 ${r - rad} ${b}H${l + rad}A${rad} ${rad} 0 0 1 ${l} ${b - rad}V${t + rad}` +
    `A${rad} ${rad} 0 0 1 ${l + rad} ${t}Z`;
}

function placeTour() {
  if (!tour.root) return;
  const gap = 10;
  const margin = 16;
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const rects = tourRects(TOUR_STEPS[tour.index]);
  tour.dim.style.clipPath = `path(evenodd, "M0 0H${vw}V${vh}H0Z${rects.map((r) => roundRectPath(r)).join("")}")`;
  tour.rings.replaceChildren(...rects.map((r) => h("div", {
    class: "tour-ring",
    style: `left:${r.left}px;top:${r.top}px;width:${r.right - r.left}px;height:${r.bottom - r.top}px`,
  })));
  const cw = tour.card.offsetWidth;
  const ch = tour.card.offsetHeight;
  let left;
  let top;
  if (!rects.length) {
    // Nothing to point at: the whole page stays dimmed and the card sits in the middle
    left = (vw - cw) / 2;
    top = Math.max(margin, (vh - ch) / 2);
  } else {
    const r = {
      left: Math.min(...rects.map((x) => x.left)),
      top: Math.min(...rects.map((x) => x.top)),
      bottom: Math.max(...rects.map((x) => x.bottom)),
    };
    top = r.bottom + gap;
    if (top + ch > vh - margin) {
      const above = r.top - gap - ch;
      top = above >= margin ? above : Math.max(margin, vh - margin - ch);
    }
    left = r.left;
  }
  left = Math.min(Math.max(margin, left), Math.max(margin, vw - margin - cw));
  Object.assign(tour.card.style, { left: `${left}px`, top: `${top}px` });
}

function tourKeys(e) {
  const keys = { Escape: endTour, ArrowLeft: () => tour.index > 0 && showTourStep(tour.index - 1) };
  keys.ArrowRight = () => (tour.index < TOUR_STEPS.length - 1 ? showTourStep(tour.index + 1) : endTour());
  // Keep the page's own shortcuts (Cmd/Ctrl+Enter, batch arrows and Esc) from firing behind the tour
  if (keys[e.key] || (e.key === "Enter" && (e.metaKey || e.ctrlKey))) {
    e.preventDefault();
    e.stopImmediatePropagation();
    keys[e.key]?.();
  }
}

$("tourBtn").addEventListener("click", startTour);
// Wait for app.js to apply the saved backend, so the tour puts back the right one
settingsReady.then(() => {
  if (!tourSeen()) startTour();
});

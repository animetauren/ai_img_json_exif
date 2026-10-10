"use strict";
// Resizable layout: drag the handles between panels to resize them. Loads after the other scripts
// and works on markup alone:
//   .split-h  a row handle between two panes ([data-pane]) stacked in a flex column. Dragging
//             sets every pane in that column to a flex-grow equal to its height in pixels, so the
//             split keeps its proportions when the window resizes.
//   .split-v  the column handle of a <main data-layout>. Dragging sets --cols, the main's
//             grid-template-columns, as a ratio of the two columns for the same reason.
// Sizes are kept in localStorage; double-click a handle to go back to the stylesheet's default.
// Arrow keys move a focused handle.

const LAYOUT_KEY = "llm-classifier-layout";
const COL_MIN = 280; // left column; the right column's minimum is the main's data-right-min
const KEY_STEP = 20;

const layout = (() => {
  try {
    const saved = JSON.parse(localStorage.getItem(LAYOUT_KEY)) || {};
    return { panes: saved.panes || {}, cols: saved.cols || {} };
  } catch {
    return { panes: {}, cols: {} };
  }
})();

function saveLayout() {
  try {
    localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
  } catch { /* storage unavailable: sizes last until reload */ }
}

const isShown = (el) => !!el && el.getClientRects().length > 0;

// ---------- rows ----------
function panesOf(handle) {
  return [...handle.parentElement.children].filter((c) => c.matches("[data-pane]"));
}

// A pane's padding and border: flex-basis 0 still keeps them, and flex-grow shares out the rest
function chrome(p) {
  const cs = getComputedStyle(p);
  return ["paddingTop", "paddingBottom", "borderTopWidth", "borderBottomWidth"].reduce((n, k) => n + (parseFloat(cs[k]) || 0), 0);
}

// Freeze every pane in the column at its current size, as flex-grow, so dragging one handle
// moves only its two neighbours. Grow is the height inside padding and border, so the shares
// reproduce the heights exactly.
function freezePanes(panes) {
  // Measure all before writing any: each write reflows the column
  const sizes = panes.map((p) => (isShown(p) ? p.getBoundingClientRect().height - chrome(p) : null));
  panes.forEach((p, i) => {
    if (sizes[i] != null) p.style.flex = `${Math.max(sizes[i], 1)} 1 0px`;
  });
}

function neighbours(handle) {
  let a = handle.previousElementSibling;
  let b = handle.nextElementSibling;
  while (a && !(a.matches("[data-pane]") && isShown(a))) a = a.previousElementSibling;
  while (b && !(b.matches("[data-pane]") && isShown(b))) b = b.nextElementSibling;
  return a && b ? [a, b] : null;
}

// In the same units as grow: the height inside padding and border
const minHeight = (el) => Math.max(0, (parseFloat(getComputedStyle(el).minHeight) || 0) - chrome(el));

// Set a's share to ha + dy, taking it from b, within both panes' minimums. ha and hb are their
// sizes when the drag started, so the boundary follows the pointer even if moves are dropped.
function moveRow(a, b, ha, hb, dy) {
  const total = ha + hb;
  const na = Math.min(Math.max(ha + dy, minHeight(a)), total - minHeight(b));
  a.style.flex = `${na} 1 0px`;
  b.style.flex = `${total - na} 1 0px`;
}

const growOf = (p) => parseFloat(p.style.flexGrow);

function storeRows(handle) {
  for (const p of panesOf(handle)) {
    if (p.style.flexGrow) layout.panes[p.dataset.pane] = parseFloat(p.style.flexGrow);
  }
  saveLayout();
}

function resetRows(handle) {
  for (const p of panesOf(handle)) {
    p.style.flex = "";
    delete layout.panes[p.dataset.pane];
  }
  saveLayout();
}

// ---------- columns ----------
function columnsOf(main) {
  return [...main.children].filter((c) => !c.classList.contains("split-v") && isShown(c));
}

function placeColumnHandle(main) {
  const handle = main.querySelector(":scope > .split-v");
  const cols = columnsOf(main);
  if (!handle || cols.length < 2 || !isShown(main)) {
    if (handle) handle.hidden = true;
    return;
  }
  const m = main.getBoundingClientRect();
  const left = cols[0].getBoundingClientRect();
  const right = cols[1].getBoundingClientRect();
  if (Math.abs(left.top - right.top) > 4) {
    handle.hidden = true; // stacked (narrow window): no columns to resize
    return;
  }
  handle.hidden = false;
  const mid = (left.right + right.left) / 2;
  Object.assign(handle.style, {
    left: `${mid - m.left - 6}px`,
    top: `${left.top - m.top}px`,
    height: `${left.height}px`,
  });
}

function setColumns(main, ratio) {
  const rightMin = Number(main.dataset.rightMin) || 320;
  main.style.setProperty("--cols", `minmax(0, ${ratio}fr) minmax(${rightMin}px, 1fr)`);
}

// Put the boundary at clientX, keeping both columns above their minimums
function moveColumns(main, clientX) {
  const cs = getComputedStyle(main);
  const m = main.getBoundingClientRect();
  const gap = parseFloat(cs.columnGap) || 0;
  const inner = m.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight) - gap;
  const rightMin = Number(main.dataset.rightMin) || 320;
  let left = clientX - m.left - parseFloat(cs.paddingLeft) - gap / 2;
  left = Math.min(Math.max(left, COL_MIN), inner - rightMin);
  const ratio = Math.round((left / (inner - left)) * 1000) / 1000;
  if (!(ratio > 0)) return;
  layout.cols[main.dataset.layout] = ratio;
  setColumns(main, ratio);
  placeColumnHandle(main);
}

function resetColumns(main) {
  main.style.removeProperty("--cols");
  delete layout.cols[main.dataset.layout];
  saveLayout();
  placeColumnHandle(main);
}

const placeAll = () => document.querySelectorAll("main[data-layout]").forEach(placeColumnHandle);

// ---------- dragging ----------
// A second press within this time resets the handle instead of dragging. Detected here rather
// than with dblclick, which Chrome doesn't fire once pointerdown's default is prevented.
const DOUBLE_PRESS_MS = 400;

function isDoublePress(handle, e) {
  const double = e.timeStamp - (handle.lastPress || -Infinity) < DOUBLE_PRESS_MS;
  handle.lastPress = double ? null : e.timeStamp;
  return double;
}

function startDrag(handle, e, onMove, onEnd) {
  e.preventDefault(); // no text selection while dragging
  handle.setPointerCapture(e.pointerId);
  handle.classList.add("dragging");
  document.body.classList.add("resizing");
  const move = (ev) => onMove(ev);
  const up = () => {
    handle.removeEventListener("pointermove", move);
    handle.removeEventListener("pointerup", up);
    handle.removeEventListener("pointercancel", up);
    handle.classList.remove("dragging");
    document.body.classList.remove("resizing");
    onEnd();
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", up);
  handle.addEventListener("pointercancel", up);
}

document.querySelectorAll(".split-h").forEach((handle) => {
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (isDoublePress(handle, e)) {
      e.preventDefault();
      resetRows(handle);
      return;
    }
    const pair = neighbours(handle);
    if (!pair) return;
    freezePanes(panesOf(handle));
    const [ha, hb] = pair.map(growOf);
    startDrag(handle, e, (ev) => moveRow(pair[0], pair[1], ha, hb, ev.clientY - e.clientY), () => storeRows(handle));
  });
  handle.addEventListener("keydown", (e) => {
    const step = { ArrowUp: -KEY_STEP, ArrowDown: KEY_STEP }[e.key];
    const pair = step && neighbours(handle);
    if (!pair) return;
    e.preventDefault();
    freezePanes(panesOf(handle));
    moveRow(pair[0], pair[1], growOf(pair[0]), growOf(pair[1]), step);
    storeRows(handle);
  });
});

document.querySelectorAll("main[data-layout]").forEach((main) => {
  const handle = main.querySelector(":scope > .split-v");
  if (!handle) return;
  handle.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    if (isDoublePress(handle, e)) {
      e.preventDefault();
      resetColumns(main);
      return;
    }
    startDrag(handle, e, (ev) => moveColumns(main, ev.clientX), saveLayout);
  });
  handle.addEventListener("keydown", (e) => {
    const step = { ArrowLeft: -KEY_STEP, ArrowRight: KEY_STEP }[e.key];
    if (!step) return;
    e.preventDefault();
    const x = handle.getBoundingClientRect().left + handle.offsetWidth / 2;
    moveColumns(main, x + step);
    saveLayout();
  });
  // Keep the handle over the gap as columns change size or the tab or side changes
  const ro = new ResizeObserver(() => placeColumnHandle(main));
  ro.observe(main);
  [...main.children].forEach((c) => c !== handle && ro.observe(c));
});

// Tabs, batch focus and the image/video switch change which columns show
new MutationObserver(() => requestAnimationFrame(placeAll))
  .observe(document.body, { attributes: true, attributeFilter: ["data-mode", "data-kind", "data-focus", "data-app", "data-vmode"] });
window.addEventListener("resize", placeAll);

// ---------- restore ----------
// A column with a pane that has no saved size (one added since) starts again from the
// stylesheet: its default share would be lost among the others' sizes in pixels
document.querySelectorAll(".split-h").forEach((handle) => {
  const panes = panesOf(handle);
  if (panes.some((p) => layout.panes[p.dataset.pane]) && panes.some((p) => !layout.panes[p.dataset.pane])) {
    panes.forEach((p) => delete layout.panes[p.dataset.pane]);
  }
});
for (const [name, grow] of Object.entries(layout.panes)) {
  const pane = document.querySelector(`[data-pane="${name}"]`);
  if (pane && grow > 0) pane.style.flex = `${grow} 1 0px`;
}
document.querySelectorAll("main[data-layout]").forEach((main) => {
  const ratio = layout.cols[main.dataset.layout];
  if (ratio > 0) setColumns(main, ratio);
});
requestAnimationFrame(placeAll);

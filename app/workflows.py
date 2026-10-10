"""Workflows: ordered chains of vision-model and LLM steps.

Each step's output is stored under its id, and later LLM prompts can pull values
from earlier steps with {{step_id.field}} placeholders, e.g. {{objects.counts.person}}.
"""

import copy
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

from .llm import chat_payload, extract_json, stream_chat

STEP_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TEMPLATE_RE = re.compile(r"\{\{\s*([A-Za-z0-9_.\-]+)\s*\}\}")
STEP_TYPES = ("vision", "llm")

log = logging.getLogger("uvicorn.error")

# ---------- starter workflows ----------

_CATALOG_SYSTEM = """You are a photo cataloguing assistant. You receive a photo plus the outputs of specialised vision models.
Trust the object detector for counts of people and objects (it is more reliable at counting than you are).
Use your own judgement for the scene, activity, setting and mood.
Respond ONLY with a JSON object, no other text."""

_CATALOG_USER = """Object detector counts: {{objects.counts}}
Scene classifier (SigLIP) top labels: {{scene.summary}}
Date taken: {{exif.DateTimeOriginal}}
Camera: {{exif.Make}} {{exif.Model}}

Return JSON in this shape:
{
  "category": "<single best category>",
  "people_count": <integer; use the detector's person count unless it is clearly wrong>,
  "objects": ["<notable objects>"],
  "activity": "<what is happening>",
  "setting": "indoor | outdoor | unknown",
  "tags": ["<short tags>"],
  "description": "<one sentence>"
}"""

_PEOPLE_SYSTEM = """You describe the people in photos for a family photo library.
A person detector has already counted the people; treat its count as ground truth unless the image clearly contradicts it.
Respond ONLY with a JSON object, no other text."""

_PEOPLE_USER = """Person detector found {{people.counts.person}} people (confidences: {{people.summary}}).

Return JSON:
{
  "people_count": <integer>,
  "shot_type": "solo portrait | couple | small group | large group | crowd | no people",
  "ages": ["<rough age groups present, e.g. child, adult, senior>"],
  "activity": "<what the people are doing>",
  "posed": <true if people are posing for the camera>,
  "description": "<one sentence>"
}"""

DEFAULT_WORKFLOWS = [
    {
        "id": "catalog",
        "name": "Catalog: object counts + SigLIP scene → LLM",
        "steps": [
            {"id": "objects", "type": "vision", "model": "rtdetr-v2-s", "enabled": True,
             "params": {"threshold": 0.5, "top_k": 20}},
            {"id": "scene", "type": "vision", "model": "siglip2-base", "enabled": True,
             "params": {"top_k": 5}},
            {"id": "interpret", "type": "llm", "model": "", "enabled": True,
             "params": {"system_prompt": _CATALOG_SYSTEM, "user_prompt": _CATALOG_USER,
                        "include_image": True, "json_mode": True, "temperature": 0.2, "max_tokens": 1024}},
        ],
    },
    {
        "id": "people",
        "name": "People: person-only detector → LLM",
        "steps": [
            {"id": "people", "type": "vision", "model": "rtdetr-v2-s", "enabled": True,
             "params": {"threshold": 0.5, "top_k": 20, "classes": ["person"]}},
            {"id": "describe", "type": "llm", "model": "", "enabled": True,
             "params": {"system_prompt": _PEOPLE_SYSTEM, "user_prompt": _PEOPLE_USER,
                        "include_image": True, "json_mode": True, "temperature": 0.2, "max_tokens": 768}},
        ],
    },
    {
        "id": "quick-tags",
        "name": "Quick tags (no LLM): RT-DETR + EfficientNet + SigLIP",
        "steps": [
            {"id": "objects", "type": "vision", "model": "rtdetr-v2-s", "enabled": True,
             "params": {"threshold": 0.5, "top_k": 10}},
            {"id": "imagenet", "type": "vision", "model": "efficientnetv2-s", "enabled": True,
             "params": {"top_k": 5}},
            {"id": "scene", "type": "vision", "model": "siglip2-base", "enabled": True,
             "params": {"top_k": 5}},
        ],
    },
]


# ---------- storage ----------

def validate(wf: dict) -> dict:
    if not isinstance(wf, dict):
        raise ValueError("Workflow must be an object.")
    if not str(wf.get("id") or "").strip():
        raise ValueError("Workflow needs an id.")
    if not str(wf.get("name") or "").strip():
        raise ValueError("Workflow needs a name.")
    steps = wf.get("steps")
    if not isinstance(steps, list):
        raise ValueError("'steps' must be a list.")
    seen = set()
    for i, step in enumerate(steps, 1):
        sid = step.get("id", "")
        if not STEP_ID_RE.match(sid):
            raise ValueError(f"Step {i}: id '{sid}' must be letters, digits or _ and not start with a digit.")
        if sid in seen or sid == "exif":
            raise ValueError(f"Step {i}: id '{sid}' is already used (or reserved).")
        seen.add(sid)
        if step.get("type") not in STEP_TYPES:
            raise ValueError(f"Step {i} ({sid}): type must be one of {', '.join(STEP_TYPES)}.")
        if step["type"] == "vision" and not step.get("model"):
            raise ValueError(f"Step {i} ({sid}): vision steps need a model.")
        step.setdefault("enabled", True)
        step.setdefault("params", {})
    return wf


class WorkflowStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> list:
        if not self.path.exists():
            return copy.deepcopy(DEFAULT_WORKFLOWS)
        return json.loads(self.path.read_text())

    def _write(self, items: list) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=2))
        os.replace(tmp, self.path)

    def list(self) -> list:
        with self._lock:
            return self._read()

    def put(self, wf: dict) -> dict:
        wf = validate(wf)
        with self._lock:
            items = self._read()
            ids = [w.get("id") for w in items]
            # Keep the workflow's position when updating; append when new
            if wf["id"] in ids:
                items[ids.index(wf["id"])] = wf
            else:
                items.append(wf)
            self._write(items)
        return wf

    def delete(self, wf_id: str) -> bool:
        with self._lock:
            items = self._read()
            kept = [w for w in items if w.get("id") != wf_id]
            self._write(kept)
            return len(kept) != len(items)


# ---------- templates ----------

def _resolve(path: str, ctx: dict):
    cur = ctx
    parts = path.split(".")
    for i, part in enumerate(parts):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif i > 0 and parts[i - 1] == "counts" and isinstance(cur, dict):
            return 0  # a label the detector didn't find is a count of zero, not unknown
        else:
            return None
    return cur


def _why_missing(path: str, ctx: dict) -> str:
    """Explain why a placeholder didn't resolve, so the user can fix the template or the step."""
    root, _, rest = path.partition(".")
    if root == "exif":
        exif = ctx.get("exif") or {}
        if not exif:
            return "no EXIF metadata was available for this image"
        return f"this image has no '{rest}' tag ({len(exif)} tags available; see the EXIF panel)"
    if root not in ctx:
        return f"no step '{root}' has run before this step (check the step id and order)"
    out = ctx[root]
    if isinstance(out, dict) and "error" in out:
        return f"step '{root}' failed: {out['error']}"
    first = rest.split(".")[0]
    if isinstance(out, dict) and first not in out:
        return f"step '{root}' has no field '{first}' (available: {', '.join(sorted(out))})"
    return "value not found"


def render(template: str, ctx: dict) -> tuple:
    """Substitute {{path}} placeholders. Returns (text, [(path, reason)] for unresolved ones)."""
    missing = []

    def sub(m):
        value = _resolve(m.group(1), ctx)
        if value is None:
            missing.append((m.group(1), _why_missing(m.group(1), ctx)))
            return "unknown"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)

    return TEMPLATE_RE.sub(sub, template or ""), missing


# ---------- runner ----------

def _vision_summary(out: dict) -> str:
    preds = out.get("predictions", [])
    if out.get("kind") == "detect":
        return ", ".join(f"{p['label']} ×{p.get('count', 1)} ({p['score']:.2f})" for p in preds) or "nothing detected"
    return ", ".join(f"{p['label'].split(',')[0]} ({p['score']:.3f})" for p in preds)


def _compact(step: dict, out: dict):
    """Short form of a step's output for the combined result."""
    if "error" in out:
        return {"error": out["error"]}
    if step["type"] == "llm":
        return out["json"] if out.get("json") is not None else out.get("text")
    compact = {k: out.get(k) for k in ("category", "tags", "confidence")}
    if out.get("counts") is not None:
        compact["counts"] = out["counts"]
    return compact


async def classify_vision(client: httpx.AsyncClient, vision_url: str, model: str, params: dict, image_data_url: str) -> dict:
    """Run one vision model on an image; used by workflow steps and the CLI's single-model runs."""
    body = {"model": model, "image_data_url": image_data_url}
    for key in ("labels", "label_template", "top_k", "threshold", "classes"):
        if params.get(key) not in (None, "", []):
            body[key] = params[key]
    r = await client.post(f"{vision_url}/classify", json=body)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"detail": r.text}
    if r.status_code >= 400:
        detail = data.get("detail") if isinstance(data.get("detail"), str) else ""
        # Keep the status on server errors so callers can tell an outage from a bad request
        raise RuntimeError(f"HTTP {r.status_code}: {detail}" if r.status_code >= 500 or not detail else detail)
    data["summary"] = _vision_summary(data)
    return data


async def _run_vision(client: httpx.AsyncClient, vision_url: str, step: dict, image_data_url: str) -> dict:
    return await classify_vision(client, vision_url, step["model"], step.get("params", {}), image_data_url)


async def run_workflow(
    wf: dict,
    *,
    image_data_url: str,
    exif: dict,
    llm: dict,
    vision_url: str,
    timeout: float,
) -> AsyncIterator[dict]:
    """Run steps in order, yielding progress events. A failed step is recorded and the run continues."""
    validate(wf)
    log.info("workflow run '%s': %s", wf.get("name"), " → ".join(
        f"{s['id']}({s.get('model') or 'default'}{'' if s.get('enabled', True) else ', off'})" for s in wf["steps"]))
    ctx: dict = {"exif": exif or {}}
    outputs: dict = {}
    started = time.perf_counter()

    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, step in enumerate(wf["steps"]):
            sid = step["id"]
            if not step.get("enabled", True):
                yield {"type": "step_skipped", "id": sid, "index": index}
                continue
            yield {"type": "step_start", "id": sid, "index": index}
            t = time.perf_counter()
            try:
                if step["type"] == "vision":
                    out = await _run_vision(client, vision_url, step, image_data_url)
                else:
                    out = None
                    async for ev in _run_llm(step, ctx, image_data_url, llm, timeout):
                        if ev["type"] == "output":
                            out = ev["output"]
                        else:
                            yield {**ev, "id": sid}
            except Exception as e:  # noqa: BLE001
                out = {"error": str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"}
                log.warning("workflow step %s failed: %s", sid, out["error"])
                yield {"type": "step_error", "id": sid, "message": out["error"]}
            else:
                yield {"type": "step_done", "id": sid, "output": out,
                       "ms": round((time.perf_counter() - t) * 1000, 1)}
            outputs[sid] = out
            ctx[sid] = out

    result = {
        "workflow": wf.get("name"),
        **{s["id"]: _compact(s, outputs[s["id"]]) for s in wf["steps"] if s["id"] in outputs},
    }
    yield {"type": "done", "result": result, "ms": round((time.perf_counter() - started) * 1000, 1)}


async def _run_llm(step: dict, ctx: dict, image_data_url: str, llm: dict, timeout: float) -> AsyncIterator[dict]:
    p = step.get("params", {})
    system_prompt, missing_sys = render(p.get("system_prompt", ""), ctx)
    user_prompt, missing_user = render(p.get("user_prompt", ""), ctx)
    missing = dict(missing_sys + missing_user)
    if missing:
        message = "Placeholders sent as 'unknown':\n" + "\n".join(
            f"• {{{{{path}}}}}: {why}" for path, why in sorted(missing.items())
        )
        log.info("workflow step %s: %s", step["id"], message.replace("\n", " "))
        yield {"type": "warning", "message": message}
    yield {"type": "prompt", "system": system_prompt, "user": user_prompt}

    payload = chat_payload(
        model=step.get("model") or llm.get("model", ""),
        system_prompt=system_prompt,
        user_prompt=user_prompt or "Describe this image.",
        image_data_url=image_data_url if p.get("include_image", True) else None,
        temperature=float(p.get("temperature", 0.2)),
        max_tokens=int(p.get("max_tokens", 1024)),
        json_mode=bool(p.get("json_mode", False)),
    )
    text, usage = "", None
    async for ev in stream_chat(llm.get("base_url", ""), llm.get("api_key", ""), payload, timeout):
        if ev["type"] == "error":
            raise RuntimeError(ev["message"])
        if ev["type"] == "token":
            text += ev["text"]
        if ev["type"] == "usage":
            usage = ev["usage"]
        yield ev

    out: dict = {"text": text, "json": None, "usage": usage, "prompt": user_prompt}
    try:
        out["json"] = extract_json(text)
    except ValueError as e:
        out["json_error"] = str(e)
    yield {"type": "output", "output": out}

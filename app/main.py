"""Small proxy + static server for trialling local vision LLMs on image classification."""

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .llm import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, chat_payload, headers as _headers, stream_chat
from . import runs
from .video import router as video_router
from .metadata import EXIFTOOL, export as export_metadata, run_exiftool
from .workflows import WorkflowStore, run_workflow, validate

STATIC_DIR = Path(__file__).parent / "static"

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://host.containers.internal:8080/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")
REQUEST_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://vision:8000").rstrip("/")
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent.parent / "data"))


app = FastAPI(title="LLM Image Classifier")
app.include_router(video_router)
store = WorkflowStore(DATA_DIR / "workflows.json")


@app.get("/api/config")
def config():
    return {
        "base_url": DEFAULT_BASE_URL,
        "has_server_api_key": bool(DEFAULT_API_KEY),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
    }


@app.get("/api/models")
async def models(base_url: str = Query(DEFAULT_BASE_URL), api_key: str = Query("")):
    url = base_url.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url, headers=_headers(api_key or DEFAULT_API_KEY))
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001 - surface any failure to the UI
        return JSONResponse({"models": [], "error": f"{type(e).__name__}: {e}"})

    ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
    # Older llama.cpp builds also expose a top-level "models" list
    if not ids:
        ids = [m.get("model") or m.get("name") for m in data.get("models", []) if m.get("model") or m.get("name")]
    return {"models": ids, "error": None}


class ClassifyRequest(BaseModel):
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = ""
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
    image_data_url: str
    temperature: float = 0.2
    max_tokens: int = 1024
    json_mode: bool = False


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


@app.post("/api/classify")
async def classify(req: ClassifyRequest):
    payload = chat_payload(
        model=req.model,
        system_prompt=req.system_prompt,
        user_prompt=req.user_prompt or DEFAULT_USER_PROMPT,
        image_data_url=req.image_data_url,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        json_mode=req.json_mode,
    )

    async def stream():
        async for event in stream_chat(req.base_url, req.api_key or DEFAULT_API_KEY, payload, REQUEST_TIMEOUT):
            yield _sse(event)
        yield _sse({"type": "done"})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/vision/models")
async def vision_models():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{VISION_BASE_URL}/models")
            r.raise_for_status()
            return {**r.json(), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"models": [], "error": f"Vision service unavailable at {VISION_BASE_URL} ({type(e).__name__}: {e})"}


@app.post("/api/vision/classify")
async def vision_classify(body: dict):
    # First use of a model downloads its weights, so allow the same generous timeout
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(f"{VISION_BASE_URL}/classify", json=body)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"detail": f"{type(e).__name__}: {e}"}, status_code=502)
    try:
        return JSONResponse(r.json(), status_code=r.status_code)
    except ValueError:
        return JSONResponse({"detail": r.text}, status_code=r.status_code)


@app.post("/api/exif")
async def exif(file: UploadFile = File(...)):
    """Return every metadata tag exiftool can find in the original (un-resized) file."""
    if not EXIFTOOL:
        return JSONResponse({"detail": "exiftool is not installed in this container."}, status_code=500)

    suffix = Path(file.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        try:
            # -G1: specific group names (IFD0, ExifIFD, GPS, Canon, XMP-dc, ...)
            # -a: keep duplicate tags, -u: include unknown tags
            full = (await run_exiftool(tmp.name, "-G1", "-a", "-u"))[0]
            gps = (await run_exiftool(tmp.name, "-n", "-GPSLatitude", "-GPSLongitude"))[0]
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"detail": f"exiftool failed: {e}"}, status_code=500)

    groups: dict = {}
    for key, value in full.items():
        if key == "SourceFile":
            continue
        group, _, tag = key.rpartition(":")
        # The System group describes our temp file (path, permissions, mtime), not the user's
        if group == "System" and tag != "FileSize":
            continue
        groups.setdefault(group or "Other", {})[tag] = value
    groups.setdefault("System", {})["FileName"] = file.filename

    location = None
    if isinstance(gps.get("GPSLatitude"), (int, float)) and isinstance(gps.get("GPSLongitude"), (int, float)):
        location = {"lat": gps["GPSLatitude"], "lon": gps["GPSLongitude"]}

    return {
        "groups": groups,
        "count": sum(len(tags) for tags in groups.values()),
        "location": location,
    }


@app.post("/api/export")
async def export_image(
    file: UploadFile = File(...),
    result: str = Form(...),
    overwrite: bool = Form(False),
    source: str = Form(""),
    save_to: str = Form(""),
    rel_path: str = Form(""),
):
    """Write the result into a copy of the original file's metadata.

    Returns the copy as a download, or with save_to (a batch run id) adds it to that run
    as <rel_path>, under its original name, and returns the report.
    """
    target = runs.target(save_to, rel_path or file.filename or "image") if save_to else None
    if not EXIFTOOL:
        return JSONResponse({"detail": "exiftool is not installed in this container."}, status_code=500)
    try:
        data = json.loads(result)
    except ValueError as e:
        raise HTTPException(400, f"Result is not valid JSON: {e}")

    name = Path(file.filename or "image").name
    suffix = Path(name).suffix
    with tempfile.TemporaryDirectory() as tmpdir:
        # exiftool picks the format from the content, but keep the suffix for its messages
        path = Path(tmpdir) / f"export{suffix}"
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            report = await export_metadata(str(path), data, source=source, overwrite=overwrite)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"detail": f"exiftool failed: {str(e).replace(str(path), name)}"}, status_code=422)
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                # "x": never replace a photo an earlier export wrote
                with path.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            except FileExistsError:
                raise HTTPException(409, f"{rel_path} is already in this run.")
            return report
        body = path.read_bytes()

    report["filename"] = f"{Path(name).stem}.tagged{suffix}"
    return Response(body, media_type=file.content_type or "application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(report['filename'])}",
        "X-Export-Report": quote(json.dumps(report, separators=(",", ":"))),
    })


class RunCreate(BaseModel):
    name: str


@app.post("/api/exports")
def create_export_run(req: RunCreate):
    """Start a batch export; photos are added with /api/export save_to=<id>."""
    return {"id": runs.create(req.name.strip())}


@app.put("/api/exports/{run_id}/manifest")
def export_manifest(run_id: str, manifest: dict):
    """Save a summary of the run; it goes into the zip as _run.json."""
    (runs.folder(run_id) / runs.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return {"saved": runs.MANIFEST_FILE}


@app.get("/api/exports/{run_id}")
def export_info(run_id: str):
    """Whether a run is still waiting to be downloaded; 404 once its zip was sent or it expired."""
    return runs.info(run_id)


@app.delete("/api/exports/{run_id}")
def discard_export(run_id: str):
    runs.discard(run_id)
    return {"discarded": run_id}


@app.get("/api/exports/{run_id}/zip")
def export_zip(run_id: str):
    filename = f"{runs.name(run_id)}.zip"
    return StreamingResponse(runs.zip_stream(run_id), media_type="application/zip", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
    })


# ---------- workflows ----------

@app.get("/api/workflows")
def list_workflows():
    return {"workflows": store.list()}


@app.put("/api/workflows/{wf_id}")
def save_workflow(wf_id: str, wf: dict):
    wf["id"] = wf_id
    try:
        return store.put(wf)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/workflows/{wf_id}")
def delete_workflow(wf_id: str):
    if not store.delete(wf_id):
        raise HTTPException(404, f"No workflow '{wf_id}'.")
    return {"deleted": wf_id}


class WorkflowRunRequest(BaseModel):
    workflow: dict  # the full definition, so unsaved edits can be run
    image_data_url: str
    exif: dict = {}
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = ""  # default for LLM steps that don't name a model


@app.post("/api/workflows/run")
async def run_workflow_endpoint(req: WorkflowRunRequest):
    try:
        validate(req.workflow)
    except ValueError as e:
        raise HTTPException(400, str(e))

    async def stream():
        try:
            async for event in run_workflow(
                req.workflow,
                image_data_url=req.image_data_url,
                exif=req.exif,
                llm={"base_url": req.base_url, "api_key": req.api_key or DEFAULT_API_KEY, "model": req.model},
                vision_url=VISION_BASE_URL,
                timeout=REQUEST_TIMEOUT,
            ):
                yield _sse(event)
        except Exception as e:  # noqa: BLE001
            yield _sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.middleware("http")
async def revalidate_assets(request, call_next):
    # Without this, browsers may reuse a cached app.js after a rebuild while loading the new
    # index.html, leaving new buttons with no handlers. no-cache still allows 304s via ETag.
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


ASSET_RE = re.compile(r'"/static/([\w.-]+\.(?:js|css))"')


@app.get("/")
def index():
    # Version each script/stylesheet URL by its mtime, so a rebuilt app is never paired
    # with a stale cached app.js (new buttons with no handlers)
    html = (STATIC_DIR / "index.html").read_text()
    html = ASSET_RE.sub(lambda m: f'"/static/{m.group(1)}?v={int((STATIC_DIR / m.group(1)).stat().st_mtime)}"', html)
    return HTMLResponse(html)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

"""Video uploads and the pass-through to the video service.

Uploads arrive in chunks and are appended straight to disk on the volume shared with the video
service, so neither the browser nor the server ever holds more than one chunk. A file of one
chunk or less goes up in a single request; larger ones resume from the server's byte count
after an interruption. Everything else (status, detection, transcripts, thumbnails, frames, tags) is forwarded to
the video service, streaming both ways, except the proxy video, which is served from disk here
so the browser can seek it with Range requests.
"""

import asyncio
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

VIDEO_DIR = Path(os.environ.get("VIDEO_DIR", "/videos"))
VIDEO_BASE_URL = os.environ.get("VIDEO_BASE_URL", "http://video:8001").rstrip("/")
CHUNK_SIZE = 64 << 20
ID_RE = re.compile(r"^[0-9a-f]{32}$")
EXT_RE = re.compile(r"^\.[a-z0-9]{1,8}$")
WRITE_BUFFER = 4 << 20

router = APIRouter(prefix="/api/video")
_client = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None))
_locks: dict = {}  # upload id -> asyncio.Lock, so two requests can't append at once


def _gb(n: float) -> str:
    return f"{n / 1e9:.1f} GB"


def _folder(video_id: str) -> Path:
    path = VIDEO_DIR / video_id
    if not ID_RE.match(video_id) or not (path / "meta.json").is_file():
        raise HTTPException(404, "Unknown or expired video.")
    return path


def _meta(root: Path) -> dict:
    return json.loads((root / "meta.json").read_text())


def _upload_state(root: Path) -> dict:
    meta = _meta(root)
    part = root / "source.part"
    if part.exists():
        return {"id": root.name, "size": meta["size"], "received": part.stat().st_size, "complete": False,
                "chunk_size": CHUNK_SIZE}
    return {"id": root.name, "size": meta["size"], "received": meta["size"], "complete": True, "chunk_size": CHUNK_SIZE}


class UploadCreate(BaseModel):
    name: str
    size: int
    last_modified: int = 0


@router.post("/uploads")
def create_upload(req: UploadCreate):
    name = Path(req.name.replace("\\", "/")).name.strip()
    if not name or req.size <= 0:
        raise HTTPException(400, "Choose a video file.")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # The same file again (name, size and date match) picks up its earlier upload
    reserved = 0
    for root in VIDEO_DIR.iterdir():
        try:
            meta = _meta(root)
        except (OSError, ValueError):
            continue
        if (meta.get("name"), meta.get("size"), meta.get("last_modified")) == (name, req.size, req.last_modified):
            return _upload_state(root)
        part = root / "source.part"
        if part.exists():
            reserved += max(0, meta.get("size", 0) - part.stat().st_size)

    # Room for the source and its 360p proxy. The proxy is usually 1-3% of the source, but noisy
    # footage at a low source bitrate has reached 12%
    need = req.size + max(req.size * 0.15, 512 << 20)
    free = shutil.disk_usage(VIDEO_DIR).free - reserved
    if free < need:
        raise HTTPException(507, f"Not enough disk space for {name}: it needs about {_gb(need)} and the video volume "
                                 f"has {_gb(max(free, 0))} free. Delete old videos, or set VIDEO_DATA to a folder on a larger disk.")

    root = VIDEO_DIR / uuid.uuid4().hex
    root.mkdir()
    suffix = Path(name).suffix.lower()
    (root / "source.part").touch()
    (root / "meta.json").write_text(json.dumps({
        "name": name, "size": req.size, "last_modified": req.last_modified, "created": time.time(),
        "ext": suffix if EXT_RE.match(suffix) else ".bin",
    }))
    return _upload_state(root)


@router.get("/uploads/{video_id}")
def upload_state(video_id: str):
    return _upload_state(_folder(video_id))


@router.put("/uploads/{video_id}")
async def upload_chunk(video_id: str, offset: int, request: Request):
    root = _folder(video_id)
    lock = _locks.setdefault(video_id, asyncio.Lock())
    async with lock:
        state = _upload_state(root)
        if state["complete"]:
            return state
        if offset != state["received"]:
            # The client is out of step (e.g. an earlier chunk was cut off): tell it where to resume
            return JSONResponse({**state, "detail": "Offset mismatch."}, status_code=409)
        part = root / "source.part"
        size = state["size"]
        written = offset
        buf = bytearray()
        # A chunk cut off part-way leaves valid bytes behind; the next request resumes after them
        with part.open("r+b") as f:
            f.seek(offset)
            async for data in request.stream():
                if written + len(buf) + len(data) > size:
                    raise HTTPException(400, "More data than the file's size.")
                buf += data
                if len(buf) >= WRITE_BUFFER:
                    await asyncio.to_thread(f.write, buf)
                    written += len(buf)
                    buf = bytearray()
            if buf:
                await asyncio.to_thread(f.write, buf)
                written += len(buf)
        if written == size:
            ext = _meta(root)["ext"]
            part.rename(root / f"source{ext}")
        return _upload_state(root)


@router.get("/{video_id}/proxy.mp4")
def proxy_video(video_id: str):
    path = _folder(video_id) / "proxy.mp4"
    if not path.exists():
        raise HTTPException(404, "The preview isn't ready yet.")
    return FileResponse(path, media_type="video/mp4")


# ---------- pass-through to the video service ----------

PASS_HEADERS = ("content-type", "content-length", "content-disposition", "cache-control", "etag", "last-modified")


async def _forward(request: Request, path: str):
    body = await request.body() if request.method in ("POST", "PUT") else None
    try:
        upstream = await _client.send(
            _client.build_request(request.method, f"{VIDEO_BASE_URL}/{path}", params=request.query_params,
                                  content=body, headers={"content-type": request.headers.get("content-type", "")}),
            stream=True,
        )
    except httpx.HTTPError as e:
        return JSONResponse({"detail": f"Video service unavailable at {VIDEO_BASE_URL} ({type(e).__name__}: {e})"},
                            status_code=502)
    headers = {k: v for k, v in upstream.headers.items() if k.lower() in PASS_HEADERS}
    return StreamingResponse(upstream.aiter_raw(), status_code=upstream.status_code, headers=headers,
                             background=BackgroundTask(upstream.aclose))


@router.get("")
async def list_videos(request: Request):
    return await _forward(request, "videos")


@router.get("/detectors")
async def detectors(request: Request):
    return await _forward(request, "detectors")


@router.get("/speech-models")
async def speech_models(request: Request):
    return await _forward(request, "speech-models")


@router.api_route("/{video_id}", methods=["GET", "DELETE"])
@router.api_route("/{video_id}/{rest:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def forward(video_id: str, request: Request, rest: str = ""):
    if not ID_RE.match(video_id):
        raise HTTPException(404, "Unknown video.")
    return await _forward(request, f"videos/{video_id}" + (f"/{rest}" if rest else ""))

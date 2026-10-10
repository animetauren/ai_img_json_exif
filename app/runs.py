"""Batch export runs: tagged copies are staged on the server, then downloaded as one zip.

The zip is streamed (stored, not compressed: photos don't compress) so large runs never
need to fit in memory or be written to disk a second time.
"""

import io
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import HTTPException

STAGING_DIR = Path(os.environ.get("EXPORT_STAGING_DIR", Path(tempfile.gettempdir()) / "photo-exports"))
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
STALE_AFTER = 24 * 3600  # runs whose zip was never downloaded are removed after a day
NAME_FILE = ".name"
MANIFEST_FILE = "_run.json"


def create(name: str) -> str:
    if not RUN_NAME_RE.match(name):
        raise HTTPException(400, "Run names use letters, digits, spaces, dots, dashes and underscores.")
    _remove_stale()
    run_id = uuid.uuid4().hex
    folder = STAGING_DIR / run_id
    folder.mkdir(parents=True)
    (folder / NAME_FILE).write_text(name)
    return run_id


def _remove_stale() -> None:
    if not STAGING_DIR.exists():
        return
    cutoff = time.time() - STALE_AFTER
    for folder in STAGING_DIR.iterdir():
        if folder.is_dir() and folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)


def folder(run_id: str) -> Path:
    path = STAGING_DIR / run_id
    if not RUN_ID_RE.match(run_id) or not path.is_dir():
        raise HTTPException(404, "Unknown or expired export run.")
    return path


def name(run_id: str) -> str:
    return (folder(run_id) / NAME_FILE).read_text()


def info(run_id: str) -> dict:
    """What a page needs to offer a run it lost track of: how many photos, and whether tagging finished."""
    root = folder(run_id)
    photos = 0
    modified = root.stat().st_mtime
    for path in root.rglob("*"):
        modified = max(modified, path.stat().st_mtime)
        if path.is_file() and not (path.parent == root and path.name in (NAME_FILE, MANIFEST_FILE)):
            photos += 1
    complete = None  # unknown: the page closed before it saved the manifest
    manifest = root / MANIFEST_FILE
    if manifest.is_file():
        try:
            complete = bool(json.loads(manifest.read_text()).get("complete"))
        except ValueError:
            pass
    return {
        "name": name(run_id), "photos": photos, "complete": complete,
        "manifest": manifest.is_file(), "idle": max(0, round(time.time() - modified)),
    }


def discard(run_id: str) -> None:
    shutil.rmtree(folder(run_id), ignore_errors=True)


def target(run_id: str, rel_path: str) -> Path:
    """Where a photo goes inside the run, refusing paths that would land outside it."""
    root = folder(run_id)
    parts = [p for p in rel_path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or ".." in parts or any("\0" in p for p in parts) or parts == [NAME_FILE]:
        raise HTTPException(400, f"Invalid file path '{rel_path}'.")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise HTTPException(400, f"Invalid file path '{rel_path}'.")
    return path


class _Sink(io.RawIOBase):
    """Write-only buffer the zip writes into; the stream drains it as it goes."""

    def __init__(self):
        self.chunks = []

    def writable(self):
        return True

    def write(self, b):
        self.chunks.append(bytes(b))
        return len(b)

    def drain(self):
        chunks, self.chunks = self.chunks, []
        return chunks


def zip_stream(run_id: str):
    """Yield a zip of the run with everything under a "<run name>/" folder.

    The run is deleted once the whole zip has been sent; an interrupted download leaves it
    for a retry until it goes stale.
    """
    root = folder(run_id)
    prefix = name(run_id)
    sink = _Sink()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or (path.parent == root and path.name == NAME_FILE):
                continue
            # from_file keeps the file's timestamp in the zip
            info = zipfile.ZipInfo.from_file(path, f"{prefix}/{path.relative_to(root).as_posix()}")
            with path.open("rb") as src, zf.open(info, "w", force_zip64=True) as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
                    yield from sink.drain()
    yield from sink.drain()
    shutil.rmtree(root, ignore_errors=True)

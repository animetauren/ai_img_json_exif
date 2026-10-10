"""Process photos from the command line with settings exported from the web UI.

    python -m app.cli catalog.settings.json ~/Pictures/trip --tag --zip

Export the settings with "Export settings" in the web UI. The CLI calls the LLM and the
vision service directly, using the same code as the server. Progress goes to stderr; each
processed photo is also printed to stdout as one JSON line. See the README's CLI section.
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from . import metadata
from .llm import DEFAULT_USER_PROMPT, chat_payload, extract_json, stream_chat
from .workflows import classify_vision, run_workflow, validate

MAX_IMAGE_EDGE = 1536  # same as the web UI, so results match
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
# Errors that mean a server is down or overloaded, not that the photo got a bad answer.
# Kept in step with TRANSIENT_RE in static/batch.js.
TRANSIENT_RE = re.compile(
    r"ConnectError|ConnectTimeout|ReadTimeout|WriteTimeout|PoolTimeout|TimeoutException|ReadError|WriteError|"
    r"RemoteProtocolError|All connection attempts failed|Connection refused|HTTP (408|429|5\d\d)\b",
    re.I,
)
RETRY_DELAYS = (5, 15, 30)
CONTAINER_HOSTS = ("host.containers.internal", "host.docker.internal")
# exiftool's System tags describe the file on this machine; the web UI drops them too
SYSTEM_TAGS = {"SourceFile", "Directory", "FileModifyDate", "FileAccessDate", "FileInodeChangeDate", "FilePermissions"}

EXIT_FAILED, EXIT_PAUSED, EXIT_INTERRUPTED = 1, 2, 130


def note(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------- settings ----------

def load_settings(path: str, args) -> dict:
    try:
        settings = json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read settings {path}: {e}")
    if settings.get("format") != "photo-organizer/settings":
        sys.exit(f"{path} is not a settings file. Use \"Export settings\" in the web UI to make one.")
    if settings.get("kind") not in ("llm", "vision", "workflow"):
        sys.exit(f"{path}: unknown kind {settings.get('kind')!r}.")
    if settings["kind"] == "workflow":
        try:
            validate(settings.get("workflow"))
        except ValueError as e:
            sys.exit(f"{path}: invalid workflow: {e}")

    llm = settings.setdefault("llm", {})
    if args.llm_url:
        llm["base_url"] = args.llm_url
    else:
        # The web UI's endpoint is as seen from inside the container
        parts = urlsplit(llm.get("base_url", ""))
        if parts.hostname in CONTAINER_HOSTS:
            netloc = parts.netloc.replace(parts.hostname, "localhost")
            llm["base_url"] = urlunsplit(parts._replace(netloc=netloc))
            uses_llm = settings["kind"] == "llm" or any(
                step.get("type") == "llm" and step.get("enabled", True) for step in settings.get("workflow", {}).get("steps", []))
            if uses_llm:
                note(f"Using {llm['base_url']} for the LLM ({parts.hostname} only works inside containers; --llm-url to change).")
    if args.model:
        llm["model"] = args.model
    return settings


def settings_label(settings: dict) -> str:
    """Same wording as the web UI's configLabel(), recorded as each result's source."""
    if settings["kind"] == "workflow":
        return f"workflow: {settings['workflow'].get('name')}"
    if settings["kind"] == "vision":
        return f"vision: {settings.get('vision', {}).get('model')}"
    return f"llm: {settings['llm'].get('model') or 'default model'}"


def settings_hash(settings: dict) -> str:
    """Identifies the settings that shape results, so resuming never mixes results from different
    settings. Endpoints and keys are left out: pointing at another server doesn't change answers."""
    kind = settings["kind"]
    llm = {k: v for k, v in settings["llm"].items() if k != "base_url"}
    relevant = {"kind": kind}
    if kind == "llm":
        relevant["llm"] = llm
    elif kind == "vision":
        relevant["vision"] = settings.get("vision")
    else:
        relevant.update(workflow=settings["workflow"], model=llm.get("model"))
    return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:12]


# ---------- photos ----------

def find_photos(paths: list, recursive: bool) -> list:
    """[(path, relative path in the output)]. Folder contents keep their paths inside the folder."""
    photos = []
    many = len(paths) > 1
    for arg in paths:
        root = Path(arg).expanduser()
        if root.is_dir():
            found = root.rglob("*") if recursive else root.glob("*")
            for f in sorted(found):
                if f.is_file() and f.suffix.lower() in PHOTO_EXTS and not f.name.startswith("."):
                    rel = f.relative_to(root).as_posix()
                    photos.append((f, f"{root.name}/{rel}" if many else rel))
        elif root.is_file():
            photos.append((root, root.name))
        else:
            sys.exit(f"Not found: {arg}")
    seen = {}
    for path, rel in photos:
        if rel in seen:
            sys.exit(f"Two photos would share the output path {rel}: {seen[rel]} and {path}")
        seen[rel] = path
    return photos


def photo_key(path: Path, rel: str) -> str:
    st = path.stat()
    return f"{rel}|{st.st_size}|{int(st.st_mtime * 1000)}"


def imaging():
    try:
        from PIL import Image, ImageOps
    except ImportError:
        sys.exit("Pillow is not installed: pip install -r requirements-cli.txt")
    try:
        from pi_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass  # HEIC/HEIF photos will fail to open
    return Image, ImageOps


def to_srgb(img, icc: Optional[bytes]):
    """Convert to sRGB like a browser does before drawing: iPhone photos are Display P3, and
    passing P3 values on as if they were sRGB shifts colours enough to change model answers."""
    if icc and img.mode in ("RGB", "CMYK"):
        try:
            from PIL import ImageCms
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            return ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"),
                                             renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode="RGB")
        except Exception:  # noqa: BLE001 - a broken profile: use the pixels as they are
            pass
    return img.convert("RGB")


def image_data_url(path: Path) -> str:
    """Scale like the web UI: upright, sRGB, longest edge at most 1536 px, JPEG quality 90."""
    Image, ImageOps = imaging()
    with Image.open(path) as im:
        icc = im.info.get("icc_profile")
        img = to_srgb(ImageOps.exif_transpose(im), icc)
    img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


async def read_exif(path: Path) -> dict:
    flat = (await metadata.run_exiftool(str(path)))[0]
    return {k: v for k, v in flat.items() if k not in SYSTEM_TAGS}


# ---------- running ----------

def error_text(e: Exception) -> str:
    return str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}"


async def run_one(settings: dict, data_url: str, exif: dict, cfg: dict) -> dict:
    """Run one image. Returns a record shaped like the web UI's (see recordResult in batch.js)."""
    kind = settings["kind"]
    llm = settings["llm"]
    if kind == "llm":
        payload = chat_payload(
            model=llm.get("model") or "",
            system_prompt=llm.get("system_prompt", ""),
            user_prompt=llm.get("user_prompt") or DEFAULT_USER_PROMPT,
            image_data_url=data_url,
            temperature=float(llm.get("temperature", 0.2)),
            max_tokens=int(llm.get("max_tokens", 1024)),
            json_mode=bool(llm.get("json_mode")),
        )
        record = {"kind": "llm", "text": "", "error": None, "json": None, "json_error": None, "usage": None}
        async for ev in stream_chat(llm.get("base_url", ""), cfg["api_key"], payload, cfg["timeout"]):
            if ev["type"] == "token":
                record["text"] += ev["text"]
            elif ev["type"] == "usage":
                record["usage"] = ev["usage"]
            elif ev["type"] == "error":
                record["error"] = ev["message"]
        if record["text"]:
            try:
                record["json"] = extract_json(record["text"])
            except ValueError as e:
                record["json_error"] = str(e)
        return record

    if kind == "vision":
        vision = settings.get("vision", {})
        params = {k: vision.get(k) for k in ("labels", "label_template", "top_k", "threshold")}
        try:
            async with httpx.AsyncClient(timeout=cfg["timeout"]) as client:
                data = await classify_vision(client, cfg["vision_url"], vision.get("model"), params, data_url)
            return {"kind": "vision", "data": data, "error": None}
        except Exception as e:  # noqa: BLE001 - recorded as the photo's error
            return {"kind": "vision", "data": None, "error": error_text(e)}

    record = {"kind": "workflow", "events": [], "result": None, "error": None}
    try:
        async for ev in run_workflow(
            settings["workflow"],
            image_data_url=data_url,
            exif=exif,
            llm={"base_url": llm.get("base_url", ""), "api_key": cfg["api_key"], "model": llm.get("model") or ""},
            vision_url=cfg["vision_url"],
            timeout=cfg["timeout"],
        ):
            if ev["type"] in ("token", "reasoning", "prompt", "usage", "timings"):
                continue
            record["events"].append(ev)
            if ev["type"] == "done":
                record["result"] = ev["result"]
    except Exception as e:  # noqa: BLE001
        record["error"] = error_text(e)
    return record


def record_result(record: dict):
    if record.get("error"):
        return None
    if record["kind"] == "llm":
        return record["json"]
    if record["kind"] == "vision":
        return record["data"]
    return record["result"]


def step_errors(record: dict) -> list:
    return [ev["message"] for ev in record.get("events", []) if ev["type"] == "step_error"]


def record_ok(record: dict) -> bool:
    return record_result(record) is not None and not step_errors(record)


def record_error(record: dict) -> Optional[str]:
    if record.get("error"):
        return record["error"]
    if record["kind"] == "llm" and record_result(record) is None:
        return record["json_error"] or "Model returned no content."
    errors = step_errors(record)
    return "; ".join(errors) if errors else None


def transient_error(record: dict) -> Optional[str]:
    for message in [record.get("error"), *step_errors(record)]:
        if message and TRANSIENT_RE.search(message):
            return message
    return None


def fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 10:
        return f"{seconds:.1f}s"
    s = round(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {s // 60 % 60:02d}m"


class Eta:
    """Average of the last 20 photos, leaving out the first (often includes loading the model)."""

    def __init__(self):
        self.times = []

    def add(self, seconds: float) -> None:
        self.times.append(seconds)

    def average(self) -> float:
        recent = (self.times[1:] if len(self.times) > 1 else self.times)[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    def describe(self, left: int) -> str:
        avg = self.average()
        return f"avg {fmt_duration(avg)} · ~{fmt_duration(avg * left)} left" if avg else ""


# ---------- outputs ----------

class Outputs:
    """results.jsonl, sidecars, tagged copies, _run.json and the zip, all under one folder."""

    def __init__(self, args, settings: dict, run_hash: str):
        self.args = args
        self.settings = settings
        self.hash = run_hash
        self.out = Path(args.out).expanduser()
        self.out.mkdir(parents=True, exist_ok=True)
        self.jsonl = self.out / "results.jsonl"
        self.tagging = args.tag or args.zip
        # --zip without --tag stages tagged copies here until the zip is built
        self.tag_dir = self.out if args.tag else self.out / ".zip-staging"
        self.previous = {} if args.rerun else self._read_previous()

    def _read_previous(self) -> dict:
        previous = {}
        if self.jsonl.exists():
            for line in self.jsonl.read_text().splitlines():
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if entry.get("settings") == self.hash:
                    previous[entry["key"]] = entry  # later lines win
        return previous

    def done_before(self, key: str) -> Optional[dict]:
        entry = self.previous.get(key)
        return entry if entry and entry.get("ok") else None

    async def finish_photo(self, path: Path, rel: str, entry: dict) -> dict:
        """Write this photo's sidecar and tagged copy as requested; returns the entry with the tag report."""
        result = entry.get("result")
        if entry.get("ok") and self.args.sidecar:
            sidecar = self.out / f"{rel}.json"
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        target = self.tag_dir / rel
        if entry.get("ok") and self.tagging and not (entry.get("tag") and target.exists()):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            try:
                report = await metadata.export(str(target), result, source=entry["source"], overwrite=self.args.overwrite)
                entry["tag"] = {k: report[k] for k in ("written", "skipped", "warnings")}
            except Exception as e:  # noqa: BLE001 - e.g. a format exiftool can't write
                target.unlink(missing_ok=True)
                entry["tag"] = {"error": str(e).replace(str(target), rel)}
        return entry

    def log(self, entry: dict) -> None:
        if self.args.jsonl:
            with self.jsonl.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.previous[entry["key"]] = entry

    def manifest(self, photos: list, complete: bool) -> dict:
        files = []
        for _, rel, key in photos:
            entry = self.previous.get(key)
            if not entry:
                continue
            item = {"path": rel, "ok": entry["ok"]}
            if entry.get("error"):
                item["error"] = entry["error"]
            if entry.get("tag"):
                item.update({k: v for k, v in entry["tag"].items() if k != "warnings" or v})
            files.append(item)
        return {
            "run": self.args.name, "source": settings_label(self.settings), "overwrite": self.args.overwrite,
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"), "complete": complete,
            "settings": self.settings, "files": files,
        }

    def write_manifest(self, photos: list, complete: bool) -> None:
        if not self.tagging:
            return
        manifest = json.dumps(self.manifest(photos, complete), indent=2, ensure_ascii=False)
        if self.args.tag:
            (self.out / "_run.json").write_text(manifest)
        if self.args.zip and complete:
            self._write_zip(photos, manifest)

    def _write_zip(self, photos: list, manifest: str) -> None:
        name = self.args.name
        zip_path = self.out / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for _, rel, key in photos:
                entry = self.previous.get(key)
                if not entry or not entry.get("ok"):
                    continue
                tagged = self.tag_dir / rel
                if tagged.exists():
                    zf.write(tagged, f"{name}/{rel}")
                sidecar = self.out / f"{rel}.json"
                if self.args.sidecar and sidecar.exists():
                    zf.write(sidecar, f"{name}/{rel}.json")
            zf.writestr(f"{name}/_run.json", manifest)
        if not self.args.tag:
            shutil.rmtree(self.tag_dir, ignore_errors=True)
        note(f"Wrote {zip_path}")


# ---------- main loop ----------

async def run(args) -> int:
    settings = load_settings(args.settings, args)
    if (args.tag or args.zip) and not metadata.EXIFTOOL:
        sys.exit("--tag and --zip need exiftool: brew install exiftool (or apt install libimage-exiftool-perl)")
    if settings["kind"] == "workflow" and not metadata.EXIFTOOL:
        note("exiftool not found: workflow {{exif.*}} placeholders will be 'unknown'.")
    imaging()  # fail early if Pillow is missing

    cfg = {
        "api_key": args.api_key or os.environ.get("LLM_API_KEY", ""),
        "vision_url": (args.vision_url or os.environ.get("VISION_BASE_URL") or "http://localhost:8000").rstrip("/"),
        "timeout": args.timeout,
    }
    label = settings_label(settings)
    run_hash = settings_hash(settings)
    outputs = Outputs(args, settings, run_hash)
    photos = [(path, rel, photo_key(path, rel)) for path, rel in find_photos(args.paths, args.recursive)]
    if not photos:
        sys.exit("No photos found.")

    todo = []
    for path, rel, key in photos:
        before = outputs.done_before(key)
        if not before:
            todo.append((path, rel, key))
        elif outputs.tagging or args.sidecar:
            # Already done with these settings: only add outputs this run asks for
            snapshot = json.dumps(before, sort_keys=True)
            entry = await outputs.finish_photo(path, rel, dict(before))
            if json.dumps(entry, sort_keys=True) != snapshot:
                outputs.log(entry)
    skipped = len(photos) - len(todo)
    if not todo:
        outputs.write_manifest(photos, complete=True)
        note(f"Nothing to do: all {len(photos)} photos already have results for {label} in {outputs.out} (--rerun to redo).")
        return 0
    note(f"{label} · {len(photos)} photos" + (f", {skipped} already done (resuming; --rerun to redo)" if skipped else "") +
         f" · output {outputs.out}")

    eta = Eta()
    failed = 0
    started = time.monotonic()
    try:
        for n, (path, rel, key) in enumerate(todo, 1):
            t = time.monotonic()
            try:
                data_url = image_data_url(path)
            except Exception as e:  # noqa: BLE001 - unreadable or unsupported image
                record = {"kind": settings["kind"], "error": f"Can't open image: {error_text(e)}".replace(str(path), rel)}
            else:
                exif = {}
                if settings["kind"] == "workflow" and metadata.EXIFTOOL:
                    try:
                        exif = await read_exif(path)
                    except Exception:  # noqa: BLE001 - run without EXIF
                        pass
                record = await run_one(settings, data_url, exif, cfg)
                retries = 0
                while transient_error(record) and retries < len(RETRY_DELAYS):
                    delay = RETRY_DELAYS[retries]
                    retries += 1
                    note(f"  {rel}: {transient_error(record)}. Retrying in {delay}s ({retries} of {len(RETRY_DELAYS)})…")
                    await asyncio.sleep(delay)
                    record = await run_one(settings, data_url, exif, cfg)
                if transient_error(record):
                    # Nothing is saved for this photo, so re-running the command picks up here
                    outputs.write_manifest(photos, complete=False)
                    note(f"Paused: {label} is not responding ({transient_error(record)}). {len(todo) - n + 1} photos left. "
                         f"Fix the server, then run the same command{'' if argv_has_out() else f' with --out {args.out}'} to resume.")
                    return EXIT_PAUSED
                if not retries:  # waiting to retry would skew the average
                    eta.add(time.monotonic() - t)

            ok = record_ok(record)
            entry = {
                "path": rel, "key": key, "settings": run_hash, "ok": ok,
                "result": record_result(record) if ok else None,
                "error": None if ok else (record_error(record) or "No result."),
                "source": label, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(time.monotonic() - t, 2),
            }
            entry = await outputs.finish_photo(path, rel, entry)
            outputs.log(entry)
            print(json.dumps({k: entry[k] for k in ("path", "ok", "result", "error")}, ensure_ascii=False), flush=True)
            if not ok:
                failed += 1
            extra = "" if ok else f" {entry['error']}"
            if entry.get("tag", {}).get("error"):
                extra += f" (not tagged: {entry['tag']['error']})"
            progress = f" · {eta.describe(len(todo) - n)}" if eta.average() and n < len(todo) else ""
            note(f"[{n}/{len(todo)}] {rel} {'✓' if ok else '✗'} {fmt_duration(entry['seconds'])}{progress}{extra}")
    except asyncio.CancelledError:
        # Ctrl-C on Python 3.11+ cancels this task; main() reports it
        outputs.write_manifest(photos, complete=False)
        raise

    outputs.write_manifest(photos, complete=True)
    avg = eta.average()
    note(f"Done: {len(todo) - failed} of {len(todo)} ok in {fmt_duration(time.monotonic() - started)}" +
         (f" (avg {fmt_duration(avg)} per photo)" if avg else "") + (f", {failed} failed" if failed else "") + ".")
    return EXIT_FAILED if failed else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Classify photos with settings exported from the web UI (\"Export settings\").",
        epilog="Exit codes: 0 all ok, 1 some photos failed, 2 paused (a server stayed unreachable), "
               "130 interrupted. Re-running the same command resumes.",
    )
    p.add_argument("settings", help="settings file exported from the web UI")
    p.add_argument("paths", nargs="+", metavar="PATH", help="photos and/or folders")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="don't look inside subfolders")
    p.add_argument("-o", "--out", help="output folder (default ./photo-organizer-runs/<name>)")
    p.add_argument("--name", help="run name, used for the default output folder and the zip (default run-<date>-<time>)")
    p.add_argument("--jsonl", action=argparse.BooleanOptionalAction, default=True,
                   help="write results.jsonl, one line per photo; needed for resuming")
    p.add_argument("--tag", action="store_true", help="write tagged copies to the output folder, keeping names and subfolders")
    p.add_argument("--zip", action="store_true", help="write <name>.zip of tagged copies")
    p.add_argument("--sidecar", action="store_true", help="write <photo>.json with each photo's result")
    p.add_argument("--overwrite", action="store_true", help="replace metadata the photos already have (default: keep it)")
    p.add_argument("--rerun", action="store_true", help="run every photo again, ignoring earlier results in the output folder")
    p.add_argument("--api-key", help="LLM API key (default $LLM_API_KEY)")
    p.add_argument("--llm-url", help="override the settings' LLM endpoint, e.g. http://localhost:1234/v1")
    p.add_argument("--model", help="override the LLM model (single LLM, or a workflow's default)")
    p.add_argument("--vision-url", help="vision service (default $VISION_BASE_URL or http://localhost:8000)")
    p.add_argument("--timeout", type=float, default=300, help="seconds per request (default 300)")
    args = p.parse_args(argv)
    if not args.name:
        args.name = datetime.now().strftime("run-%Y-%m-%d-%H%M")
    if not args.out:
        args.out = str(Path("photo-organizer-runs") / args.name)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    # The shared workflow code logs for the server; the CLI reports progress itself
    logging.getLogger("uvicorn.error").addHandler(logging.NullHandler())
    logging.getLogger("uvicorn.error").propagate = False
    try:
        return asyncio.run(run(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        note(f"\nInterrupted. Finished photos are saved in {args.out}; run the same command"
             f"{'' if argv_has_out() else f' with --out {args.out}'} to resume.")
        return EXIT_INTERRUPTED


def argv_has_out() -> bool:
    return any(a in ("-o", "--out") or a.startswith("--out=") for a in sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())

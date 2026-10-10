"""exiftool helpers: read every tag from a file, and write classification results back into one.

Export maps well-known JSON keys onto standard EXIF / XMP / IPTC tags through a fixed alias
table, and stores the whole result as JSON in UserComment so nothing is lost. Tags the file
already has are left alone unless overwrite is set.
"""

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

EXIFTOOL = shutil.which("exiftool")

# The EXIF APP1 segment holds at most 64 KB including the camera's own tags, so larger
# dumps only go to XMP, which has no such limit
MAX_EXIF_DUMP = 32_000

DESCRIPTION = ("IFD0:ImageDescription", "XMP-dc:Description", "IPTC:Caption-Abstract")
KEYWORDS = ("XMP-dc:Subject", "IPTC:Keywords")
DUMP = ("ExifIFD:UserComment", "XMP-exif:UserComment")

# JSON key (lowercase) -> (kind, target tags). "text" takes one value (last match wins),
# "list" collects values from every match, "rating" is an integer 0-5.
ALIASES = {
    "description": ("text", DESCRIPTION),
    "caption": ("text", DESCRIPTION),
    "summary": ("text", DESCRIPTION),
    "title": ("text", ("XMP-dc:Title", "IPTC:ObjectName")),
    "headline": ("text", ("XMP-photoshop:Headline", "IPTC:Headline")),
    "tags": ("list", KEYWORDS),
    "keywords": ("list", KEYWORDS),
    "labels": ("list", KEYWORDS),
    "objects": ("list", KEYWORDS),
    "category": ("list", KEYWORDS),
    "rating": ("rating", ("XMP-xmp:Rating",)),
    "artist": ("text", ("IFD0:Artist", "XMP-dc:Creator")),
    "creator": ("text", ("IFD0:Artist", "XMP-dc:Creator")),
    "author": ("text", ("IFD0:Artist", "XMP-dc:Creator")),
    "copyright": ("text", ("IFD0:Copyright", "XMP-dc:Rights")),
    "city": ("text", ("XMP-photoshop:City",)),
    "state": ("text", ("XMP-photoshop:State",)),
    "country": ("text", ("XMP-photoshop:Country",)),
    "location": ("text", ("XMP-iptcCore:Location",)),
    "event": ("text", ("XMP-iptcExt:Event",)),
    "people": ("list", ("XMP-iptcExt:PersonInImage",)),
    "persons": ("list", ("XMP-iptcExt:PersonInImage",)),
}


async def _run(*args: str) -> tuple:
    proc = await asyncio.create_subprocess_exec(
        EXIFTOOL, "-api", "largefilesupport=1", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), timeout=60)
    return proc.returncode, out.decode(errors="replace"), err.decode(errors="replace")


async def run_exiftool(path: str, *args: str) -> list:
    """Read tags as JSON."""
    code, out, err = await _run("-json", *args, path)
    # exiftool exits non-zero for minor warnings but still prints JSON
    if not out.strip():
        raise RuntimeError(err.strip() or f"exiftool exited {code}")
    return json.loads(out)


# ---------- JSON -> tags ----------

def _is_scalar(v) -> bool:
    return isinstance(v, (str, int, float)) and not isinstance(v, bool)


def _text(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".")
    return str(v).strip()


def collect(result) -> dict:
    """Map a result's known keys to tags: {tag: str | [str]}.

    Walks the JSON in order, so for workflow results (keyed by step, in run order) the last
    step's description wins while keywords are pooled from every step. An alias only applies
    when the value has the right shape; otherwise the walk descends into it, which is how a
    workflow step called "objects" (a dict) is searched rather than taken as a keyword list.
    """
    texts: dict = {}
    lists: dict = {}

    def add_list(tags, items):
        for tag in tags:
            values = lists.setdefault(tag, [])
            seen = {v.lower() for v in values}
            for item in items:
                if item and item.lower() not in seen:
                    values.append(item)
                    seen.add(item.lower())

    def apply(kind, tags, value) -> bool:
        if kind == "list":
            if _is_scalar(value):
                value = [value]
            if not isinstance(value, list) or not value or not all(_is_scalar(v) for v in value):
                return False
            add_list(tags, [_text(v) for v in value])
            return True
        if kind == "rating":
            if not _is_scalar(value) or isinstance(value, str) or not float(value).is_integer() or not 0 <= value <= 5:
                return False
            value = str(int(value))
        elif isinstance(value, list) and value and all(_is_scalar(v) for v in value):
            value = ", ".join(_text(v) for v in value)
        elif _is_scalar(value):
            value = _text(value)
        else:
            return False
        if not value:
            return False
        for tag in tags:
            texts[tag] = value
        return True

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                alias = ALIASES.get(str(key).lower())
                if not (alias and apply(*alias, value)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return {**texts, **{tag: values for tag, values in lists.items() if values}}


def _empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return all(_empty(v) for v in value)
    # Cameras often write a UserComment of spaces or NULs; filling that in loses nothing
    return not str(value).strip(" \t\r\n\0")


def _short(value) -> str:
    s = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else str(value)
    return s if len(s) <= 80 else s[:77] + "…"


# ---------- export ----------

async def export(path: str, result, *, source: str, overwrite: bool) -> dict:
    """Write `result` into the file at `path` in place. Returns what was written and skipped."""
    planned = collect(result)
    dump = json.dumps(
        {
            "generator": "photo_organizer",
            "exported": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": source,
            "result": result,
        },
        separators=(",", ":"),
        ensure_ascii=True,  # keeps the EXIF copy plain ASCII rather than UCS-2
    )
    warnings = []
    if len(dump) > MAX_EXIF_DUMP:
        planned[DUMP[1]] = dump
        warnings.append(f"The JSON dump is {len(dump) // 1000} KB, too large for EXIF, so it is only in XMP-exif:UserComment.")
    else:
        for tag in DUMP:
            planned[tag] = dump

    def current(info: dict) -> dict:
        return {k.lower(): v for k, v in info.items()}

    before = current((await run_exiftool(path, "-G1", "-IPTC:CodedCharacterSet", *(f"-{t}" for t in planned)))[0])

    skipped = []
    if not overwrite:
        for tag in list(planned):
            existing = before.get(tag.lower())
            if not _empty(existing):
                skipped.append({"tag": tag, "existing": _short(existing)})
                del planned[tag]

    if planned:
        args = []
        if any(t.startswith("IPTC:") for t in planned) and "iptc:codedcharacterset" not in before:
            args.append("-IPTC:CodedCharacterSet=UTF8")
        for tag, value in planned.items():
            # Repeating -TAG=item builds a list, replacing any existing items
            for item in value if isinstance(value, list) else [value]:
                args.append(f"-{tag}={item}")
        # -m: ignore minor problems such as IPTC strings over their length limit (they are truncated)
        code, _, err = await _run("-m", "-overwrite_original", *args, path)
        if code != 0:
            raise RuntimeError(err.strip() or f"exiftool exited {code}")

    after = current((await run_exiftool(path, "-G1", *(f"-{t}" for t in planned)))[0]) if planned else {}
    written = [t for t in planned if not _empty(after.get(t.lower()))]
    unsupported = [t for t in planned if t not in written]
    if unsupported:
        warnings.append(f"Not supported in {Path(path).suffix or 'this'} files: {', '.join(unsupported)}.")

    return {"written": written, "skipped": skipped, "warnings": warnings}

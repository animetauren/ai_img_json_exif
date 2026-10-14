"""Speech to text with Useful Sensors Moonshine (on-device, <=64 s per call).

The audio track is decoded once to 16 kHz mono PCM next to the video and read through a memory
map, then transcribed in windows of about 60 seconds. Moonshine only accepts segments up to
64 seconds in a single call, so longer audio is split; a window boundary is used directly (no
silence search, Moonshine has no word timestamps to protect).

The transcription result keeps the same shape as the Whisper version (language, probability,
segments with start/end/text/words) so downstream consumers, EXIF write-back and subtitle
export keep working. Word-level timings and per-word probabilities are not available from
Moonshine, so `words` is left empty and `cues()` falls back to one cue per segment.
"""

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

CACHE_DIR = Path(os.environ.get("VIDEO_CACHE", "/cache"))
SAMPLE_RATE = 16000
WINDOW = 60  # seconds of audio per transcribe call (Moonshine max is 64 s)
DEFAULT_MODEL = "base"


@dataclass
class Model:
    id: str
    label: str
    repo: str
    size_mb: int
    note: str


MODELS = {m.id: m for m in [
    Model("tiny", "Moonshine tiny", "UsefulSensors/moonshine", 28, "Fastest; ~10 M params."),
    Model("base", "Moonshine base", "UsefulSensors/moonshine", 76, "Good accuracy; ~30 M params."),
]}


def languages() -> list:
    # Moonshine ships English-only models.
    return ["en"]


def model_dir() -> Path:
    return CACHE_DIR / "moonshine"


def downloaded(model: Model) -> bool:
    snap = model_dir() / model.id
    return snap.exists()


def normalize(params: Optional[dict], tracks: int) -> dict:
    params = params or {}
    model = params.get("model") if params.get("model") in MODELS else DEFAULT_MODEL
    language = str(params.get("language") or "").lower()
    if language and language != "en":
        language = "en"
    try:
        track = int(params.get("track") or 0)
    except (TypeError, ValueError):
        track = 0
    return {"model": model, "language": language or "en", "track": min(max(track, 0), max(tracks - 1, 0))}


def key(params: dict) -> str:
    out = "moonshine-" + params["model"]
    return out + ("-a" + str(params["track"]) if params.get("track") else "")


def label(params: dict) -> str:
    return MODELS[params["model"]].label


# ---------- transcription ----------

def sec(x) -> float:
    return round(float(x), 3)


def transcribe(pcm_path, params, report, stage, threads) -> dict:
    """Transcribe 16 kHz mono s16le PCM. report(fraction) raises once the job is cancelled."""
    from moonshine import transcribe as ms_transcribe, load_model

    m = MODELS[params["model"]]
    stage("Loading " + m.label)
    model_name = "moonshine/tiny" if params["model"] == "tiny" else "moonshine/base"
    model = load_model(model_name)
    stage("")
    try:
        pcm = np.memmap(pcm_path, dtype="<i2", mode="r")
        total = len(pcm)
        segments = []
        start = 0
        language = params["language"] or "en"
        while start < total:
            end = min(total, start + WINDOW * SAMPLE_RATE)
            offset = start / SAMPLE_RATE
            if total:
                report(start / total)
            audio = np.asarray(pcm[start:end], dtype=np.float32) / 32768.0
            audio = audio.reshape(1, -1)
            try:
                result = ms_transcribe(audio, model=model)
            except TypeError:
                result = ms_transcribe(audio, model_name)
            text = ""
            if isinstance(result, (list, tuple)) and result:
                text = str(result[0]).strip()
            elif isinstance(result, str):
                text = result.strip()
            if text:
                segments.append({
                    "start": sec(offset), "end": sec(end / SAMPLE_RATE), "text": text,
                    "words": [],
                })
            del audio
            start = end
            report(min(0.999, start / total) if total else 1.0)
        return {"language": language, "language_probability": 1.0, "segments": segments}
    finally:
        del model
        gc.collect()


# ---------- reading a transcript ----------

def spans(segments, gap=1.0):
    out = []
    for s in segments:
        if out and s["start"] - out[-1][1] < gap:
            out[-1][1] = max(out[-1][1], round(s["end"], 1))
        else:
            out.append([round(s["start"], 1), round(s["end"], 1)])
    return out


def words_of(segments):
    out = []
    for s in segments:
        out.extend(s.get("words") or [{"start": s["start"], "end": s["end"], "word": " " + s["text"]}])
    return out


def speech_by_scene(segments, scenes):
    texts = [[] for _ in scenes]
    starts = [sc["start"] for sc in scenes]
    for w in words_of(segments):
        mid = (w["start"] + w["end"]) / 2
        i = int(np.searchsorted(starts, mid, side="right")) - 1
        if 0 <= i < len(scenes):
            texts[i].append(w["word"])
    return ["".join(t).strip() for t in texts]


# ---------- subtitles ----------

CUE_CHARS = 84
CUE_SECONDS = 6.0
LINE_CHARS = 42


def cues(segments):
    out = []
    for s in segments:
        words = s.get("words") or []
        if not words:
            out.append((s["start"], s["end"], s["text"]))
            continue
        cur = []
        for w in words:
            text = "".join(x["word"] for x in cur + [w]).strip()
            if cur and (len(text) > CUE_CHARS or w["end"] - cur[0]["start"] > CUE_SECONDS):
                out.append((cur[0]["start"], cur[-1]["end"], "".join(x["word"] for x in cur).strip()))
                cur = []
            cur.append(w)
        if cur:
            out.append((cur[0]["start"], cur[-1]["end"], "".join(x["word"] for x in cur).strip()))
    return [(a, b, wrap(t)) for a, b, t in out if t]


def wrap(text):
    if len(text) <= LINE_CHARS:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    i = min(spaces, key=lambda x: abs(x - mid))
    return text[:i] + "\n" + text[i + 1:]


def stamp(sec_val, sep):
    ms = max(0, round(sec_val * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d%s%03d" % (h, m, s, sep, ms)


def to_srt(segments):
    return "\n".join("%d\n%s --> %s\n%s\n" % (i, stamp(a, ","), stamp(b, ","), t)
                     for i, (a, b, t) in enumerate(cues(segments), 1))


def to_vtt(segments):
    return "WEBVTT\n\n" + "\n".join("%s --> %s\n%s\n" % (stamp(a, "."), stamp(b, "."), t) for a, b, t in cues(segments))


def to_text(segments):
    return "".join("[%s] %s\n" % (stamp(s["start"], ".")[:-4], s["text"]) for s in segments)

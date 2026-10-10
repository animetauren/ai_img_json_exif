"""Scene detectors. Each runs over the low-res proxy and finds the frames where new scenes start.

Every detector streams frames, so memory stays flat however long the video is.

"Live" detectors (TransNetV2, FFmpeg) return a score for every frame and leave the cutting to
a rescore function: cuts are wherever the score crosses the threshold. Their scores are saved,
so a new threshold is applied instantly without decoding again, and the browser runs the same
rule (cutsFromScores in video.js) to preview cuts while the threshold line is dragged.
PySceneDetect detectors cut inside the library, so they return cuts plus a pooled curve.
"""

import hashlib
import json
import math
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

CACHE_DIR = Path(os.environ.get("VIDEO_CACHE", "/cache"))
CURVE_BUCKETS = 2000

# Weights converted from the authors' TensorFlow checkpoint, pinned to a revision and checked
# against a hash, since torch checkpoints are pickles
TRANSNET_URL = (
    "https://huggingface.co/Sn4kehead/TransNetV2/resolve/"
    "db6ceabeb692ec71ecc6beb6d00db67ad1412d7f/transnetv2-pytorch-weights.pth"
)
TRANSNET_SHA256 = "834b10f25ae9e1b4e4f2652fe2843bd2b1388057a435d68b7c52635578fcc04d"
TRANSNET_BATCH = 4  # 100-frame windows per forward pass


class Cancelled(Exception):
    pass


@dataclass
class Param:
    name: str
    label: str
    default: float
    min: float
    max: float
    step: float
    help: str = ""


@dataclass
class Context:
    """What a detector gets: the proxy, its frame rate and length, and a progress callback."""
    proxy: str
    fps: float
    duration: float
    params: dict
    report: Callable[[float], None]  # fraction 0..1; raises Cancelled once the job is cancelled

    @property
    def frames(self) -> int:
        return max(1, round(self.duration * self.fps))

    def min_frames(self) -> int:
        return min_frames(self.params, self.fps)


def min_frames(params: dict, fps: float) -> int:
    # Rounds halves up, like Math.round in the browser's preview (Python's round() would not)
    return max(1, math.floor(params["min_scene"] * fps + 0.5))


@dataclass
class Detector:
    id: str
    label: str
    description: str
    params: List[Param]
    run: Optional[Callable[[Context], dict]]
    note: str = ""
    # (scores, threshold, min_frames) -> cuts, for detectors whose run returns per-frame scores
    rescore: Optional[Callable] = None
    curve_param: str = "threshold"  # the param the curve's dashed line shows

    @property
    def available(self) -> bool:
        return self.run is not None

    @property
    def live(self) -> bool:
        return self.rescore is not None

    def normalize(self, params: Optional[dict]) -> dict:
        """Fill in defaults, drop unknown keys and clamp each value to its range."""
        params = params or {}
        out = {}
        for p in self.params:
            try:
                value = float(params.get(p.name, p.default))
            except (TypeError, ValueError):
                value = p.default
            out[p.name] = round(min(max(value, p.min), p.max), 4)
        return out

    def key(self, params: dict) -> str:
        digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.id}-{digest}"


# ---------- shared helpers ----------

class Curve:
    """Per-frame scores pooled into CURVE_BUCKETS buckets, keeping the peak of each bucket."""

    def __init__(self, frames: int, below: bool = False):
        self.frames = max(1, frames)
        self.below = below  # cuts are where the score drops below the threshold
        self.values: list = [None] * min(CURVE_BUCKETS, self.frames)

    def add(self, frame: int, value: float) -> None:
        i = min(len(self.values) - 1, max(0, frame * len(self.values) // self.frames))
        old = self.values[i]
        if old is None or (value < old if self.below else value > old):
            self.values[i] = value

    def result(self, threshold: float) -> dict:
        return {
            "values": [None if v is None else round(float(v), 4) for v in self.values],
            "direction": "below" if self.below else "above",
            "threshold": threshold,
        }


def enforce_min_length(cuts, min_frames: int) -> list:
    out, prev = [], 0
    for c in sorted(set(int(c) for c in cuts)):
        if c - prev >= min_frames:
            out.append(c)
            prev = c
    return out


def cuts_at_peaks(scores: np.ndarray, threshold: float, min_frames: int) -> list:
    """A transition is a run of frames above the threshold. Its peak marks the last frame of the
    outgoing shot for hard cuts (checked against known cuts), so the new scene starts after it."""
    cuts, above = [], np.flatnonzero(scores > threshold)
    if len(above):
        runs = np.split(above, np.flatnonzero(np.diff(above) > 1) + 1)
        cuts = [int(r[np.argmax(scores[r])]) + 1 for r in runs]
    return enforce_min_length([c for c in cuts if 0 < c < len(scores)], min_frames)


def cuts_above(scores: np.ndarray, threshold: float, min_frames: int) -> list:
    """Every frame scoring above the threshold starts a scene."""
    cuts = np.flatnonzero(scores > threshold)
    return enforce_min_length([int(c) for c in cuts if c > 0], min_frames)


def curve_from_scores(scores: np.ndarray, threshold: float) -> dict:
    """The same pooled curve as Curve, from a whole score array at once."""
    n = len(scores)
    buckets = min(CURVE_BUCKETS, n)
    starts = np.searchsorted((np.arange(n) * buckets) // n, np.arange(buckets))
    values = np.maximum.reduceat(scores, starts)
    return {"values": [round(float(v), 4) for v in values], "direction": "above", "threshold": threshold}


class Reporter:
    """Calls ctx.report every 0.1% of frames or half a second, whichever comes first, so per-frame
    callbacks stay cheap but a cancel (raised from ctx.report) takes effect quickly."""

    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.next = 0
        self.last = time.monotonic()

    def frame(self, n: int) -> None:
        now = time.monotonic()
        if n >= self.next or now - self.last > 0.5:
            self.next = n + max(1, self.ctx.frames // 1000)
            self.last = now
            self.ctx.report(min(0.999, n / self.ctx.frames))


def ffmpeg_reader(args: list) -> subprocess.Popen:
    """Start ffmpeg writing to stdout; stderr goes to a temp file so a full pipe can't stall it."""
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", *args],
        stdout=subprocess.PIPE, stderr=err,
    )
    proc.errfile = err
    return proc


def finish(proc: subprocess.Popen, cancelled: bool = False) -> None:
    if cancelled and proc.poll() is None:
        proc.kill()
    proc.stdout.close()
    code = proc.wait()
    proc.errfile.seek(0)
    err = proc.errfile.read().decode(errors="replace").strip()
    proc.errfile.close()
    if code != 0 and not cancelled:
        raise RuntimeError(f"ffmpeg exited {code}: {err[-500:]}")


# ---------- TransNetV2 ----------

_transnet = None
_transnet_lock = threading.Lock()


def _transnet_model():
    global _transnet
    with _transnet_lock:
        if _transnet is None:
            import torch
            from transnetv2 import TransNetV2

            path = CACHE_DIR / "transnetv2" / f"{TRANSNET_SHA256[:16]}.pth"
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                part = path.with_suffix(".part")
                with urllib.request.urlopen(TRANSNET_URL, timeout=60) as r, part.open("wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                digest = hashlib.sha256(part.read_bytes()).hexdigest()
                if digest != TRANSNET_SHA256:
                    part.unlink()
                    raise RuntimeError(f"TransNetV2 weights failed their checksum (got {digest}).")
                part.rename(path)
            model = TransNetV2()
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            _transnet = model.eval()
        return _transnet


def run_transnet(ctx: Context) -> dict:
    import torch

    model = _transnet_model()
    frame_bytes = 27 * 48 * 3
    proc = ffmpeg_reader(["-i", ctx.proxy, "-map", "0:v:0", "-vf", "scale=48:27",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    reporter = Reporter(ctx)

    def read(k: int) -> np.ndarray:
        buf = bytearray(k * frame_bytes)
        view, got = memoryview(buf), 0
        while got < len(buf):
            n = proc.stdout.readinto(view[got:])
            if not n:
                break
            got += n
        got -= got % frame_bytes
        return np.frombuffer(bytes(buf[:got]), np.uint8).reshape(-1, 27, 48, 3)

    # Same windows as the authors' predict_video: the video is padded with 25 copies of its
    # first frame and 25-74 of its last, then 100-frame windows step by 50 and each keeps the
    # predictions for its middle 50 frames. Here the windows are fed as frames arrive.
    preds, seen, done = [], 0, False
    try:
        buf = read(1)
        if not len(buf):
            raise RuntimeError("The proxy has no video frames.")
        seen = 1
        buf = np.concatenate([np.repeat(buf[:1], 25, 0), buf])
        while True:
            want = 50 * TRANSNET_BATCH + 50 - len(buf)
            if want > 0 and not done:
                new = read(want)
                seen += len(new)
                done = len(new) < want
                buf = np.concatenate([buf, new])
                reporter.frame(seen)
            if done:
                pad = 25 + 50 - (seen % 50 or 50)
                buf = np.concatenate([buf, np.repeat(buf[-1:], pad, 0)])
            windows = (len(buf) - 50) // 50
            if not done:
                windows = min(windows, TRANSNET_BATCH)
            if windows > 0:
                batch = np.stack([buf[50 * j: 50 * j + 100] for j in range(windows)])
                with torch.inference_mode():
                    single, _ = model(torch.from_numpy(batch))
                preds.append(torch.sigmoid(single[:, 25:75, 0]).reshape(-1).numpy())
                buf = buf[50 * windows:]
            if done:
                break
    except BaseException:
        finish(proc, cancelled=True)
        raise
    finish(proc)

    # The probability that each frame is a transition
    return {"scores": np.concatenate(preds)[:seen].astype(np.float32), "frames": seen}


# ---------- PySceneDetect ----------

def pyscenedetect(make, metric: Callable[[dict], Optional[float]], below: bool = False):
    """Runner for a PySceneDetect detector. make(params, min_frames) builds the detector;
    metric picks the per-frame score it reports."""

    def run(ctx: Context) -> dict:
        from scenedetect import SceneManager, StatsManager, open_video

        curve = Curve(ctx.frames, below=below)
        reporter = Reporter(ctx)
        cancelled = []

        class Feed(StatsManager):
            # Detectors report every frame's metrics here. Rather than storing them (hundreds of MB
            # for a long video), take the score for the curve and use the call as a progress tick.
            def metrics_exist(self, frame_number, metric_keys):
                return False

            def set_metrics(self, frame_number, metric_kv):
                n = frame_number if isinstance(frame_number, int) else frame_number.frame_num
                value = metric(metric_kv)
                if value is not None:
                    curve.add(n, value)
                try:
                    reporter.frame(n)
                except Cancelled:
                    cancelled.append(True)
                    manager.stop()

        video = open_video(ctx.proxy, backend="opencv")
        manager = SceneManager(Feed())
        manager.add_detector(make(ctx.params, ctx.min_frames()))
        manager.detect_scenes(video)
        if cancelled:
            raise Cancelled()
        frames = video.frame_number
        cuts = [start.frame_num for start, _ in manager.get_scene_list()[1:]]
        threshold = ctx.params.get("threshold", ctx.params.get("adaptive_threshold"))
        return {"cuts": cuts, "frames": frames, "curve": curve.result(threshold)}

    return run


def _content(params, min_frames):
    from scenedetect import ContentDetector

    return ContentDetector(threshold=params["threshold"], min_scene_len=min_frames)


def _adaptive(params, min_frames):
    from scenedetect import AdaptiveDetector

    return AdaptiveDetector(
        adaptive_threshold=params["adaptive_threshold"], min_content_val=params["min_content_val"],
        min_scene_len=min_frames,
    )


def _threshold(params, min_frames):
    from scenedetect import ThresholdDetector

    return ThresholdDetector(threshold=params["threshold"], fade_bias=params["fade_bias"], min_scene_len=min_frames)


def _adaptive_ratio(kv: dict) -> Optional[float]:
    # The key carries the window width, e.g. "adaptive_ratio (w=2)"
    return next((v for k, v in kv.items() if k.startswith("adaptive_ratio")), None)


# ---------- FFmpeg scene filter ----------

def run_ffmpeg_scene(ctx: Context) -> dict:
    proc = ffmpeg_reader(["-i", ctx.proxy, "-map", "0:v:0", "-an",
                          "-vf", "select='gte(scene,0)',metadata=print:file=-", "-f", "null", "-"])
    reporter = Reporter(ctx)
    scores = np.zeros(ctx.frames + 1, np.float32)  # grown if the proxy runs longer
    frame, frames = 0, 0
    try:
        for raw in proc.stdout:
            line = raw.decode(errors="replace")
            if line.startswith("frame:"):
                # frame:12   pts:12800   pts_time:0.5
                for part in line.split():
                    if part.startswith("pts_time:"):
                        frame = round(float(part[9:]) * ctx.fps)
                frames = max(frames, frame + 1)
                reporter.frame(frame)
            elif line.startswith("lavfi.scene_score="):
                if frame >= len(scores):
                    scores = np.concatenate([scores, np.zeros(max(frame + 1 - len(scores), 1000), np.float32)])
                scores[frame] = float(line.split("=", 1)[1])
    except BaseException:
        finish(proc, cancelled=True)
        raise
    finish(proc)
    return {"scores": scores[:frames], "frames": frames}


# ---------- registry ----------

def min_scene(default: float = 0.6) -> Param:
    return Param("min_scene", "Min scene (s)", default, 0.04, 60, 0.1, "Cuts closer together than this are merged.")


DETECTORS = {
    d.id: d
    for d in [
        Detector(
            id="transnetv2", label="TransNetV2",
            description="Neural network trained on shot boundaries. Best on hard cuts, and catches most dissolves and fades.",
            params=[Param("threshold", "Threshold", 0.5, 0.05, 0.95, 0.05, "Transition probability needed for a cut."),
                    min_scene(0.5)],
            run=run_transnet, rescore=cuts_at_peaks,
        ),
        Detector(
            id="pyscenedetect-content", label="PySceneDetect · content",
            description="Change in colour and brightness between frames. Fast hard-cut detector; lower the threshold to find more cuts.",
            params=[Param("threshold", "Threshold", 27, 1, 100, 1, "Content change needed for a cut."), min_scene()],
            run=pyscenedetect(_content, lambda kv: kv.get("content_val")),
        ),
        Detector(
            id="pyscenedetect-adaptive", label="PySceneDetect · adaptive",
            description="Content change relative to neighbouring frames. Fewer false cuts in fast motion and camera pans.",
            params=[Param("adaptive_threshold", "Threshold", 3, 1, 20, 0.5, "How far a frame's change must exceed its neighbours'."),
                    Param("min_content_val", "Min change", 15, 0, 100, 1, "Content change a cut needs regardless of neighbours."),
                    min_scene()],
            run=pyscenedetect(_adaptive, _adaptive_ratio), curve_param="adaptive_threshold",
        ),
        Detector(
            id="pyscenedetect-threshold", label="PySceneDetect · fades",
            description="Fades to and from black: a scene ends where average brightness drops below the threshold.",
            params=[Param("threshold", "Threshold", 12, 1, 255, 1, "Average pixel brightness (0-255) that counts as black."),
                    Param("fade_bias", "Fade bias", 0, -1, 1, 0.1, "Where in the fade to cut: -1 at the start of black, +1 at the end."),
                    min_scene()],
            run=pyscenedetect(_threshold, lambda kv: kv.get("average_rgb"), below=True),
        ),
        Detector(
            id="ffmpeg-scene", label="FFmpeg scene filter",
            description="FFmpeg's built-in scene score. Quick rough split; some cuts score low, so try 0.2 if it misses any.",
            params=[Param("threshold", "Threshold", 0.3, 0.02, 1, 0.02, "Scene score (0-1) needed for a cut."), min_scene()],
            run=run_ffmpeg_scene, rescore=cuts_above,
        ),
        Detector(
            id="autoshot", label="AutoShot", description="Top model on the ClipShots and SHOT benchmarks.",
            params=[], run=None,
            note="Not available yet: its weights are only published on Baidu Netdisk.",
        ),
    ]
}

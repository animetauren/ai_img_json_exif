"""Inference service for non-LLM vision models (SigLIP 2, timm classifiers, RT-DETR).

Models load lazily on first use and are kept in a small LRU so the container
doesn't need RAM for every model at once. Weights are cached under VISION_CACHE.
"""

import base64
import io
import os
import threading
import time
from collections import OrderedDict
from typing import Callable, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

MAX_LOADED = int(os.environ.get("VISION_MAX_LOADED", "2"))
torch.set_num_threads(int(os.environ.get("VISION_THREADS", os.cpu_count() or 4)))

DEFAULT_LABELS = [
    "landscape", "portrait", "group of people", "selfie", "food", "pet", "wildlife",
    "document", "screenshot", "receipt", "architecture", "city street", "vehicle",
    "beach", "mountains", "snow", "night sky", "sunset", "flowers", "sports",
    "concert", "art", "meme", "whiteboard", "car interior",
]
DEFAULT_TEMPLATE = "This is a photo of {}."


# ---------- loaders / runners ----------

def load_timm(name: str):
    import timm
    from timm.data import ImageNetInfo, infer_imagenet_subset

    model = timm.create_model(name, pretrained=True).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    info = ImageNetInfo(infer_imagenet_subset(model) or "imagenet-1k")
    return model, transform, info


def run_timm(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    model, transform, info = bundle
    with torch.inference_mode():
        probs = model(transform(img).unsqueeze(0)).softmax(-1)[0]
    scores, idxs = probs.topk(req.top_k)
    preds = [
        {"label": info.index_to_description(int(i)), "score": float(s)}
        for s, i in zip(scores, idxs)
    ]
    return {"predictions": preds, "threshold": threshold}


def load_siglip(name: str):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(name).eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def run_siglip(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    model, processor = bundle
    labels = [l.strip() for l in (req.labels or DEFAULT_LABELS) if l.strip()]
    if not labels:
        raise HTTPException(400, "Zero-shot model needs at least one label.")
    template = req.label_template or DEFAULT_TEMPLATE
    texts = [template.format(l) if "{}" in template else f"{template} {l}" for l in labels]
    inputs = processor(text=texts, images=img, padding="max_length", max_length=64, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
    # SigLIP scores each label independently (sigmoid), so scores don't sum to 1
    probs = torch.sigmoid(logits)
    order = probs.argsort(descending=True)[: req.top_k]
    preds = [{"label": labels[int(i)], "score": float(probs[i])} for i in order]
    return {"predictions": preds, "threshold": threshold}


def run_siglip_nsfw(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    model, processor = bundle
    labels = ["nsfw", "safe"]
    template = req.label_template or "a photo that is {}"
    texts = [template.format(l) if "{}" in template else f"{template} {l}" for l in labels]
    inputs = processor(text=texts, images=img, padding="max_length", max_length=64, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits_per_image[0]
    probs = torch.sigmoid(logits)
    nsfw_p = float(probs[0]); safe_p = float(probs[1])
    preds = [{"label": labels[i], "score": float(probs[i])} for i in range(len(labels))]
    order = probs.argsort(descending=True)
    preds = [preds[int(i)] for i in order]
    return {"nsfw": nsfw_p, "safe": safe_p, "confidence": max(nsfw_p, safe_p),
            "predictions": preds, "threshold": threshold}



# The checkpoint uses PASCAL VOC names for a few COCO classes; report the usual COCO names,
# which are also what workflows and saved results used with YOLO
COCO_NAMES = {
    "motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch",
    "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv",
}


def load_detr(name: str):
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model = AutoModelForObjectDetection.from_pretrained(name).eval()
    processor = AutoImageProcessor.from_pretrained(name)
    names = {int(i): COCO_NAMES.get(n, n) for i, n in model.config.id2label.items()}
    return model, processor, names


def run_detr(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    model, processor, names = bundle
    wanted_ids = None
    if req.classes:
        wanted = {c.strip().lower() for c in req.classes if c.strip()}
        wanted_ids = {i for i, name in names.items() if name.lower() in wanted}
        if not wanted_ids:
            raise HTTPException(400, f"No detector classes match {sorted(wanted)}. Available: {', '.join(names.values())}")
    # Always 640x640, the size the model was trained at: larger inputs lower its scores and
    # lose objects rather than finding small ones
    inputs = processor(images=img, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    res = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=[(img.height, img.width)])[0]
    w, h = img.size
    detections = [
        {"label": names[c], "score": s, "box": [round(min(max(v, 0), lim), 1) for v, lim in zip(box, (w, h, w, h))]}
        for s, c, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist())
        if wanted_ids is None or c in wanted_ids
    ]
    # Collapse detections into one prediction per label (max score, with count)
    by_label: "OrderedDict[str, dict]" = OrderedDict()
    for d in sorted(detections, key=lambda d: -d["score"]):
        p = by_label.setdefault(d["label"], {"label": d["label"], "score": d["score"], "count": 0})
        p["count"] += 1
    preds = list(by_label.values())[: req.top_k]
    counts = {label: p["count"] for label, p in by_label.items()}
    return {"predictions": preds, "detections": detections, "counts": counts, "threshold": threshold}


def load_yolo(name: str):
    from ultralytics import YOLO

    model = YOLO(name)
    names = getattr(model, "names", {}) or {}
    names = {int(k): str(v) for k, v in names.items()}
    return model, names


def run_yolo(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    model, names = bundle
    wanted_ids = None
    if req.classes:
        wanted = {c.strip().lower() for c in req.classes if c.strip()}
        wanted_ids = {i for i, name in names.items() if name.lower() in wanted}
        if not wanted_ids:
            raise HTTPException(400, f"No detector classes match {sorted(wanted)}. Available: {', '.join(names.values())}")
    results = model.predict(source=img, conf=threshold, verbose=False)
    res = results[0] if results else None
    w, h = img.size
    detections = []
    if res is not None and getattr(res, "boxes", None) is not None:
        for box in res.boxes:
            cls = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
            score = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
            if score < threshold:
                continue
            xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy, "tolist") else list(box.xyxy[0])
            if wanted_ids is not None and cls not in wanted_ids:
                continue
            detections.append({
                "label": names.get(cls, str(cls)),
                "score": score,
                "box": [round(min(max(v, 0), lim), 1) for v, lim in zip(xyxy, (w, h, w, h))],
            })
    by_label: "OrderedDict[str, dict]" = OrderedDict()
    for d in sorted(detections, key=lambda d: -d["score"]):
        p = by_label.setdefault(d["label"], {"label": d["label"], "score": d["score"], "count": 0})
        p["count"] += 1
        if d["score"] > p["score"]:
            p["score"] = d["score"]
    preds = list(by_label.values())[: req.top_k]
    counts = {label: p["count"] for label, p in by_label.items()}
    return {"predictions": preds, "detections": detections, "counts": counts, "threshold": threshold}



def load_florence(name: str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    model = AutoModelForCausalLM.from_pretrained(name).eval()
    processor = AutoProcessor.from_pretrained(name)
    return model, processor


def run_florence(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    model, processor = bundle
    task = req.label_template
    if not (isinstance(task, str) and task.startswith("<") and task.endswith(">")):
        task = "<MORE_DETAILED_CAPTION>"
    inputs = processor(text=task, images=img, return_tensors="pt")
    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs.get("input_ids"),
            pixel_values=inputs.get("pixel_values"),
            max_new_tokens=256,
            num_beams=1,
            do_sample=False,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    caption = generated_text
    try:
        parsed = processor.post_process_generation(
            generated_text, task=task, image_size=(img.width, img.height))
        if isinstance(parsed, dict) and task in parsed:
            caption = parsed[task]
        elif isinstance(parsed, str):
            caption = parsed
    except Exception:  # noqa: BLE001 - fall back to raw decoded text
        pass
    if isinstance(caption, (list, tuple, dict)):
        caption = str(caption)
    caption = caption.strip()
    out = {"threshold": threshold, "predictions": []}
    if "OCR" in task.upper():
        out["text"] = caption
        out["segments"] = [{"text": caption, "box": None, "confidence": 1.0}]
    else:
        out["caption"] = caption
        out["description"] = caption
    return out


def load_paddleocr(name: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=True, lang="en")


def run_paddleocr(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    import numpy as np

    ocr = bundle
    result = ocr.ocr(np.array(img), cls=True)
    lines = result[0] if result and result[0] else []
    segments = []
    confidences = []
    for line in lines:
        box, (text, score) = line[0], line[1]
        score = float(score)
        if score < threshold:
            continue
        segments.append({"text": text, "box": box, "confidence": score})
        confidences.append(score)
    avg = float(sum(confidences) / len(confidences)) if confidences else 0.0
    full_text = " ".join(s["text"] for s in segments)
    return {
        "text": full_text,
        "segments": segments,
        "confidence": avg,
        "predictions": [{"label": s["text"], "score": s["confidence"]} for s in segments],
        "threshold": threshold,
    }


def load_depth_anything(name: str):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    import torch

    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModelForDepthEstimation.from_pretrained(name).eval()
    return model, processor


def run_depth_anything(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    import numpy as np
    from transformers import AutoImageProcessor

    model, processor = bundle
    inputs = processor(images=img, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    post = processor.post_process_depth_estimation(outputs, target_sizes=[(img.height, img.width)])[0]
    depth = post["predicted_depth"]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    else:
        depth = np.asarray(depth)
    dmin = float(np.min(depth)); dmax = float(np.max(depth)); dmean = float(np.mean(depth))
    denom = (dmax - dmin) if (dmax - dmin) > 1e-8 else 1.0
    normalized = (depth - dmin) / denom
    return {
        "depth_min": dmin,
        "depth_max": dmax,
        "depth_mean": dmean,
        "depth_range": [0.0, 1.0],
        "depth_normalized_mean": float(np.mean(normalized)),
        "predictions": [],
        "threshold": threshold,
    }


def load_faces(name: str):
    import insightface
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def run_faces(bundle, img: Image.Image, req: "ClassifyRequest", threshold: float) -> dict:
    import numpy as np

    app = bundle
    faces = app.get(np.array(img)[:, :, ::-1])
    out_faces = []
    for f in faces:
        score = float(getattr(f, "det_score", 1.0))
        if score < threshold:
            continue
        bbox = [float(x) for x in getattr(f, "bbox", [])]
        kps = None
        if getattr(f, "kps", None) is not None:
            kps = [[float(x), float(y)] for x, y in f.kps]
        out_faces.append({
            "bbox": bbox,
            "score": score,
            "landmarks": kps,
        })
    return {
        "count": len(out_faces),
        "faces": out_faces,
        "predictions": [],
        "threshold": threshold,
    }


class Spec(BaseModel):
    id: str
    kind: str  # "classify" | "zero-shot" | "detect"
    source: str
    description: str
    labels: str
    default_threshold: float
    loader: Callable
    runner: Callable

    model_config = {"arbitrary_types_allowed": True}


SPECS = {
    s.id: s
    for s in [
        Spec(
            id="siglip2-base", kind="zero-shot", source="google/siglip2-base-patch16-224",
            description="SigLIP 2 base — zero-shot against your own label list",
            labels="custom", default_threshold=0.01, loader=load_siglip, runner=run_siglip,
        ),
        Spec(
            id="efficientnetv2-s", kind="classify", source="tf_efficientnetv2_s.in21k_ft_in1k",
            description="EfficientNetV2-S — ImageNet-1k classifier",
            labels="imagenet-1k", default_threshold=0.05, loader=load_timm, runner=run_timm,
        ),
        Spec(
            id="resnet50", kind="classify", source="resnet50.a1_in1k",
            description="ResNet-50 — ImageNet-1k classifier",
            labels="imagenet-1k", default_threshold=0.05, loader=load_timm, runner=run_timm,
        ),
        Spec(
            id="rtdetr-v2-s", kind="detect", source="PekingU/rtdetr_v2_r18vd",
            description="RT-DETRv2-S — COCO object detection (80 classes)",
            labels="coco-80", default_threshold=0.5, loader=load_detr, runner=run_detr,
        ),
        Spec(
            id="florence2-base", kind="caption", source="microsoft/Florence-2-base",
            description="Florence-2 base — detailed image captioning",
            labels="caption", default_threshold=0.0, loader=load_florence, runner=run_florence,
        ),
        Spec(
            id="florence2-ocr", kind="ocr", source="microsoft/Florence-2-base",
            description="Florence-2 base — OCR",
            labels="text", default_threshold=0.0, loader=load_florence, runner=run_florence,
        ),
        Spec(
            id="depth-anything-v2-small", kind="depth", source="depth-anything/Depth-Anything-V2-Small-hf",
            description="Depth Anything V2 Small — monocular depth estimation",
            labels="depth", default_threshold=0.0, loader=load_depth_anything, runner=run_depth_anything,
        ),
        Spec(
            id="insightface-buffalo-l", kind="face", source="buffalo_l",
            description="InsightFace buffalo_l — face detection and landmarks (non-commercial research weights)",
            labels="face", default_threshold=0.5, loader=lambda n: load_faces("buffalo_l"), runner=run_faces,
        ),
    ]
}
# Model ids that were replaced, so saved workflows and settings files keep working
ALIASES = {"yolov8s": "rtdetr-v2-s"}


# ---------- model cache ----------

_loaded: "OrderedDict[str, object]" = OrderedDict()
_lock = threading.Lock()


def get_model(spec: Spec):
    """Return (bundle, load_ms). Caller must hold _lock."""
    if spec.id in _loaded:
        _loaded.move_to_end(spec.id)
        return _loaded[spec.id], 0.0
    while len(_loaded) >= MAX_LOADED:
        _loaded.popitem(last=False)
    t = time.perf_counter()
    bundle = spec.loader(spec.source)
    _loaded[spec.id] = bundle
    return bundle, (time.perf_counter() - t) * 1000


# ---------- API ----------

app = FastAPI(title="Vision models")


class ClassifyRequest(BaseModel):
    model: str
    image_data_url: str
    labels: Optional[List[str]] = None
    label_template: Optional[str] = None
    top_k: int = 5
    threshold: Optional[float] = None
    classes: Optional[List[str]] = None  # detection only: keep just these labels, e.g. ["person"]


def decode_image(data_url: str) -> Image.Image:
    try:
        b64 = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not decode image: {e}")


def short_label(label: str) -> str:
    # ImageNet descriptions look like "tabby, tabby cat" — keep the first name
    return label.split(",")[0].strip()


@app.get("/health")
def health():
    return {"status": "ok", "loaded": list(_loaded)}


@app.get("/models")
def models():
    return {
        "models": [
            {
                "id": s.id, "kind": s.kind, "source": s.source, "description": s.description,
                "labels": s.labels, "default_threshold": s.default_threshold,
                "loaded": s.id in _loaded,
            }
            for s in SPECS.values()
        ],
        "default_labels": DEFAULT_LABELS,
        "default_template": DEFAULT_TEMPLATE,
    }


@app.post("/classify")
def classify(req: ClassifyRequest):
    spec = SPECS.get(ALIASES.get(req.model, req.model))
    if not spec:
        raise HTTPException(404, f"Unknown model '{req.model}'. Available: {', '.join(SPECS)}")
    req.top_k = max(1, min(req.top_k, 50))
    threshold = spec.default_threshold if req.threshold is None else req.threshold
    img = decode_image(req.image_data_url)

    with _lock:
        bundle, load_ms = get_model(spec)
        t = time.perf_counter()
        out = spec.runner(bundle, img, req, threshold)
        infer_ms = (time.perf_counter() - t) * 1000

    preds = out["predictions"]
    tags = [short_label(p["label"]) for p in preds if p["score"] >= threshold]
    top = preds[0] if preds else None
    return {
        # Same top-level shape the default LLM prompt asks for, so results compare directly
        "category": short_label(top["label"]) if top else None,
        "tags": tags,
        "confidence": round(top["score"], 4) if top else 0.0,
        "model": spec.id,
        "kind": spec.kind,
        **out,
        "image_size": list(img.size),
        "timing_ms": {"load": round(load_ms, 1), "inference": round(infer_ms, 1)},
    }

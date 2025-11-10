import os
import sys
import re
import json
import math
import hashlib
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

# --- YOLO (Ultralytics) ---
from ultralytics import YOLO

# --- Optional Keras classifier (Teachable Machine export) ---
USE_CLASSIFIER = True  # set False if you don't have the Keras model yet

# Path to your Teachable Machine Keras model (SavedModel dir or .h5)
# Example for TM image project: exported_model/ (contains saved_model.pb)  OR  model.h5
KERAS_MODEL_PATH = "tm_ambulance_model"  # change to your path
CLASS_NAMES = ["Ambulance", "NotAmbulance"]  # change to match your model's label order
AMBULANCE_POSITIVE_CLASS_NAME = "Ambulance"
CLASS_THRESHOLD = 0.60  # keep crop if P(ambulance) >= 0.60

# --- I/O paths ---
INPUT_DIR = "ambulance_images"
OUTPUT_DIR = "processed_ambulance_images"

# --- YOLO model for detection ---
# If you have your own custom YOLO ambulance detector, put the .pt path here.
# Otherwise we use a general model and filter to vehicle classes.
YOLO_MODEL_PATH = "yolov8n.pt"

# COCO indices for vehicles we’ll use as proposals:
# 2=car, 3=motorbike, 5=bus, 7=truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# Output image size
OUT_SIZE = 640

# How much to expand the bounding box before cropping (as a fraction of box size)
BBOX_PAD = 0.10

# Max crops per source image (helps avoid exploding your dataset with street scenes)
MAX_CROPS_PER_IMAGE = 5


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    # Keep it Windows-safe and short-ish
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Limit length (Windows path limit safety)
    return name[:180]


def sha1_of_array(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest()


def load_keras_model():
    if not USE_CLASSIFIER:
        return None, None
    try:
        # Lazy import to avoid TF when not needed
        import tensorflow as tf
        model = tf.keras.models.load_model(KERAS_MODEL_PATH)
        # Build mapping from index -> class name
        idx_to_name = {i: n for i, n in enumerate(CLASS_NAMES)}
        return model, idx_to_name
    except Exception as e:
        print(f"[WARN] Could not load Keras model at '{KERAS_MODEL_PATH}': {e}")
        print("[INFO] Proceeding with USE_CLASSIFIER=False")
        return None, None


def keras_predict_is_ambulance(model, idx_to_name, pil_img: Image.Image) -> Tuple[bool, float, str]:
    """
    Returns (is_ambulance, probability, label_name)
    Assumes Teachable Machine default: resize to 224x224, scale [0,1]
    Adjust if your TM model requires different preprocessing.
    """
    import tensorflow as tf
    img = pil_img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    label = idx_to_name.get(idx, str(idx))
    prob = float(preds[idx])
    is_amb = (label == AMBULANCE_POSITIVE_CLASS_NAME) and (prob >= CLASS_THRESHOLD)
    return is_amb, prob, label


def pad_to_square(pil_img: Image.Image, size: int) -> Image.Image:
    w, h = pil_img.size
    side = max(w, h)
    # Create square canvas with black background (change to (255,255,255) if you prefer white)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.LANCZOS)


def crop_with_padding(img: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    """
    box = (x1, y1, x2, y2) in image coords
    Expands by BBOX_PAD and clamps to image bounds.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(BBOX_PAD * bw)
    pad_y = int(BBOX_PAD * bh)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)

    crop = img[ny1:ny2, nx1:nx2]
    return crop


def detect_vehicle_boxes(yolo_model, img_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
    """
    Returns list of boxes [(x1,y1,x2,y2,conf,cls_id)]
    Filters to VEHICLE_CLASS_IDS.
    """
    # Ultralytics expects RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model.predict(source=img_rgb, verbose=False)
    out = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cls_id = int(b.cls)
            if cls_id not in VEHICLE_CLASS_IDS:
                continue
            conf = float(b.conf)
            x1, y1, x2, y2 = map(lambda v: int(v.item()), b.xyxy[0])
            # sanity clamp
            h, w = img_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                out.append((x1, y1, x2, y2, conf, cls_id))

    # sort by area desc (largest first)
    out.sort(key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]), reverse=True)
    return out[:MAX_CROPS_PER_IMAGE]


def process_image(path: Path, yolo_model, keras_model, idx_to_name) -> int:
    """
    Reads an image, detects vehicles, optionally filters crops by Keras ambulance classifier,
    pads to square, resizes, saves.
    Returns number of crops saved.
    """
    if not path.exists():
        print(f"[MISS] File not found (skipping): {path}")
        return 0

    # cv2.imread sometimes fails on weird paths; use np.fromfile + imdecode (more robust on Windows)
    try:
        file_bytes = np.fromfile(str(path), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[ERR] Failed to read {path}: {e}")
        return 0

    if img_bgr is None:
        print(f"[ERR] OpenCV could not decode {path}")
        return 0

    boxes = detect_vehicle_boxes(yolo_model, img_bgr)
    if not boxes:
        print(f"[INFO] No vehicle boxes: {path.name}")
        return 0

    saved = 0
    base = path.stem
    for i, (x1, y1, x2, y2, conf, cls_id) in enumerate(boxes, start=1):
        crop_bgr = crop_with_padding(img_bgr, (x1, y1, x2, y2))
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_crop = Image.fromarray(crop_rgb)

        keep = True
        prob = None
        label = None
        if USE_CLASSIFIER and keras_model is not None:
            keep, prob, label = keras_predict_is_ambulance(keras_model, idx_to_name, pil_crop)

        if not keep:
            # Skip non-ambulance crops
            continue

        squared = pad_to_square(pil_crop, OUT_SIZE)
        # File naming: sourceName__i__sha1.jpg
        sha = sha1_of_array(np.array(squared))
        out_name = sanitize_filename(f"{base}__{i}__{sha}.jpg")
        out_path = Path(OUTPUT_DIR) / out_name

        try:
            squared.save(out_path)
            saved += 1
            if USE_CLASSIFIER and prob is not None:
                print(f"[OK] Saved {out_name}  (label={label}, p={prob:.2f})")
            else:
                print(f"[OK] Saved {out_name}")
        except Exception as e:
            print(f"[ERR] Save failed for {out_name}: {e}")

    return saved


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def main():
    # Ensure folders
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    # Load YOLO
    print(f"[INIT] Loading YOLO model: {YOLO_MODEL_PATH}")
    yolo = YOLO(YOLO_MODEL_PATH)

    # Load (optional) Keras classifier
    keras_model, idx_to_name = load_keras_model()
    if USE_CLASSIFIER and keras_model is None:
        print("[WARN] Keras model not available; continuing without classifier")

    images = list_images(Path(INPUT_DIR))
    if not images:
        print(f"[INFO] No images found in '{INPUT_DIR}'. Put your images there.")
        return

    print(f"[INFO] Found {len(images)} images. Processing...")
    total_crops = 0
    for p in images:
        try:
            total_crops += process_image(p, yolo, keras_model, idx_to_name)
        except Exception as e:
            print(f"[ERR] Unexpected error on {p.name}: {e}")

    print(f"[DONE] Saved {total_crops} crops to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    # If you run from PyCharm, the working dir is your project root—good.
    # If running from elsewhere, ensure INPUT_DIR paths are correct or pass absolute paths.
    main()

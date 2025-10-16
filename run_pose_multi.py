import os
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.framework.formats import landmark_pb2  # optional proto types

# --------- Model URLs (official MediaPipe Tasks) ----------
MODEL_URLS = {
    "lite":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full":  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

# ----------------- CLI -----------------
def get_args():
    p = argparse.ArgumentParser("Multi-person Pose Tracking (MediaPipe Tasks)")
    p.add_argument("--webcam", type=int, default=None, help="Webcam index (e.g., 0). If set, --video ignored.")
    p.add_argument("--video", type=str, default=None, help="Path to a video if not using webcam.")
    p.add_argument("--model_path", type=str, default="models/pose_landmarker_full.task",
                   help="Path to MediaPipe Pose Landmarker .task file (auto-downloaded if missing unless --no_download).")

    p.add_argument("--out_dir", type=str, default="outputs", help="Output folder.")
    p.add_argument("--show", action="store_true", help="Show live window.")
    p.add_argument("--record", action="store_true", help="Save overlay video.")
    p.add_argument("--csv", action="store_true", help="Write CSV per-frame landmarks for all persons.")
    p.add_argument("--max_width", type=int, default=0, help="If >0, downscale frames to this width.")

    p.add_argument("--max_poses", type=int, default=5, help="Max people to detect.")
    p.add_argument("--min_pose_score", type=float, default=0.5,
                   help="Min avg visibility (0..1) to keep a detected person.")
    p.add_argument("--model_complexity", type=str, default="full", choices=["lite","full","heavy"],
                   help="Which model to download when missing (lite/full/heavy).")

    p.add_argument("--target_fps", type=float, default=0.0, help="If >0 and recording, resample FPS.")
    p.add_argument("--fourcc", type=str, default="mp4v", help="Overlay writer codec.")
    p.add_argument("--skip_frames", type=int, default=0, help="Process every (skip_frames+1)th frame.")
    p.add_argument("--window_name", type=str, default="Multi-Person Pose")
    p.add_argument("--no_download", action="store_true",
                   help="Do not auto-download the model; error out if missing.")
    return p.parse_args()

# ----------------- Utils -----------------
def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)

def maybe_resize(img: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0:
        return img
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    s = max_w / float(w)
    return cv2.resize(img, (max_w, int(round(h * s))), interpolation=cv2.INTER_AREA)

def _safe_fps(raw_fps: float, default: float = 25.0) -> float:
    return float(raw_fps) if raw_fps and np.isfinite(raw_fps) and raw_fps > 0 else default

def open_capture(args):
    """Return (cap, fps, width, height, stream_name)"""
    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam {args.webcam}")
        fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        return cap, fps, w, h, f"webcam_{args.webcam}"
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = _safe_fps(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap, fps, w, h, Path(args.video).stem
    raise RuntimeError("No source. Use --webcam <idx> or --video <path>.")

def make_writer(path: str, w: int, h: int, fps: float, fourcc: str):
    fps = fps if fps and fps > 0 else 25.0
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Failed to create writer at {path}")
    return vw

# --------- Model handling ----------
def ensure_model_exists(model_path: str, preferred_variant: str, allow_download: bool = True) -> str:
    """
    Ensures the model file exists. If not and downloads are allowed, fetch from MediaPipe CDN.
    Returns absolute model path.
    """
    apath = str(Path(model_path).expanduser().absolute())
    if Path(apath).exists() and Path(apath).is_file() and Path(apath).stat().st_size > 0:
        print(f"[MODEL] Using existing: {apath} ({Path(apath).stat().st_size/1e6:.2f} MB)")
        return apath

    if not allow_download:
        raise FileNotFoundError(
            f"Model file not found: {apath}\n"
            f"Download one of:\n"
            f"  lite : {MODEL_URLS['lite']}\n"
            f"  full : {MODEL_URLS['full']}\n"
            f"  heavy: {MODEL_URLS['heavy']}\n"
            f"and place it at that path (or pass --model_path)."
        )

    # Decide URL from preferred variant
    url = MODEL_URLS.get(preferred_variant, MODEL_URLS["full"])
    ensure_dir(str(Path(apath).parent))
    print(f"[MODEL] Downloading {preferred_variant} model to {apath} ...")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, apath)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download model from {url}.\n"
            f"Error: {e}\n"
            f"You can download manually and place it at: {apath}"
        )

    size = Path(apath).stat().st_size if Path(apath).exists() else 0
    if size <= 0:
        raise RuntimeError(
            f"Downloaded model seems empty at {apath}. "
            f"Please download manually from:\n{url}"
        )
    print(f"[MODEL] Downloaded OK: {apath} ({size/1e6:.2f} MB)")
    return apath

# ----------------- Simple ID Tracker -----------------
class CentroidTracker:
    def __init__(self, max_dist: float = 120.0, max_age: int = 30):
        self.max_dist = max_dist
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, Dict] = {}  # id -> {cx, cy, age, last_frame}

    def update(self, detections: List[Tuple[float, float]], frame_i: int) -> List[int]:
        assigned_ids = [-1] * len(detections)
        used_ids = set()

        # Greedy nearest neighbor
        for di, (cx, cy) in enumerate(detections):
            best_id = -1
            best_d2 = float("inf")
            for tid, t in self.tracks.items():
                if tid in used_ids:
                    continue
                dx = cx - t["cx"]
                dy = cy - t["cy"]
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    best_id = tid
            if best_id != -1 and best_d2 <= self.max_dist * self.max_dist:
                self.tracks[best_id]["cx"] = cx
                self.tracks[best_id]["cy"] = cy
                self.tracks[best_id]["age"] = 0
                self.tracks[best_id]["last_frame"] = frame_i
                assigned_ids[di] = best_id
                used_ids.add(best_id)

        # New tracks
        for di, (cx, cy) in enumerate(detections):
            if assigned_ids[di] == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"cx": cx, "cy": cy, "age": 0, "last_frame": frame_i}
                assigned_ids[di] = tid

        # Age + prune
        to_del = []
        for tid, t in self.tracks.items():
            if t["last_frame"] != frame_i:
                t["age"] += 1
            if t["age"] > self.max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]

        return assigned_ids

# ----------------- Drawing helpers -----------------
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in POSE_CONNECTIONS:
        xa, ya = pts[a]; xb, yb = pts[b]
        cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)
    for (x, y) in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
    return frame

def torso_centroid(landmarks, img_w, img_h) -> Tuple[float, float]:
    idxs = [11, 12, 23, 24]
    xs, ys = [], []
    for i in idxs:
        lm = landmarks[i]
        if np.isfinite(lm.x) and np.isfinite(lm.y):
            xs.append(lm.x * img_w); ys.append(lm.y * img_h)
    if len(xs) >= 2:
        return float(np.mean(xs)), float(np.mean(ys))
    xs = [lm.x * img_w for lm in landmarks if np.isfinite(lm.x)]
    ys = [lm.y * img_h for lm in landmarks if np.isfinite(lm.y)]
    return (float(np.mean(xs)), float(np.mean(ys))) if xs and ys else (img_w * 0.5, img_h * 0.5)

def avg_visibility(landmarks) -> float:
    vals = []
    for lm in landmarks:
        v = getattr(lm, "visibility", None)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else 1.0

# ----------------- Pose Runner -----------------
def create_landmarker(model_path: str, num_poses: int) -> mp_vision.PoseLandmarker:
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=num_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO
    )
    return mp_vision.PoseLandmarker.create_from_options(options)

def process(args):
    # Ensure model file is available (download if missing)
    model_path_abs = ensure_model_exists(args.model_path, args.model_complexity, allow_download=not args.no_download)
    print(f"[MODEL] Loading from: {model_path_abs}")

    ensure_dir(args.out_dir)
    cap, src_fps, src_w, src_h, stream_name = open_capture(args)

    # Prime
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Empty stream.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = maybe_resize(frame, args.max_width)
    out_h, out_w = frame.shape[:2]

    # writer
    out_fps = args.target_fps if (args.target_fps > 0 and args.record) else src_fps
    writer = None
    overlay_path = str(Path(args.out_dir) / f"{stream_name}_multi_pose.mp4")
    if args.record:
        writer = make_writer(overlay_path, out_w, out_h, out_fps, args.fourcc)

    if args.show:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, out_w, out_h)

    # CSV buffer
    rows = []

    # Landmarker
    landmarker = create_landmarker(model_path_abs, args.max_poses)

    tracker = CentroidTracker(max_dist=0.08 * max(out_w, out_h), max_age=45)

    frame_i = 0
    written = 0
    t0 = time.time()

    iterator = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if args.webcam is None else iter(int, 1)
    if args.webcam is None:
        iterator = tqdm(iterator, desc=f"[{stream_name}]")

    for _ in iterator:
        ok, bgr = cap.read()
        if not ok:
            if args.webcam is not None:
                continue
            else:
                break

        if args.skip_frames > 0 and (frame_i % (args.skip_frames + 1) != 0):
            frame_i += 1
            continue

        bgr = maybe_resize(bgr, args.max_width)
        h, w = bgr.shape[:2]
        ts_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))

        # Build MP image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        stamp = ts_ms if ts_ms > 0 else int(frame_i * (1000.0 / max(src_fps, 1.0)))
        result = landmarker.detect_for_video(mp_image, stamp)

        if result and result.pose_landmarks:
            # Filter by avg visibility
            filtered = [lms for lms in result.pose_landmarks if avg_visibility(lms) >= args.min_pose_score]

            if filtered:
                centers = [torso_centroid(lms, w, h) for lms in filtered]
                ids = tracker.update(centers, frame_i)

                for pi, lms in enumerate(filtered):
                    pid = ids[pi]
                    draw_pose(bgr, lms)
                    cx, cy = centers[pi]
                    label = f"ID {pid}"
                    cv2.rectangle(bgr, (int(cx)-50, int(cy)-60), (int(cx)+50, int(cy)-35), (0,0,0), -1)
                    cv2.putText(bgr, label, (int(cx)-45, int(cy)-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                    if args.csv:
                        for li, lm in enumerate(lms):
                            rows.append({
                                "source": stream_name,
                                "frame": frame_i,
                                "timestamp_ms": ts_ms,
                                "person_id": pid,
                                "index": li,
                                "x": lm.x,
                                "y": lm.y,
                                "z": getattr(lm, "z", float("nan")),
                                "visibility": getattr(lm, "visibility", float("nan")),
                            })

        # Show
        if args.show:
            cv2.imshow(args.window_name, bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('p'):
                while True:
                    key2 = cv2.waitKey(30) & 0xFF
                    if key2 in (27, ord('q'), ord('p')):
                        break
                if key2 in (27, ord('q')):
                    break

        if writer is not None:
            writer.write(bgr)
            written += 1

        frame_i += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    if args.csv and rows:
        df = pd.DataFrame(rows)
        out_csv = Path(args.out_dir) / f"{stream_name}_multipose.csv"
        df.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    fps = frame_i / max(elapsed, 1e-6)
    print(f"[DONE] frames={frame_i}, wrote_frames={written}, ~{fps:.1f} fps")
    if args.record:
        print(f"Overlay: {overlay_path}")

def main():
    args = get_args()
    try:
        process(args)
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    main()

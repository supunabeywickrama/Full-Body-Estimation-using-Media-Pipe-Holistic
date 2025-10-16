import os
import cv2
import math
import time
import argparse
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import mediapipe as mp


# ---------- CLI ----------
def get_args():
    p = argparse.ArgumentParser(
        "MediaPipe Holistic | Live webcam + uploaded video processing"
    )
    # Source
    p.add_argument("--webcam", type=int, default=None,
                   help="Use live webcam by index (e.g., 0). If set, --video is ignored.")
    p.add_argument("--video", type=str, default=None,
                   help="Path to a video file (mp4/avi/mov). Used only if --webcam is not set.")

    # Output
    p.add_argument("--out_dir", type=str, default="outputs",
                   help="Where to save outputs (video/CSVs).")
    p.add_argument("--record", action="store_true",
                   help="Save an overlay video of the session.")
    p.add_argument("--csv", action="store_true",
                   help="Write CSV landmark files (pose/pose_world/face/left_hand/right_hand).")

    # Display / performance
    p.add_argument("--show", action="store_true",
                   help="Show a live window for preview.")
    p.add_argument("--max_width", type=int, default=0,
                   help="If >0, downscale frames to this width (keep aspect).")
    p.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                   help="Holistic model complexity. 0=fast, 1=balanced, 2=best.")
    p.add_argument("--smooth_landmarks", action="store_true",
                   help="Enable temporal smoothing.")
    p.add_argument("--segmentation", action="store_true",
                   help="Enable pose segmentation (slower).")
    p.add_argument("--target_fps", type=float, default=0.0,
                   help="If >0 and recording, resample output FPS. 0=keep source FPS.")
    p.add_argument("--fourcc", type=str, default="mp4v",
                   help="Video writer codec (e.g., mp4v, XVID).")
    p.add_argument("--skip_frames", type=int, default=0,
                   help="Process every (skip_frames+1)th frame for speed. 0=all.")

    # Window
    p.add_argument("--window_name", type=str, default="Holistic Live",
                   help="OpenCV window title when --show is on.")
    return p.parse_args()


# ---------- Utils ----------
def ensure_dir(d: str):
    Path(d).mkdir(parents=True, exist_ok=True)


def maybe_resize(img: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return img
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    s = max_width / float(w)
    new_w = max_width
    new_h = int(round(h * s))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def open_capture(args) -> Tuple[cv2.VideoCapture, float, int, int, Optional[str]]:
    """
    Returns: cap, fps, width, height, stream_name
    """
    if args.webcam is not None:
        cap = cv2.VideoCapture(args.webcam, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {args.webcam}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (fps > 0 and math.isfinite(fps)):
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        return cap, fps, w, h, f"webcam_{args.webcam}"
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (fps > 0 and math.isfinite(fps)):
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        stem = Path(args.video).stem
        return cap, fps, w, h, stem
    raise RuntimeError("No source. Use --webcam <index> OR --video <path>.")


def make_writer(out_path: str, width: int, height: int, fps: float, fourcc: str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer at {out_path}")
    return writer


def landmark_rows(landmarks, frame_i: int, ts_ms: int, stream: str, is_world: bool) -> List[Dict]:
    rows = []
    if landmarks is None:
        return rows
    for idx, lm in enumerate(getattr(landmarks, "landmark", [])):
        rows.append({
            "frame": frame_i,
            "timestamp_ms": ts_ms,
            "stream": stream,
            "index": idx,
            "x": getattr(lm, "x", np.nan),
            "y": getattr(lm, "y", np.nan),
            "z": getattr(lm, "z", np.nan),
            "visibility": getattr(lm, "visibility", np.nan),
            "presence": getattr(lm, "presence", np.nan),
            "is_world": bool(is_world)
        })
    return rows


def draw_all(vis: np.ndarray, res, mp_h, mp_draw, mp_styles) -> np.ndarray:
    if res.face_landmarks:
        mp_draw.draw_landmarks(
            vis, res.face_landmarks, mp_h.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
        )
    if res.pose_landmarks:
        mp_draw.draw_landmarks(
            vis, res.pose_landmarks, mp_h.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
    if res.left_hand_landmarks:
        mp_draw.draw_landmarks(vis, res.left_hand_landmarks, mp_h.HAND_CONNECTIONS)
    if res.right_hand_landmarks:
        mp_draw.draw_landmarks(vis, res.right_hand_landmarks, mp_h.HAND_CONNECTIONS)
    return vis


# ---------- Main processing ----------
def run(args):
    ensure_dir(args.out_dir)
    cap, src_fps, src_w, src_h, stream_name = open_capture(args)

    # Prime to know output size after optional resize
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Empty stream.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = maybe_resize(frame, args.max_width)
    out_h, out_w = frame.shape[:2]

    # Output paths
    overlay_path = str(Path(args.out_dir) / f"{stream_name}_overlay.mp4")
    out_fps = args.target_fps if args.target_fps > 0 else src_fps
    writer = None
    if args.record:
        writer = make_writer(overlay_path, out_w, out_h, out_fps, args.fourcc)

    # CSV buffers
    pose_rows, pose_world_rows = [], []
    face_rows, lh_rows, rh_rows = [], [], []

    # For downsampling when recording
    write_every = 1
    if args.record and args.target_fps > 0 and src_fps > 0:
        ratio = src_fps / args.target_fps
        write_every = max(1, int(round(ratio)))

    # Window setup
    if args.show:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, out_w, out_h)

    mp_h = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    holistic = mp_h.Holistic(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=args.smooth_landmarks,
        enable_segmentation=args.segmentation,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_i = 0
    written = 0
    t0 = time.time()

    with holistic:
        iterator = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) if args.webcam is None else iter(int, 1)
        if args.webcam is None:
            iterator = tqdm(iterator, desc=f"[{stream_name}]")

        for _ in iterator:
            ok, frame_bgr = cap.read()
            if not ok:
                if args.webcam is not None:
                    continue
                else:
                    break

            if args.skip_frames > 0 and (frame_i % (args.skip_frames + 1) != 0):
                frame_i += 1
                continue

            frame_bgr = maybe_resize(frame_bgr, args.max_width)

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)
            rgb.flags.writeable = True

            if args.csv:
                ts_ms = int(round(cap.get(cv2.CAP_PROP_POS_MSEC)))
                pose_rows += landmark_rows(res.pose_landmarks, frame_i, ts_ms, "pose", False)
                pose_world_rows += landmark_rows(res.pose_world_landmarks, frame_i, ts_ms, "pose", True)
                face_rows += landmark_rows(res.face_landmarks, frame_i, ts_ms, "face", False)
                lh_rows += landmark_rows(res.left_hand_landmarks, frame_i, ts_ms, "left_hand", False)
                rh_rows += landmark_rows(res.right_hand_landmarks, frame_i, ts_ms, "right_hand", False)

            vis = frame_bgr
            vis = draw_all(vis, res, mp_h, mp_draw, mp_styles)

            if args.show:
                cv2.imshow(args.window_name, vis)
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
                elif key == ord('s'):
                    snap_path = str(Path(args.out_dir) / f"{stream_name}_frame{frame_i:06d}.png")
                    cv2.imwrite(snap_path, vis)

            if writer is not None:
                if write_every <= 1 or (frame_i % write_every == 0):
                    writer.write(vis)
                    written += 1

            frame_i += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    def flush(rows, suffix):
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.insert(0, "source", stream_name)
        out_csv = Path(args.out_dir) / f"{stream_name}_{suffix}.csv"
        df.to_csv(out_csv, index=False)

    if args.csv:
        flush(pose_rows, "pose")
        flush(pose_world_rows, "pose_world")
        flush(face_rows, "face")
        flush(lh_rows, "left_hand")
        flush(rh_rows, "right_hand")

    elapsed = time.time() - t0
    fps = (frame_i / elapsed) if elapsed > 0 else 0.0
    print(f"[DONE] frames={frame_i}, wrote_frames={written}, ~{fps:.1f} fps, out={args.out_dir}")
    if args.record:
        print(f"Overlay video: {overlay_path}")


def main():
    args = get_args()
    try:
        run(args)
    except Exception as e:
        # Make sure you see errors instead of a silent exit
        print(f"[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()

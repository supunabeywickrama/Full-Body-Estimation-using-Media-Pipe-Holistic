import cv2
import json
import time
import math
import numpy as np
from collections import deque
from pathlib import Path

import mediapipe as mp

# ------------------------------
# Config
# ------------------------------
CALIB_PATH = Path("calibration.json")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUTS_DIR / "finger_ability.csv"

# Smoothing windows (frames)
ANGLE_SMOOTH_N = 5
PCT_SMOOTH_N = 5

# Labels (order matters)
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# ------------------------------
# Helpers
# ------------------------------
def angle_between(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def clamp01(x):
    return max(0.0, min(1.0, x))

def moving_avg(deque_buf, val, maxlen):
    deque_buf.append(val)
    if len(deque_buf) > maxlen:
        deque_buf.popleft()
    arr = [v for v in deque_buf if not np.isnan(v)]
    return float(np.mean(arr)) if len(arr) else np.nan

def save_calibration(calib):
    CALIB_PATH.write_text(json.dumps(calib, indent=2))

def load_calibration():
    if CALIB_PATH.exists():
        return json.loads(CALIB_PATH.read_text())
    return None

# ------------------------------
# Angle extraction per finger
# ------------------------------
# MediaPipe Hands landmark indices:
# 0: wrist
# Thumb: 1(CMC), 2(MCP), 3(IP), 4(TIP)
# Index: 5(MCP), 6(PIP), 7(DIP), 8(TIP)
# Middle: 9,10,11,12 | Ring: 13,14,15,16 | Pinky: 17,18,19,20
def finger_joint_angle(hand_landmarks, image_w, image_h, which):
    lm = hand_landmarks.landmark
    def pt(i):
        return np.array([lm[i].x * image_w, lm[i].y * image_h], dtype=np.float32)

    if which == "Thumb":
        mcp, ip_, tip = pt(2), pt(3), pt(4)
        v1 = mcp - ip_
        v2 = tip - ip_
        return angle_between(v1, v2)
    elif which == "Index":
        mcp, pip_, dip = pt(5), pt(6), pt(7)
        v1 = mcp - pip_
        v2 = dip - pip_
        return angle_between(v1, v2)
    elif which == "Middle":
        mcp, pip_, dip = pt(9), pt(10), pt(11)
        v1 = mcp - pip_
        v2 = dip - pip_
        return angle_between(v1, v2)
    elif which == "Ring":
        mcp, pip_, dip = pt(13), pt(14), pt(15)
        v1 = mcp - pip_
        v2 = dip - pip_
        return angle_between(v1, v2)
    elif which == "Pinky":
        mcp, pip_, dip = pt(17), pt(18), pt(19)
        v1 = mcp - pip_
        v2 = dip - pip_
        return angle_between(v1, v2)
    return np.nan

# ------------------------------
# Main
# ------------------------------
def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    # Try to read camera FPS; fallback to 30
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cam_fps or cam_fps <= 1:
        cam_fps = 30.0

    # State
    open_angles = {f: None for f in FINGERS}
    fist_angles = {f: None for f in FINGERS}
    prev = load_calibration()
    if prev:
        open_angles.update(prev.get("open_angles", {}))
        fist_angles.update(prev.get("fist_angles", {}))
        print("[INFO] Loaded calibration from calibration.json")

    angle_smooth = {f: deque(maxlen=ANGLE_SMOOTH_N) for f in FINGERS}
    pct_smooth = {f: deque(maxlen=PCT_SMOOTH_N) for f in FINGERS}

    # CSV logging
    recording_csv = False
    csv_file = None

    # ---- Video recording (NEW) ----
    recording_video = False
    writer = None
    out_path = None

    print("""
Controls:
  C - Capture OPEN pose (hand fully open)
  F - Capture FIST pose (hand fully flexed)
  R - Toggle CSV recording (outputs/finger_ability.csv)
  V - Toggle VIDEO recording (outputs/finger_ability_YYYYmmdd_HHMMSS.mp4)
  S - Save calibration to calibration.json
  Q - Quit
""")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        abilities = {f: np.nan for f in FINGERS}
        angles = {f: np.nan for f in FINGERS}

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            for f in FINGERS:
                ang = finger_joint_angle(hand, w, h, f)
                ang = moving_avg(angle_smooth[f], ang, ANGLE_SMOOTH_N)
                angles[f] = ang

                oa = open_angles[f]
                fa = fist_angles[f]
                if oa is not None and fa is not None and not np.isnan(ang):
                    denom = (oa - fa)
                    if abs(denom) < 1e-3:
                        pct = np.nan
                    else:
                        pct = clamp01((oa - ang) / denom) * 100.0
                else:
                    pct = np.nan
                abilities[f] = moving_avg(pct_smooth[f], pct, PCT_SMOOTH_N)

            mp_draw.draw_landmarks(
                frame, hand,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # UI panel
        panel_w = 280
        cv2.rectangle(frame, (w - panel_w, 0), (w, h), (30, 30, 30), -1)
        y = 30
        cv2.putText(frame, "Finger Ability (%)", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 20

        for f in FINGERS:
            y += 35
            label = f
            pct = abilities[f]

            bar_x = w - panel_w + 10
            bar_y = y
            bar_w = panel_w - 20
            bar_h = 18
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)

            txt = "--"
            if not np.isnan(pct):
                txt = f"{pct:5.1f}%"
                fill = int(bar_w * (pct / 100.0))
                cv2.rectangle(frame, (bar_x+1, bar_y+1), (bar_x + fill, bar_y + bar_h - 1), (120, 200, 120), -1)

            cv2.putText(frame, f"{label}: {txt}", (bar_x, bar_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Calibration + logging status
        y += 40
        open_ok = all(open_angles[f] is not None for f in FINGERS)
        fist_ok = all(fist_angles[f] is not None for f in FINGERS)
        cv2.putText(frame, f"Open calib: {'OK' if open_ok else 'Needed'}", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA); y += 20
        cv2.putText(frame, f"Fist calib: {'OK' if fist_ok else 'Needed'}", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1, cv2.LINE_AA); y += 20
        cv2.putText(frame, f"CSV: {'ON' if recording_csv else 'OFF'}", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200) if recording_csv else (180,180,180), 1, cv2.LINE_AA); y += 20
        cv2.putText(frame, f"Video: {'REC' if recording_video else 'OFF'}", (w - panel_w + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200) if recording_video else (180,180,180), 1, cv2.LINE_AA)

        # ---- Video REC badge (NEW) ----
        if recording_video:
            cv2.putText(frame, "REC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (70, 22), 8, (0, 0, 255), -1)

        # CSV logging
        if recording_csv and res.multi_hand_landmarks:
            t = time.time()
            if csv_file is None:
                csv_file = CSV_PATH.open("a", buffering=1, encoding="utf-8")
                if csv_file.tell() == 0:
                    csv_file.write("timestamp," + ",".join([f"{f}_pct" for f in FINGERS]) + "\n")
            vals = [f"{abilities[f]:.2f}" if not np.isnan(abilities[f]) else "" for f in FINGERS]
            csv_file.write(f"{t}," + ",".join(vals) + "\n")

        # ---- Write video frame if recording (NEW) ----
        if recording_video and writer is not None:
            writer.write(frame)

        cv2.imshow("Finger Ability (Rehab MVP)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c'):
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                fresh = {}
                for f in FINGERS:
                    ang = finger_joint_angle(hand, w, h, f)
                    fresh[f] = ang
                for _ in range(10):
                    ok2, fr2 = cap.read()
                    if not ok2: break
                    fr2 = cv2.flip(fr2, 1)
                    rgb2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB)
                    res2 = hands.process(rgb2)
                    if res2.multi_hand_landmarks:
                        h2 = res2.multi_hand_landmarks[0]
                        for f in FINGERS:
                            a2 = finger_joint_angle(h2, w, h, f)
                            fresh[f] = np.nanmean([fresh[f], a2])
                open_angles.update(fresh)
                print("[CALIB] OPEN set:", {k: round(v,1) if v is not None else None for k,v in open_angles.items()})

        elif key == ord('f'):
            if res.multi_hand_landmarks:
                hand = res.multi_hand_landmarks[0]
                fresh = {}
                for f in FINGERS:
                    ang = finger_joint_angle(hand, w, h, f)
                    fresh[f] = ang
                for _ in range(10):
                    ok2, fr2 = cap.read()
                    if not ok2: break
                    fr2 = cv2.flip(fr2, 1)
                    rgb2 = cv2.cvtColor(fr2, cv2.COLOR_BGR2RGB)
                    res2 = hands.process(rgb2)
                    if res2.multi_hand_landmarks:
                        h2 = res2.multi_hand_landmarks[0]
                        for f in FINGERS:
                            a2 = finger_joint_angle(h2, w, h, f)
                            fresh[f] = np.nanmean([fresh[f], a2])
                fist_angles.update(fresh)
                print("[CALIB] FIST set:", {k: round(v,1) if v is not None else None for k,v in fist_angles.items()})

        elif key == ord('s'):
            save_calibration({"open_angles": open_angles, "fist_angles": fist_angles})
            print("[INFO] Saved calibration.json")

        elif key == ord('r'):
            recording_csv = not recording_csv
            if not recording_csv and csv_file is not None:
                csv_file.close()
                csv_file = None
                print(f"[INFO] Saved CSV to {CSV_PATH.resolve()}")

        elif key == ord('v'):
            # Toggle video recording
            if not recording_video:
                # Start
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                out_name = f"finger_ability_{timestamp}.mp4"
                out_path = OUTPUTS_DIR / out_name
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # .mp4
                writer = cv2.VideoWriter(str(out_path), fourcc, cam_fps, (w, h))
                if not writer.isOpened():
                    writer = None
                    print("[ERROR] Could not start video writer.")
                else:
                    recording_video = True
                    print(f"[REC] Video recording started: {out_path}")
            else:
                # Stop
                recording_video = False
                if writer is not None:
                    writer.release()
                    writer = None
                    print(f"[REC] Video saved to: {out_path.resolve()}")
                out_path = None

    # Cleanup
    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] Saved CSV to {CSV_PATH.resolve()}")
    if writer is not None:
        writer.release()
        if out_path:
            print(f"[REC] Video saved to: {out_path.resolve()}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

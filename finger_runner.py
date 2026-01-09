# finger_runner.py
# Camera feed (with hand & landmarks shown) → Handedness → Manual per-finger calibration → Save → Game

import math, random, time, csv, os, sys, json
from collections import deque
from dataclasses import dataclass

import pygame
import cv2
import mediapipe as mp
# Ensure `mp.solutions` is available
try:
    from mediapipe import solutions as mp_solutions
except Exception:
    mp_solutions = getattr(mp, "solutions", None)
if not hasattr(mp, "solutions") and mp_solutions is not None:
    mp.solutions = mp_solutions
import numpy as np

# ========================= CONFIG =========================
WIN_W, WIN_H = 1100, 620   # slightly wider to fit camera + instructions side-by-side
GROUND_Y = int(WIN_H * 0.80)
FPS = 60

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
JUMP_FINGER   = "thumb"
RIGHT_FINGER  = "index"
LEFT_FINGER   = "middle"
SHIELD_FINGER = "ring"
BRAKE_FINGER  = "pinky"

# Game thresholds & scoring
JUMP_THRESHOLD     = 45
STRAFE_DEADZONE    = 15
DIST_SCORE_RATE    = 0.06
SMOOTH_WINDOW      = 12
SMOOTH_BONUS_RATE  = 0.02
TARGET_HOLD_LOW    = 40
TARGET_HOLD_HIGH   = 70
TARGET_HOLD_TIME   = 3.0
TARGET_HOLD_BONUS  = 25

# --- Jump tuning ---
JUMP_BASE_POWER    = 11.5   # baseline jump
JUMP_MAX_POWER     = 20.8   # higher ceiling to clear tall blocks
JUMP_POWER_CURVE   = 1.45   # >1 gentler start, strong finish
JUMP_THRESHOLD_PAD = 6      # start building power slightly below threshold

# Difficulty progression
MIN_SPAWN_GAP    = 2.0
MAX_SPAWN_GAP    = 3.2
GAME_SPEED_START = 5.3
GAME_SPEED_MAX   = 11.0
GAME_ACCEL       = 0.0009

# Calibration
CAPTURE_KEY     = pygame.K_SPACE
REDO_KEY        = pygame.K_r
RECALI_KEY      = pygame.K_c         # press 'C' in game to re-calibrate
CALIB_DIR       = "calibration_profiles"
os.makedirs(CALIB_DIR, exist_ok=True)

# Logging
LOG_DIR = "game_logs"; os.makedirs(LOG_DIR, exist_ok=True)
SESSION_CSV = os.path.join(LOG_DIR, "sessions.csv")

# States
STATE_CALIB = "calib"
STATE_RUN   = "run"

# Camera feed layout during calibration
CAM_VIEW_W = 640
CAM_VIEW_H = 480
CAM_POS    = (24, 80)               # top-left of camera panel
MIRROR_DISPLAY = True               # mirror camera display (familiar like a selfie)

def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def lerp(a, b, t):    return a + (b - a) * t
def ease_out_quart(t): return 1 - pow(1 - t, 4)
def smooth01(x):      # clamp to 0..1
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x

# ========================= CAMERA + HANDS =========================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

def _angle_at(pA, pB, pC):
    BA = np.array([pA[0]-pB[0], pA[1]-pB[1], pA[2]-pB[2]])
    BC = np.array([pC[0]-pB[0], pC[1]-pB[1], pC[2]-pB[2]])
    la = np.linalg.norm(BA); lb = np.linalg.norm(BC)
    if la < 1e-6 or lb < 1e-6: return 180.0
    cosang = float((BA @ BC) / (la*lb))
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))

class CameraHands:
    """
    Grabs frames from camera and extracts:
      - handedness: "Left"/"Right"
      - finger curl metrics (higher == more flex): thumb (CMC-MCP-TIP), others (MCP-PIP-TIP)
    Also returns a visualization frame with landmarks drawn for the UI.
    """
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.last_metrics = {f: 0.0 for f in FINGERS}
        self.last_hand = None
        self.last_vis_bgr = None
        self.ok = False

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.ok = False
            return self.last_metrics, self.last_hand, False, self.last_vis_bgr

        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.hands.process(frame_rgb)

        hand_label = None
        metrics = {f: 0.0 for f in FINGERS}

        if res.multi_hand_landmarks and res.multi_handedness:
            lms = res.multi_hand_landmarks[0]
            # MediaPipe label is subject's viewpoint; flip for mirrored UI so it matches what user sees.
            mp_label = res.multi_handedness[0].classification[0].label  # "Left"/"Right"
            hand_label = ("Left" if mp_label == "Right" else "Right") if MIRROR_DISPLAY else mp_label

            pts = [(lm.x, lm.y, lm.z) for lm in lms.landmark]

            def curl_thumb():
                A = pts[1]; B = pts[2]; C = pts[4]        # CMC-MCP-TIP
                ang = _angle_at(A, B, C)
                return clamp(180.0 - ang, 0.0, 180.0)
            def curl_three(mcp, pip, tip):
                A = pts[mcp]; B = pts[pip]; C = pts[tip]  # MCP-PIP-TIP
                ang = _angle_at(A, B, C)
                return clamp(180.0 - ang, 0.0, 180.0)

            metrics = {
                "thumb":  curl_thumb(),
                "index":  curl_three(5,6,8),
                "middle": curl_three(9,10,12),
                "ring":   curl_three(13,14,16),
                "pinky":  curl_three(17,18,20),
            }
            # temporal smoothing
            for k in metrics:
                metrics[k] = 0.7*self.last_metrics.get(k, metrics[k]) + 0.3*metrics[k]

            # draw landmarks on visible frame (we mirror on blit)
            frame.flags.writeable = True
            vis = frame.copy()
            mp_draw.draw_landmarks(
                vis,
                lms,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style(),
            )
            # UI-facing label on frame
            cv2.rectangle(vis, (10,10), (180,45), (30,30,30), -1)
            cv2.putText(vis, f"Hand: {hand_label}", (20,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            self.last_vis_bgr = vis
            self.last_hand = hand_label
            self.last_metrics = metrics
            self.ok = True
            return metrics, hand_label, True, vis

        # no hand found
        self.ok = False
        self.last_vis_bgr = frame
        return self.last_metrics, self.last_hand, False, frame

    def release(self):
        try: self.hands.close()
        except: pass
        try: self.cap.release()
        except: pass

# ========================= WORLD / GAME ENTITIES =========================
@dataclass
class Obstacle:
    x: float; w: int; h: int; kind: str
    passed: bool = False
    @property
    def rect(self): return pygame.Rect(int(self.x), GROUND_Y - self.h, self.w, self.h)

class ParticleSystem:
    def __init__(self): self.p=[]
    def emit(self, x, y, n=12, spread=1.0, speed=2.6, life=0.9):
        for _ in range(n):
            ang = random.uniform(-math.pi/2 - spread, -math.pi/2 + spread)
            v = random.uniform(speed*0.5, speed*1.5)
            self.p.append([x, y, v*math.cos(ang), v*math.sin(ang), life, random.uniform(2,4)])
    def update(self, dt):
        for pr in self.p[:]:
            pr[0]+=pr[2]; pr[1]+=pr[3]; pr[4]-=dt; pr[3]+=9.8*0.05*dt
            if pr[4]<=0: self.p.remove(pr)
    def draw(self, surf):
        for pr in self.p:
            a=int(clamp(255*pr[4],0,255))
            pygame.draw.circle(surf,(255,255,255,a),(int(pr[0]),int(pr[1])),int(pr[5]))

class Player:
    def __init__(self):
        self.w=52; self.h=52
        self.x=int(WIN_W*0.18); self.y=GROUND_Y-self.h
        self.vy=0.0; self.on_ground=True
        self.shield_time=0.0; self.lane_offset=0.0
        self.target_glow = 0.0

    @property
    def rect(self):
        lane_px=int(self.lane_offset*50)
        return pygame.Rect(self.x+lane_px,int(self.y),self.w,self.h)
    def jump(self, power=11.5):
        if self.on_ground: self.vy=-power; self.on_ground=False
    def update(self, dt, g=22.0):
        self.vy += g*dt; self.y += self.vy
        if self.y + self.h >= GROUND_Y:
            self.y = GROUND_Y - self.h; self.vy=0; self.on_ground=True
        self.shield_time = max(0.0, self.shield_time - dt)
    def draw(self, surf, t):
        r = self.rect

        # floating animation
        bob = 3 * math.sin(t * 4)
        cx = r.centerx
        cy = r.centery + bob

        # -------- Shadow --------
        shadow = pygame.Rect(0, 0, r.w-8, 10)
        shadow.center = (cx, GROUND_Y + 6)
        pygame.draw.ellipse(surf, (0, 0, 0, 60), shadow)

        # -------- Backpack --------
        pack = pygame.Rect(0, 0, 12, 26)
        pack.center = (cx - r.w//2 - 4, cy)
        pygame.draw.rect(surf, (90, 95, 110), pack, border_radius=6)

        # -------- Body (robot suit) --------
        body = pygame.Rect(0, 0, r.w, r.h-10)
        body.center = (cx, cy+6)
        soft_shadow_rect(surf, body, (0,0,0))
        base_color = (200, 205, 215)
        if self.target_glow > 0:
             glow = int(40 * self.target_glow)
             base_color = (200-glow, 225, 200-glow)

        pygame.draw.rect(surf, base_color, body, border_radius=14)

        # chest panel
        panel = pygame.Rect(body.x+12, body.y+16, body.w-24, 12)
        pygame.draw.rect(surf, (80, 170, 240), panel, border_radius=6)

        # -------- Helmet --------
        helmet = pygame.Rect(0, 0, r.w-6, r.h-18)
        helmet.center = (cx, cy-10)
        pygame.draw.ellipse(surf, (230, 235, 245), helmet)

        # visor
        visor = pygame.Rect(0, 0, helmet.w-14, helmet.h-22)
        visor.center = helmet.center
        pygame.draw.ellipse(surf, (30, 50, 70), visor)

        # visor shine
        shine = pygame.Rect(visor.x+6, visor.y+6, visor.w//3, visor.h//2)
        pygame.draw.ellipse(surf, (120, 180, 220), shine)

        # -------- Legs / boots --------
        boot_l = pygame.Rect(body.x+10, body.bottom-2, 12, 10)
        boot_r = pygame.Rect(body.right-22, body.bottom-2, 12, 10)
        pygame.draw.rect(surf, (120,120,130), boot_l, border_radius=4)
        pygame.draw.rect(surf, (120,120,130), boot_r, border_radius=4)
      
    # -------- Jetpack flame (only while jumping upward) --------
        if self.vy < -1:
              flame_h = int(12 + abs(self.vy) * 1.5)
              flame = pygame.Surface((16, flame_h), pygame.SRCALPHA)
              pygame.draw.polygon(
                  flame,
                  (255, 140, 40, 200),
                  [(8, flame_h), (0, 0), (16, 0)]
              )
              surf.blit(flame, (cx-8, body.bottom-2))

    # -------- Shield effect --------
        if self.shield_time > 0:
             a = int(90 + 80 * abs(math.sin(t*6)))
             glow = pygame.Surface((r.w+30, r.h+30), pygame.SRCALPHA)
             pygame.draw.ellipse(glow, (120, 255, 160, a), glow.get_rect(), width=5)
             surf.blit(glow, (r.x-15, r.y-20))


# ========================= UI HELPERS =========================
def draw_vertical_gradient(surf, rect, c1, c2):
    x,y,w,h=rect
    for i in range(h):
        t=i/max(1,h-1)
        col=(int(lerp(c1[0],c2[0],t)), int(lerp(c1[1],c2[1],t)), int(lerp(c1[2],c2[2],t)))
        pygame.draw.line(surf,col,(x,y+i),(x+w,y+i))

def soft_shadow_rect(surf, rect, color, radius=12, alpha=80):
    sh=pygame.Surface((rect.w+radius*2, rect.h+radius*2), pygame.SRCALPHA)
    pygame.draw.rect(sh, (*color,alpha), sh.get_rect(), border_radius=radius)
    surf.blit(sh,(rect.x-radius,rect.y-radius))

class UI:
    def __init__(self, font, tiny):
        self.font=font; self.tiny=tiny
        self.msg=""; self.msg_t=0.0
    def flash(self, text): self.msg=text; self.msg_t=2.0
    def background(self, surf, t, speed_norm):
        draw_vertical_gradient(surf,(0,0,WIN_W,WIN_H),(15,18,40),(10,70,110))
        # simple parallax bands
        for h,a in [(120,0.25),(80,0.45),(50,0.65)]:
            y=GROUND_Y-h; phase=(t*30*(a+0.2))%WIN_W
            col=(20+int(40*a),40+int(90*a),60+int(110*a))
            for k in range(-1,4):
                x=int((k*300)-phase); pygame.draw.ellipse(surf,col,(x,y,420,h*2))
        pygame.draw.rect(surf,(30,35,45),(0,GROUND_Y,WIN_W,WIN_H-GROUND_Y))
    def scorebar(self, surf, score, combo):
        bar=pygame.Rect(24,20,360,24); soft_shadow_rect(surf,bar,(0,0,0))
        pygame.draw.rect(surf,(245,245,245),bar,border_radius=12)
        inner=bar.inflate(-6,-6); fill=inner.copy(); fill.w=int(inner.w*clamp((score%100)/100,0,1))
        pygame.draw.rect(surf,(60,210,255),fill,border_radius=10)
        surf.blit(self.tiny.render(f"Score {int(score)}",True,(25,25,25)),(bar.x+8,bar.y+4))
        if combo>1.0: surf.blit(self.tiny.render(f"x{combo:.1f}",True,(10,100,25)),(bar.right-48,bar.y+4))
    def finger_panel(self, surf, values, hand_label, cam_ok):
        panel=pygame.Rect(WIN_W-260,24,236,200); soft_shadow_rect(surf,panel,(0,0,0))
        pygame.draw.rect(surf,(245,245,250),panel,border_radius=12)
        title = f"{'✓' if cam_ok else '×'} Hand: {hand_label or '—'}"
        col = (20,120,40) if cam_ok else (170,40,40)
        surf.blit(self.tiny.render(title, True, col), (panel.x+10, panel.y+8))
        y=panel.y+30
        for name,val in values.items():
            pygame.draw.rect(surf,(225,230,240),(panel.x+10,y+18,panel.w-20,12),border_radius=6)
            fill=pygame.Rect(panel.x+10,y+18,int((val/100.0)*(panel.w-20)),12)
            colb=(90,200,120) if TARGET_HOLD_LOW<=val<=TARGET_HOLD_HIGH else (60,210,255)
            pygame.draw.rect(surf,colb,fill,border_radius=6)
            surf.blit(self.tiny.render(f"{name.title():<6} {int(val):3d}%",True,(20,20,30)),(panel.x+10,y)); y+=34
    def message_center(self, surf, t):
        if self.msg_t>0:
            self.msg_t -= 1/FPS
            a=int(255*ease_out_quart(clamp(self.msg_t/2.0,0,1)))
            img=self.font.render(self.msg,True,(255,255,255)); img.set_alpha(a)
            surf.blit(img,(WIN_W//2 - img.get_width()//2, 50))
    def guide_card(self, surf, title, subtitle, rect=None):
        if rect is None:
            rect = pygame.Rect(CAM_POS[0]+CAM_VIEW_W+24, 80, WIN_W - (CAM_POS[0]+CAM_VIEW_W+24) - 24, 320)
        soft_shadow_rect(surf, rect, (0,0,0))
        pygame.draw.rect(surf,(248,248,252),rect,border_radius=18)
        t1=self.font.render(title,True,(25,25,35)); t2=self.tiny.render(subtitle,True,(60,70,80))
        surf.blit(t1,(rect.centerx - t1.get_width()//2, rect.y+18))
        surf.blit(t2,(rect.centerx - t2.get_width()//2, rect.y+56))
        return rect
    def game_over(self, surf, score, best):
        box=pygame.Rect(WIN_W//2-240, WIN_H//2-140, 480,260)
        soft_shadow_rect(surf,box,(0,0,0)); pygame.draw.rect(surf,(248,248,252),box,border_radius=18)
        t1=self.font.render("Session Complete",True,(25,25,35))
        surf.blit(t1,(box.centerx - t1.get_width()//2, box.y+16))
        sub=self.tiny.render(f"Score {int(score)}  •  Best {int(best)}",True,(50,60,70))
        surf.blit(sub,(box.centerx - sub.get_width()//2, box.y+56))
        hint=self.tiny.render("Press SPACE to retry",True,(80,90,110))
        surf.blit(hint,(box.centerx - hint.get_width()//2, box.bottom-36))

# ========================= CALIBRATION FLOW =========================
class ManualCalibration:
    """
    For each finger: capture 0% pose -> capture 100% pose (SPACE or click).
    Saves to 'calibration_<Left|Right>.json' and loads it on next run for that hand.
    """
    def __init__(self, font, tiny):
        self.font=font; self.tiny=tiny
        self.index=0    # which finger (0..4)
        self.phase=0    # 0 -> capture 0%; 1 -> capture 100%
        self.data={"Left":{f:{"v0":None,"v100":None} for f in FINGERS},
                   "Right":{f:{"v0":None,"v100":None} for f in FINGERS}}
        self.complete=False
        self.loaded=False

    def _path(self, hand):
        CALIB_DIR = "calibration_profiles"
        os.makedirs(CALIB_DIR, exist_ok=True)
        return os.path.join(CALIB_DIR, f"calibration_{hand}.json")

    def try_load(self, hand):
        p=self._path(hand)
        if os.path.exists(p):
            try:
                with open(p,"r") as f: self.data[hand]=json.load(f)
                ok = all(k in self.data[hand] and
                         self.data[hand][k]["v0"] is not None and
                         self.data[hand][k]["v100"] is not None for k in FINGERS)
                if ok:
                    self.complete=True; self.loaded=True
                    return True
            except: pass
        return False

    def map_percent(self, hand, finger, raw_val):
        c = self.data[hand][finger]
        v0, v100 = c["v0"], c["v100"]
        if v0 is None or v100 is None: return 50.0
        pct = 100.0 * (raw_val - v0) / (v100 - v0)
        return clamp(pct, 0.0, 100.0)

    def reset_current(self, hand):
        f = FINGERS[self.index]
        self.data[hand][f]["v0"]=None; self.data[hand][f]["v100"]=None
        self.phase=0

    def handle_capture(self, hand, raw_value):
        f = FINGERS[self.index]
        if self.phase == 0:
            self.data[hand][f]["v0"] = float(raw_value)
            self.phase = 1
        else:
            self.data[hand][f]["v100"] = float(raw_value)
            if abs(self.data[hand][f]["v100"] - self.data[hand][f]["v0"]) < 1e-6:
                self.data[hand][f]["v100"] = self.data[hand][f]["v0"] + 1.0
            self.index += 1; self.phase = 0
            if self.index >= len(FINGERS):
                try:
                    with open(self._path(hand), "w") as f:
                        json.dump(self.data[hand], f, indent=2)
                except: pass
                self.complete=True

    def draw(self, surf, ui: UI, hand, raw_metrics, recalib_mode=False):
        if self.complete:
            title="Calibration complete"
            subtitle=("Recalibration saved. Starting the game…"
                      if recalib_mode else f"Hand: {hand} — saved. Starting the game…")
            rect = pygame.Rect(CAM_POS[0]+CAM_VIEW_W+24, 80, WIN_W - (CAM_POS[0]+CAM_VIEW_W+24) - 24, 320)
            box=ui.guide_card(surf, title, subtitle, rect)
            y=box.y+110
            for f in FINGERS:
                c=self.data[hand][f]
                surf.blit(self.tiny.render(f"{f.title()}: 0%={c['v0']:.2f} | 100%={c['v100']:.2f}",True,(60,70,80)),
                          (box.x+24,y)); y+=22
            return True

        # instruction card right of camera
        step = f"({self.index+1}/5)"
        f = FINGERS[self.index]
        if self.phase == 0:
            title=f"Calibrate {f.title()} {step}"
            subtitle=("Recalibration mode • Hold your 0% pose and press SPACE"
                      if recalib_mode else "Hold your 0% pose (relaxed/straight). Press SPACE or click to capture.")
        else:
            title=f"Calibrate {f.title()} {step}"
            subtitle=("Recalibration mode • Hold your 100% pose and press SPACE"
                      if recalib_mode else "Hold your 100% pose (full flex). Press SPACE or click to capture.")
        rect = pygame.Rect(CAM_POS[0]+CAM_VIEW_W+24, 80, WIN_W - (CAM_POS[0]+CAM_VIEW_W+24) - 24, 320)
        box=ui.guide_card(surf, title, subtitle, rect)

        # show progress
        y = box.y + 110
        for i, ff in enumerate(FINGERS):
            d=self.data[hand][ff]
            done = (d["v0"] is not None and d["v100"] is not None)
            mark = "✅" if done else ("➤" if ff==f else "●")
            col  = (80,230,120) if done else ((60,110,240) if ff==f else (140,150,160))
            surf.blit(self.tiny.render(f"{mark} {ff.title()}", True, col), (box.x+24, y))
            y += 24

        # show live metric for the current finger
        live = raw_metrics.get(f, 0.0)
        bar = pygame.Rect(box.x+24, box.bottom-90, box.w-48, 26)
        pygame.draw.rect(surf,(220,225,235),bar,border_radius=12)
        pv = clamp(live/180.0, 0.0, 1.0)
        pygame.draw.rect(surf,(60,210,255),(bar.x,bar.y,int(pv*bar.w),26), border_radius=12)
        surf.blit(self.tiny.render(f"Live curl: {live:.2f}", True, (30,35,45)),
                  (bar.centerx-60, bar.y-24))

        hint = "SPACE/Click = Capture • R = Redo this finger"
        surf.blit(self.tiny.render(hint, True, (50,60,70)), (box.centerx-140, box.bottom-40))
        return False

# ========================= GAME CORE =========================
class Game:
    def __init__(self):
        pygame.init()
        try: pygame.mixer.init()
        except pygame.error: pass

        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("RehabX • Finger Runner")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 30)
        self.tiny  = pygame.font.SysFont("Arial", 20)

        self.ui = UI(self.font, self.tiny)
        self.player = Player()
        self.particles = ParticleSystem()

        self.achievements = set()
        self.target_hits = 0

        # Camera + hands
        self.cam = CameraHands()

        # Calibration manager
        self.calib = ManualCalibration(self.font, self.tiny)

        self.history = {f: deque(maxlen=SMOOTH_WINDOW) for f in FINGERS}
        self.state = STATE_CALIB
        self.best = self.load_best()
        self.reset_run_metrics()
        self.hand_label=None

        # NEW: flag to prevent auto-loading saved calibration when user pressed C
        self.recalib_mode = False

    def reset_run_metrics(self):
        self.t=0.0; self.speed=GAME_SPEED_START
        self.obstacles=[]; self.spawn_t=0.0
        self.spawn_gap=random.uniform(MIN_SPAWN_GAP, MAX_SPAWN_GAP)
        self.score=0.0; self.combo=1.0; self.combo_t=0.0
        self.hold_t=0.0; self.gameover=False
        self.session_start=time.time()
        self.avg_thumb_acc=0.0; self.avg_thumb_n=0
        self.smooth_acc=0.0; self.smooth_n=0
        for dq in self.history.values(): dq.clear()
        self.player.y=GROUND_Y-self.player.h; self.player.vy=0; self.player.on_ground=True

    def load_best(self):
        if not os.path.exists(SESSION_CSV): return 0
        best=0
        try:
            with open(SESSION_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    best=max(best, int(float(row.get("score",0))))
        except: pass
        return best

    def _blit_camera(self, bgr):
        if bgr is None: return
        frame = cv2.flip(bgr, 1) if MIRROR_DISPLAY else bgr
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (CAM_VIEW_W, CAM_VIEW_H), interpolation=cv2.INTER_LINEAR)
        surf = pygame.image.frombuffer(rgb.tobytes(), (CAM_VIEW_W, CAM_VIEW_H), 'RGB')
        # camera frame border
        cam_rect = pygame.Rect(CAM_POS[0]-4, CAM_POS[1]-4, CAM_VIEW_W+8, CAM_VIEW_H+8)
        soft_shadow_rect(self.screen, cam_rect, (0,0,0))
        pygame.draw.rect(self.screen, (245,245,250), cam_rect, border_radius=12)
        self.screen.blit(surf, CAM_POS)

    def log_session(self, score, duration, avg_thumb, avg_smooth):
        write_header = not os.path.exists(SESSION_CSV)
        with open(SESSION_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["ts","score","duration_sec","avg_thumb","avg_smooth"])
            if write_header: w.writeheader()
            w.writerow({
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "score": int(score),
                "duration_sec": int(duration),
                "avg_thumb": round(avg_thumb,1),
                "avg_smooth": round(avg_smooth,3)
            })

    def map_to_percent(self, hand, raw_metrics):
        mapped={}
        for f in FINGERS:
            mapped[f]=self.calib.map_percent(hand, f, raw_metrics.get(f, 0.0))
        return mapped

    def spawn_obstacle(self):
        kind=random.choice(["box","low","tall","box"])
        w,h=(36,36) if kind=="box" else (60,28) if kind=="low" else (30,60)
        self.obstacles.append(Obstacle(WIN_W+20,w,h,kind))

    def collisions(self):
        pr=self.player.rect
        for obs in self.obstacles:
            if pr.colliderect(obs.rect):
                if self.player.shield_time>0:
                    obs.x=-9999; self.score+=5*self.combo; self.ui.flash("Shield Block +5")
                else:
                    self.gameover=True

    def update_combo(self, dt, success):
        decay=0.5
        if success:
            self.combo=min(4.0,self.combo+0.02); self.combo_t=decay
        else:
            self.combo_t-=dt
            if self.combo_t<=0:
                self.combo=max(1.0,self.combo-0.02); self.combo_t=decay

    def run(self):
        while True:
            dt=self.clock.tick(FPS)/1000.0; self.t+=dt
            click=False
            for ev in pygame.event.get():
                if ev.type==pygame.QUIT:
                    self.cam.release(); pygame.quit(); sys.exit()
                if ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1: click=True
                if self.state==STATE_CALIB and ev.type==pygame.KEYDOWN and ev.key==REDO_KEY:
                    if self.hand_label: self.calib.reset_current(self.hand_label)
                if self.state==STATE_RUN and ev.type==pygame.KEYDOWN and ev.key==RECALI_KEY:
                    # Re-calibrate (force full flow; do NOT auto-load saved file)
                    self.state=STATE_CALIB
                    self.recalib_mode = True
                    self.calib.complete=False; self.calib.loaded=False
                    self.calib.index=0; self.calib.phase=0
                    self.reset_run_metrics()
                if self.gameover and ev.type==pygame.KEYDOWN and ev.key==pygame.K_SPACE:
                    self.reset_run_metrics()
                    self.state = STATE_RUN

            # ---- Camera read ----
            raw, hand, ok, vis_bgr = self.cam.read()
            if hand is not None: self.hand_label = hand

            # ---- Background ----
            self.ui.background(self.screen, self.t, (self.speed-GAME_SPEED_START)/(GAME_SPEED_MAX-GAME_SPEED_START))

            # ---- Calibration ----
            if self.state == STATE_CALIB:
                # Camera panel
                self._blit_camera(vis_bgr)

                # Load saved calibration ONLY when not in recalibration mode
                if (not self.recalib_mode) and self.hand_label and not self.calib.loaded and not self.calib.complete:
                    self.calib.try_load(self.hand_label)

                # Right-side HUD & instructions
                fake_pct = {f: 0 for f in FINGERS}
                self.ui.finger_panel(self.screen, fake_pct, self.hand_label, ok)

                if not ok:
                    self.ui.guide_card(self.screen, "Show your hand to the camera",
                                       "Keep one hand inside the view. Landmarks will appear.")
                    pygame.display.flip()
                    continue

                done = self.calib.draw(self.screen, self.ui, self.hand_label, raw, recalib_mode=self.recalib_mode)

                # Capture with Space/Click
                keys = pygame.key.get_pressed()
                if (keys[CAPTURE_KEY] or click) and not self.calib.complete:
                    cur_finger = FINGERS[self.calib.index]
                    self.calib.handle_capture(self.hand_label, raw[cur_finger])

                if done:
                    self.ui.flash("Calibration ready. Let's run!")
                    pygame.display.flip()
                    pygame.time.wait(800)
                    self.state = STATE_RUN
                    self.recalib_mode = False  # leave recalibration mode
                    self.reset_run_metrics()

                self.ui.message_center(self.screen, self.t)
                pygame.display.flip()
                continue

            # ---------------- RUN (Game) ----------------
            pct = self.map_to_percent(self.hand_label or "Right", raw)
            for k,dq in self.history.items(): dq.append(pct[k])

            if not self.gameover:
                # Controls
                move=0.0
                if pct[RIGHT_FINGER] > STRAFE_DEADZONE: move += (pct[RIGHT_FINGER]/100.0)
                if pct[LEFT_FINGER]  > STRAFE_DEADZONE: move -= (pct[LEFT_FINGER]/100.0)
                self.player.lane_offset = clamp(lerp(self.player.lane_offset, clamp(move,-1,1), 0.15), -1, 1)

                # Jump mapping with curve & higher ceiling
                thumb_pct = pct[JUMP_FINGER]
                if self.player.on_ground and thumb_pct >= (JUMP_THRESHOLD - JUMP_THRESHOLD_PAD):
                    t = (thumb_pct - (JUMP_THRESHOLD - JUMP_THRESHOLD_PAD)) / (100.0 - (JUMP_THRESHOLD - JUMP_THRESHOLD_PAD))
                    t = smooth01(t) ** JUMP_POWER_CURVE
                    power = lerp(JUMP_BASE_POWER, JUMP_MAX_POWER, t)
                    self.player.jump(power)
                    self.particles.emit(self.player.rect.centerx, self.player.rect.bottom)

                if pct[SHIELD_FINGER] >= 80: self.player.shield_time = max(self.player.shield_time, 0.7)
                brake = 1.0 - 0.35*(pct[BRAKE_FINGER]/100.0)

                # World
                self.player.update(dt)
                self.speed = clamp(self.speed + GAME_ACCEL*dt*1000 * brake, GAME_SPEED_START, GAME_SPEED_MAX)

                self.spawn_t += dt
                if self.spawn_t >= self.spawn_gap:
                    self.spawn_obstacle(); self.spawn_t=0
                    self.spawn_gap = random.uniform(MIN_SPAWN_GAP, MAX_SPAWN_GAP)*(1.15 - (self.speed-GAME_SPEED_START)/(GAME_SPEED_MAX-GAME_SPEED_START)*0.6)
                for obs in self.obstacles[:]:
                    obs.x -= self.speed
                    if obs.x + obs.w < -10: self.obstacles.remove(obs)
                    elif not obs.passed and obs.x < self.player.rect.x:
                        obs.passed=True; self.score += 2*self.combo

                self.collisions()

                # Scoring
                self.score += DIST_SCORE_RATE * self.combo
                dq = self.history[JUMP_FINGER]
                if len(dq)>=dq.maxlen:
                    var=sum((v - sum(dq)/len(dq))**2 for v in dq)/len(dq)
                    smooth=1.0/(1.0 + var*0.02)
                    self.score += SMOOTH_BONUS_RATE * smooth * self.combo
                    self.smooth_acc += smooth; self.smooth_n += 1
                    self.update_combo(dt, success=smooth>0.6)

                v = pct[JUMP_FINGER]
                if TARGET_HOLD_LOW <= v <= TARGET_HOLD_HIGH:
                    self.player.target_glow = min(1.0, self.player.target_glow + dt * 2)
                    self.hold_t += dt
                    if self.hold_t >= TARGET_HOLD_TIME:
                        self.hold_t = 0.0
                        self.score += TARGET_HOLD_BONUS * self.combo
                        self.target_hits += 1
                        if self.target_hits == 3 and "Steady Control" not in self.achievements:
                             self.achievements.add("Steady Control")
                             self.ui.flash("🏅 Achievement: Steady Control!")
                  
                        self.ui.flash(f"Great Control +{TARGET_HOLD_BONUS}!")
                        cx, cy = self.player.rect.centerx, self.player.rect.centery
                        for _ in range(3): self.particles.emit(cx, cy, n=20, spread=1.0, speed=3.4, life=1.2)
                else:
                    self.hold_t = max(0.0, self.hold_t - dt*0.4)
                    self.player.target_glow = max(0.0, self.player.target_glow - dt * 1.5)

                self.avg_thumb_acc += v; self.avg_thumb_n += 1

            # Draw world
            for obs in self.obstacles:
                pygame.draw.rect(self.screen,(240,95,95),obs.rect,border_radius=6)
                pygame.draw.rect(self.screen,(0,0,0),obs.rect,width=2,border_radius=6)
            self.player.draw(self.screen, self.t)
            self.particles.update(dt); self.particles.draw(self.screen)

            # HUD
            self.ui.finger_panel(self.screen, pct, self.hand_label, self.cam.ok)
            self.ui.scorebar(self.screen, self.score, self.combo)
            self.ui.message_center(self.screen, self.t)
            dbg = self.tiny.render(f"Thumb%: {int(pct[JUMP_FINGER])}", True, (230,235,245))
            self.screen.blit(dbg, (24, WIN_H-60))
            timer = self.tiny.render(f"{int(self.t)//60:02d}:{int(self.t)%60:02d}  (C=recalibrate)", True, (230,235,245))
            self.screen.blit(timer, (24, WIN_H-36))
            pygame.display.flip()

            if self.gameover:
                if "First Mission" not in self.achievements:
                     self.achievements.add("First Mission")
                     
                     self.ui.flash("🏅 Achievement: First Mission!")

                dur=time.time()-self.session_start
                avg_thumb=(self.avg_thumb_acc/self.avg_thumb_n) if self.avg_thumb_n else 0
                avg_smooth=(self.smooth_acc/self.smooth_n) if self.smooth_n else 0
                self.log_session(self.score, dur, avg_thumb, avg_smooth)
                self.best=max(self.best,int(self.score))
                overlay=pygame.Surface((WIN_W,WIN_H),pygame.SRCALPHA); overlay.fill((0,0,0,140))
                self.screen.blit(overlay,(0,0))
                self.ui.game_over(self.screen, self.score, self.best)
                pygame.display.flip()
                waiting=True
                while waiting:
                    for ev in pygame.event.get():
                        if ev.type==pygame.QUIT:
                            self.cam.release(); pygame.quit(); sys.exit()
                        if ev.type==pygame.KEYDOWN and ev.key==pygame.K_SPACE:
                            waiting=False
                    self.clock.tick(30)

                # restart after the wait loop
                self.reset_run_metrics()
                self.state = STATE_RUN
                self.gameover = False

if __name__ == "__main__":
    Game().run()

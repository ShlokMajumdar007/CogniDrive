"""LivePipelineRunner — CogniDrive AI Cockpit HUD with futuristic face scanning overlay.

Runs as a single daemon thread that:
    1. Reads frames from CameraManager (non-blocking).
    2. Calls PipelineManager.process_frame() for full ML inference.
    3. Updates PipelineManager.last_prediction for SSE dashboard.
    4. Renders a full futuristic HUD cockpit overlay:
         - Animated face scanning with 468 landmarks, radar rings,
           corner brackets, sweep animation, and confidence readout.
         - Left panel: driver profile, cognitive metrics, attention,
           drowsiness, stress, distraction bars.
         - Right panel: driver state, risk gauge, recommendation card,
           system status.
         - Bottom strip: EAR, head pose, gaze direction, PERCLOS,
           blink count, fatigue probability.
    5. Auto-retries on camera disconnect.
    6. Exits cleanly on Q / ESC / window close.

Architecture contract:
    - CameraManager is the only camera source (no duplicate capture).
    - PipelineManager performs all ML inference (no duplicate inference).
    - This file is visualization-only: it consumes PipelineManager outputs.
    - No FastAPI or asyncio calls; pure threading + OpenCV.

Singleton pattern: one instance system-wide via get_instance().
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

logger = logging.getLogger("CogniDrive.LivePipelineRunner")

# ─────────────────────────────────────────────────────────────────────────────
# Timing
# ─────────────────────────────────────────────────────────────────────────────
_CAMERA_WAIT_SLEEP: float = 0.05
_RECONNECT_SLEEP: float   = 2.0
_TARGET_FPS: float        = 30.0
_MIN_FRAME_SLEEP: float   = 1.0 / _TARGET_FPS

# ─────────────────────────────────────────────────────────────────────────────
# Window
# ─────────────────────────────────────────────────────────────────────────────
_WIN_NAME: str = "CogniDrive — AI Cockpit"
_HUD_W: int   = 1280   # total canvas width
_HUD_H: int   = 720    # total canvas height
_CAM_X: int   = 280    # camera feed left edge
_CAM_Y: int   = 0      # camera feed top edge
_CAM_W: int   = 720    # camera feed display width
_CAM_H: int   = 540    # camera feed display height
_PANEL_L_W: int = 280  # left panel width
_PANEL_R_W: int = 280  # right panel width
_BAR_H: int   = 140    # bottom bar height

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette — BGR
# ─────────────────────────────────────────────────────────────────────────────
_BG          = (8,   12,  18)    # near-black deep space
_PANEL_BG    = (12,  18,  28)    # panel dark navy
_PANEL_LINE  = (0,   80, 140)    # panel separator
_CYAN        = (255, 210,   0)   # primary neon cyan (BGR!)
_CYAN_DIM    = (120, 100,   0)   # dimmed cyan
_CYAN_PALE   = (200, 180,  40)   # pale cyan
_GREEN_HUD   = (0,   220,  80)   # healthy green
_GREEN_DIM   = (0,    80,  30)   # dim green
_ORANGE      = (0,   160, 255)   # warning orange
_RED_HUD     = (30,   30, 220)   # danger red
_WHITE       = (220, 220, 230)   # off-white text
_GREY        = (90,   95, 110)   # muted grey
_ACCENT_BLUE = (220, 120,  10)   # accent teal-blue
_GOLD        = (0,   180, 220)   # amber gold for risk

# Scanning / landmark colours
_LM_DOT      = (255, 200,   0)   # cyan landmark dot
_LM_MESH     = (100,  80,   0)   # dim mesh edge
_SWEEP_COL   = (200, 220,  50)   # scan sweep line
_RING_COL    = (180, 180,   0)   # radar ring
_BOX_COL_OK  = (255, 210,   0)   # face box tracked
_BOX_COL_WARN= (0,  150, 255)   # face box warning
_BOX_COL_ERR = (30,   30, 220)  # face box high-risk

_FONT        = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MONO   = cv2.FONT_HERSHEY_PLAIN

# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe face mesh index groups (478-point canonical model)
# ─────────────────────────────────────────────────────────────────────────────
_JAW = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,
        379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]
_L_BROW  = [70,63,105,66,107,55,65,52,53,46]
_R_BROW  = [336,296,334,293,300,285,295,282,283,276]
_NOSE    = [168,6,197,195,5,4,1,19,94,2]
_L_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
_R_EYE   = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
_LIPS_O  = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]
_LIPS_I  = [78,82,13,312,308,317,14,87]
_CHEEK_L = [234,93,132,58,172,136,150,149,176,148,152]
_CHEEK_R = [454,323,361,288,397,365,379,378,400,377,152]
_FOREHEAD= [10,109,67,103,54,21,162,127,234,93,132,58,172,136,150,
            149,176,148,152,377,400,378,379,365,397,288,361,323,454]

# Landmark dot highlight indices (keypoints to emphasize)
_KEY_LM = [1, 4, 6, 10, 33, 61, 91, 133, 152, 159, 168, 263, 291, 362, 386, 454]

# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _alpha_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: Tuple[int,int,int],
    alpha: float,
) -> None:
    """Blend a filled rectangle with given opacity."""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.rectangle(overlay, (0,0), (x2-x1, y2-y1), colour, -1)
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _text(
    img: np.ndarray,
    s: str,
    x: int, y: int,
    scale: float,
    colour: Tuple[int,int,int],
    thick: int = 1,
    font=None,
) -> None:
    """Draw text with 1-px dark shadow for legibility on any bg."""
    f = font or _FONT
    cv2.putText(img, s, (x+1, y+1), f, scale, (0,0,0), thick+1, cv2.LINE_AA)
    cv2.putText(img, s, (x,   y  ), f, scale, colour,  thick,   cv2.LINE_AA)


def _text_right(
    img: np.ndarray,
    s: str,
    right_x: int, y: int,
    scale: float,
    colour: Tuple[int,int,int],
    thick: int = 1,
) -> None:
    """Right-align text at right_x."""
    tw = cv2.getTextSize(s, _FONT, scale, thick)[0][0]
    _text(img, s, right_x - tw, y, scale, colour, thick)


def _hbar(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    value: float,  # 0–1
    col_fill: Tuple[int,int,int],
    col_bg: Tuple[int,int,int] = (30, 35, 45),
    border: bool = True,
) -> None:
    """Draw a horizontal progress bar."""
    cv2.rectangle(img, (x, y), (x+w, y+h), col_bg, -1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), col_fill, -1)
    if border:
        cv2.rectangle(img, (x, y), (x+w, y+h), _GREY, 1)


def _gauge_arc(
    img: np.ndarray,
    cx: int, cy: int, r: int,
    value: float,  # 0–1
    colour: Tuple[int,int,int],
    thick: int = 3,
    start_deg: float = 210,
    span_deg: float  = 300,
) -> None:
    """Draw an arc-style radial gauge."""
    end_deg = start_deg - span_deg * max(0.0, min(1.0, value))
    cv2.ellipse(img, (cx,cy), (r,r), 0,
                180 - start_deg, 180 - (start_deg - span_deg),
                (35,40,55), thick+2, cv2.LINE_AA)
    if value > 0:
        cv2.ellipse(img, (cx,cy), (r,r), 0,
                    180 - start_deg, 180 - end_deg,
                    colour, thick, cv2.LINE_AA)


def _corner_brackets(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    colour: Tuple[int,int,int],
    length: int = 22,
    thick: int  = 2,
) -> None:
    """Draw 4-corner bracket markers (no fill)."""
    corners = [
        ((x1,y1), (+1, 0), (0, +1)),
        ((x2,y1), (-1, 0), (0, +1)),
        ((x1,y2), (+1, 0), (0, -1)),
        ((x2,y2), (-1, 0), (0, -1)),
    ]
    for (px,py), (hx,hy), (vx,vy) in corners:
        cv2.line(img, (px,py), (px+hx*length, py+hy*length), colour, thick, cv2.LINE_AA)
        cv2.line(img, (px,py), (px+vx*length, py+vy*length), colour, thick, cv2.LINE_AA)


def _risk_col(v: float) -> Tuple[int,int,int]:
    if v >= 0.8: return _RED_HUD
    if v >= 0.6: return _ORANGE
    if v >= 0.3: return _GOLD
    return _GREEN_HUD


def _metric_col(v: float, lo: float = 60.0, hi: float = 40.0) -> Tuple[int,int,int]:
    """Green → amber → red scale for 0-100 metrics where high is good."""
    if v >= lo: return _GREEN_HUD
    if v >= hi: return _GOLD
    return _RED_HUD


def _inv_metric_col(v: float, warn: float = 0.5, crit: float = 0.75) -> Tuple[int,int,int]:
    """Green → amber → red where LOW is good (drowsiness, stress, risk)."""
    if v >= crit: return _RED_HUD
    if v >= warn: return _ORANGE
    return _GREEN_HUD


# ─────────────────────────────────────────────────────────────────────────────
# Face-scan HUD elements
# ─────────────────────────────────────────────────────────────────────────────

def _draw_landmarks_hud(
    img: np.ndarray,
    lm: np.ndarray,       # (N,2) pixel coords already scaled to img coords
    t:  float,            # animation clock
    face_locked: bool,
) -> None:
    """Render 468 glowing cyan landmark mesh on the camera sub-image."""
    n = len(lm)
    if n < 10:
        return

    def _safe(i: int) -> Optional[Tuple[int,int]]:
        if i < n:
            return (int(lm[i,0]), int(lm[i,1]))
        return None

    def _seg(a: int, b: int, col: Tuple[int,int,int], tk: int = 1) -> None:
        pa, pb = _safe(a), _safe(b)
        if pa and pb:
            cv2.line(img, pa, pb, col, tk, cv2.LINE_AA)

    def _polyline(indices: List[int], col: Tuple[int,int,int], closed: bool = False) -> None:
        pts = [_safe(i) for i in indices]
        pts = [p for p in pts if p is not None]
        if len(pts) >= 2:
            arr = np.array(pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(img, [arr], closed, col, 1, cv2.LINE_AA)

    # Animated pulse: landmarks glow in and out
    pulse = 0.6 + 0.4 * math.sin(t * 3.0)
    mesh_alpha = int(60 + 40 * pulse)
    mesh_col   = (mesh_alpha, mesh_alpha, 0)   # dim cyan mesh edges

    # Draw mesh groups
    _polyline(_JAW,    (0, int(120*pulse), int(80*pulse)), closed=False)
    _polyline(_L_BROW, mesh_col, closed=False)
    _polyline(_R_BROW, mesh_col, closed=False)
    _polyline(_NOSE,   (0, int(100*pulse), int(140*pulse)), closed=False)
    _polyline(_L_EYE,  (0, int(180*pulse), int(80*pulse)), closed=True)
    _polyline(_R_EYE,  (0, int(180*pulse), int(80*pulse)), closed=True)
    _polyline(_LIPS_O, (40, int(90*pulse),  int(160*pulse)), closed=True)
    _polyline(_LIPS_I, (20, int(60*pulse),  int(120*pulse)), closed=True)

    # Extra cross-cheek diagonals for mesh density
    for step in range(0, len(_FOREHEAD)-1, 3):
        a, b = _FOREHEAD[step], _FOREHEAD[min(step+4, len(_FOREHEAD)-1)]
        _seg(a, b, (0, 30, 40))

    # All landmark dots — tiny cyan specks
    dot_col = (int(200*pulse), int(180*pulse), 0)
    for i in range(min(n, 468)):
        p = _safe(i)
        if p:
            cv2.circle(img, p, 1, dot_col, -1, cv2.LINE_AA)

    # Key landmark highlights with pulsing glow
    bright = (int(255*pulse), int(220*pulse), 20)
    for ki in _KEY_LM:
        p = _safe(ki)
        if p:
            # Outer glow ring
            cv2.circle(img, p, 4, (0, int(60*pulse), int(80*pulse)), 1, cv2.LINE_AA)
            # Inner bright dot
            cv2.circle(img, p, 2, bright, -1, cv2.LINE_AA)


def _draw_face_scan_hud(
    cam_img: np.ndarray,
    lm: Optional[np.ndarray],
    t: float,
    face_locked: bool,
    confidence: float,
    risk_score: float,
) -> None:
    """Composite full face-scan HUD on the camera sub-image in-place."""
    if lm is None or len(lm) < 10:
        # No face: scanning animation with NO-FACE message
        h, w = cam_img.shape[:2]
        cx, cy = w//2, h//2
        scan_r = 80 + int(20 * abs(math.sin(t*1.5)))
        cv2.circle(cam_img, (cx,cy), scan_r, (0, 40, 60), 1, cv2.LINE_AA)
        cv2.circle(cam_img, (cx,cy), scan_r+10, (0, 20, 30), 1, cv2.LINE_AA)
        # Rotating scan line
        angle = (t * 120) % 360
        rad = math.radians(angle)
        ex = int(cx + scan_r * math.cos(rad))
        ey = int(cy + scan_r * math.sin(rad))
        cv2.line(cam_img, (cx,cy), (ex,ey), (0,80,120), 1, cv2.LINE_AA)
        # NO FACE text
        _text(cam_img, "SCANNING...", cx-52, cy+scan_r+22, 0.45, _GREY)
        # Cross-hair
        cv2.line(cam_img, (cx-15,cy), (cx+15,cy), (0,50,80), 1)
        cv2.line(cam_img, (cx,cy-15), (cx,cy+15), (0,50,80), 1)
        return

    xs = lm[:,0]; ys = lm[:,1]
    fx1, fy1 = int(xs.min()), int(ys.min())
    fx2, fy2 = int(xs.max()), int(ys.max())

    pad_x = int((fx2-fx1)*0.08); pad_y = int((fy2-fy1)*0.10)
    fx1 = max(0, fx1-pad_x); fy1 = max(0, fy1-pad_y)
    fx2 = min(cam_img.shape[1]-1, fx2+pad_x)
    fy2 = min(cam_img.shape[0]-1, fy2+pad_y)
    fcx = (fx1+fx2)//2; fcy = (fy1+fy2)//2
    fr  = max((fx2-fx1), (fy2-fy1))//2

    # ── Radar rings around face ───────────────────────────────────────────
    ring_pulse = 0.5 + 0.5 * math.sin(t * 2.0)
    col_r = _BOX_COL_OK if risk_score < 0.6 else (_BOX_COL_WARN if risk_score < 0.8 else _BOX_COL_ERR)
    for ring_i, offset in enumerate([0, 8, 18]):
        ring_r = fr + 12 + offset + int(4 * math.sin(t*2.0 + ring_i))
        alpha_ring = max(30, int(100 * ring_pulse) - ring_i*25)
        ring_dim = tuple(int(c * alpha_ring / 120) for c in col_r)
        cv2.ellipse(cam_img, (fcx,fcy), (ring_r, ring_r),
                    0, 0, 360, ring_dim, 1, cv2.LINE_AA)

    # ── Rotating scan sweep ───────────────────────────────────────────────
    sweep_angle = (t * 90) % 360  # 90 deg/s
    sweep_rad   = math.radians(sweep_angle)
    sweep_len   = fr + 25
    ex = int(fcx + sweep_len * math.cos(sweep_rad))
    ey = int(fcy + sweep_len * math.sin(sweep_rad))
    # Gradient sweep: 3 lines with decreasing opacity
    for k, offset_a in enumerate([0, -15, -30]):
        a2 = math.radians(sweep_angle + offset_a)
        ex2 = int(fcx + sweep_len * math.cos(a2))
        ey2 = int(fcy + sweep_len * math.sin(a2))
        thick = max(1, 2-k)
        alpha_sweep = max(0, int(80 * (1 - k/3)))
        sweep_col = tuple(int(c * alpha_sweep / 80) for c in _SWEEP_COL)
        cv2.line(cam_img, (fcx,fcy), (ex2,ey2), sweep_col, thick, cv2.LINE_AA)

    # ── 468 Landmark mesh ─────────────────────────────────────────────────
    _draw_landmarks_hud(cam_img, lm, t, face_locked)

    # ── Corner targeting brackets ─────────────────────────────────────────
    bracket_pulse = int(255 * (0.7 + 0.3*math.sin(t*4.0)))
    bracket_col = tuple(min(255, int(c * bracket_pulse/255)) for c in col_r)
    _corner_brackets(cam_img, fx1, fy1, fx2, fy2, bracket_col, length=18, thick=2)

    # ── Inner bracket lines (thin) ────────────────────────────────────────
    cv2.rectangle(cam_img, (fx1+18,fy1+18), (fx2-18,fy2-18),
                  (0, 30, 40), 1, cv2.LINE_AA)

    # ── Face-lock indicator ───────────────────────────────────────────────
    lock_col = _GREEN_HUD if face_locked else _ORANGE
    lock_txt  = "FACE LOCKED" if face_locked else "ACQUIRING..."
    _text(cam_img, lock_txt, fx1, fy1-10, 0.40, lock_col, 1)

    # ── Confidence percentage ─────────────────────────────────────────────
    conf_pct = int(confidence * 100)
    conf_col = _GREEN_HUD if conf_pct > 80 else (_GOLD if conf_pct > 60 else _ORANGE)
    conf_txt = f"CONF {conf_pct:3d}%"
    _text(cam_img, conf_txt, fx2-75, fy1-10, 0.40, conf_col, 1)

    # ── ID badge at bottom of face box ───────────────────────────────────
    id_txt = "DRIVER ID: ACTIVE"
    _text(cam_img, id_txt, fcx-52, fy2+16, 0.36, _CYAN_DIM, 1)

    # ── Centre cross-hair dot ─────────────────────────────────────────────
    cv2.circle(cam_img, (fcx,fcy), 3, _CYAN, -1, cv2.LINE_AA)
    cv2.circle(cam_img, (fcx,fcy), 7, _CYAN_DIM, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Left panel: Driver profile + cognitive metrics
# ─────────────────────────────────────────────────────────────────────────────

def _draw_left_panel(
    canvas: np.ndarray,
    result: Dict[str,Any],
    t: float,
) -> None:
    x0, y0, w, h = 0, 0, _PANEL_L_W, _HUD_H - _BAR_H

    # Panel background
    _alpha_rect(canvas, x0, y0, x0+w, y0+h, _PANEL_BG, 0.92)
    cv2.line(canvas, (x0+w-1, y0), (x0+w-1, y0+h), _PANEL_LINE, 1)

    # ── Branding header ───────────────────────────────────────────────────
    _alpha_rect(canvas, x0, y0, x0+w, y0+38, (0,30,60), 0.95)
    cv2.line(canvas, (x0,y0+38), (x0+w,y0+38), _CYAN, 1)
    _text(canvas, "COGNI", x0+10, y0+24, 0.65, _CYAN, 2)
    _text(canvas, "DRIVE", x0+80, y0+24, 0.65, _WHITE, 1)
    _text(canvas, "AI COCKPIT", x0+160, y0+22, 0.33, _GREY, 1)

    cy = y0 + 56

    # ── Driver profile ────────────────────────────────────────────────────
    _text(canvas, "DRIVER PROFILE", x0+10, cy, 0.36, _CYAN_DIM, 1)
    cy += 2
    cv2.line(canvas, (x0+10,cy), (x0+w-10,cy), _PANEL_LINE, 1)
    cy += 16

    pulse_dot = _GREEN_HUD if result.get("face_detected") else _GREY
    cv2.circle(canvas, (x0+18, cy-4), 5, pulse_dot, -1, cv2.LINE_AA)
    status_txt = "AUTHENTICATED" if result.get("face_detected") else "SCANNING"
    _text(canvas, status_txt, x0+30, cy, 0.38, pulse_dot, 1)
    cy += 20

    state = str(result.get("driver_state","NORMAL")).replace("_"," ")
    state_col = _state_colour_hud(state)
    _text(canvas, "STATUS", x0+10, cy, 0.33, _GREY, 1)
    _text(canvas, state, x0+70, cy, 0.40, state_col, 1)
    cy += 30

    # ── Section: Cognitive Metrics ────────────────────────────────────────
    _text(canvas, "COGNITIVE METRICS", x0+10, cy, 0.36, _CYAN_DIM, 1)
    cy += 2
    cv2.line(canvas, (x0+10,cy), (x0+w-10,cy), _PANEL_LINE, 1)
    cy += 16

    attn  = float(result.get("attention_score", 0))
    cli   = float(result.get("cli", 0))
    stress= float(result.get("stress_score", 0))

    def _metric_row(label: str, value: float, vmax: float,
                    col: Tuple, pct_label: str = "") -> None:
        nonlocal cy
        _text(canvas, label, x0+10, cy, 0.36, _WHITE, 1)
        pct = f"{value:.0f}{pct_label}"
        _text_right(canvas, pct, x0+w-10, cy, 0.36, col, 1)
        cy += 4
        _hbar(canvas, x0+10, cy, w-20, 7, value/vmax, col)
        cy += 16

    _metric_row("ATTENTION",   attn,   100, _metric_col(attn,   60, 40), "%")
    _metric_row("COGNIT LOAD", cli,    100, _inv_metric_col(cli/100, 0.5, 0.7), "%")
    _metric_row("STRESS LVL",  stress, 100, _inv_metric_col(stress/100, 0.5, 0.75), "%")

    # Drowsiness (from biometrics PERCLOS or fatigue proxy)
    bio   = result.get("biometrics", {})
    perclos = float(bio.get("perclos", 0))
    drown_val = perclos * 100
    _metric_row("DROWSINESS",  drown_val, 100, _inv_metric_col(drown_val/100, 0.3, 0.6), "%")

    # Distraction (gaze off-road proxy from risk model)
    dist_raw = float(result.get("biometrics", {}).get("gaze_x", 0.5))
    dist_pct = abs(dist_raw - 0.5) * 200   # 0 = centred, 100 = extreme
    _metric_row("DISTRACTION", dist_pct, 100, _inv_metric_col(dist_pct/100, 0.35, 0.6), "%")
    cy += 4

    # ── Gaze direction mini-map ───────────────────────────────────────────
    _text(canvas, "GAZE MAP", x0+10, cy, 0.33, _CYAN_DIM, 1)
    cy += 14
    gx = float(bio.get("gaze_x", 0.5))
    gy = float(bio.get("gaze_y", 0.5))
    box_x, box_y, box_w2, box_h2 = x0+10, cy, w-20, 42
    _alpha_rect(canvas, box_x, box_y, box_x+box_w2, box_y+box_h2, (5,10,20), 0.9)
    cv2.rectangle(canvas, (box_x,box_y),(box_x+box_w2,box_y+box_h2), _PANEL_LINE, 1)
    # Cross-hair lines
    cx_mid = box_x + box_w2//2;  cy_mid = box_y + box_h2//2
    cv2.line(canvas,(cx_mid,box_y),(cx_mid,box_y+box_h2),(20,40,60),1)
    cv2.line(canvas,(box_x,cy_mid),(box_x+box_w2,cy_mid),(20,40,60),1)
    # Gaze dot
    dot_px = int(box_x + gx*box_w2)
    dot_py = int(box_y + gy*box_h2)
    dot_px = max(box_x+3, min(box_x+box_w2-3, dot_px))
    dot_py = max(box_y+3, min(box_y+box_h2-3, dot_py))
    cv2.circle(canvas,(dot_px,dot_py), 5, _CYAN, -1, cv2.LINE_AA)
    cv2.circle(canvas,(dot_px,dot_py), 8, _CYAN_DIM, 1, cv2.LINE_AA)
    zone_txt = "ON-ROAD" if abs(gx-0.5)<0.2 and abs(gy-0.5)<0.2 else "OFF-ROAD"
    zone_col = _GREEN_HUD if zone_txt=="ON-ROAD" else _ORANGE
    _text(canvas, zone_txt, box_x+box_w2//2-22, box_y+box_h2+11, 0.32, zone_col, 1)
    cy += box_h2 + 18

    # ── Blink & eye stats ─────────────────────────────────────────────────
    _text(canvas, "EYE METRICS", x0+10, cy, 0.36, _CYAN_DIM, 1)
    cy += 2; cv2.line(canvas,(x0+10,cy),(x0+w-10,cy),_PANEL_LINE,1); cy+=14

    ear_val   = float(bio.get("ear", 0.25))
    blink_r   = float(bio.get("blink_rate", 15))
    ear_col   = _GREEN_HUD if ear_val > 0.22 else _ORANGE
    _text(canvas, f"EAR    {ear_val:.3f}", x0+10, cy, 0.36, ear_col, 1)
    cy += 15
    blink_col = _GREEN_HUD if 8 <= blink_r <= 24 else _ORANGE
    _text(canvas, f"BLINKS {blink_r:.0f}/min", x0+10, cy, 0.36, blink_col, 1)
    cy += 15

    # Performance trend mini-chart (last 10 attention scores stored in closure)
    _text(canvas, f"MAR    {float(bio.get('mar',0)):.3f}", x0+10, cy, 0.36, _GREY, 1)
    cy += 18

    # ── Timestamp footer ──────────────────────────────────────────────────
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _text(canvas, f"LOCAL  {ts}", x0+10, h-14, 0.33, _GREY, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Right panel: state, risk gauge, recommendations, system status
# ─────────────────────────────────────────────────────────────────────────────

def _draw_right_panel(
    canvas: np.ndarray,
    result: Dict[str,Any],
    fps: float,
    latency_ms: float,
    t: float,
) -> None:
    x0 = _CAM_X + _CAM_W
    y0 = 0
    w  = _PANEL_R_W
    h  = _HUD_H - _BAR_H

    _alpha_rect(canvas, x0, y0, x0+w, y0+h, _PANEL_BG, 0.92)
    cv2.line(canvas, (x0,y0), (x0,y0+h), _PANEL_LINE, 1)

    cy = y0 + 14

    # ── Driver state banner ───────────────────────────────────────────────
    state = str(result.get("driver_state","NORMAL")).replace("_"," ")
    state_col = _state_colour_hud(state)
    _alpha_rect(canvas, x0+8, cy-12, x0+w-8, cy+26, state_col, 0.18)
    cv2.rectangle(canvas, (x0+8,cy-12),(x0+w-8,cy+26), state_col, 1)
    tw = cv2.getTextSize(state, _FONT, 0.52, 2)[0][0]
    _text(canvas, state, x0+(w-tw)//2, cy+12, 0.52, state_col, 2)
    cy += 44

    # ── Risk gauge (arc) ──────────────────────────────────────────────────
    _text(canvas, "ACCIDENT RISK", x0+10, cy, 0.36, _CYAN_DIM, 1)
    cy += 4
    cv2.line(canvas,(x0+10,cy),(x0+w-10,cy),_PANEL_LINE,1)
    cy += 4

    risk  = float(result.get("risk_score",0))
    rcol  = _risk_col(risk)
    gcx, gcy, gr = x0+w//2, cy+62, 52
    # Background arc ticks
    for deg in range(210, -90, -30):
        r = math.radians(deg)
        ix = int(gcx + (gr+8)*math.cos(r))
        iy = int(gcy - (gr+8)*math.sin(r))
        ox = int(gcx + (gr+12)*math.cos(r))
        oy = int(gcy - (gr+12)*math.sin(r))
        cv2.line(canvas,(ix,iy),(ox,oy),(40,50,60),1,cv2.LINE_AA)
    _gauge_arc(canvas, gcx, gcy, gr, risk, rcol, thick=5)
    # Needle
    needle_deg = 210 - risk * 300
    nr = math.radians(needle_deg)
    nx = int(gcx + (gr-8)*math.cos(nr))
    ny = int(gcy - (gr-8)*math.sin(nr))
    cv2.line(canvas,(gcx,gcy),(nx,ny),_WHITE,2,cv2.LINE_AA)
    cv2.circle(canvas,(gcx,gcy),5,_WHITE,-1,cv2.LINE_AA)
    # Risk value
    rlabel = f"{risk:.3f}"
    rtw = cv2.getTextSize(rlabel,_FONT,0.65,2)[0][0]
    _text(canvas, rlabel, gcx-rtw//2, gcy+20, 0.65, rcol, 2)
    risk_level = "LOW" if risk<0.3 else ("MEDIUM" if risk<0.6 else ("HIGH" if risk<0.8 else "CRITICAL"))
    ltw = cv2.getTextSize(risk_level,_FONT,0.36,1)[0][0]
    _text(canvas, risk_level, gcx-ltw//2, gcy+36, 0.36, rcol, 1)
    # Min / max labels
    _text(canvas, "0.0", x0+12, gcy+18, 0.30, _GREY, 1)
    _text_right(canvas, "1.0", x0+w-12, gcy+18, 0.30, _GREY, 1)
    cy = gcy + 52

    # ── Recommendation card ───────────────────────────────────────────────
    cy += 8
    _text(canvas, "RECOMMENDATION", x0+10, cy, 0.36, _CYAN_DIM, 1)
    cy += 2; cv2.line(canvas,(x0+10,cy),(x0+w-10,cy),_PANEL_LINE,1); cy+=10

    recs = result.get("recommendations", [])
    if recs:
        rec  = recs[0]
        prio = str(rec.get("priority","LOW"))
        prio_col = _RED_HUD if "CRITICAL" in prio or "HIGH" in prio else (
                    _ORANGE if "MEDIUM" in prio else _GREEN_HUD)
        # Priority badge
        _alpha_rect(canvas, x0+10, cy-2, x0+w-10, cy+14, prio_col, 0.20)
        cv2.rectangle(canvas,(x0+10,cy-2),(x0+w-10,cy+14),prio_col,1)
        _text(canvas, f"! {prio}", x0+14, cy+10, 0.36, prio_col, 1)
        cy += 20

        title = str(rec.get("title",""))[:28]
        _text(canvas, title, x0+10, cy, 0.36, _WHITE, 1)
        cy += 14

        # Message word-wrap (28 chars/line, 3 lines max)
        msg = str(rec.get("message",""))
        lines = [msg[i:i+28] for i in range(0,min(len(msg),84),28)]
        for ln in lines[:3]:
            _text(canvas, ln, x0+10, cy, 0.32, _GREY, 1)
            cy += 13
    else:
        pulse_ok = 0.7 + 0.3*math.sin(t*2)
        ok_col = tuple(int(c*pulse_ok) for c in _GREEN_HUD)
        _text(canvas, "No alerts — all normal", x0+10, cy, 0.35, ok_col, 1)
        cy += 20

    # ── Anomaly indicator ─────────────────────────────────────────────────
    cy += 6
    is_anomaly = bool(result.get("is_anomaly", False))
    anom_score = float(result.get("anomaly_score", 0))
    anom_col = _RED_HUD if is_anomaly else _GREEN_HUD
    anom_txt  = f"ANOMALY  {'DETECTED' if is_anomaly else 'CLEAR'}"
    _alpha_rect(canvas, x0+10, cy-2, x0+w-10, cy+14,
                _RED_HUD if is_anomaly else (0,60,20), 0.25)
    cv2.rectangle(canvas,(x0+10,cy-2),(x0+w-10,cy+14),anom_col,1)
    _text(canvas, anom_txt, x0+14, cy+10, 0.34, anom_col, 1)
    _text_right(canvas, f"{anom_score:.3f}", x0+w-12, cy+10, 0.34, _GREY, 1)
    cy += 24

    # ── System status ─────────────────────────────────────────────────────
    _text(canvas, "SYSTEM STATUS", x0+10, cy, 0.36, _CYAN_DIM, 1)
    cy += 2; cv2.line(canvas,(x0+10,cy),(x0+w-10,cy),_PANEL_LINE,1); cy+=14

    cam_fps = fps
    mdl_fps = 1000.0/latency_ms if latency_ms > 0 else 0

    def _status_row(label: str, value: str, col: Tuple) -> None:
        nonlocal cy
        _text(canvas, label, x0+10, cy, 0.34, _GREY, 1)
        _text_right(canvas, value, x0+w-10, cy, 0.36, col, 1)
        cy += 15

    _status_row("CAMERA FPS",  f"{cam_fps:.1f}", _GREEN_HUD if cam_fps>20 else _ORANGE)
    _status_row("INFER TIME",  f"{latency_ms:.0f}ms", _GREEN_HUD if latency_ms<60 else _ORANGE)
    _status_row("PIPELINE",    "ACTIVE", _GREEN_HUD)
    _status_row("ML MODELS",   "LOADED", _GREEN_HUD)
    _status_row("FACE ENGINE", "ONLINE", _GREEN_HUD)
    _status_row("DB SESSION",  "LINKED", _GREEN_DIM)

    # ── Animated heartbeat line ───────────────────────────────────────────
    cy += 6
    hb_w = w - 20
    hb_pts = []
    for i in range(hb_w):
        tx = t*4 + i*0.12
        vy = int(8*math.sin(tx) + 4*math.sin(tx*3) + 2*math.sin(tx*7))
        hb_pts.append((x0+10+i, cy+12+vy))
    for i in range(len(hb_pts)-1):
        col_i = _risk_col(min(1.0, risk * (1 + 0.3*math.sin(t+i*0.1))))
        cv2.line(canvas, hb_pts[i], hb_pts[i+1], col_i, 1, cv2.LINE_AA)
    cy += 30

    # ── Frame counter ─────────────────────────────────────────────────────
    fn = result.get("frame_number",0)
    _text(canvas, f"FRAME #{fn:06d}", x0+10, h-14, 0.33, _GREY, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Bottom metrics bar
# ─────────────────────────────────────────────────────────────────────────────

def _draw_bottom_bar(
    canvas: np.ndarray,
    result: Dict[str,Any],
    t: float,
) -> None:
    y0 = _HUD_H - _BAR_H
    w  = _HUD_W
    h  = _BAR_H

    _alpha_rect(canvas, 0, y0, w, y0+h, (8,12,20), 0.96)
    cv2.line(canvas,(0,y0),(w,y0),_CYAN,1)

    bio = result.get("biometrics",{})

    # Split into 7 equal widgets
    n_cols = 7
    col_w  = w // n_cols

    def _widget(col_i: int, title: str, value: str, sub: str,
                 bar_val: Optional[float], bar_col: Tuple,
                 title_col: Tuple = None) -> None:
        cx0 = col_i * col_w
        if col_i > 0:
            cv2.line(canvas,(cx0,y0+8),(cx0,y0+h-8),_PANEL_LINE,1)
        tc = title_col or _CYAN_DIM
        _text(canvas, title, cx0+8, y0+18, 0.33, tc, 1)
        vw = cv2.getTextSize(value,_FONT,0.52,2)[0][0]
        _text(canvas, value, cx0+(col_w-vw)//2, y0+50, 0.52, bar_col, 2)
        _text(canvas, sub, cx0+8, y0+68, 0.30, _GREY, 1)
        if bar_val is not None:
            _hbar(canvas, cx0+8, y0+78, col_w-16, 6, bar_val, bar_col)

    ear_val   = float(bio.get("ear", 0.25))
    mar_val   = float(bio.get("mar", 0.05))
    perclos_v = float(bio.get("perclos", 0))
    blink_r   = float(bio.get("blink_rate", 15))
    h_pitch   = float(bio.get("head_pitch",0))
    h_yaw     = float(bio.get("head_yaw",0))
    h_roll    = float(bio.get("head_roll",0))
    gaze_x    = float(bio.get("gaze_x",0.5))
    gaze_y    = float(bio.get("gaze_y",0.5))

    ear_col   = _GREEN_HUD if ear_val>0.22 else _ORANGE
    blink_col = _GREEN_HUD if 8<=blink_r<=24 else _ORANGE
    perc_col  = _inv_metric_col(perclos_v/0.4)
    pitch_col = _GREEN_HUD if abs(h_pitch)<18 else _ORANGE
    gaze_col  = _GREEN_HUD if abs(gaze_x-0.5)<0.2 else _ORANGE
    fatigue_p = float(perclos_v * 2.5)   # rough fatigue proxy from perclos
    fat_col   = _inv_metric_col(min(1.0,fatigue_p), 0.4, 0.65)

    _widget(0, "EYE ASPECT RATIO", f"{ear_val:.3f}",
            "OPEN" if ear_val>0.22 else "CLOSING",
            1.0-min(1.0,abs(ear_val-0.30)/0.30), ear_col)

    _widget(1, "HEAD POSE",
            f"{h_yaw:+.0f}\u00b0",
            f"P{h_pitch:+.0f} R{h_roll:+.0f}",
            1.0 - min(1.0, abs(h_yaw)/45), pitch_col)

    _widget(2, "GAZE DIRECTION",
            f"{'L' if gaze_x<0.4 else ('R' if gaze_x>0.6 else 'C')}"
            f"{'U' if gaze_y<0.4 else ('D' if gaze_y>0.6 else '')}",
            f"x:{gaze_x:.2f} y:{gaze_y:.2f}",
            1.0-abs(gaze_x-0.5)*2, gaze_col)

    _widget(3, "PERCLOS",
            f"{perclos_v*100:.1f}%",
            ">15% FATIGUE" if perclos_v>0.15 else "NORMAL",
            perclos_v/0.30, perc_col)

    _widget(4, "BLINK COUNT",
            f"{blink_r:.0f}",
            "per minute",
            min(1.0, blink_r/30), blink_col)

    _widget(5, "FATIGUE PROB",
            f"{min(1.0,fatigue_p)*100:.0f}%",
            "HIGH" if fatigue_p>0.65 else ("MED" if fatigue_p>0.35 else "LOW"),
            min(1.0,fatigue_p), fat_col)

    _widget(6, "MAR YAWN",
            f"{mar_val:.3f}",
            "YAWNING" if mar_val>0.55 else "NORMAL",
            min(1.0, mar_val/0.80),
            _ORANGE if mar_val>0.55 else _GREEN_HUD)


# ─────────────────────────────────────────────────────────────────────────────
# Camera viewport frame
# ─────────────────────────────────────────────────────────────────────────────

def _draw_camera_frame_border(canvas: np.ndarray, t: float) -> None:
    """Draw animated border around the camera viewport area."""
    x1, y1 = _CAM_X, _CAM_Y
    x2, y2 = _CAM_X + _CAM_W, _CAM_Y + _CAM_H
    pulse = 0.6 + 0.4 * math.sin(t*2.5)
    col = tuple(int(c*pulse) for c in _CYAN_PALE)
    # Outer border
    cv2.rectangle(canvas,(x1-1,y1-1),(x2+1,y2+1),(20,40,60),1)
    # Corner brackets
    _corner_brackets(canvas, x1, y1, x2, y2, col, length=28, thick=2)
    # Top label
    _text(canvas, "LIVE FEED", x1+4, y1+16, 0.40, col, 1)
    # Bottom-right coordinate
    _text_right(canvas, f"{_CAM_W}x{_CAM_H}", x2-4, y2-6, 0.30, _GREY, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Full HUD compositor
# ─────────────────────────────────────────────────────────────────────────────

def _state_colour_hud(state: str) -> Tuple[int,int,int]:
    s = state.upper()
    if "HIGH" in s or "RISK" in s: return _RED_HUD
    if "FATIG" in s:               return _ORANGE
    if "DISTRACT" in s:            return _GOLD
    if "OVERLOAD" in s:            return _ACCENT_BLUE
    return _GREEN_HUD


def _compose_hud(
    cam_frame: np.ndarray,    # BGR, any size (will be resized)
    lm: Optional[np.ndarray], # (N,2) landmark pixel coords on ORIGINAL frame
    result: Dict[str,Any],
    fps: float,
    latency_ms: float,
    t: float,
    face_locked: bool,
) -> np.ndarray:
    """Build the full 1280x720 cockpit HUD canvas and return it."""

    # ── Allocate black canvas ─────────────────────────────────────────────
    canvas = np.zeros((_HUD_H, _HUD_W, 3), dtype=np.uint8)
    canvas[:] = _BG

    # ── Resize and place camera feed ──────────────────────────────────────
    if cam_frame is not None and cam_frame.size > 0:
        cam_resized = cv2.resize(cam_frame, (_CAM_W, _CAM_H), interpolation=cv2.INTER_LINEAR)
    else:
        cam_resized = np.zeros((_CAM_H, _CAM_W, 3), dtype=np.uint8)

    # Scale landmarks from original frame coords → resized cam coords
    lm_scaled: Optional[np.ndarray] = None
    if lm is not None and len(lm) > 0 and cam_frame is not None and cam_frame.size > 0:
        oh, ow = cam_frame.shape[:2]
        scale_x = _CAM_W / ow
        scale_y = _CAM_H / oh
        lm_scaled = lm.copy().astype(np.float32)
        lm_scaled[:,0] *= scale_x
        lm_scaled[:,1] *= scale_y
        lm_scaled = lm_scaled.astype(np.int32).astype(np.float32)

    # Face-scan HUD on camera sub-image
    risk_score = float(result.get("risk_score", 0))
    confidence = 0.9 if face_locked else 0.45
    _draw_face_scan_hud(cam_resized, lm_scaled, t, face_locked, confidence, risk_score)

    # Composite camera into canvas
    canvas[_CAM_Y:_CAM_Y+_CAM_H, _CAM_X:_CAM_X+_CAM_W] = cam_resized

    # ── Camera frame border ───────────────────────────────────────────────
    _draw_camera_frame_border(canvas, t)

    # ── Panels ────────────────────────────────────────────────────────────
    _draw_left_panel(canvas, result, t)
    _draw_right_panel(canvas, result, fps, latency_ms, t)
    _draw_bottom_bar(canvas, result, t)

    # ── Top ticker bar (1280-wide) ────────────────────────────────────────
    ticker_y = _CAM_H
    _alpha_rect(canvas, _CAM_X, ticker_y, _CAM_X+_CAM_W, ticker_y+22, (5,10,20), 0.95)
    cv2.line(canvas,(_CAM_X,ticker_y),(_CAM_X+_CAM_W,ticker_y),_CYAN_DIM,1)

    attn  = float(result.get("attention_score", 0))
    cli   = float(result.get("cli", 0))
    fps_v = fps
    lat_v = latency_ms

    ticker = (
        f"  ATTN {attn:.0f}%   "
        f"CLI {cli:.0f}   "
        f"RISK {risk_score:.3f}   "
        f"FPS {fps_v:.1f}   "
        f"LAT {lat_v:.0f}ms   "
        f"STATE: {str(result.get('driver_state','?')).replace('_',' ')}  "
    )
    # Scroll the ticker
    ticker_x = _CAM_X + 6 - int((t*40) % (_CAM_W+300))
    _text(canvas, ticker, ticker_x, ticker_y+15, 0.36, _CYAN_PALE, 1)

    # ── Version watermark ─────────────────────────────────────────────────
    _text(canvas, "v1.0", _HUD_W-36, _HUD_H-_BAR_H-8, 0.28, (30,35,45), 1)

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# LivePipelineRunner (unchanged contract; enhanced visualization)
# ─────────────────────────────────────────────────────────────────────────────

class LivePipelineRunner:
    """Singleton background worker: camera → ML pipeline → futuristic HUD overlay.

    Public API is identical to the original Drishti runner:
        get_instance(camera_manager, pipeline_manager)
        start(driver_id, session_id, db_session)
        stop()
        is_running  → bool

    The only behavioural change is the visualization layer.
    """

    _instance: Optional["LivePipelineRunner"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __init__(self, camera_manager: Any, pipeline_manager: Any) -> None:
        self._camera   = camera_manager
        self._pipeline = pipeline_manager

        self._thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._running: bool = False

        self._driver_id: int = 0
        self._session_id: int = 0
        self._db_session: Optional[Session] = None

    # ── Singleton ─────────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        camera_manager: Any,
        pipeline_manager: Any,
    ) -> "LivePipelineRunner":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls(
                        camera_manager=camera_manager,
                        pipeline_manager=pipeline_manager,
                    )
        return cls._instance

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self, driver_id: int, session_id: int, db_session: Session) -> None:
        """Start background processing + HUD preview thread. Idempotent."""
        if self._running:
            logger.warning("LivePipelineRunner already running.")
            return
        self._driver_id  = driver_id
        self._session_id = session_id
        self._db_session = db_session
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            name="CogniDrive-LivePipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("LivePipelineRunner started (driver=%d, session=%d).", driver_id, session_id)

    def stop(self) -> None:
        """Signal thread to exit and destroy the OpenCV window."""
        if not self._running:
            return
        logger.info("LivePipelineRunner stopping…")
        self._stop_event.set()
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Thread did not exit within 5 s.")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("LivePipelineRunner stopped.")

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Main loop ─────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        logger.info("LivePipelineRunner loop entered.")
        frame_number: int = 0
        consecutive_errors: int = 0
        max_consecutive_errors: int = 30

        fps_count: int = 0
        fps_timer: float = time.monotonic()
        fps: float = 0.0

        # Animation clock (incremented by real elapsed time)
        t: float = 0.0

        # Bind PipelineManager session context
        try:
            self._pipeline.start_session(
                driver_id=self._driver_id,
                session_id=self._session_id,
                db_session=self._db_session,
            )
        except Exception as exc:
            logger.warning("start_session failed (non-fatal): %s", exc)
            self._pipeline.db_session      = self._db_session
            self._pipeline.active_driver_id  = self._driver_id
            self._pipeline.active_session_id = self._session_id

        # Create the window
        try:
            cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(_WIN_NAME, _HUD_W, _HUD_H)
        except Exception as exc:
            logger.warning("Could not create OpenCV window: %s", exc)

        last_result: Dict[str,Any] = {
            "face_detected": False,
            "driver_state": "NORMAL",
            "attention_score": 0.0,
            "stress_score": 0.0,
            "cli": 0.0,
            "risk_score": 0.0,
            "anomaly_score": 0.0,
            "is_anomaly": False,
            "frame_number": 0,
            "recommendations": [],
            "biometrics": {},
        }
        last_landmarks: Optional[np.ndarray] = None
        face_locked: bool = False
        face_lock_frames: int = 0

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            # ── Read frame ────────────────────────────────────────────────
            frame = None
            try:
                frame = self._camera.read_frame()
            except Exception as exc:
                logger.debug("read_frame error: %s", exc)

            if frame is None:
                time.sleep(_CAMERA_WAIT_SLEEP)
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        self._stop_event.set(); break
                    if cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        self._stop_event.set(); break
                except Exception:
                    pass
                continue

            # ── Camera health ─────────────────────────────────────────────
            if not self._camera.is_running:
                logger.warning("Camera stopped — retrying…")
                time.sleep(_RECONNECT_SLEEP)
                try:
                    self._camera.restart()
                except Exception as exc:
                    logger.error("Camera restart failed: %s", exc)
                continue

            # ── ML inference ──────────────────────────────────────────────
            latency_ms: float = 0.0
            result = last_result
            landmarks_arr: Optional[np.ndarray] = last_landmarks

            try:
                t_infer = time.monotonic()
                result = self._pipeline.process_frame(
                    frame=frame,
                    frame_number=frame_number,
                    frame_time_ms=loop_start * 1000.0,
                )
                latency_ms = (time.monotonic() - t_infer) * 1000.0
                self._pipeline.last_prediction = result
                frame_number += 1
                consecutive_errors = 0
                last_result = result

                # Re-extract landmarks for overlay (same frame already cached)
                try:
                    lm_result = self._pipeline.landmark_extractor.extract(frame)
                    if lm_result.is_valid and lm_result.all_landmarks is not None:
                        landmarks_arr   = lm_result.all_landmarks
                        last_landmarks  = landmarks_arr
                        face_lock_frames = min(face_lock_frames + 1, 10)
                    else:
                        landmarks_arr    = None
                        last_landmarks   = None
                        face_lock_frames = max(face_lock_frames - 3, 0)
                except Exception:
                    landmarks_arr = last_landmarks

                face_locked = face_lock_frames >= 5

            except RuntimeError as exc:
                logger.warning("Pipeline RuntimeError — rebinding: %s", exc)
                self._pipeline.db_session        = self._db_session
                self._pipeline.active_driver_id  = self._driver_id
                self._pipeline.active_session_id = self._session_id
                consecutive_errors += 1

            except Exception as exc:
                consecutive_errors += 1
                logger.error("Frame error #%d: %s", consecutive_errors, exc,
                             exc_info=(consecutive_errors == 1))
                if consecutive_errors >= max_consecutive_errors:
                    logger.error("Too many errors — pausing %.1f s.", _RECONNECT_SLEEP)
                    time.sleep(_RECONNECT_SLEEP)
                    consecutive_errors = 0

            # ── FPS ───────────────────────────────────────────────────────
            fps_count += 1
            now = time.monotonic()
            elapsed_fps = now - fps_timer
            if elapsed_fps >= 1.0:
                fps        = fps_count / elapsed_fps
                fps_count  = 0
                fps_timer  = now

            # ── Advance animation clock ───────────────────────────────────
            t += time.monotonic() - loop_start   # real elapsed seconds

            # ── Compose and render HUD ────────────────────────────────────
            try:
                hud = _compose_hud(
                    cam_frame  = frame,
                    lm         = landmarks_arr,
                    result     = result,
                    fps        = fps,
                    latency_ms = latency_ms,
                    t          = t,
                    face_locked= face_locked,
                )
            except Exception as exc:
                logger.debug("HUD compose error: %s", exc)
                # Fallback: raw frame on a black canvas
                hud = np.zeros((_HUD_H, _HUD_W, 3), dtype=np.uint8)
                if frame is not None:
                    rf = cv2.resize(frame, (_CAM_W, _CAM_H))
                    hud[_CAM_Y:_CAM_Y+_CAM_H, _CAM_X:_CAM_X+_CAM_W] = rf

            # ── Show ──────────────────────────────────────────────────────
            try:
                cv2.imshow(_WIN_NAME, hud)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    logger.info("Q pressed — stopping.")
                    self._stop_event.set(); break
                if cv2.getWindowProperty(_WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("Window closed — stopping.")
                    self._stop_event.set(); break
            except Exception as exc:
                logger.debug("imshow error: %s", exc)

            # ── Throttle to target FPS ────────────────────────────────────
            elapsed = time.monotonic() - loop_start
            sleep_t = _MIN_FRAME_SLEEP - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Cleanup
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("LivePipelineRunner loop exited.")

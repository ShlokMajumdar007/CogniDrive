# CogniDrive HUD — Change Log

## [HUD 1.0] — Futuristic AI Cockpit Visualization

### What changed
Only `live_pipeline_runner.py` was modified.
All ML pipeline, prediction logic, API routes, database, and
CameraManager code is **untouched**.

---

### Architecture preserved
| Component | Status |
|---|---|
| `CameraManager` | Unchanged — sole camera source |
| `PipelineManager.process_frame()` | Unchanged — sole inference path |
| `PipelineManager.last_prediction` | Unchanged — SSE dashboard still works |
| `LandmarkExtractor` | Unchanged — re-used for overlay coords |
| FastAPI routes / DB | Unchanged |
| `LivePipelineRunner` public API | Unchanged — `get_instance / start / stop / is_running` |

---

### Issues fixed in original `_draw_hud` / `_draw_landmarks` / `_draw_face_box`
1. **Text rendering** — old code had a redundant `tw, _ = ...` line that
   computed getTextSize twice; removed.
2. **face_box sizing** — old 5 % padding now 8 % (x) / 10 % (y) for a
   less tight fit.
3. **Colour scheme** — all colours moved to named constants so they are
   trivially theme-able.

---

### New visualization features

#### Face Scan HUD (center camera viewport)
- **468 MediaPipe landmark mesh** — all points rendered as 1 px cyan specks;
  key anatomy groups (jaw, brows, nose, eyes, lips) drawn as polylines with
  region-specific colours.
- **16 keypoint highlights** — pulsing glow rings + bright dots at anatomical
  landmarks (nose tip, eye corners, lip corners, etc.).
- **Animated mesh pulse** — landmark brightness oscillates at 3 Hz via
  `sin(t * 3)`.
- **Rotating scan sweep** — 3-layer gradient sweep line revolves at 90 °/s
  around the face centre.
- **Radar rings** — 3 concentric animated ellipses with risk-driven colour
  (green → orange → red).
- **Corner targeting brackets** — 4-corner L-shaped brackets, colour pulsing
  at 4 Hz; risk-driven colour.
- **Face lock indicator** — text reads "FACE LOCKED" (green) after 5
  consecutive valid frames, "ACQUIRING…" (orange) otherwise; uses a
  hysteresis counter (+1 per lock frame, −3 per miss).
- **Confidence percentage** — `CONF xx%` displayed; 90 % when locked, 45 %
  when acquiring.
- **No-face scanning mode** — when no landmarks available: rotating arc +
  crosshair + "SCANNING…" text.

#### Left Panel — Driver Profile + Cognitive Metrics
- **Branding header** — "COGNI DRIVE AI COCKPIT" with cyan accent.
- **Authentication dot** — green pulse when face detected, grey when not.
- **5 metric rows** with animated horizontal bars:
  Attention, Cognitive Load, Stress Level, Drowsiness (PERCLOS-derived),
  Distraction (gaze deviation).
- **Gaze mini-map** — 280×42 px grid with live cyan dot tracking
  horizontal/vertical gaze ratio; "ON-ROAD" / "OFF-ROAD" label.
- **Eye metrics** — EAR value + colour-coded status, blink rate/min, MAR.
- **Timestamp footer**.

#### Right Panel — State, Risk, Recommendations, System
- **Driver state banner** — full-width colour block matching state severity.
- **Arc risk gauge** — 300° sweep gauge with rotating needle; risk value and
  level label below; 0.0 / 1.0 min-max labels.
- **Recommendation card** — priority badge (red/orange/green), title, 3-line
  word-wrapped message; "No alerts" with pulsing green text when clear.
- **Anomaly row** — green "CLEAR" / red "DETECTED" badge with anomaly score.
- **System status table** — camera FPS, inference time, pipeline/ML/face-
  engine/DB status.
- **Animated heartbeat line** — composite sine wave in risk colour.

#### Bottom Bar (7 widgets)
Full-width strip below the camera viewport:
1. **Eye Aspect Ratio** — value + open/closing label + bar.
2. **Head Pose** — yaw angle in degrees (±), pitch & roll sub-text + bar.
3. **Gaze Direction** — L/C/R U/D cardinal labels + x/y values + bar.
4. **PERCLOS** — percentage + ">15% FATIGUE" / "NORMAL" + bar.
5. **Blink Count** — blinks/min + bar.
6. **Fatigue Probability** — derived from PERCLOS × 2.5 + HIGH/MED/LOW.
7. **MAR Yawn** — mouth aspect ratio + YAWNING/NORMAL + bar.

#### Layout (1280 × 720 canvas)
```
┌──────────────┬──────────────────────────┬──────────────┐
│  Left Panel  │   Camera Viewport (HUD)  │  Right Panel │
│    280 px    │         720 × 540        │    280 px    │
│              ├──────────────────────────┤              │
│              │   Scrolling ticker bar   │              │
├──────────────┴──────────────────────────┴──────────────┤
│               Bottom Metrics Bar  (7 widgets, 140 px)  │
└─────────────────────────────────────────────────────────┘
```

#### Animation clock
- `t` accumulates real elapsed seconds; all animations are time-based
  (not frame-count-based) so they remain smooth at any FPS.

#### Performance
- All rendering is pure NumPy + OpenCV2 — no Python loops over
  individual pixel coordinates.
- Landmark polylines use `cv2.polylines` (single C call per group).
- Gauge arcs use `cv2.ellipse`.
- Panel backgrounds use `cv2.addWeighted` for glassmorphism blending.
- Target: ≥ 25 FPS on a CPU-only laptop running the full ML pipeline.

---

### Files changed
| File | Change |
|---|---|
| `backend/engine/live_pipeline_runner.py` | Full rewrite of visualization layer; `LivePipelineRunner` class and public API unchanged |

### Files NOT changed
Everything else in the repository.

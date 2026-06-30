# CogniDrive HUD — Installation & Run Guide

## 1. Drop the file into the repository

```bash
# From your project root (the folder that contains backend/)
cp live_pipeline_runner.py backend/engine/live_pipeline_runner.py
```

## 2. Install / verify dependencies

```bash
# Activate your project venv first
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r backend/requirements.txt

# For the HUD window you need a display-capable OpenCV build
# (not the headless variant):
pip uninstall -y opencv-python-headless 2>/dev/null || true
pip install opencv-python
```

> **Server environment / SSH**: set `DISPLAY=:0` or use a virtual framebuffer:
> ```bash
> Xvfb :99 &
> export DISPLAY=:99
> ```

## 3. Start the backend (required for full pipeline)

```bash
cd /path/to/project   # folder containing backend/
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Wait for:
```
INFO | CogniDrive backend is ready — all subsystems initialized.
```

## 4. Run the HUD

The HUD is launched by `LivePipelineRunner` — typically from the FastAPI
lifespan or a standalone script. A minimal launcher:

```python
# run_hud.py  (place next to backend/)
import sys, time
sys.path.insert(0, ".")

from backend.app.config import get_settings
from backend.database.session import SessionLocal, create_database
from backend.database.models.driver_profile import DriverProfile
from backend.database.models.session_data import SessionData
from backend.vision.camera_manager import CameraManager
from backend.engine.pipeline_manager import PipelineManager
from backend.engine.live_pipeline_runner import LivePipelineRunner

settings = get_settings()
create_database()

db = SessionLocal()
driver = DriverProfile(name="HUD Demo Driver", experience_years=3)
db.add(driver); db.commit(); db.refresh(driver)
session = SessionData(driver_id=driver.id)
db.add(session); db.commit(); db.refresh(session)

cam = CameraManager(camera_index=0,
                    frame_width=settings.FRAME_WIDTH,
                    frame_height=settings.FRAME_HEIGHT)
cam.start()

pipeline = PipelineManager(camera_manager=cam, settings=settings)

runner = LivePipelineRunner.get_instance(cam, pipeline)
runner.start(driver_id=driver.id, session_id=session.id, db_session=db)

try:
    while runner.is_running:
        time.sleep(0.5)
except KeyboardInterrupt:
    pass
finally:
    runner.stop()
    cam.release()
    db.close()
```

```bash
python run_hud.py
```

## 5. HUD controls

| Key | Action |
|-----|--------|
| `Q` | Quit cleanly |
| `ESC` | Quit cleanly |
| Close window button | Quit cleanly |

## 6. Layout overview

```
┌──── Left Panel (280px) ────┬──── Camera + Face HUD (720px) ────┬── Right Panel (280px) ──┐
│ COGNIDRVE AI COCKPIT       │  ┌──────────────────────────────┐  │ [ DRIVER STATE ]        │
│                            │  │                              │  │                         │
│ ● AUTHENTICATED            │  │  468-point landmark mesh     │  │   Risk Arc Gauge        │
│                            │  │  Radar rings                 │  │       0.123             │
│ COGNITIVE METRICS          │  │  Sweep animation             │  │       LOW               │
│ ATTENTION    ████░░  78%   │  │  Corner brackets             │  │                         │
│ COGNIT LOAD  ██░░░░  32%   │  │  FACE LOCKED  CONF 90%      │  │ ! Recommendation card   │
│ STRESS LVL   █░░░░░  18%   │  │                              │  │   title + message       │
│ DROWSINESS   ░░░░░░   3%   │  └──────────────────────────────┘  │                         │
│ DISTRACTION  █░░░░░  12%   │  ┄ scrolling ticker ┄ ATTN 78% ┄  │ ANOMALY   CLEAR  0.021  │
│                            │                                    │                         │
│ GAZE MAP                   │                                    │ SYSTEM STATUS           │
│ ┌────────────────────────┐ │                                    │ CAMERA FPS   29.8       │
│ │           •            │ │                                    │ INFER TIME   38ms       │
│ └────────────────────────┘ │                                    │ PIPELINE     ACTIVE     │
│ ON-ROAD                    │                                    │ ~~heartbeat line~~      │
│                            │                                    │                         │
│ EYE METRICS                │                                    │                         │
│ EAR    0.278               │                                    │                         │
│ BLINKS 16/min              │                                    │                         │
│ MAR    0.042               │                                    │                         │
├────────────────────────────┴────────────────────────────────────┴─────────────────────────┤
│  EAR 0.278  │  HEAD +2°   │  GAZE C    │ PERCLOS 4% │ BLINKS 16 │ FATIGUE 10% │ MAR 0.042│
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 7. Colour key

| Colour | Meaning |
|--------|---------|
| Cyan `#00d8ff` | Primary HUD / healthy |
| Green `#00dc50` | Normal / good state |
| Amber / Gold | Warning |
| Orange | Alert |
| Red | Danger / high risk |
| Grey | Inactive / metadata |

## 8. Troubleshooting

| Problem | Fix |
|---------|-----|
| "Could not create OpenCV window" | Install `opencv-python` (not headless); set `DISPLAY` |
| Black camera viewport | Check `camera_index` setting; `CameraManager.is_running` must be True |
| No landmark mesh | MediaPipe model not found or frame too dark; check `FACE_LANDMARKER_MODEL_NAME` |
| HUD < 15 FPS | Reduce `_TARGET_FPS` to 20; disable `commit_online_stats` in heavy sessions |
| Window too large | Resize manually or change `_HUD_W / _HUD_H` constants at top of file |

"""
CogniDrive Live Prediction Demo

Connects to the backend's /pipeline/status endpoint to display what the
server's continuous LivePipelineRunner is currently predicting.

In Drishti mode the server handles camera capture and ML inference
automatically — this script is an optional monitoring client only.

Run (with the backend already started):
    python backend/demo/live_prediction.py

Press Q to quit.
"""

from __future__ import annotations

import sys
import time
from typing import Any

try:
    import requests
except ImportError:
    print("requests not installed — run: pip install requests")
    sys.exit(1)

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    print("cv2 not available — will print predictions to console only.")

BASE_URL = "http://127.0.0.1:8000/api/v1"
STATUS_URL = f"{BASE_URL}/pipeline/status"
POLL_INTERVAL = 0.1  # seconds between polls (~10 Hz display)


def _draw_overlay(frame: Any, data: dict) -> Any:
    """Overlay prediction text onto a blank frame."""
    y = 30
    lines = [
        f"CogniDrive  [server pipeline]",
        f"camera_running : {data.get('camera_running', '?')}",
        f"pipeline_running: {data.get('pipeline_running', '?')}",
    ]
    pred = data.get("last_prediction") or {}
    for key in [
        "face_detected",
        "driver_state",
        "attention_score",
        "stress_score",
        "cli",
        "risk_score",
        "inference_time_ms",
    ]:
        if key in pred:
            lines.append(f"{key}: {pred[key]}")

    for line in lines:
        cv2.putText(
            frame, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80), 1,
        )
        y += 24
    return frame


def _print_status(data: dict) -> None:
    pred = data.get("last_prediction") or {}
    print(
        f"\rcam={data.get('camera_running')} pipe={data.get('pipeline_running')} "
        f"state={pred.get('driver_state','?'):12s} "
        f"attn={pred.get('attention_score', 0.0):5.1f} "
        f"risk={pred.get('risk_score', 0.0):.3f}",
        end="",
        flush=True,
    )


def main() -> None:
    print("=" * 60)
    print("CogniDrive Live Demo  (monitoring server pipeline)")
    print(f"Polling: {STATUS_URL}")
    if _CV2:
        print("Press Q in the window to quit.")
    else:
        print("Press Ctrl+C to quit.")
    print("=" * 60)

    if _CV2:
        blank = __import__("numpy").zeros((300, 640, 3), dtype=__import__("numpy").uint8)

    while True:
        try:
            resp = requests.get(STATUS_URL, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
            else:
                data = {"camera_running": False, "pipeline_running": False,
                        "error": resp.status_code}
        except Exception as exc:
            data = {"camera_running": False, "pipeline_running": False,
                    "error": str(exc)}

        if _CV2:
            import numpy as np
            frame = blank.copy()
            frame = _draw_overlay(frame, data)
            cv2.imshow("CogniDrive — Server Pipeline Monitor", frame)
            key = cv2.waitKey(int(POLL_INTERVAL * 1000)) & 0xFF
            if key == ord("q"):
                break
        else:
            _print_status(data)
            time.sleep(POLL_INTERVAL)

    if _CV2:
        cv2.destroyAllWindows()
    print("\nDone.")


if __name__ == "__main__":
    main()

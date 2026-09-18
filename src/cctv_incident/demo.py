import json
from pathlib import Path

import cv2
import numpy as np

from .calibration import estimate_calibration


def generate_demo(root, negative=False):
    root = Path(root)
    samples = root / "data" / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    path = samples / ("negative.mp4" if negative else "demo.mp4")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 360))
    if not writer.isOpened():
        raise RuntimeError("Demo video encoder unavailable")
    try:
        for index in range(160):
            timestamp = index / 20
            frame = np.full((360, 640, 3), 45, np.uint8)
            for y in (90, 280):
                cv2.line(frame, (0, y), (639, y), (210, 210, 210), 3)
            travel = timestamp if negative else min(timestamp, 2)
            x1, x2 = 100 + 90 * travel, 500 - 90 * travel
            y1, y2 = (140, 230) if negative else (190, 190)
            for x, y, color in [(x1, y1, (0, 220, 0)), (x2, y2, (220, 0, 0))]:
                if 20 < x < 620:
                    cv2.rectangle(frame, (int(x - 14), y - 16), (int(x + 14), y + 16), color, -1)
            cv2.putText(
                frame,
                "SYNTHETIC TEST - NOT CCTV INFERENCE",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()
    points = [[0, 0], [639, 0], [639, 359], [0, 359]]
    destination = (np.asarray(points) * 0.05).tolist()
    calibration = estimate_calibration("synthetic", (640, 360), points, destination)
    calibration.save(root / "data" / "calibration" / "synthetic.yaml")
    annotations = (
        []
        if negative
        else [{"camera_id": "synthetic", "clip_id": "demo", "impact_time_s": 2.0, "label": 1}]
    )
    (samples / ("negative.events.json" if negative else "demo.events.json")).write_text(
        json.dumps(annotations, indent=2), encoding="utf-8"
    )
    return path

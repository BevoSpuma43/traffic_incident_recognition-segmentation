import cv2
import numpy as np


def color_for(track_id):
    return tuple(int(x) for x in np.random.default_rng(track_id).integers(70, 255, 3))


def render_frame(image, tracks, motions, histories, decision, calibration, render_masks=True):
    canvas = image.copy()
    for track in tracks:
        color = color_for(track.track_id)
        if render_masks:
            mask = track.instance.mask
            canvas[mask] = (canvas[mask] * 0.6 + np.array(color) * 0.4).astype(np.uint8)
        x1, y1, x2, y2 = map(int, track.instance.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track.track_id}"
        motion = motions.get(track.track_id)
        if motion and calibration.metric_valid():
            label += f" | {motion.speed * 3.6:.1f} km/h"
        elif motion and calibration.units == "px":
            label += f" | {motion.speed:.0f} px/s"
        cv2.putText(canvas, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        history = histories.get(track.track_id)
        if history:
            points = np.array([m.observation.point_px for m in history.motions], np.int32)
            if len(points) >= 2:
                cv2.polylines(canvas, [points], False, color, 2)
    cv2.polylines(canvas, [np.asarray(calibration.roi_px, np.int32)], True, (255, 220, 100), 1)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 45), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        f"{decision.state} | score {decision.score:.2f} | {calibration.units}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 220, 255),
        2,
    )
    return canvas


def render_bird_eye(motions, histories, calibration, candidate_ids=(), size=(480, 480)):
    width, height = size
    canvas = np.full((height, width, 3), (28, 32, 38), np.uint8)
    bounds = np.asarray(calibration.destination_points)
    minimum, maximum = bounds.min(axis=0) - 3, bounds.max(axis=0) + 3
    scale = min(
        (width - 40) / max(1, maximum[0] - minimum[0]),
        (height - 40) / max(1, maximum[1] - minimum[1]),
    )

    def pixel(point):
        xy = (np.asarray(point) - minimum) * scale + 20
        return int(np.clip(xy[0], -10000, 10000)), height - int(np.clip(xy[1], -10000, 10000))

    for track_id, motion in motions.items():
        color = (0, 100, 255) if track_id in candidate_ids else color_for(track_id)
        history = histories[track_id]
        points = np.array([pixel(m.position) for m in history.motions], np.int32)
        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, 2)
        pos = pixel(motion.position)
        cv2.circle(canvas, pos, 6, color, -1)
        cv2.arrowedLine(canvas, pos, pixel(motion.position + motion.velocity * 0.5), color, 2)
        cv2.putText(canvas, str(track_id), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    label = "METRI" if calibration.metric_valid() else "NON METRICA / INVALIDA"
    cv2.putText(canvas, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return canvas

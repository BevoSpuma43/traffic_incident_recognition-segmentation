"""Conservative motion fallback for detections that fail the IoU association."""

import numpy as np


def motion_costs(costs, tracks, detections, history, timestamp, frame_id, match_threshold):
    if not tracks or not detections:
        return costs
    predicted, eligible = [], []
    for track in tracks:
        samples = history.get(track.track_id, [])
        recent = bool(
            samples and track.frame_id == frame_id - 1 and 0 < timestamp - samples[-1][0] <= 0.25
        )
        eligible.append(recent)
        box = track.xyxy.copy()
        if recent:
            previous_time, previous_box = samples[-1]
            box = previous_box.copy()
            if len(samples) >= 2:
                old_time, old_box = samples[-2]
                velocity = ((box[:2] + box[2:]) - (old_box[:2] + old_box[2:])) / (
                    2 * (previous_time - old_time)
                )
                box += np.tile(velocity * (timestamp - previous_time), 2)
        predicted.append(box)
    a = np.asarray(predicted)
    b = np.asarray([d.xyxy for d in detections])
    centers_a, centers_b = (a[:, :2] + a[:, 2:]) / 2, (b[:, :2] + b[:, 2:]) / 2
    scales_a = np.linalg.norm(a[:, 2:] - a[:, :2], axis=1)
    scales_b = np.linalg.norm(b[:, 2:] - b[:, :2], axis=1)
    distances = np.linalg.norm(centers_a[:, None] - centers_b[None, :], axis=2) / np.maximum(
        1, (scales_a[:, None] + scales_b[None, :]) / 2
    )
    refined = costs.copy()
    for i, track in enumerate(tracks):
        threshold = match_threshold if track.is_activated else 0.7
        if not eligible[i] or costs[i].min() <= threshold:
            continue
        j = int(np.argmin(distances[i]))
        if (
            int(np.argmin(distances[:, j])) != i
            or distances[i, j] > 0.6
            or track.cls != detections[j].cls
        ):
            continue
        # Both directions must have a clear nearest neighbour. Ambiguous traffic
        # keeps the original association; no extrapolated observation is exported.
        if len(detections) > 1 and np.sort(distances[i])[1] < distances[i, j] + 0.2:
            continue
        if len(tracks) > 1 and np.sort(distances[:, j])[1] < distances[i, j] + 0.2:
            continue
        refined[i, j] = min(costs[i, j], 0.5 + 0.18 * distances[i, j] / 0.6)
    return refined

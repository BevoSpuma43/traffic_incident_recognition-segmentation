import cv2
import numpy as np


def ground_point(mask: np.ndarray, bbox: np.ndarray, roi=None):
    """Use the largest component, rejecting isolated low pixels and weak masks."""
    fallback = (float((bbox[0] + bbox[2]) / 2), float(bbox[3]))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return fallback, 0.2
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == component)
    if len(xs) < 12:
        return fallback, 0.2
    low_y = float(np.quantile(ys, 0.99))
    band = max(2, (low_y - float(ys.min())) * 0.06)
    selection = (ys >= low_y - band) & (ys <= low_y)
    point = (float(np.median(xs[selection])), float(np.median(ys[selection])))
    quality = min(1.0, len(xs) / max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
    if roi is not None and cv2.pointPolygonTest(np.asarray(roi, np.float32), point, False) < 0:
        return point, 0.0
    if low_y >= mask.shape[0] - 2:
        quality *= 0.4
    return point, float(quality)

from itertools import combinations

import cv2
import numpy as np


def fit_lines(skeleton):
    raw = cv2.HoughLinesP(skeleton, 1, np.pi / 180, 25, minLineLength=30, maxLineGap=15)
    if raw is None:
        return [], []
    families = [[], []]
    for segment in raw[:, 0].astype(float):
        direction = segment[2:] - segment[:2]
        length = np.linalg.norm(direction)
        line = np.cross([*segment[:2], 1], [*segment[2:], 1])
        line /= max(1e-9, np.linalg.norm(line[:2]))
        group = 0 if abs(direction[0]) > abs(direction[1]) else 1
        families[group].append((segment.tolist(), line, float(length)))
    return families


def vanishing_point(family):
    if len(family) < 2:
        return None, float("inf")
    intersections = []
    for a, b in combinations(family[:40], 2):
        point = np.cross(a[1], b[1])
        if abs(point[2]) > 1e-6:
            point = point[:2] / point[2]
            if np.isfinite(point).all() and np.linalg.norm(point) < 1e6:
                intersections.append(point)
    if not intersections:
        # Parallel image lines have a point at infinity, not a finite metric solution.
        return None, float("inf")
    center = np.median(intersections, axis=0)
    dispersion = float(np.median(np.linalg.norm(np.asarray(intersections) - center, axis=1)))
    return center.tolist(), dispersion


def extreme_lines(family, center):
    # Compare signed distances with consistent line orientation; suppress near-duplicate edges.
    aligned = []
    reference = family[0][1][:2]
    for _, line, length in family:
        line = line if line[:2] @ reference >= 0 else -line
        aligned.append((float(line @ [*center, 1]), line, length))
    aligned.sort(key=lambda item: item[0])
    return aligned[0][1], aligned[-1][1]

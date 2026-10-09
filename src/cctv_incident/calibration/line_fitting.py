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


def projective_families(segments, image_size, *, seed=42, max_families=4, hypotheses=512):
    """Bounded robust homogeneous VP fitting, including parallel lines at infinity.

    Segment coordinates are normalized with a uniform image scale. Residuals
    measure angular agreement at each segment midpoint, not pixel distance to a
    possibly very remote vanishing point. No horizontal/vertical image buckets.
    """
    segments = np.asarray(segments, dtype=float).reshape(-1, 4)
    if not np.isfinite(segments).all() or len(segments) > 160:
        raise ValueError("Expected at most 160 finite segments")
    if len(image_size) != 2 or min(image_size) < 2:
        raise ValueError("Expected positive image dimensions")
    if not 1 <= max_families <= 4 or not 1 <= hypotheses <= 1024:
        raise ValueError("Invalid line fitting budget")
    if not len(segments):
        return []
    scale = float(max(image_size))
    center = np.asarray(image_size, dtype=float) / 2
    ends = (segments.reshape(-1, 2, 2) - center) / scale
    lines = np.cross(np.c_[ends[:, 0], np.ones(len(ends))], np.c_[ends[:, 1], np.ones(len(ends))])
    norms = np.linalg.norm(lines[:, :2], axis=1)
    if (norms < 1e-9).any():
        raise ValueError("Zero-length line segment")
    lines /= norms[:, None]
    midpoints = ends.mean(axis=1)
    weights = np.minimum(np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1), 0.25)
    remaining = np.arange(len(lines))
    rng = np.random.default_rng(seed)
    families = []

    def residual(point, indices):
        directions = point[:2] - point[2] * midpoints[indices]
        lengths = np.linalg.norm(directions, axis=1)
        return np.abs(np.sum(lines[indices, :2] * directions, axis=1)) / np.maximum(lengths, 1e-12)

    for _ in range(max_families):
        if len(remaining) < 2:
            break
        pairs = list(combinations(remaining.tolist(), 2))
        if len(pairs) > hypotheses:
            pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), hypotheses, replace=False))]
        best = None
        for a, b in pairs:
            point = np.cross(lines[a], lines[b])
            norm = np.linalg.norm(point)
            if norm < 1e-8:
                continue  # Collinear fragments are not two distinct supporting lines.
            point /= norm
            inliers = remaining[residual(point, remaining) < np.sin(np.deg2rad(4))]
            score = float(weights[inliers].sum())
            if len(inliers) >= 2 and (best is None or score > best[0]):
                best = (score, point, inliers)
        if best is None:
            break
        _, point, inliers = best
        for _ in range(2):
            _, _, vh = np.linalg.svd(
                lines[inliers] * np.sqrt(weights[inliers, None]), full_matrices=True
            )
            refined = vh[-1]
            refined_inliers = remaining[residual(refined, remaining) < np.sin(np.deg2rad(4))]
            if len(refined_inliers) < 2:
                break
            point, inliers = refined, refined_inliers
        if point[2] < 0:
            point = -point
        if abs(point[2]) < 1e-5:
            native = np.r_[point[:2] / np.linalg.norm(point[:2]), 0.0]
            kind = "infinite"
        else:
            native = np.r_[scale * point[:2] / point[2] + center, 1.0]
            kind = "finite"
        families.append(
            {
                "kind": kind,
                "homogeneous": native.tolist(),
                "segment_indices": inliers.tolist(),
                "median_angular_error_deg": float(
                    np.rad2deg(np.arcsin(np.clip(np.median(residual(point, inliers)), 0, 1)))
                ),
            }
        )
        remaining = remaining[~np.isin(remaining, inliers)]
    return families

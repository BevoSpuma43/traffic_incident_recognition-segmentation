from collections import defaultdict
from itertools import combinations

import numpy as np

from .types import PairFeatures


def bbox_diagonal(bbox):
    return max(1.0, float(np.linalg.norm(np.maximum(0, bbox[2:] - bbox[:2]))))


def bbox_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def bbox_contact_coverage(a, b):
    """Overlap relative to the smaller vehicle, so a truck does not dilute contact."""
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    smaller = min(np.maximum(0, a[2:] - a[:2]).prod(), np.maximum(0, b[2:] - b[:2]).prod())
    return float(intersection / smaller) if smaller > 0 else 0.0


def closest_approach(relative_position, relative_velocity, collision_radius):
    p, v = np.asarray(relative_position, float), np.asarray(relative_velocity, float)
    vv, pv, pp = float(v @ v), float(p @ v), float(p @ p)
    if vv < 1e-9:
        return 0.0, float(np.sqrt(pp)), None, False
    t = max(0.0, -pv / vv)
    distance = float(np.linalg.norm(p + v * t))
    ttc = None
    if pv < 0:
        discriminant = pv * pv - vv * (pp - collision_radius**2)
        if pp <= collision_radius**2:
            ttc = 0.0
        elif discriminant >= 0:
            root = (-pv - np.sqrt(discriminant)) / vv
            if root >= 0:
                ttc = float(root)
    return t, distance, ttc, pv < 0


def compute_pairs(motions, config, coordinate_mode="metric"):
    grid = defaultdict(list)
    image_mode = coordinate_mode == "image"
    radius = (
        (
            max((bbox_diagonal(m.observation.bbox) for m in motions.values()), default=1)
            * config.image.pair_radius_diagonals
        )
        if image_mode
        else config.pair_radius_m
    )
    for track_id, motion in motions.items():
        cell = tuple(np.floor(motion.position / radius).astype(int))
        grid[cell].append(track_id)
    pairs = []
    for cell, ids in grid.items():
        candidates = list(combinations(ids, 2))
        for dx, dy in [(0, 1), (1, -1), (1, 0), (1, 1)]:
            candidates.extend(
                (a, b) for a in ids for b in grid.get((cell[0] + dx, cell[1] + dy), [])
            )
        for first, second in candidates:
            a, b = motions[first], motions[second]
            p, v = b.position - a.position, b.velocity - a.velocity
            distance = float(np.linalg.norm(p))
            scale = (bbox_diagonal(a.observation.bbox) + bbox_diagonal(b.observation.bbox)) / 2
            pair_radius = scale * config.image.pair_radius_diagonals if image_mode else radius
            if distance > pair_radius:
                continue
            collision_radius = (
                scale * config.image.collision_distance_diagonals
                if image_mode
                else config.collision_distance_m
            )
            t, closest, ttc, converging = closest_approach(p, v, collision_radius)
            pairs.append(
                PairFeatures(
                    tuple(sorted((first, second))),
                    distance,
                    float(np.linalg.norm(v)),
                    t,
                    closest,
                    ttc,
                    converging,
                    bbox_iou(a.observation.bbox, b.observation.bbox),
                    min(a.quality, b.quality),
                )
            )
    return pairs

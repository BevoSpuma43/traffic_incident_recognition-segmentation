"""Bounded recovery of chipped paint and repeated planar marking patterns.

These are geometric hypotheses, never semantic recognition or metric measurements.
"""

from itertools import combinations

import cv2
import numpy as np

from .coordinates import validate_quad
from .homography import order_quad, transform_points


def edge_support(quad, distance):
    support, gaps = [], []
    for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True):
        samples = np.linspace(a, b, min(128, max(8, round(np.linalg.norm(b - a)))))
        observed = (
            distance[np.rint(samples[:, 1]).astype(int), np.rint(samples[:, 0]).astype(int)] <= 2
        )
        support.append(float(observed.mean()))
        runs = np.diff(np.r_[False, ~observed, False].astype(int))
        starts, stops = np.flatnonzero(runs == 1), np.flatnonzero(runs == -1)
        gaps.append(float(max(stops - starts, default=0) / len(samples)))
    return support, gaps


def _elongated(contour):
    (x, y), (w, h), angle = cv2.minAreaRect(contour)
    if min(w, h) < 6 or max(w, h) < 25 or max(w, h) / min(w, h) < 2:
        return None
    if h > w:
        angle += 90
    direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    return np.array([x, y]), direction, min(w, h), max(w, h)


def recover_markings(contours, mask, distance, border_distance, known_quads):
    """Recover hulls only with paint fill, evidence on every edge and bounded gaps.

    Never extrapolate a marking beyond the frame/ROI, or merge neighbouring lanes.
    """
    size = (mask.shape[1], mask.shape[0])
    contours = [c for c in contours[:80] if cv2.contourArea(c) >= 25]
    hypotheses = [(c, 1) for c in contours]
    elongated = [(c, _elongated(c)) for c in contours]
    elongated = [(c, shape) for c, shape in elongated if shape is not None]
    for (a, sa), (b, sb) in combinations(elongated[:40], 2):
        ca, da, wa, la = sa
        cb, db, wb, lb = sb
        delta = cb - ca
        if abs(da @ db) < np.cos(np.deg2rad(10)) or max(wa, wb) > 1.5 * min(wa, wb):
            continue
        across = abs(da[0] * delta[1] - da[1] * delta[0])
        along = abs(da @ delta)
        gap = along - (la + lb) / 2
        if across > 0.35 * min(wa, wb) or not 0 <= gap <= 0.35 * min(la, lb):
            continue
        hypotheses.append((np.concatenate([a, b]), 2))
        if len(hypotheses) >= 120:
            break
    recovered = []
    for contour, fragments in hypotheses:
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.01, 0.02, 0.03):
            polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True).reshape(-1, 2)
            if len(polygon) != 4:
                continue
            try:
                quad = order_quad(polygon)
                validate_quad(quad, size, min_area=60)
            except ValueError:
                continue
            if any(np.linalg.norm(quad.mean(axis=0) - q.mean(axis=0)) < 5 for q in known_quads):
                break
            sides = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if min(sides) < 6 or max(sides) / min(sides) < 1.8:
                continue
            if min(border_distance[int(y), int(x)] for x, y in quad) < 3:
                continue
            support, gaps = edge_support(quad, distance)
            if min(support) < 0.5 or np.mean(support) < 0.7 or max(gaps) > 0.35:
                continue
            inside = np.zeros_like(mask)
            cv2.fillPoly(inside, [quad.astype(np.int32)], 255)
            fill = float((mask[inside > 0] > 0).mean())
            if fill < 0.7:
                continue
            recovered.append(
                (
                    quad,
                    support,
                    {
                        "reconstructed": True,
                        "fragments": fragments,
                        "largest_edge_gaps": gaps,
                        "mask_fill": fill,
                    },
                )
            )
            known_quads.append(quad)
            break
        if len(recovered) >= 20:
            break
    return recovered


def repeated_patterns(candidates, image_size):
    """Group >=3 bars only after projective rectification and alignment checks.

    Side-by-side bars must share endpoints. End-to-end dashes must share lateral
    edges. Equal spacing is checked in rectified coordinates, never in raw pixels.
    """
    bars = [c for c in candidates if c.reference_type in {"painted_bar", "reconstructed_bar"}][:40]
    patterns, seen = [], set()
    for anchor in bars:
        h = cv2.getPerspectiveTransform(
            np.float32(anchor.points_px), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        )
        normalized = []
        for i, bar in enumerate(bars):
            denominators = np.c_[bar.points_px, np.ones(4)] @ h[2]
            if np.min(denominators) * np.max(denominators) <= 0:
                continue
            try:
                quad = transform_points(h, bar.points_px)
            except ValueError:
                continue
            lower, upper = quad.min(axis=0), quad.max(axis=0)
            span = upper - lower
            # A marking crossing the rectification horizon is not a road neighbour.
            if (
                not cv2.isContourConvex(quad.astype(np.float32))
                or (span < 0.65).any()
                or (span > 1.5).any()
            ):
                continue
            sides = np.roll(quad, -1, axis=0) - quad
            if max(abs(sides[[0, 2], 1])) > 0.15 or max(abs(sides[[1, 3], 0])) > 0.15:
                continue
            normalized.append((i, lower, upper, quad.mean(axis=0)))
        for axis, kind in ((0, "repeated_bars"), (1, "repeated_dashes")):
            other = 1 - axis
            aligned = [
                n for n in normalized if abs(n[1][other]) <= 0.15 and abs(n[2][other] - 1) <= 0.15
            ]
            aligned.sort(key=lambda n: n[3][axis])
            # Split at large gaps/overlaps; do not enclose unrelated components.
            clusters, current = [], []
            for item in aligned:
                if current:
                    gap = item[1][axis] - current[-1][2][axis]
                    if not 0.15 <= gap <= 4:
                        clusters.append(current)
                        current = []
                current.append(item)
            clusters.append(current)
            for cluster in clusters:
                if len(cluster) < 3:
                    continue
                ids = tuple(sorted(n[0] for n in cluster))
                if (kind, ids) in seen:
                    continue
                spacing = np.diff([n[3][axis] for n in cluster])
                if np.max(abs(spacing - np.median(spacing))) > 0.3 * np.median(spacing):
                    continue
                lower = np.min([n[1] for n in cluster], axis=0)
                upper = np.max([n[2] for n in cluster], axis=0)
                corners = np.array(
                    [
                        [lower[0], lower[1]],
                        [upper[0], lower[1]],
                        [upper[0], upper[1]],
                        [lower[0], upper[1]],
                    ]
                )
                try:
                    quad = order_quad(transform_points(np.linalg.inv(h), corners))
                    validate_quad(quad, image_size, min_area=100)
                except (ValueError, np.linalg.LinAlgError):
                    continue
                lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
                if lengths[0] + lengths[2] > lengths[1] + lengths[3]:
                    quad = np.roll(quad, -1, axis=0)
                members = [bars[i] for i in ids]
                patterns.append(
                    anchor.model_copy(
                        update={
                            "candidate_id": "",
                            "reference_type": kind,
                            "points_px": tuple(tuple(float(v) for v in point) for point in quad),
                            "support_lines_px": tuple(
                                s for c in members for s in c.support_lines_px
                            ),
                            "geometric_quality": min(
                                0.64, min(c.geometric_quality for c in members) * 0.85
                            ),
                            "reasons": (
                                f"Gruppo di {len(members)} tratti con allineamento e ripetizione coerenti.",
                                "Verifica il rettangolo complessivo e misura entrambi i lati; i preset di un singolo tratto non si applicano.",
                            ),
                            "diagnostics": {
                                "member_count": len(members),
                                "member_points_px": [c.points_px for c in members],
                                "rectified_spacing": spacing.tolist(),
                                "single_marking": False,
                                "reconstructed": any(
                                    c.diagnostics.get("reconstructed") for c in members
                                ),
                                "complete_marking_verified": False,
                            },
                        }
                    )
                )
                seen.add((kind, ids))
                if len(patterns) == 8:
                    return patterns
    return patterns

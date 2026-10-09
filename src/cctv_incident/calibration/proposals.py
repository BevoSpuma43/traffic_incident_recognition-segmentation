"""Conservative road-marking proposals; geometry never supplies a metric scale."""

from dataclasses import dataclass, field

import cv2
import numpy as np
from pydantic import Field

from .coordinates import validate_quad, validate_roi
from .homography import order_quad
from .lane_mask import lane_mask
from .line_fitting import projective_families
from .marking_geometry import edge_support, recover_markings, repeated_patterns
from .records import Automation, Distance, RecordModel, Vertex
from .repository import pixel_sha256

ALGORITHM_VERSION = "painted-patterns-v2"


class ProposalParameters(RecordModel):
    max_dimension: int = Field(1280, ge=320, le=1920)
    max_candidates: int = Field(5, ge=1, le=8)
    max_segments: int = Field(160, ge=16, le=160)
    seed: int = Field(42, ge=0, le=2**32 - 1)
    min_edge_support: float = Field(0.7, ge=0.5, le=1)
    recover_incomplete: bool = True
    group_markings: bool = True


class ProposalCandidate(RecordModel):
    candidate_id: str
    reference_type: str
    points_px: tuple[tuple[float, float], ...] = Field(min_length=4, max_length=4)
    support_lines_px: tuple[tuple[float, float, float, float], ...]
    geometric_quality: float = Field(ge=0, le=0.79)
    reasons: tuple[str, ...]
    diagnostics: dict
    width: Distance = Field(default_factory=Distance)
    length: Distance = Field(default_factory=Distance)


@dataclass(frozen=True)
class ProposalResult:
    candidates: tuple[ProposalCandidate, ...]
    diagnostics: dict
    mask: np.ndarray
    preview: np.ndarray
    evidence_images: dict[str, np.ndarray] = field(default_factory=dict)


def _segments(edges, quads, limit):
    detected = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(edges)[0]
    lines = [] if detected is None else detected[:, 0].tolist()
    # Contour sides have been independently checked against observed mask edges.
    for quad, _ in quads:
        lines.extend(
            [list(a) + list(b) for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True)]
        )
    lines.sort(key=lambda s: (-np.linalg.norm(np.array(s[2:]) - s[:2]), *s))
    unique = []
    for line in lines:
        a, b = np.asarray(line[:2]), np.asarray(line[2:])
        if np.linalg.norm(a - b) < 5:
            continue
        if any(
            min(
                np.linalg.norm(a - u[:2]) + np.linalg.norm(b - u[2:]),
                np.linalg.norm(a - u[2:]) + np.linalg.norm(b - u[:2]),
            )
            < 5
            for u in unique
        ):
            continue
        unique.append(np.asarray(line))
        if len(unique) == limit:
            break
    return np.asarray(unique).reshape(-1, 4)


def generate_proposals(image, roi=None, *, parameters=None):
    parameters = parameters or ProposalParameters()
    if not isinstance(parameters, ProposalParameters):
        parameters = ProposalParameters.model_validate(parameters)
    if not isinstance(image, np.ndarray) or (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) < 8
    ):
        raise ValueError("Expected an 8-bit BGR frame")
    original_size = (image.shape[1], image.shape[0])
    if roi is not None:
        validate_roi(roi, original_size)
    scale = min(1, parameters.max_dimension / max(original_size))
    work_size = tuple(max(8, round(v * scale)) for v in original_size)
    frame = cv2.resize(image, work_size) if work_size != original_size else image.copy()
    ratios = np.asarray(original_size) / np.asarray(work_size)
    working_roi = None if roi is None else (np.asarray(roi) / ratios).tolist()
    mask, _ = lane_mask(frame, working_roi, retain_wide=True)
    edges = cv2.Canny(mask, 50, 150)
    distance = cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)
    region = np.zeros(mask.shape, np.uint8)
    if working_roi is None:
        region[mask.shape[0] // 3 :] = 255
    else:
        cv2.fillPoly(region, [np.asarray(working_roi, np.int32)], 255)
    # Include the image boundary, which OpenCV's distance transform otherwise ignores.
    region_padded = np.pad(region, 1)
    border_distance = cv2.distanceTransform(region_padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:160]
    quads, discarded, reconstruction = [], {}, {}

    def reject(reason):
        discarded[reason] = discarded.get(reason, 0) + 1

    for contour in contours:
        if cv2.contourArea(contour) < 25:
            reject("area_insufficiente")
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True).reshape(-1, 2)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon.reshape(-1, 1, 2)):
            reject("contorno_non_rettangolare")
            continue
        try:
            quad = order_quad(polygon)
            validate_quad(quad, work_size, min_area=25)
        except ValueError:
            reject("geometria_degenere")
            continue
        if min(border_distance[int(y), int(x)] for x, y in quad) < 3:
            reject("riferimento_tagliato_da_frame_o_roi")
            continue
        lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        if min(lengths) < 6:
            reject("lato_troppo_corto")
            continue
        support, _ = edge_support(quad, distance)
        if min(support) < parameters.min_edge_support:
            reject("bordi_non_supportati")
            continue
        # Width P1→P2 denotes the shorter observed side pair; users must check
        # physical semantics under perspective before selecting a size preset.
        if lengths[0] + lengths[2] > lengths[1] + lengths[3]:
            quad = np.roll(quad, -1, axis=0)
            support = support[1:] + support[:1]
        quads.append((quad, support))
    if parameters.recover_incomplete:
        recovered = recover_markings(
            contours, mask, distance, border_distance, [q for q, _ in quads]
        )
        for quad, support, metadata in recovered:
            lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
            if lengths[0] + lengths[2] > lengths[1] + lengths[3]:
                quad = np.roll(quad, -1, axis=0)
                support = support[1:] + support[:1]
                metadata["largest_edge_gaps"] = (
                    metadata["largest_edge_gaps"][1:] + metadata["largest_edge_gaps"][:1]
                )
            reconstruction[len(quads)] = metadata
            quads.append((quad, support))
    segments = _segments(edges, quads, parameters.max_segments)
    families = projective_families(segments, work_size, seed=parameters.seed)
    candidates = []
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    paint = ((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 170)) | (
        (hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 110)
    )
    for quad_index, (quad, support) in enumerate(quads):
        center = quad.mean(axis=0)
        directions = np.roll(quad, -1, axis=0) - quad
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        best_families = []
        local_families = []
        for indices in [(0, 2), (1, 3)]:
            lines = []
            for index in indices:
                line = np.cross([*quad[index], 1], [*quad[(index + 1) % 4], 1])
                lines.append(line / np.linalg.norm(line[:2]))
            point = np.cross(*lines)
            if abs(point[2]) < 1e-7:
                point = np.r_[point[:2] / np.linalg.norm(point[:2]), 0]
                kind = "infinite"
            else:
                point /= point[2]
                kind = "finite"
            local_families.append(
                {"kind": kind, "homogeneous": point.tolist(), "side_indices": list(indices)}
            )
            scores = []
            for family in families:
                vp = np.asarray(family["homogeneous"])
                midpoints = (quad + np.roll(quad, -1, axis=0)) / 2
                rays = vp[:2] - vp[2] * midpoints
                rays /= np.maximum(np.linalg.norm(rays, axis=1)[:, None], 1e-9)
                error = np.abs(directions[:, 0] * rays[:, 1] - directions[:, 1] * rays[:, 0])
                scores.append(float(max(error[list(indices)])))
            best_families.append(int(np.argmin(scores)) if scores and min(scores) < 0.15 else None)
        # A closed contour gives two *observed* opposite-side families even when
        # unrelated background lines dominate the global robust fit. Do not form
        # rectangles by mixing extrema from unrelated components.
        inside = np.zeros(mask.shape, np.uint8)
        cv2.fillPoly(inside, [quad.astype(np.int32)], 255)
        fill = float(paint[inside > 0].mean())
        side_lengths = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
        aspect = (side_lengths[1] + side_lengths[3]) / (side_lengths[0] + side_lengths[2])
        reference_type = "painted_bar" if fill > 0.65 and aspect >= 1.8 else "outlined_rectangle"
        quality = float(min(0.79, 0.79 * min(support) * min(1, min(side_lengths) / 12)))
        recovered = reconstruction.get(quad_index)
        if recovered:
            if fill <= 0.65:
                reject("colore_non_compatibile_con_segnaletica")
                continue
            reference_type = "reconstructed_bar"
            quality = float(
                min(0.65, 0.79 * np.mean(support) * 0.9 * min(1, min(side_lengths) / 12))
            )
        native_quad = quad * ratios
        native_segments = tuple(
            tuple((np.r_[a, b] * np.tile(ratios, 2)).tolist())
            for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True)
        )
        candidates.append(
            ProposalCandidate(
                candidate_id="",
                reference_type=reference_type,
                points_px=tuple(map(tuple, native_quad)),
                support_lines_px=native_segments,
                geometric_quality=quality,
                reasons=(
                    "Barra ricostruita da bordi parziali; controlla i vertici stimati."
                    if recovered
                    else "Quattro bordi visibili e due direzioni prospettiche compatibili.",
                    "Verifica che il riferimento sia un rettangolo sul piano stradale; la scala resta da specificare.",
                ),
                diagnostics={
                    **(recovered or {}),
                    "edge_support": support,
                    "paint_fill": fill,
                    "observed_aspect": float(aspect),
                    "family_indices": best_families,
                    "opposite_side_families": local_families,
                    "line_family_coordinate_system": "processed_pixels",
                    "complete_marking_verified": False,
                    "centroid_px": (center * ratios).tolist(),
                    "single_marking": reference_type == "painted_bar",
                },
            )
        )
    if parameters.group_markings:
        candidates.extend(repeated_patterns(candidates, original_size))
    candidates.sort(
        key=lambda c: (
            -c.geometric_quality,
            -abs(cv2.contourArea(np.float32(c.points_px))),
            c.points_px,
        )
    )
    unique = []
    for candidate in candidates:
        if any(
            candidate.reference_type == c.reference_type
            and np.linalg.norm(np.mean(candidate.points_px, axis=0) - np.mean(c.points_px, axis=0))
            < 5 * max(ratios)
            for c in unique
        ):
            continue
        unique.append(candidate)
    chosen = unique[: parameters.max_candidates]
    # Keep a group available even when numerous small complete bars rank first.
    groups = [c for c in unique if c.reference_type.startswith("repeated_")]
    if (
        groups
        and parameters.max_candidates >= 3
        and not any(c.reference_type.startswith("repeated_") for c in chosen)
    ):
        chosen[-1] = groups[0]
    unique = [
        c.model_copy(update={"candidate_id": f"candidate_{i}"}) for i, c in enumerate(chosen, 1)
    ]
    preview = frame.copy()
    for index, family in enumerate(families):
        for s in segments[family["segment_indices"]]:
            cv2.line(
                preview,
                tuple(np.rint(s[:2]).astype(int)),
                tuple(np.rint(s[2:]).astype(int)),
                [(255, 150, 0), (0, 220, 255), (220, 0, 220), (100, 220, 100)][index],
                1,
            )
    for candidate in unique:
        quad = np.rint(np.asarray(candidate.points_px) / ratios).astype(np.int32)
        cv2.polylines(preview, [quad], True, (0, 255, 0), 2)
        cv2.putText(
            preview,
            candidate.candidate_id,
            tuple(quad[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    reason = (
        "Scegli una proposta e controlla punti e misure."
        if unique
        else "Non sono visibili quattro bordi di un riferimento planare con due direzioni sufficienti. Usa i clic o le coordinate numeriche."
    )
    diagnostics = {
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": parameters.model_dump(),
        "seed": parameters.seed,
        "processed_size": work_size,
        "original_size": original_size,
        "reference_pixel_sha256": pixel_sha256(image),
        "search_region": "roi" if roi is not None else "lower_two_thirds",
        "line_families": families,
        "segments_px": (segments * np.tile(ratios, 2)).tolist(),
        "discarded": discarded,
        "reconstructed_markings": len(reconstruction),
        "candidate_types": {
            kind: sum(c.reference_type == kind for c in unique)
            for kind in sorted({c.reference_type for c in unique})
        },
        "reason": reason,
        "requires_review": True,
        "metric_scale_available": False,
    }
    return ProposalResult(tuple(unique), diagnostics, mask, preview)


def candidate_changes(candidate, result):
    """Applying a candidate always clears old distances and previous acceptance."""
    return {
        "geometry_mode": "rectangle",
        "vertices": [
            Vertex(id=f"P{i}", x=x, y=y).model_dump()
            for i, (x, y) in enumerate(candidate.points_px, 1)
        ],
        "destination_points": [],
        "explicit_scale": None,
        "width": {**candidate.width.model_dump(), "user_confirmed": False},
        "length": {**candidate.length.model_dump(), "user_confirmed": False},
        "geometric_quality": candidate.geometric_quality,
        "automation": Automation(
            method="auto_assisted",
            algorithm_version=result.diagnostics.get("algorithm_version", ALGORITHM_VERSION),
            parameters=result.diagnostics["parameters"],
            seed=result.diagnostics["seed"],
            initial_points_px=candidate.points_px,
            diagnostics={
                **result.diagnostics,
                "selected_candidate": candidate.model_dump(mode="json"),
            },
        ).model_dump(),
    }

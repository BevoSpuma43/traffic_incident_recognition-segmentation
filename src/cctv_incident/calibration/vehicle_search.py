"""Opt-in vehicle detections and reviewable experimental road-plane proposals."""

from pathlib import Path

import av
import cv2
import numpy as np

from ..config import Perception
from ..ground_point import ground_point
from .coordinates import validate_quad, validate_roi
from .homography import transform_points
from .proposals import ProposalCandidate, ProposalResult
from .records import Distance
from .repository import file_sha256, pixel_sha256
from .temporal import stationary_scene
from .vehicle_geometry import (
    ALGORITHM_VERSION,
    REFERENCE_PAPER,
    VehicleDimensions,
    VehicleFitError,
    fit_vehicle_camera,
)


def _sample_frames(source, reference, seconds, count):
    targets = np.linspace(0, seconds, count)
    selected, diagnostics, target = [], [], 1
    try:
        with av.open(str(source)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 900:
                    break
                if index == 0:
                    image = frame.to_ndarray(format="bgr24")
                    if pixel_sha256(image) != pixel_sha256(reference):
                        raise ValueError(
                            "Il video è cambiato rispetto al fotogramma di riferimento"
                        )
                    start = frame.time
                    selected.append((index, start, image))
                    if seconds == 0 or start is None:
                        break
                    continue
                if frame.time is None:
                    continue
                elapsed = float(frame.time - start)
                if elapsed > seconds or target >= len(targets):
                    break
                if elapsed < targets[target]:
                    continue
                while target < len(targets) and targets[target] <= elapsed:
                    target += 1
                image = frame.to_ndarray(format="bgr24")
                stability = stationary_scene(reference, image)
                diagnostics.append(
                    {"frame_index": index, "timestamp_s": float(frame.time), **stability}
                )
                if stability["accepted"]:
                    selected.append((index, float(frame.time), image))
    except av.FFmpegError as exc:
        raise ValueError("Video non leggibile durante la ricerca delle automobili") from exc
    if not selected:
        raise ValueError("Il video non contiene un fotogramma leggibile")
    return selected, diagnostics


def _select_observations(observations, size):
    # Deduplicate parked/static detections; repeated moving cars are observations,
    # not an asserted count of different vehicles.
    unique = []
    for item in sorted(observations, key=lambda row: -row["confidence"]):
        if any(
            np.max(np.abs(np.asarray(item["bbox_px"]) - other["bbox_px"])) < 3 for other in unique
        ):
            continue
        unique.append(item)
    if len(unique) <= 12:
        return unique
    boxes = np.asarray([row["bbox_px"] for row in unique])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2 / size
    features = np.c_[centers, np.log((boxes[:, 3] - boxes[:, 1]) / size[1]) * 0.3]
    chosen = [0]
    while len(chosen) < 12:
        distances = np.linalg.norm(features[:, None] - features[chosen], axis=2).min(axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return [unique[i] for i in chosen]


def _road_rectangle(fit, image_size, dimensions, roi):
    heading = fit["camera"][3]
    basis = np.array([[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]])
    positions = fit["positions_m"][fit["inliers"]] @ basis
    lower = positions.min(axis=0) - [dimensions.width_m / 2, dimensions.length_m / 2]
    upper = positions.max(axis=0) + [dimensions.width_m / 2, dimensions.length_m / 2]
    center = (lower + upper) / 2
    for shrink in (1, 0.85, 0.7, 0.55, 0.4):
        span = (upper - lower) * shrink
        start, end = center - span / 2, center + span / 2
        local = np.array(
            [[start[0], start[1]], [end[0], start[1]], [end[0], end[1]], [start[0], end[1]]]
        )
        quad = transform_points(fit["road_to_image"], local @ basis.T)
        try:
            validate_quad(quad, image_size, min_area=np.prod(image_size) * 0.01)
        except ValueError:
            continue
        if roi is not None:
            probes = np.vstack([quad, (quad + np.roll(quad, -1, axis=0)) / 2, quad.mean(axis=0)])
            if any(
                cv2.pointPolygonTest(np.float32(roi), tuple(point), False) < 0 for point in probes
            ):
                continue
        return quad, span
    raise ValueError(
        "La porzione stradale stimata non forma un rettangolo utile dentro il frame e la ROI"
    )


def generate_vehicle_proposals(
    source, reference, roi=None, *, model_path=None, dimensions=None, seconds=5.0, segmenter=None
):
    """Detect cars with local weights only. Never starts incident analysis or saves records."""
    dimensions = VehicleDimensions.model_validate(dimensions or {})
    if not np.isfinite(seconds) or not 0 <= seconds <= 15:
        raise ValueError("La ricerca delle automobili ammette da 0 a 15 secondi")
    if (
        not isinstance(reference, np.ndarray)
        or reference.dtype != np.uint8
        or reference.ndim != 3
        or reference.shape[2] != 3
    ):
        raise ValueError("Fotogramma BGR non valido")
    size = (reference.shape[1], reference.shape[0])
    if roi is not None:
        validate_roi(roi, size)
    source = Path(source)
    before = source.stat()
    frames, stability = _sample_frames(source, reference, seconds, 6)
    model_hash = file_sha256(model_path) if model_path and Path(model_path).is_file() else None
    if segmenter is None:
        from ..segmenter import Segmenter

        if model_path is None or not Path(model_path).is_file():
            raise ValueError(
                "Seleziona un modello di segmentazione locale per riconoscere le automobili"
            )
        segmenter = Segmenter(
            Perception(
                model=Path(model_path),
                classes=["car", "bus", "truck"],
                confidence=0.45,
                max_detections=24,
            )
        )
    work_size = tuple(max(8, round(v * min(1, 1280 / max(size)))) for v in size)
    ratios = np.asarray(size) / work_size
    mask = np.zeros(work_size[::-1], np.uint8)
    observations, evidence, discarded = [], {}, {}

    def reject(reason):
        discarded[reason] = discarded.get(reason, 0) + 1

    for index, timestamp, image in frames:
        work = cv2.resize(image, work_size)
        instances = segmenter.predict(work)
        for instance in instances[:24]:
            box = np.asarray(instance.bbox, dtype=float)
            if instance.class_id != 2 or instance.confidence < 0.45:
                reject("classe_o_confidenza")
                continue
            if (
                box.shape != (4,)
                or not np.isfinite(box).all()
                or (box[:2] < 3).any()
                or (box[2:] > np.asarray(work_size) - 4).any()
                or min(box[2:] - box[:2]) < 18
            ):
                reject("auto_piccola_o_tagliata")
                continue
            if instance.mask.shape != work.shape[:2]:
                raise ValueError("Maschera del veicolo non coerente con il fotogramma")
            point, quality = ground_point(instance.mask, box)
            native_point = np.asarray(point) * ratios
            if (
                quality < 0.3
                or (roi is None and native_point[1] < size[1] / 3)
                or (
                    roi is not None
                    and cv2.pointPolygonTest(np.float32(roi), tuple(native_point), False) < 0
                )
            ):
                reject("fuori_strada_o_maschera_debole")
                continue
            occluded = False
            for other in instances:
                if other is instance:
                    continue
                b = np.asarray(other.bbox)
                intersection = np.prod(
                    np.maximum(0, np.minimum(box[2:], b[2:]) - np.maximum(box[:2], b[:2]))
                )
                if (
                    intersection / max(1, min(np.prod(box[2:] - box[:2]), np.prod(b[2:] - b[:2])))
                    > 0.2
                ):
                    occluded = True
                    break
            if occluded:
                reject("auto_sovrapposte")
                continue
            observations.append(
                {
                    "bbox_px": (box * np.tile(ratios, 2)).tolist(),
                    "confidence": float(instance.confidence),
                    "frame_index": index,
                    "timestamp_s": timestamp,
                    "pixel_sha256": pixel_sha256(image),
                }
            )
            mask[instance.mask.astype(bool)] = 255
            evidence[str(index)] = work
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Il video è cambiato durante la ricerca delle automobili")
    observations = _select_observations(observations, np.asarray(size))
    observed_evidence = {}
    for i, row in enumerate(observations):
        key = str(row["frame_index"])
        if key not in observed_evidence:
            observed_evidence[key] = evidence[key].copy()
        box = np.rint(np.asarray(row["bbox_px"]) / np.tile(ratios, 2)).astype(int)
        cv2.rectangle(observed_evidence[key], tuple(box[:2]), tuple(box[2:]), (0, 180, 255), 1)
        cv2.putText(
            observed_evidence[key],
            str(i + 1),
            tuple(box[:2]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 180, 255),
            1,
        )
    diagnostics = {
        "algorithm_version": ALGORITHM_VERSION,
        "parameters": {
            "dimensions": dimensions.model_dump(),
            "seconds": seconds,
            "sample_count": 6,
            "max_observations": 12,
        },
        "seed": 42,
        "reference_pixel_sha256": pixel_sha256(reference),
        "original_size": size,
        "processed_size": work_size,
        "requires_review": True,
        "metric_scale_available": False,
        "scale_origin": "experimental",
        "assumption_source": dimensions.source(),
        "model_path": str(model_path) if model_path else None,
        "model_sha256": model_hash,
        "observations": observations,
        "discarded": discarded,
        "stability_checks": stability,
        "reference_paper": REFERENCE_PAPER,
        "assumptions": [
            "road_planar",
            "camera_fixed",
            "camera_roll_zero",
            "square_pixels",
            "principal_point_at_image_center",
            "common_vehicle_heading",
            "assumed_vehicle_dimensions",
        ],
    }
    preview = observed_evidence.get("0", cv2.resize(reference, work_size)).copy()
    try:
        fit = fit_vehicle_camera([row["bbox_px"] for row in observations], size, dimensions)
        quad, span = _road_rectangle(fit, size, dimensions, roi)
    except ValueError as exc:
        if isinstance(exc, VehicleFitError):
            diagnostics["rejected_fit"] = exc.diagnostics
        diagnostics["reason"] = (
            f"Calibrazione dalle automobili non disponibile: {exc}. Usa un riferimento stradale o la selezione manuale."
        )
        return ProposalResult((), diagnostics, mask, preview, observed_evidence)
    camera = fit["camera"]
    diagnostics["fit"] = {
        "focal_ratio": float(camera[0]),
        "pitch_degrees": float(np.rad2deg(camera[1])),
        "camera_height_m": float(camera[2]),
        "heading_degrees": float(np.rad2deg(camera[3])),
        "inlier_count": int(fit["inliers"].sum()),
        "median_relative_box_error": fit["median_relative_box_error"],
        "alternative_scale_ratio": fit["alternative_scale_ratio"],
        "relative_box_errors": fit["relative_box_errors"].tolist(),
        "optimizer_success": fit["optimizer_success"],
        "camera_attempts": fit["camera_attempts"],
        "road_to_image": fit["road_to_image"].tolist(),
        "projected_cuboids_px": fit["corners_px"].tolist(),
    }
    # Model wireframes show height explicitly; their image-box corners are not
    # calibration vertices. The proposed rectangle is a separate Z=0 road patch.
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    used_evidence = {}
    for i, row in enumerate(observations):
        if not fit["inliers"][i]:
            continue
        key = str(row["frame_index"])
        if key not in used_evidence:
            used_evidence[key] = evidence[key].copy()
        corners = np.rint(fit["corners_px"][i] / ratios).astype(np.int32)
        for a, b in edges:
            cv2.line(used_evidence[key], tuple(corners[a]), tuple(corners[b]), (255, 180, 0), 1)
    if "0" in used_evidence:
        preview = used_evidence["0"].copy()
    cv2.polylines(preview, [np.rint(quad / ratios).astype(np.int32)], True, (0, 255, 0), 2)
    source_description = dimensions.source()
    candidate = ProposalCandidate(
        candidate_id="candidate_1",
        reference_type="vehicle_reference",
        points_px=tuple(map(tuple, quad)),
        support_lines_px=tuple(
            tuple(np.r_[a, b]) for a, b in zip(quad, np.roll(quad, -1, axis=0), strict=True)
        ),
        geometric_quality=0.6,
        width=Distance(value=float(span[0]), origin="experimental", source=source_description),
        length=Distance(value=float(span[1]), origin="experimental", source=source_description),
        reasons=(
            f"Scala sperimentale da {int(fit['inliers'].sum())} osservazioni compatibili con l'auto tipo.",
            "I quattro punti delimitano una porzione stimata del piano stradale. Controlla prospettiva e dimensioni; le misure sono approssimative.",
        ),
        diagnostics={
            "single_marking": False,
            "scale_origin": "experimental",
            "dimensions": dimensions.model_dump(),
            "fit": diagnostics["fit"],
            "complete_marking_verified": False,
        },
    )
    diagnostics.update(
        metric_scale_available=True,
        reason="Controlla il piano stradale e conferma la scala sperimentale ricavata dalle automobili.",
    )
    return ProposalResult((candidate,), diagnostics, mask, preview, used_evidence)

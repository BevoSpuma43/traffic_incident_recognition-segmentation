"""Experimental US metric hypotheses. Classification and dimensions are assumptions.

Paint geometry is not semantic recognition. Repetition is preferred to an isolated
bar, and a lane needs a longitudinal reference as well as an assumed width.
"""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from pydantic import Field

from .coordinates import rectangle_destination, validate_quad
from .homography import transform_points
from .presets import SOURCE_EDITION, SOURCE_URL
from .proposals import ProposalParameters, ProposalResult
from .records import Distance, RecordModel
from .temporal import search_initial_frames
from .vehicle_geometry import VehicleDimensions
from .vehicle_search import generate_vehicle_proposals

ALGORITHM_VERSION = "experimental-us-metric-v1"
LANE_SOURCE = "https://highways.dot.gov/safety/other/road-diets/road-diet-informational-guide/4-designing-road-diet"


class AutomaticParameters(RecordModel):
    policy: Literal["experimental_usa_v1"] = "experimental_usa_v1"
    assume_usa: Literal[True] = True
    seconds: float = Field(10.0, gt=0, le=15)
    min_quality: float = Field(0.2, ge=0.1, le=0.6)
    dash_width_m: float = Field(0.1524, ge=0.1016, le=0.1524)
    dash_length_m: float = Field(3.048, gt=0)
    crosswalk_bar_width_m: float = Field(0.4572, ge=0.3048, le=0.6096)
    crosswalk_bar_length_m: float = Field(3.0, gt=0)
    lane_width_m: float = Field(3.6576, ge=3.048, le=3.6576)
    vehicle: VehicleDimensions = Field(default_factory=VehicleDimensions)


def _distance(value, reference, explanation):
    return Distance(
        value=float(value),
        origin="experimental",
        preset=reference,
        source=f"USA ipotizzati per tutti i video. {explanation}",
    )


def _metric_candidate(candidate, width, length, kind, explanation, quality_cap, **diagnostics):
    return candidate.model_copy(
        update={
            "reference_type": kind,
            "width": _distance(width, kind, explanation),
            "length": _distance(length, kind, explanation),
            "geometric_quality": min(quality_cap, candidate.geometric_quality),
            "reasons": (
                explanation,
                "Scala approssimativa; classificazione geometrica sperimentale.",
            ),
            "diagnostics": {**candidate.diagnostics, **diagnostics, "usa_assumed": True},
        }
    )


def marking_hypotheses(result, parameters=None):
    """Attach metric hypotheses, preserving the actual orientation of each quad."""
    p = AutomaticParameters.model_validate(parameters or {})
    source = f"{SOURCE_EDITION}, §§3A.04/3C.06; {SOURCE_URL}. "
    candidates = []
    for candidate in result.candidates:
        kind = candidate.reference_type
        if kind in {"repeated_bars", "repeated_dashes"}:
            members = candidate.diagnostics.get("member_points_px", [])
            if len(members) < 3:
                continue
            # The group may be rotated relative to its members. Rectify a member,
            # scale its axes, then measure the GROUP edges in that metric plane.
            crosswalk = kind == "repeated_bars"
            width = p.crosswalk_bar_width_m if crosswalk else p.dash_width_m
            length = p.crosswalk_bar_length_m if crosswalk else p.dash_length_m
            h = cv2.getPerspectiveTransform(
                np.float32(members[0]), np.float32(rectangle_destination(width, length))
            )
            world = transform_points(h, candidate.points_px)
            sides = np.linalg.norm(np.roll(world, -1, axis=0) - world, axis=1)
            if not np.isfinite(sides).all() or min(sides) <= 0:
                continue
            explanation = source + (
                f"Barre affiancate interpretate come attraversamento: barra {width:g} × {length:g} m. "
                "La lunghezza di 3 m è una scelta del progetto, non una lunghezza imposta dal MUTCD."
                if crosswalk
                else f"Tratti in successione interpretati come linea discontinua: tratto {length:g} m, "
                f"larghezza {width:g} m; intervalli ricavati dalla geometria rettificata. "
                "Il rapporto guida 10/30 ft non è universale."
            )
            candidates.append(
                _metric_candidate(
                    candidate,
                    np.mean(sides[[0, 2]]),
                    np.mean(sides[[1, 3]]),
                    "us_crosswalk" if crosswalk else "us_dash_group",
                    explanation,
                    0.6,
                    member_dimensions_m=[width, length],
                )
            )
        elif kind == "painted_bar" and candidate.diagnostics.get("single_marking"):
            # An isolated rectangle can be several different objects. Keep a low
            # quality cap and disclose this ambiguity rather than label it verified.
            candidates.append(
                _metric_candidate(
                    candidate,
                    p.dash_width_m,
                    p.dash_length_m,
                    "us_single_dash",
                    source
                    + f"Singola barra interpretata come tratto ordinario da {p.dash_length_m:g} m "
                    f"e {p.dash_width_m:g} m di larghezza; tipo non verificato semanticamente.",
                    0.35,
                    semantic_ambiguity=True,
                )
            )
    candidates.extend(_lane_hypotheses(result, p))
    return candidates


def _lane_hypotheses(result, p):
    """Adjacent aligned dash strips supply two directions and a lane width prior.

    Width alone cannot determine a homography. A complete dash supplies the
    longitudinal extent; raw segment pairs or a whole carriageway do not.
    """
    bars = [c for c in result.candidates if c.reference_type == "painted_bar"]
    lanes = []
    for i, anchor in enumerate(bars):
        h = cv2.getPerspectiveTransform(
            np.float32(anchor.points_px), np.float32(rectangle_destination(1, 1))
        )
        for neighbour in bars[i + 1 :]:
            q = transform_points(h, neighbour.points_px)
            lower, upper = q.min(axis=0), q.max(axis=0)
            sides = np.roll(q, -1, axis=0) - q
            x = float(q[:, 0].mean())
            # A lane should span about 20–36 normal paint widths. Side-by-side
            # crosswalk bars must not be mistaken for adjacent lane boundaries.
            if (
                not 18 <= abs(x - 0.5) <= 36
                or not 0.65 <= upper[0] - lower[0] <= 1.5
                or abs(lower[1]) > 0.15
                or abs(upper[1] - 1) > 0.15
                or max(abs(sides[[0, 2], 1])) > 0.15
                or max(abs(sides[[1, 3], 0])) > 0.15
            ):
                continue
            ends = sorted([0.5, x])
            corners = [[ends[0], 0], [ends[1], 0], [ends[1], 1], [ends[0], 1]]
            quad = transform_points(np.linalg.inv(h), corners)
            try:
                validate_quad(quad, result.diagnostics["original_size"], min_area=100)
            except ValueError:
                continue
            base = anchor.model_copy(
                update={
                    "points_px": tuple(map(tuple, quad)),
                    "geometric_quality": min(anchor.geometric_quality, neighbour.geometric_quality),
                }
            )
            lanes.append(
                _metric_candidate(
                    base,
                    p.lane_width_m,
                    p.dash_length_m,
                    "us_lane_and_dash",
                    f"Corsia ipotizzata fra due linee tratteggiate: {p.lane_width_m:g} m (12 ft); "
                    f"tratto lungo {p.dash_length_m:g} m. Le corsie possono variare, ad esempio 10–12 ft: "
                    f"{LANE_SOURCE}. Tratti: {SOURCE_URL}. Non è una misura dell'intera carreggiata.",
                    0.5,
                    boundary_points_px=[anchor.points_px, neighbour.points_px],
                )
            )
    return lanes


def generate_automatic_proposals(
    source,
    reference,
    *,
    roi=None,
    parameters=None,
    automatic_parameters=None,
    model_path=None,
    segmenter=None,
):
    p = AutomaticParameters.model_validate(automatic_parameters or {})
    painted = search_initial_frames(
        source,
        reference,
        roi,
        seconds=p.seconds,
        parameters=ProposalParameters.model_validate(parameters or {}),
    )
    candidates = marking_hypotheses(painted, p)
    evidence = dict(painted.evidence_images)
    vehicle_diagnostics = {}
    # Cars are a fallback, avoiding an expensive fit when repeated road markings
    # already supply a usable metric hypothesis.
    strong = any(
        c.reference_type != "us_single_dash" and c.geometric_quality >= p.min_quality
        for c in candidates
    )
    if not strong and (segmenter is not None or (model_path and Path(model_path).is_file())):
        try:
            vehicles = generate_vehicle_proposals(
                source,
                reference,
                roi,
                seconds=p.seconds,
                model_path=model_path,
                segmenter=segmenter,
                dimensions=p.vehicle,
            )
            vehicle_diagnostics = vehicles.diagnostics
            candidates.extend(vehicles.candidates)
            evidence.update(vehicles.evidence_images)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            vehicle_diagnostics = {"reason": f"Automobili: {type(exc).__name__}: {exc}"}
    else:
        vehicle_diagnostics = {
            "reason": "Riferimento stradale disponibile"
            if strong
            else "Pesi locali non disponibili"
        }
    ranks = {
        "us_crosswalk": 4,
        "us_dash_group": 4,
        "us_lane_and_dash": 3,
        "vehicle_reference": 2,
        "us_single_dash": 1,
    }
    candidates = [c for c in candidates if c.geometric_quality >= p.min_quality]
    candidates.sort(key=lambda c: (-ranks.get(c.reference_type, 2), -c.geometric_quality))
    chosen = tuple(
        c.model_copy(update={"candidate_id": f"candidate_{i}"})
        for i, c in enumerate(candidates[:8], 1)
    )
    diagnostics = {
        **painted.diagnostics,
        "algorithm_version": ALGORITHM_VERSION,
        "automatic_parameters": p.model_dump(mode="json"),
        "vehicle_search": vehicle_diagnostics,
        "metric_scale_available": bool(chosen),
        "requires_review": False,
        "approximate": True,
        "reason": (
            "Ipotesi USA sperimentale disponibile; misure approssimative."
            if chosen
            else "Nessuna ipotesi metrica utilizzabile: calibrazione manuale richiesta. "
            + vehicle_diagnostics.get("reason", "")
        ),
    }
    preview = cv2.resize(reference, (painted.preview.shape[1], painted.preview.shape[0]))
    if chosen:
        ratio = np.array(
            [reference.shape[1] / preview.shape[1], reference.shape[0] / preview.shape[0]]
        )
        cv2.polylines(
            preview,
            [np.rint(np.array(chosen[0].points_px) / ratio).astype(np.int32)],
            True,
            (0, 255, 0),
            2,
        )
    return ProposalResult(chosen, diagnostics, painted.mask, preview, evidence)

import cv2
import numpy as np
import pytest

from cctv_incident.calibration import automatic
from cctv_incident.calibration.automatic import AutomaticParameters, marking_hypotheses
from cctv_incident.calibration.marking_geometry import repeated_patterns
from cctv_incident.calibration.proposals import ProposalCandidate, ProposalResult


def bar(x, y, width=20, length=100):
    return ProposalCandidate(
        candidate_id=f"{x}_{y}",
        reference_type="painted_bar",
        points_px=((x, y), (x + width, y), (x + width, y + length), (x, y + length)),
        geometric_quality=0.7,
        support_lines_px=(),
        reasons=(),
        diagnostics={"single_marking": True},
    )


def result(candidates):
    return ProposalResult(
        tuple(candidates),
        {
            "original_size": (1000, 800),
            "seed": 42,
            "parameters": {},
            "reference_pixel_sha256": "0" * 64,
        },
        np.zeros((800, 1000), np.uint8),
        np.zeros((800, 1000, 3), np.uint8),
    )


def test_crosswalk_group_scale_follows_member_axes_after_group_rotation():
    bars = [bar(x, 200) for x in (100, 140, 180, 220)]
    groups = repeated_patterns(bars, (1000, 800))
    crosswalk = next(
        c for c in marking_hypotheses(result(groups)) if c.reference_type == "us_crosswalk"
    )
    # The group is 140 x 100 px; its first edge is the 100px length of a bar.
    assert crosswalk.width.value == pytest.approx(3.0)
    assert crosswalk.length.value == pytest.approx(7 * 0.4572)
    assert crosswalk.width.origin == "experimental"
    assert not crosswalk.width.user_confirmed
    assert "non una lunghezza imposta" in crosswalk.width.source


def test_dash_group_includes_rectified_gaps_in_its_metric_length():
    bars = [bar(100, y, 10, 50) for y in (100, 300, 500)]
    groups = repeated_patterns(bars, (1000, 800))
    dashes = next(
        c for c in marking_hypotheses(result(groups)) if c.reference_type == "us_dash_group"
    )
    assert dashes.width.value == pytest.approx(0.1524)
    assert dashes.length.value == pytest.approx(9 * 3.048)


def test_lane_requires_adjacent_dash_boundaries_and_longitudinal_extent():
    proposal = result([bar(100, 200), bar(580, 200)])
    lane = next(c for c in marking_hypotheses(proposal) if c.reference_type == "us_lane_and_dash")
    assert lane.width.value == pytest.approx(3.6576)
    assert lane.length.value == pytest.approx(3.048)
    assert lane.points_px[0] == pytest.approx((110, 200))
    assert lane.points_px[1] == pytest.approx((590, 200))
    assert not any(
        c.reference_type == "us_lane_and_dash"
        for c in marking_hypotheses(result([bar(100, 200), bar(140, 200)]))
    )


def test_crosswalk_group_dimensions_are_projectively_invariant():
    bars = [bar(x, 200) for x in (100, 140, 180, 220)]
    h = np.array([[1, 0.12, 40], [0.1, 1, 30], [0.0003, 0.0005, 1]], dtype=float)
    warped = [
        b.model_copy(
            update={
                "points_px": tuple(
                    map(tuple, cv2.perspectiveTransform(np.float32(b.points_px)[None], h)[0])
                )
            }
        )
        for b in bars
    ]
    group = next(
        c
        for c in marking_hypotheses(result(repeated_patterns(warped, (1000, 800))))
        if c.reference_type == "us_crosswalk"
    )
    assert sorted([group.width.value, group.length.value]) == pytest.approx(
        sorted([3, 7 * 0.4572]), rel=1e-5
    )


def test_vehicle_fallback_uses_same_ten_seconds_and_can_survive_model_failure(monkeypatch):
    calls = []
    paint = result([bar(100, 200)])

    def search(*args, **kwargs):
        calls.append(("paint", kwargs["seconds"]))
        return paint

    def vehicles(*args, **kwargs):
        calls.append(("cars", kwargs["seconds"]))
        assert kwargs["dimensions"] == AutomaticParameters().vehicle
        raise RuntimeError("model failed")

    monkeypatch.setattr(automatic, "search_initial_frames", search)
    monkeypatch.setattr(automatic, "generate_vehicle_proposals", vehicles)
    proposal = automatic.generate_automatic_proposals(
        "unused.mp4", paint.preview, segmenter=object()
    )
    assert calls == [("paint", 10.0), ("cars", 10.0)]
    assert proposal.candidates[0].reference_type == "us_single_dash"
    assert proposal.candidates[0].geometric_quality == 0.35
    assert "model failed" in proposal.diagnostics["vehicle_search"]["reason"]


def test_no_reference_stays_empty_and_unknown_rectangles_get_no_dimensions(monkeypatch):
    paint = result([bar(100, 200).model_copy(update={"reference_type": "outlined_rectangle"})])
    monkeypatch.setattr(automatic, "search_initial_frames", lambda *a, **k: paint)
    proposal = automatic.generate_automatic_proposals("unused", paint.preview)
    assert not proposal.candidates
    assert not proposal.diagnostics["metric_scale_available"]

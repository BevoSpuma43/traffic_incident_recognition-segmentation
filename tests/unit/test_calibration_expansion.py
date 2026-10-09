import cv2
import numpy as np
import pytest

from cctv_incident.calibration.coordinates import validate_quad
from cctv_incident.calibration.presets import preset_changes
from cctv_incident.calibration.proposals import (
    ProposalParameters,
    candidate_changes,
    generate_proposals,
)
from cctv_incident.calibration.records import (
    CalibrationRecord,
    ReferenceFrame,
    VideoIdentity,
    edit_record,
)
from cctv_incident.calibration.repository import pixel_sha256
from cctv_incident.calibration.temporal import stationary_scene


def pattern(kind="bars"):
    image = np.full((640, 640, 3), 45, np.uint8)
    if kind in {"chipped", "fragmented", "large_gap", "cropped"}:
        cv2.rectangle(
            image, (200, 240), (250, 639 if kind == "cropped" else 580), (245, 245, 245), -1
        )
        if kind == "chipped":
            cv2.rectangle(image, (190, 350), (225, 425), (45, 45, 45), -1)
        elif kind in {"fragmented", "large_gap"}:
            cv2.rectangle(
                image, (190, 385), (260, 405 if kind == "fragmented" else 480), (45, 45, 45), -1
            )
    elif kind == "ellipse":
        cv2.ellipse(image, (225, 430), (25, 160), 0, 0, 360, (245, 245, 245), -1)
    elif kind == "dashes":
        for y in (250, 380, 510):
            cv2.rectangle(image, (200, y), (220, y + 65), (245, 245, 245), -1)
    else:
        for i, x in enumerate((100, 230, 360)):
            offset = i * 65 if kind == "unaligned" else 0
            cv2.rectangle(image, (x, 260 + offset), (x + 45, 590), (245, 245, 245), -1)
    return image


@pytest.mark.parametrize("kind,fragments", [("chipped", 1), ("fragmented", 2)])
def test_partial_paint_recovery_retains_evidence_and_does_not_invent_scale(kind, fragments):
    result = generate_proposals(pattern(kind))
    candidate = next(c for c in result.candidates if c.reference_type == "reconstructed_bar")
    assert candidate.diagnostics["fragments"] == fragments
    assert min(candidate.diagnostics["edge_support"]) >= 0.5
    assert max(candidate.diagnostics["largest_edge_gaps"]) <= 0.35
    assert candidate.geometric_quality <= 0.65
    assert candidate.width.value is None and candidate.length.value is None
    assert not candidate.diagnostics["single_marking"]
    validate_quad(candidate.points_px, (640, 640))
    xs, ys = np.asarray(candidate.points_px).T
    np.testing.assert_allclose(
        [xs.min(), xs.max(), ys.min(), ys.max()], [200, 250, 240, 580], atol=2
    )
    assert not any(
        c.reference_type == "reconstructed_bar"
        for c in generate_proposals(
            pattern(kind), parameters=ProposalParameters(recover_incomplete=False)
        ).candidates
    )


@pytest.mark.parametrize("kind", ["ellipse", "cropped", "large_gap"])
def test_recovery_rejects_curved_clipped_and_heavily_missing_references(kind):
    result = generate_proposals(pattern(kind))
    assert not any(c.reference_type == "reconstructed_bar" for c in result.candidates)


@pytest.mark.parametrize(
    "kind,expected", [("bars", "repeated_bars"), ("dashes", "repeated_dashes")]
)
@pytest.mark.parametrize("perspective", [False, True])
def test_pattern_grouping_uses_rectified_geometry(kind, expected, perspective):
    image = pattern(kind)
    if perspective:
        matrix = cv2.getPerspectiveTransform(
            np.float32([[0, 0], [639, 0], [639, 639], [0, 639]]),
            np.float32([[110, 130], [540, 80], [620, 620], [25, 625]]),
        )
        image = cv2.warpPerspective(image, matrix, (640, 640), borderValue=(45, 45, 45))
    result = generate_proposals(image)
    group = next(c for c in result.candidates if c.reference_type == expected)
    assert group.diagnostics["member_count"] == 3
    assert not group.diagnostics["single_marking"]
    assert group.width.value is None and group.length.value is None
    validate_quad(group.points_px, (640, 640))
    # Also exercise JSON round-trip used by preparation, not just model_copy.
    type(group).model_validate_json(group.model_dump_json())


def test_unaligned_bars_do_not_form_a_shared_rectangle():
    result = generate_proposals(pattern("unaligned"))
    assert not any(c.reference_type.startswith("repeated_") for c in result.candidates)


def test_contrast_enhancement_does_not_turn_gray_objects_into_reconstructed_paint():
    image = pattern("chipped")
    image[image[:, :, 0] > 200] = 140
    result = generate_proposals(image)
    assert not any(c.reference_type == "reconstructed_bar" for c in result.candidates)


def test_applying_a_saved_proposal_preserves_its_algorithm_version():
    from dataclasses import replace

    result = generate_proposals(pattern())
    result = replace(
        result, diagnostics={**result.diagnostics, "algorithm_version": "painted-contours-v1"}
    )
    changes = candidate_changes(result.candidates[0], result)
    assert changes["automation"]["algorithm_version"] == "painted-contours-v1"


def test_group_remains_available_among_many_complete_markings():
    image = np.full((640, 640, 3), 45, np.uint8)
    for x in range(30, 580, 80):
        cv2.rectangle(image, (x, 270), (x + 25, 590), (245, 245, 245), -1)
    result = generate_proposals(image)
    assert len(result.candidates) == 5
    assert any(c.reference_type == "repeated_bars" for c in result.candidates)


@pytest.mark.parametrize(
    "kind,reference_type", [("bars", "repeated_bars"), ("fragmented", "reconstructed_bar")]
)
def test_single_marking_presets_are_blocked_on_groups_and_reconstructed_references(
    kind, reference_type
):
    image = pattern(kind)
    result = generate_proposals(image)
    candidate = next(c for c in result.candidates if c.reference_type == reference_type)
    digest = pixel_sha256(image)
    record = CalibrationRecord(
        video=VideoIdentity(
            source_path="scene.mp4",
            original_name="scene.mp4",
            sha256="a" * 64,
            first_frame_sha256=digest,
            size_bytes=100,
            image_size=(640, 640),
        ),
        reference=ReferenceFrame(frame_index=0, pixel_sha256=digest),
        camera_id="camera",
    )
    record = edit_record(record, **candidate_changes(candidate, result))
    with pytest.raises(ValueError, match="singola barra"):
        preset_changes(
            record,
            "us_broken_10ft",
            usa_verified=True,
            marking_kind="broken_line",
            reference_verified=True,
        )


def textured_scene():
    rng = np.random.default_rng(42)
    image = rng.integers(30, 100, (480, 640, 3), dtype=np.uint8)
    # Static detail distributed across the scene, independent of the road marking.
    for _ in range(150):
        x, y = rng.integers([15, 15], [620, 460])
        cv2.rectangle(image, (x, y), (x + 7, y + 7), (15, 15, 15), 2)
    return image


def test_static_background_can_validate_a_newly_uncovered_marking():
    first = textured_scene()
    uncovered = first.copy()
    cv2.rectangle(uncovered, (260, 240), (310, 440), (245, 245, 245), -1)
    assert stationary_scene(first, uncovered)["accepted"]


@pytest.mark.parametrize("motion", ["pan", "zoom", "cut", "blank"])
def test_temporal_search_rejects_motion_cuts_and_insufficient_background(motion):
    image = textured_scene()
    if motion == "pan":
        other = cv2.warpAffine(image, np.float32([[1, 0, 12], [0, 1, 0]]), (640, 480))
    elif motion == "zoom":
        other = cv2.warpAffine(image, cv2.getRotationMatrix2D((320, 240), 0, 1.08), (640, 480))
    elif motion == "cut":
        other = np.random.default_rng(12).integers(30, 100, image.shape, dtype=np.uint8)
    else:
        image = np.full_like(image, 45)
        other = np.full_like(image, 60)
    assert not stationary_scene(image, other)["accepted"]

import cv2
import numpy as np
import pytest

from cctv_incident.calibration.coordinates import validate_quad
from cctv_incident.calibration.line_fitting import projective_families
from cctv_incident.calibration.presets import PRESETS, preset_changes
from cctv_incident.calibration.proposals import (
    ProposalParameters,
    candidate_changes,
    generate_proposals,
)
from cctv_incident.calibration.records import (
    CalibrationRecord,
    Distance,
    ReferenceFrame,
    VideoIdentity,
    confirm_record,
    edit_record,
)
from cctv_incident.calibration.repository import pixel_sha256


def scene(kind="bar", size=(640, 480)):
    image = np.full((size[1], size[0], 3), 45, np.uint8)
    if kind == "bar":
        cv2.rectangle(image, (180, 220), (225, 430), (245, 245, 245), -1)
    elif kind == "bars":
        for x in [100, 220, 340]:
            cv2.rectangle(image, (x, 230), (x + 45, 440), (245, 245, 245), -1)
    elif kind == "outline":
        cv2.rectangle(image, (80, 210), (540, 430), (245, 245, 245), 3)
    elif kind == "perspective":
        cv2.fillPoly(
            image, [np.int32([[200, 200], [240, 205], [400, 420], [290, 420]])], (245, 245, 245)
        )
    elif kind == "rotated":
        cv2.fillPoly(image, [np.int32(cv2.boxPoints(((320, 340), (40, 180), 35)))], (245, 245, 245))
    elif kind == "parallel":
        for x in [100, 200, 350]:
            cv2.line(image, (x, 0), (x, size[1] - 1), (245, 245, 245), 4)
    elif kind == "circle":
        cv2.circle(image, (320, 330), 80, (245, 245, 245), -1)
    elif kind == "cropped":
        cv2.rectangle(image, (200, 220), (250, size[1] - 1), (245, 245, 245), -1)
    return image


def proposal_record(image, result=None):
    result = result or generate_proposals(image)
    digest = pixel_sha256(image)
    record = CalibrationRecord(
        video=VideoIdentity(
            source_path="scene.mp4",
            original_name="scene.mp4",
            sha256="a" * 64,
            first_frame_sha256=digest,
            size_bytes=100,
            image_size=(image.shape[1], image.shape[0]),
        ),
        reference=ReferenceFrame(frame_index=0, pixel_sha256=digest),
        camera_id="camera",
    )
    return edit_record(record, **candidate_changes(result.candidates[0], result))


@pytest.mark.parametrize("kind", ["blank", "parallel", "circle", "cropped"])
def test_unsupported_scenes_return_reasoned_manual_fallback(kind):
    result = generate_proposals(scene(kind))
    assert not result.candidates
    assert "Usa i clic" in result.diagnostics["reason"]
    assert (
        result.diagnostics["requires_review"] and not result.diagnostics["metric_scale_available"]
    )
    if kind == "cropped":
        assert result.diagnostics["discarded"]["riferimento_tagliato_da_frame_o_roi"]


@pytest.mark.parametrize("kind", ["bar", "bars", "outline", "perspective", "rotated"])
def test_observed_quads_have_support_but_never_invent_dimensions(kind):
    result = generate_proposals(scene(kind))
    assert result.candidates, result.diagnostics
    for candidate in result.candidates:
        validate_quad(candidate.points_px, (640, 480))
        if candidate.reference_type.startswith("repeated_"):
            assert candidate.diagnostics["member_count"] >= 3
            assert len(candidate.support_lines_px) == 4 * candidate.diagnostics["member_count"]
        else:
            assert len(candidate.support_lines_px) == 4
            assert min(candidate.diagnostics["edge_support"]) >= 0.7
            assert len(candidate.diagnostics["opposite_side_families"]) == 2
        assert candidate.width.value is None and candidate.length.value is None
        assert not candidate.width.user_confirmed and not candidate.length.user_confirmed
        assert candidate.geometric_quality <= 0.79
    if kind == "bars":
        assert [c.reference_type for c in result.candidates].count("painted_bar") == 3
        assert [c.reference_type for c in result.candidates].count("repeated_bars") == 1


def test_roi_clipping_is_not_a_complete_marking_and_preview_is_bounded():
    image = scene()
    result = generate_proposals(image, [[170, 230], [240, 230], [240, 400], [170, 400]])
    assert not result.candidates
    assert result.diagnostics["discarded"]["riferimento_tagliato_da_frame_o_roi"]
    large = cv2.resize(image, (2560, 1920), interpolation=cv2.INTER_NEAREST)
    result = generate_proposals(large, parameters=ProposalParameters(max_dimension=640))
    assert max(result.preview.shape[:2]) <= 640 and len(result.diagnostics["segments_px"]) <= 160
    assert result.candidates
    np.testing.assert_allclose(
        result.candidates[0].points_px,
        np.asarray(generate_proposals(image).candidates[0].points_px) * 4,
        atol=4,
    )


def test_candidates_and_diagnostics_are_deterministic_and_capped():
    image = scene("bars")
    parameters = ProposalParameters(max_candidates=2, seed=17)
    a, b = (
        generate_proposals(image, parameters=parameters),
        generate_proposals(image, parameters=parameters),
    )
    assert (
        len(a.candidates) == 2 and a.candidates == b.candidates and a.diagnostics == b.diagnostics
    )
    assert [c.candidate_id for c in a.candidates] == ["candidate_1", "candidate_2"]
    assert np.array_equal(a.preview, b.preview)


def test_vanishing_families_handle_finite_infinite_rotated_and_outliers():
    segments = []
    for x in [80, 120, 180, 240, 390, 450]:
        segments.append([x, 200, 320 + (x - 320) * 0.5, 350])  # VP at (320, 500).
    for y in [220, 260, 300, 350, 400, 440]:
        segments.append([80, y, 550, y])
    segments.extend([[20, 20, 50, 90], [590, 180, 550, 450]])
    families = projective_families(segments, (640, 600))
    finite = next(f for f in families if f["kind"] == "finite" and len(f["segment_indices"]) >= 5)
    np.testing.assert_allclose(finite["homogeneous"], [320, 500, 1], atol=1e-6)
    assert any(f["kind"] == "infinite" and len(f["segment_indices"]) >= 6 for f in families)
    rotation = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]])
    rotated = (np.asarray(segments).reshape(-1, 2) @ rotation.T).reshape(-1, 4)
    assert any(
        f["kind"] == "finite" and len(f["segment_indices"]) >= 5
        for f in projective_families(rotated, (640, 600))
    )


def test_applying_proposal_clears_old_scale_and_records_initial_geometry():
    image = scene()
    result = generate_proposals(image)
    record = proposal_record(image, result)
    scaled = edit_record(
        record,
        width=Distance(value=5, origin="measured", user_confirmed=True),
        length=Distance(value=12, origin="measured", user_confirmed=True),
    )
    confirmed = confirm_record(scaled)
    changed = edit_record(confirmed, **candidate_changes(result.candidates[0], result))
    assert changed.status == "draft" and changed.runtime is None
    assert changed.width.value is None and changed.length.value is None
    assert changed.automation.initial_points_px == result.candidates[0].points_px
    assert changed.automation.algorithm_version and changed.automation.seed == 42
    assert changed.automation.diagnostics["selected_candidate"]["candidate_id"] == "candidate_1"
    with pytest.raises(ValueError, match="confirmed provenance"):
        confirm_record(changed)


@pytest.mark.parametrize("preset", PRESETS)
def test_presets_are_explicit_hypotheses_bound_to_one_marking(preset):
    record = proposal_record(scene())
    changes = preset_changes(
        record,
        preset.id,
        usa_verified=True,
        marking_kind=preset.marking_kind,
        reference_verified=True,
        assumed_width=sum(preset.width_range_m) / 2,
    )
    changed = edit_record(record, **changes)
    assert changed.width.origin == "standard" and not changed.width.user_confirmed
    assert changed.width.preset == preset.id and "December 2025" in changed.width.source
    assert changed.length.value == preset.length_m
    assert changed.status == "draft" and changed.geometric_quality == record.geometric_quality
    assert changed.automation.diagnostics["preset_application"]["usa_verified"]


@pytest.mark.parametrize("case", ["country", "kind", "review", "outlined", "multi_bar", "width"])
def test_unsafe_preset_applications_are_rejected(case):
    record = proposal_record(scene("outline" if case == "outlined" else "bar"))
    kwargs = dict(usa_verified=True, marking_kind="broken_line", reference_verified=True)
    if case == "country":
        kwargs["usa_verified"] = False
    if case == "kind":
        kwargs["marking_kind"] = "crosswalk_bar"
    if case == "review":
        kwargs["reference_verified"] = False
    if case == "width":
        kwargs["assumed_width"] = 1
    if case == "multi_bar":
        record = edit_record(
            record,
            vertices=[
                {"id": f"P{i}", "x": x, "y": y}
                for i, (x, y) in enumerate([[80, 210], [540, 210], [540, 430], [80, 430]], 1)
            ],
        )
    with pytest.raises(ValueError):
        preset_changes(record, "us_broken_10ft", **kwargs)


def test_normal_dash_length_does_not_invent_width_or_gap_length():
    record = proposal_record(scene())
    changed = edit_record(
        record,
        **preset_changes(
            record,
            "us_broken_10ft",
            usa_verified=True,
            marking_kind="broken_line",
            reference_verified=True,
        ),
    )
    assert changed.width.value is None and changed.length.value == 3.048
    with pytest.raises(ValueError, match="Seleziona una larghezza"):
        preset_changes(
            record,
            "us_crosswalk_bar",
            usa_verified=True,
            marking_kind="crosswalk_bar",
            reference_verified=True,
        )


def test_changing_preset_clears_obsolete_hypotheses_but_keeps_measured_distance():
    record = proposal_record(scene())
    changes = preset_changes(
        record,
        "us_broken_10ft",
        usa_verified=True,
        marking_kind="broken_line",
        reference_verified=True,
        assumed_width=0.12,
    )
    record = edit_record(record, **changes)
    record = edit_record(
        record,
        **preset_changes(
            record,
            "us_crosswalk_bar",
            usa_verified=True,
            marking_kind="crosswalk_bar",
            reference_verified=True,
            assumed_width=0.45,
        ),
    )
    assert record.length.value is None and record.width.value == 0.45
    record = edit_record(record, width=Distance(value=0.5, origin="measured", user_confirmed=True))
    record = edit_record(
        record,
        **preset_changes(
            record,
            "us_broken_10ft",
            usa_verified=True,
            marking_kind="broken_line",
            reference_verified=True,
        ),
    )
    assert (
        record.width.value == 0.5
        and record.width.origin == "measured"
        and record.length.value == 3.048
    )

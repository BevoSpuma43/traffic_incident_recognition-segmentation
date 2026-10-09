import numpy as np
import pytest

from cctv_incident.calibration.editor import EditorSession, geometric_quality, metric_preview
from cctv_incident.calibration.records import (
    CalibrationRecord,
    Distance,
    ReferenceFrame,
    VideoIdentity,
    confirm_record,
    edit_record,
)


@pytest.fixture
def session():
    identity = VideoIdentity(
        source_path="clip.mp4",
        original_name="clip.mp4",
        sha256="a" * 64,
        first_frame_sha256="b" * 64,
        size_bytes=100,
        image_size=(160, 96),
    )
    record = CalibrationRecord(
        video=identity,
        reference=ReferenceFrame(frame_index=0, pixel_sha256="b" * 64),
        camera_id="video_a",
    )
    record = edit_record(
        record,
        vertices=[
            {"id": f"P{i}", "x": x, "y": y}
            for i, (x, y) in enumerate([(10, 10), (140, 10), (140, 80), (10, 80)], 1)
        ],
        width=Distance(value=5, origin="measured", user_confirmed=True),
        length=Distance(value=12, origin="measured", user_confirmed=True),
        geometric_quality=0.63,
    )
    return EditorSession(
        ("clip",), confirm_record(record), np.zeros((96, 160, 3), np.uint8), visually_accepted=True
    )


def payload(session, **changes):
    record = session.record
    data = dict(
        record_id=record.record_id,
        source_sha256=record.video.sha256,
        revision=record.revision,
        epoch=session.epoch,
        event_id="edit-1",
        mode="calibration",
        vertices=[v.model_dump() for v in record.vertices],
    )
    data.update(changes)
    return data


def test_drag_invalidation_and_undo_retain_ids_without_reaccepting(session):
    event = payload(session)
    event["vertices"][0]["x"] = 20
    assert session.receive(event)
    assert session.record.status == "draft" and session.record.runtime is None
    assert not session.visually_accepted and session.record.geometric_quality == 0
    assert session.record.vertices[0].id == "P1"
    assert not session.receive(event)  # Duplicate, even after rerun.
    session.undo()
    assert session.record.vertices[0].x == 10
    assert session.record.status == "draft" and not session.visually_accepted


@pytest.mark.parametrize(
    "field,value",
    [("source_sha256", "c" * 64), ("record_id", "f" * 32), ("revision", 42), ("epoch", 42)],
)
def test_late_events_from_previous_video_revision_or_edit_are_ignored(session, field, value):
    assert not session.receive(payload(session, **{field: value}))
    assert session.record.status == "confirmed" and session.visually_accepted


def test_roi_is_independent_and_invalidates_confirmation(session):
    vertices = session.record.vertices
    roi = [[0, 0], [100, 0], [100, 90], [0, 90]]
    assert session.receive(payload(session, mode="roi", roi_px=roi))
    assert session.record.vertices == vertices and session.record.roi_px == tuple(map(tuple, roi))
    assert not session.visually_accepted
    assert confirm_record(session.record).runtime.roi_px == roi


def test_invalid_coordinates_are_rejected_and_locked_editor_ignores_edits(session):
    event = payload(session)
    event["vertices"][0]["x"] = 160
    with pytest.raises(ValueError, match="outside"):
        session.receive(event)
    with pytest.raises(ValueError, match="outside"):
        session.receive(
            payload(session, event_id="roi", mode="roi", roi_px=[[0, 0], [160, 0], [0, 95]])
        )
    session.locked = True
    assert not session.receive(payload(session, event_id="locked"))


def test_no_op_and_each_measure_change(session):
    assert not session.change(vertices=[v.model_dump() for v in session.record.vertices])
    assert session.visually_accepted
    assert session.change(width=Distance(value=6, origin="measured").model_dump())
    assert not session.visually_accepted
    with pytest.raises(ValueError, match="confirmed provenance"):
        confirm_record(session.record)


def test_preview_is_bounded_for_large_measures_and_grid_has_sensible_spacing(session):
    image, step = metric_preview(session.frame, session.record)
    assert max(image.shape[:2]) <= 640 and step == 1
    record = edit_record(session.record, width=Distance(value=1e7), length=Distance(value=2e7))
    image, step = metric_preview(session.frame, record)
    assert max(image.shape[:2]) <= 640 and image.nbytes <= 640 * 640 * 3
    assert step >= 1e6


def test_geometry_quality_is_not_metric_provenance_and_bad_shapes_cannot_preview(session):
    assert 0.55 <= geometric_quality(session.record) <= 0.79
    record = edit_record(session.record, width=Distance(value=5), length=Distance(value=12))
    assert geometric_quality(record) == geometric_quality(session.record)
    with pytest.raises(ValueError, match="confirmed provenance"):
        confirm_record(record)
    crossed = edit_record(record, vertices=[record.vertices[i] for i in [0, 2, 1, 3]])
    # IDs determine correspondence order; changing array order does not cross the polygon.
    metric_preview(session.frame, crossed)
    vertices = [v.model_dump() for v in record.vertices]
    vertices[1]["x"], vertices[1]["y"] = 140, 80
    vertices[2]["x"], vertices[2]["y"] = 140, 10
    with pytest.raises(ValueError, match="crossings"):
        metric_preview(session.frame, edit_record(record, vertices=vertices))


def test_concave_roi_is_supported_but_self_intersections_are_not_confirmable(session):
    concave = [[5, 5], [150, 5], [150, 90], [75, 45], [5, 90]]
    record = edit_record(session.record, roi_px=concave)
    assert confirm_record(record).runtime.roi_px == concave
    # A crossed polygon can have nonzero signed area, so an area-only test is insufficient.
    crossed = [[5, 5], [140, 5], [20, 80], [140, 80], [5, 60]]
    with pytest.raises(ValueError, match="incrociarsi"):
        confirm_record(edit_record(session.record, roi_px=crossed))

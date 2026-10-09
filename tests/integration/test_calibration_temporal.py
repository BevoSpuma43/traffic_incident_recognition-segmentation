from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from cctv_incident.calibration.proposals import candidate_changes
from cctv_incident.calibration.records import Distance, confirm_record, edit_record
from cctv_incident.calibration.repository import (
    compatibility_reasons,
    create_draft,
    inspect_video,
    load_record,
    save_record,
)
from cctv_incident.calibration.temporal import search_initial_frames


@pytest.fixture
def occluded_video(tmp_path):
    return write_sequence(tmp_path)


def write_sequence(tmp_path, *, moving=False):
    path = tmp_path / ("moving.mkv" if moving else "uncovered.mkv")
    rng = np.random.default_rng(42)
    background = rng.integers(30, 100, (480, 640, 3), dtype=np.uint8)
    for _ in range(150):
        x, y = rng.integers([15, 15], [620, 460])
        cv2.rectangle(background, (x, y), (x + 7, y + 7), (15, 15, 15), 2)
    cv2.rectangle(background, (230, 210), (610, 465), (45, 45, 45), -1)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 640, 480, "bgr0"
        for i in range(21):
            image = background.copy()
            if i >= 6:
                x = 260 + 110 * ((i - 6) // 6) if moving else 260
                cv2.rectangle(image, (x, 240), (x + 50, 440), (245, 245, 245), -1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_moving_bright_surfaces_do_not_become_temporal_calibration_references(tmp_path):
    path = write_sequence(tmp_path, moving=True)
    _, first = create_draft(path, tmp_path)
    result = search_initial_frames(path, first, seconds=1.8, sample_count=4)
    assert not result.candidates
    rows = result.diagnostics["temporal_search"]["frames"]
    assert all(row["accepted"] and row["candidate_count"] >= 1 for row in rows)
    assert result.diagnostics["temporal_search"]["unconfirmed_observations"] >= 3


def test_uncovered_reference_keeps_first_frame_identity_and_survives_metric_save(
    occluded_video, tmp_path
):
    record, first = create_draft(occluded_video, tmp_path)
    result = search_initial_frames(occluded_video, first, seconds=1.8, sample_count=4)
    assert result.candidates
    assert result.diagnostics["reference_pixel_sha256"] == record.reference.pixel_sha256
    candidate = result.candidates[0]
    source = candidate.diagnostics["source_frame"]
    assert source["frame_index"] >= 6
    assert source["pixel_sha256"] in result.evidence_images
    xs, ys = np.asarray(candidate.points_px).T
    np.testing.assert_allclose(
        [xs.min(), xs.max(), ys.min(), ys.max()], [260, 310, 240, 440], atol=2
    )
    assert candidate.width.value is None and candidate.length.value is None
    updated = edit_record(record, **candidate_changes(candidate, result))
    confirmed = confirm_record(
        edit_record(
            updated,
            width=Distance(value=0.5, origin="measured", user_confirmed=True),
            length=Distance(value=2, origin="measured", user_confirmed=True),
        )
    )
    path = save_record(confirmed, tmp_path / "archive", reference_image=first)
    saved = load_record(path)
    assert saved.reference.pixel_sha256 == record.reference.pixel_sha256
    assert saved.reference.frame_index == 0
    assert not compatibility_reasons(saved, inspect_video(occluded_video, tmp_path))
    assert (
        saved.automation.diagnostics["selected_candidate"]["diagnostics"]["source_frame"] == source
    )


def test_initial_window_and_roi_are_respected(occluded_video, tmp_path):
    _, first = create_draft(occluded_video, tmp_path)
    early = search_initial_frames(occluded_video, first, seconds=0.3, sample_count=3)
    assert not early.candidates
    assert all(row["timestamp_s"] <= 0.3 for row in early.diagnostics["temporal_search"]["frames"])
    outside = search_initial_frames(
        occluded_video, first, [[20, 170], [180, 170], [180, 440], [20, 440]], seconds=1.8
    )
    assert not outside.candidates
    assert outside.diagnostics["search_region"] == "roi"


def test_temporal_search_rejects_a_reference_from_another_video(occluded_video, tmp_path):
    _, first = create_draft(occluded_video, tmp_path)
    with pytest.raises(ValueError, match="video è cambiato"):
        search_initial_frames(occluded_video, np.zeros_like(first), seconds=1)


@pytest.mark.parametrize("seconds,count", [(16, 6), (0, 6), (float("nan"), 6), (5, 13), (5, 1)])
def test_temporal_limits_are_validated_before_reading_video(seconds, count):
    with pytest.raises(ValueError, match="intervallo"):
        search_initial_frames(
            Path("missing.mp4"),
            np.zeros((100, 100, 3), np.uint8),
            seconds=seconds,
            sample_count=count,
        )


def test_temporal_editor_shows_evidence_and_preserves_corrections(occluded_video, tmp_path):
    from streamlit.testing.v1 import AppTest
    from test_calibration_ui import button, editor_app

    app = AppTest.from_function(editor_app, args=(str(occluded_video), str(tmp_path))).run(
        timeout=20
    )
    next(w for w in app.number_input if w.label == "Secondi iniziali da esaminare").set_value(1.5)
    button(app, "Cerca nei primi secondi").click().run(timeout=20)
    assert not app.exception
    session = app.session_state["single_calibration_editor"]
    assert session.record.automation.diagnostics["temporal_search"]["seconds"] == 1.5
    assert session.record.vertices and session.record.reference.frame_index == 0
    assert any("Riferimento trovato a" in w.value for w in app.caption)
    assert button(app, "Conferma e salva calibrazione").disabled
    point = session.record.vertices[0].x
    component = app.get("bidi_component")[0]
    import json

    data = json.loads(component.proto.json)
    payload = {k: data[k] for k in ("record_id", "source_sha256", "revision", "epoch")}
    payload.update(event_id="temporal-drag", mode="calibration", vertices=data["vertices"])
    payload["vertices"][0]["x"] = point + 1
    states = app._tree.get_widget_states()
    widget = states.widgets.add()
    widget.id, widget.json_value = component.proto.id, "{}"
    trigger = states.widgets.add()
    trigger.id = "$$STREAMLIT_INTERNAL_KEY_" + component.proto.id + "__events"
    trigger.json_trigger_value = json.dumps([{"event": "edit", "value": payload}])
    app._run(states)
    app.run(timeout=20)
    assert not app.exception and session.record.vertices[0].x == point + 1
    button(app, "Salva bozza").click().run()
    saved = load_record(tmp_path / "data/calibration/videos/uncovered.yaml")
    assert saved.vertices[0].x == point + 1
    reopened = AppTest.from_function(editor_app, args=(str(occluded_video), str(tmp_path))).run(
        timeout=20
    )
    assert not reopened.exception
    assert (
        reopened.session_state["single_calibration_editor"].record.reference.pixel_sha256
        == saved.reference.pixel_sha256
    )

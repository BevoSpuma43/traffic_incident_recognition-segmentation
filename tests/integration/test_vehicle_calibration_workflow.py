import json

import av
import cv2
import numpy as np
import pytest

from cctv_incident.calibration.editor import geometric_quality
from cctv_incident.calibration.proposals import candidate_changes
from cctv_incident.calibration.records import confirm_record, edit_record
from cctv_incident.calibration.repository import create_draft, load_record, save_record
from cctv_incident.calibration.vehicle_geometry import (
    VehicleDimensions,
    project_cuboids,
    projected_boxes,
)
from cctv_incident.calibration.vehicle_search import generate_vehicle_proposals
from cctv_incident.types import Instance


class KnownCars:
    def __init__(self, class_id=2, confidence=0.96):
        self.calls = 0
        self.class_id, self.confidence = class_id, confidence

    def predict(self, frame):
        self.calls += 1
        positions = np.array([[-5, 18], [-4, 25], [-2, 40], [2, 21], [5, 33], [10, 60]], float)
        camera = np.array([1, np.deg2rad(18), 9, np.deg2rad(18)])
        boxes = projected_boxes(
            project_cuboids((1280, 960), camera, positions, VehicleDimensions())
        )
        instances = []
        for box in boxes:
            mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.rectangle(mask, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), 1, -1)
            instances.append(Instance(box.copy(), mask > 0, self.class_id, self.confidence))
        return instances


@pytest.fixture
def vehicle_video(tmp_path):
    path = tmp_path / "cars.mkv"
    image = np.full((960, 1280, 3), 45, np.uint8)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 1280, 960, "bgr0"
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="bgr24")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_vehicle_proposal_is_metric_but_experimental_and_unconfirmed(vehicle_video, tmp_path):
    record, first = create_draft(vehicle_video, tmp_path)
    detector = KnownCars()
    result = generate_vehicle_proposals(vehicle_video, first, segmenter=detector, seconds=0)
    assert result.candidates, result.diagnostics["reason"]
    assert detector.calls == 1
    assert not (tmp_path / "data/calibration").exists()
    candidate = result.candidates[0]
    assert candidate.width.origin == candidate.length.origin == "experimental"
    assert candidate.width.value > 1 and candidate.length.value > 1
    assert not candidate.width.user_confirmed and not candidate.length.user_confirmed
    assert result.diagnostics["metric_scale_available"]
    draft = edit_record(record, **candidate_changes(candidate, result))
    assert draft.automation.algorithm_version == "mean-vehicle-cuboid-v1"
    assert geometric_quality(draft) <= 0.6
    with pytest.raises(ValueError, match="confirmed provenance"):
        confirm_record(draft)
    confirmed = confirm_record(
        edit_record(
            draft,
            width=draft.width.model_copy(update={"user_confirmed": True}),
            length=draft.length.model_copy(update={"user_confirmed": True}),
        )
    )
    path = save_record(confirmed, tmp_path / "archive", reference_image=first)
    saved = load_record(path)
    assert saved.width.origin == saved.length.origin == "experimental"
    assert saved.reference.pixel_sha256 == record.reference.pixel_sha256
    assert saved.automation.diagnostics["parameters"]["dimensions"]["length_m"] == 4.7


@pytest.mark.parametrize("class_id,confidence", [(7, 0.99), (5, 0.99), (2, 0.2)])
def test_trucks_buses_and_weak_detections_cannot_supply_the_car_scale(
    vehicle_video, tmp_path, class_id, confidence
):
    _, first = create_draft(vehicle_video, tmp_path)
    result = generate_vehicle_proposals(
        vehicle_video, first, segmenter=KnownCars(class_id, confidence), seconds=0
    )
    assert not result.candidates and not result.diagnostics["metric_scale_available"]
    assert result.diagnostics["discarded"]["classe_o_confidenza"] == 6


def test_first_frame_identity_is_checked_before_vehicle_inference(vehicle_video, tmp_path):
    _, first = create_draft(vehicle_video, tmp_path)
    detector = KnownCars()
    with pytest.raises(ValueError, match="video è cambiato"):
        generate_vehicle_proposals(vehicle_video, np.zeros_like(first), segmenter=detector)
    assert detector.calls == 0


def test_local_weights_are_required_without_downloads(vehicle_video, tmp_path):
    _, first = create_draft(vehicle_video, tmp_path)
    with pytest.raises(ValueError, match="modello di segmentazione locale"):
        generate_vehicle_proposals(
            vehicle_video, first, model_path=tmp_path / "missing.pt", seconds=0
        )


def test_roi_prevents_cars_outside_the_selected_road_from_setting_scale(vehicle_video, tmp_path):
    _, first = create_draft(vehicle_video, tmp_path)
    result = generate_vehicle_proposals(
        vehicle_video,
        first,
        [[0, 0], [150, 0], [150, 900], [0, 900]],
        segmenter=KnownCars(),
        seconds=0,
    )
    assert not result.candidates


def test_vehicle_editor_requires_experimental_confirmation_and_keeps_dimensions_on_reopen(
    vehicle_video, tmp_path, monkeypatch
):
    from streamlit.testing.v1 import AppTest
    from test_calibration_ui import button, editor_app

    model = tmp_path / "example-seg.pt"
    model.write_bytes(b"test adapter only")
    monkeypatch.setattr(
        "cctv_incident.calibration_proposal_ui._vehicle_model_paths", lambda root: [model]
    )
    calls = []

    def generate(source, reference, roi=None, **kwargs):
        calls.append(kwargs)
        return generate_vehicle_proposals(
            source,
            reference,
            roi,
            dimensions=kwargs["dimensions"],
            seconds=0,
            segmenter=KnownCars(),
        )

    monkeypatch.setattr(
        "cctv_incident.calibration_proposal_ui.generate_vehicle_proposals", generate
    )
    app = AppTest.from_function(editor_app, args=(str(vehicle_video), str(tmp_path))).run(
        timeout=20
    )
    for label, value in (
        ("Lunghezza auto tipo (m)", 5.17),
        ("Larghezza auto tipo (m)", 1.98),
        ("Altezza auto tipo (m)", 1.65),
    ):
        next(w for w in app.number_input if w.label == label).set_value(value)
    next(
        w for w in app.number_input if w.label == "Secondi iniziali per osservare le auto"
    ).set_value(0.0)
    app.run()
    button(app, "Usa automobili (sperimentale)").click().run(timeout=20)
    assert not app.exception
    state = app.session_state["single_calibration_editor"]
    assert state.record.width.origin == "experimental"
    assert button(app, "Conferma e salva calibrazione").disabled
    assert calls[0]["dimensions"].length_m == 5.17
    assert not any(w.label == "Preset dimensionale" for w in app.selectbox)
    app.run()
    assert len(calls) == 1
    for checkbox in [w for w in app.checkbox if w.label == "Confermo valore e origine"]:
        checkbox.check()
    next(w for w in app.checkbox if w.label.startswith("Ho verificato il fotogramma")).check()
    app.run()
    # The confirmation changes the epoch, which requires renewed visual review.
    next(w for w in app.checkbox if w.label.startswith("Ho verificato il fotogramma")).check().run()
    assert not button(app, "Conferma e salva calibrazione").disabled
    button(app, "Conferma e salva calibrazione").click().run()
    assert not app.exception and not button(app, "Analisi di prova").disabled
    saved = load_record(tmp_path / "data/calibration/videos/cars.yaml")
    assert saved.width.origin == "experimental" and saved.runtime.confidence <= 0.6
    reopened = AppTest.from_function(editor_app, args=(str(vehicle_video), str(tmp_path))).run(
        timeout=20
    )
    assert not reopened.exception
    assert (
        reopened.session_state["single_calibration_editor"].record.automation.algorithm_version
        == "mean-vehicle-cuboid-v1"
    )
    component = json.loads(reopened.get("bidi_component")[0].proto.json)
    assert component["width_m"] == saved.width.value
    assert (
        next(w for w in reopened.number_input if w.label == "Lunghezza auto tipo (m)").value == 5.17
    )
    assert (
        next(
            w for w in reopened.number_input if w.label == "Secondi iniziali per osservare le auto"
        ).value
        == 0.0
    )

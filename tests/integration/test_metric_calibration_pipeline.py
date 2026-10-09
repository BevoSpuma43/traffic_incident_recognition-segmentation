"""Audit regressions for video-bound calibration and partial phase 4 integration."""

import json

import av
import cv2
import numpy as np
import pytest

from cctv_incident.calibration import load_calibration
from cctv_incident.calibration.records import Distance, confirm_record, edit_record
from cctv_incident.calibration.repository import create_draft, load_record, save_record
from cctv_incident.calibration.runtime import load_run_calibration
from cctv_incident.pipeline import Pipeline
from cctv_incident.replay import export_replay, list_saved_runs


class EmptySegmenter:
    def predict(self, image):
        return []


def write_video(path, value=50):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for index in range(6):
            frame = av.VideoFrame.from_ndarray(
                np.full((360, 640, 3), value + index, np.uint8), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def metric_config(tmp_path, config, monkeypatch):
    video = tmp_path / "clip.mp4"
    write_video(video)
    draft, image = create_draft(video, tmp_path)
    record = confirm_record(
        edit_record(
            draft,
            vertices=[
                {"id": f"P{i}", "x": x, "y": y}
                for i, (x, y) in enumerate([(0, 0), (639, 0), (639, 359), (0, 359)], 1)
            ],
            width=Distance(value=31.95, origin="measured", user_confirmed=True),
            length=Distance(
                value=17.95, origin="experimental", source="audit fixture", user_confirmed=True
            ),
            geometric_quality=0.79,
        )
    )
    path = save_record(record, tmp_path / "archive", reference_image=image)
    saved = load_record(path)
    config.project.root_dir = tmp_path
    config.project.output_dir = tmp_path / "outputs"
    config.video.source = str(video)
    config.video.target_fps = 10
    config.video.max_width = 321
    config.perception.backend = "synthetic"
    config.calibration.file = path
    config.calibration.camera_id = saved.camera_id
    config.calibration.expected_record_id = saved.record_id
    config.calibration.expected_revision = saved.revision
    config.calibration.detect_camera_motion = False
    monkeypatch.setattr("cctv_incident.pipeline.hardware_info", lambda: {})
    return config


def test_record_resize_and_portable_provenance(metric_config):
    cfg = metric_config
    summary = Pipeline(cfg, EmptySegmenter()).run()
    assert summary["evaluable"] and summary["completed"] and summary["events"] == 0
    run_dir = cfg.project.output_dir / "runs" / summary["run_id"]
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["original_image_size"] == [640, 360]
    assert metadata["processed_image_size"] == [321, 181]
    assert metadata["calibration_transform"]["scale"] == [321 / 640, 181 / 360]
    assert metadata["calibration_provenance"]["revision"] == cfg.calibration.expected_revision
    assert metadata["calibration_provenance"]["length"]["origin"] == "experimental"
    original = load_calibration(run_dir / "calibration-original.yaml")
    effective = load_calibration(run_dir / "calibration-effective.yaml")
    for point in [(0, 0), (639, 359), (213, 170)]:
        np.testing.assert_allclose(
            effective.transform(np.asarray(point) * [321 / 640, 181 / 360]),
            original.transform(point),
            atol=1e-6,
        )
    # Portable snapshot works after removing the entire central archive from view.
    cfg.calibration.file.parent.rename(cfg.calibration.file.parent.with_name("archive-moved"))
    portable = load_record(run_dir / "calibration-record.yaml")
    assert portable.record_id == cfg.calibration.expected_record_id
    assert portable.runtime.image_size == (640, 360)
    reference = cv2.imread(str(run_dir / effective.reference_image))
    assert reference.shape[1::-1] == (321, 181)


@pytest.mark.parametrize(
    "fault", ["revision", "record_id", "video", "canonical", "confidence", "draft"]
)
def test_metric_preflight_rejects_before_creating_run(metric_config, fault):
    cfg = metric_config
    if fault == "revision":
        cfg.calibration.expected_revision += 1
    elif fault == "record_id":
        cfg.calibration.expected_record_id = "0" * 32
    elif fault == "video":
        write_video(cfg.project.root_dir / "other.mp4", value=100)
        cfg.video.source = str(cfg.project.root_dir / "other.mp4")
    elif fault == "confidence":
        cfg.calibration.min_confidence = 0.9
    elif fault == "draft":
        save_record(edit_record(load_record(cfg.calibration.file)), cfg.calibration.file.parent)
    else:
        legacy = load_calibration(cfg.calibration.file)
        legacy.units = "canonical"
        legacy.save(cfg.calibration.file)
    with pytest.raises(ValueError):
        Pipeline(cfg, EmptySegmenter()).run()
    assert not (cfg.project.output_dir / "runs").exists()


def test_reference_changed_during_preflight_is_rejected(metric_config, monkeypatch):
    from cctv_incident.calibration import runtime

    cfg = metric_config
    inspect = runtime.inspect_video
    record = load_record(cfg.calibration.file)

    def replace_reference(*args):
        identity = inspect(*args)
        replacement = np.full((360, 640, 3), 200, np.uint8)
        cv2.imwrite(str(cfg.calibration.file.parent / record.reference.image_path), replacement)
        return identity

    monkeypatch.setattr(runtime, "inspect_video", replace_reference)
    with pytest.raises(ValueError, match="Reference image"):
        load_run_calibration(cfg)


@pytest.mark.parametrize("mode", ["metric", "image"])
def test_camera_motion_produces_non_evaluable_summary_and_replay(metric_config, monkeypatch, mode):
    cfg = metric_config
    cfg.events.coordinate_mode = mode
    cfg.calibration.detect_camera_motion = True
    observed = []

    class MovedCamera:
        def __init__(self, reference, threshold):
            observed.append((reference.shape[1::-1], threshold))

        def update(self, image, exclusion):
            return True

    monkeypatch.setattr("cctv_incident.pipeline.CameraMotionGuard", MovedCamera)
    summary = Pipeline(cfg, EmptySegmenter()).run()
    assert observed[0][0] == (321, 181)
    if mode == "metric":
        assert observed[0][1] == pytest.approx(cfg.calibration.motion_threshold_px * 321 / 640)
    assert summary["completed"] and not summary["calibration_valid"]
    assert not summary["evaluable"]
    assert summary["non_evaluable_reasons"] == ["camera_motion"]
    assert summary["events"] == 0 and summary["track_observations"] == 0
    assert list_saved_runs(cfg.project.output_dir)[0]["evaluable"] is False
    replay = export_replay(summary["run_dir"])
    assert replay["evaluable"] is False
    assert replay["non_evaluable_reasons"] == ["camera_motion"]


def test_replay_rejects_replaced_video_with_same_resolution(metric_config):
    cfg = metric_config
    summary = Pipeline(cfg, EmptySegmenter()).run()
    write_video(cfg.video.source, value=190)
    with pytest.raises(ValueError, match="[Vv]ideo.*(cambiato|differs)"):
        export_replay(summary["run_dir"])


def test_native_resolution_change_on_skipped_frame_invalidates(metric_config, monkeypatch):
    from cctv_incident.video import VideoSource

    cfg = metric_config
    cfg.video.target_fps = 2
    decode = VideoSource._decode

    def changed_decode(self):
        for packet in decode(self):
            if packet.frame_index == 1:  # At 0.1s: not selected for inference at 2 fps.
                packet.original_image_size = (1280, 720)
            yield packet

    monkeypatch.setattr(VideoSource, "_decode", changed_decode)
    summary = Pipeline(cfg, EmptySegmenter()).run()
    assert not summary["evaluable"]
    assert summary["calibration_invalidations"] == [
        {"reason": "decoded_resolution_changed", "timestamp_s": 0.1, "frame_index": 1}
    ]
    assert summary["state_counts"]["PAUSED"] >= 1


def test_legacy_processed_calibration_remains_supported(metric_config):
    from cctv_incident.calibration.coordinates import ImageTransform

    cfg = metric_config
    legacy = ImageTransform.resize((640, 360), (321, 181)).adapt_calibration(
        load_calibration(cfg.calibration.file)
    )
    legacy.reference_image = None
    legacy.save(cfg.calibration.file)
    cfg.calibration.expected_revision = cfg.calibration.expected_record_id = None
    summary = Pipeline(cfg, EmptySegmenter()).run()
    assert summary["evaluable"]
    metadata = json.loads(
        (cfg.project.output_dir / "runs" / summary["run_id"] / "run.json").read_text()
    )
    assert metadata["calibration_transform"]["coordinate_basis"] == "legacy_processed_pixels"


def controlled_motion_config(cfg, *, shift_camera=False):
    """Known simulated 0.05 m/px plane, textured static background and a 5 m/s target."""
    video = cfg.project.root_dir / "controlled.mkv"
    gray = np.random.default_rng(17).integers(60, 180, (360, 640), dtype=np.uint8)
    background = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    with av.open(str(video), "w") as container:
        # Lossless RGB keeps the analytic rectangle intact: MPEG-4 chroma artifacts
        # alter the color mask and confound geometry error with segmentation error.
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "bgr0"
        for index in range(15):
            pixels = background.copy()
            center = 100 + index * 10
            cv2.rectangle(pixels, (center - 20, 196), (center + 20, 236), (0, 230, 0), -1)
            if shift_camera and index >= 4:
                pixels = cv2.warpAffine(pixels, np.float32([[1, 0, 36], [0, 1, 24]]), (640, 360))
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    previous = load_record(cfg.calibration.file)
    draft, reference = create_draft(video, cfg.project.root_dir)
    # These are declared simulation dimensions, not measured dimensions of a real road.
    record = confirm_record(
        edit_record(
            draft,
            vertices=previous.vertices,
            width=Distance(
                value=31.95,
                origin="experimental",
                source="simulated 0.05 m/px",
                user_confirmed=True,
            ),
            length=Distance(
                value=17.95,
                origin="experimental",
                source="simulated 0.05 m/px",
                user_confirmed=True,
            ),
            geometric_quality=0.79,
        )
    )
    cfg.calibration.file = save_record(
        record, cfg.project.root_dir / "controlled-archive", reference_image=reference
    )
    record = load_record(cfg.calibration.file)
    cfg.calibration.camera_id = record.camera_id
    cfg.calibration.expected_record_id = record.record_id
    cfg.calibration.expected_revision = record.revision
    cfg.calibration.detect_camera_motion = True
    cfg.calibration.motion_check_s = 0.1
    cfg.video.source = str(video)
    cfg.features.ground_point_method = "box"
    cfg.features.ema_alpha = 1
    return cfg


@pytest.mark.parametrize("max_width", [None, 321])
def test_controlled_metric_trajectory_with_real_tracking_and_motion_guard(metric_config, max_width):
    cfg = controlled_motion_config(metric_config)
    cfg.video.max_width = max_width
    previews = []
    summary = Pipeline(cfg).run(lambda packet: previews.append(packet["frame"].shape))
    assert summary["evaluable"] and summary["calibration_valid"]
    assert summary["processed_frames"] == 15 and summary["track_observations"] == 15
    assert len(previews) == 15
    run = cfg.project.output_dir / "runs" / summary["run_id"]
    rows = [json.loads(line) for line in (run / "trajectories.jsonl").read_text().splitlines()]
    assert len({row["track_id"] for row in rows}) == 1
    # Oracle is the generated target trajectory, independent of serialized H or T.
    expected = np.array([[(100.5 + 10 * row["frame_index"]) * 0.05, 237 * 0.05] for row in rows])
    observed = np.array([row["point_world"] for row in rows])
    assert np.max(np.linalg.norm(observed - expected, axis=1)) < 0.15
    speed = (observed[-1, 0] - observed[0, 0]) / (rows[-1]["timestamp_s"] - rows[0]["timestamp_s"])
    assert speed == pytest.approx(5, abs=0.15)
    assert all(row["coordinate_units"] == "m" for row in rows)
    replay = export_replay(run)
    assert replay["evaluable"] and replay["frames"] == 15


def test_real_camera_shift_invalidates_reduced_metric_pipeline(metric_config):
    cfg = controlled_motion_config(metric_config, shift_camera=True)
    summary = Pipeline(cfg).run()
    assert summary["completed"] and not summary["evaluable"]
    assert summary["non_evaluable_reasons"] == ["camera_motion"]
    assert summary["events"] == 0
    invalidation = summary["calibration_invalidations"][0]
    assert 0.4 <= invalidation["timestamp_s"] <= 0.8
    run = cfg.project.output_dir / "runs" / summary["run_id"]
    features = [json.loads(line) for line in (run / "features.jsonl").read_text().splitlines()]
    paused = [row for row in features if row["timestamp_s"] >= invalidation["timestamp_s"]]
    assert paused and all(
        not row["calibration_valid"] and row["state"] == "PAUSED" for row in paused
    )
    trajectories = [
        json.loads(line) for line in (run / "trajectories.jsonl").read_text().splitlines()
    ]
    assert trajectories and all(
        row["timestamp_s"] < invalidation["timestamp_s"] for row in trajectories
    )


@pytest.mark.parametrize("finish", ["stop", "error"])
def test_incomplete_metric_analysis_is_persisted_as_non_evaluable(metric_config, finish):
    cfg = metric_config

    class FailingSegmenter(EmptySegmenter):
        calls = 0

        def predict(self, image):
            self.calls += 1
            if self.calls == 2 and finish == "error":
                raise RuntimeError("test inference failure")
            return []

    segmenter = FailingSegmenter()
    pipeline = Pipeline(cfg, segmenter)
    if finish == "error":
        with pytest.raises(RuntimeError, match="test inference failure"):
            pipeline.run()
    else:
        pipeline.run(stop_requested=lambda: segmenter.calls >= 2)
    summary = pipeline.summary
    assert not summary["completed"] and not summary["evaluable"]
    reason = "analysis_stopped" if finish == "stop" else "analysis_error"
    assert summary["non_evaluable_reasons"] == [reason]
    run = cfg.project.output_dir / "runs" / summary["run_id"]
    for filename in ("run.json", "metrics.json"):
        stored = json.loads((run / filename).read_text())
        assert stored["evaluation_status"] == "not_evaluable"
        assert stored["non_evaluable_reasons"] == [reason]

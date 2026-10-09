"""Metric batch snapshots, refusal to score invalid runs and independent experiments."""

import csv
import os
import time
from pathlib import Path

import av
import numpy as np
import pytest

from cctv_incident import batch
from cctv_incident.calibration.batch_snapshot import InvalidBatchCalibration, archive_mapping
from cctv_incident.calibration.records import Distance, confirm_record, edit_record
from cctv_incident.calibration.repository import create_draft, load_record, save_record
from cctv_incident.calibration.runtime import load_run_calibration


class EmptySegmenter:
    def predict(self, image):
        return []


@pytest.fixture
def metric_inputs(tmp_path, config, monkeypatch):
    folder = tmp_path / "dataset/standard_dataset"
    folder.mkdir(parents=True)
    archive = tmp_path / "data/calibration/videos"
    mapping = {}
    for index, name in enumerate(("a.mp4", "b.mp4")):
        source = folder / name
        with av.open(str(source), "w") as container:
            stream = container.add_stream("mpeg4", rate=10)
            stream.width, stream.height, stream.pix_fmt = 160, 90, "yuv420p"
            for frame_index in range(6):
                frame = av.VideoFrame.from_ndarray(
                    np.full((90, 160, 3), 30 + index * 70 + frame_index, np.uint8),
                    format="bgr24",
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        record, image = create_draft(source, tmp_path)
        record = confirm_record(
            edit_record(
                record,
                vertices=[
                    {"id": f"P{i}", "x": x, "y": y}
                    for i, (x, y) in enumerate([(0, 0), (159, 0), (159, 89), (0, 89)], 1)
                ],
                width=Distance(value=8 + index, origin="measured", user_confirmed=True),
                length=Distance(
                    value=5 + index, origin="experimental", source="fixture", user_confirmed=True
                ),
                geometric_quality=0.8,
            )
        )
        mapping[name] = save_record(record, archive, reference_image=image)
    metadata = folder.parent / "metadata-real.csv"
    metadata.write_text(
        "path,type,accident_time,duration\na.mp4,normal,,0.6\nb.mp4,normal,,0.6\n",
        encoding="utf-8",
    )
    config.project.root_dir = tmp_path
    config.perception.backend = "synthetic"
    config.perception.model = tmp_path / "synthetic.pt"
    config.events.coordinate_mode = "metric"
    config.calibration.detect_camera_motion = False
    config.video.target_fps = 10
    monkeypatch.setattr("cctv_incident.pipeline.hardware_info", lambda: {})
    return config, folder, metadata, mapping


def prepare(inputs):
    cfg, folder, metadata, mapping = inputs
    return batch.prepare_job(cfg, folder, metadata, calibration_map=mapping)


def summary(cfg, **overrides):
    return {
        "run_id": Path(cfg.video.source).stem,
        "stopped": False,
        "completed": True,
        "error": None,
        "calibration_valid": True,
        "metric_calibration_available": True,
        "evaluable": True,
        "source_duration_s": 0.6,
        "elapsed_s": 0.1,
        **overrides,
    }


def test_two_models_two_modes_use_distinct_experiments_and_correct_snapshots(metric_inputs):
    cfg, folder, metadata, mapping = metric_inputs
    identities, jobs, models_loaded = [], [], []
    seen = []

    class InspectPipeline:
        def __init__(self, config, segmenter):
            self.cfg = config
            assert segmenter is models_loaded[-1]

        def run(self, **kwargs):
            if self.cfg.events.coordinate_mode == "metric":
                run = load_run_calibration(self.cfg)
                seen.append((Path(self.cfg.video.source).name, run.record.width.value))
                assert self.cfg.calibration.file.is_relative_to(jobs[-1])
                assert self.cfg.calibration.file not in mapping.values()
            return summary(self.cfg)

    def model_factory(_):
        model = object()
        models_loaded.append(model)
        return model

    for model in ("small.pt", "medium.pt"):
        cfg.perception.model = cfg.project.root_dir / model
        cfg.perception.model.write_bytes(model.encode())
        for mode in ("image", "metric"):
            cfg.events.coordinate_mode = mode
            job = batch.prepare_job(
                cfg, folder, metadata, calibration_map=mapping if mode == "metric" else None
            )
            jobs.append(job)
            manifest = batch.read_json(job / "manifest.json")
            assert manifest["schema_version"] == 2
            assert manifest["coordinate_mode"] == mode
            identities.append(manifest["protocol_sha256"])
            batch.run_job(job, InspectPipeline, model_factory)
            assert batch.snapshot(job)["status"] == "completed"
            assert batch.read_json(job / "metrics.json")["video"]["true_negatives"] == 2
    assert len(set(jobs)) == len(set(identities)) == len(models_loaded) == 4
    assert seen == [("a.mp4", 8), ("b.mp4", 9)] * 2
    assert {item["coordinate_mode"] for item in batch.list_jobs(jobs[0].parent)} == {
        "image",
        "metric",
    }


def test_real_pipeline_works_after_archive_removed_and_csv_records_provenance(metric_inputs):
    cfg, _, _, mapping = metric_inputs
    job = prepare(metric_inputs)
    archive = next(iter(mapping.values())).parent
    archive.rename(archive.with_name("archive-moved"))
    batch.validate_manifest(batch.read_json(job / "manifest.json"), job)
    batch.run_job(job, segmenter_factory=lambda _: EmptySegmenter())
    assert batch.snapshot(job)["status"] == "completed"
    results = batch.completed_results(job)
    assert all(r["summary"]["evaluable"] for r in results)
    assert len({r["calibration"]["record_id"] for r in results}) == 2
    with (job / "videos.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row, result in zip(rows, results, strict=True):
        assert row["coordinate_mode"] == "metric"
        assert row["calibration_sha256"] == batch.sha256(job / row["calibration_path"])
        assert row["calibration_record_id"] == result["calibration"]["record_id"]
        assert row["width_origin"] == "measured"
        assert row["length_origin"] == "experimental"
        assert row["explicit_scale_origin"] == ""
    assert cfg.calibration.camera_id == "camera_01"  # Caller configuration was not mutated.


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "draft", "wrong_video", "confidence", "reference"]
)
def test_all_calibrations_must_pass_before_job_is_published(metric_inputs, fault):
    cfg, _, _, mapping = metric_inputs
    if fault == "missing":
        del mapping["b.mp4"]
    elif fault == "extra":
        mapping["unexpected.mp4"] = mapping["a.mp4"]
    elif fault == "draft":
        path = mapping["b.mp4"]
        save_record(edit_record(load_record(path)), path.parent)
    elif fault == "wrong_video":
        mapping["b.mp4"] = mapping["a.mp4"]
    elif fault == "confidence":
        cfg.calibration.min_confidence = 0.95
    else:
        path = mapping["b.mp4"]
        (path.parent / load_record(path).reference.image_path).write_bytes(b"bad png")
    with pytest.raises(InvalidBatchCalibration):
        prepare(metric_inputs)
    root = cfg.project.root_dir / "outputs/batches"
    assert batch.list_jobs(root) == []
    assert not list(root.glob("*/checkpoint.json"))
    assert not list(root.glob(".preparing-*"))


@pytest.mark.parametrize("target", ["record", "reference", "source", "missing"])
def test_snapshot_tampering_blocks_before_model_and_requires_new_experiment(metric_inputs, target):
    job = prepare(metric_inputs)
    manifest = batch.read_json(job / "manifest.json")
    data = manifest["videos"][1]["calibration"]
    suffix = {
        "record": "calibration-record.yaml",
        "source": "calibration-source.yaml",
        "reference": ".png",
        "missing": ".png",
    }[target]
    path = job / next(name for name in data["files"] if name.endswith(suffix))
    if target == "missing":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(InvalidBatchCalibration):
        batch.run_job(job, segmenter_factory=lambda _: pytest.fail("Model must not load"))
    assert batch.snapshot(job)["requires_new_experiment"]
    assert not batch.completed_results(job)
    with pytest.raises(InvalidBatchCalibration):
        batch.start_job(job)


def test_metric_stop_resume_skips_commits_and_uses_original_archive_revision(metric_inputs):
    job = prepare(metric_inputs)
    calls, revisions = [], []

    class InterruptedPipeline:
        def __init__(self, cfg, segmenter):
            self.cfg = cfg

        def run(self, stop_requested, **kwargs):
            name = Path(self.cfg.video.source).name
            calls.append(name)
            revisions.append(load_run_calibration(self.cfg).record.revision)
            if name == "b.mp4" and calls.count(name) == 1:
                batch.stop_job(job)
            stopped = stop_requested()
            return summary(self.cfg, stopped=stopped, completed=not stopped, evaluable=not stopped)

    batch.run_job(job, InterruptedPipeline, lambda _: object())
    assert batch.snapshot(job)["status"] == "paused"
    assert len(batch.completed_results(job)) == 1
    path = metric_inputs[3]["b.mp4"]
    old = load_record(path)
    save_record(edit_record(old), path.parent)  # Central archive is now an unconfirmed revision.
    (job / "videos.csv").unlink()
    (job / "stop.request").unlink()
    batch.run_job(job, InterruptedPipeline, lambda _: object())
    batch.run_job(job, InterruptedPipeline, lambda _: object())
    assert calls == ["a.mp4", "b.mp4", "b.mp4"]
    assert revisions == [old.revision] * 3
    assert batch.snapshot(job)["status"] == "completed"
    assert len(batch.completed_results(job)) == 2
    with (job / "videos.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_real_camera_motion_stops_without_committing_negative(metric_inputs, monkeypatch):
    cfg = metric_inputs[0]
    cfg.calibration.detect_camera_motion = True
    calls = []

    class MovedCamera:
        def __init__(self, reference, threshold):
            pass

        def update(self, image, exclusion):
            calls.append(True)
            return True

    monkeypatch.setattr("cctv_incident.pipeline.CameraMotionGuard", MovedCamera)
    job = prepare(metric_inputs)
    with pytest.raises(InvalidBatchCalibration, match="camera_motion"):
        batch.run_job(job, segmenter_factory=lambda _: EmptySegmenter())
    assert len(calls) == 1  # The worker stops at the next frame, not at end of the video.
    assert not batch.completed_results(job)
    assert batch.read_json(job / "metrics.json")["completed_videos"] == 0
    assert batch.snapshot(job)["requires_new_experiment"]


def test_tampering_during_inference_prevents_commit(metric_inputs):
    job = prepare(metric_inputs)

    class TamperingPipeline:
        def __init__(self, cfg, segmenter):
            self.cfg = cfg

        def run(self, **kwargs):
            self.cfg.calibration.file.write_bytes(b"changed during inference")
            return summary(self.cfg)

    with pytest.raises(InvalidBatchCalibration):
        batch.run_job(job, TamperingPipeline, lambda _: object())
    assert not batch.completed_results(job)


def test_video_hash_detects_change_with_same_size_and_mtime(metric_inputs):
    job = prepare(metric_inputs)
    source = metric_inputs[1] / "a.mp4"
    stat = source.stat()
    content = bytearray(source.read_bytes())
    content[-1] ^= 1
    source.write_bytes(content)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="Video modificato"):
        batch.run_job(job, segmenter_factory=lambda _: pytest.fail("Model must not load"))
    assert not batch.completed_results(job)


def test_archive_mapping_lists_missing_calibrations_without_skipping(metric_inputs):
    cfg, folder, _, mapping = metric_inputs
    assert archive_mapping(folder, cfg.project.root_dir, list(mapping)) == mapping
    path = mapping["b.mp4"]
    save_record(edit_record(load_record(path)), path.parent)
    with pytest.raises(InvalidBatchCalibration, match="b.mp4"):
        archive_mapping(folder, cfg.project.root_dir, list(mapping))


def test_legacy_exports_leave_missing_provenance_empty(metric_inputs):
    cfg, folder, metadata, _ = metric_inputs
    cfg.events.coordinate_mode = "image"
    job = batch.prepare_job(cfg, folder, metadata)
    manifest = batch.read_json(job / "manifest.json")
    manifest["schema_version"] = 1
    del manifest["coordinate_mode"]
    for video in manifest["videos"]:
        del video["sha256"]
    batch.write_json(job / "manifest.json", manifest)
    batch.run_job(job, segmenter_factory=lambda _: EmptySegmenter())
    for path in (job / "results").glob("*.json"):
        result = batch.read_json(path)
        result.pop("calibration")
        result.pop("coordinate_mode")
        batch.write_json(path, result)
    batch.export_results(job)
    with (job / "videos.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(
        row["coordinate_mode"] == row["calibration_sha256"] == row["width_origin"] == ""
        for row in rows
    )


def test_metric_ui_separates_modes_and_preserves_preparation(metric_inputs, monkeypatch):
    from streamlit.testing.v1 import AppTest

    cfg, folder, metadata, mapping = metric_inputs
    models = cfg.project.root_dir / "models"
    models.mkdir()
    model = models / "fixture-seg.pt"
    model.write_bytes(b"test model")
    cfg.perception.model = model
    image_cfg = cfg.model_copy(deep=True)
    image_cfg.events.coordinate_mode = "image"
    image_job = batch.prepare_job(image_cfg, folder, metadata)
    monkeypatch.setattr("cctv_incident.config.load_config", lambda _: cfg.model_copy(deep=True))
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    next(x for x in app.selectbox if x.label == "Modalita").select(
        "standard_dataset analisi in batch - con omografia"
    ).run(timeout=20)
    assert not app.exception
    assert not any(x.label == "Prepara batch" for x in app.button)
    app.radio(key="metric_batch_stage").set_value("Analisi batch").run(timeout=20)
    assert not any(x.label == "Esperimento salvato per questo modello" for x in app.selectbox)
    next(x for x in app.button if x.label == "Prepara batch").click().run(timeout=20)
    assert not app.exception
    metric_job = app.session_state["metric_batch_selected_job"]
    assert metric_job != str(image_job)
    assert batch.read_json(Path(metric_job) / "manifest.json")["coordinate_mode"] == "metric"
    selected = app.selectbox(key="metric_batch_selected_job")
    assert len(selected.options) == 1
    next(x for x in app.selectbox if x.label == "Modalita").select(
        "standard_dataset analisi in batch - no omografia"
    ).run(timeout=20)
    assert not app.exception
    assert app.session_state["batch_selected_job"] == str(image_job)
    assert len(app.selectbox(key="batch_selected_job").options) == 1


def test_copied_reference_is_verified_before_publication(metric_inputs, monkeypatch):
    from cctv_incident.calibration.runtime import RunCalibration

    original = RunCalibration.write_original

    def corrupt_copy(self, directory):
        original(self, directory)
        (directory / "calibration-reference-original.png").write_bytes(b"bad copy")

    monkeypatch.setattr(RunCalibration, "write_original", corrupt_copy)
    with pytest.raises(InvalidBatchCalibration, match="Reference image"):
        prepare(metric_inputs)
    assert batch.list_jobs(metric_inputs[0].project.root_dir / "outputs/batches") == []


@pytest.mark.parametrize("evaluable", [False, None])
def test_completed_metric_run_requires_explicit_evaluable_result(metric_inputs, evaluable):
    job = prepare(metric_inputs)

    class NonEvaluablePipeline:
        def __init__(self, cfg, segmenter):
            self.cfg = cfg

        def run(self, **kwargs):
            result = summary(self.cfg, evaluable=evaluable)
            if evaluable is None:
                del result["evaluable"]
            return result

    with pytest.raises(RuntimeError, match="non valutabile"):
        batch.run_job(job, NonEvaluablePipeline, lambda _: object())
    assert not batch.completed_results(job)
    assert batch.read_json(job / "metrics.json")["completed_videos"] == 0


def test_metric_subprocess_stop_resume_uses_snapshots(metric_inputs):
    cfg, folder, metadata, mapping = metric_inputs
    output = cfg.project.root_dir / "worker-batches"
    cfg.project.root_dir = Path.cwd()
    job = batch.prepare_job(cfg, folder, metadata, output, calibration_map=mapping)

    def await_exit():
        deadline = time.monotonic() + 45
        while batch.worker_alive(job) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not batch.worker_alive(job), (job / "worker.log").read_text()

    try:
        batch.start_job(job)
        batch.stop_job(job)
        await_exit()
        assert batch.snapshot(job)["status"] == "paused", (job / "worker.log").read_text()
        archive = mapping["a.mp4"].parent
        archive.rename(archive.with_name("hidden-archive"))
        batch.start_job(job)
        await_exit()
        assert batch.snapshot(job)["status"] == "completed", (job / "worker.log").read_text()
        assert len(batch.completed_results(job)) == 2
        assert all(r["summary"]["evaluable"] for r in batch.completed_results(job))
    finally:
        batch.stop_job(job)
        await_exit()

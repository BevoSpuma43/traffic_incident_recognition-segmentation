import csv
import time
from pathlib import Path

import pytest

from cctv_incident.batch import (
    completed_results,
    exclusive_lock,
    prepare_job,
    read_json,
    run_job,
    snapshot,
    start_job,
    stop_job,
    worker_alive,
)
from cctv_incident.config import AppConfig


@pytest.fixture
def batch_inputs(tmp_path):
    folder = tmp_path / "subset"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (folder / name).write_bytes(b"fake video")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "path,type,accident_time,duration\nreal_videos/a.mp4,t-bone,1,3\n"
        "real_videos/b.mp4,rear-end,1,3\n",
        encoding="utf-8",
    )
    config = AppConfig()
    config.project.root_dir = tmp_path
    config.events.coordinate_mode = "image"
    config.perception.backend = "synthetic"
    config.calibration.detect_camera_motion = False
    return config, folder, metadata


def fake_summary(cfg, stopped=False):
    return {
        "run_id": Path(cfg.video.source).stem,
        "stopped": stopped,
        "completed": not stopped,
        "error": None,
        "source_duration_s": 3.0,
        "elapsed_s": 0.1,
    }


def test_stop_resume_commits_only_complete_videos_and_recovers_csv(batch_inputs):
    job = prepare_job(*batch_inputs)
    saved_configuration = (job / "batch_config.json").read_bytes()
    calls = []

    class FakePipeline:
        def __init__(self, cfg, segmenter):
            self.cfg = cfg

        def run(self, stop_requested, progress_callback):
            name = Path(self.cfg.video.source).name
            calls.append(name)
            progress_callback(
                {"duration_s": 3, "timestamp_s": 1, "run_id": name, "processed_frames": 8}
            )
            if name == "b.mp4" and calls.count(name) == 1:
                stop_job(job)
            return fake_summary(self.cfg, stop_requested())

    run_job(job, FakePipeline, lambda cfg: object())
    assert snapshot(job)["status"] == "paused"
    assert len(completed_results(job)) == 1
    assert read_json(job / "metrics.json")["completed_videos"] == 1
    # Simulate losing an export after committing a result: the next run rebuilds it.
    (job / "videos.csv").unlink()
    (job / "stop.request").unlink()
    run_job(job, FakePipeline, lambda cfg: object())
    assert calls == ["a.mp4", "b.mp4", "b.mp4"]
    assert snapshot(job)["status"] == "completed"
    assert read_json(job / "metrics.json")["video"]["false_negatives"] == 2
    with (job / "videos.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 2
    run_job(job, FakePipeline, lambda cfg: object())
    assert len(calls) == 3  # Completed work is never rerun or double counted.
    assert (job / "batch_config.json").read_bytes() == saved_configuration


def test_models_and_settings_have_independent_jobs_and_changed_inputs_are_rejected(batch_inputs):
    config, folder, metadata = batch_inputs
    first = prepare_job(config, folder, metadata)
    repeated = prepare_job(config, folder, metadata)
    assert repeated != first
    assert (
        read_json(repeated / "manifest.json")["protocol_sha256"]
        == read_json(first / "manifest.json")["protocol_sha256"]
    )
    config.perception.model = folder / "other-seg.pt"
    config.perception.model.write_bytes(b"model")
    second = prepare_job(config, folder, metadata)
    assert first != second
    config.video.target_fps = 15
    assert prepare_job(config, folder, metadata) not in (first, second)
    (folder / "a.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Video modificato"):
        run_job(first, segmenter_factory=lambda cfg: object())
    assert snapshot(first)["status"] == "error"
    assert not completed_results(first)


def test_batch_saves_gui_settings_original_yaml_and_effective_parameters(batch_inputs):
    import yaml

    config, folder, metadata = batch_inputs
    original = folder.parent / "custom.yaml"
    original.write_text("# Original settings\nvideo:\n  target_fps: 8\n", encoding="utf-8")
    config.video.target_fps = 15
    job = prepare_job(
        config,
        folder,
        metadata,
        mode="standard_dataset analisi in batch - no omografia",
        config_path=original,
        dataset_directory=folder.parent,
        dataset_selection="subset",
    )
    settings = read_json(job / "batch_config.json")
    assert settings["mode"] == "standard_dataset analisi in batch - no omografia"
    assert settings["configuration_yaml"] == str(original.resolve())
    assert settings["yolo_model"] == str(config.perception.model.resolve())
    assert settings["dataset_directory"] == str(folder.parent.resolve())
    assert settings["dataset_selection"] == "subset"
    assert settings["video_directory"] == str(folder.resolve())
    assert settings["labels_csv"] == str(metadata.resolve())
    assert settings["analysis_fps"] == 15
    assert (job / "source_config.yaml").read_text(encoding="utf-8") == original.read_text(
        encoding="utf-8"
    )
    resolved = yaml.safe_load((job / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["video"]["target_fps"] == 15
    assert resolved["project"]["output_dir"] == str(job / "pipeline")
    assert resolved == read_json(job / "manifest.json")["config"]
    original.write_text("video:\n  target_fps: 30\n", encoding="utf-8")
    assert "target_fps: 8" in (job / "source_config.yaml").read_text(encoding="utf-8")


def test_unknown_subset_video_is_not_silently_a_negative(batch_inputs):
    config, folder, metadata = batch_inputs
    (folder / "unknown.mp4").touch()
    with pytest.raises(ValueError, match="senza etichetta"):
        prepare_job(config, folder, metadata)


def test_predictions_from_interrupted_attempt_are_excluded_from_csv(batch_inputs):
    from cctv_incident.storage import EventStorage
    from cctv_incident.types import Event

    config, folder, metadata = batch_inputs
    (folder / "b.mp4").unlink()
    job = prepare_job(config, folder, metadata)
    attempts = []

    class FakePipeline:
        def __init__(self, cfg, segmenter):
            self.cfg = cfg

        def run(self, stop_requested, progress_callback):
            run_id = f"attempt-{len(attempts)}"
            attempts.append(run_id)
            storage = EventStorage(self.cfg.project.output_dir)
            try:
                storage.save(
                    Event(
                        run_id,
                        "camera",
                        0,
                        2.9 if len(attempts) == 1 else 1.0,
                        3,
                        [1, 2],
                        0.9,
                        0,
                        [],
                        run_id=run_id,
                    )
                )
            finally:
                storage.close()
            if len(attempts) == 1:
                stop_job(job)
            return {**fake_summary(self.cfg, stop_requested()), "run_id": run_id}

    run_job(job, FakePipeline, lambda cfg: object())
    assert not completed_results(job)
    (job / "stop.request").unlink()
    run_job(job, FakePipeline, lambda cfg: object())
    with (job / "events.csv").open(newline="") as handle:
        events = list(csv.DictReader(handle))
    assert len(events) == 1
    assert events[0]["run_id"] == "attempt-1"
    assert events[0]["match"] == "TP"
    metrics = read_json(job / "metrics.json")
    assert metrics["event"]["true_positives"] == 1
    assert metrics["event"]["false_positives"] == 0


def test_file_lock_prevents_two_workers(tmp_path):
    path = tmp_path / "worker.lock"
    with exclusive_lock(path), pytest.raises(RuntimeError, match="già in corso"):
        with exclusive_lock(path):
            pytest.fail("Lock must be exclusive")
    with exclusive_lock(path):
        pass  # Released after leaving the context.


def test_real_background_worker_stops_and_resumes_without_ui(tmp_path):
    pytest.importorskip("ultralytics")
    from cctv_incident.demo import generate_demo

    source = generate_demo(tmp_path, negative=True)
    metadata = tmp_path / "labels.csv"
    metadata.write_text(
        f"path,type,accident_time,duration\n{source.name},normal,,8\n", encoding="utf-8"
    )
    # generate_demo also writes the other clip: use a separate subset with one video.
    folder = tmp_path / "subset"
    folder.mkdir()
    import shutil

    shutil.copyfile(source, folder / source.name)
    cfg = AppConfig()
    cfg.project.root_dir = Path.cwd()
    cfg.perception.backend = "synthetic"
    cfg.perception.model = tmp_path / "synthetic.pt"
    cfg.events.coordinate_mode = "image"
    cfg.calibration.detect_camera_motion = False
    job = prepare_job(cfg, folder, metadata, tmp_path / "batches")

    def await_exit():
        deadline = time.monotonic() + 45
        while worker_alive(job) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not worker_alive(job), (job / "worker.log").read_text()

    try:
        start_job(job)
        with pytest.raises(RuntimeError, match="già attivo"):
            start_job(job)
        stop_job(job)
        await_exit()
        assert snapshot(job)["status"] == "paused", (job / "worker.log").read_text()
        assert not completed_results(job)
        start_job(job)
        await_exit()
        assert snapshot(job)["status"] == "completed", (job / "worker.log").read_text()
        assert read_json(job / "metrics.json")["video"]["true_negatives"] == 1
        # Recover even if the last result committed but the final CSV/checkpoint update failed.
        from cctv_incident.batch import write_json

        saved_runs = list((job / "pipeline/runs").iterdir())
        checkpoint = read_json(job / "checkpoint.json")
        write_json(job / "checkpoint.json", {**checkpoint, "status": "running"})
        (job / "metrics.csv").unlink()
        start_job(job)
        await_exit()
        assert snapshot(job)["status"] == "completed"
        assert (job / "metrics.csv").is_file()
        assert list((job / "pipeline/runs").iterdir()) == saved_runs
    finally:
        stop_job(job)
        await_exit()


def test_pipeline_cooperative_stop_mid_video(tmp_path):
    from cctv_incident.demo import generate_demo
    from cctv_incident.pipeline import Pipeline

    cfg = AppConfig()
    cfg.video.source = str(generate_demo(tmp_path, negative=True))
    cfg.project.output_dir = tmp_path / "pipeline"
    cfg.perception.backend = "synthetic"
    cfg.events.coordinate_mode = "image"
    cfg.calibration.detect_camera_motion = False
    progress = []
    summary = Pipeline(cfg).run(
        stop_requested=lambda: len(progress) >= 3,
        progress_callback=progress.append,
    )
    assert summary["stopped"]
    assert not summary["completed"]
    assert summary["processed_frames"] == 3
    assert len(progress) == 3
    assert progress[-1]["duration_s"] > progress[-1]["timestamp_s"]

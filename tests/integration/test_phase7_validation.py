"""Historical read-only access, paired controls and honest proposal review reports."""

import csv
from pathlib import Path

import numpy as np
import pytest
from test_metric_batch import EmptySegmenter
from test_metric_batch import metric_inputs as metric_inputs

from cctv_incident import batch
from cctv_incident.batch_comparison import compare_jobs
from cctv_incident.calibration.evaluation import review_report, sample_proposals
from cctv_incident.calibration.proposals import ProposalCandidate, ProposalResult
from cctv_incident.calibration.repository import create_draft
from cctv_incident.pipeline import Pipeline


def tree_hashes(root):
    return {str(p.relative_to(root)): batch.sha256(p) for p in root.rglob("*") if p.is_file()}


def test_legacy_read_export_and_refused_resume_do_not_modify_job(metric_inputs, monkeypatch):
    cfg, folder, metadata, _ = metric_inputs
    cfg.events.coordinate_mode = "image"
    job = batch.prepare_job(cfg, folder, metadata)
    batch.run_job(job, segmenter_factory=lambda _: EmptySegmenter())
    manifest = batch.read_json(job / "manifest.json")
    manifest.update(schema_version=1, code_sha256="historical-code")
    manifest.pop("coordinate_mode")
    manifest.pop("batch_settings")
    batch.write_json(job / "manifest.json", manifest)
    state = batch.read_json(job / "checkpoint.json")
    batch.write_json(job / "checkpoint.json", {**state, "status": "paused"})
    for path in (job / "results").glob("*.json"):
        result = batch.read_json(path)
        result.pop("coordinate_mode")
        result.pop("calibration")
        batch.write_json(path, result)
    before = tree_hashes(job)
    compatible, message = batch.resume_compatibility(job)
    assert not compatible and "codice originale" in message
    destination = job.parent / "historical-export"
    batch.export_results(job, destination)
    with (destination / "videos.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["calibration_sha256"] == rows[0]["coordinate_mode"] == ""
    monkeypatch.setattr(
        batch.subprocess, "Popen", lambda *a, **k: pytest.fail("No worker may launch")
    )
    with pytest.raises(ValueError, match="codice"):
        batch.start_job(job)
    assert tree_hashes(job) == before


def test_ui_historical_job_remains_readable_without_weights(metric_inputs, monkeypatch):
    from streamlit.testing.v1 import AppTest

    cfg, folder, metadata, _ = metric_inputs
    cfg.events.coordinate_mode = "image"
    job = batch.prepare_job(cfg, folder, metadata)
    manifest = batch.read_json(job / "manifest.json")
    manifest["code_sha256"] = "old-code"
    batch.write_json(job / "manifest.json", manifest)
    before = tree_hashes(job)
    monkeypatch.setattr("cctv_incident.config.load_config", lambda _: cfg.model_copy(deep=True))
    app = AppTest.from_file(str(Path("apps/streamlit_app.py").resolve())).run(timeout=20)
    next(w for w in app.selectbox if w.label == "Modalita").select(
        "standard_dataset analisi in batch - no omografia"
    ).run(timeout=20)
    assert not app.exception
    assert any("codice originale" in w.value for w in app.warning)
    assert next(w for w in app.button if w.label == "Riprendi").disabled
    assert next(w for w in app.button if w.label == "Avvia batch").disabled
    assert len(app.get("download_button")) == 3
    assert tree_hashes(job) == before


@pytest.fixture
def paired_jobs(metric_inputs):
    cfg, folder, metadata, mapping = metric_inputs
    cfg.perception.model.write_bytes(b"controlled-model")
    cfg.events.coordinate_mode = "image"
    image_job = batch.prepare_job(cfg, folder, metadata)
    cfg.events.coordinate_mode = "metric"
    metric_job = batch.prepare_job(cfg, folder, metadata, calibration_map=mapping)
    for job in (image_job, metric_job):
        batch.run_job(job, Pipeline, lambda _: EmptySegmenter())
    return image_job, metric_job


def test_paired_comparison_of_real_pipelines_declares_detector_difference(paired_jobs):
    report = compare_jobs(*paired_jobs)
    assert report["comparable"], report["reasons"]
    assert report["metrics"]["image"]["video"]["true_negatives"] == 2
    assert report["metrics"]["metric"]["video"]["true_negatives"] == 2
    assert "not an isolated" in report["interpretation"]


@pytest.mark.parametrize(
    "fault", ["fps", "model", "labels", "source", "incomplete", "invalid", "snapshot"]
)
def test_paired_comparison_refuses_uncontrolled_or_incomplete_runs(paired_jobs, fault):
    _, job = paired_jobs
    manifest = batch.read_json(job / "manifest.json")
    if fault == "fps":
        manifest["config"]["video"]["target_fps"] += 1
    elif fault == "model":
        manifest["model_sha256"] = "different"
    elif fault == "labels":
        manifest["videos"][0]["label"]["type"] = "changed"
    elif fault == "source":
        manifest["videos"][0]["sha256"] = "different"
    elif fault == "incomplete":
        next((job / "results").glob("*.json")).unlink()
    elif fault == "invalid":
        path = next((job / "results").glob("*.json"))
        result = batch.read_json(path)
        result["summary"]["evaluable"] = False
        batch.write_json(path, result)
    else:
        (job / manifest["videos"][0]["calibration"]["path"]).write_bytes(b"tampered")
    batch.write_json(job / "manifest.json", manifest)
    report = compare_jobs(*paired_jobs)
    assert not report["comparable"] and report["reasons"]
    assert report["metrics"] is None


def test_label_blind_sample_is_repeatable_and_leaves_archive_unchanged(metric_inputs, monkeypatch):
    from cctv_incident.calibration import evaluation

    cfg, folder, metadata, mapping = metric_inputs
    archive = mapping["a.mp4"].parent
    before = tree_hashes(archive)
    metadata.unlink()  # Proposal evaluation must not require labels.

    def propose(image, **kwargs):
        return ProposalResult((), {}, np.zeros(image.shape[:2], np.uint8), image)

    monkeypatch.setattr(evaluation, "generate_proposals", propose)
    directories = [cfg.project.root_dir / name for name in ("sample1", "sample2")]
    for directory in directories:
        report = sample_proposals(folder, directory, count=1)
        assert report["sample_size"] == 1 and report["raw_proposal_coverage"] == 0
        assert report["mean_review_seconds"] is None and report["metric_error_m"] is None
    assert (
        batch.read_json(directories[0] / "sample.json")["selected_videos"]
        == batch.read_json(directories[1] / "sample.json")["selected_videos"]
    )
    assert tree_hashes(archive) == before
    with pytest.raises(FileExistsError):
        sample_proposals(folder, directories[0], count=1)


def test_review_reports_only_observed_corrections_and_time(metric_inputs, monkeypatch):
    from cctv_incident.calibration import evaluation

    cfg, folder, _, _ = metric_inputs
    _, image = create_draft(folder / "a.mp4", cfg.project.root_dir)
    candidate = ProposalCandidate(
        candidate_id="candidate_1",
        reference_type="painted_bar",
        points_px=((10, 10), (100, 10), (100, 70), (10, 70)),
        support_lines_px=(),
        geometric_quality=0.7,
        reasons=(),
        diagnostics={},
    )
    monkeypatch.setattr(
        evaluation,
        "generate_proposals",
        lambda *a, **k: ProposalResult(
            (candidate,), {}, np.zeros(image.shape[:2], np.uint8), image
        ),
    )
    directory = cfg.project.root_dir / "review-sample"
    sample_proposals(folder, directory, count=1)
    reviews = batch.read_json(directory / "reviews.json")
    reviews[0].update(
        decision="correct",
        candidate_id="candidate_1",
        corrected_points_px=[[11, 10], [100, 10], [100, 70], [10, 70]],
        review_seconds=18.5,
        reviewer="test fixture",
    )
    batch.write_json(directory / "reviews.json", reviews)
    report = review_report(directory)
    assert report["corrected_points"] == report["corrected_videos"] == 1
    assert report["mean_review_seconds"] == 18.5
    assert report["metric_error_m"] is None
    reviews[0]["review_seconds"] = float("nan")
    with pytest.raises(ValueError):
        from cctv_incident.calibration.evaluation import Review

        Review.model_validate(reviews[0])
    (directory / "000/preview.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        review_report(directory)


def test_paired_comparison_refuses_different_analysis_areas(metric_inputs):
    from cctv_incident.calibration.records import confirm_record, edit_record
    from cctv_incident.calibration.repository import load_record, save_record

    cfg, folder, metadata, mapping = metric_inputs
    path = mapping["b.mp4"]
    record = confirm_record(
        edit_record(
            load_record(path),
            roi_px=[(5, 5), (150, 5), (150, 80), (5, 80)],
            geometric_quality=0.8,
        )
    )
    save_record(record, path.parent)
    cfg.perception.model.write_bytes(b"model")
    cfg.events.coordinate_mode = "image"
    image_job = batch.prepare_job(cfg, folder, metadata)
    cfg.events.coordinate_mode = "metric"
    metric_job = batch.prepare_job(cfg, folder, metadata, calibration_map=mapping)
    for job in (image_job, metric_job):
        batch.run_job(job, segmenter_factory=lambda _: EmptySegmenter())
    report = compare_jobs(image_job, metric_job)
    assert not report["comparable"]
    assert any("analysis area" in reason for reason in report["reasons"])


def test_shared_example_configuration_loads_in_both_modes():
    from cctv_incident.config import AppConfig, load_config

    cfg = load_config("configs/paired-evaluation.yaml")
    for mode in ("image", "metric"):
        cfg.events.coordinate_mode = mode
        assert AppConfig.model_validate(cfg.model_dump()).video.target_fps == 8

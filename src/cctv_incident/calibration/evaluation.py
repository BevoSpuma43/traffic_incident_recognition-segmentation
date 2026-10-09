"""Label-blind proposal sampling and explicit, separately recorded human review."""

import random
import time
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field

from ..batch import code_signature, read_json, write_json
from ..video_inputs import list_dataset_videos
from .proposals import ALGORITHM_VERSION, ProposalParameters, generate_proposals
from .repository import create_draft, file_sha256


class Review(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    video: str
    decision: Literal["pending", "accept", "correct", "reject", "no_reference"] = "pending"
    candidate_id: str | None = None
    corrected_points_px: list[tuple[float, float]] | None = Field(None, min_length=4, max_length=4)
    review_seconds: float | None = Field(None, gt=0)
    reviewer: str | None = None
    notes: str = ""


def sample_proposals(folder, output, *, count=5, seed=42, project_root=None):
    """Read first frames only; never read accident labels or mutate calibration archives."""
    folder, output = Path(folder).resolve(), Path(output).resolve()
    names = list_dataset_videos(folder)
    if not 1 <= count <= min(20, len(names)):
        raise ValueError("Sample count must be between 1 and min(20, available videos)")
    selected = sorted(random.Random(seed).sample(sorted(names), count))
    output.mkdir(parents=True, exist_ok=False)
    parameters = ProposalParameters()
    manifest = {
        "schema_version": 1,
        "folder": str(folder),
        "selection": "uniform sample of sorted relative paths, without reading labels",
        "seed": seed,
        "population_size": len(names),
        "algorithm_version": ALGORITHM_VERSION,
        "code_sha256": code_signature(),
        "parameters": parameters.model_dump(mode="json"),
        "selected_videos": selected,
        "videos": [],
    }
    write_json(output / "sample.json", manifest)
    for index, name in enumerate(selected):
        directory = output / f"{index:03d}"
        directory.mkdir()
        row = {"video": name, "error": None, "candidates": [], "generation_seconds": None}
        try:
            record, image = create_draft(folder / name, project_root or folder)
            started = time.perf_counter()
            result = generate_proposals(image, parameters=parameters)
            row.update(
                identity=record.video.model_dump(mode="json"),
                generation_seconds=time.perf_counter() - started,
                candidates=[c.model_dump(mode="json") for c in result.candidates],
                diagnostics=result.diagnostics,
            )
            artifacts = {}
            for filename, pixels in (
                ("reference.png", image),
                ("preview.png", result.preview),
                ("mask.png", result.mask),
            ):
                path = directory / filename
                if not cv2.imwrite(str(path), pixels):
                    raise OSError(f"Cannot write {path}")
                artifacts[path.relative_to(output).as_posix()] = file_sha256(path)
            row["artifacts"] = artifacts
        except (OSError, ValueError, cv2.error) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        manifest["videos"].append(row)
        write_json(output / "sample.json", manifest)
    write_json(output / "reviews.json", [Review(video=name).model_dump() for name in selected])
    report = review_report(output)
    write_json(output / "report.json", report)
    return report


def review_report(directory):
    directory = Path(directory)
    manifest = read_json(directory / "sample.json")
    rows = manifest["videos"]
    if len(rows) != len(manifest["selected_videos"]):
        raise ValueError("Sampling is incomplete")
    by_name = {row["video"]: row for row in rows}
    for row in rows:
        for relative, digest in row.get("artifacts", {}).items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()) or file_sha256(path) != digest:
                raise ValueError(f"Sample artifact changed: {relative}")
    reviews = [Review.model_validate(row) for row in read_json(directory / "reviews.json")]
    if len(reviews) != len(by_name) or {r.video for r in reviews} != set(by_name):
        raise ValueError("Reviews must match the sample exactly, without duplicates")
    corrections = []
    for review in reviews:
        candidates = {c["candidate_id"]: c for c in by_name[review.video]["candidates"]}
        if review.decision == "pending":
            if review.review_seconds is not None or review.corrected_points_px is not None:
                raise ValueError("Pending reviews cannot claim review time or corrections")
            continue
        if not review.reviewer or not review.reviewer.strip():
            raise ValueError("Completed reviews require an identified reviewer")
        if review.decision in {"accept", "correct"} and review.candidate_id not in candidates:
            raise ValueError("Accepted/corrected review must identify an existing candidate")
        if review.decision == "no_reference" and candidates:
            raise ValueError("Use reject for generated candidates that are unusable")
        if review.decision == "correct":
            if review.corrected_points_px is None:
                raise ValueError("Corrected review requires the four final points")
            from .coordinates import validate_quad

            validate_quad(
                review.corrected_points_px, by_name[review.video]["identity"]["image_size"]
            )
            before = candidates[review.candidate_id]["points_px"]
            corrections.append(
                sum(
                    tuple(a) != tuple(b)
                    for a, b in zip(before, review.corrected_points_px, strict=True)
                )
            )
        elif review.corrected_points_px is not None:
            raise ValueError("Only a corrected review can provide corrected points")
    reviewed = [r for r in reviews if r.decision != "pending"]
    times = [r.review_seconds for r in reviewed if r.review_seconds is not None]
    proposed = sum(bool(row["candidates"]) and row["error"] is None for row in rows)
    return {
        "sample_size": len(rows),
        "processing_errors": sum(row["error"] is not None for row in rows),
        "videos_with_proposals": proposed,
        "raw_proposal_coverage": proposed / len(rows),
        "coverage_definition": "at least one proposal / all sampled videos; not validated accuracy",
        "reviewed_videos": len(reviewed),
        "accepted_without_correction": sum(r.decision == "accept" for r in reviewed),
        "corrected_videos": len(corrections),
        "corrected_points": sum(corrections) if corrections else None,
        "review_times_available": len(times),
        "mean_review_seconds": sum(times) / len(times) if times else None,
        "metric_error_m": None,
        "metric_error_status": "not measured; independent distances required",
    }

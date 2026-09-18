"""Inspect ACCIDENT metadata, resolve packaged annotation paths and sample video readability."""

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import av


def resolve_annotation(root, name):
    requested = (root / name).resolve()
    if not requested.is_relative_to(root):
        raise ValueError("Annotation path escapes the dataset directory")
    candidates = [requested]
    if requested.suffix == ".gz":
        plain = requested.with_suffix("")
        # Kaggle v9 packages name.json.gz as name.json/name.json.
        candidates.extend([plain, plain / plain.name])
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Resolved annotation escapes the dataset directory")
        if candidate.is_file():
            return candidate
    return None


def inspect_subset(root, subset, samples):
    metadata = root / f"metadata-{subset}.csv"
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    path_column = "path" if subset == "real" else "rgb_path"
    missing, missing_annotations, invalid_times = [], [], []
    annotation_map, by_class = {}, {}
    for row in rows:
        path = (root / row[path_column]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Metadata path escapes the dataset directory")
        if not path.is_file():
            missing.append(row[path_column])
        if not 0 <= float(row["accident_time"]) <= float(row["duration"]):
            invalid_times.append(row[path_column])
        if row.get("annotations_path"):
            annotation = resolve_annotation(root, row["annotations_path"])
            if annotation is None:
                missing_annotations.append(row["annotations_path"])
            else:
                annotation_map[row["annotations_path"]] = annotation.relative_to(root).as_posix()
        by_class.setdefault(row["type"], []).append(row)
    checks, selected = [], []
    for group in by_class.values():
        step = max(1, len(group) // max(1, samples // len(by_class)))
        selected.extend(group[::step][: max(1, samples // len(by_class))])
    for row in selected[:samples]:
        try:
            with av.open(str(root / row[path_column])) as container:
                first = next(container.decode(container.streams.video[0]))
                check = {
                    "path": row[path_column],
                    "first_frame_decoded": True,
                    "image_size": [first.width, first.height],
                    "metadata_image_size_matches": [first.width, first.height]
                    == [int(row["width"]), int(row["height"])],
                }
            if row.get("annotations_path"):
                annotation = resolve_annotation(root, row["annotations_path"])
                if annotation is None:
                    raise FileNotFoundError(row["annotations_path"])
                opener = gzip.open if annotation.suffix == ".gz" else open
                with opener(annotation, "rt", encoding="utf-8") as handle:
                    json.load(handle)
                check["annotation_json_readable"] = True
            checks.append(check)
        except Exception as exc:
            checks.append({"path": row[path_column], "error": f"{type(exc).__name__}: {exc}"})
    split_columns = [name for name in rows[0] if name.startswith("split_")] if rows else []
    return {
        "metadata": metadata.name,
        "rows": len(rows),
        "videos_on_disk": len(list((root / f"{subset}_videos").rglob("*.mp4"))),
        "duration_hours": sum(float(row["duration"]) for row in rows) / 3600,
        "types": dict(Counter(row["type"] for row in rows)),
        "official_splits": {key: dict(Counter(row[key] for row in rows)) for key in split_columns},
        "columns": list(rows[0]) if rows else [],
        "missing_files": missing,
        "missing_annotations": missing_annotations,
        "invalid_accident_times": invalid_times,
        "annotation_path_map": annotation_map,
        "sample_decode_checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/ACCIDENT")
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    root = Path(args.root).resolve()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "real": inspect_subset(root, "real", args.samples),
        "synthetic": inspect_subset(root, "synthetic", args.samples),
    }
    mapping = {}
    for subset in ("real", "synthetic"):
        paths = report[subset].pop("annotation_path_map")
        mapping.update(paths)
        report[subset]["resolved_annotations"] = len(paths)
    (root / "annotation-path-map.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (root / "inspection-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        subset: {
            "videos": report[subset]["videos_on_disk"],
            "metadata_rows": report[subset]["rows"],
            "missing_files": len(report[subset]["missing_files"]),
            "missing_annotations": len(report[subset]["missing_annotations"]),
            "resolved_annotations": report[subset]["resolved_annotations"],
            "invalid_timestamps": len(report[subset]["invalid_accident_times"]),
            "sampled_videos": len(report[subset]["sample_decode_checks"]),
            "decode_errors": sum("error" in row for row in report[subset]["sample_decode_checks"]),
        }
        for subset in ("real", "synthetic")
    }
    print(json.dumps(summary, indent=2))
    for subset in ("real", "synthetic"):
        if (
            summary[subset]["missing_files"]
            or summary[subset]["missing_annotations"]
            or summary[subset]["invalid_timestamps"]
            or summary[subset]["decode_errors"]
        ):
            raise SystemExit("Dataset inspection reported an issue; see inspection-report.json")


if __name__ == "__main__":
    main()

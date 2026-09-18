"""Deterministic, bounded ACCIDENT sampling; annotations never enter inference config."""

import csv
import hashlib
from collections import Counter
from pathlib import Path

CLASSES = ("head-on", "rear-end", "sideswipe", "single", "t-bone")


def source_id(path):
    stem = Path(path).stem
    head, separator, tail = stem.rpartition("_")
    return head if separator and tail.isdigit() else stem


def select_sample(metadata, per_class=2, seed=42, split="test", max_duration_s=45):
    if not 1 <= per_class <= 4:
        raise ValueError("Use 1 to 4 clips per class (at most 20 videos)")
    with Path(metadata).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_splits = {}
    for row in rows:
        source_splits.setdefault(source_id(row["path"]), set()).add(row["split_in_distribution"])
    eligible = [
        row
        for row in rows
        if row["split_in_distribution"] == split
        and source_splits[source_id(row["path"])] == {split}
        and 10 <= float(row["duration"]) <= max_duration_s
        and 2.5 <= float(row["accident_time"]) <= float(row["duration"]) - 3
    ]
    selected, used_sources = [], set()
    for category in CLASSES:
        candidates = [row for row in eligible if row["type"] == category]
        candidates.sort(
            key=lambda row: hashlib.sha256(f"{seed}:{row['path']}".encode()).hexdigest()
        )
        chosen = []
        for slot in range(per_class):
            available = [row for row in candidates if source_id(row["path"]) not in used_sources]
            if not available:
                raise ValueError(f"Not enough independent sources for {category}")
            # Metadata-only diversity: alternate random draw and illumination/weather diversity.
            if slot and chosen:
                first = chosen[0]
                available.sort(
                    key=lambda row: (
                        -sum(row[key] != first[key] for key in ("day_time", "weather", "quality"))
                    )
                )
            row = available[0]
            chosen.append(row)
            used_sources.add(source_id(row["path"]))
        selected.extend(chosen)
    return selected, {
        "seed": seed,
        "split_column": "split_in_distribution",
        "split": split,
        "per_class": per_class,
        "max_duration_s": max_duration_s,
        "minimum_pre_impact_s": 2.5,
        "minimum_post_impact_s": 3,
        "source_id_kind": "filename_source_proxy_not_verified_camera",
        "exclude_sources_crossing_official_splits": True,
        "eligible_clips": len(eligible),
        "selected_clips": len(selected),
        "duration_s": sum(float(row["duration"]) for row in selected),
        "class_counts": dict(Counter(row["type"] for row in selected)),
        "selection_uses_model_predictions": False,
    }

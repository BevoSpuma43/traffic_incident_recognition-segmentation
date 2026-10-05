"""ACCIDENT labels and two distinct evaluation units: videos and timed events."""

import csv
import math
from pathlib import Path

POSITIVE_TYPES = {"rear-end", "t-bone", "single", "head-on", "sideswipe"}
NEGATIVE_TYPES = {"normal", "negative", "no-accident", "no_accident"}


def load_labels(path):
    """Match copied subsets by unchanged filename; missing labels are never negatives."""
    labels = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "type", "accident_time", "duration"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Colonne mancanti nei metadati: {sorted(required)}")
        for row in reader:
            name = Path(row["path"].replace("\\", "/")).name
            if not name or name in labels:
                raise ValueError(f"Nome video mancante o ambiguo nei metadati: {name}")
            kind = row["type"].strip().lower()
            duration = float(row["duration"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Durata non valida: {name}")
            if kind in POSITIVE_TYPES:
                timestamp = float(row["accident_time"])
                if not math.isfinite(timestamp) or not 0 <= timestamp <= duration:
                    raise ValueError(f"Timestamp incidente non valido: {name}")
            elif kind in NEGATIVE_TYPES and not row["accident_time"].strip():
                timestamp = None
            else:
                raise ValueError(f"Etichetta non riconosciuta o negativa ambigua: {name}")
            labels[name] = {
                "metadata_path": row["path"],
                "type": kind,
                "accident_time_s": timestamp,
                "duration_s": duration,
                "positive": timestamp is not None,
                "split_in_distribution": row.get("split_in_distribution", ""),
                "split_geo_aware": row.get("split_geo_aware", ""),
            }
    return labels


def compare_video(label, predictions, tolerance_s=1.0):
    """One label per ACCIDENT video. Match at most one prediction, nearest first."""
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("Tolleranza temporale non valida")
    target = label["accident_time_s"]
    events = sorted(predictions, key=lambda item: item["impact_time_s"])
    for event in events:
        if not math.isfinite(event["impact_time_s"]):
            raise ValueError("Timestamp predetto non finito")
    eligible = [
        i
        for i, event in enumerate(events)
        if target is not None and abs(event["impact_time_s"] - target) <= tolerance_s
    ]
    matched = (
        min(eligible, key=lambda i: abs(events[i]["impact_time_s"] - target)) if eligible else None
    )
    event_rows = []
    for i, event in enumerate(events):
        event_rows.append(
            {
                "event_id": event["event_id"],
                "impact_time_s": event["impact_time_s"],
                "confirm_time_s": event.get("confirm_time_s"),
                "score": event.get("score"),
                "match": "TP" if i == matched else "FP",
                "time_error_s": event["impact_time_s"] - target if target is not None else None,
                "track_ids": event.get("track_ids", []),
            }
        )
    positive, detected = label["positive"], bool(events)
    outcome = ("TP" if detected else "FN") if positive else ("FP" if detected else "TN")
    return {
        "video_outcome": outcome,
        "predicted_positive": detected,
        "event_tp": int(matched is not None),
        "event_fp": len(events) - int(matched is not None),
        "event_fn": int(positive and matched is None),
        "events": event_rows,
    }


def scores(tp, fp, fn, tn=None):
    def divide(a, b):
        return a / b if b else None

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "accuracy": divide(tp + tn, tp + fp + fn + tn) if tn is not None else None,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
    }


def aggregate_results(results, total, tolerance_s=1.0):
    outcomes = [r["comparison"]["video_outcome"] for r in results]
    video = scores(*(outcomes.count(name) for name in ("TP", "FP", "FN", "TN")))
    event = scores(
        *(sum(r["comparison"][f"event_{name}"] for r in results) for name in ("tp", "fp", "fn"))
    )
    positives = sum(r["label"]["positive"] for r in results)
    return {
        "completed_videos": len(results),
        "total_videos": total,
        "complete": len(results) == total,
        "positive_videos": positives,
        "negative_videos": len(results) - positives,
        "tolerance_s": tolerance_s,
        "video": video,
        "event": event,
    }

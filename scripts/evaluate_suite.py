import argparse
import json
from pathlib import Path

from cctv_incident.config import load_config
from cctv_incident.metrics import evaluate_events
from cctv_incident.pipeline import Pipeline
from cctv_incident.storage import EventStorage

parser = argparse.ArgumentParser(
    description="Evaluate camera-disjoint videos and controlled ablations"
)
parser.add_argument(
    "manifest", help="JSON list: config, clip_id, source_id, camera_id, split, truth"
)
parser.add_argument("--split", choices=["val", "test"], default="test")
parser.add_argument("--ablations", action="store_true")
parser.add_argument("--output", default="outputs/metrics/suite.json")
args = parser.parse_args()
manifest = Path(args.manifest).resolve()
rows = json.loads(manifest.read_text(encoding="utf-8"))
for field in ("camera_id", "source_id", "clip_id"):
    assignments = {}
    for row in rows:
        if assignments.setdefault(row[field], row["split"]) != row["split"]:
            raise ValueError(f"Split leakage: {field}={row[field]}")
selected = [row for row in rows if row["split"] == args.split]
if not selected:
    raise ValueError("No videos selected")
variants = (
    ["baseline", "box_ground_point", "no_smoothing", "long_confirmation"]
    if args.ablations
    else ["baseline"]
)
reports = {}
for variant in variants:
    predictions, truth, runs = [], [], []
    duration = 0.0
    for row in selected:
        cfg = load_config(manifest.parent / row["config"])
        cfg.video.clip_id = row["clip_id"]
        if cfg.calibration.camera_id != row["camera_id"]:
            raise ValueError("Camera mismatch between manifest and configuration")
        cfg.project.output_dir = cfg.project.output_dir / "evaluation" / variant
        if variant == "box_ground_point":
            cfg.features.ground_point_method = "box"
        elif variant == "no_smoothing":
            cfg.features.ema_alpha = 1.0
        elif variant == "long_confirmation":
            cfg.events.confirm_duration_s = 0.8
        result = Pipeline(cfg).run()
        runs.append(result)
        duration += result["media_duration_s"]
        storage = EventStorage(cfg.project.output_dir)
        try:
            predictions.extend(storage.list_events(result["run_id"]))
        finally:
            storage.close()
        annotations = json.loads((manifest.parent / row["truth"]).read_text(encoding="utf-8"))
        if any(
            (event["camera_id"], event["clip_id"]) != (row["camera_id"], row["clip_id"])
            for event in annotations
        ):
            raise ValueError("Ground truth must use manifest camera_id and clip_id")
        truth.extend(annotations)
    reports[variant] = {"events": evaluate_events(predictions, truth, duration), "runs": runs}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(reports, indent=2), encoding="utf-8")
print(json.dumps({key: value["events"] for key, value in reports.items()}, indent=2))

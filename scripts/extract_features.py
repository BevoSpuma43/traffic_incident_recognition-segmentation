import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from cctv_incident.classifier import FEATURE_NAMES, feature_vector

parser = argparse.ArgumentParser(
    description="Aggregate trajectory features into bounded temporal windows"
)
parser.add_argument("--features", required=True)
parser.add_argument(
    "--annotations", required=True, help="JSON events with camera_id, clip_id, impact_time_s"
)
parser.add_argument("--group", required=True, help="Independent camera/source split group")
parser.add_argument("--split", choices=["train", "val", "test"], required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--window", type=float, default=3)
args = parser.parse_args()
if args.window <= 0:
    parser.error("--window must be positive")
annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
history = defaultdict(lambda: deque(maxlen=500))
last_emit = {}
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
with (
    Path(args.features).open(encoding="utf-8") as handle,
    output.open("w", newline="", encoding="utf-8") as sink,
):
    writer = csv.DictWriter(
        sink,
        fieldnames=[
            "camera_id",
            "clip_id",
            "group",
            "split",
            "window_start",
            "window_end",
            "track_a",
            "track_b",
            *FEATURE_NAMES,
            "label",
        ],
    )
    writer.writeheader()
    for line in handle:
        row = json.loads(line)
        if row.get("coordinate_mode", "metric") != "metric":
            raise ValueError("Image-coordinate logs require a separate classifier schema")
        timestamp = row["timestamp_s"]
        for key in list(history):
            while history[key] and history[key][0][0] < timestamp - args.window:
                history[key].popleft()
            if not history[key]:
                del history[key]
                last_emit.pop(key, None)
        for pair in row["pairs"]:
            key = tuple(pair["track_ids"])
            members = [row["motions"][str(track_id)] for track_id in key]
            history[key].append((timestamp, feature_vector(pair, members)))
            if timestamp - last_emit.get(key, -1e9) < args.window / 2:
                continue
            if len(history[key]) < 3:
                continue
            start = history[key][0][0]
            vector = np.mean([value for _, value in history[key]], axis=0)
            label = int(
                any(
                    event["camera_id"] == row["camera_id"]
                    and event["clip_id"] == row["clip_id"]
                    and start <= event["impact_time_s"] <= timestamp
                    for event in annotations
                )
            )
            writer.writerow(
                dict(
                    camera_id=row["camera_id"],
                    clip_id=row["clip_id"],
                    group=args.group,
                    split=args.split,
                    window_start=start,
                    window_end=timestamp,
                    track_a=key[0],
                    track_b=key[1],
                    label=label,
                    **dict(zip(FEATURE_NAMES, vector, strict=True)),
                )
            )
            last_emit[key] = timestamp

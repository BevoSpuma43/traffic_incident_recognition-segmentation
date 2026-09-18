import argparse
import json
from pathlib import Path

import av
import cv2

from cctv_incident.calibration.assisted import propose_calibration
from cctv_incident.calibration.background import sample_background
from cctv_incident.calibration.manual import manual_calibration

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--camera-id", default="camera_01")
parser.add_argument("--output", default="data/calibration/camera_01.yaml")
parser.add_argument(
    "--points",
    help="JSON file: source_points_px (optional), destination_points, units, roi_px (optional)",
)
parser.add_argument("--auto", action="store_true")
parser.add_argument("--width", type=float)
parser.add_argument("--length", type=float)
args = parser.parse_args()
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
if args.auto:
    frame, unstable, size = sample_background(args.source)
    frame = cv2.resize(frame, size)
    calibration, diagnostics, mask, preview = propose_calibration(
        frame, args.camera_id, width=args.width, length=args.length
    )
    cv2.imwrite(str(output.with_suffix(".lanes.png")), mask)
    cv2.imwrite(str(output.with_suffix(".lines.jpg")), preview)
    cv2.imwrite(str(output.with_suffix(".unstable.png")), unstable)
    output.with_suffix(".diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, indent=2))
    if calibration is None:
        raise SystemExit("No reliable proposal. Use --points for manual calibration.")
else:
    if not args.points:
        parser.error("--points is required for manual calibration")
    with av.open(args.source) as container:
        frame = next(container.decode(video=0)).to_ndarray(format="bgr24")
    points = json.loads(Path(args.points).read_text(encoding="utf-8"))
    calibration = manual_calibration(
        frame,
        points["destination_points"],
        args.camera_id,
        points.get("units", "canonical"),
        points.get("source_points_px"),
        points.get("roi_px"),
    )
reference = output.with_suffix(".reference.jpg")
cv2.imwrite(str(reference), frame)
calibration.reference_image = reference.name
calibration.save(output)
print(f"Saved {output}; accepted={calibration.accepted}; units={calibration.units}")

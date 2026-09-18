import argparse
import json
from pathlib import Path

from cctv_incident.metrics import evaluate_events

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
parser.add_argument("--truth", required=True)
parser.add_argument(
    "--duration",
    type=float,
    required=True,
    help="Total evaluated video seconds, including negatives",
)
parser.add_argument("--tolerance", type=float, default=2)
parser.add_argument("--output", default="outputs/metrics/evaluation.json")
args = parser.parse_args()
predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
truth = json.loads(Path(args.truth).read_text(encoding="utf-8"))
report = evaluate_events(predictions, truth, args.duration, args.tolerance)
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

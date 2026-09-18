import argparse
import json

from cctv_incident.config import load_config
from cctv_incident.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--max-frames", type=int)
args = parser.parse_args()
cfg = load_config(args.config)
if args.max_frames:
    cfg.video.max_frames = args.max_frames
print(json.dumps(Pipeline(cfg).run(), indent=2))

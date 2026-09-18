import argparse
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data", required=True, help="YOLO segmentation dataset YAML with camera-disjoint splits"
)
parser.add_argument("--weights", default="models/yolo26n-seg.pt")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--device", default="cpu")
parser.add_argument("--freeze", type=int, default=0)
parser.add_argument("--batch", type=int, default=4)
args = parser.parse_args()
if not Path(args.weights).is_file():
    parser.error("Local pretrained weights are required; run download_model.py")
model = YOLO(args.weights, task="segment")
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=640,
    device=args.device,
    batch=args.batch,
    freeze=args.freeze,
    patience=15,
    seed=42,
    deterministic=True,
    project="models/training",
    name="vehicle-seg",
)
model.val(data=args.data, split="val", device=args.device)

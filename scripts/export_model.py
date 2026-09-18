import argparse
from pathlib import Path

from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--format", choices=["onnx", "openvino", "engine"], required=True)
parser.add_argument("--half", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--data", help="Representative train/validation calibration set; never test")
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
if not Path(args.weights).is_file():
    parser.error("Local weights required")
if args.int8 and not args.data:
    parser.error("INT8 requires --data")
if args.half and args.int8:
    parser.error("Choose FP16 or INT8")
if args.format == "onnx" and args.int8:
    parser.error("Use OpenVINO or TensorRT for this INT8 export workflow")
kwargs = {"format": args.format, "imgsz": 640, "device": args.device, "dynamic": False}
if args.half:
    kwargs["quantize"] = 16
if args.int8:
    kwargs.update(quantize=8, data=args.data)
print(YOLO(args.weights, task="segment").export(**kwargs))

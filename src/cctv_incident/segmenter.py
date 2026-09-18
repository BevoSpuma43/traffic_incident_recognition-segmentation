import importlib.util
import os
from pathlib import Path

import cv2
import numpy as np

from .types import Instance


def configure_offline_runtime():
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"


class Segmenter:
    def __init__(self, config):
        configure_offline_runtime()
        required = {
            "pytorch": "torch",
            "onnx": "onnxruntime",
            "openvino": "openvino",
            "tensorrt": "tensorrt",
        }
        module = required.get(config.backend)
        if module and importlib.util.find_spec(module) is None:
            raise RuntimeError(
                f"Backend dependency missing: {module}. Install it explicitly before inference."
            )
        from ultralytics import YOLO

        if not config.model.exists():
            raise FileNotFoundError(
                f"Local model missing: {config.model}. Run scripts/download_model.py first."
            )
        expected = {"pytorch": ".pt", "onnx": ".onnx", "tensorrt": ".engine"}
        if config.backend in expected and config.model.suffix != expected[config.backend]:
            raise ValueError(f"Model path does not match backend {config.backend}")
        if config.backend == "openvino" and not config.model.is_dir():
            raise ValueError("OpenVINO expects an exported model directory")
        self.config = config
        self.model = YOLO(str(Path(config.model)), task="segment")
        names = self.model.names
        self.class_ids = [i for i, name in names.items() if name in config.classes]
        if not self.class_ids:
            raise ValueError("None of the requested vehicle classes exist in the model")

    def predict(self, frame):
        result = (
            self.model.predict(
                frame,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                classes=self.class_ids,
                device=self.config.device,
                quantize=16 if self.config.half else 32,
                max_det=self.config.max_detections,
                retina_masks=True,
                verbose=False,
            )[0]
            .cpu()
            .numpy()
        )
        if result.masks is None:
            return []
        masks = result.masks.data
        if masks.shape[1:] != frame.shape[:2]:
            raise ValueError("Expected retina masks in original frame coordinates")
        return [
            Instance(box[:4].copy(), mask > 0.5, int(box[5]), float(box[4]))
            for box, mask in zip(result.boxes.data, masks, strict=True)
        ]


class SyntheticSegmenter:
    """Color masks exclusively for the generated regression/demo videos, never CCTV."""

    def predict(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        instances = []
        for low, high in [((40, 110, 110), (85, 255, 255)), ((100, 110, 110), (135, 255, 255))]:
            binary = cv2.inRange(hsv, np.array(low), np.array(high))
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
            for index in range(1, count):
                x, y, w, h, area = stats[index]
                if area >= 60:
                    instances.append(
                        Instance(np.array([x, y, x + w, y + h], float), labels == index, 2, 0.99)
                    )
        return instances


def create_segmenter(config):
    return SyntheticSegmenter() if config.backend == "synthetic" else Segmenter(config)

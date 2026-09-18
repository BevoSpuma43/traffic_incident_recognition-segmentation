import json
import sys
from pathlib import Path

import cv2

from cctv_incident.config import load_config
from cctv_incident.pipeline import hardware_info
from cctv_incident.segmenter import Segmenter

cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml")
report = hardware_info()
if cfg.perception.model.exists():
    segmenter = Segmenter(cfg.perception)
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    if image is None:
        raise FileNotFoundError("Bundled Ultralytics bus image missing")
    instances = segmenter.predict(image)
    report["model_smoke"] = {
        "instances": len(instances),
        "mask_shapes": [list(x.mask.shape) for x in instances],
        "model": str(cfg.perception.model),
        "device": str(segmenter.model.predictor.device),
    }
else:
    report["model_smoke"] = "Not run: local weights are missing"
report["opencv"] = cv2.__version__
Path("outputs").mkdir(exist_ok=True)
Path("outputs/environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))

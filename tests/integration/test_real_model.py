from pathlib import Path

import cv2
import pytest

from cctv_incident.config import Perception
from cctv_incident.segmenter import Segmenter


def test_local_yolo_returns_vehicle_masks():
    path = Path("models/yolo26n-seg.pt")
    if not path.exists():
        pytest.skip("Run scripts/download_model.py to enable the real-model smoke test")
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    segmenter = Segmenter(Perception(model=path.resolve()))
    instances = segmenter.predict(image)
    assert instances, "The bundled bus image should contain a detected vehicle"
    assert any(instance.class_id == 5 for instance in instances)
    assert all(instance.mask.shape == image.shape[:2] for instance in instances)
    assert all(instance.mask.any() for instance in instances)


def test_real_yolo_video_serializes_trajectories(tmp_path, config):
    import json

    from cctv_incident.calibration import estimate_calibration
    from cctv_incident.pipeline import Pipeline

    path = Path("models/yolo26n-seg.pt").resolve()
    if not path.exists():
        pytest.skip("Local weights not present")
    from ultralytics.utils import ASSETS

    image = cv2.imread(str(ASSETS / "bus.jpg"))
    height, width = image.shape[:2]
    video = tmp_path / "bus.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(image)
    writer.release()
    points = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    calibration = estimate_calibration(
        "camera_01", (width, height), points, points, units="canonical"
    )
    config.calibration.file = tmp_path / "calibration.yaml"
    calibration.save(config.calibration.file)
    config.calibration.detect_camera_motion = False
    config.video.source = str(video)
    config.perception.model = path
    config.project.output_dir = tmp_path / "outputs"
    result = Pipeline(config).run()
    assert result["processed_frames"] == 4
    rows = [
        json.loads(line)
        for line in (Path(result["run_dir"]) / "trajectories.jsonl").read_text().splitlines()
    ]
    assert len(rows) >= 4
    assert all(isinstance(row["quality"], float) for row in rows)

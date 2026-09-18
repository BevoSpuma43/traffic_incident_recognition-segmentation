import numpy as np
import pytest
from pydantic import ValidationError

from cctv_incident.classifier import TemporalClassifier
from cctv_incident.config import AppConfig, load_config


class RecordingModel:
    def predict_proba(self, rows):
        self.rows = rows
        return np.array([[0.2, 0.8]])


def test_classifier_uses_temporal_mean_and_expires():
    model = RecordingModel()
    classifier = TemporalClassifier(model, window_seconds=3)
    classifier.score((1, 2), 0, [2, 4])
    assert classifier.score((1, 2), 1, [4, 8]) == 0.8
    np.testing.assert_allclose(model.rows, [[3, 6]])
    classifier.expire(5)
    assert not classifier.histories


def test_config_validation_and_relative_paths(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "test.yaml"
    path.write_text("project:\n  root_dir: ..\nvideo:\n  source: clip.mp4\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.video.source == str(tmp_path / "clip.mp4")
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"perception": {"confidence": 0.9}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"video": {"target_fps": 0}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unknown": 1})

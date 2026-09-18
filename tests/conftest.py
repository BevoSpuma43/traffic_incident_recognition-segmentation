import numpy as np
import pytest

from cctv_incident.calibration import estimate_calibration
from cctv_incident.config import AppConfig
from cctv_incident.types import Observation


@pytest.fixture
def config():
    cfg = AppConfig()
    cfg.features.ema_alpha = 1.0
    cfg.events.pre_event_s = 1
    cfg.events.post_event_s = 1
    return cfg


@pytest.fixture
def calibration():
    return estimate_calibration(
        "camera_01",
        (640, 360),
        [[0, 0], [639, 0], [639, 359], [0, 359]],
        [[0, 0], [31.95, 0], [31.95, 17.95], [0, 17.95]],
    )


def observation(track_id, timestamp, x, y=0, quality=1):
    return Observation(
        track_id,
        timestamp,
        np.array([x, y], float),
        (x * 20, y * 20),
        np.array([x * 20 - 10, y * 20 - 10, x * 20 + 10, y * 20 + 10]),
        quality=quality,
    )

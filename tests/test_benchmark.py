import pytest

from src.benchmark import parse_config_overrides, select_annotations
from src.calibration import Annotation
from src.config import AppConfig


def test_select_annotations_filters_split_type_and_limit() -> None:
    annotations = [
        Annotation(
            "real_videos/a.mp4",
            10,
            10,
            label="rear-end",
            split_in_distribution="test",
        ),
        Annotation(
            "real_videos/b.mp4",
            20,
            20,
            label="t-bone",
            split_in_distribution="test",
        ),
        Annotation(
            "real_videos/c.mp4",
            30,
            30,
            label="rear-end",
            split_in_distribution="train",
        ),
    ]

    selected = select_annotations(
        annotations,
        split_field="split_in_distribution",
        split="test",
        accident_types={"REAR-END"},
        video_names={"a.mp4"},
        limit=1,
    )

    assert [item.video_path for item in selected] == ["real_videos/a.mp4"]


def test_config_overrides_use_the_declared_field_types() -> None:
    overrides = parse_config_overrides(
        [
            "track_reid_enabled=true",
            "young_pair_max_frames=5",
            "impulse_approach_speed_multiplier=2.5",
            "model_path=yolo26s-seg.pt",
        ]
    )

    assert overrides == {
        "track_reid_enabled": True,
        "young_pair_max_frames": 5,
        "impulse_approach_speed_multiplier": 2.5,
        "model_path": "yolo26s-seg.pt",
    }
    # Devono restare applicabili a una configurazione reale.
    assert AppConfig(**overrides).track_reid_enabled is True


@pytest.mark.parametrize(
    "override",
    ["campo_inesistente=1", "track_reid_enabled", "track_reid_enabled=forse"],
)
def test_invalid_config_override_is_rejected(override: str) -> None:
    with pytest.raises(ValueError):
        parse_config_overrides([override])

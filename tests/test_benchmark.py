from src.benchmark import select_annotations
from src.calibration import Annotation


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

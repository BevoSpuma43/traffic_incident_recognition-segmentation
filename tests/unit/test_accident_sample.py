import csv

import pytest

from cctv_incident.accident_sample import CLASSES, select_sample, source_id


def test_sample_is_bounded_deterministic_and_avoids_source_overlap(tmp_path):
    rows = []
    for index, category in enumerate(CLASSES):
        for sample in range(6):
            rows.append(
                dict(
                    path=f"real_videos/source{index}_{sample}_00.mp4",
                    type=category,
                    split_in_distribution="test",
                    duration="20",
                    accident_time="5",
                    day_time="day" if sample % 2 else "night",
                    weather="normal",
                    quality="Good",
                )
            )
    rows.append(
        {
            **rows[0],
            "path": rows[0]["path"].replace("_00.mp4", "_01.mp4"),
            "split_in_distribution": "train",
        }
    )
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selected, policy = select_sample(metadata)
    assert (selected, policy) == select_sample(metadata)
    assert len(selected) == 10
    assert len({source_id(row["path"]) for row in selected}) == 10
    assert source_id(rows[0]["path"]) not in {source_id(row["path"]) for row in selected}
    assert policy["class_counts"] == dict.fromkeys(CLASSES, 2)
    assert not policy["selection_uses_model_predictions"]
    with pytest.raises(ValueError, match="at most 20"):
        select_sample(metadata, per_class=5)

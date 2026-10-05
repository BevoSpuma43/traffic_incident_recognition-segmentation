import csv

import pytest

from cctv_incident.batch_evaluation import aggregate_results, compare_video, load_labels


def label(positive=True):
    return {"positive": positive, "accident_time_s": 10.0 if positive else None}


def prediction(timestamp, event_id="event"):
    return {"event_id": event_id, "impact_time_s": timestamp}


@pytest.mark.parametrize("timestamp,expected", [(9, 1), (11, 1), (8.999, 0), (11.001, 0)])
def test_one_second_inclusive_tolerance(timestamp, expected):
    result = compare_video(label(), [prediction(timestamp)])
    assert result["event_tp"] == expected
    assert result["event_fp"] == 1 - expected
    assert result["event_fn"] == 1 - expected
    assert result["video_outcome"] == "TP"  # Clip presence is a different question.


def test_one_to_one_matching_counts_duplicate_and_off_time_alarms():
    result = compare_video(
        label(), [prediction(10.8, "a"), prediction(10.2, "b"), prediction(20, "c")]
    )
    assert (result["event_tp"], result["event_fp"], result["event_fn"]) == (1, 2, 0)
    assert [e["event_id"] for e in result["events"] if e["match"] == "TP"] == ["b"]


def test_video_confusion_matrix_includes_explicit_negatives():
    inputs = [(True, [prediction(10)]), (True, []), (False, [prediction(10)]), (False, [])]
    results = [
        {"label": label(positive), "comparison": compare_video(label(positive), predictions)}
        for positive, predictions in inputs
    ]
    metrics = aggregate_results(results, 4)
    assert metrics["complete"]
    assert metrics["video"] == {
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "true_negatives": 1,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert metrics["event"]["true_negatives"] is None
    assert metrics["event"]["accuracy"] is None
    assert aggregate_results([], 4)["video"]["accuracy"] is None


def test_labels_match_basename_and_reject_ambiguous_or_missing_truth(tmp_path):
    path = tmp_path / "metadata.csv"

    def write(rows):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "type", "accident_time", "duration"])
            writer.writerows(rows)

    write([["real_videos/a.mp4", "t-bone", 9, 30], ["real_videos/b.mp4", "normal", "", 20]])
    labels = load_labels(path)
    assert labels["a.mp4"]["accident_time_s"] == 9
    assert not labels["b.mp4"]["positive"]
    write([["a.mp4", "t-bone", "", 30]])
    with pytest.raises(ValueError):
        load_labels(path)
    write([["a.mp4", "t-bone", 9, 30], ["other/a.mp4", "t-bone", 9, 30]])
    with pytest.raises(ValueError, match="ambiguo"):
        load_labels(path)

import pytest

from cctv_incident.metrics import evaluate_events


def event(timestamp, camera="c", clip="a"):
    return {
        "camera_id": camera,
        "clip_id": clip,
        "impact_time_s": timestamp,
        "confirm_time_s": timestamp + 1,
    }


def test_one_to_one_and_false_alarms():
    report = evaluate_events([event(10), event(10.2), event(30)], [event(10), event(50)], 3600)
    assert report["true_positives"] == 1
    assert report["false_positives"] == 2
    assert report["false_negatives"] == 1
    assert report["false_alarms_per_hour"] == 2
    assert report["delay_mean_s"] == 1


def test_cross_clip_matching_forbidden():
    report = evaluate_events([event(10, clip="a")], [event(10, clip="b")], 60)
    assert report["true_positives"] == 0
    with pytest.raises(ValueError, match="clip_id"):
        evaluate_events([{"camera_id": "c", "impact_time_s": 10}], [], 60)


def test_empty_evaluation():
    assert evaluate_events([], [], 60)["delay_mean_s"] is None
    with pytest.raises(ValueError):
        evaluate_events([], [], 0)

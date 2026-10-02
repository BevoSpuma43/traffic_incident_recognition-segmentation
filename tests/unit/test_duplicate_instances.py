import numpy as np
import pytest

from cctv_incident.segmenter import suppress_duplicate_instances
from cctv_incident.types import Instance


@pytest.mark.parametrize("other_class", [2, 7])
def test_same_silhouette_keeps_only_highest_confidence_detection(other_class):
    mask = np.zeros((40, 40), bool)
    mask[5:30, 5:30] = True
    low = Instance(np.array([5, 5, 30, 30]), mask.copy(), other_class, 0.4)
    high = Instance(np.array([5, 5, 30, 30]), mask.copy(), 2, 0.8)
    result = suppress_duplicate_instances([low, high])
    assert len(result) == 1 and result[0] is high


def test_overlapping_boxes_of_distinct_silhouettes_are_preserved():
    a = np.zeros((40, 40), bool)
    b = a.copy()
    a[5:30, 5:17] = True
    b[5:30, 18:30] = True
    instances = [
        Instance(np.array([5, 5, 30, 30]), a, 2, 0.8),
        Instance(np.array([5, 5, 30, 30]), b, 2, 0.7),
    ]
    result = suppress_duplicate_instances(instances)
    assert len(result) == 2
    assert result[0] is instances[0] and result[1] is instances[1]


def test_combined_mask_does_not_suppress_individual_collision_vehicles():
    a = np.zeros((40, 40), bool)
    b = a.copy()
    a[5:25, 2:17] = True
    b[5:25, 19:34] = True
    combined = Instance(np.array([2, 5, 34, 25]), a | b, 7, 0.6)
    vehicles = [
        Instance(np.array([2, 5, 17, 25]), a, 2, 0.5),
        Instance(np.array([19, 5, 34, 25]), b, 2, 0.4),
    ]
    assert len(suppress_duplicate_instances([combined, *vehicles])) == 3

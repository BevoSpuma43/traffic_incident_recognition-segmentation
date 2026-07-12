import numpy as np
import pytest

from src.geometry import (
    compute_bbox_from_polygon,
    compute_centroid_from_polygon,
    mask_intersection_area,
    masks_touch,
    polygon_to_binary_mask,
    safe_polygon_array,
)


def _point_xy(point: object) -> tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(getattr(point, "x")), float(getattr(point, "y"))

    if isinstance(point, (tuple, list, np.ndarray)) and len(point) == 2:
        return float(point[0]), float(point[1])

    raise AssertionError("Unsupported centroid format")


def _bbox_xyxy(
    bbox: object,
    expected_min_x: int,
    expected_min_y: int,
    expected_max_x: int,
    expected_max_y: int,
) -> tuple[int, int, int, int]:
    if all(hasattr(bbox, attr) for attr in ("x1", "y1", "x2", "y2")):
        return (
            int(getattr(bbox, "x1")),
            int(getattr(bbox, "y1")),
            int(getattr(bbox, "x2")),
            int(getattr(bbox, "y2")),
        )

    if all(hasattr(bbox, attr) for attr in ("x", "y", "w", "h")):
        x = int(getattr(bbox, "x"))
        y = int(getattr(bbox, "y"))
        w = int(getattr(bbox, "w"))
        h = int(getattr(bbox, "h"))
        return x, y, x + w, y + h

    if isinstance(bbox, (tuple, list, np.ndarray)) and len(bbox) == 4:
        x0, y0, a, b = [int(v) for v in bbox]
        width = expected_max_x - expected_min_x
        height = expected_max_y - expected_min_y

        if (x0, y0, a, b) == (expected_min_x, expected_min_y, width, height):
            return x0, y0, x0 + a, y0 + b

        return x0, y0, a, b

    raise AssertionError("Unsupported bbox format")


def test_safe_polygon_array_converts_points_to_numpy_array() -> None:
    polygon = [(1, 1), (4, 1), (4, 3), (1, 3)]

    result = safe_polygon_array(polygon)

    assert isinstance(result, np.ndarray)
    assert result.shape == (4, 2)
    assert np.issubdtype(result.dtype, np.number)
    np.testing.assert_array_equal(
        result,
        np.array([[1, 1], [4, 1], [4, 3], [1, 3]], dtype=result.dtype),
    )


def test_polygon_to_binary_mask_creates_foreground_inside_polygon() -> None:
    polygon = np.array([[2, 2], [5, 2], [5, 5], [2, 5]], dtype=np.int32)

    mask = polygon_to_binary_mask(polygon, (8, 8))

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (8, 8)
    assert mask[3, 3] > 0
    assert mask[0, 0] == 0
    assert np.count_nonzero(mask) > 0


def test_compute_centroid_from_polygon_returns_center_of_rectangle() -> None:
    polygon = np.array([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=np.int32)

    centroid = compute_centroid_from_polygon(polygon)
    x, y = _point_xy(centroid)

    assert x == pytest.approx(2.0)
    assert y == pytest.approx(2.0)


def test_compute_bbox_from_polygon_returns_expected_bounds() -> None:
    polygon = np.array([[1, 2], [4, 2], [4, 6], [1, 6]], dtype=np.int32)

    bbox = compute_bbox_from_polygon(polygon)
    x1, y1, x2, y2 = _bbox_xyxy(
        bbox,
        expected_min_x=1,
        expected_min_y=2,
        expected_max_x=4,
        expected_max_y=6,
    )

    assert (x1, y1, x2, y2) == (1, 2, 4, 6)


def test_mask_intersection_area_counts_shared_pixels() -> None:
    mask_a = np.zeros((6, 6), dtype=np.uint8)
    mask_b = np.zeros((6, 6), dtype=np.uint8)

    mask_a[1:4, 1:4] = 1
    mask_b[2:5, 2:5] = 1

    area = mask_intersection_area(mask_a, mask_b)

    assert area == 4


def test_masks_touch_is_true_when_masks_overlap() -> None:
    mask_a = np.zeros((6, 6), dtype=np.uint8)
    mask_b = np.zeros((6, 6), dtype=np.uint8)

    mask_a[1:4, 1:4] = 1
    mask_b[3:5, 3:5] = 1

    assert masks_touch(mask_a, mask_b) is True


def test_masks_touch_is_false_when_masks_are_separate() -> None:
    mask_a = np.zeros((6, 6), dtype=np.uint8)
    mask_b = np.zeros((6, 6), dtype=np.uint8)

    mask_a[1:3, 1:3] = 1
    mask_b[4:6, 4:6] = 1

    assert masks_touch(mask_a, mask_b) is False
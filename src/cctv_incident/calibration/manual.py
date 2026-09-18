import cv2
import numpy as np

from .homography import estimate_calibration


def select_points(image, count=4):
    points = []
    name = "Road points: click in correspondence order; Enter accept; Backspace undo; Esc cancel"

    def clicked(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append([float(x), float(y)])

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, clicked)
    try:
        while True:
            preview = image.copy()
            for i, point in enumerate(points):
                xy = tuple(map(int, point))
                cv2.circle(preview, xy, 5, (0, 255, 255), -1)
                cv2.putText(preview, str(i + 1), xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(name, preview)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                raise ValueError("Calibration cancelled")
            if key in (8, 127) and points:
                points.pop()
            if key in (10, 13) and len(points) == count:
                return np.asarray(points)
    finally:
        cv2.destroyWindow(name)


def manual_calibration(image, destination, camera_id, units="m", source=None, roi=None):
    source = select_points(image, len(destination)) if source is None else source
    return estimate_calibration(
        camera_id, (image.shape[1], image.shape[0]), source, destination, units=units, roi=roi
    )

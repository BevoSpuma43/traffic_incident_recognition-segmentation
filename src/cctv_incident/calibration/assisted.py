import cv2
import numpy as np

from .homography import estimate_calibration, order_quad
from .lane_mask import lane_mask
from .line_fitting import extreme_lines, fit_lines, vanishing_point


def propose_calibration(image, camera_id, roi=None, width=None, length=None):
    mask, skeleton = lane_mask(image, roi)
    families = fit_lines(skeleton)
    preview = image.copy()
    diagnostics = {"accepted": False, "requires_review": True, "reason": "", "vanishing_points": []}
    for group, family in enumerate(families):
        point, dispersion = vanishing_point(family)
        diagnostics["vanishing_points"].append(
            {"point": point, "dispersion": dispersion if np.isfinite(dispersion) else None}
        )
        for segment, _, _ in family:
            x1, y1, x2, y2 = map(int, segment)
            cv2.line(preview, (x1, y1), (x2, y2), (0, 255, 255) if group else (255, 150, 0), 2)
    if any(len(family) < 2 for family in families):
        diagnostics["reason"] = (
            "Two independent line families are required; use manual calibration."
        )
        return None, diagnostics, mask, preview
    center = np.array([image.shape[1] / 2, image.shape[0] / 2])
    horizontal, vertical = [extreme_lines(family, center) for family in families]
    points = []
    for a in horizontal:
        for b in vertical:
            intersection = np.cross(a, b)
            if abs(intersection[2]) < 1e-8:
                diagnostics["reason"] = "Unstable line intersections; use manual calibration."
                return None, diagnostics, mask, preview
            points.append(intersection[:2] / intersection[2])
    try:
        points = order_quad(points)
        if (
            (points < 0).any()
            or (points[:, 0] >= image.shape[1]).any()
            or (points[:, 1] >= image.shape[0]).any()
        ):
            raise ValueError("Proposed corners fall outside the image")
        if cv2.contourArea(points) < image.shape[0] * image.shape[1] * 0.01:
            raise ValueError("Insufficient road coverage")
        metric = width is not None and length is not None
        if metric and (width <= 0 or length <= 0):
            raise ValueError("Measured width and length must be positive")
        w, h = (width, length) if metric else (10, 30)
        calibration = estimate_calibration(
            camera_id,
            (image.shape[1], image.shape[0]),
            points,
            [[0, 0], [w, 0], [w, h], [0, h]],
            units="m" if metric else "canonical",
            method="auto_assisted",
            accepted=False,
            roi=roi,
        )
        # These are proposed painted-line intersections; their physical meaning needs review.
        total_length = sum(item[2] for family in families for item in family)
        calibration.confidence = min(0.79, 0.4 + 0.3 * min(1, total_length / image.shape[1] / 4))
        diagnostics["reason"] = (
            "Review corner correspondences and measured distances before accepting."
        )
        diagnostics["confidence"] = calibration.confidence
        return calibration, diagnostics, mask, preview
    except ValueError as exc:
        diagnostics["reason"] = str(exc)
        return None, diagnostics, mask, preview

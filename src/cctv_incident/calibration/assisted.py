"""Legacy tuple API backed by reviewable contour proposals."""

import numpy as np

from .homography import estimate_calibration
from .proposals import generate_proposals


def propose_calibration(image, camera_id, roi=None, width=None, length=None):
    result = generate_proposals(image, roi)
    diagnostics = {
        **result.diagnostics,
        "accepted": False,
        "candidates": [c.model_dump(mode="json") for c in result.candidates],
    }
    if not result.candidates:
        return None, diagnostics, result.mask, result.preview
    candidate = result.candidates[0]
    metric = width is not None and length is not None
    if metric and (not np.isfinite([width, length]).all() or min(width, length) <= 0):
        diagnostics["reason"] = "Measured width and length must be finite and positive"
        return None, diagnostics, result.mask, result.preview
    # Legacy compatibility only: an explicitly nonmetric unit square.
    # The new editor never receives a canonical fallback as a metric scale.
    w, h = (width, length) if metric else (1, 1)
    try:
        calibration = estimate_calibration(
            camera_id,
            (image.shape[1], image.shape[0]),
            candidate.points_px,
            [[0, 0], [w, 0], [w, h], [0, h]],
            units="m" if metric else "canonical",
            roi=roi,
            method="auto_assisted",
            accepted=False,
        )
        calibration.confidence = candidate.geometric_quality
        diagnostics["confidence"] = candidate.geometric_quality
        return calibration, diagnostics, result.mask, result.preview
    except ValueError as exc:
        diagnostics["reason"] = str(exc)
        return None, diagnostics, result.mask, result.preview

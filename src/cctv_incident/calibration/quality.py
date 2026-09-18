import cv2
import numpy as np

from .homography import transform_points


def held_out_error(calibration, source, destination):
    errors = np.linalg.norm(
        transform_points(calibration.homography, source) - np.asarray(destination), axis=1
    )
    return {
        "mean": float(errors.mean()),
        "p95": float(np.quantile(errors, 0.95)),
        "units": calibration.units,
    }


class CameraMotionGuard:
    def __init__(self, reference, threshold_px=8):
        self.reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        self.orb = cv2.ORB_create(nfeatures=1200)
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.reference, None)
        self.threshold = threshold_px
        self.strikes = 0

    def update(self, image, exclusion_mask=None):
        if image.shape[:2] != self.reference.shape:
            return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (
            None
            if exclusion_mask is None
            else (~exclusion_mask.astype(bool)).astype(np.uint8) * 255
        )
        keys, descriptors = self.orb.detectAndCompute(gray, mask)
        if self.descriptors is None or descriptors is None or len(descriptors) < 8:
            return False
        matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.descriptors, descriptors, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
        if len(good) < 12:
            return False
        source = np.float32([self.keypoints[m.queryIdx].pt for m in good])
        target = np.float32([keys[m.trainIdx].pt for m in good])
        affine, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=2
        )
        if affine is None or inliers.sum() < 10 or inliers.mean() < 0.5:
            return False
        h, w = gray.shape
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        displacement = np.median(
            np.linalg.norm(np.c_[corners, np.ones(4)] @ affine.T - corners, axis=1)
        )
        self.strikes = self.strikes + 1 if displacement > self.threshold else 0
        return self.strikes >= 2

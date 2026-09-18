import cv2
import numpy as np
from skimage.morphology import skeletonize


def lane_mask(image, roi=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
    white = (hsv[:, :, 1] < 65) & (light > 185)
    yellow = (hsv[:, :, 0] > 15) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 70) & (light > 110)
    top_hat = cv2.morphologyEx(light, cv2.MORPH_TOPHAT, np.ones((15, 15), np.uint8))
    mask = ((white | yellow) & (top_hat > 15)).astype(np.uint8) * 255
    road = np.zeros(mask.shape, np.uint8)
    if roi is None:
        road[mask.shape[0] // 3 :] = 255
    else:
        cv2.fillPoly(road, [np.asarray(roi, np.int32)], 255)
    mask &= road
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 15:
            clean[labels == i] = 255
    return clean, skeletonize(clean > 0).astype(np.uint8) * 255

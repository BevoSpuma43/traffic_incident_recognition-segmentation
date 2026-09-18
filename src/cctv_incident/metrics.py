import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np
from scipy.optimize import linear_sum_assignment


class Profiler:
    def __init__(self, max_samples=10000):
        self.samples = defaultdict(lambda: deque(maxlen=max_samples))
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def measure(self, stage):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.samples[stage].append(duration)
            self.totals[stage] += duration
            self.counts[stage] += 1

    def summary(self):
        return {
            stage: {
                "count": self.counts[stage],
                "mean_ms": self.totals[stage] / self.counts[stage] * 1000,
                "median_ms": float(np.median(values)) * 1000,
                "p95_ms": float(np.quantile(values, 0.95)) * 1000,
                "quantile_sample_count": len(values),
            }
            for stage, values in self.samples.items()
        }


def evaluate_events(predictions, truth, duration_s, tolerance_s=2.0):
    if duration_s <= 0 or tolerance_s < 0:
        raise ValueError("Duration must be positive and tolerance non-negative")
    for row in [*predictions, *truth]:
        if not np.isfinite(row["impact_time_s"]):
            raise ValueError("Event timestamps must be finite")
        if "camera_id" not in row or "clip_id" not in row:
            raise ValueError(
                "Each event requires camera_id and clip_id to prevent cross-video matches"
            )
    n, m = len(predictions), len(truth)
    delays, matched = [], 0
    if n and m:
        # Dummy nodes allow unmatched events; the penalty prioritizes cardinality over timing.
        penalty = (max(n, m) + 1) * (tolerance_s + 1)
        cost = np.full((n, m + n), penalty)
        valid = np.zeros((n, m), bool)
        for i, pred in enumerate(predictions):
            for j, target in enumerate(truth):
                delta = abs(pred["impact_time_s"] - target["impact_time_s"])
                if (pred["camera_id"], pred["clip_id"]) == (
                    target["camera_id"],
                    target["clip_id"],
                ) and delta <= tolerance_s:
                    cost[i, j], valid[i, j] = delta, True
                else:
                    cost[i, j] = penalty * 3
        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols, strict=True):
            if j < m and valid[i, j]:
                matched += 1
                delays.append(
                    predictions[i].get("confirm_time_s", predictions[i]["impact_time_s"])
                    - truth[j]["impact_time_s"]
                )
    false_positive, false_negative = n - matched, m - matched
    precision = matched / n if n else 0.0
    recall = matched / m if m else 0.0
    return {
        "true_positives": matched,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarms_per_hour": false_positive * 3600 / duration_s,
        "delay_mean_s": float(np.mean(delays)) if delays else None,
        "delay_p95_s": float(np.quantile(delays, 0.95)) if delays else None,
        "duration_s": duration_s,
        "tolerance_s": tolerance_s,
    }

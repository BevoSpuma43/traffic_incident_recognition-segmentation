FEATURE_NAMES = [
    "distance",
    "relative_speed",
    "closest_time",
    "closest_distance",
    "ttc",
    "bbox_iou",
    "quality",
    "max_speed",
    "max_deceleration",
    "max_jerk",
    "max_heading_change",
    "max_stop_duration",
    "max_prior_speed",
]


def feature_vector(pair, members):
    def get(obj, name):
        return obj[name] if isinstance(obj, dict) else getattr(obj, name)

    ttc = get(pair, "ttc")
    return [
        get(pair, "distance"),
        get(pair, "relative_speed"),
        min(30, get(pair, "closest_time")),
        get(pair, "closest_distance"),
        30 if ttc is None else min(30, ttc),
        get(pair, "bbox_iou"),
        get(pair, "quality"),
        *[
            max(get(member, field) for member in members)
            for field in [
                "speed",
                "deceleration",
                "jerk",
                "heading_change",
                "stop_duration_s",
                "prior_speed",
            ]
        ],
    ]


class TemporalClassifier:
    """Same causal rolling mean used by extract_features.py."""

    def __init__(self, model, window_seconds=3, max_pairs=5000):
        self.model, self.window = model, window_seconds
        self.histories = {}
        self.max_pairs = max_pairs

    def expire(self, timestamp):
        for key in list(self.histories):
            history = self.histories[key]
            while history and history[0][0] < timestamp - self.window:
                history.popleft()
            if not history:
                del self.histories[key]

    def score(self, key, timestamp, vector):
        from collections import deque

        import numpy as np

        if key not in self.histories:
            if len(self.histories) >= self.max_pairs:
                oldest = min(self.histories, key=lambda k: self.histories[k][-1][0])
                del self.histories[oldest]
            self.histories[key] = deque(maxlen=500)
        self.histories[key].append((timestamp, vector))
        mean = np.mean([row for _, row in self.histories[key]], axis=0)
        return float(self.model.predict_proba([mean])[0, 1])

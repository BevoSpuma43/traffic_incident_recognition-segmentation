from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .types import Motion


@dataclass
class History:
    motions: deque = field(default_factory=deque)
    started: float = 0
    stop_since: float | None = None


class TrajectoryStore:
    def __init__(self, tracking, features, coordinate_mode="metric"):
        self.coordinate_mode = coordinate_mode
        self.config, self.features = tracking, features
        self.histories: dict[int, History] = {}
        self.last_timestamp = -float("inf")

    def update(self, observations, timestamp):
        if timestamp <= self.last_timestamp:
            return {}
        self.last_timestamp = timestamp
        cutoff = timestamp - self.config.history_seconds
        for key in list(self.histories):
            history = self.histories[key]
            while history.motions and history.motions[0].observation.timestamp_s < cutoff:
                history.motions.popleft()
            if not history.motions:
                del self.histories[key]
        current = {}
        for obs in observations:
            if obs.timestamp_s != timestamp or not np.isfinite(obs.point).all():
                continue
            if obs.track_id not in self.histories:
                if len(self.histories) >= self.config.max_tracks:
                    oldest = min(
                        self.histories,
                        key=lambda k: self.histories[k].motions[-1].observation.timestamp_s,
                    )
                    del self.histories[oldest]
                self.histories[obs.track_id] = History(started=timestamp)
            history = self.histories[obs.track_id]
            prev = history.motions[-1] if history.motions else None
            dt = timestamp - prev.observation.timestamp_s if prev else 0
            continuous = prev is not None and 0 < dt <= self.features.max_gap_s
            if self.coordinate_mode == "image":
                scale = max(1.0, float(np.linalg.norm(obs.bbox[2:] - obs.bbox[:2])))
                max_speed = scale * self.features.image.max_speed_diagonals_s
                stop_speed = scale * self.features.image.stop_speed_diagonals_s
            else:
                max_speed = self.features.max_speed_mps
                stop_speed = self.features.stop_speed_mps
            if continuous and np.linalg.norm(obs.point - prev.observation.point) / dt > max_speed:
                # Start a new reliable history after an association jump.
                history.motions.clear()
                continuous = False
            if not continuous:
                history.started, history.stop_since = timestamp, None
                position, velocity, acceleration = obs.point.copy(), np.zeros(2), np.zeros(2)
                jerk = heading_change = deceleration = 0.0
                quality = 0.0
            else:
                alpha = 1 - (1 - self.features.ema_alpha) ** (
                    dt / self.features.smoothing_reference_s
                )
                position = prev.position + alpha * (obs.point - prev.position)
                velocity = (position - prev.position) / dt
                acceleration = (velocity - prev.velocity) / dt
                jerk = float(np.linalg.norm(acceleration - prev.acceleration) / dt)
                heading_change = 0.0
                if np.linalg.norm(velocity) > stop_speed and prev.speed > stop_speed:
                    a, b = (
                        np.arctan2(velocity[1], velocity[0]),
                        np.arctan2(prev.velocity[1], prev.velocity[0]),
                    )
                    heading_change = float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))
                deceleration = float(max(0, (prev.speed - np.linalg.norm(velocity)) / dt))
                quality = min(obs.quality, prev.observation.quality) * (0.3 if obs.predicted else 1)
            quality *= obs.confidence
            speed = float(np.linalg.norm(velocity))
            if continuous and speed <= stop_speed:
                if history.stop_since is None:
                    history.stop_since = timestamp
            else:
                history.stop_since = None
            recent = [
                m.speed
                for m in history.motions
                if timestamp - m.observation.timestamp_s <= self.features.window_seconds
            ]
            motion = Motion(
                obs,
                position,
                velocity,
                acceleration,
                jerk,
                speed,
                deceleration,
                heading_change,
                timestamp - history.started,
                timestamp - history.stop_since if history.stop_since is not None else 0,
                max(recent, default=speed),
                quality,
            )
            history.motions.append(motion)
            while len(history.motions) > self.config.max_samples:
                history.motions.popleft()
            current[obs.track_id] = motion
        return current

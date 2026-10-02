"""Side-contact confirmation from observed shape, direction and speed changes."""

from collections import deque
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np

from .features import bbox_contact_coverage, bbox_diagonal
from .types import Event


@dataclass
class SideContact:
    timestamp: float
    boxes: dict
    velocities: dict
    prior_boxes: dict = field(default_factory=dict)
    rotated: set = field(default_factory=set)
    crossing: bool = False
    hits: int = 1
    responders: set = field(default_factory=set)
    post_since: dict = field(default_factory=dict)


class SideswipeDetector:
    """Confirm lateral and crossing contacts from observed aftermath.

    Side contacts need repeated overlap and independent vehicle reactions.
    Crossing contacts need mature tracks with a previously observed separation,
    followed by strong overlap, rotation and sustained slowdown near the contact.
    """

    def __init__(self, config, features, camera_id):
        self.config, self.features, self.camera_id = config, features, camera_id
        self.histories = {}
        self.contacts = {}

    def reset(self):
        self.histories.clear()
        self.contacts.clear()

    def update(self, motions, pairs, t, blocked):
        cfg = self.config.image
        side = cfg.sideswipe
        if not side.enabled:
            self.reset()
            return []
        history_seconds = side.confirmation_window_s + side.velocity_window_s
        reliable = {}
        velocities = {}
        for k, m in motions.items():
            if (
                m.observation.predicted
                or m.observation.timestamp_s != t
                or m.observation.quality < cfg.min_quality
            ):
                continue
            box = m.observation.bbox
            if not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                continue
            samples = self.histories.setdefault(k, deque(maxlen=256))
            while samples and t - samples[0][0] > history_seconds:
                samples.popleft()
            if samples and t - samples[-1][0] > self.features.max_gap_s:
                samples.clear()
            if samples:
                old = next(
                    (r for r in reversed(samples) if t - r[0] >= side.velocity_window_s), samples[0]
                )
                velocities[k] = ((box[:2] + box[2:]) - (old[1][:2] + old[1][2:])) / 2 / (t - old[0])
            samples.append((t, box.copy()))
            if m.quality >= cfg.min_quality and k in velocities:
                reliable[k] = m
        for k in list(self.histories):
            if t - self.histories[k][-1][0] > history_seconds:
                del self.histories[k]
        for pair in pairs:
            key = pair.track_ids
            if any(k in blocked or k not in reliable for k in key):
                continue
            a, b = (reliable[k] for k in key)
            boxes = [m.observation.bbox for m in [a, b]]
            coverage = bbox_contact_coverage(*boxes)
            if coverage < side.min_contact_coverage:
                continue
            if key in self.contacts:
                self.contacts[key].hits += 1
                continue
            va, vb = (velocities[k] for k in key)
            p = ((boxes[1][:2] + boxes[1][2:]) - (boxes[0][:2] + boxes[0][2:])) / 2
            scale = sum(bbox_diagonal(b) for b in boxes) / 2
            closing = -float(p @ (vb - va)) / max(1, np.linalg.norm(p)) / scale
            speeds = [np.linalg.norm(v) for v in [va, vb]]
            alignment = float(va @ vb) / max(1, speeds[0] * speeds[1])
            crossing = abs(alignment) < side.min_direction_cosine
            prior_boxes = {}
            if crossing:
                other_samples = dict(self.histories[key[1]])
                separate_samples = [
                    (sample_time, box, other_samples[sample_time])
                    for sample_time, box in self.histories[key[0]]
                    if sample_time in other_samples
                    and 0 < t - sample_time <= cfg.occlusion_confirmation_s
                    and bbox_contact_coverage(box, other_samples[sample_time])
                    < side.min_contact_coverage
                ]
                if separate_samples:
                    prior_boxes = dict(zip(key, separate_samples[-1][1:], strict=True))
                if (
                    not separate_samples
                    or coverage < cfg.min_contact_coverage
                    or min(a.age_s, b.age_s) < self.config.min_track_age_s
                ):
                    continue
            if (
                closing < cfg.min_approach_speed_diagonals_s
                or (not crossing and alignment < side.min_direction_cosine)
                or any(
                    speed / bbox_diagonal(box) < cfg.min_prior_speed_diagonals_s
                    for speed, box in zip(speeds, boxes, strict=True)
                )
            ):
                continue
            self.contacts[key] = SideContact(
                t,
                dict(zip(key, boxes, strict=True)),
                dict(zip(key, [va, vb], strict=True)),
                crossing=crossing,
                prior_boxes=prior_boxes,
            )
        events = []
        for key, c in list(self.contacts.items()):
            if t - c.timestamp > side.confirmation_window_s or any(k in blocked for k in key):
                del self.contacts[key]
                continue
            for k in key:
                if k not in reliable:
                    continue
                delta = np.linalg.norm(velocities[k] - c.velocities[k]) / bbox_diagonal(c.boxes[k])
                if delta >= side.min_velocity_change_diagonals_s:
                    c.responders.add(k)
            for k in key:
                m = reliable.get(k)
                if m is None or m.age_s < self.config.min_track_age_s:
                    c.post_since.pop(k, None)
                    continue
                # A contact box can cover two occluded vehicles. Compare shape
                # against the last observation in which they were still separate.
                old = c.prior_boxes.get(k, c.boxes[k])
                box = m.observation.bbox
                old_aspect = (old[2] - old[0]) / max(1, old[3] - old[1])
                aspect = (box[2] - box[0]) / max(1, box[3] - box[1])
                shape = max(aspect / old_aspect, old_aspect / aspect)
                prior = c.velocities[k]
                velocity = velocities[k]
                angle = np.arccos(
                    np.clip(
                        float(prior @ velocity)
                        / max(1, np.linalg.norm(prior) * np.linalg.norm(velocity)),
                        -1,
                        1,
                    )
                )
                slow = (
                    np.linalg.norm(velocity) <= np.linalg.norm(prior) * cfg.post_impact_speed_ratio
                )
                contact_region = np.r_[
                    np.minimum.reduce([b[:2] for b in c.boxes.values()]),
                    np.maximum.reduce([b[2:] for b in c.boxes.values()]),
                ]
                near_contact = (
                    bbox_contact_coverage(box, contact_region) >= cfg.min_contact_coverage
                )
                # Retain rotation only after sustained joint evidence; small box
                # fluctuations must not erase an already observed reaction.
                if (
                    c.crossing
                    and shape >= side.min_shape_change
                    and k in c.post_since
                    and t - c.post_since[k] >= self.config.candidate_duration_s
                ):
                    c.rotated.add(k)
                good = (
                    (c.crossing or c.hits >= 2)
                    and (shape >= side.min_shape_change or k in c.rotated)
                    and (c.crossing or angle >= side.min_heading_change_rad)
                    and (not c.crossing or near_contact)
                    and slow
                    and (c.crossing or any(i != k for i in c.responders))
                )
                if not good:
                    c.post_since.pop(k, None)
                    continue
                c.post_since.setdefault(k, t)
                if t - c.post_since[k] >= self.config.confirm_duration_s:
                    events.append(
                        Event(
                            self.camera_id + "_" + uuid4().hex[:16],
                            self.camera_id,
                            c.timestamp,
                            c.timestamp,
                            t,
                            list(key),
                            0.9,
                            0,
                            [
                                "image_coordinates",
                                "crossing_contact" if c.crossing else "side_contact",
                                "vehicle_rotation",
                                "post_impact_slowdown",
                                "crossing_approach" if c.crossing else "two_vehicle_reaction",
                            ],
                            coordinate_mode="image",
                        )
                    )
                    for i in key:
                        blocked[i] = t + self.config.cooldown_s
                    del self.contacts[key]
                    break
        return events

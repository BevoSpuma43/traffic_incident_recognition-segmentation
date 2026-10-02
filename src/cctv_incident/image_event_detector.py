"""Uncalibrated baseline using pixel motion relative to apparent vehicle size."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .event_detector import EventDetector
from .features import bbox_contact_coverage, bbox_diagonal
from .sideswipe_detector import SideswipeDetector
from .types import Decision


@dataclass
class ContactMemory:
    last_seen: float
    region: np.ndarray
    prior_speeds: dict[int, float]


class ImageEventDetector(EventDetector):
    def __init__(self, config, features, camera_id):
        super().__init__(config, features, camera_id)
        self.coordinate_mode = "image"
        self.box_histories: dict[int, deque] = {}
        self.contacts: dict[tuple[int, ...], ContactMemory] = {}
        self.sideswipes = SideswipeDetector(config, features, camera_id)

    def reset(self):
        super().reset()
        self.contacts.clear()
        self.box_histories.clear()
        self.sideswipes.reset()

    def _update_box_history(self, motions, timestamp):
        # Ground points can jump within a stationary mask; retain observed box motion.
        window = max(self.features.max_gap_s, self.config.confirm_duration_s)
        for key in list(self.box_histories):
            rows = self.box_histories[key]
            while rows and timestamp - rows[0][0] > window:
                rows.popleft()
            if not rows:
                del self.box_histories[key]
        for key, motion in motions.items():
            observation = motion.observation
            if observation.predicted or observation.timestamp_s != timestamp:
                continue
            if observation.quality < self.config.image.min_quality:
                continue
            rows = self.box_histories.setdefault(key, deque(maxlen=256))
            rows.append((timestamp, (observation.bbox[:2] + observation.bbox[2:]) / 2))

    def _box_is_stationary(self, motion):
        rows = self.box_histories.get(motion.observation.track_id, ())
        if not rows or rows[-1][0] - rows[0][0] + 1e-6 < self.config.confirm_duration_s:
            return False
        displacement = np.linalg.norm(np.ptp(np.array([point for _, point in rows]), axis=0))
        tolerance = (
            bbox_diagonal(motion.observation.bbox)
            * self.features.image.stop_speed_diagonals_s
            * self.config.confirm_duration_s
        )
        return displacement <= tolerance

    def update(self, motions, pairs, timestamp, calibration):
        if timestamp <= self.last_timestamp:
            return Decision()
        if timestamp - self.last_timestamp > self.features.max_gap_s:
            self.sideswipes.reset()
        self.last_timestamp = timestamp
        self.cooldowns = {key: until for key, until in self.cooldowns.items() if until > timestamp}
        if not calibration.valid or calibration.units != "px":
            self.reset()
            return Decision("PAUSED", reasons=["image_reference_invalid"])
        cfg = self.config.image
        self._update_box_history(motions, timestamp)
        evidence = {}
        for pair in pairs:
            members = [motions[track_id] for track_id in pair.track_ids]
            if (
                pair.quality < cfg.min_quality
                or min(m.age_s for m in members) < self.config.min_track_age_s
            ):
                continue
            scales = [bbox_diagonal(m.observation.bbox) for m in members]
            scale = sum(scales) / len(scales)
            close = pair.distance / scale <= self.features.image.collision_distance_diagonals
            plausible = (
                pair.closest_distance / scale <= self.features.image.collision_distance_diagonals
            )
            converging = pair.converging and plausible
            relative_position = members[1].position - members[0].position
            relative_velocity = members[1].velocity - members[0].velocity
            closing_speed = max(
                0.0, -float(relative_position @ relative_velocity) / max(pair.distance, 1.0)
            )
            # TTC can be zero for overlapping stopped boxes despite negligible motion.
            low_ttc = (
                pair.ttc is not None
                and pair.ttc <= self.config.low_ttc_s
                and converging
                and closing_speed / scale >= cfg.min_approach_speed_diagonals_s
            )
            overlap = (
                pair.bbox_iou >= cfg.min_bbox_iou
                or bbox_contact_coverage(members[0].observation.bbox, members[1].observation.bbox)
                >= cfg.min_contact_coverage
            )
            braking = any(
                m.deceleration / size >= cfg.deceleration_diagonals_s2
                and m.prior_speed / size >= cfg.min_prior_speed_diagonals_s
                for m, size in zip(members, scales, strict=True)
            )
            stopped = any(
                m.stop_duration_s >= self.config.confirm_duration_s
                and m.prior_speed / size >= cfg.min_prior_speed_diagonals_s
                for m, size in zip(members, scales, strict=True)
            )
            if not (low_ttc or close or pair.track_ids in self.candidates):
                continue
            reasons = {"image_coordinates"}
            if low_ttc:
                reasons.add("image_convergence")
            if overlap:
                reasons.add("image_box_overlap")
            if braking:
                reasons.add("relative_deceleration")
            if stopped:
                reasons.add("post_impact_stop")
            score = min(
                1.0,
                0.35 * (close or plausible)
                + 0.3 * low_ttc
                + 0.3 * braking
                + 0.15 * overlap
                + 0.2 * stopped,
            )
            motion_supported = any(not self._box_is_stationary(m) for m in members)
            trigger = motion_supported and (low_ttc or (close and overlap and braking))
            impact = motion_supported and close and overlap and (braking or stopped)
            post = close and overlap and stopped
            # Require independent reactions in two observed vehicles before bridging
            # an occlusion. One braking vehicle behind a passing car is insufficient.
            two_vehicle_reaction = any(
                turning != slowing
                and members[turning].heading_change >= cfg.heading_change_rad
                and members[turning].prior_speed / scales[turning]
                >= cfg.min_prior_speed_diagonals_s
                and members[slowing].deceleration / scales[slowing] >= cfg.deceleration_diagonals_s2
                and members[slowing].prior_speed / scales[slowing]
                >= cfg.min_prior_speed_diagonals_s
                for turning in range(2)
                for slowing in range(2)
            )
            if (
                impact
                and braking
                and two_vehicle_reaction
                and not any(m.observation.predicted for m in members)
            ):
                reasons.add("two_vehicle_reaction")
                if pair.track_ids not in self.contacts:
                    boxes = [member.observation.bbox for member in members]
                    self.contacts[pair.track_ids] = ContactMemory(
                        timestamp,
                        np.r_[
                            np.minimum(boxes[0][:2], boxes[1][:2]),
                            np.maximum(boxes[0][2:], boxes[1][2:]),
                        ],
                        {m.observation.track_id: m.prior_speed for m in members},
                    )
                self.contacts[pair.track_ids].last_seen = timestamp
            evidence[pair.track_ids] = score, trigger, impact, post, reasons
        self._continue_contacts(motions, evidence, timestamp)
        paired = {track_id for pair in evidence for track_id in pair}
        for track_id, motion in motions.items():
            key = (track_id,)
            if track_id in paired and key not in self.candidates:
                continue
            if motion.quality < cfg.min_quality or motion.age_s < self.config.min_track_age_s:
                continue
            scale = bbox_diagonal(motion.observation.bbox)
            moving_before = motion.prior_speed / scale >= cfg.min_prior_speed_diagonals_s
            abrupt = (
                moving_before
                and not self._box_is_stationary(motion)
                and motion.deceleration / scale >= cfg.deceleration_diagonals_s2
                and motion.heading_change >= cfg.heading_change_rad
            )
            item = self.candidates.get(key)
            sustained_slowdown = (
                item is not None and motion.speed <= motion.prior_speed * 0.35 and moving_before
            )
            stopped = motion.stop_duration_s >= self.config.confirm_duration_s
            if abrupt or item is not None:
                reasons = {"image_coordinates"}
                if abrupt:
                    reasons.update(("relative_deceleration", "abrupt_direction_change"))
                if stopped:
                    reasons.add("post_impact_stop")
                evidence[key] = (
                    0.65 if abrupt or sustained_slowdown else 0.0,
                    abrupt or sustained_slowdown,
                    abrupt,
                    stopped,
                    reasons,
                )
        decision = self._advance(evidence, timestamp, 0.0)
        side_events = self.sideswipes.update(motions, pairs, timestamp, self.cooldowns)
        if side_events:
            decision.events.extend(side_events)
            decision.state = "CONFIRMED"
            decision.score = max(decision.score, *(e.score for e in side_events))
            decision.candidate_ids = sorted({i for e in decision.events for i in e.track_ids})
            decision.reasons = sorted({r for e in decision.events for r in e.reasons})
        return decision

    def _continue_contacts(self, motions, evidence, timestamp):
        """Confirm a measured contact using a still-observed vehicle, never a prediction."""
        cfg = self.config.image
        for key, contact in list(self.contacts.items()):
            if timestamp - contact.last_seen > cfg.occlusion_confirmation_s or any(
                track_id in self.cooldowns for track_id in key
            ):
                del self.contacts[key]
                continue
            candidate = self.candidates.get(key)
            if candidate is None or candidate.impact is None:
                continue
            visible = [
                motions[track_id]
                for track_id in key
                if track_id in motions
                and motions[track_id].quality >= cfg.min_quality
                and not motions[track_id].observation.predicted
                and motions[track_id].observation.timestamp_s == timestamp
            ]
            slowdown = braking = False
            for motion in visible:
                peak = contact.prior_speeds[motion.observation.track_id]
                scale = bbox_diagonal(motion.observation.bbox)
                if peak / scale < cfg.min_prior_speed_diagonals_s:
                    continue
                if (
                    bbox_contact_coverage(motion.observation.bbox, contact.region)
                    < cfg.min_contact_coverage
                ):
                    continue
                slowdown |= motion.speed <= peak * cfg.post_impact_speed_ratio
                braking |= motion.deceleration / scale >= cfg.deceleration_diagonals_s2
            if not (slowdown or braking):
                if key not in evidence:
                    candidate.post_since = None
                continue
            reasons = set(evidence.get(key, (0, False, False, False, set()))[4])
            reasons.update(("image_coordinates", "two_vehicle_reaction"))
            if slowdown:
                reasons.add("post_impact_slowdown")
            if len(visible) < len(key):
                reasons.add("occluded_partner")
            evidence[key] = candidate.peak_score, True, False, slowdown, reasons

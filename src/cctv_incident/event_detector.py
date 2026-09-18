from dataclasses import dataclass, field
from uuid import uuid4

from .types import Decision, Event


@dataclass
class Candidate:
    start: float
    last_seen: float
    observations: int = 0
    impact: float | None = None
    post_since: float | None = None
    reasons: set = field(default_factory=set)
    peak_score: float = 0
    armed: bool = False


class EventDetector:
    def __init__(
        self, config, features, camera_id, min_calibration_confidence=0.55, classifier=None
    ):
        self.config, self.features = config, features
        self.camera_id = camera_id
        self.coordinate_mode = "metric"
        self.min_calibration_confidence = min_calibration_confidence
        self.classifier = classifier
        self.candidates = {}
        self.cooldowns = {}
        self.last_timestamp = -float("inf")

    def reset(self):
        self.candidates.clear()

    def update(self, motions, pairs, timestamp, calibration):
        decision = Decision()
        if timestamp <= self.last_timestamp:
            return decision
        self.last_timestamp = timestamp
        if self.classifier is not None:
            self.classifier.expire(timestamp)
        self.cooldowns = {k: t for k, t in self.cooldowns.items() if t > timestamp}
        if not calibration.metric_valid(self.min_calibration_confidence):
            self.candidates.clear()
            return Decision("PAUSED", reasons=["calibration_not_metric_or_invalid"])
        evidence = {}
        for pair in pairs:
            members = [motions[i] for i in pair.track_ids]
            if (
                pair.quality < self.config.min_quality
                or min(m.age_s for m in members) < self.config.min_track_age_s
            ):
                continue
            close = pair.distance <= self.features.collision_distance_m
            low_ttc = pair.ttc is not None and pair.ttc <= self.config.low_ttc_s
            braking = any(
                m.deceleration >= self.config.deceleration_mps2
                and m.prior_speed >= self.config.min_prior_speed_mps
                for m in members
            )
            stopped = any(
                m.stop_duration_s >= self.config.confirm_duration_s
                and m.prior_speed >= self.config.min_prior_speed_mps
                for m in members
            )
            classifier_score = None
            if self.classifier is not None:
                from .classifier import feature_vector

                classifier_score = self.classifier.score(
                    pair.track_ids, timestamp, feature_vector(pair, members)
                )
            if not (close or low_ttc or pair.track_ids in self.candidates):
                continue
            reasons = set()
            if close:
                reasons.add("spatial_proximity")
            if low_ttc:
                reasons.add("low_ttc")
            if braking:
                reasons.add("high_deceleration")
            if stopped:
                reasons.add("post_impact_stop")
            # Proximity alone cannot arm a candidate or confirm an event.
            plausible = close or (
                pair.converging and pair.closest_distance <= self.features.collision_distance_m
            )
            score = min(1.0, 0.35 * plausible + 0.3 * low_ttc + 0.3 * braking + 0.35 * stopped)
            trigger = (plausible and low_ttc) or (close and braking)
            impact = close and (braking or stopped)
            post = close and stopped
            if classifier_score is not None and trigger:
                score = classifier_score
            evidence[pair.track_ids] = (score, trigger, impact, post, reasons)

        # A single-vehicle impact requires both abrupt deceleration and direction change.
        paired_ids = {i for ids in evidence for i in ids}
        for track_id, m in motions.items():
            key = (track_id,)
            if track_id in paired_ids and key not in self.candidates:
                continue
            if m.quality < self.config.min_quality or m.age_s < self.config.min_track_age_s:
                continue
            abrupt = m.deceleration >= self.config.deceleration_mps2 and m.heading_change > 0.6
            stopped = m.stop_duration_s >= self.config.confirm_duration_s
            trigger = abrupt and m.prior_speed >= self.config.min_prior_speed_mps
            if trigger or key in self.candidates:
                reasons = {"abrupt_direction_change", "high_deceleration"} if trigger else set()
                if stopped:
                    reasons.add("post_impact_stop")
                evidence[key] = (
                    0.65 if trigger else 0.35 * stopped,
                    trigger,
                    trigger,
                    stopped,
                    reasons,
                )

        return self._advance(evidence, timestamp, calibration.confidence)

    def _advance(self, evidence, timestamp, calibration_confidence):
        decision = Decision()
        rejected = False
        for key in list(self.candidates):
            item = self.candidates[key]
            # Once contact is observed, allow a bounded post-impact confirmation phase.
            phase_start = item.impact if item.impact is not None else item.start
            if (
                timestamp - item.last_seen > self.config.evidence_gap_s
                or timestamp - phase_start > self.config.candidate_timeout_s
            ):
                del self.candidates[key]
                rejected = True
        for key, (score, trigger, impact, post, reasons) in evidence.items():
            if any(track_id in self.cooldowns for track_id in key):
                continue
            item = self.candidates.get(key)
            if item is None:
                if not trigger or score < self.config.candidate_threshold:
                    continue
                item = self.candidates[key] = Candidate(timestamp, timestamp)
            if timestamp - item.last_seen > self.config.evidence_gap_s:
                del self.candidates[key]
                continue
            item.last_seen = timestamp
            item.observations += 1
            item.reasons.update(reasons)
            item.peak_score = max(item.peak_score, score)
            if (
                trigger
                and timestamp - item.start >= self.config.candidate_duration_s
                and item.observations >= 3
            ):
                item.armed = True
            if impact and item.impact is None:
                item.impact = timestamp
            if post and item.impact is not None:
                if item.post_since is None:
                    item.post_since = timestamp
            else:
                item.post_since = None
            combined = min(1.0, item.peak_score + (0.25 if post and item.impact is not None else 0))
            if combined >= decision.score:
                decision.score, decision.reasons = combined, sorted(item.reasons)
                decision.candidate_ids = list(key)
            decision.state = "CANDIDATE"
            if (
                item.armed
                and item.post_since is not None
                and timestamp - item.post_since >= self.config.confirm_duration_s
                and combined >= self.config.confirm_threshold
            ):
                event = Event(
                    event_id=f"{self.camera_id}_{uuid4().hex[:16]}",
                    camera_id=self.camera_id,
                    start_time_s=item.start,
                    impact_time_s=item.impact,
                    confirm_time_s=timestamp,
                    track_ids=list(key),
                    score=combined,
                    calibration_confidence=calibration_confidence,
                    coordinate_mode=self.coordinate_mode,
                    reasons=sorted(item.reasons),
                )
                decision.events.append(event)
                decision.state = "CONFIRMED"
                for track_id in key:
                    self.cooldowns[track_id] = timestamp + self.config.cooldown_s
                del self.candidates[key]
        if decision.events:
            decision.state = "CONFIRMED"
        if decision.state == "NORMAL" and rejected:
            decision.state = "REJECTED"
        if decision.state == "NORMAL" and self.cooldowns:
            decision.state = "COOLDOWN"
        return decision

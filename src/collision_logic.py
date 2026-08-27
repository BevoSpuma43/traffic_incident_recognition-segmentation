"""Rilevamento temporale e spiegabile delle collisioni tra veicoli."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import AppConfig
from src.geometry import (
    bbox_distance,
    dilated_masks_touch_in_roi,
    mask_intersection_area_in_roi,
    normalized_overlap,
)
from src.models import CollisionEvent, PairDiagnostic, VehicleState


@dataclass(slots=True)
class _SpatialEvidence:
    contact: bool
    overlap_area: int
    overlap_ratio: float
    distance_px: float


@dataclass(slots=True)
class _PairState:
    """Memoria temporale di una coppia ordinata di track."""

    status: str = "clear"
    last_seen_frame: int = -1
    first_contact_frame: int | None = None
    last_contact_frame: int | None = None
    last_dynamic_frame: int | None = None
    last_approach_frame: int | None = None
    candidate_frames: int = 0
    cooldown_until: int = -1
    saw_hard_deceleration: bool = False
    saw_dual_stop: bool = False
    saw_stop_transition: bool = False
    max_overlap_area: int = 0
    max_overlap_ratio: float = 0.0
    min_distance_px: float = math.inf
    max_closing_speed_px: float = 0.0

    def reset_candidate(self, *, keep_temporal: bool = False) -> None:
        previous_approach = self.last_approach_frame
        previous_closing_speed = self.max_closing_speed_px
        previous_dynamic = self.last_dynamic_frame
        previous_hard_deceleration = self.saw_hard_deceleration
        previous_dual_stop = self.saw_dual_stop
        previous_stop_transition = self.saw_stop_transition
        self.status = "clear"
        self.first_contact_frame = None
        self.last_contact_frame = None
        self.last_dynamic_frame = None
        self.last_approach_frame = None
        self.candidate_frames = 0
        self.saw_hard_deceleration = False
        self.saw_dual_stop = False
        self.saw_stop_transition = False
        self.max_overlap_area = 0
        self.max_overlap_ratio = 0.0
        self.min_distance_px = math.inf
        self.max_closing_speed_px = 0.0
        if keep_temporal:
            self.last_approach_frame = previous_approach
            self.max_closing_speed_px = previous_closing_speed
            self.last_dynamic_frame = previous_dynamic
            self.saw_hard_deceleration = previous_hard_deceleration
            self.saw_dual_stop = previous_dual_stop
            self.saw_stop_transition = previous_stop_transition


class CollisionDetector:
    """Combina prossimita spaziale e anomalie cinematiche nel tempo."""

    def __init__(
        self,
        mask_overlap_threshold: int,
        stopped_frames_threshold: int,
        stopped_speed_threshold: float,
        strong_deceleration_threshold: float,
        *,
        mask_overlap_ratio_threshold: float = 0.015,
        contact_distance_threshold_px: float = 8.0,
        mask_dilation_pixels: int = 3,
        min_preimpact_speed_px: float = 3.0,
        min_closing_speed_px: float = 1.5,
        impact_window_frames: int = 5,
        collision_confirmation_frames: int = 2,
        collision_cooldown_frames: int = 30,
        pair_state_ttl_frames: int = 45,
    ) -> None:
        self.mask_overlap_threshold = max(0, int(mask_overlap_threshold))
        self.stopped_frames_threshold = max(1, int(stopped_frames_threshold))
        self.stopped_speed_threshold = max(0.0, float(stopped_speed_threshold))
        self.strong_deceleration_threshold = float(strong_deceleration_threshold)
        self.mask_overlap_ratio_threshold = max(0.0, float(mask_overlap_ratio_threshold))
        self.contact_distance_threshold_px = max(0.0, float(contact_distance_threshold_px))
        self.mask_dilation_pixels = max(0, int(mask_dilation_pixels))
        self.min_preimpact_speed_px = max(0.0, float(min_preimpact_speed_px))
        self.min_closing_speed_px = max(0.0, float(min_closing_speed_px))
        self.impact_window_frames = max(1, int(impact_window_frames))
        self.collision_confirmation_frames = max(1, int(collision_confirmation_frames))
        self.collision_cooldown_frames = max(1, int(collision_cooldown_frames))
        self.pair_state_ttl_frames = max(1, int(pair_state_ttl_frames))
        self._pair_states: dict[tuple[int, int], _PairState] = {}
        self.last_diagnostics: list[PairDiagnostic] = []

    @classmethod
    def from_config(cls, config: AppConfig) -> CollisionDetector:
        return cls(
            mask_overlap_threshold=config.mask_overlap_threshold,
            stopped_frames_threshold=config.stopped_frames_threshold,
            stopped_speed_threshold=config.stopped_speed_threshold,
            strong_deceleration_threshold=config.strong_deceleration_threshold,
            mask_overlap_ratio_threshold=config.mask_overlap_ratio_threshold,
            contact_distance_threshold_px=config.contact_distance_threshold_px,
            mask_dilation_pixels=config.mask_dilation_pixels,
            min_preimpact_speed_px=config.min_preimpact_speed_px,
            min_closing_speed_px=config.min_closing_speed_px,
            impact_window_frames=config.impact_window_frames,
            collision_confirmation_frames=config.collision_confirmation_frames,
            collision_cooldown_frames=config.collision_cooldown_frames,
            pair_state_ttl_frames=config.pair_state_ttl_frames,
        )

    def detect_collisions(
        self,
        vehicle_states: list[VehicleState] | dict[int, VehicleState],
        frame_index: int,
    ) -> list[CollisionEvent]:
        """Aggiorna gli stati di coppia ed emette solo nuove conferme."""
        states = (
            list(vehicle_states.values())
            if isinstance(vehicle_states, dict)
            else list(vehicle_states)
        )
        self._remove_stale_pairs(frame_index)
        self.last_diagnostics = []
        collisions: list[CollisionEvent] = []
        seen_pairs: set[tuple[int, int]] = set()

        for index, vehicle_a in enumerate(states):
            for vehicle_b in states[index + 1 :]:
                if vehicle_a.track_id == vehicle_b.track_id:
                    continue
                pair = self._pair_key(vehicle_a.track_id, vehicle_b.track_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                pair_state = self._pair_states.setdefault(pair, _PairState())
                pair_state.last_seen_frame = frame_index
                if pair_state.status == "cooldown":
                    if frame_index < pair_state.cooldown_until:
                        continue
                    pair_state.reset_candidate()

                self._expire_temporal_evidence(pair_state, frame_index)

                spatial = self._spatial_evidence(vehicle_a, vehicle_b)
                closing_speed = self._closing_speed(vehicle_a, vehicle_b)
                hard_deceleration = self._hard_deceleration(vehicle_a, vehicle_b)
                dual_stop = self._both_stopped(vehicle_a, vehicle_b)
                stop_transition = self._recent_stop_transition(
                    vehicle_a, frame_index
                ) or self._recent_stop_transition(vehicle_b, frame_index)
                had_motion = self._had_recent_motion(
                    vehicle_a, frame_index
                ) or self._had_recent_motion(vehicle_b, frame_index)

                if spatial.contact:
                    if pair_state.first_contact_frame is None:
                        pair_state.first_contact_frame = frame_index
                    pair_state.last_contact_frame = frame_index
                    pair_state.max_overlap_area = max(
                        pair_state.max_overlap_area, spatial.overlap_area
                    )
                    pair_state.max_overlap_ratio = max(
                        pair_state.max_overlap_ratio, spatial.overlap_ratio
                    )
                    pair_state.min_distance_px = min(
                        pair_state.min_distance_px, spatial.distance_px
                    )

                if closing_speed >= self.min_closing_speed_px:
                    pair_state.last_approach_frame = frame_index
                    pair_state.max_closing_speed_px = max(
                        pair_state.max_closing_speed_px, closing_speed
                    )

                if hard_deceleration or stop_transition or (dual_stop and had_motion):
                    pair_state.last_dynamic_frame = frame_index
                    pair_state.saw_hard_deceleration |= hard_deceleration
                    pair_state.saw_stop_transition |= stop_transition
                    pair_state.saw_dual_stop |= dual_stop

                has_recent_contact = self._is_recent(
                    pair_state.last_contact_frame, frame_index
                )
                has_recent_dynamic = self._is_recent(
                    pair_state.last_dynamic_frame, frame_index
                )
                has_recent_approach = self._is_recent(
                    pair_state.last_approach_frame, frame_index
                )
                strong_overlap = pair_state.max_overlap_ratio >= max(
                    0.05, self.mask_overlap_ratio_threshold * 2.0
                )
                emitted = False
                if (
                    has_recent_contact
                    and has_recent_dynamic
                    and had_motion
                    and (has_recent_approach or strong_overlap)
                ):
                    pair_state.status = "contact_candidate"
                    pair_state.candidate_frames += 1
                elif not has_recent_contact:
                    pair_state.reset_candidate(keep_temporal=True)
                else:
                    pair_state.candidate_frames = 0

                if (
                    pair_state.candidate_frames
                    >= self.collision_confirmation_frames
                ):
                    ordered_a, ordered_b = self._ordered_states(vehicle_a, vehicle_b)
                    collisions.append(
                        self._build_event(
                            pair, pair_state, ordered_a, ordered_b, frame_index
                        )
                    )
                    emitted = True
                    pair_state.status = "cooldown"
                    pair_state.cooldown_until = (
                        frame_index + self.collision_cooldown_frames
                    )
                    pair_state.candidate_frames = 0

                self.last_diagnostics.append(
                    PairDiagnostic(
                        frame_index=frame_index,
                        track_id_a=pair[0],
                        track_id_b=pair[1],
                        contact=spatial.contact,
                        overlap_area=spatial.overlap_area,
                        overlap_ratio=spatial.overlap_ratio,
                        spatial_distance_px=spatial.distance_px,
                        closing_speed_px=closing_speed,
                        hard_deceleration=hard_deceleration,
                        dual_stop=dual_stop,
                        stop_transition=stop_transition,
                        had_recent_motion=had_motion,
                        candidate_frames=pair_state.candidate_frames,
                        pair_status=pair_state.status,
                        emitted=emitted,
                    )
                )

        return collisions

    @staticmethod
    def _pair_key(id_a: int, id_b: int) -> tuple[int, int]:
        return (id_a, id_b) if id_a <= id_b else (id_b, id_a)

    @staticmethod
    def _ordered_states(
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> tuple[VehicleState, VehicleState]:
        if vehicle_a.track_id <= vehicle_b.track_id:
            return vehicle_a, vehicle_b
        return vehicle_b, vehicle_a

    def _spatial_evidence(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> _SpatialEvidence:
        distance = bbox_distance(vehicle_a.bbox, vehicle_b.bbox)
        if distance > self.contact_distance_threshold_px:
            return _SpatialEvidence(False, 0, 0.0, distance)

        overlap_area = mask_intersection_area_in_roi(
            vehicle_a.mask,
            vehicle_b.mask,
            vehicle_a.bbox,
            vehicle_b.bbox,
            padding=self.mask_dilation_pixels,
        )
        overlap_ratio = normalized_overlap(
            overlap_area,
            self._current_mask_area(vehicle_a),
            self._current_mask_area(vehicle_b),
        )
        meaningful_overlap = overlap_area > 0 and (
            overlap_area >= self.mask_overlap_threshold
            or overlap_ratio >= self.mask_overlap_ratio_threshold
        )
        near_contact = dilated_masks_touch_in_roi(
            vehicle_a.mask,
            vehicle_b.mask,
            vehicle_a.bbox,
            vehicle_b.bbox,
            self.mask_dilation_pixels,
        )
        return _SpatialEvidence(
            meaningful_overlap or near_contact,
            overlap_area,
            overlap_ratio,
            distance,
        )

    @staticmethod
    def _current_mask_area(vehicle: VehicleState) -> int:
        if vehicle.history:
            return vehicle.history[-1].mask_area
        if vehicle.mask is None:
            return 0
        return int((vehicle.mask > 0).sum())

    @staticmethod
    def _closing_speed(vehicle_a: VehicleState, vehicle_b: VehicleState) -> float:
        if not vehicle_a.kinematics_valid or not vehicle_b.kinematics_valid:
            return 0.0
        anchor_a = vehicle_a.motion_anchor or vehicle_a.centroid
        anchor_b = vehicle_b.motion_anchor or vehicle_b.centroid
        if anchor_a is None or anchor_b is None:
            return 0.0
        dx = float(anchor_b.x) - float(anchor_a.x)
        dy = float(anchor_b.y) - float(anchor_a.y)
        distance = math.hypot(dx, dy)
        if distance <= 1e-6:
            return 0.0
        relative_vx = vehicle_b.velocity_x_px - vehicle_a.velocity_x_px
        relative_vy = vehicle_b.velocity_y_px - vehicle_a.velocity_y_px
        distance_rate = (relative_vx * dx + relative_vy * dy) / distance
        return max(0.0, -distance_rate)

    def _both_stopped(self, vehicle_a: VehicleState, vehicle_b: VehicleState) -> bool:
        return (
            vehicle_a.kinematics_valid
            and vehicle_b.kinematics_valid
            and vehicle_a.stopped_frames >= self.stopped_frames_threshold
            and vehicle_b.stopped_frames >= self.stopped_frames_threshold
            and vehicle_a.speed_px <= self.stopped_speed_threshold
            and vehicle_b.speed_px <= self.stopped_speed_threshold
        )

    def _hard_deceleration(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> bool:
        return (
            vehicle_a.kinematics_valid
            and vehicle_a.acceleration_px <= self.strong_deceleration_threshold
        ) or (
            vehicle_b.kinematics_valid
            and vehicle_b.acceleration_px <= self.strong_deceleration_threshold
        )

    def _had_recent_motion(self, vehicle: VehicleState, frame_index: int) -> bool:
        history_window = self.impact_window_frames * 2
        return any(
            frame_index - sample.frame_index <= history_window
            and sample.speed_px >= self.min_preimpact_speed_px
            for sample in vehicle.history
        )

    def _recent_stop_transition(self, vehicle: VehicleState, frame_index: int) -> bool:
        if not vehicle.kinematics_valid or vehicle.speed_px > self.stopped_speed_threshold:
            return False
        return any(
            0 < frame_index - sample.frame_index <= self.impact_window_frames
            and sample.speed_px >= self.min_preimpact_speed_px
            for sample in vehicle.history
        )

    def _is_recent(self, evidence_frame: int | None, frame_index: int) -> bool:
        return evidence_frame is not None and (
            0 <= frame_index - evidence_frame <= self.impact_window_frames
        )

    def _expire_temporal_evidence(
        self,
        pair_state: _PairState,
        frame_index: int,
    ) -> None:
        if not self._is_recent(pair_state.last_dynamic_frame, frame_index):
            pair_state.last_dynamic_frame = None
            pair_state.saw_hard_deceleration = False
            pair_state.saw_dual_stop = False
            pair_state.saw_stop_transition = False
        if not self._is_recent(pair_state.last_approach_frame, frame_index):
            pair_state.last_approach_frame = None
            pair_state.max_closing_speed_px = 0.0

    def _build_event(
        self,
        pair: tuple[int, int],
        pair_state: _PairState,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
        frame_index: int,
    ) -> CollisionEvent:
        reasons: list[str] = ["temporal_contact"]
        if pair_state.saw_hard_deceleration:
            reasons.append("hard_deceleration")
        if pair_state.saw_stop_transition:
            reasons.append("stop_transition")
        if pair_state.saw_dual_stop:
            reasons.append("dual_stop")

        confidence = 0.35 + min(0.2, pair_state.max_overlap_ratio)
        confidence += 0.2 if pair_state.saw_hard_deceleration else 0.0
        confidence += 0.15 if pair_state.saw_stop_transition else 0.0
        confidence += 0.1 if pair_state.max_closing_speed_px > 0.0 else 0.0

        return CollisionEvent(
            frame_index=frame_index,
            track_id_a=pair[0],
            track_id_b=pair[1],
            overlap_area=pair_state.max_overlap_area,
            reason="_and_".join(reasons),
            speed_a=vehicle_a.speed_px,
            speed_b=vehicle_b.speed_px,
            acceleration_a=vehicle_a.acceleration_px,
            acceleration_b=vehicle_b.acceleration_px,
            overlap_ratio=pair_state.max_overlap_ratio,
            spatial_distance_px=(
                0.0 if math.isinf(pair_state.min_distance_px) else pair_state.min_distance_px
            ),
            closing_speed_px=pair_state.max_closing_speed_px,
            confidence=min(1.0, confidence),
            first_contact_frame=pair_state.first_contact_frame,
            confirmation_frame=frame_index,
        )

    def _remove_stale_pairs(self, frame_index: int) -> None:
        stale = [
            pair
            for pair, state in self._pair_states.items()
            if frame_index - state.last_seen_frame > self.pair_state_ttl_frames
        ]
        for pair in stale:
            del self._pair_states[pair]

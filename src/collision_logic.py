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
from src.kinematics import compute_bbox_scale_px
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
    approach_frames: int = 0
    candidate_frames: int = 0
    cooldown_until: int = -1
    saw_hard_deceleration: bool = False
    saw_dual_stop: bool = False
    saw_stop_transition: bool = False
    saw_trajectory_deflection: bool = False
    saw_target_impulse: bool = False
    saw_occlusion_bridge: bool = False
    sustained_approach: bool = False
    reaction_frames: int = 0
    last_crossing_approach_frame: int | None = None
    last_observation_gap_frame: int | None = None
    observation_gap_frames: int = 0
    max_overlap_area: int = 0
    max_overlap_ratio: float = 0.0
    min_distance_px: float = math.inf
    max_closing_speed_px: float = 0.0
    preexisting_contact: bool = False
    separation_frames: int = 0
    preexisting_frames: int = 0

    def reset_candidate(self, *, keep_temporal: bool = False) -> None:
        previous_approach = self.last_approach_frame
        previous_approach_frames = self.approach_frames
        previous_closing_speed = self.max_closing_speed_px
        previous_dynamic = self.last_dynamic_frame
        previous_hard_deceleration = self.saw_hard_deceleration
        previous_dual_stop = self.saw_dual_stop
        previous_stop_transition = self.saw_stop_transition
        previous_trajectory_deflection = self.saw_trajectory_deflection
        previous_target_impulse = self.saw_target_impulse
        previous_occlusion_bridge = self.saw_occlusion_bridge
        previous_sustained_approach = self.sustained_approach
        previous_reaction_frames = self.reaction_frames
        previous_crossing_approach = self.last_crossing_approach_frame
        previous_gap_frame = self.last_observation_gap_frame
        previous_gap_frames = self.observation_gap_frames
        self.status = "clear"
        self.first_contact_frame = None
        self.last_contact_frame = None
        self.last_dynamic_frame = None
        self.last_approach_frame = None
        self.approach_frames = 0
        self.candidate_frames = 0
        self.saw_hard_deceleration = False
        self.saw_dual_stop = False
        self.saw_stop_transition = False
        self.saw_trajectory_deflection = False
        self.saw_target_impulse = False
        self.saw_occlusion_bridge = False
        self.sustained_approach = False
        self.reaction_frames = 0
        self.last_crossing_approach_frame = None
        self.last_observation_gap_frame = None
        self.observation_gap_frames = 0
        self.max_overlap_area = 0
        self.max_overlap_ratio = 0.0
        self.min_distance_px = math.inf
        self.max_closing_speed_px = 0.0
        if keep_temporal:
            self.last_approach_frame = previous_approach
            self.approach_frames = previous_approach_frames
            self.max_closing_speed_px = previous_closing_speed
            self.last_dynamic_frame = previous_dynamic
            self.saw_hard_deceleration = previous_hard_deceleration
            self.saw_dual_stop = previous_dual_stop
            self.saw_stop_transition = previous_stop_transition
            self.saw_trajectory_deflection = previous_trajectory_deflection
            self.saw_target_impulse = previous_target_impulse
            self.saw_occlusion_bridge = previous_occlusion_bridge
            self.sustained_approach = previous_sustained_approach
            self.reaction_frames = previous_reaction_frames
            self.last_crossing_approach_frame = previous_crossing_approach
            self.last_observation_gap_frame = previous_gap_frame
            self.observation_gap_frames = previous_gap_frames


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
        motion_confirmation_frames: int = 3,
        min_closing_speed_px: float = 1.5,
        approach_confirmation_frames: int = 3,
        approach_evidence_window_frames: int = 2,
        impact_window_frames: int = 5,
        trajectory_history_frames: int = 12,
        trajectory_reaction_lag_frames: int = 2,
        trajectory_min_displacement_px: float = 12.0,
        trajectory_deflection_angle_deg: float = 45.0,
        crossing_history_frames: int = 6,
        crossing_min_angle_deg: float = 45.0,
        target_impulse_acceleration_threshold: float = 8.0,
        crossing_dual_stop_bridge_frames: int = 12,
        crossing_dual_stop_min_gap_frames: int = 2,
        stationary_history_frames: int = 10,
        stationary_history_ratio: float = 0.75,
        impact_reaction_confirmation_frames: int = 2,
        max_contact_candidate_age_frames: int = 5,
        collision_confirmation_frames: int = 2,
        collision_cooldown_frames: int = 30,
        pair_state_ttl_frames: int = 45,
        preexisting_contact_release_frames: int = 3,
        preexisting_contact_max_track_age_frames: int = 10,
        preexisting_contact_max_frames: int = 45,
        kinematic_normalization_enabled: bool = False,
        normalize_time_windows: bool = False,
        reference_scale_px: float = 90.0,
        reference_fps: float = 15.0,
        video_fps: float = 0.0,
    ) -> None:
        self.mask_overlap_threshold = max(0, int(mask_overlap_threshold))
        self.stopped_frames_threshold = max(1, int(stopped_frames_threshold))
        self.stopped_speed_threshold = max(0.0, float(stopped_speed_threshold))
        self.strong_deceleration_threshold = float(strong_deceleration_threshold)
        self.mask_overlap_ratio_threshold = max(0.0, float(mask_overlap_ratio_threshold))
        self.contact_distance_threshold_px = max(0.0, float(contact_distance_threshold_px))
        self.mask_dilation_pixels = max(0, int(mask_dilation_pixels))
        self.min_preimpact_speed_px = max(0.0, float(min_preimpact_speed_px))
        self.motion_confirmation_frames = max(1, int(motion_confirmation_frames))
        self.min_closing_speed_px = max(0.0, float(min_closing_speed_px))
        self.approach_confirmation_frames = max(
            1, int(approach_confirmation_frames)
        )
        self.approach_evidence_window_frames = max(
            0, int(approach_evidence_window_frames)
        )
        self.impact_window_frames = max(1, int(impact_window_frames))
        self.trajectory_history_frames = max(3, int(trajectory_history_frames))
        self.trajectory_reaction_lag_frames = max(
            1, int(trajectory_reaction_lag_frames)
        )
        self.trajectory_min_displacement_px = max(
            0.0, float(trajectory_min_displacement_px)
        )
        self.trajectory_deflection_angle_deg = min(
            180.0, max(0.0, float(trajectory_deflection_angle_deg))
        )
        self.crossing_history_frames = max(2, int(crossing_history_frames))
        self.crossing_min_angle_deg = min(
            89.0, max(0.0, float(crossing_min_angle_deg))
        )
        self.target_impulse_acceleration_threshold = max(
            0.0, float(target_impulse_acceleration_threshold)
        )
        self.crossing_dual_stop_bridge_frames = max(
            1, int(crossing_dual_stop_bridge_frames)
        )
        self.crossing_dual_stop_min_gap_frames = max(
            1, int(crossing_dual_stop_min_gap_frames)
        )
        self.stationary_history_frames = max(3, int(stationary_history_frames))
        self.stationary_history_ratio = min(
            1.0, max(0.0, float(stationary_history_ratio))
        )
        self.impact_reaction_confirmation_frames = max(
            1, int(impact_reaction_confirmation_frames)
        )
        self.max_contact_candidate_age_frames = max(
            0, int(max_contact_candidate_age_frames)
        )
        self.collision_confirmation_frames = max(1, int(collision_confirmation_frames))
        self.collision_cooldown_frames = max(1, int(collision_cooldown_frames))
        self.pair_state_ttl_frames = max(1, int(pair_state_ttl_frames))
        self.preexisting_contact_release_frames = max(
            0, int(preexisting_contact_release_frames)
        )
        self.preexisting_contact_max_track_age_frames = max(
            0, int(preexisting_contact_max_track_age_frames)
        )
        self.preexisting_contact_max_frames = max(
            0, int(preexisting_contact_max_frames)
        )
        self.kinematic_normalization_enabled = bool(
            kinematic_normalization_enabled
        )
        self.normalize_time_windows = bool(normalize_time_windows)
        self.reference_scale_px = max(1e-6, float(reference_scale_px))
        self.reference_fps = max(1e-6, float(reference_fps))
        self.video_fps = max(0.0, float(video_fps))
        # Converte una soglia per-frame tarata a `reference_fps` verso il frame
        # rate reale: a fps piu alto lo stesso moto fisico produce meno pixel
        # per frame, quindi le soglie di velocita devono scendere e le finestre
        # espresse in frame devono allungarsi. Calcolato una volta sola perche
        # viene usato in ogni confronto di ogni coppia di ogni frame.
        self._time_factor = 1.0
        if self.kinematic_normalization_enabled and self.video_fps > 0.0:
            self._time_factor = self.reference_fps / self.video_fps
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
            motion_confirmation_frames=config.motion_confirmation_frames,
            min_closing_speed_px=config.min_closing_speed_px,
            approach_confirmation_frames=config.approach_confirmation_frames,
            approach_evidence_window_frames=(
                config.approach_evidence_window_frames
            ),
            impact_window_frames=config.impact_window_frames,
            trajectory_history_frames=config.trajectory_history_frames,
            trajectory_reaction_lag_frames=(
                config.trajectory_reaction_lag_frames
            ),
            trajectory_min_displacement_px=(
                config.trajectory_min_displacement_px
            ),
            trajectory_deflection_angle_deg=(
                config.trajectory_deflection_angle_deg
            ),
            crossing_history_frames=config.crossing_history_frames,
            crossing_min_angle_deg=config.crossing_min_angle_deg,
            target_impulse_acceleration_threshold=(
                config.target_impulse_acceleration_threshold
            ),
            crossing_dual_stop_bridge_frames=(
                config.crossing_dual_stop_bridge_frames
            ),
            crossing_dual_stop_min_gap_frames=(
                config.crossing_dual_stop_min_gap_frames
            ),
            stationary_history_frames=config.stationary_history_frames,
            stationary_history_ratio=config.stationary_history_ratio,
            impact_reaction_confirmation_frames=(
                config.impact_reaction_confirmation_frames
            ),
            max_contact_candidate_age_frames=(
                config.max_contact_candidate_age_frames
            ),
            collision_confirmation_frames=config.collision_confirmation_frames,
            collision_cooldown_frames=config.collision_cooldown_frames,
            pair_state_ttl_frames=config.pair_state_ttl_frames,
            preexisting_contact_release_frames=(
                config.preexisting_contact_release_frames
            ),
            preexisting_contact_max_track_age_frames=(
                config.preexisting_contact_max_track_age_frames
            ),
            preexisting_contact_max_frames=(
                config.preexisting_contact_max_frames
            ),
            kinematic_normalization_enabled=(
                config.kinematic_normalization_enabled
            ),
            normalize_time_windows=config.normalize_time_windows,
            reference_scale_px=config.reference_scale_px,
            reference_fps=config.reference_fps,
            video_fps=config.video_fps,
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

                is_new_pair = pair not in self._pair_states
                pair_state = self._pair_states.setdefault(pair, _PairState())
                previous_pair_frame = pair_state.last_seen_frame
                pair_state.last_seen_frame = frame_index
                if (
                    previous_pair_frame >= 0
                    and frame_index - previous_pair_frame > 1
                ):
                    pair_state.approach_frames = 0
                    pair_state.observation_gap_frames = (
                        frame_index - previous_pair_frame - 1
                    )
                    pair_state.last_observation_gap_frame = frame_index
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
                stationary_role = self._stationary_target_and_mover(
                    vehicle_a, vehicle_b, frame_index
                )
                deflection_a = self._trajectory_deflection(
                    vehicle_a, frame_index
                )
                deflection_b = self._trajectory_deflection(
                    vehicle_b, frame_index
                )
                crossing_trajectories = self._crossing_trajectories(
                    vehicle_a, vehicle_b, frame_index
                )
                target_impulse = self._target_impulse(vehicle_a, vehicle_b)
                moving_vehicle_reaction = False
                trajectory_deflection = False
                if stationary_role is not None:
                    mover, _ = stationary_role
                    mover_hard_deceleration = self._vehicle_hard_deceleration(
                        mover
                    )
                    mover_stop_transition = self._recent_stop_transition(
                        mover, frame_index
                    )
                    trajectory_deflection = (
                        deflection_a if mover is vehicle_a else deflection_b
                    )
                    moving_vehicle_reaction = (
                        mover_hard_deceleration
                        or mover_stop_transition
                        or trajectory_deflection
                    )
                    # In una coppia veicolo-in-moto/bersaglio-fermo, rumore
                    # cinematico del bersaglio non e una reazione d'impatto.
                    hard_deceleration = mover_hard_deceleration
                    stop_transition = mover_stop_transition

                contact_context = spatial.contact or self._is_recent(
                    pair_state.last_contact_frame, frame_index
                )
                if stationary_role is not None and contact_context:
                    if moving_vehicle_reaction:
                        pair_state.reaction_frames += 1
                    else:
                        pair_state.reaction_frames = 0
                else:
                    pair_state.reaction_frames = 0

                if (
                    is_new_pair
                    and spatial.contact
                    and self.preexisting_contact_release_frames > 0
                    and self._both_tracks_are_new(vehicle_a, vehicle_b)
                ):
                    pair_state.preexisting_contact = True
                    pair_state.status = "preexisting_contact"

                if pair_state.preexisting_contact:
                    pair_state.preexisting_frames += 1
                    if spatial.contact:
                        pair_state.separation_frames = 0
                    else:
                        pair_state.separation_frames += 1
                    if self._preexisting_contact_is_released(pair_state):
                        pair_state.preexisting_contact = False
                        pair_state.separation_frames = 0
                        pair_state.preexisting_frames = 0
                        pair_state.reset_candidate()

                    self._record_diagnostic(
                        frame_index=frame_index,
                        pair=pair,
                        pair_state=pair_state,
                        spatial=spatial,
                        closing_speed=closing_speed,
                        hard_deceleration=hard_deceleration,
                        dual_stop=dual_stop,
                        stop_transition=stop_transition,
                        had_motion=had_motion,
                        stationary_target=stationary_role is not None,
                        moving_vehicle_reaction=moving_vehicle_reaction,
                        trajectory_deflection=trajectory_deflection,
                        crossing_trajectories=crossing_trajectories,
                        target_impulse=target_impulse,
                        bridged_dual_stop=False,
                        bridged_strong_impact=False,
                        emitted=False,
                    )
                    continue

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

                if closing_speed >= self._pair_speed_threshold(
                    self.min_closing_speed_px, vehicle_a, vehicle_b
                ):
                    pair_state.approach_frames += 1
                    pair_state.max_closing_speed_px = max(
                        pair_state.max_closing_speed_px, closing_speed
                    )
                    if pair_state.approach_frames >= self._window(
                        self.approach_confirmation_frames
                    ):
                        pair_state.last_approach_frame = frame_index
                        pair_state.sustained_approach = True
                        if crossing_trajectories:
                            pair_state.last_crossing_approach_frame = frame_index
                else:
                    pair_state.approach_frames = 0

                dynamic_evidence = (
                    moving_vehicle_reaction
                    if stationary_role is not None
                    else hard_deceleration
                    or stop_transition
                    or (dual_stop and had_motion)
                    or target_impulse
                )
                if dynamic_evidence:
                    pair_state.last_dynamic_frame = frame_index
                    pair_state.saw_hard_deceleration |= hard_deceleration
                    pair_state.saw_stop_transition |= stop_transition
                    pair_state.saw_dual_stop |= dual_stop
                    pair_state.saw_trajectory_deflection |= trajectory_deflection
                    pair_state.saw_target_impulse |= target_impulse

                has_recent_contact = self._is_recent(
                    pair_state.last_contact_frame, frame_index
                )
                has_recent_dynamic = self._is_recent(
                    pair_state.last_dynamic_frame, frame_index
                )
                has_recent_approach = self._is_approach_recent(
                    pair_state.last_approach_frame, frame_index
                )
                strong_overlap = pair_state.max_overlap_ratio >= max(
                    0.05, self.mask_overlap_ratio_threshold * 2.0
                )
                bridge_frames = self._window(self.crossing_dual_stop_bridge_frames)
                min_gap_frames = self._window(self.crossing_dual_stop_min_gap_frames)
                recent_observation_gap = (
                    pair_state.last_observation_gap_frame is not None
                    and 0
                    <= frame_index - pair_state.last_observation_gap_frame
                    <= bridge_frames
                    and pair_state.observation_gap_frames >= min_gap_frames
                )
                contact_age = self._contact_age(pair_state, frame_index)
                max_contact_age = self._window(
                    self.max_contact_candidate_age_frames
                )
                fresh_contact = (
                    max_contact_age == 0 or contact_age <= max_contact_age
                )
                confirmed_moving_reaction = (
                    pair_state.reaction_frames
                    >= self._window(self.impact_reaction_confirmation_frames)
                )
                late_stationary_impact = (
                    stationary_role is not None
                    and pair_state.sustained_approach
                    and confirmed_moving_reaction
                    and trajectory_deflection
                )
                reaction_is_valid = (
                    stationary_role is None or confirmed_moving_reaction
                )
                crossing_disruption = (
                    target_impulse or deflection_a or deflection_b
                )
                recent_crossing_approach = self._is_crossing_approach_recent(
                    pair_state.last_crossing_approach_frame, frame_index
                )
                crossing_context = (
                    crossing_trajectories or recent_crossing_approach
                )
                bridged_dual_stop = (
                    spatial.contact
                    and spatial.overlap_area > 0
                    and dual_stop
                    and had_motion
                    and recent_crossing_approach
                    and pair_state.last_crossing_approach_frame is not None
                    and pair_state.last_observation_gap_frame is not None
                    and pair_state.last_observation_gap_frame
                    > pair_state.last_crossing_approach_frame
                    and pair_state.observation_gap_frames >= min_gap_frames
                    and 0
                    <= frame_index - pair_state.last_observation_gap_frame
                    <= bridge_frames
                )
                bridged_strong_impact = (
                    spatial.contact
                    and spatial.overlap_area > 0
                    and strong_overlap
                    and had_motion
                    and pair_state.sustained_approach
                    and recent_observation_gap
                    and (
                        pair_state.saw_hard_deceleration
                        or crossing_disruption
                    )
                )
                crossing_evidence_is_valid = (
                    not crossing_context
                    or crossing_disruption
                    or bridged_dual_stop
                    or bridged_strong_impact
                )
                closing_speed_threshold = self._pair_speed_threshold(
                    self.min_closing_speed_px, vehicle_a, vehicle_b
                )
                high_confidence_crossing_impulse = (
                    crossing_trajectories
                    and target_impulse
                    and spatial.contact
                    and spatial.overlap_area
                    >= self._area_threshold(
                        self.mask_overlap_threshold, vehicle_a, vehicle_b
                    )
                    and closing_speed >= closing_speed_threshold
                )
                weak_contact_without_current_evidence = (
                    spatial.contact
                    and spatial.overlap_area == 0
                    and not dynamic_evidence
                    and closing_speed < closing_speed_threshold
                )
                confirmation_frames = self._window(
                    self.collision_confirmation_frames
                )
                emitted = False
                if (
                    has_recent_contact
                    and (
                        fresh_contact
                        or late_stationary_impact
                        or bridged_dual_stop
                        or bridged_strong_impact
                    )
                    and has_recent_dynamic
                    and had_motion
                    and reaction_is_valid
                    and crossing_evidence_is_valid
                    and not weak_contact_without_current_evidence
                    and (
                        has_recent_approach
                        or late_stationary_impact
                        or high_confidence_crossing_impulse
                        or bridged_dual_stop
                        or bridged_strong_impact
                        or (strong_overlap and pair_state.saw_hard_deceleration)
                    )
                ):
                    pair_state.status = "contact_candidate"
                    pair_state.saw_occlusion_bridge |= (
                        bridged_dual_stop or bridged_strong_impact
                    )
                    pair_state.candidate_frames += (
                        confirmation_frames
                        if (
                            high_confidence_crossing_impulse
                            or bridged_dual_stop
                            or bridged_strong_impact
                        )
                        else 1
                    )
                elif not has_recent_contact:
                    pair_state.reset_candidate(keep_temporal=True)
                else:
                    pair_state.candidate_frames = 0

                if pair_state.candidate_frames >= confirmation_frames:
                    ordered_a, ordered_b = self._ordered_states(vehicle_a, vehicle_b)
                    collisions.append(
                        self._build_event(
                            pair, pair_state, ordered_a, ordered_b, frame_index
                        )
                    )
                    emitted = True
                    pair_state.status = "cooldown"
                    pair_state.cooldown_until = frame_index + self._window(
                        self.collision_cooldown_frames
                    )
                    pair_state.candidate_frames = 0

                self._record_diagnostic(
                    frame_index=frame_index,
                    pair=pair,
                    pair_state=pair_state,
                    spatial=spatial,
                    closing_speed=closing_speed,
                    hard_deceleration=hard_deceleration,
                    dual_stop=dual_stop,
                    stop_transition=stop_transition,
                    had_motion=had_motion,
                    stationary_target=stationary_role is not None,
                    moving_vehicle_reaction=moving_vehicle_reaction,
                    trajectory_deflection=trajectory_deflection,
                    crossing_trajectories=crossing_trajectories,
                    target_impulse=target_impulse,
                    bridged_dual_stop=bridged_dual_stop,
                    bridged_strong_impact=bridged_strong_impact,
                    emitted=emitted,
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

    # ------------------------------------------------------------------
    # Normalizzazione delle soglie
    #
    # Le soglie di configurazione sono tarate su un veicolo di scala
    # `reference_scale_px` in un video a `reference_fps`. Qui vengono convertite
    # nelle unita effettive del veicolo e del video in esame. A condizioni di
    # riferimento tutti i fattori valgono 1 e i valori restano identici.
    # ------------------------------------------------------------------

    def _vehicle_scale_factor(self, vehicle: VehicleState) -> float:
        """Rapporto fra la scala apparente del veicolo e quella di riferimento."""
        if not self.kinematic_normalization_enabled:
            return 1.0
        scale = vehicle.scale_px
        if scale <= 0.0:
            scale = compute_bbox_scale_px(vehicle.bbox)
        if scale <= 0.0:
            return 1.0
        return scale / self.reference_scale_px

    def _pair_scale_factor(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> float:
        """Scala di riferimento della coppia: media delle due scale.

        La media e simmetrica rispetto all'ordine dei track, a differenza di
        min/max, e non fa dipendere l'esito dal veicolo che si e segmentato
        peggio in quel frame.
        """
        if not self.kinematic_normalization_enabled:
            return 1.0
        return (
            self._vehicle_scale_factor(vehicle_a)
            + self._vehicle_scale_factor(vehicle_b)
        ) / 2.0

    def _speed_threshold(self, base: float, vehicle: VehicleState) -> float:
        """Soglia di velocita: px/frame e proporzionale a scala / fps."""
        return base * self._vehicle_scale_factor(vehicle) * self._time_factor

    def _pair_speed_threshold(
        self,
        base: float,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> float:
        return (
            base * self._pair_scale_factor(vehicle_a, vehicle_b) * self._time_factor
        )

    def _acceleration_threshold(self, base: float, vehicle: VehicleState) -> float:
        """Soglia di accelerazione: px/frame^2 e proporzionale a scala / fps^2."""
        return (
            base
            * self._vehicle_scale_factor(vehicle)
            * self._time_factor
            * self._time_factor
        )

    def _distance_threshold(
        self,
        base: float,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState | None = None,
    ) -> float:
        """Soglia di lunghezza: i pixel scalano con la scala, non con gli fps."""
        if vehicle_b is None:
            return base * self._vehicle_scale_factor(vehicle_a)
        return base * self._pair_scale_factor(vehicle_a, vehicle_b)

    def _area_threshold(
        self,
        base: float,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> float:
        """Soglia di area: i pixel quadrati scalano con il quadrato della scala."""
        factor = self._pair_scale_factor(vehicle_a, vehicle_b)
        return base * factor * factor

    def _window(self, frames: int) -> int:
        """Converte una finestra tarata a `reference_fps` nel frame rate reale.

        Uno zero conserva il proprio significato di "controllo disattivato" e
        non viene mai trasformato in un frame.

        La conversione e disattivata di default: misurata sul dataset reale
        peggiora il rilevamento, perche moltiplica i frame di conferma richiesti
        proprio alle coppie che ne hanno meno a disposizione.
        """
        if frames <= 0 or not self.normalize_time_windows:
            return int(frames)
        if self._time_factor == 1.0:
            return int(frames)
        return max(1, int(round(frames / self._time_factor)))

    def _both_tracks_are_new(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> bool:
        """Vero solo se nessuno dei due track era gia noto da diversi frame.

        Una coppia nuova fra un track maturo e un ID appena creato non prova
        che i veicoli fossero gia accostati: nella maggior parte dei casi il
        tracker ha semplicemente riassegnato un ID mentre le sagome si
        occludevano, cioe proprio durante l'urto. Solo quando entrambi i track
        sono appena comparsi l'avvicinamento e realmente non osservabile.
        """
        limit = self._window(self.preexisting_contact_max_track_age_frames)
        return (
            vehicle_a.observed_frames <= limit
            and vehicle_b.observed_frames <= limit
        )

    def _preexisting_contact_is_released(self, pair_state: _PairState) -> bool:
        """Rilascia il disarmo per separazione stabile oppure per scadenza."""
        if pair_state.separation_frames >= self._window(
            self.preexisting_contact_release_frames
        ):
            return True
        expiry = self._window(self.preexisting_contact_max_frames)
        return expiry > 0 and pair_state.preexisting_frames >= expiry

    def _spatial_evidence(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> _SpatialEvidence:
        distance = bbox_distance(vehicle_a.bbox, vehicle_b.bbox)
        if distance > self._distance_threshold(
            self.contact_distance_threshold_px, vehicle_a, vehicle_b
        ):
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
            overlap_area
            >= self._area_threshold(
                self.mask_overlap_threshold, vehicle_a, vehicle_b
            )
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
        stopped_frames = self._window(self.stopped_frames_threshold)
        return (
            vehicle_a.kinematics_valid
            and vehicle_b.kinematics_valid
            and vehicle_a.stopped_frames >= stopped_frames
            and vehicle_b.stopped_frames >= stopped_frames
            and vehicle_a.speed_px
            <= self._speed_threshold(self.stopped_speed_threshold, vehicle_a)
            and vehicle_b.speed_px
            <= self._speed_threshold(self.stopped_speed_threshold, vehicle_b)
        )

    def _hard_deceleration(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> bool:
        return self._vehicle_hard_deceleration(
            vehicle_a
        ) or self._vehicle_hard_deceleration(vehicle_b)

    def _vehicle_hard_deceleration(self, vehicle: VehicleState) -> bool:
        return vehicle.kinematics_valid and (
            vehicle.acceleration_px
            <= self._acceleration_threshold(
                self.strong_deceleration_threshold, vehicle
            )
        )

    def _stationary_target_and_mover(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
        frame_index: int,
    ) -> tuple[VehicleState, VehicleState] | None:
        """Restituisce (veicolo in moto, bersaglio fermo) se i ruoli sono stabili."""
        a_stationary = self._is_stably_stationary(vehicle_a, frame_index)
        b_stationary = self._is_stably_stationary(vehicle_b, frame_index)
        a_moving = self._had_recent_motion(vehicle_a, frame_index)
        b_moving = self._had_recent_motion(vehicle_b, frame_index)
        if a_moving and b_stationary and not b_moving:
            return vehicle_a, vehicle_b
        if b_moving and a_stationary and not a_moving:
            return vehicle_b, vehicle_a
        return None

    def _is_stably_stationary(
        self,
        vehicle: VehicleState,
        frame_index: int,
    ) -> bool:
        history_frames = self._window(self.stationary_history_frames)
        samples = [
            sample
            for sample in vehicle.history
            if 0 <= frame_index - sample.frame_index < history_frames
        ]
        minimum_samples = max(3, int(math.ceil(history_frames * 0.6)))
        if len(samples) < minimum_samples:
            return False
        stopped_speed = self._speed_threshold(self.stopped_speed_threshold, vehicle)
        stationary_samples = sum(
            sample.speed_px <= stopped_speed for sample in samples
        )
        return stationary_samples / len(samples) >= self.stationary_history_ratio

    def _trajectory_deflection(
        self,
        vehicle: VehicleState,
        frame_index: int,
    ) -> bool:
        """Rileva una deviazione persistente rispetto alla traiettoria pre-impatto."""
        if not vehicle.kinematics_valid:
            return False
        speed = math.hypot(vehicle.velocity_x_px, vehicle.velocity_y_px)
        if speed < self._speed_threshold(self.min_preimpact_speed_px, vehicle):
            return False

        lag_frames = self._window(self.trajectory_reaction_lag_frames)
        history_frames = self._window(self.trajectory_history_frames)
        baseline_samples = [
            sample
            for sample in vehicle.history
            if lag_frames <= frame_index - sample.frame_index <= history_frames
        ]
        if len(baseline_samples) < 2:
            return False
        first = baseline_samples[0].motion_anchor
        last = baseline_samples[-1].motion_anchor
        baseline_x = float(last.x) - float(first.x)
        baseline_y = float(last.y) - float(first.y)
        displacement = math.hypot(baseline_x, baseline_y)
        if displacement < self._distance_threshold(
            self.trajectory_min_displacement_px, vehicle
        ):
            return False

        cosine = (
            baseline_x * vehicle.velocity_x_px
            + baseline_y * vehicle.velocity_y_px
        ) / (displacement * speed)
        cosine = min(1.0, max(-1.0, cosine))
        angle_deg = math.degrees(math.acos(cosine))
        return angle_deg >= self.trajectory_deflection_angle_deg

    def _crossing_trajectories(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
        frame_index: int,
    ) -> bool:
        """Distingue direttrici incrociate da moto parallelo o opposto."""
        direction_a = self._recent_trajectory_vector(vehicle_a, frame_index)
        direction_b = self._recent_trajectory_vector(vehicle_b, frame_index)
        if direction_a is None or direction_b is None:
            return False
        ax, ay = direction_a
        bx, by = direction_b
        norm_a = math.hypot(ax, ay)
        norm_b = math.hypot(bx, by)
        cosine = (ax * bx + ay * by) / (norm_a * norm_b)
        cosine = abs(min(1.0, max(-1.0, cosine)))
        return cosine <= math.cos(math.radians(self.crossing_min_angle_deg))

    def _recent_trajectory_vector(
        self,
        vehicle: VehicleState,
        frame_index: int,
    ) -> tuple[float, float] | None:
        history_frames = self._window(self.crossing_history_frames)
        samples = [
            sample
            for sample in vehicle.history
            if 0 <= frame_index - sample.frame_index < history_frames
        ]
        if len(samples) >= 2:
            first = samples[0].motion_anchor
            last = samples[-1].motion_anchor
            dx = float(last.x) - float(first.x)
            dy = float(last.y) - float(first.y)
            if math.hypot(dx, dy) >= self._distance_threshold(
                self.min_preimpact_speed_px, vehicle
            ):
                return dx, dy
        if vehicle.kinematics_valid and vehicle.speed_px >= self._speed_threshold(
            self.min_preimpact_speed_px, vehicle
        ):
            return vehicle.velocity_x_px, vehicle.velocity_y_px
        return None

    def _target_impulse(
        self,
        vehicle_a: VehicleState,
        vehicle_b: VehicleState,
    ) -> bool:
        """Rileva un trasferimento improvviso di moto sul veicolo piu lento."""
        return self._is_impulsed_target(vehicle_a, vehicle_b) or self._is_impulsed_target(
            vehicle_b, vehicle_a
        )

    def _is_impulsed_target(
        self,
        target: VehicleState,
        other: VehicleState,
    ) -> bool:
        if not target.kinematics_valid or not other.kinematics_valid:
            return False
        if target.acceleration_px < self._acceleration_threshold(
            self.target_impulse_acceleration_threshold, target
        ):
            return False
        other_reference_speed = max(other.prev_speed_px, other.speed_px)
        return (
            other_reference_speed
            >= self._speed_threshold(self.min_preimpact_speed_px, other)
            and target.prev_speed_px < other_reference_speed
        )

    def _had_recent_motion(self, vehicle: VehicleState, frame_index: int) -> bool:
        history_window = self._window(self.impact_window_frames) * 2
        return self._has_sustained_motion(
            vehicle,
            frame_index,
            history_window,
        )

    def _recent_stop_transition(self, vehicle: VehicleState, frame_index: int) -> bool:
        if not vehicle.kinematics_valid or vehicle.speed_px > self._speed_threshold(
            self.stopped_speed_threshold, vehicle
        ):
            return False
        return self._has_sustained_motion(
            vehicle,
            frame_index,
            self._window(self.impact_window_frames),
            exclude_current=True,
        )

    def _has_sustained_motion(
        self,
        vehicle: VehicleState,
        frame_index: int,
        window_frames: int,
        *,
        exclude_current: bool = False,
    ) -> bool:
        motion_speed = self._speed_threshold(self.min_preimpact_speed_px, vehicle)
        required = self._window(self.motion_confirmation_frames)
        consecutive = 0
        for sample in vehicle.history:
            age = frame_index - sample.frame_index
            in_window = 0 <= age <= window_frames
            if exclude_current and age == 0:
                in_window = False
            if in_window and sample.speed_px >= motion_speed:
                consecutive += 1
                if consecutive >= required:
                    return True
            else:
                consecutive = 0
        return False

    def _record_diagnostic(
        self,
        *,
        frame_index: int,
        pair: tuple[int, int],
        pair_state: _PairState,
        spatial: _SpatialEvidence,
        closing_speed: float,
        hard_deceleration: bool,
        dual_stop: bool,
        stop_transition: bool,
        had_motion: bool,
        stationary_target: bool,
        moving_vehicle_reaction: bool,
        trajectory_deflection: bool,
        crossing_trajectories: bool,
        target_impulse: bool,
        bridged_dual_stop: bool,
        bridged_strong_impact: bool,
        emitted: bool,
    ) -> None:
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
                preexisting_contact=pair_state.preexisting_contact,
                separation_frames=pair_state.separation_frames,
                approach_frames=pair_state.approach_frames,
                sustained_approach=pair_state.sustained_approach,
                stationary_target=stationary_target,
                moving_vehicle_reaction=moving_vehicle_reaction,
                trajectory_deflection=trajectory_deflection,
                crossing_trajectories=crossing_trajectories,
                target_impulse=target_impulse,
                observation_gap_frames=pair_state.observation_gap_frames,
                bridged_dual_stop=bridged_dual_stop,
                bridged_strong_impact=bridged_strong_impact,
                reaction_frames=pair_state.reaction_frames,
                contact_age_frames=self._contact_age(pair_state, frame_index),
                candidate_frames=pair_state.candidate_frames,
                pair_status=pair_state.status,
                emitted=emitted,
            )
        )

    def _is_recent(self, evidence_frame: int | None, frame_index: int) -> bool:
        return evidence_frame is not None and (
            0
            <= frame_index - evidence_frame
            <= self._window(self.impact_window_frames)
        )

    def _is_approach_recent(
        self,
        evidence_frame: int | None,
        frame_index: int,
    ) -> bool:
        return evidence_frame is not None and (
            0
            <= frame_index - evidence_frame
            <= self._window(self.approach_evidence_window_frames)
        )

    def _is_crossing_approach_recent(
        self,
        evidence_frame: int | None,
        frame_index: int,
    ) -> bool:
        return evidence_frame is not None and (
            0
            <= frame_index - evidence_frame
            <= self._window(self.crossing_dual_stop_bridge_frames)
        )

    @staticmethod
    def _contact_age(pair_state: _PairState, frame_index: int) -> int:
        if pair_state.first_contact_frame is None:
            return 0
        return max(0, frame_index - pair_state.first_contact_frame)

    def _expire_temporal_evidence(
        self,
        pair_state: _PairState,
        frame_index: int,
    ) -> None:
        contact_is_recent = self._is_recent(
            pair_state.last_contact_frame, frame_index
        )
        if (
            pair_state.last_contact_frame is not None
            and not contact_is_recent
        ):
            pair_state.sustained_approach = False
            pair_state.reaction_frames = 0
        if not self._is_recent(pair_state.last_dynamic_frame, frame_index):
            pair_state.last_dynamic_frame = None
            pair_state.saw_hard_deceleration = False
            pair_state.saw_dual_stop = False
            pair_state.saw_stop_transition = False
            pair_state.saw_trajectory_deflection = False
            pair_state.saw_target_impulse = False
        if not self._is_approach_recent(
            pair_state.last_approach_frame, frame_index
        ):
            pair_state.last_approach_frame = None
            pair_state.max_closing_speed_px = 0.0
            if not contact_is_recent:
                pair_state.sustained_approach = False
        if not self._is_crossing_approach_recent(
            pair_state.last_crossing_approach_frame, frame_index
        ):
            pair_state.last_crossing_approach_frame = None
        if (
            pair_state.last_observation_gap_frame is not None
            and frame_index - pair_state.last_observation_gap_frame
            > self._window(self.crossing_dual_stop_bridge_frames)
        ):
            pair_state.last_observation_gap_frame = None
            pair_state.observation_gap_frames = 0

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
        if pair_state.saw_trajectory_deflection:
            reasons.append("trajectory_deflection")
        if pair_state.saw_target_impulse:
            reasons.append("target_impulse")
        if pair_state.saw_occlusion_bridge:
            reasons.append("occlusion_bridge")

        confidence = 0.35 + min(0.2, pair_state.max_overlap_ratio)
        confidence += 0.2 if pair_state.saw_hard_deceleration else 0.0
        confidence += 0.15 if pair_state.saw_stop_transition else 0.0
        confidence += 0.1 if pair_state.max_closing_speed_px > 0.0 else 0.0
        confidence += 0.1 if pair_state.saw_trajectory_deflection else 0.0
        confidence += 0.15 if pair_state.saw_target_impulse else 0.0
        confidence += 0.1 if pair_state.saw_occlusion_bridge else 0.0

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
        ttl = self._window(self.pair_state_ttl_frames)
        stale = [
            pair
            for pair, state in self._pair_states.items()
            if frame_index - state.last_seen_frame > ttl
        ]
        for pair in stale:
            del self._pair_states[pair]

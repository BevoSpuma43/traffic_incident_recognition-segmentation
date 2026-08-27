"""
src/video_pipeline.py
=====================
Pipeline principale per il processing frame-by-frame del video.

Coordina i quattro componenti del sistema in sequenza, per ogni frame:
  1. VehicleSegmenter   -> rilevamento e tracking YOLO + ByteTrack
  2. VehicleStateStore  -> aggiornamento cinematica e pulizia track stale
  3. CollisionDetector  -> rilevamento collisioni tra coppie di veicoli
  4. render_frame       -> composizione del frame annotato

Dipendenze interne (bottom-up):
  src.config          -> AppConfig
  src.models          -> CollisionEvent
  src.detector        -> VehicleSegmenter
  src.tracker_state   -> VehicleStateStore
  src.collision_logic -> CollisionDetector
  src.renderer        -> render_frame
"""

from __future__ import annotations

import numpy as np

from src.calibration import CalibrationRecorder
from src.collision_logic import CollisionDetector
from src.config import AppConfig
from src.detector import VehicleSegmenter
from src.models import CollisionEvent
from src.renderer import render_frame
from src.tracker_state import VehicleStateStore


class TrafficAccidentPipeline:
    """
    Orchestratore della pipeline di rilevamento incidenti stradali.

    Inizializza e mantiene tutti i componenti del sistema (segmenter,
    state store, collision detector) e coordina il processing di ogni
    frame video in ordine deterministico.
    """

    def __init__(self, config: AppConfig) -> None:
        """
        Inizializza tutti i componenti della pipeline dalla configurazione.

        Parameters
        ----------
        config : AppConfig
            Configurazione dell'applicazione con tutti i parametri necessari
            per i sottocomponenti (path modello, soglie, parametri di tracking).
        """
        self.config = config

        # Segmenter YOLO + ByteTrack: rilevamento e tracking per ogni frame.
        self.segmenter: VehicleSegmenter = VehicleSegmenter(config)

        # Store degli stati cinematici: mantiene il dizionario {track_id: VehicleState}.
        self.state_store: VehicleStateStore = VehicleStateStore(config)

        # Rilevatore di collisioni: usa from_config per evitare parametri posizionali.
        # Viene creato UNA SOLA VOLTA e riutilizzato per ogni frame (era un bug
        # nella versione originale dove veniva ricreato dentro process_frame).
        self.collision_detector: CollisionDetector = CollisionDetector.from_config(config)

        # Gli eventi restano unici per logging, ma l'alert visuale persiste
        # abbastanza a lungo da essere visibile durante la riproduzione.
        self._visible_collisions: dict[tuple[int, int], tuple[CollisionEvent, int]] = {}
        calibration_path = str(getattr(config, "calibration_log_path", "")).strip()
        self._calibration_recorder: CalibrationRecorder | None = None
        if calibration_path:
            self._calibration_recorder = CalibrationRecorder(
                calibration_path,
                video_path=config.video_path,
                config=config,
            )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        *,
        render: bool = True,
        timestamp_s: float | None = None,
    ) -> tuple[np.ndarray, list[CollisionEvent]]:
        """
        Processa un singolo frame video e restituisce il frame annotato.

        Esegue la pipeline completa in 4 step:
          1. Segmentazione e tracking YOLO (produce DetectionResult)
          2. Aggiornamento stato cinematico e pulizia track stale
          3. Rilevamento collisioni tra tutte le coppie di veicoli attivi
          4. Rendering del frame annotato con maschere, ID e alert collisioni

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR originale (H x W x 3), come restituito da cv2.VideoCapture.
        frame_index : int
            Indice progressivo del frame, usato per la gestione dei track stale
            e per il logging degli eventi di collisione.

        Returns
        -------
        tuple[np.ndarray, list[CollisionEvent]]
            - Frame BGR annotato con tutte le sovrapposizioni (maschere,
              centroidi, ID, rettangoli CRASH).
            - Lista degli eventi di collisione rilevati nel frame corrente
              (lista vuota se nessuna collisione).
        """
        # --- Step 1: Segmentazione e tracking ---
        # VehicleSegmenter chiama YOLO.track() e normalizza l'output in
        # DetectionResult (poligono, maschera, bbox, track_id, class).
        detections = self.segmenter.segment_and_track(frame=frame)

        # --- Step 2: Aggiornamento stato cinematico ---
        # Calcola velocita, accelerazione e contatore stop per ogni track.
        # I track non presenti in questa detection vengono marcati assenti.
        active_states = self.state_store.update(
            detections=detections,
            frame_index=frame_index,
        )

        # Rimuove i track non rilevati da piu di max_missing_frames frame.
        # Fatto DOPO update() per non perdere l'ultimo stato cinematico.
        self.state_store.remove_stale_tracks(frame_index=frame_index)

        # --- Step 3: Rilevamento collisioni ---
        # Analizza tutte le coppie di veicoli attivi: overlap pixel-level
        # + anomalia cinematica (dual stop o hard deceleration).
        # NOTA: self.collision_detector e creato in __init__ e riutilizzato,
        # NON ricreato ad ogni frame come nel bug originale.
        collisions: list[CollisionEvent] = self.collision_detector.detect_collisions(
            vehicle_states=active_states,
            frame_index=frame_index,
        )
        if self._calibration_recorder is not None:
            self._calibration_recorder.record_frame(
                frame_index,
                self.collision_detector.last_diagnostics,
                collisions,
                timestamp_s=timestamp_s,
                detection_count=len(detections),
                active_track_count=len(active_states),
            )

        display_frames = max(
            1, int(getattr(self.config, "collision_display_frames", 15))
        )
        for event in collisions:
            pair = (event.track_id_a, event.track_id_b)
            self._visible_collisions[pair] = (event, frame_index + display_frames)
        expired_pairs = [
            pair
            for pair, (_, expires_at) in self._visible_collisions.items()
            if frame_index >= expires_at
        ]
        for pair in expired_pairs:
            del self._visible_collisions[pair]
        visible_collisions = [
            event for event, _ in self._visible_collisions.values()
        ]

        # --- Step 4: Rendering ---
        annotated = (
            self._render(frame, active_states, visible_collisions)
            if render
            else frame
        )

        return annotated, collisions

    def close(self) -> None:
        """Chiude l'eventuale log di calibrazione."""
        if self._calibration_recorder is not None:
            self._calibration_recorder.close()
            self._calibration_recorder = None

    def _render(
        self,
        frame: np.ndarray,
        active_states: list,
        collisions: list[CollisionEvent],
    ) -> np.ndarray:
        """
        Compone il frame annotato chiamando render_frame con i parametri corretti.

        Legge le opzioni di visualizzazione dalla configurazione (alpha_overlay,
        draw_ids, draw_bbox) e le passa direttamente a render_frame.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR originale.
        active_states : list
            Lista degli stati cinematici attivi (VehicleState).
        collisions : list[CollisionEvent]
            Collisioni rilevate nel frame corrente.

        Returns
        -------
        np.ndarray
            Frame annotato con tutte le sovrapposizioni.
        """
        return render_frame(
            frame=frame,
            vehicle_states=active_states,
            collisions=collisions,
            # alpha_overlay e il quarto parametro OBBLIGATORIO di render_frame.
            # Nella versione originale veniva passato frame_index (int) al suo
            # posto, causando un valore clippato a 1.0 (opaco) dopo il frame 1.
            alpha_overlay=float(getattr(self.config, "alpha_overlay", 0.4)),
            draw_masks=bool(getattr(self.config, "draw_masks", True)),
            draw_ids=bool(getattr(self.config, "draw_ids", True)),
            draw_bbox=bool(getattr(self.config, "draw_bbox", False)),
        )

"""
src/tracker_state.py
====================
Gestione dello store dei track attivi per il progetto traffic-accident-seg.

Questo modulo mantiene un dizionario aggiornato degli stati dinamici di tutti
i veicoli attualmente tracciati. Ad ogni frame:
  1. aggiorna i track già noti con le nuove detection (cinematica + maschera)
  2. crea nuovi stati per i track appena apparsi
  3. rimuove i track "stale" (non rilevati per troppi frame consecutivi)

Dipendenze interne (bottom-up):
  src.config   → AppConfig (parametri di soglia)
  src.models   → Point2D, DetectionResult, VehicleState
  src.geometry → compute_centroid_from_polygon
  src.kinematics → compute_speed_px, compute_acceleration, update_stopped_counter
"""

from __future__ import annotations

from collections import deque

from src.config import AppConfig
from src.geometry import compute_centroid_from_polygon
from src.kinematics import (
    compute_acceleration,
    compute_bbox_scale_px,
    compute_delta_v_px,
    compute_velocity_px,
    exponential_moving_average,
    update_stopped_counter,
)
from src.models import DetectionResult, Point2D, TrackSample, VehicleState


class VehicleStateStore:
    """
    Store aggiornabile degli stati cinematici dei veicoli tracciati.

    Mantiene internamente un dizionario ``{track_id: VehicleState}`` e
    fornisce metodi per aggiornarlo ad ogni frame, rimuovere i track morti
    e leggere l'elenco degli stati attivi.
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        """
        Inizializza lo store con la configurazione dell'applicazione.

        Parameters
        ----------
        config : AppConfig | None
            Configurazione dell'applicazione. Se None vengono usati i
            valori di default definiti direttamente in AppConfig.
        """
        self._config = config

        # Dizionario principale: track_id → stato cinematico corrente.
        self._states: dict[int, VehicleState] = {}

        # Soglia di pulizia: dopo quanti frame di assenza rimuovere un track.
        self._max_missing_frames: int = int(
            getattr(config, "max_missing_frames", 10)
        )
        self._history_size = max(2, int(getattr(config, "motion_history_size", 15)))
        self._velocity_alpha = float(getattr(config, "velocity_ema_alpha", 0.45))
        self._max_kinematic_gap = max(
            1, int(getattr(config, "max_kinematic_gap_frames", 2))
        )
        self._scale_alpha = float(getattr(config, "scale_ema_alpha", 0.25))
        # Con la normalizzazione attiva anche il contatore di stop deve usare
        # una soglia proporzionale alla scala: un veicolo lontano non puo
        # essere dichiarato fermo solo perche si sposta di pochi pixel.
        self._normalize = bool(
            getattr(config, "kinematic_normalization_enabled", False)
        )
        self._reference_scale_px = max(
            1e-6, float(getattr(config, "reference_scale_px", 90.0))
        )
        self._reference_fps = max(1e-6, float(getattr(config, "reference_fps", 15.0)))
        self._video_fps = max(0.0, float(getattr(config, "video_fps", 0.0)))
        # Re-identificazione dopo un cambio di ID (punto 2b).
        self._reid_enabled = bool(getattr(config, "track_reid_enabled", False))
        self._reid_max_gap = max(
            1, int(getattr(config, "track_reid_max_gap_frames", 5))
        )
        self._reid_max_distance_scale = max(
            0.0, float(getattr(config, "track_reid_max_distance_scale", 0.8))
        )
        self._reid_max_scale_ratio = max(
            1.0, float(getattr(config, "track_reid_max_scale_ratio", 2.0))
        )

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def update(
        self,
        detections: list[DetectionResult],
        frame_index: int,
        stopped_speed_threshold: float | None = None,
    ) -> list[VehicleState]:
        """
        Aggiorna lo store con le detection del frame corrente.

        Per ogni detection con track_id valido:
          - se il track esiste già → aggiorna cinematica e maschera
          - se è nuovo → crea un nuovo VehicleState con valori iniziali

        Parameters
        ----------
        detections : list[DetectionResult]
            Output di VehicleSegmenter.segment_and_track() per il frame corrente.
        frame_index : int
            Indice progressivo del frame (usato per gestire i track stale).
        stopped_speed_threshold : float | None
            Soglia di velocità (pixel/frame) al di sotto della quale un
            veicolo è considerato fermo. Se None viene letto da AppConfig.

        Returns
        -------
        list[VehicleState]
            Lista degli stati aggiornati in questo frame (solo i track
            presenti nelle detection correnti, non tutti quelli in store).
        """
        # Lettura della soglia dalla configurazione se non passata esplicitamente.
        if stopped_speed_threshold is None:
            stopped_speed_threshold = float(
                getattr(self._config, "stopped_speed_threshold", 2.5)
            )

        updated_states: list[VehicleState] = []

        detections_by_id: dict[int, DetectionResult] = {}
        for detection in detections:
            if detection.track_id is None:
                continue
            previous = detections_by_id.get(detection.track_id)
            if previous is None or detection.confidence > previous.confidence:
                detections_by_id[detection.track_id] = detection

        # Un track presente in questo frame non puo essere donatore di storia:
        # il suo ID non e sparito, quindi non c'e nessuna riassegnazione.
        active_ids = set(detections_by_id)
        inherited_ids: set[int] = set()

        for detection in detections_by_id.values():
            # I track senza ID assegnato da ByteTrack (es. primo frame
            # o detection momentaneamente non associate) vengono scartati.
            track_id = detection.track_id
            if track_id is None:
                continue

            # Calcola il centroide del veicolo come Point2D.
            # Tenta prima dai momenti del poligono, poi dalla bbox come fallback.
            centroid: Point2D = self._compute_detection_centroid(detection)
            motion_anchor = self._compute_motion_anchor(detection, centroid)

            if track_id not in self._states:
                donor_id = self._find_reid_donor(
                    detection=detection,
                    motion_anchor=motion_anchor,
                    frame_index=frame_index,
                    active_ids=active_ids,
                    inherited_ids=inherited_ids,
                )
                if donor_id is not None:
                    inherited_ids.add(donor_id)
                    self._inherit_track_history(donor_id, track_id)

            if track_id in self._states:
                # --- Aggiornamento di un track già noto ---
                self._update_existing_state(
                    track_id=track_id,
                    detection=detection,
                    centroid=centroid,
                    motion_anchor=motion_anchor,
                    frame_index=frame_index,
                    stopped_speed_threshold=stopped_speed_threshold,
                )
                updated_states.append(self._states[track_id])
            else:
                # --- Creazione di un nuovo track ---
                new_state = self._build_new_state(
                    detection=detection,
                    centroid=centroid,
                    motion_anchor=motion_anchor,
                    frame_index=frame_index,
                )
                self._states[track_id] = new_state
                updated_states.append(new_state)

        return updated_states

    def get_active_states(self) -> list[VehicleState]:
        """
        Restituisce tutti gli stati correnti nello store.

        Returns
        -------
        list[VehicleState]
            Snapshot della lista degli stati attivi. Non include i track
            rimossi da remove_stale_tracks().
        """
        return list(self._states.values())

    def remove_stale_tracks(self, frame_index: int) -> None:
        """
        Rimuove dallo store i track non rilevati per troppi frame consecutivi.

        Un track è considerato "stale" se non compare nelle detection da
        più di ``max_missing_frames`` frame. Questo evita l'accumulo di
        stati per veicoli che hanno lasciato il campo visivo.

        Parameters
        ----------
        frame_index : int
            Indice del frame corrente, confrontato con ``last_seen_frame``
            di ogni stato per calcolare i frame di assenza.
        """
        # Raccoglie gli ID da eliminare in una lista separata per evitare
        # la modifica del dizionario durante l'iterazione.
        stale_ids: list[int] = [
            track_id
            for track_id, state in self._states.items()
            if frame_index - state.last_seen_frame > self._max_missing_frames
        ]

        for track_id in stale_ids:
            del self._states[track_id]

    # ------------------------------------------------------------------
    # Metodi privati
    # ------------------------------------------------------------------

    def _update_existing_state(
        self,
        track_id: int,
        detection: DetectionResult,
        centroid: Point2D,
        motion_anchor: Point2D,
        frame_index: int,
        stopped_speed_threshold: float,
    ) -> None:
        """
        Aggiorna in-place lo stato di un track già presente nello store.

        Calcola la nuova velocità come distanza euclidea tra il centroide
        precedente e quello corrente, l'accelerazione come differenza di
        velocità, e aggiorna il contatore di stop.

        Parameters
        ----------
        track_id : int
            ID del track da aggiornare.
        detection : DetectionResult
            Detection corrente per questo track.
        centroid : Point2D
            Centroide calcolato per il frame corrente.
        frame_index : int
            Indice del frame corrente.
        stopped_speed_threshold : float
            Soglia di velocità per il contatore di stop.
        """
        state = self._states[track_id]

        # Salva i valori del frame precedente prima di sovrascriverli.
        previous_centroid: Point2D | None = state.centroid
        previous_anchor: Point2D | None = state.motion_anchor or state.centroid
        previous_speed_px: float = state.speed_px
        # Letto prima di sovrascriverlo: un delta-V ha senso solo fra due frame
        # entrambi misurati, altrimenti il primo campione varrebbe l'intera
        # velocita e ogni track nuovo sembrerebbe appena stato urtato.
        previous_kinematics_valid: bool = state.kinematics_valid
        frame_delta = frame_index - state.last_seen_frame
        kinematics_valid = 0 < frame_delta <= self._max_kinematic_gap

        if kinematics_valid:
            raw_vx, raw_vy = compute_velocity_px(
                previous_anchor, motion_anchor, frame_delta
            )
            if state.observed_frames > 1:
                velocity_x = exponential_moving_average(
                    state.velocity_x_px, raw_vx, self._velocity_alpha
                )
                velocity_y = exponential_moving_average(
                    state.velocity_y_px, raw_vy, self._velocity_alpha
                )
            else:
                velocity_x, velocity_y = raw_vx, raw_vy
            speed_px = (velocity_x ** 2 + velocity_y ** 2) ** 0.5
            acceleration_px = compute_acceleration(
                speed_px, previous_speed_px, frame_delta
            )
            # Il delta-V usa la velocita grezza: l'EMA e tarato per stabilizzare
            # la traiettoria e smorzerebbe proprio il picco da rilevare.
            delta_v_px = (
                compute_delta_v_px(
                    (state.raw_velocity_x_px, state.raw_velocity_y_px),
                    (raw_vx, raw_vy),
                    frame_delta,
                )
                if previous_kinematics_valid
                else 0.0
            )
        else:
            raw_vx = 0.0
            raw_vy = 0.0
            velocity_x = 0.0
            velocity_y = 0.0
            speed_px = 0.0
            acceleration_px = 0.0
            delta_v_px = 0.0

        scale_px = self._smoothed_scale(detection, state.scale_px)

        if kinematics_valid and frame_delta == 1:
            stopped_frames = update_stopped_counter(
                prev_counter=state.stopped_frames,
                speed_px=speed_px,
                stopped_speed_threshold=self._effective_stopped_threshold(
                    stopped_speed_threshold, scale_px
                ),
            )
        else:
            stopped_frames = 0

        # Aggiornamento in-place dei campi dello stato.
        # VehicleState ha slots=True: setattr funziona solo per campi esistenti.
        state.prev_centroid = previous_centroid
        state.centroid = centroid
        state.prev_motion_anchor = previous_anchor
        state.motion_anchor = motion_anchor
        state.prev_speed_px = previous_speed_px
        state.speed_px = speed_px
        state.acceleration_px = acceleration_px
        state.velocity_x_px = velocity_x
        state.velocity_y_px = velocity_y
        state.raw_velocity_x_px = raw_vx
        state.raw_velocity_y_px = raw_vy
        state.delta_v_px = delta_v_px
        state.kinematics_valid = kinematics_valid
        state.observed_frames += 1
        state.confidence = detection.confidence
        state.class_id = detection.class_id
        state.class_name = detection.class_name
        state.polygon = detection.polygon
        state.mask = detection.mask
        state.bbox = detection.bbox
        state.last_seen_frame = frame_index
        state.stopped_frames = stopped_frames
        state.scale_px = scale_px
        state.history.append(self._build_sample(state, frame_index))

    def _find_reid_donor(
        self,
        *,
        detection: DetectionResult,
        motion_anchor: Point2D,
        frame_index: int,
        active_ids: set[int],
        inherited_ids: set[int],
    ) -> int | None:
        """Cerca il track sparito da cui questo nuovo ID puo ereditare la storia.

        Quando due sagome si occludono durante un urto ByteTrack perde
        l'associazione e riassegna un ID nuovo: il veicolo e lo stesso, ma per
        la pipeline nasce in quel momento e tutta la cinematica pre-impatto
        sparisce proprio nel frame in cui servirebbe. Qui l'ID nuovo viene
        ricollegato al track sparito piu compatibile, se ne esiste uno solo
        plausibile per posizione prevista e dimensione.

        Returns
        -------
        int | None
            ID del track donatore, oppure None se nessun candidato e
            sufficientemente vicino.
        """
        if not self._reid_enabled:
            return None
        scale_px = compute_bbox_scale_px(detection.bbox)
        if scale_px <= 0.0:
            return None

        best_id: int | None = None
        best_distance = float("inf")
        for candidate_id, state in self._states.items():
            if candidate_id in active_ids or candidate_id in inherited_ids:
                continue
            gap_frames = frame_index - state.last_seen_frame
            if gap_frames <= 0 or gap_frames > self._reid_max_gap:
                continue
            anchor = state.motion_anchor or state.centroid
            if anchor is None:
                continue
            # Posizione attesa se il veicolo avesse proseguito col suo moto:
            # durante un urto la velocita cambia, ma su 1-5 frame l'estrapolazione
            # resta molto piu vicina del salto che separa due veicoli distinti.
            predicted_x = float(anchor.x) + state.velocity_x_px * gap_frames
            predicted_y = float(anchor.y) + state.velocity_y_px * gap_frames
            distance = (
                (float(motion_anchor.x) - predicted_x) ** 2
                + (float(motion_anchor.y) - predicted_y) ** 2
            ) ** 0.5
            reference_scale = max(scale_px, state.scale_px)
            if distance > self._reid_max_distance_scale * reference_scale:
                continue
            if state.scale_px > 0.0:
                ratio = max(
                    scale_px / state.scale_px, state.scale_px / scale_px
                )
                if ratio > self._reid_max_scale_ratio:
                    continue
            if distance < best_distance:
                best_id = candidate_id
                best_distance = distance
        return best_id

    def _inherit_track_history(self, donor_id: int, track_id: int) -> None:
        """Sposta lo stato del track sparito sotto il nuovo ID.

        Riusare lo stato invece di copiarne i campi fa si che il frame
        successivo passi dal normale ``_update_existing_state``: velocita,
        accelerazione e validita cinematica vengono calcolate sullo
        spostamento reale attraverso il cambio di ID, non ricostruite a mano.
        """
        state = self._states.pop(donor_id)
        state.track_id = track_id
        self._states[track_id] = state

    def _build_new_state(
        self,
        detection: DetectionResult,
        centroid: Point2D,
        motion_anchor: Point2D,
        frame_index: int,
    ) -> VehicleState:
        """
        Crea un nuovo VehicleState con valori iniziali per un track appena apparso.

        Al primo frame di un track la velocità e l'accelerazione sono 0.0.
        Il centroide precedente è inizializzato uguale a quello corrente
        (nessuno spostamento misurabile al primo frame).

        Parameters
        ----------
        detection : DetectionResult
            Detection dal quale leggere track_id, class_id, class_name,
            polygon, mask e bbox.
        centroid : Point2D
            Centroide calcolato per questo primo frame del track.
        frame_index : int
            Indice del frame corrente (usato come last_seen_frame iniziale).
        Returns
        -------
        VehicleState
            Nuovo stato cinematico con tutti i campi popolati.
        """
        state = VehicleState(
            track_id=detection.track_id,          # int (già validato non-None)
            class_id=detection.class_id,
            class_name=detection.class_name,
            centroid=centroid,
            prev_centroid=centroid,                # uguale al corrente: nessuno spostamento misurabile
            speed_px=0.0,
            prev_speed_px=0.0,
            acceleration_px=0.0,
            polygon=detection.polygon,
            mask=detection.mask,
            bbox=detection.bbox,
            last_seen_frame=frame_index,
            stopped_frames=0,
            motion_anchor=motion_anchor,
            prev_motion_anchor=motion_anchor,
            kinematics_valid=False,
            observed_frames=1,
            confidence=detection.confidence,
            scale_px=compute_bbox_scale_px(detection.bbox),
            history=deque(maxlen=self._history_size),
        )
        state.history.append(self._build_sample(state, frame_index))
        return state

    def _smoothed_scale(
        self,
        detection: DetectionResult,
        previous_scale_px: float,
    ) -> float:
        """Filtra la scala apparente per non far oscillare le soglie."""
        measured = compute_bbox_scale_px(detection.bbox)
        if measured <= 0.0:
            return previous_scale_px
        if previous_scale_px <= 0.0:
            return measured
        return exponential_moving_average(
            previous_scale_px, measured, self._scale_alpha
        )

    def _effective_stopped_threshold(
        self,
        stopped_speed_threshold: float,
        scale_px: float,
    ) -> float:
        """Adegua la soglia di arresto alla scala apparente e al frame rate.

        Una velocita in pixel/frame e proporzionale a ``scala / fps``: senza
        questa correzione un veicolo lontano risulta sempre fermo e uno vicino
        non lo risulta mai.
        """
        if not self._normalize or scale_px <= 0.0:
            return stopped_speed_threshold
        factor = scale_px / self._reference_scale_px
        if self._video_fps > 0.0:
            factor *= self._reference_fps / self._video_fps
        return stopped_speed_threshold * factor

    @staticmethod
    def _compute_motion_anchor(
        detection: DetectionResult,
        fallback: Point2D,
    ) -> Point2D:
        """Usa il punto inferiore centrale della bbox come ancora di moto."""
        bbox = detection.bbox
        if bbox is None or len(bbox) < 4:
            return fallback
        x1, _, x2, y2 = bbox[:4]
        return Point2D(x=(float(x1) + float(x2)) / 2.0, y=float(y2))

    @staticmethod
    def _build_sample(state: VehicleState, frame_index: int) -> TrackSample:
        centroid = state.centroid or Point2D(0.0, 0.0)
        anchor = state.motion_anchor or centroid
        mask_area = 0
        if state.mask is not None:
            mask_area = int((state.mask > 0).sum())
        return TrackSample(
            frame_index=frame_index,
            centroid=centroid,
            motion_anchor=anchor,
            speed_px=state.speed_px,
            velocity_x_px=state.velocity_x_px,
            velocity_y_px=state.velocity_y_px,
            acceleration_px=state.acceleration_px,
            mask_area=mask_area,
            scale_px=state.scale_px,
            delta_v_px=state.delta_v_px,
        )

    def _compute_detection_centroid(self, detection: DetectionResult) -> Point2D:
        """
        Calcola il centroide del veicolo e lo restituisce come Point2D.

        Strategia a due livelli:
          1. Prova a calcolare il centroide dal poligono di segmentazione
             tramite i momenti di immagine (più preciso, usa tutta la forma).
          2. Fallback: calcola il centro della bounding box se il poligono
             non è disponibile o è degenere.
          3. Ultimo fallback: origine (0, 0) se nessun dato è disponibile.

        Parameters
        ----------
        detection : DetectionResult
            Detection dalla quale leggere polygon e bbox.

        Returns
        -------
        Point2D
            Centroide come oggetto Point2D con attributi .x e .y (float).
            Compatibile con il Protocol _HasXY usato da compute_speed_px().
        """
        # Calcola il centroide dal poligono via momenti di immagine OpenCV.
        # compute_centroid_from_polygon() restituisce tuple[int,int] | None,
        # quindi avvolgiamo il risultato in Point2D per soddisfare _HasXY.
        centroid_tuple = compute_centroid_from_polygon(detection.polygon)

        if centroid_tuple is not None:
            cx, cy = centroid_tuple
            # Conversione esplicita da tuple a Point2D: le tuple non hanno
            # .x e .y, che sono richiesti dal Protocol _HasXY in kinematics.py.
            return Point2D(x=float(cx), y=float(cy))

        # Fallback: centro della bounding box (x1+x2)/2, (y1+y2)/2.
        bbox = detection.bbox
        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return Point2D(
                x=float(x1 + x2) / 2.0,
                y=float(y1 + y2) / 2.0,
            )

        # Ultimo fallback: origine. Non dovrebbe mai accadere in condizioni
        # normali, ma evita un crash se entrambe polygon e bbox sono None.
        return Point2D(x=0.0, y=0.0)

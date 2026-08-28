"""
src/models.py
=============
Dataclass di dominio per il progetto traffic-accident-seg.

Questo modulo definisce le strutture dati condivise dall'intera pipeline:
  - Point2D          → coordinata 2-D in spazio pixel
  - DetectionResult  → singolo output del detector YOLO
  - VehicleState     → stato cinematico di un veicolo tracciato
  - CollisionEvent   → evento di collisione rilevato tra due veicoli

Nessun import circolare: questo modulo non importa altri moduli interni.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Punto 2-D in spazio pixel
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Point2D:
    """
    Coordinata bidimensionale in spazio pixel.

    I campi sono float (e non int) per due motivi:
      1. tracker_state.py calcola il centroide della bbox come media
         di coordinate intere, ottenendo un float (es. (x1+x2)/2.0).
      2. Il Protocol _HasXY in kinematics.py dichiara x/y come float,
         quindi Point2D deve essere compatibile per duck-typing.
    """

    # Coordinata orizzontale (colonna), in pixel.
    x: float
    # Coordinata verticale (riga), in pixel.
    y: float


@dataclass(slots=True)
class TrackSample:
    """Campione cinematico osservato per un singolo track."""

    frame_index: int
    centroid: Point2D
    motion_anchor: Point2D
    speed_px: float
    velocity_x_px: float
    velocity_y_px: float
    acceleration_px: float
    mask_area: int


# ---------------------------------------------------------------------------
# Risultato di una singola detection YOLO + ByteTrack
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DetectionResult:
    """
    Output di una singola istanza rilevata dal modello YOLO segmenter.

    Viene prodotto da VehicleSegmenter.segment_and_track() in detector.py
    e consumato da VehicleStateStore.update() in tracker_state.py.
    """

    # ID univoco assegnato da ByteTrack a questo veicolo.
    # Può essere None se ByteTrack non ha ancora associato un track
    # (primo frame o track perso).
    track_id: int | None

    # Indice numerico della classe COCO (es. 2=car, 7=truck).
    class_id: int

    # Nome testuale della classe (es. "car"), derivato dal dict result.names.
    class_name: str

    # Score di confidenza della detection restituito da YOLO (0.0 – 1.0).
    # Campo aggiunto rispetto alla versione originale: detector.py lo passa
    # esplicitamente come keyword argument; senza questo campo si ottiene
    # un TypeError immediato.
    confidence: float

    # Poligono del contorno della maschera di segmentazione, in formato
    # OpenCV: array di shape (N, 1, 2), dtype int32, con N >= 3 vertici.
    polygon: np.ndarray

    # Maschera binaria uint8 (0/255) delle stesse dimensioni del frame.
    # Generata da polygon_to_binary_mask() in geometry.py.
    mask: np.ndarray

    # Bounding box (x_min, y_min, x_max, y_max) in pixel interi,
    # derivata dal poligono tramite compute_bbox_from_polygon() in geometry.py.
    bbox: tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Stato cinematico di un veicolo tracciato
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VehicleState:
    """
    Stato dinamico aggiornato frame-per-frame per un veicolo tracciato.

    Viene mantenuto in VehicleStateStore (tracker_state.py) e aggiornato
    ad ogni chiamata a update(). I campi cinematici (speed_px,
    acceleration_px) sono calcolati da kinematics.py.
    """

    # ID univoco del track, assegnato da ByteTrack.
    track_id: int

    # Classe COCO del veicolo (es. 2=car, 5=bus).
    class_id: int

    # Nome testuale della classe.
    class_name: str

    # Centroide corrente del veicolo nello spazio pixel (Point2D).
    # È None solo prima della prima detection valida.
    centroid: Point2D | None

    # Centroide del frame precedente, usato per calcolare la velocità.
    prev_centroid: Point2D | None

    # Velocità corrente in pixel/frame (norma euclidea dello spostamento
    # del centroide tra due frame consecutivi).
    speed_px: float

    # Velocità del frame precedente, usata per calcolare l'accelerazione.
    prev_speed_px: float

    # Accelerazione in pixel/frame²: differenza tra speed_px e prev_speed_px.
    # Un valore fortemente negativo indica una frenata brusca.
    acceleration_px: float

    # Poligono della maschera corrente (formato OpenCV, shape (N,1,2)).
    polygon: np.ndarray | None

    # Maschera binaria uint8 corrente, usata per il calcolo dell'overlap
    # in collision_logic.py.
    mask: np.ndarray | None

    # Bounding box corrente (x_min, y_min, x_max, y_max).
    bbox: tuple[int, int, int, int] | None

    # Indice dell'ultimo frame in cui questo veicolo è stato rilevato.
    # Usato da remove_stale_tracks() per eliminare i track "morti".
    last_seen_frame: int

    # Contatore di frame consecutivi in cui il veicolo risultava fermo
    # (speed_px <= stopped_speed_threshold). Azzerato non appena il
    # veicolo torna in movimento.
    stopped_frames: int

    # Punto inferiore centrale della bbox, meno sensibile del centroide alle
    # occlusioni parziali della maschera.
    motion_anchor: Point2D | None = None
    prev_motion_anchor: Point2D | None = None
    velocity_x_px: float = 0.0
    velocity_y_px: float = 0.0
    kinematics_valid: bool = False
    observed_frames: int = 1
    confidence: float = 0.0
    history: deque[TrackSample] = field(default_factory=deque)


# ---------------------------------------------------------------------------
# Evento di collisione rilevato
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CollisionEvent:
    """
    Evento di possibile collisione tra due veicoli tracciati.

    Prodotto da CollisionDetector.detect_collisions() in collision_logic.py
    e consumato da renderer.py per evidenziare visivamente i veicoli
    coinvolti con bbox e label rossi.

    Nota sui nomi dei campi
    -----------------------
    I campi identificativi usano il prefisso ``track_id_`` (es. track_id_a,
    track_id_b) per coerenza con la nomenclatura di collision_logic.py e
    renderer.py, che accedono a questi attributi via getattr().

    I campi cinematici e spaziali rendono l'evento spiegabile e vengono
    popolati dal rilevatore al momento della conferma.
    """

    # Indice del frame in cui è stata rilevata la collisione.
    frame_index: int

    # ID del primo veicolo coinvolto (sempre il minore tra i due, per
    # garantire chiavi di coppia ordinate in collision_logic.py).
    track_id_a: int

    # ID del secondo veicolo coinvolto.
    track_id_b: int

    # Area di sovrapposizione in pixel tra le due maschere binarie.
    overlap_area: int

    # Stringa descrittiva del motivo del rilevamento. Valori possibili:
    #   "overlap_and_dual_stop"
    #   "overlap_and_hard_deceleration"
    #   "overlap_and_dual_stop_and_hard_deceleration"
    reason: str

    # Velocità del veicolo A al momento dell'evento (pixel/frame).
    # Default 0.0: non popolato dalla pipeline attuale, disponibile per
    # estensioni future di logging.
    speed_a: float = 0.0

    # Velocità del veicolo B al momento dell'evento (pixel/frame).
    speed_b: float = 0.0

    # Accelerazione del veicolo A al momento dell'evento (pixel/frame²).
    acceleration_a: float = 0.0

    # Accelerazione del veicolo B al momento dell'evento (pixel/frame²).
    acceleration_b: float = 0.0

    overlap_ratio: float = 0.0
    spatial_distance_px: float = 0.0
    closing_speed_px: float = 0.0
    confidence: float = 0.0
    first_contact_frame: int | None = None
    confirmation_frame: int | None = None


@dataclass(slots=True)
class PairDiagnostic:
    """Evidenze frame-per-frame usate per calibrare una coppia di track."""

    frame_index: int
    track_id_a: int
    track_id_b: int
    contact: bool
    overlap_area: int
    overlap_ratio: float
    spatial_distance_px: float
    closing_speed_px: float
    hard_deceleration: bool
    dual_stop: bool
    stop_transition: bool
    had_recent_motion: bool
    preexisting_contact: bool
    separation_frames: int
    approach_frames: int
    sustained_approach: bool
    stationary_target: bool
    moving_vehicle_reaction: bool
    trajectory_deflection: bool
    crossing_trajectories: bool
    target_impulse: bool
    observation_gap_frames: int
    bridged_dual_stop: bool
    bridged_strong_impact: bool
    reaction_frames: int
    contact_age_frames: int
    candidate_frames: int
    pair_status: str
    emitted: bool

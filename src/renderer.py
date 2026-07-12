"""
src/renderer.py
===============
Funzioni di rendering per annotare i frame video con maschere, centroidi,
bounding box ed evidenziazioni di eventi di collisione.

Questo modulo contiene solo logica di visualizzazione (OpenCV puro) e non
ha effetti collaterali sul sistema di tracking o rilevamento.

Funzioni esportate
------------------
render_frame            -- entry point principale: compone tutte le annotazioni
draw_mask_overlay       -- overlay colorato per le maschere di segmentazione
draw_centroids_and_ids  -- cerchi e ID dei track
draw_bboxes             -- rettangoli bounding box per ogni veicolo
draw_collisions         -- evidenziazione rossa dei veicoli in collisione
get_color_for_track     -- colore deterministico per track ID

Dipendenze interne:
  src.models -> CollisionEvent, VehicleState
"""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from src.models import CollisionEvent, VehicleState

# Colore BGR usato per evidenziare le collisioni (rosso pieno).
_CRASH_COLOR: tuple[int, int, int] = (0, 0, 255)

# Sentinel restituito da compute_bbox_from_polygon() per bbox degenere.
# Ridefinito qui per evitare import circolare con detector.py.
_DEGENERATE_BBOX: tuple[int, int, int, int] = (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Funzioni helper private (prefisso _)
# ---------------------------------------------------------------------------

def _extract_centroid(state: VehicleState) -> tuple[int, int] | None:
    """
    Estrae il centroide da uno stato veicolo come coordinate intere pixel.

    Tenta prima l'accesso diretto a Point2D (attributi .x e .y), poi
    i fallback per tuple/ndarray (compatibilita con codice legacy).

    Parameters
    ----------
    state : VehicleState
        Stato del veicolo da cui estrarre il centroide.

    Returns
    -------
    tuple[int, int] | None
        Coppia (cx, cy) arrotondata a pixel interi, oppure None
        se il centroide non e disponibile o non e convertibile.
    """
    centroid = state.centroid
    if centroid is None:
        return None

    # Caso principale (dopo fix tracker_state.py): centroid e sempre Point2D.
    x = getattr(centroid, "x", None)
    y = getattr(centroid, "y", None)
    if x is not None and y is not None:
        return int(round(float(x))), int(round(float(y)))

    # Fallback difensivi per compatibilita con formati legacy (tuple, ndarray).
    if isinstance(centroid, (tuple, list)) and len(centroid) >= 2:
        return int(round(float(centroid[0]))), int(round(float(centroid[1])))

    if isinstance(centroid, np.ndarray) and centroid.size >= 2:
        return int(round(float(centroid.flat[0]))), int(round(float(centroid.flat[1])))

    return None


def _extract_bbox(state: VehicleState) -> tuple[int, int, int, int] | None:
    """
    Estrae la bounding box da uno stato veicolo come coordinate intere pixel.

    Parameters
    ----------
    state : VehicleState
        Stato del veicolo da cui estrarre la bbox.

    Returns
    -------
    tuple[int, int, int, int] | None
        Tupla (x1, y1, x2, y2) arrotondata, oppure None se la bbox e
        assente o degenere (sentinel (0,0,0,0) da compute_bbox_from_polygon).
    """
    bbox = state.bbox
    if bbox is None:
        return None

    # Converti ndarray in lista prima di accedere agli elementi.
    if isinstance(bbox, np.ndarray):
        bbox = bbox.tolist()

    if isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        result = (
            int(round(float(x1))),
            int(round(float(y1))),
            int(round(float(x2))),
            int(round(float(y2))),
        )
        # Scarta il sentinel (0,0,0,0): indica un poligono degenere in
        # compute_bbox_from_polygon(), non una bbox reale.
        if result == _DEGENERATE_BBOX:
            return None
        return result

    return None


def _extract_mask(state: VehicleState) -> np.ndarray | None:
    """
    Estrae la maschera binaria da uno stato veicolo.

    Parameters
    ----------
    state : VehicleState
        Stato del veicolo da cui estrarre la maschera.

    Returns
    -------
    np.ndarray | None
        Array NumPy della maschera, oppure None se assente o vuota.
    """
    mask = state.mask
    if mask is None:
        return None

    # Coercizione a ndarray: la maschera dovrebbe gia esserlo, ma il renderer
    # deve essere robusto a tipi inaspettati per non crashare la UI.
    if not isinstance(mask, np.ndarray):
        try:
            mask = np.asarray(mask)
        except Exception:
            return None

    if mask.size == 0:
        return None

    return mask


def _build_state_map(vehicle_states: Iterable[VehicleState]) -> dict[int, VehicleState]:
    """
    Costruisce un dizionario {track_id: VehicleState} per accesso O(1).

    Usato da draw_collisions per trovare rapidamente lo stato del veicolo
    dato il track_id estratto dall'evento di collisione.

    Parameters
    ----------
    vehicle_states : Iterable[VehicleState]
        Lista o iterabile degli stati attivi nel frame corrente.

    Returns
    -------
    dict[int, VehicleState]
        Mappa track_id -> stato, con solo gli stati che hanno track_id valido.
    """
    state_map: dict[int, VehicleState] = {}
    for state in vehicle_states:
        track_id = state.track_id
        if track_id is None:
            continue
        state_map[int(track_id)] = state
    return state_map


def _extract_collision_track_ids(collision: CollisionEvent) -> list[int]:
    """
    Estrae i track ID dei veicoli coinvolti in un evento di collisione.

    Accede direttamente ai campi track_id_a e track_id_b definiti
    in CollisionEvent (dopo il fix di models.py).

    Parameters
    ----------
    collision : CollisionEvent
        Evento di collisione da cui estrarre gli ID.

    Returns
    -------
    list[int]
        Lista dei due track ID coinvolti nell'evento.
    """
    # Accesso diretto agli attributi canonici di CollisionEvent.
    # La versione originale tentava prima "track_ids" (attributo inesistente)
    # e poi ricadeva su un for-loop con quattro nomi alternativi: ora semplificato.
    return [collision.track_id_a, collision.track_id_b]


def _label_anchor(state: VehicleState) -> tuple[int, int]:
    """
    Sceglie un punto di ancoraggio stabile per il testo da disegnare sul veicolo.

    Preferisce il bordo superiore della bbox (visivamente meno sovrapposto
    alla maschera) e usa il centroide come fallback.

    Parameters
    ----------
    state : VehicleState
        Stato del veicolo per cui calcolare l'ancoraggio.

    Returns
    -------
    tuple[int, int]
        Coordinate (x, y) del punto in cui inizia il testo.
    """
    bbox = _extract_bbox(state)
    if bbox is not None:
        x1, y1, _, _ = bbox
        # Posiziona il testo 10 pixel sopra il bordo superiore della bbox.
        return x1, max(0, y1 - 10)

    centroid = _extract_centroid(state)
    if centroid is not None:
        cx, cy = centroid
        return cx + 6, max(0, cy - 6)

    # Fallback di sicurezza: angolo in alto a sinistra.
    return 10, 20


# ---------------------------------------------------------------------------
# Funzioni di colore
# ---------------------------------------------------------------------------

def get_color_for_track(track_id: int) -> tuple[int, int, int]:
    """
    Restituisce un colore BGR deterministico e stabile per un dato track ID.

    Usa un hash lineare congruente (LCG) per distribuire i colori in modo
    pseudo-casuale ma riproducibile: lo stesso track ID produce sempre lo
    stesso colore, anche tra frame diversi.

    Parameters
    ----------
    track_id : int
        ID del track da colorare.

    Returns
    -------
    tuple[int, int, int]
        Colore BGR con componenti nell'intervallo [64, 191].
        Il range evita colori troppo scuri (< 64) o troppo chiari (> 191).
    """
    # Hash LCG: moltiplicatore e incremento standard (stile glibc).
    seed = int(track_id) * 1103515245 + 12345
    blue  = 64 + (seed & 0x7F)           # bit 0-6
    green = 64 + ((seed >> 8) & 0x7F)    # bit 8-14
    red   = 64 + ((seed >> 16) & 0x7F)   # bit 16-22
    return int(blue), int(green), int(red)


# ---------------------------------------------------------------------------
# Funzioni di rendering pubbliche
# ---------------------------------------------------------------------------

def draw_mask_overlay(
    frame: np.ndarray,
    vehicle_states: Iterable[VehicleState],
    alpha_overlay: float,
) -> np.ndarray:
    """
    Disegna un overlay colorato semitrasparente per le maschere di segmentazione.

    Per ogni veicolo tracciato, riempie la regione della sua maschera con il
    colore deterministico del track ID, poi applica alpha blending con il frame
    originale per ottenere l'effetto di semitrasparenza.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR originale di input.
    vehicle_states : Iterable[VehicleState]
        Stati dei veicoli con maschere da disegnare.
    alpha_overlay : float
        Intensita dell'overlay (0.0 = invisibile, 1.0 = opaco).
        Valori tipici: 0.35-0.5 per buona visibilita del video sottostante.

    Returns
    -------
    np.ndarray
        Frame annotato con gli overlay delle maschere.
    """
    base_frame = frame.copy()   # copia originale per il blending finale
    overlay = frame.copy()      # copia su cui disegniamo le maschere

    for state in vehicle_states:
        mask = _extract_mask(state)
        if mask is None:
            continue

        # Se la maschera ha dimensioni diverse dal frame (es. resize del frame
        # tra detection e rendering), la adatta con INTER_NEAREST per
        # preservare i bordi binari della segmentazione.
        if mask.shape[:2] != base_frame.shape[:2]:
            try:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (base_frame.shape[1], base_frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            except Exception:
                continue

        # Maschera booleana: True nei pixel del veicolo, False nello sfondo.
        mask_binary = mask > 0
        if not np.any(mask_binary):
            # Maschera completamente nera: segmentazione fallita o veicolo
            # fuori campo. Skip per non sprecare cicli CPU.
            continue

        color = get_color_for_track(int(state.track_id or 0))
        # Assegna il colore del track a tutti i pixel della maschera sull'overlay.
        overlay[mask_binary] = color

    # Alpha blending: overlay * alpha + base_frame * (1 - alpha) + 0 (gamma).
    alpha = float(np.clip(alpha_overlay, 0.0, 1.0))
    return cv2.addWeighted(overlay, alpha, base_frame, 1.0 - alpha, 0.0)


def draw_centroids_and_ids(
    frame: np.ndarray,
    vehicle_states: Iterable[VehicleState],
    draw_ids: bool = True,
) -> np.ndarray:
    """
    Disegna un cerchio nel centroide di ogni veicolo e, opzionalmente, il track ID.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR su cui disegnare.
    vehicle_states : Iterable[VehicleState]
        Stati dei veicoli da annotare.
    draw_ids : bool
        Se True, aggiunge il testo "ID X" accanto al centroide.

    Returns
    -------
    np.ndarray
        Frame annotato con cerchi e ID.
    """
    output = frame.copy()

    for state in vehicle_states:
        centroid = _extract_centroid(state)
        if centroid is None:
            continue

        track_id = int(state.track_id or 0)
        color = get_color_for_track(track_id)

        # Cerchio pieno (thickness=-1) nel centroide, raggio 4 pixel.
        cv2.circle(output, centroid, 4, color, thickness=-1, lineType=cv2.LINE_AA)

        if draw_ids:
            text = f"ID {track_id}"
            # Testo spostato leggermente in alto a destra rispetto al centro.
            text_pos = (centroid[0] + 6, max(0, centroid[1] - 6))
            cv2.putText(
                output,
                text,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return output


def draw_bboxes(frame: np.ndarray, vehicle_states: Iterable[VehicleState]) -> np.ndarray:
    """
    Disegna i rettangoli delle bounding box per ogni veicolo tracciato.

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR su cui disegnare.
    vehicle_states : Iterable[VehicleState]
        Stati dei veicoli le cui bbox devono essere disegnate.

    Returns
    -------
    np.ndarray
        Frame annotato con i rettangoli delle bbox.
    """
    output = frame.copy()

    for state in vehicle_states:
        bbox = _extract_bbox(state)
        if bbox is None:
            # _extract_bbox restituisce None anche per il sentinel (0,0,0,0),
            # quindi i poligoni degeneri non producono rettangoli spurii.
            continue

        track_id = int(state.track_id or 0)
        color = get_color_for_track(track_id)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    return output


def draw_collisions(
    frame: np.ndarray,
    collisions: Iterable[CollisionEvent],
    vehicle_states: Iterable[VehicleState],
) -> np.ndarray:
    """
    Evidenzia in rosso i veicoli coinvolti in un possibile evento di collisione.

    Per ogni evento, disegna rettangolo rosso spesso (3px), cerchio rosso
    sul centroide ed etichetta testuale "CRASH?".

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR su cui disegnare.
    collisions : Iterable[CollisionEvent]
        Lista degli eventi di collisione rilevati nel frame corrente.
    vehicle_states : Iterable[VehicleState]
        Stati di tutti i veicoli attivi (serve per le coordinate).

    Returns
    -------
    np.ndarray
        Frame annotato con le evidenziazioni delle collisioni.
    """
    output = frame.copy()

    # Mappa {track_id: state} per lookup O(1) durante l'iterazione sulle collisioni.
    state_map = _build_state_map(vehicle_states)

    for collision in collisions:
        # Estrae i due track ID coinvolti (track_id_a, track_id_b da CollisionEvent).
        for track_id in _extract_collision_track_ids(collision):
            state = state_map.get(track_id)
            if state is None:
                # Il track potrebbe essere gia stato rimosso come stale.
                continue

            bbox = _extract_bbox(state)
            centroid = _extract_centroid(state)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                # Rettangolo rosso piu spesso (3px) per distinguerlo dai
                # rettangoli normali dei veicoli non in collisione (2px).
                cv2.rectangle(output, (x1, y1), (x2, y2), _CRASH_COLOR, 3, lineType=cv2.LINE_AA)

            if centroid is not None:
                # Cerchio vuoto (thickness=2): non copre il cerchio pieno del centroide.
                cv2.circle(output, centroid, 8, _CRASH_COLOR, 2, lineType=cv2.LINE_AA)

            label_pos = _label_anchor(state)
            cv2.putText(
                output,
                "CRASH?",
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                _CRASH_COLOR,
                2,
                cv2.LINE_AA,
            )

    return output


def render_frame(
    frame: np.ndarray,
    vehicle_states: Iterable[VehicleState],
    collisions: Iterable[CollisionEvent],
    alpha_overlay: float,
    draw_masks: bool = True,
    draw_ids: bool = True,
    draw_bbox: bool = False,
) -> np.ndarray:
    """
    Compone il frame finale annotato applicando tutte le funzioni di rendering.

    Ordine di applicazione (dal piu basso al piu visibile):
      1. Overlay maschere di segmentazione (se draw_masks=True)
      2. Bounding box (se draw_bbox=True)
      3. Centroidi e ID dei track (sempre, se draw_ids=True)
      4. Evidenziazioni collisioni (sempre, se presenti)

    Parameters
    ----------
    frame : np.ndarray
        Frame BGR originale di input (non modificato in-place).
    vehicle_states : Iterable[VehicleState]
        Stati dei veicoli attivi nel frame corrente.
    collisions : Iterable[CollisionEvent]
        Collisioni rilevate per il frame corrente.
    alpha_overlay : float
        Intensita dell'overlay delle maschere (0.0-1.0).
    draw_masks : bool
        Se True, disegna gli overlay delle maschere di segmentazione.
    draw_ids : bool
        Se True, disegna i cerchi dei centroidi e i track ID testuali.
    draw_bbox : bool
        Se True, disegna i rettangoli delle bounding box.

    Returns
    -------
    np.ndarray
        Frame annotato con tutte le sovrapposizioni richieste.
    """
    # Materializza gli iterabili in lista: alcune funzioni li consumano
    # piu di una volta (es. _build_state_map + iterazione per drawing).
    vehicle_states_list = list(vehicle_states)
    collisions_list = list(collisions)

    annotated = frame.copy()

    # Strato 1: maschere di segmentazione semitrasparenti.
    if draw_masks:
        annotated = draw_mask_overlay(annotated, vehicle_states_list, alpha_overlay)

    # Strato 2: bounding box (opzionale, disattivato di default).
    if draw_bbox:
        annotated = draw_bboxes(annotated, vehicle_states_list)

    # Strato 3: centroidi e ID dei track.
    annotated = draw_centroids_and_ids(annotated, vehicle_states_list, draw_ids=draw_ids)

    # Strato 4: evidenziazioni in rosso per i veicoli in collisione (sopra tutto).
    annotated = draw_collisions(annotated, collisions_list, vehicle_states_list)

    return annotated
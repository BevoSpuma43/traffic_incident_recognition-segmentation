"""
src/kinematics.py
=================
Funzioni pure per il calcolo della cinematica in spazio pixel.

Questo modulo non importa nessun modulo interno del progetto: opera
esclusivamente su tipi Python standard e sul Protocol _HasXY, che
definisce il contratto minimo per un oggetto con coordinate 2-D.

Funzioni esportate
------------------
compute_speed_px       -- velocità euclidea tra due centroidi (pixel/frame)
compute_acceleration   -- accelerazione frame-to-frame (pixel/frame²)
update_stopped_counter -- contatore di frame consecutivi da fermo
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol


class _HasXY(Protocol):
    """
    Protocol di duck-typing per oggetti che espongono coordinate 2-D.

    Qualsiasi oggetto con attributi ``x: float`` e ``y: float`` soddisfa
    questo Protocol, incluso models.Point2D (dopo il fix in models.py).

    Le funzioni pubbliche accettano anche coppie numeriche per agevolare test
    e integrazioni, pur usando Point2D nel flusso principale.
    """

    x: float
    y: float


def _coordinates(point: _HasXY | Sequence[float]) -> tuple[float, float]:
    """Estrae coordinate da Point2D-like o da una coppia numerica."""
    if hasattr(point, "x") and hasattr(point, "y"):
        return float(point.x), float(point.y)
    if isinstance(point, Sequence) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise TypeError("Il punto deve esporre x/y oppure contenere due coordinate")


def compute_velocity_px(
    prev_centroid: _HasXY | Sequence[float] | None,
    curr_centroid: _HasXY | Sequence[float] | None,
    frame_delta: int = 1,
) -> tuple[float, float]:
    """Calcola il vettore velocita normalizzato per i frame trascorsi."""
    if prev_centroid is None or curr_centroid is None or frame_delta <= 0:
        return 0.0, 0.0
    prev_x, prev_y = _coordinates(prev_centroid)
    curr_x, curr_y = _coordinates(curr_centroid)
    delta = float(frame_delta)
    return (curr_x - prev_x) / delta, (curr_y - prev_y) / delta


def exponential_moving_average(previous: float, current: float, alpha: float) -> float:
    """Filtra un valore imponendo alpha nell'intervallo [0, 1]."""
    weight = min(1.0, max(0.0, float(alpha)))
    return weight * float(current) + (1.0 - weight) * float(previous)


def compute_bbox_scale_px(bbox: Sequence[float] | None) -> float:
    """Scala apparente di un veicolo: diagonale del bounding box in pixel.

    La diagonale e un proxy del rapporto pixel/metro nel punto della scena
    occupato dal veicolo: a parita di velocita reale, un veicolo lontano si
    sposta di pochi pixel per frame e uno vicino di molti. Normalizzare le
    soglie su questa grandezza le rende indipendenti dalla prospettiva.

    Si preferisce la diagonale all'altezza o all'area della maschera perche e
    meno sensibile sia all'orientamento del veicolo sia alle occlusioni
    parziali della segmentazione.

    Returns
    -------
    float
        Diagonale in pixel, oppure 0.0 se la bbox manca o e degenere.
    """
    if bbox is None or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    return math.hypot(x2 - x1, y2 - y1)


def compute_speed_px(
    prev_centroid: _HasXY | Sequence[float] | None,
    curr_centroid: _HasXY | Sequence[float] | None,
    frame_delta: int = 1,
) -> float:
    """
    Calcola la velocità istantanea di un veicolo come distanza euclidea
    tra il centroide del frame precedente e quello del frame corrente.

    La formula è la norma L2 dello spostamento nel piano pixel:
        speed = sqrt((curr_x - prev_x)^2 + (curr_y - prev_y)^2)

    Il risultato è sempre >= 0 (la funzione restituisce la magnitudine,
    non un vettore direzionale).

    Parameters
    ----------
    prev_centroid : _HasXY | None
        Centroide del frame precedente (attributi .x e .y in pixel).
        Se None, la velocità non è calcolabile: restituisce 0.0.
    curr_centroid : _HasXY | None
        Centroide del frame corrente (attributi .x e .y in pixel).
        Se None, la velocità non è calcolabile: restituisce 0.0.

    Returns
    -------
    float
        Velocità in pixel/frame. Restituisce 0.0 se uno dei due
        centroidi è None (veicolo appena apparso o perso dal tracker).

    Note sull'ordine dei parametri
    --------------------------------
    I parametri sono (prev_centroid, curr_centroid) — centroide vecchio
    prima, nuovo dopo — per coerenza con la chiamata in tracker_state.py:
        compute_speed_px(previous_centroid, centroid)
    Il segno dello spostamento non influenza il risultato perché
    math.hypot è simmetrico: hypot(dx, dy) == hypot(-dx, -dy).
    """
    # Entrambi i centroidi devono essere disponibili per calcolare lo
    # spostamento. Se uno manca, il veicolo è appena entrato in scena
    # o il tracker lo ha perso: la velocità è convenzionalmente 0.
    if prev_centroid is None or curr_centroid is None:
        return 0.0

    dx, dy = compute_velocity_px(prev_centroid, curr_centroid, frame_delta)

    # math.hypot è numericamente più stabile di sqrt(dx**2 + dy**2)
    # perché evita overflow/underflow su valori molto grandi o molto piccoli.
    return math.hypot(dx, dy)


def compute_acceleration(
    curr_speed: float,
    prev_speed: float,
    frame_delta: int = 1,
) -> float:
    """
    Calcola l'accelerazione frame-to-frame come variazione della velocità.

    L'accelerazione è definita come:
        a = v_corrente - v_precedente  (pixel/frame²)

    Un valore negativo indica decelerazione (frenata). Un valore fortemente
    negativo (< strong_deceleration_threshold in config.py) è un indicatore
    di incidente o frenata brusca.

    Parameters
    ----------
    curr_speed : float
        Velocità del frame corrente in pixel/frame.
    prev_speed : float
        Velocità del frame precedente in pixel/frame.

    Returns
    -------
    float
        Accelerazione in pixel/frame². Può essere negativa (decelerazione)
        o positiva (accelerazione). Il valore è esatto a meno di errori
        di arrotondamento floating-point.
    """
    # Differenza finita del primo ordine: approssimazione discreta della
    # derivata della velocità rispetto al tempo (in unità di frame).
    if frame_delta <= 0:
        return 0.0
    return (float(curr_speed) - float(prev_speed)) / float(frame_delta)


def update_stopped_counter(
    prev_counter: int | None = None,
    speed_px: float = 0.0,
    stopped_speed_threshold: float = 0.0,
    *,
    stopped_counter: int | None = None,
) -> int:
    """
    Aggiorna il contatore di frame consecutivi in cui il veicolo è fermo.

    Il contatore viene incrementato di 1 ad ogni frame in cui la velocità
    è al di sotto della soglia di stop, e azzerato non appena il veicolo
    torna in movimento.

    Questo meccanismo consente di distinguere una breve oscillazione del
    tracker (pochi frame a bassa velocità) da un vero arresto prolungato
    (più frame consecutivi). La soglia è configurabile in AppConfig.

    Parameters
    ----------
    prev_counter : int
        Valore del contatore al frame precedente.
    speed_px : float
        Velocità attuale del veicolo in pixel/frame.
    stopped_speed_threshold : float
        Soglia massima (pixel/frame) al di sotto della quale il veicolo
        viene considerato fermo. Corrisponde a AppConfig.stopped_speed_threshold.

    Returns
    -------
    int
        Contatore aggiornato:
          - prev_counter + 1  se speed_px <= stopped_speed_threshold
          - 0                 se il veicolo è tornato in movimento

    Note sull'ordine dei parametri
    --------------------------------
    L'ordine (prev_counter, speed_px, stopped_speed_threshold) è canonico
    e documentato qui per eliminare il try/except di fallback in
    tracker_state.py._update_stopped_counter(), che sarà rimosso in step 5.
    """
    # ``stopped_counter`` mantiene compatibilita con i chiamanti precedenti.
    if prev_counter is None:
        prev_counter = 0 if stopped_counter is None else stopped_counter

    # Il veicolo è considerato fermo se la sua velocità non supera la soglia.
    # Il confronto usa <= per includere velocità esattamente uguali alla soglia.
    if speed_px <= stopped_speed_threshold:
        # Incrementa: un altro frame consecutivo da fermo.
        return prev_counter + 1

    # Il veicolo si è mosso: azzera il contatore per ricominciare a contare
    # dal prossimo stop consecutivo.
    return 0

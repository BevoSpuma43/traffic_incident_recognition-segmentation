"""
src/geometry.py
===============
Funzioni matematiche pure per l'analisi geometrica delle maschere di
segmentazione nel progetto traffic-accident-seg.

Questo modulo non importa nessun altro modulo interno del progetto per
evitare dipendenze circolari. Tutte le funzioni operano su array NumPy
e tipi Python standard, senza dipendenze da config, models o tracker.

Funzioni esportate
------------------
safe_polygon_array         -- normalizza un poligono grezzo per OpenCV
polygon_to_binary_mask     -- crea una maschera binaria da un poligono
compute_centroid_from_polygon -- calcola il centroide via momenti di immagine
compute_bbox_from_polygon  -- calcola il bounding box da un poligono
mask_intersection_area     -- conta i pixel di sovrapposizione tra maschere
masks_touch                -- booleano: le due maschere si toccano?
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def safe_polygon_array(polygon: Any) -> np.ndarray | None:
    """
    Normalizza un poligono grezzo nel formato atteso da OpenCV.

    YOLO restituisce i contorni delle maschere come array di shape (N, 2)
    con coordinate floating-point. OpenCV richiede shape (N, 1, 2) con
    dtype int32. Questa funzione gestisce anche input malformati
    (None, array vuoti, shape incompatibili, valori NaN/Inf).

    Parameters
    ----------
    polygon : Any
        Qualsiasi oggetto array-like che rappresenti un poligono 2-D.
        Tipicamente un array NumPy di shape (N, 2) o (N, 1, 2).

    Returns
    -------
    np.ndarray | None
        Array di shape (N, 1, 2) con dtype int32 se il poligono è valido
        (almeno 3 vertici, tutti finiti), altrimenti None.
    """
    if polygon is None:
        return None

    # Tentativo di conversione a ndarray; cattura qualsiasi eccezione
    # per input non convertibili (es. oggetti arbitrari).
    try:
        array = np.asarray(polygon)
    except Exception:
        return None

    # Scarta array vuoti prima del reshape.
    if array.size == 0:
        return None

    # Porta il poligono a shape (N, 2) per uniformare i diversi formati
    # di input: (N, 2), (N, 1, 2), lista di tuple, ecc.
    try:
        array = array.reshape(-1, 2)
    except ValueError:
        # Il numero totale di elementi non è divisibile per 2:
        # il poligono è malformato.
        return None

    # Un triangolo è il poligono valido minimo: almeno 3 vertici.
    if array.shape[0] < 3:
        return None

    # Scarta poligoni con coordinate NaN o Inf, che causerebbero
    # comportamenti non deterministici in cv2.fillPoly.
    if not np.isfinite(array).all():
        return None

    # Formato finale richiesto da OpenCV: (N, 1, 2), dtype int32.
    # Il reshape aggiunge la dimensione intermedia "1" che OpenCV usa
    # per distinguere i contorni annidati nelle operazioni di drawing.
    return array.astype(np.int32).reshape(-1, 1, 2)


def polygon_to_binary_mask(
    polygon: Any,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """
    Crea una maschera binaria uint8 riempiendo il poligono dato sul canvas.

    La maschera risultante ha lo stesso layout spaziale del frame originale:
    i pixel interni al poligono valgono 255, tutti gli altri 0.

    Parameters
    ----------
    polygon : Any
        Poligono grezzo o già normalizzato (viene ri-validato internamente
        tramite safe_polygon_array per robustezza).
    frame_shape : tuple[int, int]
        Dimensioni del frame (height, width) in pixel.
        Corrisponde a ``frame.shape[:2]`` di un array OpenCV.

    Returns
    -------
    np.ndarray
        Maschera uint8 di shape (height, width). In caso di poligono
        invalido restituisce un array di soli zeri (maschera vuota).

    Note
    ----
    L'ordine degli argomenti è (polygon, frame_shape) per coerenza con
    il chiamante in detector.py:
        ``polygon_to_binary_mask(polygon, frame.shape[:2])``
    La versione originale aveva gli argomenti invertiti, causando maschere
    sempre vuote (il poligono veniva interpretato come frame_shape e
    viceversa, con safe_polygon_array che restituiva None).
    """
    height = int(frame_shape[0])
    width = int(frame_shape[1])

    # Canvas inizialmente nero (nessun pixel attivo).
    mask = np.zeros((height, width), dtype=np.uint8)

    # Validazione e normalizzazione del poligono nel formato OpenCV.
    polygon_array = safe_polygon_array(polygon)
    if polygon_array is None:
        # Poligono invalido: restituisce maschera vuota invece di sollevare.
        return mask

    # cv2.fillPoly richiede una lista di contorni; usiamo il singolo
    # poligono normalizzato. Il valore 255 identifica i pixel "attivi".
    cv2.fillPoly(mask, [polygon_array], 255)
    return mask


def compute_centroid_from_polygon(
    polygon: Any,
) -> tuple[int, int] | None:
    """
    Calcola il centroide geometrico di un poligono tramite momenti di immagine.

    Il centroide è calcolato come:
        cx = m10 / m00
        cy = m01 / m00
    dove m00 è l'area del poligono e m10, m01 sono i momenti del primo
    ordine rispettivamente lungo x e y.

    Parameters
    ----------
    polygon : Any
        Poligono grezzo nel formato accettato da safe_polygon_array.

    Returns
    -------
    tuple[int, int] | None
        Coppia (cx, cy) in pixel interi, oppure None se il poligono è
        invalido o ha area zero (poligono degenere).

    Note
    ----
    Il valore restituito è una tupla Python, non un Point2D. tracker_state.py
    è responsabile della conversione a Point2D prima di memorizzarlo nello
    stato del veicolo.
    """
    polygon_array = safe_polygon_array(polygon)
    if polygon_array is None:
        return None

    # cv2.moments calcola i momenti geometrici del contorno.
    # m00 corrisponde all'area (in pixel) del poligono riempito.
    moments = cv2.moments(polygon_array)
    m00 = moments.get("m00", 0.0)

    # m00 == 0 indica un poligono degenere (area nulla): impossibile
    # dividere per zero per calcolare il centroide.
    if m00 == 0.0:
        return None

    # Centroide come quoziente dei momenti del primo ordine e dell'area.
    cx = int(moments["m10"] / m00)
    cy = int(moments["m01"] / m00)
    return cx, cy


def compute_bbox_from_polygon(
    polygon: Any,
) -> tuple[int, int, int, int]:
    """
    Calcola il bounding box axis-aligned (AABB) dal poligono dato.

    Parameters
    ----------
    polygon : Any
        Poligono grezzo nel formato accettato da safe_polygon_array.

    Returns
    -------
    tuple[int, int, int, int]
        Bounding box come (x_min, y_min, x_max, y_max) in pixel interi.
        Restituisce (0, 0, 0, 0) come sentinel se il poligono è invalido.
        Nota: detector.py controlla ``if bbox is None`` che non scatterà
        mai con questo return; il controllo sarà uniformato in detector.py.
    """
    polygon_array = safe_polygon_array(polygon)
    if polygon_array is None:
        # Sentinel: bbox degenere. detector.py gestisce questo caso.
        return 0, 0, 0, 0

    # Porta il poligono a shape (N, 2) per accedere alle colonne x e y.
    points = polygon_array.reshape(-1, 2)

    # Min/max lungo la dimensione dei punti per ottenere i quattro angoli.
    x_min = int(np.min(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    x_max = int(np.max(points[:, 0]))
    y_max = int(np.max(points[:, 1]))
    return x_min, y_min, x_max, y_max


def mask_intersection_area(
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
) -> int:
    """
    Conta i pixel sovrapposti (foreground) tra due maschere binarie.

    Utilizza cv2.bitwise_AND per calcolare l'intersezione pixel-a-pixel
    tra le due maschere. Il risultato è il numero di pixel in cui entrambe
    le maschere sono attive (valore > 0), che corrisponde all'area di
    sovrapposizione in pixel².

    Parameters
    ----------
    mask_a : np.ndarray | None
        Prima maschera binaria uint8 di shape (H, W).
    mask_b : np.ndarray | None
        Seconda maschera binaria uint8 di shape (H, W).
        Deve avere la stessa shape di mask_a.

    Returns
    -------
    int
        Numero di pixel sovrapposti. Restituisce 0 in caso di input
        invalidi o shape incompatibili (fail-safe silenzioso).
    """
    # Controlli difensivi: None, non-array, shape diverse.
    if mask_a is None or mask_b is None:
        return 0

    if not isinstance(mask_a, np.ndarray) or not isinstance(mask_b, np.ndarray):
        return 0

    # Le due maschere devono avere esattamente la stessa forma spaziale
    # perché bitwise_and opera elemento per elemento.
    if mask_a.shape != mask_b.shape:
        return 0

    # bitwise_AND: pixel = 255 se entrambe le maschere sono attive (255 & 255 = 255),
    # 0 altrimenti. Il confronto "> 0" trasforma il risultato in booleano
    # prima del conteggio, rendendo il codice robusto a maschere non binarie.
    intersection = cv2.bitwise_and(mask_a, mask_b)
    return int(np.count_nonzero(intersection > 0))


def masks_touch(
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
) -> bool:
    """
    Controlla se due maschere di segmentazione si sovrappongono.

    Wrapper booleano di mask_intersection_area: restituisce True se esiste
    almeno un pixel in cui entrambe le maschere sono attive.

    Parameters
    ----------
    mask_a : np.ndarray | None
        Prima maschera binaria uint8.
    mask_b : np.ndarray | None
        Seconda maschera binaria uint8.

    Returns
    -------
    bool
        True se le maschere si toccano (overlap > 0 pixel), False altrimenti.
    """
    return mask_intersection_area(mask_a, mask_b) > 0


def bbox_distance(
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
) -> float:
    """Distanza euclidea minima tra due bounding box axis-aligned."""
    if bbox_a is None or bbox_b is None:
        return math.inf
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    dx = max(float(ax1 - bx2), float(bx1 - ax2), 0.0)
    dy = max(float(ay1 - by2), float(by1 - ay2), 0.0)
    return math.hypot(dx, dy)


def mask_intersection_area_in_roi(
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
    padding: int = 0,
) -> int:
    """Calcola l'intersezione limitandola all'unione locale delle bbox."""
    if (
        mask_a is None
        or mask_b is None
        or bbox_a is None
        or bbox_b is None
        or mask_a.shape != mask_b.shape
        or mask_a.ndim < 2
    ):
        return 0
    height, width = mask_a.shape[:2]
    x1 = max(0, min(bbox_a[0], bbox_b[0]) - padding)
    y1 = max(0, min(bbox_a[1], bbox_b[1]) - padding)
    x2 = min(width, max(bbox_a[2], bbox_b[2]) + padding + 1)
    y2 = min(height, max(bbox_a[3], bbox_b[3]) + padding + 1)
    if x1 >= x2 or y1 >= y2:
        return 0
    intersection = cv2.bitwise_and(mask_a[y1:y2, x1:x2], mask_b[y1:y2, x1:x2])
    return int(np.count_nonzero(intersection > 0))


def normalized_overlap(
    overlap_area: int,
    area_a: int,
    area_b: int,
) -> float:
    """Normalizza l'intersezione rispetto alla sagoma piu piccola."""
    denominator = min(int(area_a), int(area_b))
    if denominator <= 0:
        return 0.0
    return float(overlap_area) / float(denominator)


def dilated_masks_touch_in_roi(
    mask_a: np.ndarray | None,
    mask_b: np.ndarray | None,
    bbox_a: tuple[int, int, int, int] | None,
    bbox_b: tuple[int, int, int, int] | None,
    dilation_pixels: int,
) -> bool:
    """Verifica la prossimita delle sagome tramite dilatazione locale."""
    radius = max(0, int(dilation_pixels))
    if radius == 0:
        return mask_intersection_area_in_roi(mask_a, mask_b, bbox_a, bbox_b) > 0
    if (
        mask_a is None
        or mask_b is None
        or bbox_a is None
        or bbox_b is None
        or mask_a.shape != mask_b.shape
        or mask_a.ndim < 2
    ):
        return False
    height, width = mask_a.shape[:2]
    x1 = max(0, min(bbox_a[0], bbox_b[0]) - radius)
    y1 = max(0, min(bbox_a[1], bbox_b[1]) - radius)
    x2 = min(width, max(bbox_a[2], bbox_b[2]) + radius + 1)
    y2 = min(height, max(bbox_a[3], bbox_b[3]) + radius + 1)
    if x1 >= x2 or y1 >= y2:
        return False
    kernel_size = radius * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    region_a = (mask_a[y1:y2, x1:x2] > 0).astype(np.uint8)
    region_b = (mask_b[y1:y2, x1:x2] > 0).astype(np.uint8)
    dilated_a = cv2.dilate(region_a, kernel, iterations=1)
    return bool(np.any((dilated_a > 0) & (region_b > 0)))

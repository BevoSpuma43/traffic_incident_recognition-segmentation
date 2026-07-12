"""
src/detector.py
===============
Integrazione con il modello YOLO segmenter e il tracker ByteTrack.

Questo modulo espone la classe VehicleSegmenter che, dato un frame video,
restituisce la lista delle detection per il frame corrente, ognuna arricchita
con poligono di segmentazione, maschera binaria, bounding box e track ID.

Dipendenze interne (bottom-up):
  src.config   → AppConfig (parametri del modello)
  src.models   → DetectionResult
  src.geometry → safe_polygon_array, polygon_to_binary_mask, compute_bbox_from_polygon
"""

from __future__ import annotations

from os import PathLike

import numpy as np
from ultralytics import YOLO

from src.config import AppConfig
from src.geometry import (
    compute_bbox_from_polygon,
    polygon_to_binary_mask,
    safe_polygon_array,
)
from src.models import DetectionResult

# Sentinel restituito da compute_bbox_from_polygon() quando il poligono è invalido.
# Una bbox con tutti zero è degenere: va scartata come se fosse None.
_DEGENERATE_BBOX: tuple[int, int, int, int] = (0, 0, 0, 0)


class VehicleSegmenter:
    """
    Segmenter di veicoli basato su YOLO con tracking persistente ByteTrack.

    Wrappa il modello YOLO in modalità segmentazione e restituisce, per ogni
    frame, la lista dei veicoli rilevati come DetectionResult (incluse maschera
    binaria e poligono normalizzato per OpenCV).

    Modalità di costruzione
    -----------------------
    Il costruttore accetta indifferentemente:
      a) un oggetto AppConfig  → estrae model_path e parametri dalla config
      b) un path (str o Path)  → usa i parametri passati esplicitamente

    Questa doppia modalità consente sia l'integrazione con la pipeline
    (video_pipeline.py passa AppConfig) sia l'uso stand-alone nei test.
    """

    def __init__(
        self,
        model_path_or_config: AppConfig | str | PathLike[str],
        vehicle_classes: list[int] | None = None,
        confidence: float | None = None,
        iou_threshold: float | None = None,
        tracker: str = "bytetrack.yaml",
    ) -> None:
        """
        Inizializza il segmenter YOLO con i parametri necessari.

        Parameters
        ----------
        model_path_or_config : AppConfig | str | PathLike[str]
            Se è un oggetto AppConfig, i parametri vengono estratti dalla
            config (modalità pipeline). Se è un path, vengono usati i
            parametri opzionali passati direttamente (modalità test/standalone).
        vehicle_classes : list[int] | None
            Classi COCO da rilevare. Usato solo in modalità path.
        confidence : float | None
            Soglia di confidenza YOLO (0.0–1.0). Usato solo in modalità path.
        iou_threshold : float | None
            Soglia IoU per NMS interno YOLO (0.0–1.0). Usato solo in modalità path.
        tracker : str
            Nome del file YAML di configurazione del tracker ByteTrack.
        """
        # Modalità config: il primo argomento è un AppConfig, non un path.
        # Si distingue tramite isinstance: str e PathLike sono path, tutto il
        # resto viene trattato come config (duck-typing).
        if not isinstance(model_path_or_config, (str, PathLike)):
            config = model_path_or_config
            model_path_or_config = str(getattr(config, "model_path"))
            vehicle_classes = list(getattr(config, "vehicle_classes", [2, 3, 5, 7]))
            confidence = float(getattr(config, "confidence", 0.35))
            iou_threshold = float(getattr(config, "iou_threshold", 0.45))

        # Caricamento del modello YOLO. str() garantisce compatibilità con PathLike.
        self.model: YOLO = YOLO(str(model_path_or_config))

        # Classi COCO da rilevare (default: car=2, motorcycle=3, bus=5, truck=7).
        self.vehicle_classes: list[int] = list(vehicle_classes or [2, 3, 5, 7])

        # Soglie del modello con valori di default nel caso in cui non siano
        # stati specificati nemmeno tramite config.
        self.confidence: float = float(0.35 if confidence is None else confidence)
        self.iou_threshold: float = float(0.45 if iou_threshold is None else iou_threshold)

        # Nome del file YAML del tracker ByteTrack (ricercato nella cartella
        # interna di ultralytics o nel percorso specificato).
        self.tracker: str = tracker

    # ------------------------------------------------------------------
    # Interfaccia pubblica
    # ------------------------------------------------------------------

    def segment_and_track(
        self,
        frame: np.ndarray,
        vehicle_classes: list[int] | None = None,
    ) -> list[DetectionResult]:
        """
        Esegue segmentazione e tracking su un singolo frame video.

        Chiama YOLO in modalità `track` con ByteTrack persistente, poi
        normalizza ogni detection in un DetectionResult con poligono OpenCV,
        maschera binaria e bounding box.

        Parameters
        ----------
        frame : np.ndarray
            Frame BGR di shape (H, W, 3), come restituito da cv2.VideoCapture.
        vehicle_classes : list[int] | None
            Override opzionale delle classi da rilevare per questo frame.
            Se None, vengono usate quelle salvate in self.vehicle_classes.

        Returns
        -------
        list[DetectionResult]
            Lista delle detection valide (poligono ≥ 3 vertici, maschera
            non vuota, bbox non degenere). Può essere vuota se YOLO non
            rileva nulla o se tutti i poligoni sono invalidi.
        """
        if vehicle_classes is None:
            vehicle_classes = self.vehicle_classes

        # Invoca YOLO con ByteTrack persistente.
        # persist=True mantiene lo stato del tracker tra frame consecutivi,
        # garantendo la continuità degli ID dei track.
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker,
            classes=vehicle_classes,
            verbose=False,
            conf=self.confidence,
            iou=self.iou_threshold,
        )

        # Verifica che YOLO abbia restituito almeno un risultato.
        if not results:
            return []

        result = results[0]

        # Controllo difensivo: YOLO può restituire result senza maschere
        # (es. frame senza veicoli, modello non-seg, classi assenti).
        if result is None or result.masks is None or result.boxes is None:
            return []

        # result.masks.xy contiene i poligoni di segmentazione come lista
        # di array NumPy di shape (N, 2) con coordinate float.
        polygons_raw = result.masks.xy
        boxes = result.boxes

        if not polygons_raw or len(boxes) == 0:
            return []

        # Estrazione dei tensori PyTorch → liste Python standard.
        class_ids: list[object] = self._to_list(boxes.cls)
        confidences: list[object] = self._to_list(boxes.conf)
        track_ids: list[object] = self._to_list(boxes.id)
        names: object = getattr(result, "names", {})

        # Numero di istanze da processare: limitato al minimo tra poligoni,
        # class_ids e confidences per evitare IndexError in caso di
        # disallineamento tra le liste (anomalia rara ma possibile).
        if not class_ids or not confidences:
            return []

        total_instances: int = min(len(polygons_raw), len(class_ids), len(confidences))

        detections: list[DetectionResult] = []

        for index in range(total_instances):
            try:
                # Normalizza il poligono grezzo nel formato OpenCV (N, 1, 2) int32.
                # YOLO restituisce float, OpenCV richiede int32 per cv2.fillPoly.
                polygon = safe_polygon_array(polygons_raw[index])
                if polygon is None or polygon.shape[0] < 3:
                    # Poligono invalido o con meno di 3 vertici: skip.
                    continue

                # Crea la maschera binaria uint8 (0/255) dalle stesse dimensioni
                # del frame. L'ordine (polygon, frame.shape[:2]) è corretto
                # dopo il fix di geometry.py (versione originale era invertito).
                mask = polygon_to_binary_mask(polygon, frame.shape[:2])
                if mask.size == 0:
                    # Maschera vuota (frame di shape (0,0)?): skip difensivo.
                    continue

                # Calcola il bounding box axis-aligned dal poligono.
                # compute_bbox_from_polygon restituisce (0,0,0,0) come sentinel
                # invece di None quando il poligono è invalido, quindi
                # controlliamo esplicitamente il sentinel invece di `is None`.
                bbox = compute_bbox_from_polygon(polygon)
                if bbox == _DEGENERATE_BBOX:
                    continue

                class_id: int = int(class_ids[index])
                class_name: str = self._resolve_class_name(names, class_id)
                conf_value: float = float(confidences[index])
                track_id: int | None = self._resolve_track_id(track_ids, index)

                detections.append(
                    DetectionResult(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=class_name,
                        confidence=conf_value,
                        polygon=polygon,
                        mask=mask,
                        bbox=bbox,
                    )
                )

            except Exception:
                # Skip difensivo: una singola detection malformata non deve
                # interrompere il processing dell'intero frame.
                # Gli errori tipici qui sono IndexError su liste di lunghezza
                # inaspettata o ValueError su tensori NaN.
                continue

        return detections

    # ------------------------------------------------------------------
    # Metodi statici di supporto
    # ------------------------------------------------------------------

    @staticmethod
    def _to_list(values: object) -> list[object]:
        """
        Converte tensori PyTorch o array NumPy in una lista Python standard.

        YOLO restituisce cls, conf e id come tensori 1-D. Questo metodo
        gestisce la catena di conversione: Tensor → CPU → NumPy → list,
        con fallback per tipi già convertiti o scalari.

        Parameters
        ----------
        values : object
            Tensore PyTorch, array NumPy, lista o scalare da convertire.

        Returns
        -------
        list[object]
            Lista piatta degli elementi. Lista vuota se values è None.
        """
        if values is None:
            return []

        # Sposta il tensore dalla GPU alla CPU prima di accedere ai dati.
        if hasattr(values, "cpu"):
            values = values.cpu()

        # Converte da tensore PyTorch a ndarray NumPy.
        if hasattr(values, "numpy"):
            values = values.numpy()

        # Appiattisce l'ndarray e lo converte in lista Python nativa.
        if isinstance(values, np.ndarray):
            return values.reshape(-1).tolist()

        # Gestisce oggetti con .tolist() (es. tensori già su CPU).
        if hasattr(values, "tolist"):
            converted = values.tolist()
            # .tolist() su un tensore scalare restituisce un Python scalar,
            # non una lista: lo avvolgiamo in lista per uniformità.
            if isinstance(converted, list):
                return converted
            return [converted]

        # Fallback generico: tenta la conversione a lista.
        try:
            return list(values)
        except TypeError:
            # Scalare non iterabile: lo avvolge in una lista singola.
            return [values]

    @staticmethod
    def _resolve_class_name(names: object, class_id: int) -> str:
        """
        Risolve il nome testuale della classe dal dizionario YOLO.

        YOLO restituisce `result.names` come dizionario {int: str} o come
        lista, a seconda della versione. Questo metodo gestisce entrambi i
        formati con fallback al numero di classe come stringa.

        Parameters
        ----------
        names : object
            Dizionario o lista dei nomi delle classi da result.names.
        class_id : int
            Indice numerico della classe COCO.

        Returns
        -------
        str
            Nome della classe (es. "car") oppure str(class_id) come fallback.
        """
        if isinstance(names, dict):
            # Formato standard YOLO: {2: "car", 3: "motorcycle", ...}
            return str(names.get(class_id, str(class_id)))

        if isinstance(names, list) and 0 <= class_id < len(names):
            # Formato lista: ["person", "bicycle", "car", ...]
            return str(names[class_id])

        # Fallback: restituisce l'ID numerico come stringa.
        return str(class_id)

    @staticmethod
    def _resolve_track_id(track_ids: list[object], index: int) -> int | None:
        """
        Estrae e valida il track ID di ByteTrack per una detection.

        ByteTrack può non assegnare un ID in alcuni frame (es. primo frame,
        detection con score basso, track perso). In questi casi `boxes.id`
        è None o contiene NaN come placeholder.

        Parameters
        ----------
        track_ids : list[object]
            Lista degli ID di tracking estratti da boxes.id.
        index : int
            Indice della detection corrente.

        Returns
        -------
        int | None
            ID intero del track, oppure None se:
              - la lista è vuota
              - l'indice è fuori range
              - il valore è None
              - il valore è NaN (ByteTrack non ha ancora assegnato un ID)
        """
        # Lista vuota o indice fuori range: track non disponibile.
        if not track_ids or index >= len(track_ids):
            return None

        raw_track_id = track_ids[index]

        if raw_track_id is None:
            return None

        try:
            # ByteTrack usa NaN come placeholder quando non ha ancora
            # associato un track stabile alla detection.
            if isinstance(raw_track_id, float) and np.isnan(raw_track_id):
                return None

            return int(raw_track_id)

        except (TypeError, ValueError):
            # Valore non convertibile a int: track non disponibile.
            return None
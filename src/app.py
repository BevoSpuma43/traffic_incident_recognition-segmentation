"""
src/app.py
==========
Entry point per l'esecuzione da riga di comando (CLI headless) del sistema
di rilevamento incidenti stradali.

Avvia la pipeline su un file video specificato nella configurazione e mostra
il risultato in una finestra OpenCV. Stampa sulla console gli eventi di
collisione rilevati con frame, track ID e area di sovrapposizione.

Utilizzo:
    python -m src.app
    oppure: python src/app.py

Dipendenze interne:
    src.config        -> AppConfig, load_default_config
    src.video_pipeline -> TrafficAccidentPipeline
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import replace

import cv2

# Aggiunge la cartella radice del progetto al sys.path quando il modulo viene
# eseguito direttamente (python src/app.py) invece che come pacchetto
# (python -m src.app). In questo modo gli import "from src.*" funzionano
# indipendentemente dalla directory di lavoro.
if __package__ is None or __package__ == "":
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from src.config import AppConfig, load_default_config
from src.video_pipeline import TrafficAccidentPipeline


def main() -> None:
    """
    Entry point principale per la modalita CLI.

    Legge la configurazione dal default config (AppConfig con load_default_config),
    apre il video indicato da config.video_path, esegue la pipeline frame-by-frame
    e mostra il risultato in una finestra OpenCV.

    Tasti durante la riproduzione:
      q -- interrompe la riproduzione e chiude la finestra
    """
    # Carica la configurazione dal default: usa load_default_config() che
    # restituisce un AppConfig gia popolato con tutti i valori di default.
    parser = argparse.ArgumentParser(description="Traffic accident detector")
    parser.add_argument("--video", help="Video da analizzare")
    parser.add_argument("--model", help="Modello YOLO segmentation")
    parser.add_argument("--calibration-log", help="Output diagnostico JSONL")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Non apre la finestra OpenCV (utile per calibrazione batch)",
    )
    args = parser.parse_args()

    config: AppConfig = load_default_config()
    config = replace(
        config,
        video_path=args.video or config.video_path,
        model_path=args.model or config.model_path,
        calibration_log_path=(
            args.calibration_log
            if args.calibration_log is not None
            else config.calibration_log_path
        ),
    )

    # Apertura del file video tramite OpenCV VideoCapture.
    video_path: str = getattr(config, "video_path", "")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"[ERRORE] Impossibile aprire la sorgente video: '{video_path}'")
        return

    frame_index: int = 0
    video_fps = float(capture.get(cv2.CAP_PROP_FPS))
    # Il rilevatore normalizza le soglie cinematiche sul frame rate reale:
    # senza questo valore resterebbero tarate sul frame rate di riferimento.
    config = replace(config, video_fps=max(0.0, video_fps))
    pipeline: TrafficAccidentPipeline | None = None

    try:
        # Inizializzazione della pipeline completa (YOLO + tracker + collision detector).
        pipeline = TrafficAccidentPipeline(config)
        while True:
            success, frame = capture.read()
            if not success:
                # Fine del video o errore di lettura: uscita normale dal loop.
                break

            # process_frame restituisce sempre (annotated_frame, list[CollisionEvent]).
            # annotated_frame: frame BGR con maschere, centroidi e alert CRASH disegnati.
            # collision_events: lista degli eventi rilevati nel frame corrente.
            timestamp_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if frame_index > 0 and timestamp_s <= 0.0 and video_fps > 0.0:
                timestamp_s = frame_index / video_fps
            annotated_frame, collision_events = pipeline.process_frame(
                frame,
                frame_index,
                timestamp_s=timestamp_s,
            )

            # Mostra il frame annotato nella finestra OpenCV.
            if not args.headless:
                cv2.imshow("Traffic Accident Segmentation", annotated_frame)

            # Stampa sulla console gli eventi di collisione rilevati in questo frame.
            _log_collisions(collision_events, frame_index)

            frame_index += 1

            # Attende 1 ms per il prossimo frame e controlla il tasto 'q'.
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    finally:
        # Rilascio garantito delle risorse anche in caso di eccezione.
        if pipeline is not None:
            pipeline.close()
        capture.release()
        cv2.destroyAllWindows()


def _log_collisions(
    collision_events: object,
    current_frame_index: int,
) -> None:
    """
    Stampa sulla console i dettagli degli eventi di collisione rilevati.

    Per ogni evento, mostra:
      - indice del frame
      - track ID dei due veicoli coinvolti (track_id_a, track_id_b)
      - area di sovrapposizione pixel-level
      - motivo del rilevamento (dual stop / hard deceleration)

    Parameters
    ----------
    collision_events : object
        Lista (o iterabile) di CollisionEvent restituita da process_frame.
    current_frame_index : int
        Indice del frame corrente, usato come fallback se l'evento non ha
        il campo frame_index.
    """
    # Verifica che collision_events sia effettivamente iterabile prima di iterare.
    if not isinstance(collision_events, Iterable):
        return

    for event in collision_events:
        # Legge i track ID direttamente dai campi canonici di CollisionEvent
        # (track_id_a e track_id_b, dopo il fix di models.py).
        track_id_a = getattr(event, "track_id_a", None)
        track_id_b = getattr(event, "track_id_b", None)

        frame_number = getattr(event, "frame_index", current_frame_index)
        overlap_area = getattr(event, "overlap_area", None)
        reason = getattr(event, "reason", "")
        overlap_ratio = getattr(event, "overlap_ratio", None)
        confidence = getattr(event, "confidence", None)
        first_contact = getattr(event, "first_contact_frame", None)

        # Costruisce il messaggio di log in modo incrementale.
        message = f"[COLLISION] frame={frame_number}"

        if track_id_a is not None and track_id_b is not None:
            tracks = sorted([int(track_id_a), int(track_id_b)])
            message += f" tracks={tracks}"

        if overlap_area is not None:
            message += f" overlap={overlap_area}px"

        if reason:
            message += f" reason={reason}"

        if overlap_ratio is not None:
            message += f" overlap_ratio={float(overlap_ratio):.3f}"

        if confidence is not None:
            message += f" confidence={float(confidence):.2f}"

        if first_contact is not None:
            message += f" first_contact={first_contact}"

        print(message)


if __name__ == "__main__":
    main()

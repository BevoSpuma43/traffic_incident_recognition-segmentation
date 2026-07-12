"""
src/ui_main.py
==============
Interfaccia grafica (GUI) per il sistema di rilevamento incidenti stradali.

Usa customtkinter per una finestra di controllo minima con:
  - Campo per il percorso del video
  - Campo per il percorso del modello YOLO
  - Pulsante Start / Stop
  - Label di stato

Il processing video avviene in un thread daemon separato per non bloccare
il mainloop della GUI. La finestra OpenCV con il video annotato viene aperta
e gestita direttamente dal thread worker.

Dipendenze interne:
    src.config          -> AppConfig, load_default_config
    src.video_pipeline  -> TrafficAccidentPipeline
"""

from __future__ import annotations

import os
import sys

# Aggiunge la root del progetto a sys.path quando ui_main.py viene eseguito
# direttamente come script (python src/ui_main.py) invece che come modulo
# (python -m src.ui_main). Senza questa guardia, Python trova src/ nel path
# invece della root del progetto e tutti gli import 'from src.*' falliscono.
if __package__ is None or __package__ == "":
    _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import threading
from dataclasses import is_dataclass, replace
from typing import Any

import cv2
import customtkinter as ctk

from src.config import AppConfig, load_default_config
from src.video_pipeline import TrafficAccidentPipeline


class TrafficApp(ctk.CTk):
    """
    Finestra principale della GUI per il controllo della pipeline.

    Eredita da customtkinter.CTk per una UI moderna con supporto al
    tema scuro. Gestisce il ciclo di vita del thread di processing e
    aggiorna la label di stato in modo thread-safe tramite self.after().
    """

    def __init__(self) -> None:
        """Inizializza la finestra principale e costruisce il layout."""
        super().__init__()
        self.title("traffic-accident-seg")
        self.geometry("720x240")
        self.resizable(False, False)

        # Evento di sincronizzazione: viene settato quando l'utente
        # preme Stop o chiude la finestra, segnalando al thread di fermarsi.
        self._stop_event = threading.Event()

        # Riferimento al thread worker (None se non e in esecuzione).
        self._worker_thread: threading.Thread | None = None

        # Nome della finestra OpenCV: costante per poterla chiudere in modo mirato.
        self._window_name = "Traffic Accident Seg"

        self._build_layout()

        # Intercetta la chiusura della finestra (X) per fermare il thread
        # prima di distruggere la GUI.
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Costruzione del layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """
        Crea e posiziona tutti i widget della finestra principale.

        Layout a griglia (2 colonne):
          - Riga 0: etichetta + campo percorso video
          - Riga 1: etichetta + campo percorso modello
          - Riga 2: pulsanti Start e Stop
          - Riga 3: etichetta di stato
        """
        # Carica i valori di default dalla configurazione per pre-popolare i campi.
        defaults = load_default_config()
        default_video_path = self._safe_attr(defaults, "video_path")
        default_model_path = self._safe_attr(defaults, "model_path")

        # Colonna 1 (campo di testo) si espande orizzontalmente con la finestra.
        self.grid_columnconfigure(1, weight=1)

        # --- Riga 0: percorso video ---
        video_label = ctk.CTkLabel(self, text="Video path")
        video_label.grid(row=0, column=0, padx=16, pady=(20, 8), sticky="w")

        self.video_entry = ctk.CTkEntry(self, width=520)
        self.video_entry.grid(row=0, column=1, padx=(0, 16), pady=(20, 8), sticky="ew")
        self.video_entry.insert(0, default_video_path)

        # --- Riga 1: percorso modello YOLO ---
        model_label = ctk.CTkLabel(self, text="Model path")
        model_label.grid(row=1, column=0, padx=16, pady=8, sticky="w")

        self.model_entry = ctk.CTkEntry(self, width=520)
        self.model_entry.grid(row=1, column=1, padx=(0, 16), pady=8, sticky="ew")
        self.model_entry.insert(0, default_model_path)

        # --- Riga 2: pulsanti Start e Stop ---
        buttons_frame = ctk.CTkFrame(self, fg_color="transparent")
        buttons_frame.grid(row=2, column=0, columnspan=2, padx=16, pady=(16, 8), sticky="w")

        self.start_button = ctk.CTkButton(
            buttons_frame,
            text="Start",
            command=self.start_processing,
            width=120,
        )
        self.start_button.grid(row=0, column=0, padx=(0, 8), pady=0)

        # Stop disabilitato all'avvio: si attiva quando il processing e in corso.
        self.stop_button = ctk.CTkButton(
            buttons_frame,
            text="Stop",
            command=self.stop_processing,
            width=120,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=1, padx=0, pady=0)

        # --- Riga 3: label di stato ---
        self.status_label = ctk.CTkLabel(self, text="Status: idle", anchor="w")
        self.status_label.grid(row=3, column=0, columnspan=2, padx=16, pady=(8, 16), sticky="ew")

    # ------------------------------------------------------------------
    # Costruzione della configurazione runtime
    # ------------------------------------------------------------------

    def build_config(self) -> AppConfig:
        """
        Costruisce la configurazione runtime dai valori inseriti nella GUI.

        Parte dal config di default e sovrascrive solo video_path e model_path
        con i valori digitati dall'utente nei campi di testo.

        Returns
        -------
        AppConfig
            Configurazione con video_path e model_path aggiornati.
        """
        default_config = load_default_config()
        video_path = self.video_entry.get().strip()
        model_path = self.model_entry.get().strip()

        # AppConfig e una dataclass: usa dataclasses.replace() per creare
        # una copia immutabile con solo i campi modificati.
        if is_dataclass(default_config):
            try:
                return replace(
                    default_config,
                    video_path=video_path,
                    model_path=model_path,
                )
            except TypeError:
                # Fallback nel caso in cui video_path o model_path non siano
                # campi della dataclass (versione non standard di AppConfig).
                pass

        # Fallback: setattr diretto (per AppConfig senza slots o SimpleNamespace).
        try:
            setattr(default_config, "video_path", video_path)
            setattr(default_config, "model_path", model_path)
            return default_config
        except Exception:
            # Ultimo fallback: crea AppConfig dai __dict__ della config default.
            config_data: dict[str, Any] = {}
            if hasattr(default_config, "__dict__"):
                config_data.update(vars(default_config))
            config_data["video_path"] = video_path
            config_data["model_path"] = model_path
            return AppConfig(**config_data)

    # ------------------------------------------------------------------
    # Controllo del processing
    # ------------------------------------------------------------------

    def start_processing(self) -> None:
        """
        Avvia il thread di processing video se non e gia in esecuzione.

        Verifica che video_path e model_path siano stati specificati prima
        di avviare il thread worker.
        """
        # Evita di avviare un secondo thread se uno e gia attivo.
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._set_status("gia in esecuzione")
            return

        config = self.build_config()
        if not self._safe_attr(config, "video_path"):
            self._set_status("percorso video mancante")
            return
        if not self._safe_attr(config, "model_path"):
            self._set_status("percorso modello mancante")
            return

        # Resetta l'evento di stop e aggiorna lo stato della GUI.
        self._stop_event.clear()
        self._set_controls(is_running=True)
        self._set_status("avvio in corso...")

        # Thread daemon: viene terminato automaticamente alla chiusura dell'app
        # anche se il worker non ha finito (failsafe).
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="traffic-processing-thread",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_processing(self) -> None:
        """
        Segnala al thread worker di fermarsi.

        Non blocca: imposta solo l'evento di stop che il thread controlla
        ad ogni frame nel suo ciclo principale.
        """
        self._stop_event.set()
        self._set_status("arresto in corso...")

    # ------------------------------------------------------------------
    # Thread worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Ciclo principale di processing video, eseguito nel thread worker.

        Apre il file video, inizializza la pipeline e processa i frame
        in sequenza finche il video non finisce o l'utente non preme Stop.
        La finestra OpenCV con il video annotato viene gestita interamente
        in questo thread per evitare conflitti con il thread della GUI.
        """
        cap: cv2.VideoCapture | None = None
        pipeline: TrafficAccidentPipeline | None = None
        final_status = "idle"

        try:
            config = self.build_config()

            # Apertura del file video.
            cap = cv2.VideoCapture(config.video_path)
            if not cap.isOpened():
                final_status = "impossibile aprire il video"
                return

            # Inizializzazione della pipeline (carica YOLO, ByteTrack, detector).
            pipeline = TrafficAccidentPipeline(config)
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._set_status("in esecuzione - premi Q per fermare")

            # Contatore del frame corrente: necessario per la logica di tracking
            # (remove_stale_tracks e cinematica dipendono dall'indice crescente).
            frame_index: int = 0

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    # Fine del video raggiunta normalmente.
                    final_status = "completato"
                    break

                # Esegue la pipeline completa: segmentazione, tracking,
                # rilevamento collisioni e rendering delle annotazioni.
                annotated_frame, collision_events = self._process_pipeline_frame(
                    pipeline, frame, frame_index
                )

                # Mostra il frame annotato nella finestra OpenCV.
                cv2.imshow(self._window_name, annotated_frame)

                # Aggiorna la label di stato se ci sono collisioni in questo frame.
                if collision_events:
                    self._set_status(
                        f"in esecuzione - {len(collision_events)} collisione/i al frame {frame_index}"
                    )

                frame_index += 1

                # Legge un tasto OpenCV: 'q' interrompe il loop anche senza
                # premere il pulsante Stop nella GUI.
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    final_status = "fermato dall'utente"
                    self._stop_event.set()
                    break
            else:
                # Il ciclo e uscito perche _stop_event e stato settato da Stop button.
                final_status = "fermato"

        except Exception as exc:
            # Cattura qualsiasi eccezione non gestita e la mostra nella status label.
            final_status = f"errore: {exc}"

        finally:
            # Pulizia garantita: rilascio risorse pipeline e video, chiusura
            # finestra OpenCV e ripristino dei controlli della GUI.
            self._cleanup_pipeline(pipeline)
            if cap is not None:
                cap.release()
            self._destroy_cv_window()
            self._stop_event.set()
            self._worker_thread = None
            self._set_controls(is_running=False)
            self._set_status(final_status)

    # ------------------------------------------------------------------
    # Metodi privati di supporto al worker
    # ------------------------------------------------------------------

    def _process_pipeline_frame(
        self,
        pipeline: TrafficAccidentPipeline,
        frame: Any,
        frame_index: int,
    ) -> tuple[Any, list]:
        """
        Esegue un singolo frame attraverso la pipeline di processing.

        Chiama `pipeline.process_frame(frame, frame_index)` che restituisce
        sempre (annotated_frame, list[CollisionEvent]).

        Parameters
        ----------
        pipeline : TrafficAccidentPipeline
            Pipeline inizializzata.
        frame : Any
            Frame BGR come np.ndarray da cv2.VideoCapture.
        frame_index : int
            Indice progressivo del frame corrente (obbligatorio per
            remove_stale_tracks e per il logging degli eventi).

        Returns
        -------
        tuple[Any, list]
            (annotated_frame, collision_events):
              - annotated_frame: frame BGR con le annotazioni disegnate.
              - collision_events: lista di CollisionEvent (puo essere vuota).
        """
        if hasattr(pipeline, "process_frame"):
            # Chiamata alla firma corretta: (frame, frame_index).
            # La versione originale passava solo frame, causando TypeError.
            return pipeline.process_frame(frame, frame_index)

        if hasattr(pipeline, "process"):
            # Fallback per versioni alternative della pipeline.
            return pipeline.process(frame), []

        raise AttributeError("TrafficAccidentPipeline non ha un metodo di processing frame")

    def _extract_rendered_frame(self, fallback_frame: Any, output: Any) -> Any:
        """
        Estrae un frame visualizzabile dall'output della pipeline.

        Supporta output di tipo np.ndarray, tuple e dict per compatibilita
        con versioni passate della pipeline che restituivano formati diversi.

        Parameters
        ----------
        fallback_frame : Any
            Frame originale non annotato: usato come fallback se non si
            riesce ad estrarre un frame dall'output.
        output : Any
            Output restituito da process_frame().

        Returns
        -------
        Any
            Frame visualizzabile (np.ndarray con .shape e .dtype).
        """
        # Caso principale: output e direttamente un frame (np.ndarray).
        if self._is_frame_like(output):
            return output

        # Caso tuple: process_frame restituisce (annotated_frame, collisions).
        # Il primo elemento e sempre il frame annotato.
        if isinstance(output, tuple):
            for item in output:
                if self._is_frame_like(item):
                    return item

        # Caso dict: compatibilita con pipeline che restituivano un dizionario.
        if isinstance(output, dict):
            for key in ("rendered_frame", "output_frame", "annotated_frame", "frame", "image"):
                item = output.get(key)
                if self._is_frame_like(item):
                    return item

        # Fallback su attributi nominati dell'output.
        for attr_name in ("rendered_frame", "output_frame", "annotated_frame", "frame", "image"):
            if hasattr(output, attr_name):
                item = getattr(output, attr_name)
                if self._is_frame_like(item):
                    return item

        # Ultimo fallback: il frame originale non annotato.
        return fallback_frame

    def _cleanup_pipeline(self, pipeline: TrafficAccidentPipeline | None) -> None:
        """
        Rilascia le risorse opzionali della pipeline.

        Chiama il primo metodo di chiusura disponibile tra close(),
        release() e shutdown(), se presenti.

        Parameters
        ----------
        pipeline : TrafficAccidentPipeline | None
            Pipeline da chiudere. Se None, non fa nulla.
        """
        if pipeline is None:
            return

        # Tenta i metodi di chiusura piu comuni in ordine di preferenza.
        for method_name in ("close", "release", "shutdown"):
            method = getattr(pipeline, method_name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break

    def _destroy_cv_window(self) -> None:
        """
        Chiude la finestra OpenCV in modo sicuro.

        Prima tenta la chiusura mirata per nome, poi chiude tutte
        le finestre OpenCV come failsafe.
        """
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Aggiornamento thread-safe della GUI
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        """
        Aggiorna la label di stato in modo thread-safe.

        Usa self.after(0, ...) per schedulare l'aggiornamento nel thread
        principale della GUI (Tkinter non e thread-safe per accessi diretti
        da thread secondari).

        Parameters
        ----------
        text : str
            Testo da visualizzare nella label di stato (senza prefisso "Status:").
        """
        def update() -> None:
            if self.winfo_exists():
                self.status_label.configure(text=f"Status: {text}")

        try:
            self.after(0, update)
        except Exception:
            pass

    def _set_controls(self, is_running: bool) -> None:
        """
        Abilita o disabilita i pulsanti Start/Stop in modo thread-safe.

        Parameters
        ----------
        is_running : bool
            True se il processing e in corso (Start disabilitato, Stop abilitato).
            False se e fermo (Start abilitato, Stop disabilitato).
        """
        def update() -> None:
            if not self.winfo_exists():
                return
            self.start_button.configure(state="disabled" if is_running else "normal")
            self.stop_button.configure(state="normal" if is_running else "disabled")

        try:
            self.after(0, update)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Gestione chiusura finestra
    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        """
        Gestisce l'evento di chiusura della finestra (pulsante X).

        Segnala lo stop al worker e attende la sua terminazione prima
        di distruggere la finestra GUI, per evitare errori di accesso
        a widget gia distrutti.
        """
        self.stop_processing()
        self._wait_for_thread_and_close()

    def _wait_for_thread_and_close(self) -> None:
        """
        Attende la terminazione del thread worker prima di chiudere la GUI.

        Si ri-schedula ogni 100ms finche il thread e attivo. Una volta
        terminato, chiama self.destroy() per chiudere definitivamente
        la finestra e il mainloop.
        """
        if self._worker_thread is not None and self._worker_thread.is_alive():
            # Il thread e ancora attivo: controlla di nuovo tra 100ms.
            self.after(100, self._wait_for_thread_and_close)
            return

        # Thread terminato: distrugge la finestra in modo sicuro.
        self.destroy()

    # ------------------------------------------------------------------
    # Metodi statici di utilita
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_attr(obj: Any, name: str) -> str:
        """
        Legge un attributo come stringa in modo sicuro (non solleva eccezioni).

        Parameters
        ----------
        obj : Any
            Oggetto da cui leggere l'attributo.
        name : str
            Nome dell'attributo.

        Returns
        -------
        str
            Valore dell'attributo come stringa, oppure "" se l'attributo
            non esiste o ha un valore None.
        """
        value = getattr(obj, name, "")
        return value if isinstance(value, str) else str(value)

    @staticmethod
    def _is_frame_like(value: Any) -> bool:
        """
        Verifica se un valore sembra un frame immagine NumPy.

        Un array NumPy e riconoscibile dalla presenza simultanea degli
        attributi .shape e .dtype (entrambi assenti su tipi non-ndarray).

        Parameters
        ----------
        value : Any
            Valore da verificare.

        Returns
        -------
        bool
            True se il valore e un array NumPy-like (ha .shape e .dtype).
        """
        return hasattr(value, "shape") and hasattr(value, "dtype")


def main() -> None:
    """Avvia l'applicazione GUI e entra nel mainloop di Tkinter."""
    app = TrafficApp()
    app.mainloop()


if __name__ == "__main__":
    main()

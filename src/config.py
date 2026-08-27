"""
src/config.py
=============
Configurazione centralizzata dell'applicazione traffic-accident-seg.

Questo modulo definisce un'unica dataclass (AppConfig) che raccoglie tutti
i parametri di soglia e di rendering usati dall'intera pipeline.
Tutte le soglie sono raggruppate qui per garantire un'unica sorgente di verità
e rendere semplice la calibrazione senza dover toccare la logica di business.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AppConfig:
    """
    Parametri globali dell'applicazione.

    La classe è una normale dataclass (non frozen) perché ui_main.py crea
    copie aggiornate tramite dataclasses.replace() quando l'utente modifica
    i path dalla GUI.
    """

    # ------------------------------------------------------------------
    # Sorgenti dati
    # ------------------------------------------------------------------

    # Percorso del file video da analizzare (relativo alla CWD del processo).
    # Assicurarsi di lanciare lo script dalla root del progetto affinché
    # il path relativo sia risolto correttamente.
    video_path: str = "dataset/real_videos/Z4kg2Ev3vhk_00.mp4"

    # Percorso del modello YOLO segmenter (es. yolo26n-seg.pt).
    # Il file deve essere presente nella root del progetto o indicato
    # con un percorso assoluto.
    model_path: str = "yolo26n-seg.pt"

    # ------------------------------------------------------------------
    # Parametri del rilevatore YOLO + ByteTrack
    # ------------------------------------------------------------------

    # Classi COCO da rilevare (solo veicoli):
    #   2  = car
    #   3  = motorcycle
    #   5  = bus
    #   7  = truck
    # Modificare questa lista per aggiungere/escludere categorie.
    vehicle_classes: list[int] = field(default_factory=lambda: [2, 3, 5, 7])

    # Soglia minima di confidenza per accettare una detection (0.0 – 1.0).
    # Valori più bassi aumentano i rilevamenti ma introducono più falsi positivi.
    confidence: float = 0.35

    # Soglia IoU usata dall'NMS interno di YOLO per sopprimere bounding box
    # ridondanti sullo stesso oggetto (0.0 – 1.0).
    iou_threshold: float = 0.45

    # ------------------------------------------------------------------
    # Parametri di collisione e cinematica
    # ------------------------------------------------------------------

    # Numero minimo di pixel sovrapposti tra due maschere binarie affinché
    # la coppia di veicoli venga considerata in possibile collisione.
    # Unità: pixel (area dell'intersezione bit-a-bit delle maschere).
    # Abbassare questo valore rende il rilevamento più sensibile ma rumoroso.
    mask_overlap_threshold: int = 120

    # Velocità massima (in pixel/frame) al di sotto della quale un veicolo
    # viene considerato "fermo". Dipende dalla risoluzione e dal frame-rate
    # del video: calibrare se il video ha FPS o scala diversa dal default.
    stopped_speed_threshold: float = 2.5

    # Numero consecutivo di frame in cui un veicolo deve risultare "fermo"
    # prima che la condizione di stop venga confermata.
    # Valore basso (es. 3) è sensibile ma genera falsi positivi ai semafori;
    # alzarlo (es. 8–10) riduce i falsi ma aumenta la latenza del rilevamento.
    stopped_frames_threshold: int = 3

    # Soglia di decelerazione brusca (pixel/frame²) al di sotto della quale
    # scatta l'anomalia cinematica. Il valore è negativo perché rappresenta
    # una variazione negativa della velocità (frenata improvvisa).
    strong_deceleration_threshold: float = -4.0

    # Parametri temporali e spaziali del rilevatore di collisioni.
    motion_history_size: int = 15
    velocity_ema_alpha: float = 0.45
    max_kinematic_gap_frames: int = 2
    mask_overlap_ratio_threshold: float = 0.015
    contact_distance_threshold_px: float = 8.0
    mask_dilation_pixels: int = 3
    min_preimpact_speed_px: float = 3.0
    min_closing_speed_px: float = 1.5
    impact_window_frames: int = 5
    collision_confirmation_frames: int = 2
    collision_cooldown_frames: int = 30
    collision_display_frames: int = 15
    pair_state_ttl_frames: int = 45

    # Se valorizzato, la pipeline salva diagnostica per coppia ed eventi in
    # JSON Lines, utilizzabile dal modulo src.calibration.
    calibration_log_path: str = ""
    # Disattivabile nei benchmark estesi: i record per ogni coppia e frame
    # possono essere molto voluminosi; gli eventi restano sempre salvati.
    calibration_log_diagnostics: bool = True

    # ------------------------------------------------------------------
    # Opzioni di rendering
    # ------------------------------------------------------------------

    # Se True, sovrappone al frame le maschere di segmentazione colorate.
    draw_masks: bool = True

    # Se True, mostra l'ID univoco del track sopra ogni veicolo.
    draw_ids: bool = True

    # Se True, disegna il bounding box rettangolare attorno a ogni veicolo.
    # Disabilitato di default per non sovraffollare il video con le maschere.
    draw_bbox: bool = False

    # Opacità dell'overlay delle maschere di segmentazione (0.0 = trasparente,
    # 1.0 = completamente opaco). Valori tipici: 0.3 – 0.5.
    alpha_overlay: float = 0.4

    # ------------------------------------------------------------------
    # Gestione dei track "stale"
    # ------------------------------------------------------------------

    # Numero massimo di frame consecutivi in cui un track può non essere
    # rilevato prima di essere rimosso dallo store interno.
    # Aumentare se i veicoli vengono persi frequentemente per occlusioni.
    max_missing_frames: int = 10


def load_default_config() -> AppConfig:
    """
    Istanzia e restituisce la configurazione con i valori di default.

    Questa funzione factory viene importata da ui_main.py per popolare
    i widget della GUI con i valori predefiniti senza istanziare AppConfig
    direttamente nel layer di presentazione.

    Returns
    -------
    AppConfig
        Istanza con tutti i parametri impostati ai valori di default.
    """
    return AppConfig()

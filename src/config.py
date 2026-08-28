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

    # ------------------------------------------------------------------
    # Normalizzazione cinematica (scala apparente e frame rate)
    # ------------------------------------------------------------------
    # Tutte le soglie cinematiche sotto sono espresse in pixel/frame e restano
    # tarate su un veicolo di scala `reference_scale_px` in un video a
    # `reference_fps`. Con la normalizzazione attiva vengono riscalate a ogni
    # confronto sulla scala apparente del veicolo e sul frame rate reale:
    #   - una velocita in px/frame e proporzionale a  scala / fps
    #   - un'accelerazione in px/frame^2 e proporzionale a  scala / fps^2
    #   - una distanza in px e proporzionale a  scala
    #   - un'area in px^2 e proporzionale a  scala^2
    # Disattivando il flag il rilevatore torna esattamente al comportamento
    # precedente, utile per confrontare le due configurazioni sullo stesso log.
    #
    # DEFAULT OFF, e la ragione va letta prima di cambiarlo. Sul campione di 50
    # video la normalizzazione completa (magnitudini + finestre) e risultata una
    # regressione: 1 -> 0 veri positivi e 3 -> 5 falsi positivi. Le coppie che
    # collidono hanno scala apparente sopra la mediana del dataset (128 px
    # contro 90), quindi normalizzare gli ALZA le soglie invece di abbassarle.
    # La configurazione con le sole magnitudini non e ancora stata misurata:
    # attivarla solo dopo aver rieseguito il benchmark. Vedi
    # CRASH_DETECTION_PLAN.md, punto 2.
    kinematic_normalization_enabled: bool = False
    # La dilatazione delle finestre e corretta in teoria - una durata e una
    # durata - ma e la meta piu dannosa della normalizzazione: a 30 fps
    # raddoppia i frame di conferma richiesti, e i track coinvolti in un urto
    # vivono spesso pochi frame perche il tracker riassegna gli ID durante
    # l'occlusione. Resta separata dalle magnitudini per poterle misurare
    # indipendentemente.
    normalize_time_windows: bool = False
    # Mediana della diagonale delle bbox misurata sul dataset reale (90,7 px).
    reference_scale_px: float = 90.0
    # Mediana del frame rate del dataset reale (14,99 fps).
    reference_fps: float = 15.0
    # Frame rate del video in analisi. 0 significa sconosciuto: in quel caso
    # non viene applicata alcuna correzione temporale. La CLI e il benchmark lo
    # valorizzano con il valore restituito dal decoder.
    video_fps: float = 0.0
    # La scala apparente e filtrata piu della velocita: cambia lentamente e un
    # salto della maschera non deve alterare le soglie di un intero frame.
    scale_ema_alpha: float = 0.25

    # Parametri temporali e spaziali del rilevatore di collisioni.
    motion_history_size: int = 15
    velocity_ema_alpha: float = 0.45
    max_kinematic_gap_frames: int = 2
    mask_overlap_ratio_threshold: float = 0.015
    contact_distance_threshold_px: float = 8.0
    mask_dilation_pixels: int = 3
    min_preimpact_speed_px: float = 3.0
    # Numero minimo di campioni consecutivi sopra la velocita pre-impatto.
    # Evita che un singolo salto della maschera venga interpretato come moto.
    motion_confirmation_frames: int = 3
    min_closing_speed_px: float = 1.5
    # L'avvicinamento relativo deve essere continuo: un singolo picco dovuto
    # alla prospettiva o alla maschera non dimostra una traiettoria convergente.
    approach_confirmation_frames: int = 3
    approach_evidence_window_frames: int = 2
    impact_window_frames: int = 5
    # Memoria separata usata per stimare la direzione abituale del veicolo.
    # Non prolunga la validita di overlap o frenate isolate.
    trajectory_history_frames: int = 12
    trajectory_reaction_lag_frames: int = 2
    trajectory_min_displacement_px: float = 12.0
    trajectory_deflection_angle_deg: float = 45.0
    # Due direzioni sono considerate incrociate quando formano almeno questo
    # angolo (ignorando il verso opposto sulla stessa direttrice).
    crossing_history_frames: int = 6
    crossing_min_angle_deg: float = 45.0
    # Accelerazione positiva improvvisa del veicolo inizialmente piu lento:
    # rappresenta il trasferimento di moto tipico del veicolo colpito.
    target_impulse_acceleration_threshold: float = 8.0
    # Collega avvicinamento e doppio arresto attraverso una breve occlusione
    # del tracker, tipica del momento in cui due sagome entrano in collisione.
    crossing_dual_stop_bridge_frames: int = 12
    crossing_dual_stop_min_gap_frames: int = 2
    # Un veicolo e un bersaglio stabilmente fermo solo se la maggioranza dei
    # suoi campioni recenti e sotto la soglia di stop.
    stationary_history_frames: int = 10
    stationary_history_ratio: float = 0.75
    # Una reazione del veicolo in moto deve persistere: elimina frenate o
    # salti di maschera di un singolo frame durante le svolte all'incrocio.
    impact_reaction_confirmation_frames: int = 2
    # Un'anomalia comparsa molto dopo l'inizio di un overlap persistente non
    # viene attribuita allo stesso contatto (tipico del traffico affiancato).
    max_contact_candidate_age_frames: int = 5
    collision_confirmation_frames: int = 2
    collision_cooldown_frames: int = 30
    collision_display_frames: int = 15
    pair_state_ttl_frames: int = 45
    # Una coppia gia in contatto alla prima osservazione e disarmata finche
    # le sagome non risultano separate per questo numero di frame consecutivi.
    preexisting_contact_release_frames: int = 3
    # Il contatto e davvero preesistente solo se nessuno dei due track era gia
    # noto: una coppia nuova fra un track maturo e un ID appena creato nasce
    # quasi sempre da una riassegnazione del tracker durante l'occlusione
    # dell'urto, non da due veicoli gia accostati. Un track e considerato nuovo
    # finche non supera questo numero di frame osservati; con 0 nessun track e
    # nuovo e il disarmo non scatta mai.
    preexisting_contact_max_track_age_frames: int = 10
    # Il disarmo non e permanente: dopo questo numero di frame consecutivi la
    # coppia torna alle regole normali anche se le sagome non si sono mai
    # separate. Senza questo limite due veicoli che restano a contatto dopo un
    # urto non possono piu generare alcun evento. Con 0 il limite e disattivato.
    preexisting_contact_max_frames: int = 45

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

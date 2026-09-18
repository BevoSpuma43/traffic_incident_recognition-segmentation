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

    # ------------------------------------------------------------------
    # Punto 2b: coppie senza storia cinematica
    # ------------------------------------------------------------------
    # Misurando il punto 2 e emerso che il vero collo di bottiglia non sono le
    # soglie ma le coppie che nascono nell'istante dell'urto: in 5 dei 6 video
    # bloccati su `sustained_approach` la sequenza diagnostica della coppia
    # inizia esattamente al frame del primo contatto, perche il tracker
    # riassegna un ID durante l'occlusione dell'impatto. Su una coppia che
    # esiste da 1-3 frame nessun gate basato sulla persistenza e soddisfacibile.
    # Due rimedi indipendenti, entrambi disattivabili separatamente.

    # (a) Re-identificazione: quando un ID nuovo compare dove un altro e appena
    # sparito, il nuovo track eredita la storia cinematica del precedente
    # invece di ripartire da zero. Attacca il problema alla radice: rimette in
    # funzione tutti i gate storici (had_recent_motion, closing_speed,
    # _is_stably_stationary, traiettoria) invece di aggirarne uno solo.
    #
    # DEFAULT OFF perche misurata inefficace, non perche non sia stata provata.
    # Su 34 video scatta 44 volte senza spostare un solo stadio del funnel e
    # senza cambiare un solo evento: gli ID adottati sono track periferici.
    # Il motivo e nella geometria, non nella logica: il 59% delle nascite di ID
    # non ha alcun track sparito nei 5 frame precedenti (sono veicoli che
    # entrano in scena, non riassegnazioni), e fra le altre il donatore migliore
    # dista in mediana 2,19 diagonali di bbox. Non esiste una soglia che separi.
    # Alzarla comprerebbe copertura tirando a indovinare. Il codice resta perche
    # l'impalcatura e riusabile: a dover cambiare e la funzione di somiglianza,
    # che su questi dati richiede un descrittore d'aspetto e non la posizione.
    # Vedi CRASH_DETECTION_PLAN.md, punto 2b.
    track_reid_enabled: bool = False
    # Un ID puo ereditare solo da un track sparito da pochi frame: oltre questa
    # distanza l'associazione diventa una scommessa.
    track_reid_max_gap_frames: int = 5
    # Distanza massima fra la posizione prevista del track sparito e quella del
    # nuovo, espressa in frazioni della diagonale della bbox: in pixel assoluti
    # sarebbe di nuovo dipendente dalla prospettiva.
    track_reid_max_distance_scale: float = 0.8
    # Due sagome di dimensione molto diversa non sono lo stesso veicolo.
    track_reid_max_scale_ratio: float = 2.0

    # (b) Avvicinamento impulsivo: su una coppia appena nata l'intensita
    # sostituisce la persistenza. Un singolo frame con closing speed molto
    # superiore alla soglia vale come avvicinamento sostenuto, perche
    # `approach_confirmation_frames` non e soddisfacibile da una coppia che
    # esiste da meno frame di quanti ne richiede.
    #
    # DEFAULT ON, ma l'evidenza e sottile e va letta prima di farci affidamento.
    # Su 30 video stratificati recupera un vero positivo (t-bone notturno,
    # coppia corretta, +0,60 s dall'annotazione) senza aggiungere nessun falso
    # positivo: 1 -> 2 TP, 2 FP invariati. Sui 4 video di regressione non cambia
    # un solo evento pur attivandosi 52 volte. Il totale resta pero di 2 TP su
    # 30 video: e un miglioramento reale ma piccolo, misurato su un campione
    # dove il sistema emette pochissimo.
    #
    # La sua sicurezza e in prestito. Sulle 159 righe in cui scatta, il 97%
    # muore con la coppia in `status=clear` e solo il 7% ha contatto spaziale:
    # a filtrare non e questa condizione ma la congiunzione con contatto, moto
    # precedente ed evidenza dinamica. Se un intervento futuro indebolisce il
    # requisito di contatto, questa scorciatoia va rimisurata insieme a quello.
    impulse_approach_enabled: bool = True
    # Quante volte la soglia ordinaria deve valere il closing speed di un
    # singolo frame. Nei log un urto reale produce 16 px/f contro una soglia
    # di 1,5: il margine e ampio e la soglia non e delicata.
    impulse_approach_speed_multiplier: float = 4.0
    # Una coppia e "giovane" finche e stata osservata per meno di questi frame.
    # La scorciatoia vale solo li: su una coppia matura la persistenza e
    # disponibile e va richiesta.
    young_pair_max_frames: int = 3

    # ------------------------------------------------------------------
    # Punto 3: delta-V vettoriale
    # ------------------------------------------------------------------
    # `strong_deceleration_threshold` confronta la derivata della MAGNITUDINE
    # della velocita, quindi non vede un urto che devia un veicolo senza
    # rallentarlo: un t-bone a 90 gradi a modulo costante produce accelerazione
    # scalare zero. Il delta-V |v(t) - v(t-1)| e la grandezza fisica giusta,
    # quella che compare nella conservazione della quantita di moto, e viene
    # usato come evidenza dinamica ACCANTO alla decelerazione, non al suo posto.
    #
    # Misurato sul delta-V grezzo: nella finestra dell'incidente la
    # decelerazione scalare scatta su 4 campioni su 445, mentre |dv| >= 10 ne
    # prende 30, di cui 26 (87%) invisibili alla derivata del modulo.
    #
    # DEFAULT ON. Su 30 video recupera un vero positivo - un t-bone in cui il
    # delta-V e l'UNICA evidenza dinamica presente - senza aggiungere falsi
    # positivi, e non cambia nulla sui 4 casi di regressione: 2 -> 3 TP, 2 FP
    # invariati. Entrambi i TP recuperati fra punto 2b e punto 3 sono t-bone,
    # cioe la classe cieca per costruzione alla derivata del modulo.
    #
    # Controindicazione misurata: il delta-V scatta anche sui falsi positivi
    # esistenti e ne alza la confidenza (0,75 -> 0,90 su uno dei due). Non ne
    # crea di nuovi, ma rende quelli che ci sono meno separabili da una futura
    # soglia sul punteggio - cattiva notizia per il punto 5.
    delta_v_evidence_enabled: bool = True
    # Soglia in pixel/frame^2, tarata sulla distribuzione del delta-V grezzo e
    # non su quella ricostruita dai log filtrati, che avrebbe dato un valore
    # quasi doppio troppo basso. Il valore e il ginocchio della curva: fra 8 e
    # 10 l'arricchimento nella finestra passa da 2,5x a 3,4x e poi si appiattisce
    # attorno a 3,5x. Sotto i 10 si comprano campioni di fondo, sopra i 15 si
    # perde meta del segnale senza guadagnare selettivita.
    delta_v_impact_threshold_px: float = 10.0
    # DEFAULT OFF, misurato dannoso. L'idea era buona - un bersaglio deviato da
    # un t-bone ha delta-V alto ma spesso nessuna `trajectory_deflection`, che
    # pretende velocita e spostamento di base minimi - ma annulla la protezione
    # del punto 3.7 di PROJECT_STATUS: un veicolo che attraversa piu flussi ha
    # delta-V alto su OGNI coppia prospettica, quindi la perturbazione smette di
    # essere locale e torna a essere condivisa. Su `987C4_UdnJE_01` si torna
    # esattamente ai 5 eventi di prima della correzione e la precision sui casi
    # di regressione crolla da 0,60 a 0,38.
    delta_v_crossing_disruption_enabled: bool = False

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

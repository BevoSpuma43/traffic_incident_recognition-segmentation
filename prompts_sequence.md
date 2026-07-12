Usa questa impostazione: **un prompt base fisso**, sempre uguale, e poi **un prompt operativo per un solo file alla volta**. Così puoi aprire chat separate, incollare prima il blocco fisso e poi il task specifico, ottenendo una prima versione funzionante molto più facilmente.

Questa sequenza è pensata per arrivare in fretta a un **MVP coerente e avviabile**, con soglie iniziali già ragionate, import prevedibili e responsabilità dei moduli ben separate.

## Prompt base fisso

Incolla questo blocco **all’inizio di ogni richiesta**, sempre uguale.

```text
Contesto progetto:
Sto sviluppando un progetto Python chiamato "traffic-accident-seg" per rilevare incidenti stradali in tempo reale tramite instance segmentation, tracking e analisi pixel-level delle maschere.

Obiettivo MVP:
- leggere un video
- segmentare e tracciare i veicoli
- costruire una mask binaria per ogni veicolo
- aggiornare stato cinematico per track_id
- rilevare una possibile collisione tramite overlap tra maschere + anomalia cinematica
- renderizzare overlay, centroidi, ID e collisioni
- fornire una versione CLI funzionante subito
- aggiungere una GUI semplice separata

Stack obbligatorio:
- Python 3.11
- ultralytics
- opencv-python
- numpy
- customtkinter
- pillow
- pytest

Struttura progetto obbligatoria:
traffic-accident-seg/
├─ app.py
├─ requirements.txt
├─ README.md
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ geometry.py
│  ├─ kinematics.py
│  ├─ detector.py
│  ├─ tracker_state.py
│  ├─ collision_logic.py
│  ├─ renderer.py
│  ├─ video_pipeline.py
│  └─ ui_main.py
└─ tests/
   ├─ test_geometry.py
   ├─ test_kinematics.py
   └─ test_collision_logic.py

Vincoli architetturali:
- architettura multi-file ordinata dentro src/
- target CPU x86
- niente librerie extra non richieste
- type hints obbligatori
- docstring brevi
- codice semplice, robusto e modulare
- niente file monolitici
- niente logica duplicata
- separare bene detection, geometria, cinematica, collision logic, rendering, pipeline, UI
- ogni risposta deve generare SOLO il file richiesto
- non aggiungere spiegazioni fuori dal codice
- non usare markdown
- non usare triple backticks

Assunzioni operative iniziali:
- classi veicoli COCO: [2, 3, 5, 7]
- confidence: 0.35
- iou_threshold: 0.45
- mask_overlap_threshold: 120
- stopped_speed_threshold: 2.5
- stopped_frames_threshold: 3
- strong_deceleration_threshold: -4.0
- alpha_overlay: 0.4
- max_missing_frames: 10

Interpretazione delle soglie:
- speed e acceleration sono in pixel/frame e pixel/frame^2
- mask_overlap_threshold è in pixel condivisi
- i valori devono essere messi in config.py come default iniziali, facili da modificare

Dipendenze tra moduli da rispettare:
- src/config.py: solo standard library
- src/models.py: standard library + numpy
- src/geometry.py: numpy + cv2
- src/kinematics.py: standard library
- src/detector.py: ultralytics + numpy + src.models + src.geometry
- src/tracker_state.py: src.models + src.geometry + src.kinematics
- src/collision_logic.py: src.models + src.geometry
- src/renderer.py: cv2 + numpy + src.models
- src/video_pipeline.py: src.config + src.detector + src.tracker_state + src.collision_logic + src.renderer + src.models
- src/ui_main.py: customtkinter + threading + cv2 opzionale + src.config + src.video_pipeline
- app.py: cv2 + src.config + src.video_pipeline
- tests/*: pytest + moduli del progetto

Regole sui modelli dati:
- usare dataclass dove appropriato
- Point2D per centroidi
- DetectionResult per output del detector
- VehicleState per stato dinamico dei track
- CollisionEvent per evento di crash

Regole di output obbligatorie:
1. prima riga: FILE: <percorso_file>
2. dalla seconda riga in poi: SOLO il contenuto del file
3. non aggiungere testo prima o dopo
4. se qualcosa manca, fai l’assunzione più semplice e robusta possibile
5. non introdurre librerie non richieste
6. non accorpare più moduli in un solo file
```

## Ordine pratico consigliato

Questa è la sequenza migliore per ottenere presto una versione che gira davvero:

1. `requirements.txt`
2. `src/__init__.py`
3. `src/config.py`
4. `src/models.py`
5. `src/geometry.py`
6. `src/kinematics.py`
7. `src/detector.py`
8. `src/tracker_state.py`
9. `src/collision_logic.py`
10. `src/renderer.py`
11. `src/video_pipeline.py`
12. `app.py`
13. `src/ui_main.py`
14. `tests/test_geometry.py`
15. `tests/test_kinematics.py`
16. `tests/test_collision_logic.py`
17. `README.md`

Sotto trovi i prompt già pronti.

## 1) requirements.txt

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file requirements.txt con solo le dipendenze minime necessarie per il progetto.

Percorso file: requirements.txt
```

## 2) src/__init__.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera un file minimale e valido per inizializzare il package Python.

Percorso file: src/__init__.py
```

## 3) src/config.py

Questo prompt è importante perché stabilisce i default dell’intero MVP.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/config.py.

Requisiti:
- usare dataclass
- definire AppConfig
- includere questi campi:
  - video_path: str = "dataset/autostrada.mp4"
  - model_path: str = "yolo11n-seg.pt"
  - vehicle_classes: list[int] = [2, 3, 5, 7]
  - confidence: float = 0.35
  - iou_threshold: float = 0.45
  - mask_overlap_threshold: int = 120
  - stopped_speed_threshold: float = 2.5
  - stopped_frames_threshold: int = 3
  - strong_deceleration_threshold: float = -4.0
  - draw_masks: bool = True
  - draw_ids: bool = True
  - draw_bbox: bool = False
  - alpha_overlay: float = 0.4
  - max_missing_frames: int = 10
- aggiungere una funzione load_default_config() -> AppConfig
- usare solo standard library
- docstring brevi

Percorso file: src/config.py
```

## 4) src/models.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/models.py.

Requisiti:
- importare numpy come np
- usare dataclass
- definire:

1. Point2D
- x: int
- y: int

2. DetectionResult
- track_id: int
- class_id: int
- class_name: str
- polygon: np.ndarray
- mask: np.ndarray
- bbox: tuple[int, int, int, int]

3. VehicleState
- track_id: int
- class_id: int
- class_name: str
- centroid: Point2D | None
- prev_centroid: Point2D | None
- speed_px: float
- prev_speed_px: float
- acceleration_px: float
- polygon: np.ndarray | None
- mask: np.ndarray | None
- bbox: tuple[int, int, int, int] | None
- last_seen_frame: int
- stopped_frames: int

4. CollisionEvent
- frame_index: int
- vehicle_a: int
- vehicle_b: int
- overlap_area: int
- speed_a: float
- speed_b: float
- acceleration_a: float
- acceleration_b: float
- reason: str

- solo strutture dati, niente logica complessa
- type hints completi
- docstring brevi

Percorso file: src/models.py
```

## 5) src/geometry.py

Questo è uno dei file più critici del progetto.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/geometry.py.

Requisiti:
- importare cv2 e numpy
- implementare le seguenti funzioni:

1. safe_polygon_array(polygon) -> np.ndarray | None
- converte l'input in un array numpy int32 di shape compatibile con OpenCV
- se input è None, vuoto o invalido ritorna None

2. polygon_to_binary_mask(frame_shape, polygon) -> np.ndarray
- crea una mask binaria 2D uint8 con valori 0 o 255
- usa fillPoly
- se il poligono è invalido, ritorna una mask vuota della giusta shape

3. compute_centroid_from_polygon(polygon) -> tuple[int, int] | None
- usa cv2.moments
- se area nulla o poligono invalido ritorna None

4. compute_bbox_from_polygon(polygon) -> tuple[int, int, int, int]
- ritorna x_min, y_min, x_max, y_max
- se poligono invalido ritorna 0, 0, 0, 0

5. mask_intersection_area(mask_a, mask_b) -> int
- usa operazione bitwise and
- conta i pixel > 0 in intersezione
- se shape incompatibili o None ritorna 0

6. masks_touch(mask_a, mask_b) -> bool
- ritorna True se area di intersezione > 0

- codice robusto e semplice
- nessuna dipendenza extra oltre numpy e cv2

Percorso file: src/geometry.py
```

## 6) src/kinematics.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/kinematics.py.

Requisiti:
- usare solo standard library
- implementare:

1. compute_speed_px(curr_centroid, prev_centroid) -> float
- se uno dei due centroidi è None, ritorna 0.0
- distanza euclidea in pixel/frame

2. compute_acceleration(curr_speed, prev_speed) -> float
- ritorna curr_speed - prev_speed

3. update_stopped_counter(prev_counter, speed_px, stopped_speed_threshold) -> int
- incrementa il contatore se speed_px <= stopped_speed_threshold
- altrimenti azzera a 0

- funzioni pure, semplici e testabili
- type hints completi

Percorso file: src/kinematics.py
```

## 7) src/detector.py

Qui conviene chiedere esplicitamente una prima implementazione robusta ma semplice, senza complicare troppo con casi rari.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/detector.py.

Requisiti:
- importare:
  - numpy as np
  - ultralytics.YOLO
  - DetectionResult da src.models
  - polygon_to_binary_mask, compute_bbox_from_polygon, safe_polygon_array da src.geometry
- definire la classe VehicleSegmenter

Metodi richiesti:
1. __init__(self, model_path: str, confidence: float = 0.35, iou_threshold: float = 0.45)
- carica il modello YOLO

2. segment_and_track(self, frame, vehicle_classes: list[int]) -> list[DetectionResult]
- usa self.model.track con:
  - source=frame
  - persist=True
  - tracker="bytetrack.yaml"
  - classes=vehicle_classes
  - verbose=False
  - conf=self.confidence
  - iou=self.iou_threshold
- gestisce bene il caso di nessun risultato
- usa result = results[0]
- se mancano masks o boxes, ritorna lista vuota
- estrae track_id da result.boxes.id se disponibile
- estrae class_id e class_name
- per ogni istanza:
  - prende polygon da result.masks.xy
  - normalizza polygon con safe_polygon_array
  - costruisce la mask binaria con polygon_to_binary_mask
  - costruisce bbox con compute_bbox_from_polygon
  - crea un DetectionResult
- salta istanze invalide senza fallire
- ritorna una lista di DetectionResult

- niente GUI
- niente logica di collisione
- codice pulito e robusto

Percorso file: src/detector.py
```

## 8) src/tracker_state.py

Questo prompt deve guidare bene l’aggiornamento dello stato.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/tracker_state.py.

Requisiti:
- importare:
  - Point2D, DetectionResult, VehicleState da src.models
  - compute_centroid_from_polygon da src.geometry
  - compute_speed_px, compute_acceleration, update_stopped_counter da src.kinematics
- definire la classe VehicleStateStore
- usare self._states: dict[int, VehicleState]

Metodi richiesti:
1. __init__(self)
- inizializza il dizionario degli stati

2. update(self, detections: list[DetectionResult], frame_index: int, stopped_speed_threshold: float) -> list[VehicleState]
- per ogni detection:
  - calcola centroid dal polygon
  - se il track_id esiste, aggiorna prev_centroid, prev_speed_px, speed_px, acceleration_px, polygon, mask, bbox, last_seen_frame
  - aggiorna stopped_frames usando update_stopped_counter
  - se il track_id non esiste, crea un nuovo VehicleState con valori iniziali sensati
- ritorna la lista degli stati aggiornati relativi alle detection del frame corrente

3. get_active_states(self) -> list[VehicleState]
- ritorna tutti gli stati correnti

4. remove_stale_tracks(self, frame_index: int, max_missing_frames: int) -> None
- rimuove i track non visti da più di max_missing_frames

- codice semplice e molto leggibile
- niente dipendenze extra

Percorso file: src/tracker_state.py
```

## 9) src/collision_logic.py

Qui conviene rendere la logica esplicita e modulare.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/collision_logic.py.

Requisiti:
- importare:
  - VehicleState, CollisionEvent da src.models
  - mask_intersection_area da src.geometry
- definire la classe CollisionDetector

Costruttore:
- __init__(
    self,
    mask_overlap_threshold: int,
    stopped_frames_threshold: int,
    stopped_speed_threshold: float,
    strong_deceleration_threshold: float
  )

Metodi pubblici:
1. detect_collisions(self, vehicle_states: list[VehicleState], frame_index: int) -> list[CollisionEvent]
- confronta ogni coppia una sola volta
- evita duplicati A-B / B-A
- se mask mancanti, salta la coppia
- calcola overlap_area
- genera CollisionEvent solo se:
  - overlap_area >= mask_overlap_threshold
  - e inoltre vale almeno una delle due:
    a) entrambi i veicoli hanno stopped_frames >= stopped_frames_threshold
    b) almeno uno dei due ha acceleration_px <= strong_deceleration_threshold
- reason deve spiegare il motivo, ad esempio:
  - "overlap_and_dual_stop"
  - "overlap_and_hard_deceleration"

Metodi privati richiesti:
- _pair_key(id_a: int, id_b: int) -> tuple[int, int]
- _has_overlap(vehicle_a: VehicleState, vehicle_b: VehicleState) -> tuple[bool, int]
- _both_stopped(vehicle_a: VehicleState, vehicle_b: VehicleState) -> bool
- _hard_deceleration(vehicle_a: VehicleState, vehicle_b: VehicleState) -> bool

- implementazione prudente, semplice, testabile
- niente GUI
- niente dipendenze extra

Percorso file: src/collision_logic.py
```

## 10) src/renderer.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/renderer.py.

Requisiti:
- importare cv2, numpy
- importare VehicleState e CollisionEvent da src.models

Implementare:
1. get_color_for_track(track_id: int) -> tuple[int, int, int]
- colore stabile e deterministico basato sul track_id

2. draw_mask_overlay(frame, vehicle_states, alpha_overlay: float) -> np.ndarray
- crea overlay con maschere colorate
- usa mask > 0 per colorare
- usa cv2.addWeighted

3. draw_centroids_and_ids(frame, vehicle_states, draw_ids: bool = True) -> np.ndarray
- disegna centroide se presente
- disegna testo ID se draw_ids è True

4. draw_bboxes(frame, vehicle_states) -> np.ndarray
- disegna bbox se disponibile

5. draw_collisions(frame, collisions, vehicle_states) -> np.ndarray
- evidenzia i veicoli coinvolti
- scrive una label tipo "CRASH?" vicino alla bbox o al centroide
- recupera i VehicleState tramite track_id

6. render_frame(frame, vehicle_states, collisions, alpha_overlay: float, draw_masks: bool = True, draw_ids: bool = True, draw_bbox: bool = False) -> np.ndarray
- esegue le varie fasi di disegno in ordine corretto
- ritorna il frame annotato finale

- codice semplice e robusto
- non modificare il frame originale in modo distruttivo se evitabile

Percorso file: src/renderer.py
```

## 11) src/video_pipeline.py

Questo deve diventare il centro del progetto.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/video_pipeline.py.

Requisiti:
- importare numpy as np
- importare:
  - AppConfig da src.config
  - CollisionEvent da src.models
  - VehicleSegmenter da src.detector
  - VehicleStateStore da src.tracker_state
  - CollisionDetector da src.collision_logic
  - render_frame da src.renderer
- definire la classe TrafficAccidentPipeline

Metodi richiesti:
1. __init__(self, config: AppConfig)
- istanzia:
  - self.config
  - self.segmenter
  - self.state_store
  - self.collision_detector

2. process_frame(self, frame, frame_index: int) -> tuple[np.ndarray, list[CollisionEvent]]
- esegue:
  1. detections = segmenter.segment_and_track(...)
  2. active_states = state_store.update(...)
  3. state_store.remove_stale_tracks(...)
  4. collisions = collision_detector.detect_collisions(active_states, frame_index)
  5. annotated = render_frame(...)
- ritorna annotated, collisions

- design pulito
- niente GUI
- niente logica duplicata

Percorso file: src/video_pipeline.py
```

## 12) app.py

Questo deve produrre una CLI avviabile subito.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file app.py.

Requisiti:
- importare cv2
- importare load_default_config da src.config
- importare TrafficAccidentPipeline da src.video_pipeline

Implementare:
1. funzione main()
- carica config = load_default_config()
- apre il video con cv2.VideoCapture(config.video_path)
- se non si apre, stampa errore e termina
- crea pipeline = TrafficAccidentPipeline(config)
- legge i frame in loop
- per ogni frame:
  - chiama process_frame(frame, frame_index)
  - mostra il risultato con cv2.imshow
  - stampa gli eventi di collisione in console in modo leggibile
  - incrementa frame_index
- uscita con tasto q
- release e destroyAllWindows in modo corretto

2. blocco:
if __name__ == "__main__":
    main()

- implementazione semplice e robusta
- pronta per una prima esecuzione reale

Percorso file: app.py
```

## 13) src/ui_main.py

Per la prima versione, meglio una GUI di controllo semplice, non una GUI complessa con embedded video.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file src/ui_main.py.

Requisiti:
- importare customtkinter come ctk
- usare threading
- importare cv2
- importare load_default_config, AppConfig da src.config
- importare TrafficAccidentPipeline da src.video_pipeline

Obiettivo:
- GUI semplice di controllo, non serve preview del video dentro la finestra
- la GUI può aprire una finestra OpenCV separata durante l'elaborazione

Implementare:
1. classe TrafficApp(ctk.CTk)
- campi input per:
  - video path
  - model path
- pulsanti:
  - Start
  - Stop
- label di stato
- flag interno per stop thread

2. metodo per costruire AppConfig partendo dai campi GUI
- riusa i default di load_default_config
- sovrascrive solo video_path e model_path

3. metodo start_processing()
- avvia un thread se non già attivo

4. metodo stop_processing()
- segnala stop e aggiorna stato

5. metodo worker thread
- apre il video
- crea TrafficAccidentPipeline
- processa frame in loop
- mostra risultato in finestra OpenCV
- termina se richiesto stop o se si preme q
- rilascia correttamente le risorse

6. funzione main() che lancia la GUI

7. blocco if __name__ == "__main__": main()

- niente duplicazione della business logic della pipeline
- implementazione prudente e lineare

Percorso file: src/ui_main.py
```

## 14) tests/test_geometry.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file tests/test_geometry.py.

Requisiti:
- usare pytest e numpy
- importare funzioni da src.geometry
- scrivere test semplici e leggibili per:
  - safe_polygon_array
  - polygon_to_binary_mask
  - compute_centroid_from_polygon
  - compute_bbox_from_polygon
  - mask_intersection_area
  - masks_touch
- usare poligoni sintetici piccoli e facili da capire
- nessuna dipendenza extra

Percorso file: tests/test_geometry.py
```

## 15) tests/test_kinematics.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file tests/test_kinematics.py.

Requisiti:
- usare pytest
- importare funzioni da src.kinematics
- scrivere test semplici e deterministici per:
  - compute_speed_px
  - compute_acceleration
  - update_stopped_counter
- includere almeno:
  - caso con centroidi None
  - caso con distanza nota tipo 3-4-5
  - caso di reset del contatore di stop

Percorso file: tests/test_kinematics.py
```

## 16) tests/test_collision_logic.py

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file tests/test_collision_logic.py.

Requisiti:
- usare pytest e numpy
- importare CollisionDetector da src.collision_logic
- importare Point2D, VehicleState da src.models

Richieste:
- creare una piccola funzione helper locale per costruire VehicleState sintetici
- creare mask binarie sintetiche con overlap controllato
- testare almeno:
  1. collisione rilevata con overlap sufficiente + entrambi fermi
  2. collisione rilevata con overlap sufficiente + forte decelerazione
  3. nessuna collisione senza overlap
  4. nessun duplicato A-B / B-A
- test leggibili e stabili

Percorso file: tests/test_collision_logic.py
```

## 17) README.md

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Genera il file README.md.

Requisiti:
- descrivere il progetto in tono accademico ma chiaro
- includere:
  - titolo
  - obiettivo
  - struttura del progetto
  - stack tecnologico
  - installazione
  - esecuzione CLI
  - esecuzione GUI
  - esecuzione test
  - spiegazione sintetica della logica di rilevamento collisione
  - note sui limiti del prototipo e sulla taratura delle soglie
- markdown pulito
- niente testo superfluo

Percorso file: README.md
```

## Prompt di correzione universale

Quando un file generato non funziona, invece di chiedere “riscrivilo meglio”, usa questo prompt. È molto più efficace con modelli piccoli.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Rigenera il file <PERCORSO_FILE> correggendo solo i problemi seguenti:

- <ERRORE 1>
- <ERRORE 2>
- <ERRORE 3>

Vincoli:
- mantieni invariata l'API pubblica del file se possibile
- non cambiare nomi di classi o funzioni già definiti
- non introdurre nuove dipendenze
- correggi il minimo necessario per ottenere una versione funzionante

Percorso file: <PERCORSO_FILE>
```

Esempio pratico:

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Rigenera il file src/detector.py correggendo solo i problemi seguenti:

- gestisci il caso result.boxes.id is None senza fallire
- salta le istanze con polygon invalido
- assicurati che la mask binaria abbia la shape del frame

Vincoli:
- mantieni invariata l'API pubblica del file se possibile
- non cambiare nomi di classi o funzioni già definiti
- non introdurre nuove dipendenze
- correggi il minimo necessario per ottenere una versione funzionante

Percorso file: src/detector.py
```

## Prompt di verifica architetturale

Prima di generare gli ultimi file, puoi fare anche questo passaggio di controllo.

```text
[INCOLLA PRIMA IL PROMPT BASE FISSO]

Senza scrivere codice, verifica la coerenza architetturale del progetto.

Output richiesto:
- per ogni file del progetto, elenca solo gli import che dovrebbero comparire
- segnala eventuali dipendenze circolari da evitare
- non scrivere implementazioni
- non aggiungere testo superfluo
```

## Strategia migliore per ottenere subito una prima versione funzionante

In pratica, la procedura più efficace è questa:

1. genera fino a `app.py`
2. crea davvero i file localmente
3. esegui:
   ```bash
   pip install -r requirements.txt
   python app.py
   ```
4. se hai errori, usa il **prompt di correzione universale** passando il traceback
5. solo dopo passa a GUI, test e README

Il motivo è semplice: per un MVP di computer vision, la priorità non è la perfezione teorica ma ottenere presto una pipeline che:
- apra il video,
- segmenti i veicoli,
- aggiorni lo stato,
- segnali collisioni candidate,
- disegni tutto a schermo.

## Valori iniziali sensati delle soglie

Questi sono già incorporati nel prompt base, ma ti esplicito perché sono buoni per la prima prova:

- `confidence = 0.35`: abbastanza prudente, evita troppe detection deboli.
- `iou_threshold = 0.45`: equilibrato per tracking/inferenza.
- `mask_overlap_threshold = 120`: evita falsi contatti dovuti a pochi pixel rumorosi.
- `stopped_speed_threshold = 2.5`: ragionevole come “quasi fermo” in pixel/frame su video standard.
- `stopped_frames_threshold = 3`: evita di scambiare un rallentamento istantaneo per crash.
- `strong_deceleration_threshold = -4.0`: buona soglia iniziale per evidenziare frenate o impatti apparenti.
- `max_missing_frames = 10`: abbastanza permissivo per piccole perdite del tracker.

Sono soglie da prototipo, quindi poi andranno calibrate sul tuo dataset, ma per partire sono coerenti e difendibili anche in sede d’esame.

## Sequenza minima super-operativa

Se vuoi andare ancora più veloce, usa solo questa mini-sequenza iniziale:

1. `requirements.txt`
2. `src/__init__.py`
3. `src/config.py`
4. `src/models.py`
5. `src/geometry.py`
6. `src/kinematics.py`
7. `src/detector.py`
8. `src/tracker_state.py`
9. `src/collision_logic.py`
10. `src/renderer.py`
11. `src/video_pipeline.py`
12. `app.py`

Con questi 12 prompt dovresti già ottenere la **prima versione funzionante**. GUI, test e README vengono bene dopo.

In sintesi, la chiave è: **stesso prompt base, un solo file per chat, output rigidissimo, soglie già definite, dipendenze già pensate**. È il modo più robusto per usare modelli piccoli senza farli deragliare.

Se vuoi, nel prossimo messaggio posso fare un passo ancora più pratico: prepararti una **versione “ultra-compatta” dei prompt**, più corta da copiare, ma che conserva quasi tutta l’efficacia operativa.

A proposito, Image Generation mode è attiva — se vuoi posso creare uno schema visivo dell’architettura software o del flusso `segmentazione -> tracking -> overlap -> crash detection`.
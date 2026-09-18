# Piano di implementazione — Incident Detection da CCTV

Versione: 1.0  
Obiettivo: realizzare un sistema locale e leggero che rilevi possibili incidenti stradali in video CCTV usando segmentazione delle istanze, tracking, omografia del piano stradale e analisi temporale delle traiettorie.

---

## 1. Risultato finale atteso

Il sistema deve:

1. Leggere un video MP4 e, opzionalmente, un flusso RTSP.
2. Segmentare separatamente automobili, camion, autobus e motocicli.
3. Assegnare un identificatore persistente a ogni veicolo.
4. Stimare il punto di contatto di ogni veicolo con la strada.
5. Trasformare tale punto in coordinate bird's-eye mediante omografia.
6. Calcolare traiettorie, velocità, accelerazioni e relazioni tra veicoli.
7. Generare un punteggio di incidente su una finestra temporale.
8. Confermare o rifiutare il possibile incidente mediante evidenze successive.
9. Mostrare a video maschere, ID, traiettorie, vista dall'alto e punteggio.
10. Salvare timestamp, veicoli coinvolti e una clip dell'evento.
11. Funzionare interamente in locale su una singola telecamera alla volta.

### Criteri generali di successo

- Elaborazione minima utile: 5–10 FPS sulla macchina target.
- Nessuna crescita non limitata della memoria durante video lunghi.
- Misure e derivate temporali basate sui timestamp, non sul numero nominale di frame.
- Valutazione per evento, non soltanto per singolo frame.
- Presenza di un fallback manuale quando la calibrazione automatica non è affidabile.
- Codice, configurazioni, pesi e risultati riproducibili.

---

## 2. Perimetro del progetto

### Funzionalità obbligatorie — MVP

- Video preregistrato.
- Una telecamera fissa.
- Segmentazione dei veicoli.
- ByteTrack.
- Omografia impostata manualmente con quattro o più corrispondenze.
- Estrazione di traiettorie sul piano stradale.
- Rilevamento mediante regole temporali.
- Salvataggio e visualizzazione degli eventi.
- Misurazione di precisione, recall, falsi allarmi per ora e velocità di esecuzione.

### Estensioni principali

- Individuazione automatica delle strisce.
- Stima automatica o assistita dell'omografia.
- Classificatore leggero addestrato sulle feature delle traiettorie.
- Quantizzazione INT8 o inferenza FP16.
- Lettura RTSP.

### Fuori perimetro

- Invio automatico di chiamate o segnalazioni di emergenza.
- Garanzia di funzionamento su qualsiasi telecamera senza calibrazione.
- Fusione di più telecamere.
- Ricostruzione 3D completa della scena.
- Anticipazione dell'incidente molti secondi prima dell'impatto.
- Uso di Video Transformer pesanti in tempo reale.

---

## 3. Architettura logica

### Calibrazione, eseguita all'avvio o dopo lo spostamento della camera

1. Campionamento di frame.
2. Costruzione dello sfondo mediano.
3. Estrazione della regione stradale e delle strisce.
4. Skeletonization della maschera.
5. Rilevamento e raggruppamento robusto delle linee.
6. Stima dei punti di fuga.
7. Costruzione delle corrispondenze strada–bird's-eye.
8. Calcolo dell'omografia.
9. Valutazione della confidenza.
10. Salvataggio della calibrazione oppure apertura del fallback manuale.

### Pipeline continua

1. Decode del frame e lettura del timestamp.
2. Segmentazione delle istanze.
3. Tracking.
4. Estrazione del punto a terra dalla maschera.
5. Trasformazione del punto mediante omografia.
6. Aggiornamento e filtraggio delle traiettorie.
7. Calcolo delle feature per singolo veicolo e per coppia.
8. Aggiornamento del detector temporale.
9. Conferma dell'evento.
10. Aggiornamento della UI e salvataggio della clip.

---

## 4. Stack tecnologico

| Area | Scelta principale | Note |
| --- | --- | --- |
| Linguaggio | Python 3.11 | Ampio supporto per PyTorch, OpenCV e runtime |
| Gestione ambiente | uv oppure venv + pip | Usare un lockfile |
| Video | PyAV/FFmpeg | Timestamp e RTSP più affidabili |
| Computer vision | OpenCV, NumPy | Maschere, linee, omografia e rendering |
| Elaborazione immagini | SciPy, scikit-image | Filtri e skeletonization |
| Segmentazione | YOLO26n-seg | Modello nano, da validare e fine-tunare |
| Tracking | ByteTrack | Basso overhead, adatto a camera fissa |
| Training | PyTorch + Ultralytics | Solo sviluppo e addestramento |
| Classificazione evento | scikit-learn | HistGradientBoosting come prima scelta |
| Inference Intel | OpenVINO | CPU, iGPU e possibile INT8 |
| Inference NVIDIA | TensorRT | Preferibilmente FP16 |
| Inference generica | ONNX Runtime | Fallback portabile |
| Annotazione | CVAT | Installabile localmente |
| Demo | Streamlit | Interfaccia rapida per l'esame |
| Configurazione | YAML + Pydantic | Parametri validati |
| Persistenza | SQLite + JSON + MP4 | Nessun server esterno |
| Test | pytest | Unit, integration e regression test |
| Qualità codice | Ruff | Lint e formatting |

### Nota sulla licenza

Ultralytics e i suoi modelli sono distribuiti con licenza AGPL-3.0, salvo licenza enterprise. Per un progetto universitario pubblicato integralmente come open source è normalmente una scelta praticabile. Se il repository deve restare proprietario, valutare MMDetection con RTMDet-Ins-tiny e verificare separatamente la licenza dei pesi e dei dataset.

---

## 5. Struttura consigliata del repository

~~~text
cctv-accident-detection/
├── README.md
├── pyproject.toml
├── uv.lock
├── configs/
│   ├── default.yaml
│   ├── cameras/
│   │   └── camera_01.yaml
│   ├── bytetrack.yaml
│   └── vehicles.yaml
├── data/
│   ├── raw/
│   ├── annotations/
│   ├── splits/
│   ├── calibration/
│   ├── processed/
│   └── samples/
├── models/
│   ├── training/
│   └── exported/
├── notebooks/
│   ├── 01_dataset_analysis.ipynb
│   ├── 02_calibration_analysis.ipynb
│   └── 03_event_features.ipynb
├── src/
│   └── cctv_incident/
│       ├── __init__.py
│       ├── config.py
│       ├── video.py
│       ├── segmenter.py
│       ├── tracker.py
│       ├── ground_point.py
│       ├── calibration/
│       │   ├── manual.py
│       │   ├── background.py
│       │   ├── lane_mask.py
│       │   ├── line_fitting.py
│       │   ├── homography.py
│       │   └── quality.py
│       ├── trajectories.py
│       ├── features.py
│       ├── event_detector.py
│       ├── clip_buffer.py
│       ├── storage.py
│       ├── visualization.py
│       └── pipeline.py
├── apps/
│   └── streamlit_app.py
├── scripts/
│   ├── prepare_dataset.py
│   ├── train_segmenter.py
│   ├── export_model.py
│   ├── calibrate_camera.py
│   ├── extract_features.py
│   ├── train_event_classifier.py
│   ├── evaluate.py
│   └── benchmark.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── outputs/
│   ├── events/
│   ├── clips/
│   ├── metrics/
│   └── debug/
└── docs/
    ├── architecture.md
    ├── dataset-card.md
    └── experiment-log.md
~~~

Non inserire nel repository pubblico i video per i quali non esiste un diritto di redistribuzione. Documentare invece la procedura per ottenerli.

---

## 6. Fase 0 — Bloccare obiettivi e protocollo sperimentale

### Attività

- [ ] Definire se il sistema deve rilevare il momento dell'impatto oppure anche anticiparlo.
- [ ] Limitare la prima versione a una telecamera fissa.
- [ ] Scegliere una risoluzione target iniziale: 512 o 640 px per il lato di input.
- [ ] Definire un target di elaborazione: inizialmente 5–10 FPS.
- [ ] Definire le classi veicolo: car, truck, bus, motorcycle; opzionali bicycle e person.
- [ ] Definire cosa costituisce un evento vero positivo.
- [ ] Stabilire la tolleranza temporale, ad esempio una finestra attorno all'impatto annotato.
- [ ] Scegliere le metriche prima di osservare i risultati finali.
- [ ] Definire la macchina target e registrare CPU, GPU, RAM, sistema operativo e versione dei driver.

### Definizione operativa suggerita

Un incidente è confermato quando due o più tracce mostrano una collisione spazialmente plausibile e almeno un segnale post-impatto, oppure quando un singolo veicolo mostra un cambiamento improvviso compatibile con un urto seguito da arresto o traiettoria anomala.

### Output

- <code>docs/architecture.md</code> con il problema definito.
- Una tabella con metriche e target.
- Scheda della macchina target.

### Criterio di completamento

Il sistema può essere valutato con una definizione non ambigua di evento, input, output e hardware.

---

## 7. Fase 1 — Creare l'ambiente riproducibile

### Attività

1. Inizializzare il repository Git.
2. Creare l'ambiente Python.
3. Installare soltanto un backend di inference per volta.
4. Bloccare le versioni nel lockfile.
5. Verificare OpenCV, PyTorch e il dispositivo disponibile.

### Comandi indicativi

~~~bash
uv init --python 3.11
uv add av opencv-python numpy scipy scikit-image scikit-learn pyyaml pydantic streamlit
uv add ultralytics torch torchvision
uv add --dev pytest ruff
~~~

Per macchina Intel:

~~~bash
uv add openvino
~~~

Per fallback ONNX:

~~~bash
uv add onnx onnxruntime
~~~

Su NVIDIA installare la combinazione di CUDA e TensorRT compatibile con driver e sistema operativo, seguendo la documentazione ufficiale. Non mescolare casualmente versioni CUDA, PyTorch e TensorRT.

### Smoke test

- Aprire un'immagine.
- Eseguire una prediction con il modello preaddestrato.
- Stampare forma e numero delle maschere.
- Verificare che il backend selezionato usi realmente il dispositivo previsto.

### Output

- <code>pyproject.toml</code>.
- Lockfile.
- <code>scripts/check_environment.py</code>.

### Criterio di completamento

Un secondo computer può installare l'ambiente ed eseguire lo smoke test senza modificare il codice.

---

## 8. Fase 2 — Preparare dati e split

### Dataset consigliati

- ACCIDENT per localizzazione temporale e spaziale di incidenti CCTV.
- CADP come sorgente secondaria.
- Video normali provenienti da telecamere fisse.
- Clip difficili senza incidente: frenate, code, veicoli molto vicini, occlusioni, svolte, passaggi sotto ponti, pioggia e notte.

### Attività

- [ ] Leggere e documentare licenze e condizioni d'uso.
- [ ] Generare un identificatore per sorgente, telecamera e clip.
- [ ] Rimuovere duplicati e versioni quasi duplicate.
- [ ] Uniformare i timestamp senza cambiare la velocità temporale.
- [ ] Conservare FPS e risoluzione originali nei metadati.
- [ ] Dividere train, validation e test per telecamera o sorgente.
- [ ] Vietare che frame della stessa clip compaiano in split diversi.
- [ ] Creare una lista separata di hard negatives.
- [ ] Registrare giorno/notte, meteo, compressione, scena e tipo di incidente quando disponibili.

### Quantità iniziali suggerite

- Segmentazione: 300–800 frame annotati, campionati da condizioni diverse.
- Calibrazione: almeno una sequenza di 100–200 frame per camera.
- Detector di evento: tutte le clip di incidente disponibili e un numero almeno comparabile di finestre negative.

Le quantità sono obiettivi di partenza, non garanzie di accuratezza.

### Output

- <code>data/splits/train.txt</code>.
- <code>data/splits/val.txt</code>.
- <code>data/splits/test.txt</code>.
- <code>docs/dataset-card.md</code>.

### Criterio di completamento

Ogni campione ha provenienza, split, licenza, timestamp e annotazioni tracciabili.

---

## 9. Fase 3 — Implementare la segmentazione dei veicoli

### Passo 3.1 — Baseline preaddestrata

1. Caricare <code>yolo26n-seg.pt</code>.
2. Conservare soltanto le classi stradali necessarie.
3. Applicare una ROI stradale prima o dopo l'inferenza.
4. Salvare maschera, box, classe e confidenza.
5. Valutare separatamente veicoli vicini, piccoli e parzialmente occlusi.

### Passo 3.2 — Annotazione

- Usare poligoni o maschere per ogni istanza.
- Annotare anche veicoli parzialmente visibili.
- Stabilire una regola coerente per veicoli tagliati dal bordo.
- Evitare frame consecutivi quasi identici come maggioranza del dataset.
- Controllare manualmente almeno un campione delle annotazioni di ogni sessione.

### Passo 3.3 — Fine-tuning

1. Iniziare dal peso preaddestrato.
2. Usare immagini a 640 px.
3. Congelare inizialmente parte del backbone se la GPU è limitata.
4. Usare early stopping.
5. Salvare best e last checkpoint.
6. Valutare sempre sullo split per telecamera.
7. Controllare qualitativamente i falsi negativi lontani.

### Metriche

- mAP50-95 per le maschere.
- Precision e recall per classe.
- Recall dei veicoli piccoli nella ROI.
- Latenza media e p95.

### Output

- <code>models/training/vehicle-seg-best.pt</code>.
- Report delle metriche.
- Galleria di errori.

### Criterio di completamento

Le maschere dei veicoli importanti per l'evento sono sufficientemente stabili e la latenza è compatibile con il budget.

---

## 10. Fase 4 — Integrare ByteTrack

### Attività

1. Passare box e confidence del segmenter a ByteTrack.
2. Conservare l'associazione tra track ID e maschera corrente.
3. Usare timestamp reali per aggiornare la storia.
4. Configurare una durata massima di traccia persa.
5. Evitare ReID nella prima versione.
6. Registrare ID switch, tracce spezzate e false tracce.
7. Mantenere una history limitata, ad esempio gli ultimi 3–5 secondi.

### Parametri da validare

- Soglia per nuove tracce.
- Soglia per associazione.
- Buffer delle tracce perse.
- Soglia IoU.
- Confidenza minima del detector.

Non ottimizzare questi parametri sul test set.

### Output dati suggerito

~~~text
TrackObservation
  camera_id
  frame_index
  timestamp_s
  track_id
  class_id
  confidence
  bbox_xyxy
  mask
  ground_point_px
  ground_point_world
~~~

### Criterio di completamento

Gli ID rimangono stabili nei video di validazione e gli errori di associazione vengono salvati per l'analisi.

---

## 11. Fase 5 — Implementare prima l'omografia manuale

L'omografia manuale deve esistere prima della versione automatica: fornisce una baseline e il ground truth operativo con cui confrontare l'auto-calibrazione.

### Passo 5.1 — Selezione dei punti

1. Mostrare un frame senza traffico intenso.
2. Far selezionare almeno quattro punti complanari sulla strada.
3. Preferire intersezioni di strisce, stop line, bordi corsia o attraversamenti.
4. Distribuire i punti su una porzione ampia della ROI.
5. Associare a ciascun punto una coordinata reale o canonica.

### Passo 5.2 — Scala

Per coordinate metriche è necessaria almeno una misura reale:

- larghezza misurata della corsia;
- distanza tra due segni stradali;
- dimensione nota di un marker temporaneo;
- rilievo o planimetria disponibile.

Non assumere una larghezza universale della corsia senza dichiararlo.

### Passo 5.3 — Calcolo

- Con esattamente quattro coppie: <code>cv2.getPerspectiveTransform</code>.
- Con più di quattro coppie: <code>cv2.findHomography</code> con RANSAC.
- Trasformare le coordinate con <code>cv2.perspectiveTransform</code>.
- Usare <code>cv2.warpPerspective</code> soltanto per la visualizzazione.

### Passo 5.4 — Persistenza

Salvare un file per camera:

~~~yaml
camera_id: camera_01
image_size: [1920, 1080]
method: manual
lane_width_m: 3.4
source_points_px:
  - [420.0, 760.0]
  - [990.0, 760.0]
  - [760.0, 420.0]
  - [910.0, 420.0]
destination_points_m:
  - [0.0, 0.0]
  - [3.4, 0.0]
  - [0.0, 20.0]
  - [3.4, 20.0]
homography:
  - [0.0, 0.0, 0.0]
  - [0.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0]
confidence: 1.0
~~~

### Passo 5.5 — Validazione

- Proiettare linee di corsia nella vista bird's-eye.
- Verificare che risultino approssimativamente parallele.
- Misurare l'errore su punti non usati per stimare H.
- Controllare distanze note in più zone dell'immagine.

### Criterio di completamento

È possibile ottenere traiettorie coerenti sul piano stradale e quantificare l'errore della calibrazione.

---

## 12. Fase 6 — Auto-calibrazione mediante strisce

### Vincolo teorico

Le sole strisce longitudinali parallele forniscono soprattutto una direzione e un punto di fuga. Per una rettifica metrica completa servono:

- una seconda famiglia di linee non parallele, come stop line o attraversamenti;
- oppure informazioni sulla camera;
- e almeno una misura di scala reale.

La procedura deve quindi produrre una stima con confidenza e non fingere che ogni scena sia completamente calibrabile.

### Passo 6.1 — Sfondo mediano

1. Campionare 100–200 frame distribuiti nella sequenza.
2. Ridimensionarli alla risoluzione di calibrazione.
3. Calcolare la mediana per pixel.
4. Escludere frame con cambi improvvisi di esposizione.
5. Salvare lo sfondo e una maschera delle aree instabili.

### Passo 6.2 — ROI stradale

Ordine di preferenza:

1. ROI configurata una volta dall'utente.
2. Maschera drivable-area da modello leggero.
3. Regione inferiore dell'immagine come fallback.

La ROI riduce edifici, guardrail, pali e linee non stradali.

### Passo 6.3 — Maschera delle strisce

Baseline classica:

1. Conversione in LAB e HSV.
2. CLAHE sul canale di luminanza.
3. Soglia per bianco e giallo.
4. Top-hat morfologico per strutture chiare e sottili.
5. Canny o filtro delle creste.
6. Chiusura morfologica moderata.
7. Eliminazione delle componenti troppo piccole.

Alternativa learned:

- TwinLiteNet+ o un piccolo segmenter binario.
- Fine-tuning con frame delle stesse CCTV.
- Esecuzione solo all'avvio, non necessariamente per ogni frame.

### Passo 6.4 — Linee robuste

1. Applicare skeletonization.
2. Estrarre segmenti con LSD o HoughLinesP.
3. Eliminare segmenti corti o quasi orizzontali se non appartengono a stop line.
4. Raggruppare gli angoli.
5. Fittare le famiglie di linee con RANSAC.
6. Pesare i segmenti in base a lunghezza e probabilità della maschera.

### Passo 6.5 — Punti di fuga

- Intersecare coppie di linee della stessa famiglia.
- Stimare il punto di fuga con mediana robusta o RANSAC.
- Calcolare la dispersione delle intersezioni.
- Cercare una famiglia longitudinale e, quando disponibile, una trasversale.
- Rifiutare punti di fuga numericamente instabili.

### Passo 6.6 — Generazione dei punti

Caso preferito:

1. Selezionare due bordi di corsia.
2. Selezionare due linee trasversali.
3. Calcolare le quattro intersezioni.
4. Ordinare i vertici in modo consistente.
5. Mapparli sul rettangolo bird's-eye.

Caso parziale:

1. Selezionare due bordi corsia.
2. Usare due sezioni immagine predefinite.
3. Generare una rettifica canonica non metrica.
4. Marcare l'output come approximate.
5. Disabilitare velocità in km/h finché non viene fornita una scala.

### Passo 6.7 — Confidence score

Combinare:

- quantità e lunghezza delle linee valide;
- dispersione del punto di fuga;
- errore di riproiezione;
- parallelismo dopo il warp;
- stabilità tra sottoinsiemi di frame;
- plausibilità della larghezza corsia;
- condizionamento numerico della matrice H.

Esempio di politica:

- confidenza alta: accettazione automatica;
- confidenza media: mostrare i punti proposti e chiedere conferma;
- confidenza bassa: avviare la calibrazione manuale.

### Passo 6.8 — Rilevamento di spostamento camera

- Confrontare periodicamente keypoint statici o lo sfondo.
- Se la registrazione supera una soglia di errore, invalidare H.
- Mettere in pausa le stime metriche.
- Rieseguire la calibrazione.

### Valutazione

Confrontare H automatica e manuale su:

- errore di riproiezione;
- errore di distanza;
- successo/fallimento della procedura;
- stabilità giorno/notte;
- tempo di calibrazione.

### Criterio di completamento

Il sistema non usa una calibrazione automatica inaffidabile senza segnalarlo e produce un confronto quantitativo con la baseline manuale.

---

## 13. Fase 7 — Estrarre correttamente il punto a terra

### Algoritmo iniziale

1. Individuare il valore y massimo della maschera.
2. Considerare una fascia inferiore pari a circa il 3–8% dell'altezza della maschera.
3. Calcolare la mediana delle coordinate x dei pixel della fascia.
4. Usare una statistica robusta di y invece di un singolo pixel estremo.
5. Vincolare il punto alla ROI stradale.
6. Trasformarlo con H.

### Perché non usare il centro del box

Il centro del box appartiene visivamente alla carrozzeria e non al piano stradale. Applicargli l'omografia introduce un errore dipendente da altezza del veicolo e prospettiva.

### Gestione delle occlusioni

- Se la parte inferiore della maschera è occlusa, usare la predizione del tracker.
- Marcare il punto come predicted.
- Ridurre il peso delle feature derivate.
- Non calcolare accelerazioni forti da una singola osservazione riapparsa.

### Criterio di completamento

Il punto trasformato segue visivamente il veicolo nella vista dall'alto senza salti sistematici dovuti alla forma del box.

---

## 14. Fase 8 — Costruire traiettorie e feature

### Filtraggio

- Ordinare le osservazioni per timestamp.
- Rimuovere duplicati temporali.
- Usare EMA, Kalman o Savitzky–Golay su una finestra corta.
- Non filtrare oltre il punto da introdurre un ritardo incompatibile con il detector.
- Calcolare derivate usando il delta temporale reale.

### Feature per singolo veicolo

- Posizione x/y.
- Velocità x/y e modulo.
- Accelerazione x/y e modulo.
- Jerk.
- Heading e variazione di heading.
- Durata della traccia.
- Durata di arresto.
- Confidenza media della segmentazione.
- Percentuale di osservazioni predette.
- Distanza dal bordo della ROI.

### Feature per coppia di veicoli

- Distanza corrente.
- Velocità relativa.
- Direzione relativa.
- Tempo al punto di minima distanza.
- Distanza minima prevista.
- Time-to-collision quando definito.
- IoU dei box e delle maschere in immagine.
- Convergenza o divergenza delle traiettorie.
- Variazione di velocità prima e dopo il punto di massimo avvicinamento.

### Ottimizzazione

Non confrontare tutte le coppie se il traffico è elevato:

- creare una griglia spaziale nella vista bird's-eye;
- confrontare soltanto veicoli entro un raggio configurabile;
- escludere coppie chiaramente divergenti.

### Dataset di feature

Salvare una riga per finestra temporale:

~~~text
camera_id, clip_id, window_start, window_end,
track_a, track_b, features..., label, impact_time
~~~

### Criterio di completamento

Le feature sono riproducibili da una clip e possono essere ispezionate senza rieseguire il segmenter.

---

## 15. Fase 9 — Detector di incidente a regole

### Macchina a stati

~~~text
NORMAL
  -> CANDIDATE
  -> CONFIRMED
  -> COOLDOWN
  -> NORMAL

CANDIDATE
  -> REJECTED
  -> NORMAL
~~~

### Attivazione del candidato

Usare più condizioni, ad esempio:

- tempo al minimo avvicinamento ridotto;
- distanza minima prevista inferiore a una soglia;
- traiettorie convergenti;
- IoU o sovrapposizione prospettica in crescita;
- brusco cambiamento di velocità.

### Conferma post-impatto

Richiedere almeno una evidenza temporale:

- forte decelerazione;
- cambio improvviso di heading;
- arresto di uno o più veicoli;
- separazione anomala dopo il contatto;
- traccia che termina vicino al punto di possibile impatto;
- moto residuo incoerente con la traiettoria precedente.

### Regole di robustezza

- Richiedere più osservazioni consecutive.
- Ignorare picchi prodotti da una sola maschera.
- Ridurre il punteggio con tracking instabile.
- Non confermare se la calibrazione è invalida e le sole evidenze sono metriche.
- Usare un cooldown per non contare più volte lo stesso evento.
- Memorizzare le ragioni che hanno prodotto il punteggio.

### Output evento

~~~json
{
  "event_id": "camera_01_000042",
  "camera_id": "camera_01",
  "start_time_s": 118.4,
  "impact_time_s": 120.1,
  "confirm_time_s": 121.2,
  "track_ids": [17, 23],
  "score": 0.91,
  "calibration_confidence": 0.88,
  "reasons": ["low_ttc", "high_deceleration", "post_impact_stop"],
  "clip_path": "outputs/clips/camera_01_000042.mp4"
}
~~~

### Criterio di completamento

Il detector produce eventi spiegabili, non genera duplicati e può essere eseguito su almeno un'ora di video normale per misurare i falsi allarmi.

---

## 16. Fase 10 — Classificatore temporale leggero

Implementarlo solo dopo una baseline a regole funzionante.

### Preparazione delle finestre

1. Definire una durata, ad esempio 2–4 secondi.
2. Centrare le finestre positive attorno all'impatto.
3. Campionare negative dalla stessa scena e da hard negatives.
4. Evitare finestre quasi duplicate in split diversi.
5. Inserire indicatori di qualità di tracking e calibrazione.
6. Normalizzare soltanto usando statistiche del train set.

### Modelli da provare

Ordine consigliato:

1. Logistic Regression come baseline.
2. HistGradientBoostingClassifier.
3. XGBoost, se giustificato.
4. Piccola GRU o TCN soltanto se le feature tabellari non sono sufficienti.

### Addestramento

- Gestire lo sbilanciamento con class weights o campionamento.
- Selezionare iperparametri sul validation set.
- Calibrare il punteggio se viene mostrato come probabilità.
- Scegliere la soglia in base al compromesso recall–falsi allarmi/ora.
- Non scegliere la soglia sul test.

### Integrazione ibrida

Pipeline raccomandata:

1. Le regole economiche generano candidati.
2. Il classificatore assegna il punteggio.
3. La logica post-impatto conferma.
4. La macchina a stati elimina duplicati.

### Interpretabilità

- Salvare l'importanza delle feature.
- Mostrare le feature determinanti nella demo.
- Confrontare il modello con la baseline a regole.

### Criterio di completamento

Il classificatore migliora almeno una metrica rilevante senza peggiorare in modo non accettabile falsi allarmi o latenza.

---

## 17. Fase 11 — Buffer e salvataggio delle clip

### Implementazione

- Mantenere un ring buffer compresso o di frame recenti.
- Durata suggerita: 5 secondi prima dell'evento.
- Dopo la conferma continuare a registrare per 10 secondi.
- Unire eventi molto vicini relativi agli stessi track ID.
- Scrivere la clip in un thread separato.
- Usare timestamp originali.
- Limitare lo spazio totale con una politica di retention configurabile.

### Criterio di completamento

Ogni evento confermato ha metadati e una clip che comprende il contesto precedente e successivo.

---

## 18. Fase 12 — Ottimizzare l'inference locale

### Passo 12.1 — Profilare prima di ottimizzare

Misurare separatamente:

- decode;
- preprocessing;
- segmentazione;
- postprocessing delle maschere;
- tracking;
- trasformazioni geometriche;
- feature ed event detector;
- rendering;
- scrittura video.

Registrare media, mediana e p95.

### Passo 12.2 — Ridurre il carico

- Limitare la ROI.
- Usare input 512 o 640.
- Elaborare il modello a 5–10 FPS.
- Usare batch 1 per bassa latenza.
- Non trasformare l'intero frame in bird's-eye durante la pipeline.
- Non eseguire il detector di strisce per ogni frame.
- Conservare solo history limitate.
- Separare rendering e inferenza.

### Passo 12.3 — Coda a bassa latenza

- Decoder, inference e UI in componenti separati.
- Coda di inference con dimensione 1 o 2.
- Se il sistema è in ritardo, scartare frame vecchi invece di accumularli.
- Non scartare i timestamp.
- Non perdere i frame del ring buffer necessari alla clip.

### Passo 12.4 — Export

OpenVINO:

~~~python
from ultralytics import YOLO

model = YOLO("models/training/vehicle-seg-best.pt")
model.export(
    format="openvino",
    imgsz=640,
)
~~~

ONNX:

~~~python
from ultralytics import YOLO

model = YOLO("models/training/vehicle-seg-best.pt")
model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
)
~~~

TensorRT:

~~~python
from ultralytics import YOLO

model = YOLO("models/training/vehicle-seg-best.pt")
model.export(
    format="engine",
    imgsz=640,
    quantize=16,
)
~~~

Verificare gli argomenti supportati dalla versione bloccata nel progetto.

### Passo 12.5 — Quantizzazione

1. Conservare FP32 o FP16 come riferimento.
2. Preparare un calibration set rappresentativo.
3. Generare INT8.
4. Rivalutare segmentazione ed evento, non solo la latenza.
5. Accettare INT8 soltanto se il calo sui veicoli piccoli è compatibile con gli obiettivi.

### Criterio di completamento

La pipeline raggiunge il target sulla macchina indicata e il report riporta sia prestazioni sia eventuale perdita di accuratezza.

---

## 19. Fase 13 — Interfaccia Streamlit

### Vista principale

- Video originale.
- Maschere e track ID.
- Tracce recenti.
- Stato della pipeline.
- Punteggio di incidente.
- FPS e latenza.

### Vista bird's-eye

- ROI stradale.
- Posizione dei veicoli.
- Direzione e velocità.
- Coppia candidata.
- Punto stimato dell'impatto.

### Pannello calibrazione

- Sfondo mediano.
- Maschera delle strisce.
- Linee accettate e rifiutate.
- Punti di fuga.
- Quattro punti proposti.
- Confidence score.
- Pulsanti accetta, correggi e calibrazione manuale.

### Pannello eventi

- Timeline.
- Elenco eventi.
- Track ID coinvolti.
- Cause del punteggio.
- Riproduzione della clip.
- Esportazione JSON/CSV.

### Criterio di completamento

Durante la demo è possibile comprendere perché il sistema ha prodotto o rifiutato un allarme.

---

## 20. Configurazione centrale suggerita

~~~yaml
project:
  seed: 42
  output_dir: outputs

video:
  source: data/samples/demo.mp4
  target_fps: 8
  queue_size: 2
  reconnect_rtsp: true

perception:
  model: models/exported/vehicle-seg
  backend: openvino
  image_size: 640
  confidence: 0.30
  classes: [car, truck, bus, motorcycle]

tracking:
  tracker: bytetrack
  config: configs/bytetrack.yaml
  history_seconds: 5.0

calibration:
  camera_id: camera_01
  mode: auto_assisted
  file: data/calibration/camera_01.yaml
  lane_width_m: null
  min_confidence_auto: 0.80
  min_confidence_assisted: 0.55

features:
  smoothing: ema
  ema_alpha: 0.35
  pair_radius_m: 15.0
  window_seconds: 3.0

events:
  candidate_threshold: 0.55
  confirm_threshold: 0.80
  confirm_duration_s: 0.40
  cooldown_s: 8.0
  pre_event_s: 5.0
  post_event_s: 10.0

storage:
  sqlite_path: outputs/events/events.sqlite
  clips_dir: outputs/clips
  retention_gb: 10

ui:
  render_masks: true
  render_bird_eye: true
  render_debug_calibration: false
~~~

I valori sono iniziali e devono essere scelti sul validation set.

---

## 21. Pseudocodice della pipeline

~~~python
config = load_config()
video = VideoSource(config.video)
segmenter = Segmenter(config.perception)
tracker = VehicleTracker(config.tracking)

calibration = load_valid_calibration(config.calibration)
if calibration is None:
    proposal = auto_calibrate(video.sample_frames())
    calibration = accept_or_correct(proposal)

trajectory_store = TrajectoryStore(max_seconds=5)
event_detector = EventDetector(config.events)
clip_buffer = ClipBuffer(config.events)

for frame, timestamp in video:
    clip_buffer.append(frame, timestamp)

    if segmenter.is_due(timestamp):
        instances = segmenter.predict(frame)
        tracks = tracker.update(instances, timestamp)
    else:
        tracks = tracker.predict(timestamp)

    observations = []
    for track in tracks:
        point_px, point_quality = ground_point(track.mask, track.bbox)
        point_world = calibration.transform(point_px)
        observations.append(
            build_observation(
                track,
                timestamp,
                point_px,
                point_world,
                point_quality,
            )
        )

    trajectory_store.update(observations)
    features = compute_features(trajectory_store, calibration.quality)
    decision = event_detector.update(features, timestamp)

    if decision.just_confirmed:
        save_event(decision)
        clip_buffer.schedule_export(decision)

    render(frame, observations, features, decision, calibration)
~~~

---

## 22. Test

### Unit test

- Ordinamento corretto dei quattro punti.
- Omografia identità.
- Trasformazione di punti noti.
- Rifiuto di matrici degeneri.
- Estrazione robusta del punto inferiore della maschera.
- Derivate con timestamp irregolari.
- TTC nei casi convergente, parallelo e divergente.
- Transizioni della macchina a stati.
- Cooldown e deduplicazione.
- Ring buffer prima/dopo evento.
- Serializzazione della calibrazione.

### Integration test

- Video breve → segmentazione → tracking → file di traiettorie.
- Calibrazione salvata → coordinate bird's-eye deterministiche.
- Clip positiva → un solo evento.
- Clip negativa → nessun evento.
- Invalidation della calibrazione dopo uno spostamento simulato.

### Regression test

Mantenere un piccolo insieme di video:

- incidente evidente;
- quasi incidente;
- occlusione;
- notte;
- pioggia;
- traffico fermo;
- camera con strisce non visibili.

Salvare metriche e output attesi entro tolleranze.

---

## 23. Protocollo di valutazione

### Segmentazione

- mAP50-95 mask.
- Precision e recall.
- Recall per dimensione apparente del veicolo.
- Latenza media e p95.

### Tracking

- HOTA o IDF1, se sono disponibili annotazioni.
- Numero di ID switch.
- Percentuale di tracce frammentate.
- Durata media delle tracce.

### Calibrazione

- Errore di riproiezione.
- Errore su distanze note.
- Stabilità della matrice nel tempo.
- Percentuale di scene calibrate automaticamente.
- Percentuale di scene inviate al fallback.

### Evento

- Precision.
- Recall.
- F1.
- Falsi allarmi per ora.
- Ritardo medio e p95 di rilevamento.
- Temporal IoU, se richiesta dal dataset.
- Prestazioni per giorno/notte e scena.

### Runtime

- FPS effettivi.
- Latenza end-to-end media e p95.
- Uso CPU.
- Uso GPU.
- RAM e VRAM.
- Dimensione dei modelli.
- Tempo di avvio e calibrazione.

### Regola di split

Riportare i risultati soltanto su un test set separato per telecamera o sorgente. Non effettuare tuning sul test.

---

## 24. Ablation study consigliato

| Esperimento | Segmentazione | Coordinate | Calibrazione | Detector evento | Quantizzazione |
| --- | --- | --- | --- | --- | --- |
| E0 | Box | Immagine | Nessuna | Regole | No |
| E1 | Maschera | Immagine | Nessuna | Regole | No |
| E2 | Maschera | Bird's-eye | Manuale | Regole | No |
| E3 | Maschera | Bird's-eye | Automatica | Regole | No |
| E4 | Maschera | Bird's-eye | Manuale | Classificatore | No |
| E5 | Maschera | Bird's-eye | Automatica | Classificatore | No |
| E6 | Maschera | Bird's-eye | Automatica | Classificatore | INT8/FP16 |

Domande a cui rispondere:

1. Le maschere migliorano il punto a terra rispetto ai box?
2. L'omografia migliora le feature di distanza e il detector?
3. Quanto perde l'auto-calibrazione rispetto alla calibrazione manuale?
4. Il classificatore migliora i falsi allarmi rispetto alle regole?
5. Qual è il costo in accuratezza della quantizzazione?

---

## 25. Analisi degli errori

Per ogni falso positivo o falso negativo salvare:

- frame prima, durante e dopo;
- maschere;
- track ID;
- traiettorie;
- qualità della calibrazione;
- feature;
- stato della macchina a stati;
- motivi della decisione.

Classificare l'errore come:

- veicolo non segmentato;
- maschera instabile;
- ID switch;
- punto a terra errato;
- omografia errata;
- occlusione;
- soglia non corretta;
- etichetta ambigua;
- scena fuori distribuzione.

Non limitarsi a mostrare la confusion matrix: collegare ogni errore al modulo che lo ha originato.

---

## 26. Rischi e fallback

| Rischio | Effetto | Mitigazione |
| --- | --- | --- |
| Strisce invisibili | H automatica assente | Confidence score e fallback manuale |
| Una sola famiglia di linee | Scala/metrica indeterminata | Richiedere misura o modalità non metrica |
| Veicoli piccoli | Falsi negativi | Fine-tuning, ROI, input più grande |
| Occlusioni | ID switch e salti | ByteTrack tuning e quality flag |
| False collisioni prospettiche | Falsi allarmi | Bird's-eye e conferma temporale |
| Video compressi | Maschere instabili | Augmentation JPEG/blur |
| Pochi incidenti | Overfitting | Hard negatives, split per sorgente, modello semplice |
| Camera spostata | Misure errate | Rilevamento movimento e invalidazione H |
| Pipeline lenta | Ritardo crescente | Queue corta e drop dei frame vecchi |
| INT8 perde veicoli piccoli | Recall inferiore | Confronto con FP16/FP32 e rollback |

---

## 27. Piano temporale indicativo di sei settimane

### Settimana 1

- Definizione del protocollo.
- Ambiente.
- Raccolta e split dei dati.
- Baseline preaddestrata.

### Settimana 2

- Annotazione e fine-tuning.
- ByteTrack.
- Esportazione delle traiettorie in coordinate immagine.

### Settimana 3

- Calibrazione manuale.
- Bird's-eye.
- Estrazione feature.
- Prima baseline a regole.

### Settimana 4

- Sfondo mediano.
- Segmentazione delle strisce.
- Line fitting e punti di fuga.
- Auto-calibrazione assistita.

### Settimana 5

- Dataset di feature.
- Classificatore temporale.
- Analisi falsi positivi e falsi negativi.

### Settimana 6

- Export OpenVINO/ONNX/TensorRT.
- Benchmark.
- UI Streamlit.
- Ablation study.
- Relazione, demo e backup.

---

## 28. Priorità se il tempo è limitato

### Must have

1. Segmentazione.
2. Tracking.
3. Omografia manuale.
4. Traiettorie bird's-eye.
5. Detector a regole.
6. Metriche e demo.

### Should have

1. Auto-calibrazione assistita.
2. Confidence score.
3. Classificatore di feature.
4. Salvataggio clip.

### Nice to have

1. RTSP.
2. INT8.
3. Rilevamento automatico dello spostamento camera.
4. Confronto con un piccolo modello temporale neurale.

Se il tempo termina, non sacrificare valutazione e baseline manuale per aggiungere un modello più complesso.

---

## 29. Struttura della relazione d'esame

1. Problema e motivazione.
2. Requisiti di esecuzione locale.
3. Dataset e protocollo di split.
4. Segmentazione delle istanze.
5. Tracking.
6. Modello geometrico del piano stradale.
7. Calibrazione manuale.
8. Auto-calibrazione e suoi limiti.
9. Feature temporali.
10. Detector di incidente.
11. Ottimizzazione del runtime.
12. Metriche.
13. Ablation study.
14. Analisi degli errori.
15. Limiti, privacy e sviluppi futuri.

### Contributo sperimentale da evidenziare

La domanda centrale può essere:

> La rettifica del piano stradale, combinata con punti a terra ottenuti dalle maschere, migliora il rilevamento leggero degli incidenti rispetto a feature calcolate direttamente nello spazio immagine?

Questa formulazione rende segmentazione e omografia parti necessarie dell'esperimento, non semplici elementi grafici.

---

## 30. Checklist per la demo

- [ ] Ambiente ricreabile da lockfile.
- [ ] Modelli già presenti in locale.
- [ ] Nessun download necessario durante la presentazione.
- [ ] Video positivo breve.
- [ ] Video negativo breve.
- [ ] Calibrazione manuale salvata.
- [ ] Auto-calibrazione dimostrabile.
- [ ] Fallback manuale funzionante.
- [ ] Vista con maschere e track ID.
- [ ] Vista bird's-eye.
- [ ] Punteggio e motivi dell'evento.
- [ ] Clip salvata.
- [ ] Tabella delle metriche.
- [ ] Grafico precision–recall o soglia–falsi allarmi.
- [ ] Benchmark sulla macchina target.
- [ ] Piano B con video dell'esecuzione.

---

## 31. Definition of Done

Il progetto è completato quando:

- [ ] Elabora end-to-end un video mai visto.
- [ ] Produce segmentazioni e track ID.
- [ ] Trasforma correttamente i punti a terra.
- [ ] Supporta calibrazione manuale.
- [ ] Propone automaticamente la calibrazione quando le strisce sono sufficienti.
- [ ] Rifiuta o richiede conferma quando la calibrazione è debole.
- [ ] Genera eventi temporali deduplicati.
- [ ] Salva metadati e clip.
- [ ] È valutato su split per sorgente.
- [ ] Riporta falsi allarmi per ora e ritardo.
- [ ] Riporta FPS, latenza e memoria sulla macchina target.
- [ ] Include almeno tre ablation significative.
- [ ] Documenta errori, limiti e licenze.
- [ ] Può essere avviato con un singolo comando documentato.

---

## 32. Riferimenti tecnici

- Ultralytics instance segmentation: https://docs.ultralytics.com/tasks/segment/
- Ultralytics tracking: https://docs.ultralytics.com/modes/track/
- Ultralytics export: https://docs.ultralytics.com/modes/export/
- Licenza Ultralytics: https://www.ultralytics.com/license
- OpenCV homography: https://docs.opencv.org/4.13.0/d9/dab/tutorial_homography.html
- OpenCV geometric transformations: https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html
- ONNX Runtime Execution Providers: https://onnxruntime.ai/docs/execution-providers/
- OpenVINO documentation: https://docs.openvino.ai/
- ByteTrack: https://github.com/FoundationVision/ByteTrack
- TwinLiteNet+: https://arxiv.org/abs/2403.16958
- ACCIDENT benchmark: https://accidentbench.github.io/
- CADP: https://arxiv.org/abs/1809.05782


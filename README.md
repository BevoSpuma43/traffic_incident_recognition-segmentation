# CCTV Incident Detection

Sistema Python locale per individuare **possibili incidenti** da una telecamera fissa.
Include YOLO26-seg, ByteTrack, omografia manuale, proposta assistita dalle strisce,
traiettorie metriche, detector temporale spiegabile, clip MP4, SQLite, UI Streamlit e
strumenti per dataset, training, export e valutazione.

Il piano originale resta in [piano-implementazione-cctv-incident-detection.md](piano-implementazione-cctv-incident-detection.md).
Lo stato verificato e i limiti sperimentali sono in [docs/implementation-status.md](docs/implementation-status.md).

## Avvio su Windows

Dalla radice del repository, con [uv](https://docs.astral.sh/uv/) e Python 3.12
installato da [python.org](https://www.python.org/downloads/windows/):

```powershell
uv sync --frozen --extra inference --extra ui --extra ml --dev
.venv\Scripts\python.exe scripts/download_model.py
.venv\Scripts\cctv-incident.exe demo
```

Il progetto usa Python 3.12 e configura `uv` con `python-preference = "only-system"`.
Sul PC di sviluppo Windows blocca alcune DLL del Python gestito da `uv`; la
procedura usa quindi l'interprete già installato sul sistema. In caso di errore
`DLL load failed while importing _ssl`, vedere la
[procedura di ripristino](docs/ripristino-python-windows.md).

L'ultimo comando genera una scena sintetica positiva, la calibrazione e il risultato completo.
Per visualizzarla durante l'elaborazione aggiungere `--show`; premere Q per chiudere.
Per il controllo negativo:

```powershell
.venv\Scripts\cctv-incident.exe demo --negative
```

**La demo sintetica usa maschere dei colori**, con il vero ByteTrack e tutta la pipeline successiva.
Non misura la qualità di YOLO né l'accuratezza su incidenti reali.
I pesi YOLO vengono usati da `configs/default.yaml`; il download è una preparazione esplicita,
mai un fallback durante l'analisi.

Interfaccia completa, dopo l'installazione:

```powershell
.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Per **Segmentazione + omografia**, selezionare un video locale e completare
la scheda **Calibrazione**: quattro punti trascinabili, dimensioni in metri
con origine confermata e ROI separata. È possibile salvare una bozza e riprenderla;
una calibrazione confermata va revisionata sul fotogramma prima del riuso.
L'analisi si abilita dopo la conferma. Uso, verifiche e limiti:
[editor di calibrazione](docs/fase-2-editor.md).

Il pulsante **Calibrazione automatica** mostra subito la prima proposta con
punti trascinabili e nasconde i controlli di inserimento manuale. Le alternative
si applicano direttamente dal menu; le distanze disponibili o applicate da
un preset scelto compaiono nei campi modificabili. Le misure sconosciute restano
vuote. **Torna alla selezione manuale** riapre coordinate numeriche e ROI.

L'espansore **Calibrazione dalle automobili · sperimentale** aggiunge una stima
approssimativa del piano stradale da più sagome di auto e dimensioni ipotizzate
modificabili. Il valore iniziale è 4,7 × 1,8 × 1,5 m, scelto per il progetto;
non rappresenta una media statistica verificata del parco USA. Uso e limiti:
[calibrazione dalle automobili](docs/calibrazione-dalle-automobili.md).

La pipeline adatta la calibrazione alla risoluzione elaborata, conserva gli
snapshot nel run e segnala gli esiti non valutabili dopo invalidazioni o
interruzioni. La fase 4 è verificata con **231 test superati** nella `.venv`
esistente, senza blocchi Windows. Restano controllo visivo nel browser e
verifica dell'accuratezza fisica su misure reali:
[implementazione, artefatti e collaudo](docs/fase-4-pipeline.md).

La voce **standard_dataset analisi in batch - con omografia** ora permette
la **preparazione delle calibrazioni**: sessioni persistenti, proposte in un
worker separato con stop/ripresa, tabella degli stati ed editor condiviso.
Le conferme restano individuali. Dopo la revisione, **Analisi batch** permette
di creare esperimenti metrici con copie verificate delle calibrazioni, stop/ripresa
e risultati separati per modalità e modello. Se la geometria diventa invalida,
il video non viene conteggiato e occorre un nuovo esperimento.
Guide: [preparazione](docs/fase-5-preparazione.md) e
[batch metrico e snapshot](docs/fase-6-batch-metrico.md).

La fase 7 aggiunge consultazione degli esperimenti storici anche senza pesi,
blocco preventivo delle riprese incompatibili, esportazione separata e strumenti
per campionamento/revisione e confronto controllato. Su 5 scene reali sono state
generate proposte per 4: è copertura grezza, non accuratezza metrica. Mancano
misure indipendenti, revisioni cronometrate e collaudo visivo della GUI; il
confronto reale richiede calibrazioni confermate. Procedura e stato effettivo:
[validazione della fase 7](docs/fase-7-validazione.md).

Su Linux/macOS usare gli eseguibili in `.venv/bin/`; il lockfile e la configurazione
PyTorch CPU sono stati verificati su Windows. Nessun servizio cloud è necessario.

Per i video reali, **Qualità analisi → Rapida** usa YOLO26s-seg a 8 FPS
richiesti; **Accurata** usa YOLO26m-seg a 15 FPS. Il modello small conserva meglio
i veicoli poco visibili rispetto al precedente nano, con un costo di calcolo
maggiore. Accurata rimane più lenta su CPU. Preparare esplicitamente i modelli:

```powershell
.venv\Scripts\python.exe scripts/download_model.py --output models/yolo26s-seg.pt
.venv\Scripts\python.exe scripts/download_model.py --output models/yolo26m-seg.pt
```

Gli urti con veicoli ancora in movimento possono essere confermati anche da un
rallentamento persistente dopo un contatto osservato con reazioni di entrambi i
veicoli. Una breve perdita di una traccia non cancella subito quel contesto.
Per gli urti laterali vengono cercati anche cambiamenti persistenti di forma,
direzione e velocità dopo contatti ripetuti. Il tracker usa il movimento per
mantenere l'identità dei veicoli veloci quando le bbox si sovrappongono poco;
le associazioni ambigue vengono escluse. Negli urti trasversali viene conservato
il primo contatto tra due veicoli prima separati, confermandolo con rotazione e
rallentamento persistenti nella zona dell'urto. Rilevazioni duplicate della stessa
sagoma e salti del punto della maschera su bbox ferme vengono filtrati.
I risultati verificati sui video segnalati sono in
[docs/real-video-validation.md](docs/real-video-validation.md).

## Scegliere un video dal frontend

Selezionare **Solo Segmentazione** e scegliere la **Sorgente video**:

- **Carica un video** apre la selezione dei file del computer, fino a 1 GB per file.
  Sono accettati MP4, AVI, MOV, MKV, WebM e altri formati video comuni.
- **Video del dataset ACCIDENT** mostra tutti i video locali del dataset: digitare
  il nome nel menu per cercare il file, anche fuori dal campione di dieci clip.
- **Percorso o URL RTSP** permette di indicare un file locale, anche oltre il limite
  di caricamento, o uno stream RTSP.

Scegliere Rapida o Accurata e premere **Avvia analisi**. Il caricamento e disponibile
anche nella modalita con calibrazione metrica. I file caricati restano in
`data/uploads/` per consentire i replay successivi. Video diversi con lo stesso
nome vengono conservati separatamente. La decodifica dipende dai codec supportati
nell'ambiente; il risultato annotato viene esportato in MP4.

## Rivedere un'analisi

Per elaborare una cartella intera con checkpoint, selezionare **standard_dataset analisi in batch - no omografia**:
scegliere cartella e modello locale, premere **Prepara batch**, poi **Avvia batch**.
**Stop** interrompe il video corrente; **Riprendi** salta quelli completati e ricomincia
il video interrotto. Risultati e checkpoint sono separati per modello/configurazione in
`outputs/batches/`. Sono disponibili CSV per video, per evento e metriche aggregate,
con confronto temporale entro **±1 secondo**. Procedura e definizioni di TP/FP/FN/TN:
[analisi batch](docs/batch-analysis.md).

Al termine dell'analisi, la scheda **Rivedi analisi** offre un video completo con
bounding box, ID, traiettorie e velocità. La barra del lettore permette di scorrere
il video; **Scarica video annotato** salva una copia MP4. Il video mantiene i
timestamp originali; le annotazioni si aggiornano alla frequenza dell'analisi.

Gli impatti rilevati sono segnalati da un mirino rosso nel punto stimato dalle
bounding box dei veicoli coinvolti. Il segnale compare dal tempo stimato dell'urto,
anche se la conferma arriva dopo. Il menu **Vai a un impatto** porta un secondo
prima dell'evento. Se il detector non rileva eventi, non vengono aggiunti segnali.

Le analisi precedenti sono disponibili nella stessa scheda: **Prepara video
annotato** usa i risultati salvati senza ripetere YOLO. Serve il video originale.
Ogni analisi conserva `annotated.mp4` e `replay.json` nella propria cartella
`outputs/.../runs/<run_id>/`. Le riproduzioni non includono l'audio né le maschere
di segmentazione, che non sono conservate nei log. Le registrazioni RTSP non
salvate come file non possono essere ricostruite.

## Analizzare un video CCTV

1. Mettere il video autorizzato in `data/raw/`.
2. Creare un JSON con almeno quattro coppie di punti strada–piano. Usare
   `configs/cameras/camera_01.points.example.json` come struttura; le coordinate di esempio
   sono canoniche e vanno sostituite con misure della propria scena.
3. Calibrare e salvare la geometria:

```powershell
.venv\Scripts\python.exe scripts/calibrate_camera.py --source data/raw/strada.mp4 --points miei-punti.json --output data/calibration/camera_01.yaml
```

Se il JSON contiene solo `destination_points`, si apre la selezione visuale:
click nell'ordine delle corrispondenze, Backspace annulla, Invio conferma, Esc esce.
Non riordinare indipendentemente i punti sorgente e destinazione. Impostare `units: "m"`
solo per coordinate misurate in metri. Salvare almeno un'immagine di riferimento è automatico.

4. In `configs/default.yaml` impostare il video, la camera e il modello. Avviare:

```powershell
.venv\Scripts\cctv-incident.exe run --config configs/default.yaml --source data/raw/strada.mp4
```

La calibrazione deve corrispondere esattamente a risoluzione e camera del video.
Una calibrazione canonica permette la visualizzazione ma **sospende le decisioni metriche**.
Il punteggio mostrato è un indice euristico, non una probabilità calibrata.

Per la proposta dalle strisce:

```powershell
.venv\Scripts\python.exe scripts/calibrate_camera.py --source data/raw/strada.mp4 --auto --output data/calibration/proposal.yaml
```

Il comando salva sfondo/riferimento, strisce, linee e diagnostica. La proposta richiede revisione:
nell'interfaccia copiarne/correggerne i punti, associare misure reali e salvare come calibrazione
manuale. Due sole linee longitudinali non determinano una calibrazione metrica completa.

## Video reali senza calibrazione e campione ACCIDENT

Per i video ACCIDENT usare `configs/accident-image.yaml`, oppure scegliere
**Solo Segmentazione** nella UI. La nuova modalita usa coordinate
immagine e soglie relative ai veicoli.

```powershell
.venv\Scripts\cctv-incident.exe run --config configs/accident-image.yaml
.venv\Scripts\python.exe scripts/benchmark_accident.py --per-class 2 --seed 42 --output outputs/accident-sample
```

Il benchmark elabora solo **10 clip reali** e salva anteprime, manifest, CSV e report.
Per ripeterlo scegliere una nuova cartella di output.
Procedura, protocollo e limiti: [docs/real-videos.md](docs/real-videos.md).
Risultati: [verifica reale](docs/real-video-validation.md), [report locale completo](outputs/accident-sample/report.md). Il primo campione ha prodotto TP=0, FP=2, FN=10: la baseline richiede miglioramenti.

## Output e configurazione

- `outputs/runs/<run_id>/run.json`: configurazione risolta, geometria, hardware e hash del modello.
- `trajectories.jsonl`: osservazioni in pixel e sul piano, timestamp, ID e qualità.
- `features.jsonl`: feature per veicolo/coppia, stato, score e ragioni.
- `metrics.json`: FPS effettivi, tempo video, RAM, CPU, fallback timestamp, latenze medie/p95.
- `outputs/events/`: SQLite e un JSON per evento.
- `outputs/clips/`: MP4 con timestamp variabili; offset originali e troncamenti nei metadati.
- La demo scrive in `outputs/demo/`.

Le opzioni sono validate in `src/cctv_incident/config.py`. Percorsi relativi alla
`project.root_dir`, a sua volta relativa al file YAML; non dipendono dal terminale.
`video.clip_id` deve coincidere con le annotazioni, altrimenti usa il nome del video senza estensione.
Le soglie iniziali richiedono tuning sul validation set.

Il buffer JPEG ha un limite in byte; le clip attive e in coda hanno un limite numerico e in byte
per clip. A fine video le clip parziali vengono salvate e marcate `clip_truncated`.
La retention elimina solo MP4 dalla cartella clip e aggiorna i metadati con stato `expired`.
I log di esperimento rimangono su disco per l'audit e vanno archiviati periodicamente.

## Verifiche

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check src apps scripts tests
.venv\Scripts\ruff.exe format --check src apps scripts tests
.venv\Scripts\python.exe scripts/check_environment.py
```

Il test reale YOLO usa l'immagine bus inclusa in Ultralytics e viene saltato se i pesi locali
non sono disponibili. I test geometrici e temporali non scaricano nulla.
Lo smoke test non equivale a una valutazione su un dataset CCTV.

## Dataset, training e valutazione

Vedere [docs/dataset-card.md](docs/dataset-card.md) per il manifest e le annotazioni.

ACCIDENT v9 è stato scaricato in `data/raw/ACCIDENT/` (2.027 video reali e 2.211 sintetici).
Percorsi, split ufficiali, licenza e controlli: [docs/accident-dataset.md](docs/accident-dataset.md).

```powershell
.venv\Scripts\python.exe scripts/prepare_dataset.py data/manifest.jsonl
.venv\Scripts\python.exe scripts/train_segmenter.py --data data/vehicles-seg.yaml --device cpu
.venv\Scripts\python.exe scripts/extract_features.py --features outputs/runs/RUN/features.jsonl --annotations eventi.json --group sorgente01 --split train --output data/processed/train.csv
.venv\Scripts\python.exe scripts/train_event_classifier.py data/processed/train.csv data/processed/val.csv
```

Il classificatore usa medie temporali causali delle stesse feature in training e runtime.
Impostare `events.classifier` al file joblib per attivarlo e scegliere le soglie sul validation set.
Il training rifiuta il test set e gruppi/camere/clip presenti in split diversi. L'artefatto
registra schema e durata finestra; il punteggio non viene dichiarato una probabilità calibrata.

Esportazione degli eventi e valutazione per evento, con matching uno-a-uno per camera e clip:

```powershell
.venv\Scripts\cctv-incident.exe export-events --config configs/default.yaml --run-id RUN --output outputs/predictions.json
.venv\Scripts\python.exe scripts/evaluate.py --predictions outputs/predictions.json --truth eventi.json --duration 3600
.venv\Scripts\python.exe scripts/evaluate_suite.py data/evaluation.json --split test --ablations
```

La durata deve includere **tutti** i video valutati, anche i negativi. Il report contiene
precision, recall, F1, falsi allarmi/ora e ritardo medio/p95. Le ablation disponibili confrontano
baseline, punto inferiore del box, assenza di smoothing e conferma più lunga. Non sostituiscono
gli esperimenti E0–E6 del piano; aggiungerli dopo aver raccolto dati e calibrazioni reali.

## Backend e RTSP

L'ambiente predefinito usa PyTorch **CPU**. Per ONNX o OpenVINO aggiungere rispettivamente
`--extra onnx` o `--extra openvino` a `uv sync`, mantenendo `--extra inference`.
Gli export usano gli argomenti della versione Ultralytics bloccata:

```powershell
.venv\Scripts\python.exe scripts/export_model.py --weights models/yolo26n-seg.pt --format onnx
.venv\Scripts\python.exe scripts/export_model.py --weights models/yolo26n-seg.pt --format openvino --int8 --data data/vehicles-seg.yaml
```

Impostare percorso e `perception.backend` coerentemente. TensorRT richiede una macchina NVIDIA
e un ambiente CUDA dedicato: questa installazione non include TensorRT né CUDA.
FP16/INT8 sono workflow di export, non risultati di accuratezza già validati.

RTSP si attiva con un URL `rtsp://...` o `rtsps://...`. Il decoder ha una coda corta,
conserva nel buffer i frame prima dell'eventuale scarto per inferenza e limita i tentativi
di riconnessione. Conserva i PTS relativi e segnala discontinuità dopo una riconnessione.
Non è stato verificato contro una telecamera RTSP fisica.

## Riferimenti e licenze

API di riferimento: [segmentazione](https://docs.ultralytics.com/tasks/segment/),
[ByteTrack](https://docs.ultralytics.com/reference/trackers/byte_tracker/),
[export](https://docs.ultralytics.com/modes/export/).
Versioni effettive in `uv.lock`, non la versione corrente della documentazione.

Ultralytics e i pesi hanno [condizioni di licenza proprie](https://www.ultralytics.com/license);
vedere [docs/licenses.md](docs/licenses.md). Il repository non contiene dataset CCTV di terzi.

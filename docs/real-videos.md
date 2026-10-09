# Video reali e campione ACCIDENT

La configurazione `configs/accident-image.yaml` abilita YOLO26n-seg, ByteTrack
e un detector temporale in coordinate immagine. Permette di elaborare CCTV
per cui non sono disponibili misure della strada, come i video reali ACCIDENT.

Posizioni e velocita sono espresse in pixel e px/s. Distanze, decelerazioni
e soglie di arresto sono confrontate con la diagonale apparente dei veicoli.
Non vengono stimate distanze in metri, velocita in km/h o una vista stradale metrica.
La trasformazione identita salvata in `image-reference.yaml` descrive solo
il sistema di coordinate; ha confidenza metrica zero.

Per le coppie servono convergenza o frenata con prossimita, sovrapposizione dei box
e arresto persistente dopo il possibile impatto. Per il singolo veicolo
servono decelerazione e variazione di direzione, seguite da rallentamento e arresto.
Il detector conserva conferma temporale, cooldown e controllo del movimento camera.
Le regole sono una baseline euristica; il punteggio non e una probabilita.

## Avvio

Dalla radice del progetto:

```powershell
.venv\Scripts\cctv-incident.exe run --config configs/accident-image.yaml
.venv\Scripts\cctv-incident.exe run --config configs/accident-image.yaml --source data/raw/ACCIDENT/real_videos/50uYv-SxT-o_00.mp4 --show
.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Nella UI scegliere **Solo Segmentazione**. Se esiste il manifest
del benchmark, e disponibile anche il selettore delle sole clip del campione.
Il comando `run` analizza un singolo video. Per una camera misurata resta
disponibile `configs/default.yaml`, che richiede una calibrazione valida.

## Benchmark limitato

```powershell
.venv\Scripts\python.exe scripts/benchmark_accident.py --per-class 2 --seed 42 --output outputs/accident-sample
```

Il campione predefinito contiene **10 clip reali**, due per ciascuna delle cinque
classi, circa 4 minuti e 24 secondi. Il programma accetta da una a quattro
clip per classe, mai piu di 20; non offre un'opzione per l'intero dataset.
I video selezionati vengono elaborati integralmente. Non viene usato un
ritaglio temporale centrato sull'incidente.

La selezione e deterministica: ordine SHA256 con seed, bilanciamento per tipo,
diversita giorno/notte, meteo e qualita. Usa lo split test ufficiale
`split_in_distribution`; esclude sorgenti nominali presenti anche nel train.
La sorgente ricavata dal nome del file e un identificatore approssimativo,
non una camera fisica verificata. Le clip ammesse durano 10-45 s e contengono
almeno 2,5 s prima e 3 s dopo l'impatto annotato. Questi filtri vanno considerati
nell'interpretazione delle metriche.

Il manifest, i checksum dei video e del modello, gli hash del codice e la
configurazione sono salvati prima dell'inferenza. Le annotazioni sono esportate
separatamente e usate solo nel matching finale, entro +/-2 s. Il detector riceve
immagini, timestamp e identificatori. Le soglie non vengono adattate al test.

Per preparare solo un nuovo manifest aggiungere `--prepare-only`.
Una cartella con `sample.json` non viene sovrascritta: per una nuova esecuzione
specificare un'altra directory `--output`.

## Output

- `report.md` e `summary.json`: metriche aggregate, limiti e risultati per clip.
- `sample.json`, `config.json`, `configs/`: campione e parametri riproducibili.
- `per-video.csv`: TP/FP/FN, rilevamenti, tracce, FPS, latenza p95, RAM e quota PAUSED.
- `previews/`: video H.264 annotati ai timestamp originali dei frame elaborati.
- `snapshots/` e `contact-sheet.jpg`: fotogrammi scelti dal massimo score del detector.
- `ground-truth.json` e `predictions.json`: input del confronto per evento.
- `pipeline/runs/`: log causali delle traiettorie, feature, configurazione e hardware.
- `pipeline/clips/` e `pipeline/events/`: clip degli allarmi e archivio SQLite/JSON.

L'inferenza richiede 8 FPS, con lato immagine del modello 640 e larghezza
massima del video elaborato 1280. I timestamp restano quelli del video.
Gli FPS misurati comprendono decodifica, inferenza, tracking, logging e
rendering/encoding durante l'analisi; escludono il flush finale dell'anteprima,
l'aggregazione del report e il caricamento iniziale condiviso del modello.
I p95 per frame escludono decode e buffer, che sono registrati separatamente.

Il campione contiene soltanto clip con incidenti: precision, recall e F1
descrivono questo campione; non misurano la capacita di operare nel traffico normale.
Il tasso di falsi allarmi su video negativi resta non disponibile.
Sono ancora necessari dati negativi e un validation set distinto per il tuning.
Il classificatore metrico esistente rifiuta feature in coordinate immagine.

Risultati misurati e limiti del campione: [real-video-validation.md](real-video-validation.md).

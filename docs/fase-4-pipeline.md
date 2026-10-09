# Fase 4 — Video singolo e pipeline metrica

Validazione conclusa il **7 ottobre 2026** nella `.venv` esistente.
Implementazione e verifiche automatiche completate: **231 test superati in
45,85 s**, nessun fallimento, errore o test saltato. Resta il controllo visivo
del canvas in un browser reale; non è incluso nella validazione automatica.

## Comportamento implementato

- In **Segmentazione + omografia**, la GUI richiede una calibrazione del video
  selezionato, salvata, confermata e revisionata. Passa alla pipeline percorso,
  camera, ID del record e revisione attesi. La modalità scelta nel menu prevale
  sulla modalità contenuta in uno YAML personalizzato: un file configurato in
  coordinate immagine non può eludere la revisione della calibrazione metrica.
- Il preflight rifiuta bozze, record/revisioni cambiati, video incompatibili,
  riferimenti alterati, calibrazioni non accettate/invalide, confidenza sotto
  soglia e unità diverse da `m`. Non carica un modello prima di questi controlli.
- Dopo la decodifica, usa le dimensioni native effettive e quelle elaborate.
  Adatta punti, ROI, omografia, riferimento e soglia della guardia camera prima
  di usarli: `H_elaborata = H_originale @ inverse(T)`.
- Un cambio di risoluzione o il movimento della camera invalidano la geometria,
  azzerano lo storico e sospendono il detector. Vengono registrati motivo,
  timestamp e frame; anche i frame saltati dal campionamento sono controllati
  per i cambi di risoluzione. I controlli di camera motion restano configurabili
  con gli stessi parametri precedenti e non sono stati disabilitati.
- `completed` indica il termine dell'elaborazione; `evaluable` richiede anche
  geometria valida. Invalidazioni, arresti ed errori producono esito non
  valutabile. Zero allarmi non viene presentato come negativo valido in questi
  casi. La GUI conserva il riepilogo disponibile anche dopo errori di runtime.
- Il replay conserva valutabilità e motivi, mostra `NON VALUTABILE` sul video
  pertinente e, per i nuovi record, verifica l'hash della sorgente. Le analisi
  immagine continuano a usare pixel; i vecchi YAML metrici CLI restano leggibili,
  anche quando calibrati direttamente sulla risoluzione già ridotta.

## Artefatti del run

In `outputs/.../runs/<run_id>/`:

| File | Contenuto |
| --- | --- |
| `calibration-source.yaml` | Byte esatti dell'input, per tracciabilità; i suoi percorsi originari non sono necessariamente portabili. |
| `calibration-original.yaml` | Geometria originale con riferimento locale al run. |
| `calibration-record.yaml` | Record portabile con identità/revisione e provenienza, presente per il nuovo formato. |
| `calibration-effective.yaml` | Geometria alla risoluzione elaborata, con validità finale. |
| `calibration-reference-original.png`, `calibration-reference-effective.png` | Riferimenti originale/elaborato, quando disponibili. |
| `run.json` | Configurazione, provenienza delle distanze, trasformazioni, dimensioni, soglie, invalidazioni ed esito. |
| `metrics.json` | Riepilogo con `completed`, `evaluable`, `evaluation_status` e `non_evaluable_reasons`. |
| `features.jsonl` | Include indice del frame e validità della calibrazione per ricostruire l'andamento temporale. |
| `replay.json`, `annotated.mp4` | Report e video annotato, quando viene richiesto il replay o generato dalla GUI. |

Gli snapshot consentono di ricostruire la calibrazione senza dipendere
dall'archivio centrale. Il riferimento originale conserva i pixel verificati.
I run legacy senza identità del video mantengono i controlli precedenti di
dimensione e numero di frame; non viene inventato retroattivamente un hash.

## Verifiche eseguite

- **35 test mirati superati in 31,08 s**: pipeline metrica/immagine, editor,
  riuso del record nella GUI e replay, oltre alle regressioni dell'audit.
- **231 test nella suite completa**, inclusi YOLO locale, tracking, detector,
  eventi/clip, batch e AppTest dell'app principale.
- Cinque gruppi JavaScript dell'editor superati in V8; Ruff, controllo del
  formato di 99 file e `git diff --check` superati.
- Video sintetico con piano dichiarato a 0,05 m/pixel e un bersaglio a 5 m/s:
  decodifica, segmentazione a colori, ByteTrack, omografia e replay eseguiti
  sia a 640×360 sia a 321×181. Tutte le 15 osservazioni restano entro 0,15 m
  dalla traiettoria analitica; velocità media entro 0,15 m/s dal valore atteso.
- Spostamento introdotto nel video con sfondo testurizzato: la vera guardia
  OpenCV invalida la calibrazione ridotta, il detector entra in `PAUSED` e
  non vengono più prodotte osservazioni metriche dopo l'invalidazione.
- Prove esplicite di arresto ed errore: gli artefatti registrano un esito non
  valutabile; snapshot portabile verificato anche senza archivio centrale.

La nuova fixture geometrica inizialmente usava MPEG-4 con perdita: gli
artefatti cromatici deformavano la maschera del bersaglio e confondevano errore
di segmentazione ed errore geometrico. Ora usa FFV1 senza perdita; la tolleranza
non è stata allargata. Corretta anche un'asserzione GUI sul testo dell'avviso.

Questi risultati validano il software e una geometria simulata nota, non
l'accuratezza di misure o preset su una strada reale. Le distanze della scena
sintetica sono dichiarate sperimentali. Nessuna calibrazione del dataset reale
è stata confermata automaticamente e nessun benchmark completo è stato avviato.

Rapporti: `outputs/phase4-verification/targeted.xml`, `pytest-full.xml` e
`validation-summary.json`. Windows non ha bloccato questa validazione; non
sono state cambiate protezioni, dipendenze o ambiente virtuale.

## Ripetere i test

Dalla radice del repository, in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --junitxml=outputs/phase4-verification/pytest-full.xml
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check src apps scripts tests
```

## Controllo manuale rimasto

Il browser integrato non era disponibile agli strumenti (`Browser is not
available: iab`). Per il controllo visivo, avviare:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

1. Selezionare **Segmentazione + omografia** e un video locale con un riferimento
   stradale identificabile. Prima della conferma, **Avvia analisi** deve essere
   disabilitato.
2. Nella scheda **Calibrazione**, selezionare P1–P4, trascinarli e ridimensionare
   la finestra. Numeri, lati e coordinate devono restare associati agli stessi
   punti; il disegno deve coincidere con il fotogramma.
3. Verificare ROI indipendente, annullamento e salvataggio bozza. Inserire le
   dimensioni note del riferimento, oppure dichiarare esplicitamente eventuali
   ipotesi e la loro fonte. Confermare misure, revisione e salvataggio.
4. Avviare l'analisi, verificare il video in **Rivedi analisi** e la presenza
   degli snapshot sopra elencati. Per provare il resize usare un video più largo
   di 640 pixel e una copia della configurazione con `video.max_width: 640`.
5. Riaprire l'app e lo stesso video: deve richiedere **Riutilizza calibrazione**.
   Modificare un punto o una misura: l'analisi deve richiedere una nuova conferma.

Per attestare l'accuratezza fisica servono inoltre distanze indipendenti sul
piano stradale, non usate per costruire l'omografia. Questo controllo sui dati
reali è distinto dal collaudo software della fase 4. Le fasi 5–7 e il batch
metrico restano da implementare.

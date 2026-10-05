# Contesto per riprendere il progetto

Aggiornato il **5 ottobre 2026**, fuso dell'utente **Europe/Rome**.
Questo documento riassume la conversazione e il lavoro effettivamente svolto.
È un passaggio di consegne, non una richiesta di avviare automaticamente nuovi esperimenti.
Prima di intervenire, verificare lo stato corrente dei file: potrebbe essere cambiato dopo questa fotografia.

## 1. Stato da cui ripartire

L'utente sta sviluppando un **progetto universitario di traffic incident recognition basato sulla segmentazione dei veicoli**. Comunica in italiano e lavora con VS Code su Windows.

È stata implementata e verificata una nuova modalità della GUI Streamlit che:

- analizza sequenzialmente tutti i video di una cartella;
- permette di scegliere il modello YOLO tra i pesi locali;
- confronta i risultati con le annotazioni ACCIDENT;
- esporta CSV per video, per evento e per metriche aggregate;
- conserva checkpoint ed esperimenti distinti per modello/configurazione/sottoinsieme;
- permette Stop e Riprendi, mostrando nome del video, posizione sul totale e percentuale;
- recupera il lavoro dopo riapertura della GUI o arresto del processo.

**L'implementazione è conclusa, non soltanto pianificata.** L'ultima suite completa ha riportato **128 test superati**. È stata eseguita anche una prova con YOLO26s su due video reali, interrompendo e riprendendo il primo video.

**Non è stata eseguita l'analisi completa dei 150 video.** Alla verifica effettuata per questo recap non risultano manifest in `outputs/batches/`; esiste invece l'esperimento di verifica separato in `outputs/batch-smoke/`.

L'ultima richiesta dell'utente è stata produrre questo `LLM_recap.md` per consentire a un altro LLM di ripartire dal contesto. Non ci sono chiarimenti in attesa, commit o pubblicazioni richiesti, né un benchmark completo già avviato dall'assistente da continuare.

## 2. Workspace e ambiente

Radice del progetto:

```text
C:\Users\leona\dip_crash_detection_workspace\traffic_incident_recognition-segmentation
```

- Shell: PowerShell.
- Branch al momento del recap: `main_v3`.
- HEAD: `916b5feac848950904cea141341ea333e3c7acfd`.
- Ambiente locale: `.venv`, Python 3.11.15.
- Gestore dipendenze: `uv`; specifiche in `pyproject.toml`, versioni bloccate in `uv.lock`.
- PyTorch configurato per CPU; la verifica dell'ambiente svolta nella conversazione riportava `torch 2.14.0+cpu` e CUDA non disponibile.
- Streamlit, Ultralytics e PyTorch sono ora presenti e utilizzabili. Il primo sopralluogo li aveva trovati mancanti: quella fotografia iniziale è superata.

Nel sandbox dell'assistente alcuni file delle dipendenze `.venv`, DLL PyTorch e directory temporanee di pytest restituivano `Access denied`. Gli stessi comandi hanno funzionato fuori dal sandbox con l'approvazione prevista dagli strumenti. Non interpretare automaticamente questi errori come installazione corrotta; non aggirare le restrizioni. L'uso di `-p no:cacheprovider` nei test evita anche un warning di permessi sulla cache pytest.

## 3. Architettura esistente

Flusso principale:

```text
Video/PyAV
  -> segmentazione di istanze YOLO
  -> tracking ByteTrack
  -> punto ricavato dalla maschera e traiettorie
  -> velocità, decelerazioni, contatti e cambiamenti di direzione
  -> detector temporale a regole
  -> eventi, clip, log, metriche e replay
```

YOLO segmenta i veicoli: **non è un classificatore end-to-end degli incidenti**.
La decisione di incidente è prodotta dai moduli temporali successivi. Esiste un classificatore opzionale per la modalità metrica, non attivo nel nuovo batch ACCIDENT.

| Percorso | Ruolo |
| --- | --- |
| `apps/streamlit_app.py` | Entrypoint Streamlit; modalità singolo video, demo, calibrazione, replay ed eventi, più nuovo batch |
| `src/cctv_incident/pipeline.py` | Orchestrazione dell'analisi di un video |
| `segmenter.py` | YOLO locale, maschere, filtro classi e duplicati; backend sintetico per test |
| `tracker.py`, `track_association.py` | ByteTrack e associazione tramite movimento |
| `ground_point.py`, `trajectories.py`, `features.py` | Geometria, storia del moto e feature |
| `image_event_detector.py`, `sideswipe_detector.py` | Riconoscimento in coordinate immagine |
| `event_detector.py` | Detector metrico |
| `src/cctv_incident/calibration/` | Omografia manuale/assistita e controlli della camera |
| `video.py`, `clip_buffer.py`, `storage.py`, `replay.py` | Decodifica, buffer/clip, SQLite/JSON, ricostruzione video annotato |
| `configs/` | Preset validati da `config.py` |
| `scripts/` | Download, calibrazione, training, benchmark, valutazione ed export |
| `tests/` | Test unitari e d'integrazione |
| `docs/` | Architettura, procedure e risultati storici |

Per ACCIDENT usare **coordinate immagine**, senza calibrazione metrica: posizioni e velocità sono in pixel e px/s, non metri e km/h. La demo sintetica usa maschere da colori; non misura la qualità di YOLO su video reali.

## 4. Dati locali e pesi

Percorsi reali dell'installazione corrente, relativi alla radice:

```text
dataset/real_videos/          2.027 video reali
dataset/standard_dataset/      150 video selezionati dall'utente
dataset/metadata-real.csv    2.027 righe di annotazioni
dataset/annotation_classes.yaml
```

Tutti i **150 video del sottoinsieme** sono associabili ai metadati per nome file e sono etichettati come incidenti.

Alcuni documenti storici e il selettore originale del singolo video usano `data/raw/ACCIDENT/`. Quel percorso non va confuso con la collocazione attuale `dataset/`. Il nuovo batch permette di scegliere `standard_dataset` o inserire un altro percorso. Per il singolo video restano disponibili caricamento e percorso manuale.

Pesi attualmente presenti in `models/`, con rispettivi `.sha256.json`:

| File | Dimensione in byte |
| --- | ---: |
| `yolo26n-seg.pt` | 6.719.965 |
| `yolo26s-seg.pt` | 23.467.933 |
| `yolo26m-seg.pt` | 54.750.385 |

Un errore precedente `Local model missing: .../yolo26s-seg.pt` è stato risolto scaricando esplicitamente il modello small e verificando inferenza e checksum:

```text
3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4
```

Non occorre risolvere nuovamente quell'errore se i pesi sono ancora presenti. Durante l'analisi non vengono scaricati modelli automaticamente.

## 5. Requisiti dell'utente e decisioni concordate

L'utente ha richiesto analisi di una cartella un video per volta, salvataggio CSV, confronto con `metadata-real.csv`, TP/FP/FN/TN e accuracy/precision/recall/F1, checkpoint persistenti per ciascun modello YOLO, indicazione `i/n`, percentuale del video e pulsanti Stop/Riprendi.

È stato chiesto come trattare le metriche, perché i video forniti sono tutti positivi. L'utente ha risposto esplicitamente:

> Va bene entrambi i report ma la corrispondenza deve essere dentro +- 1 secondo.

Decisioni da preservare:

1. **Due report distinti: per video e per evento.**
2. **Tolleranza temporale ±1 secondo, estremi inclusi.** Non tornare ai ±2 secondi presenti in alcune valutazioni storiche del repository.
3. La ripresa conserva i video conclusi e **ricomincia dall'inizio del video interrotto**. È stato comunicato all'utente; non è implementato il ripristino dello stato del tracker a metà video.
4. I modelli nel batch partono tutti da **8 FPS richiesti**, dimensione inferenza **640**, massimo lato orizzontale del video **1280**, CPU, tramite `configs/accident-image.yaml`. Gli FPS sono selezionabili. Non cambiare automaticamente FPS in base al modello: il confronto deve mantenere condizioni equivalenti.
5. La configurazione salvata appartiene all'esperimento: **Riprendi non applica le nuove impostazioni della sidebar**.

## 6. Metadati e valutazione

Colonne di `metadata-real.csv`:

```text
path,type,rollover,accident_time,accident_frame,center_x,center_y,
x1,y1,x2,y2,region,scene_layout,weather,day_time,quality,
no_frames,duration,height,width,split_in_distribution,split_geo_aware
```

- Una riga e un incidente annotato per video.
- `path` contiene ad esempio `real_videos/50uYv-SxT-o_00.mp4`.
- Il batch usa il **basename invariato**, perché i video sono stati copiati in un'altra directory.
- `accident_time` è il timestamp in secondi; `duration` è la durata dichiarata.
- Le cinque classi positive sono `single` (680), `t-bone` (657), `rear-end` (328), `sideswipe` (245), `head-on` (117).
- Nomi ambigui/duplicati, file senza etichetta e annotazioni non valide vengono rifiutati; assenza di etichetta non significa negativo.
- Tipo, timestamp e posizione dell'incidente annotato non vengono forniti al detector. La durata dei metadati può servire da fallback per la barra di progresso.
- Non viene valutata la classificazione del tipo di urto o la localizzazione spaziale.

### Per video

Predizione positiva se c'è almeno un evento confermato, a qualsiasi timestamp.
Verità positiva se il video ha un incidente annotato. Si calcola la matrice binaria TP/FP/FN/TN.

Con i dati attuali tutti positivi: **TN=0 e FP=0**, accuracy=recall; precision=1 quando esiste almeno una predizione positiva. Questo report non misura da solo gli allarmi fuori tempo.

Sono supportati eventuali negativi aggiunti con etichetta esplicita `normal`, `negative`, `no-accident` o `no_accident`, `accident_time` vuoto e durata positiva.

### Per evento

- Confrontare `impact_time_s` con `accident_time`, **non** il successivo `confirm_time_s`.
- TP se l'errore assoluto è ≤1 s nello stesso video.
- Matching uno-a-uno: tra più candidati scegliere il più vicino all'annotazione.
- Altri rilevamenti, inclusi duplicati nella finestra, sono FP.
- Annotazione senza corrispondenza: FN.
- Un unico allarme fuori finestra in un video positivo produce **FP=1 e FN=1** per eventi, pur essendo TP per video.
- TN e accuracy degli eventi puntuali sono **non definiti**; non inventare finestre negative o contare automaticamente i frame senza incidente.

Formule: accuracy=(TP+TN)/(TP+FP+FN+TN), precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN).
Denominatori zero producono `null` nel JSON e celle vuote nel CSV.

## 7. Modifiche implementate

### Nuovi moduli

**`src/cctv_incident/batch_evaluation.py`**

- `load_labels`: parsing e validazione delle annotazioni.
- `compare_video`: confronto binario e matching temporale entro ±1 s.
- `scores`, `aggregate_results`: metriche parziali/finali separate per unità.

**`src/cctv_incident/batch.py`**

- `prepare_job`: valida cartella/metadati/pesi, congela configurazione e manifest, identifica o recupera l'esperimento.
- Identità derivata da modello/hash, cartella/lista video, metadati, configurazione, protocollo e firma del codice.
- `start_job`: avvia `python -m cctv_incident.batch --job <directory>` in un subprocess locale, nascosto su Windows.
- `stop_job`: scrive `stop.request`, consumato come richiesta cooperativa tra fotogrammi.
- `run_job`: carica il segmenter una volta, elabora in sequenza, salta risultati già confermati, aggiorna progresso e checkpoint.
- `export_results`: rigenera CSV e metriche dai risultati completi persistenti.
- `snapshot`, `worker_alive`, `list_jobs`: stato e recupero esperimenti indipendenti dalla sessione Streamlit.
- Lock OS `msvcrt`/`fcntl` e PID con tempo di creazione impediscono avvii concorrenti e confusione con PID riutilizzati; un batch attivo per directory radice di output.
- Scritture atomiche tramite file temporaneo, flush/fsync e replace. Il JSON di un video completo è confermato prima dell'export CSV.
- Tentativi interrotti/falliti rimangono nei log della pipeline, ma non nei conteggi. Gli eventi vengono letti dal solo `run_id` completato.
- Errori fermano il batch sul video interessato senza trasformarli in FN.
- La ripresa ripara export incompleti anche quando era già stato confermato l'ultimo risultato.
- La firma del codice copre i `.py` di `src/cctv_incident/`, escluso `batch_ui.py`. Modifiche a quei file possono impedire la ripresa di esperimenti precedenti: preparare un nuovo batch, non bypassare la verifica mescolando risultati.
- Alla ripresa si verificano hash di metadati/pesi e dimensione/mtime dei video; ogni risultato conserva anche SHA256 del video analizzato.

**`src/cctv_incident/batch_ui.py`**

- Pagina batch con selezione cartella, percorso personalizzato, metadati, modelli `.pt`, FPS ed esperimenti salvati.
- `@st.fragment(run_every=1.0)` aggiorna stato e controlli senza bloccare la GUI durante l'inferenza.
- Pulsanti Prepara batch, Avvia batch, Stop, Riprendi; avanzamento cartella e video corrente.
- Download dei tre CSV e confronto fra esperimenti.
- I pesi selezionati devono essere di segmentazione; un modello di altro task viene rifiutato dal worker.

### File esistenti modificati

- `apps/streamlit_app.py`: modalità batch rinominata **standard_dataset analisi in batch - no omografia**, instradata alla pagina batch; preservate le modalità precedenti. Aggiunta anche **standard_dataset analisi in batch - con omografia**, per ora un segnaposto che mostra un messaggio senza caricare configurazioni né avviare elaborazioni. L'utente intende confrontare in futuro le prestazioni su standard_dataset con e senza omografia; il batch con omografia non è ancora implementato.
- `src/cctv_incident/pipeline.py`: `Pipeline.run` accetta `progress_callback` oltre a callback visuale e stop; summary con `stopped` e `completed`.
- `README.md`: aggiunta procedura batch e collegamento alla guida.
- `docs/batch-analysis.md`: nuova guida completa a uso, persistenza, schema CSV e definizioni delle metriche.

### Test nuovi

- `tests/unit/test_batch_evaluation.py`.
- `tests/integration/test_batch.py`.
- `tests/integration/test_batch_ui.py`.

Coprono confini ±1 s, duplicati, negativi espliciti, metadati ambigui, stop/ripresa, esclusione di tentativi parziali, separazione modelli/configurazioni, cambiamenti degli input, lock, subprocess reale, recupero dopo l'ultimo commit, stop a metà video e GUI tramite Streamlit AppTest.

## 8. File prodotti e uso della GUI

```text
outputs/batches/<modello>-<identificatore>/
  manifest.json
  checkpoint.json
  results/000000.json, ...
  videos.csv
  events.csv
  metrics.csv
  metrics.json
  worker.json
  worker.log
  stop.request              quando viene richiesto Stop
  pipeline/                 runs, traiettorie, feature, eventi, clip
```

- CSV: UTF-8, virgola come separatore, punto decimale; liste di timestamp/ID codificate come stringhe JSON.
- `videos.csv`: una riga per video completato, anche senza rilevamenti; label, esito binario, TP/FP/FN temporali, timestamp, durata, runtime, hash e split.
- `events.csv`: una riga per rilevamento del tentativo completato, con TP/FP ed errore temporale; sola intestazione quando non ci sono eventi.
- `metrics.csv`: due righe, `unit=video` e `unit=event`, con flag completo/parziale.
- Il batch non genera automaticamente il replay completo annotato per ogni video; conserva log e clip degli eventi.

Procedura utente:

1. Avviare Streamlit.
2. Scegliere **Modalita → standard_dataset analisi in batch - no omografia**.
3. Scegliere modello e cartella **standard_dataset**; CSV predefinito `dataset/metadata-real.csv`.
4. **Prepara batch → Avvia batch**.
5. Per alternare i modelli: **Stop**, attendere l'arresto, scegliere l'altro modello/esperimento e avviare o riprendere.
6. Dopo riapertura, selezionare modello ed esperimento salvato e premere **Riprendi**.

Stop attende il fotogramma in corso e il flush delle clip. **Chiudere il browser non interrompe il subprocess**: usare il pulsante Stop. Non promettere ripresa dal frame esatto.

## 9. Comandi utili

Eseguire dalla radice del repository in PowerShell.

```powershell
# Installazione riproducibile, solo se necessaria
uv sync --frozen --extra inference --extra ui --extra ml --dev

# GUI
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py

# Preparazione esplicita di un modello mancante
.\.venv\Scripts\python.exe scripts/download_model.py --output models/yolo26s-seg.pt

# Verifica modello/ambiente
.\.venv\Scripts\python.exe scripts/check_environment.py configs/accident-image.yaml

# Singolo video reale, senza calibrazione
.\.venv\Scripts\cctv-incident.exe run --config configs/accident-image.yaml --source "dataset/real_videos/50uYv-SxT-o_00.mp4" --show

# Test e controlli
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src apps scripts tests
.\.venv\Scripts\python.exe -m ruff format --check src apps scripts tests
git diff --check
```

Usare la GUI per avviare/riprendere normalmente: `start_job` si occupa anche della registrazione del processo e della rimozione della precedente richiesta Stop. L'entrypoint del worker non è un sostituto completo dei controlli di avvio della GUI.

## 10. Verifiche effettivamente eseguite

Ultima verifica prima di questa richiesta di recap:

- **128 passed in 25.42s** con pytest, senza cacheprovider.
- Ruff: **All checks passed**.
- Ruff format: **82 files already formatted**.
- `git diff --check`: nessun errore.
- GUI verificata con **Streamlit AppTest**; non è stata effettuata una verifica visuale completa in un browser reale durante l'implementazione batch.

Prova reale separata:

```text
outputs/batch-smoke/verify.py
outputs/batch-smoke/input/
outputs/batch-smoke/jobs/yolo26s-seg-2bf4b4ab84a02068/
```

Video: `6oymx7wBmjw_02.mp4` e `987C4_UdnJE_00.mp4`, circa tre secondi ciascuno, copiati dal sottoinsieme dell'utente. Modello: YOLO26s-seg, CPU.

Il primo video è stato fermato durante l'inferenza (circa 12–15%), con **0 video confermati**. Dopo Riprendi, entrambi sono stati completati e il CSV contiene esattamente due video distinti, senza incorporare il tentativo parziale.

Risultati di questa sola verifica:

| Unità | TP | FP | FN | TN | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Video | 1 | 0 | 1 | 0 | 0,5 | 1 | 0,5 | 0,6667 |
| Evento ±1 s | 1 | 0 | 1 | non definito | non definita | 1 | 0,5 | 0,6667 |

Questi due video verificano il funzionamento del batch, **non rappresentano una valutazione dell'accuratezza sui 150 video**. Lo script di verifica non è pensato per essere rilanciato sul medesimo job già completato senza predisporre un nuovo esperimento.

I report storici di `docs/real-video-validation.md` appartengono ad altre esecuzioni e protocolli; non attribuirli al nuovo batch o alla macchina attuale. Il detector resta una baseline sperimentale con falsi positivi e falsi negativi possibili.

## 11. Stato Git e precauzioni per chi riprende

Le modifiche non sono state committate. Alla verifica per questo recap:

```text
 M .gitignore
 M README.md
 M apps/streamlit_app.py
 M src/cctv_incident/pipeline.py
?? dataset/
?? docs/batch-analysis.md
?? src/cctv_incident/batch.py
?? src/cctv_incident/batch_evaluation.py
?? src/cctv_incident/batch_ui.py
?? tests/integration/test_batch.py
?? tests/integration/test_batch_ui.py
?? tests/unit/test_batch_evaluation.py
```

A questi file si aggiunge ora `LLM_recap.md`.

- La modifica a `.gitignore` e la directory `dataset/` erano già presenti prima del lavoro batch: **sono dell'utente, non vanno annullate**.
- `.gitignore` esclude `.venv/`, `outputs/`, i pesi e `*.mp4`; `dataset/` non è esclusa interamente, quindi i suoi metadati risultano non tracciati.
- Non usare `git reset`, clean, cancellazioni dei dati o commit indiscriminati per ripulire questo stato.
- Non avviare automaticamente il benchmark completo solo perché si sta leggendo questo recap.
- Non rimuovere i controlli del manifest per forzare una ripresa dopo cambiamenti del codice/dati.
- Non trasformare crash, video senza label o elaborazioni parziali in risultati negativi.

Per attività Streamlit è stata usata la skill `developing-with-streamlit`, disponibile durante la sessione in `C:/Users/leona/.agents/skills/developing-with-streamlit/SKILL.md`. La relativa documentazione è stata scoperta nell'installazione locale Streamlit; seguire le skill e istruzioni disponibili nella nuova sessione, senza presumere che gli stessi percorsi o permessi siano ancora validi.

## 12. Come continuare in una nuova sessione

1. Leggere questo recap e `docs/batch-analysis.md`, poi controllare `git status` e gli eventuali job su disco.
2. Se l'utente segnala un errore batch, partire da `checkpoint.json`, `worker.log`, `manifest.json` e dai risultati completi dell'esperimento selezionato. Verificare prima se il processo è ancora attivo.
3. Se chiede di eseguire il confronto, usare i 150 video di `dataset/standard_dataset`, il CSV locale e gli stessi parametri per ogni modello; conservare entrambi i report con tolleranza ±1 s.
4. Se chiede modifiche, preservare funzionamento singolo video, checkpoint e semantica delle metriche; eseguire test adeguati al cambiamento.
5. Non confondere il completamento del software con il completamento degli esperimenti universitari: il confronto completo fra modelli e con l'altro software resta da eseguire.

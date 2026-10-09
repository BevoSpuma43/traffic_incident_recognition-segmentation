# Piano operativo per GPT-6.1 Sol: segmentazione e omografia

Data: 6 ottobre 2026.
Stato aggiornato al 7 ottobre 2026: Fasi 0–4 implementate e verificate
automaticamente. Fase 4: **231 test superati nella `.venv` esistente**, inclusi
flusso GUI/pipeline/replay e traiettoria metrica nota a risoluzione ridotta.
Windows non ha bloccato la validazione. Resta il controllo visivo del canvas
nel browser e la verifica dell'accuratezza fisica sui dati reali.
Dettagli: [Fase 4](docs/fase-4-pipeline.md) e
[audit precedente](docs/audit-fasi-1-4.md).
Fase 5 implementata il 7 ottobre: sessioni persistenti, worker di proposte con
stop/ripresa, tabella ed editor condiviso. **248 test superati**, inclusi 17 nuovi
casi. Dettagli: [Fase 5](docs/fase-5-preparazione.md).
Fase 6 implementata: batch metrico, snapshot verificati per video, stop/ripresa,
provenienza nei CSV e blocco dei risultati con calibrazione invalida.
**271 test superati**, inclusi 23 nuovi casi, senza blocchi Windows.
Dettagli e validazione: [Fase 6](docs/fase-6-batch-metrico.md).
Fase 7: implementati compatibilità, esportazione separata, campionamento e
confronto controllato. Campione reale di 5 video: 4 con proposte, zero errori;
non è una misura di accuratezza. Validazione reale ancora parziale: assenti
misure indipendenti e calibrazioni reali confermate; browser non controllabile
dagli strumenti. Restano collaudo visivo, revisione cronometrata e confronto
reale. Procedura: [Fase 7](docs/fase-7-validazione.md).
Suite finale: **285 test superati**, 14 nuovi casi, cinque gruppi JavaScript e
controlli statici superati. I 33 file del batch storico sono invariati.
Verifiche: [baseline Fase 0](docs/fase-0-baseline.md), [backend Fase 1](docs/fase-1-calibrazione.md)
e [editor Fase 2](docs/fase-2-editor.md). Stato della validazione:
[proposte Fase 3](docs/fase-3-proposte.md).
Destinatario: agente di sviluppo GPT-6.1 Sol con accesso al repository.

## 1. Mandato e risultato atteso

Implementare un flusso di calibrazione stradale per video locali, utilizzabile sia in **Segmentazione + omografia** sia in **standard_dataset analisi in batch - con omografia**.

L'utente deve poter selezionare quattro punti sul primo frame, trascinarli per correggerli, inserire le distanze in metri, salvare la calibrazione con il nome del video e riutilizzarla. Deve inoltre poter premere **Calibrazione automatica** per ricevere una proposta di quattro punti e, dove giustificato, di distanze modificabili. La proposta deve essere revisionabile nello stesso editor della calibrazione manuale.

Il batch deve usare la calibrazione specifica di ogni video, conservare configurazioni e risultati separati per esperimento/modello e mantenere stop, ripresa e valutazione delle prestazioni.

Questo file è un piano da eseguire per fasi quando l'utente le autorizza.
La richiesta originale autorizzava la pianificazione; nella sessione successiva
l'utente ha autorizzato le Fasi 0–7 e richiesto la validazione nella `.venv`
esistente. Prima della ripresa della Fase 4 ha richiesto l'audit delle Fasi 1–4,
ora eseguito; resta la verifica visiva del componente nel browser.
Non è autorizzato dal documento
l'avvio automatico di inferenze lunghe o delle fasi successive.

## 2. Documenti da leggere e verifica iniziale

Leggere, in questo ordine:

1. Eventuali `AGENTS.md` applicabili e istruzioni del workspace.
2. Questo documento.
3. `docs/piano-calibrazione-omografia.md`: motivazioni geometriche, ricerca sulle librerie, fonti FHWA e requisiti funzionali.
4. `LLM_recap.md`, `README.md`, `docs/batch-analysis.md`, `docs/real-videos.md`, `start_tutorial.md`.
5. Codice e test elencati nella sezione 4. Il codice corrente prevale sulle descrizioni storiche dello stato del repository.

All'inizio eseguire `git status --short`, ispezionare le differenze esistenti e verificare l'ambiente `.venv`. Il workspace contiene già modifiche dell'utente e delle attività precedenti: non ripristinarle o riscriverle indiscriminatamente.

Usare la skill `developing-with-streamlit` disponibile nell'ambiente per consultare le API della versione installata. La versione minima dichiarata in `pyproject.toml` non garantisce che una nuova API sia disponibile: verificare la versione effettiva e dichiarare un requisito minimo adeguato se si adotta un componente più recente.

## 3. Decisioni già prese: non richiedere nuovamente chiarimenti

- GUI in italiano; conservare i quattro nomi delle modalità attuali. Non reintrodurre **Demo sintetica**.
- Calibrazione sul primo frame decodificabile del video. Salvare indice/timestamp effettivo.
- Quattro punti modificabili tramite trascinamento reale nel browser, non soltanto campi numerici o copia di JSON.
- Prima versione basata sui vertici di un rettangolo fisico sul piano stradale, con larghezza e lunghezza. Il rettangolo può apparire trapezoidale in prospettiva.
- Conservare un eventuale editor avanzato con corrispondenze metriche esplicite, senza complicare il flusso principale.
- Calibrazioni archiviate per video e riutilizzabili fra modelli YOLO; conferma visiva al riutilizzo, senza obbligare a selezionare nuovamente i punti.
- Misure note, ipotesi da standard e stime sperimentali devono rimanere distinguibili.
- Se mancano riferimenti, non inventare distanze o punti: mostrare il motivo e consentire la calibrazione manuale.
- Batch sequenziale; il video interrotto riparte dall'inizio. I video completati non vengono ricalcolati nella ripresa.
- Ogni nuovo esperimento ha una cartella unica; **Riprendi** continua un esperimento esistente con la configurazione salvata.
- Valutazione per video e per evento; associazione temporale **±1 secondo inclusivo**. Non modificare il protocollo per favorire la nuova modalità.
- Nessuna inferenza sullo stato di incidente a partire dalle etichette durante calibrazione o rilevamento. I metadati degli incidenti servono alla valutazione.
- Video non calibrabile, errore tecnico o calibrazione invalidata non equivalgono a una predizione negativa.

Provenienza del sottoinsieme verificata nella sessione: 150 video, di cui 96 con località statunitensi, 46 `World`, 8 `UAE`. Ricalcolare questi conteggi se il dataset cambia. Non trattare `World` come USA.

## 4. Mappa del codice e interventi previsti

Tutti i percorsi `calibration/...` riportati sotto sono relativi a `src/cctv_incident/`.

| File esistente | Stato verificato | Intervento |
|---|---|---|
| `apps/streamlit_app.py` | Quattro modalità; batch metrico fermato da `st.stop()`; calibrazione JSON e proposta da copiare | Integrare editor, controllo preliminare e apertura della modalità batch metrica |
| `calibration/homography.py` | `Calibration`, `estimate_calibration`, `load_calibration`; quattro punti con OpenCV | Validazione, adattamento delle coordinate, caricamento compatibile |
| `calibration/assisted.py` | `propose_calibration`, proposta non accettata; misure mancanti in unità canoniche | Restituire candidati geometrici e provenienza delle misure senza convertirli implicitamente in metri |
| `calibration/lane_mask.py` | Filtri bianchi/gialli, morfologia e skeleton | Riutilizzare e rendere configurabile la zona cercata |
| `calibration/line_fitting.py` | Hough, famiglie basate sugli assi immagine e linee estreme | Raggruppamento prospettico robusto e costruzione di candidati coerenti |
| `calibration/background.py` | Sfondo mediano da campioni nel video | Non chiamarlo implicitamente nel nuovo flusso sul primo frame |
| `calibration/quality.py` | Guardia per movimento telecamera | Integrare invalidazione esplicita e coerente col batch |
| `config.py` | Modelli Pydantic con `extra=forbid` | Aggiungere solo impostazioni necessarie, con default compatibili |
| `pipeline.py` | Carica un file di calibrazione, confronta `camera_id`, rifiuta risoluzioni diverse | Usare calibrazione corretta e adattata al frame elaborato |
| `batch.py` | `prepare_job`, `run_job`, `validate_manifest`, `export_results`; accetta solo coordinate immagine | Estendere protocollo, snapshot per video, verifica e risultati metrici |
| `batch_ui.py` | Scelta modello/cartella, subprocess, monitor e checkpoint | Preparazione calibrazioni, separazione modalità e riuso del monitor |
| `batch_evaluation.py` | Confronti video/evento già presenti | Conservare la matematica del protocollo; aggiungere metadati altrove |
| `video_inputs.py` | Elenco video e upload | Riutilizzare identificazione e lettura percorsi |

Moduli nuovi proposti, da accorpare se una separazione non porta benefici:

- `calibration/records.py`: documento versionato, bozze, identità e provenienza delle misure.
- `calibration/repository.py`: ricerca, salvataggio atomico, revisioni e verifica compatibilità.
- `calibration/coordinates.py`: trasformazioni immagine originale/elaborata e omografia.
- `calibration/presets.py`: riferimenti dimensionali documentati e condizioni di applicazione.
- `calibration_ui.py` e `components/calibration_editor/`: flusso Streamlit e componente grafico.
- `calibration/preparation.py`: preparazione persistente delle proposte per una cartella.

Non introdurre un secondo motore di inferenza batch: estendere quello esistente.

## 5. Architettura e contratti

### 5.1 Separare bozza, calibrazione confermata e oggetto runtime

Una bozza può avere meno di quattro punti o distanze mancanti. La classe `Calibration` attuale non è adatta a rappresentarla perché richiede una geometria valida.

Proposta: un documento `CalibrationRecord` versionato contiene identità del video, stato, dati dell'editor, misure e metadati. Il campo con la `Calibration` runtime è assente finché non esistono corrispondenze valide. La pipeline deve ricevere soltanto una calibrazione confermata, valida e con scala esplicitamente definita.

Campi minimi proposti:

| Gruppo | Campi |
|---|---|
| Versione | `schema_version`, `record_id`, `revision`, date creazione/modifica |
| Video | percorso relativo, nome originale, SHA-256, dimensioni originali |
| Riferimento | indice/timestamp primo frame, percorso immagine e suo hash |
| Editor | punti originali in pixel, ID stabili P1–P4, ROI indipendente, larghezza/lunghezza |
| Misure | valore, unità, origine per ciascuna distanza, fonte/preset, conferma dell'utente |
| Automazione | metodo, versione, parametri, seed, proposta iniziale e diagnostica |
| Stato | `draft`, `confirmed`, `invalid`; motivi di incompatibilità separati |
| Runtime | oggetto `Calibration` con punti, destinazioni, matrice e unità |

Usare origine per singola distanza: una può essere misurata e l'altra assunta. Distinguere la conferma umana dalla qualità geometrica e dall'accuratezza metrica. Non portare automaticamente a 1 il punteggio perché l'utente ha premuto conferma.

Mantenere leggibili i vecchi YAML di `Calibration`. L'importazione nell'archivio per video richiede associazione verificata al video e indicazione della provenienza; non assegnare automaticamente una vecchia configurazione generica a qualunque sorgente.

### 5.2 Archivio

Percorso principale: `data/calibration/videos/<video-stem>.yaml`, con immagine di riferimento associata. In caso di collisione usare sottocartelle per dataset/identità, mantenendo il nome del video nel file YAML. Gestire anche `clip.mp4` e `clip.avi`, upload rinominati e nomi non validi su Windows.

Il percorso individua un candidato; la compatibilità dipende dal contenuto e dalla risoluzione del video. Un video sostituito mantenendo il nome non deve riutilizzare silenziosamente la calibrazione precedente.

API indicative, non firme vincolanti:

```python
inspect_video(source) -> VideoIdentity
load_record(path) -> CalibrationRecord
find_record(identity) -> CompatibilityResult
save_record(record, root) -> Path
confirm_record(draft) -> CalibrationRecord
to_runtime_calibration(record) -> Calibration
```

Usare scrittura temporanea e sostituzione atomica. Per YAML e immagine: scrivere prima la nuova immagine con nome di revisione, poi pubblicare il YAML che la referenzia. Non creare finestre in cui il file principale punta a un'immagine non ancora disponibile.

### 5.3 Coordinate e dimensioni

Salvare sempre i punti nella risoluzione originale. La GUI può ridimensionare il frame, ma deve usare la trasformazione effettiva, compresi eventuali margini. Non salvare coordinate CSS come coordinate del video.

Definire `T` come trasformazione da pixel originali a pixel elaborati. Se `H_original` porta dalla sorgente al piano metrico:

```text
p_processed = T * p_original
H_processed = H_original * inverse(T)
roi_processed = T * roi_original
```

Adattare anche immagine di riferimento e parametri espressi in pixel usati dalla guardia del movimento. Non sovrascrivere il record originale. Usare le dimensioni effettive del decoder, non soltanto il `max_width` richiesto. Verificare orientamento e metadati di rotazione tra il primo frame estratto e i frame della pipeline.

Il rettangolo di calibrazione non determina implicitamente la ROI di valutazione. Esporre la ROI separatamente e conservare la stessa area nei confronti con/senza omografia. Se si aggiunge un filtro ROI comune, mantenerlo opzionale e preservare il comportamento preesistente quando non è configurato.

## 6. Sequenza di implementazione

### Fase 0 — Ricognizione e baseline

- Verificare stato del repository, versioni Python/Streamlit/OpenCV e test pertinenti.
- Rileggere le funzioni nella sezione 4 e la struttura di un manifest esistente senza modificarlo.
- Controllare se sono presenti batch attivi prima di modificare runtime condiviso; non terminare processi dell'utente implicitamente.
- Registrare anomalie già presenti distinguendole dalle regressioni introdotte.

Completamento: elenco preciso dei punti di integrazione, ambiente utilizzabile e baseline dei test mirati.

### Fase 1 — Record, archivio e geometria

- Implementare schema per bozza/conferma e caricamento dei vecchi file.
- Estrarre il primo frame con PyAV, chiudendo sempre il decoder; gestire video vuoti/non decodificabili.
- Implementare identità, ricerca e salvataggio delle calibrazioni.
- Implementare costruzione delle quattro corrispondenze rettangolari e trasformazione delle coordinate.
- Verificare punti finiti/distinti, area, convessità, ordine, distanze positive e matrice non degenere.
- Conservare gli ID dei vertici: riordinamenti dopo un trascinamento non devono scambiare larghezza e lunghezza.

Completamento: round trip su disco, incompatibilità rilevata per video sostituito e distanze invarianti dopo resize.

### Fase 2 — Editor grafico riutilizzabile

- Scegliere il componente usando la documentazione della versione Streamlit installata. Preferire un componente locale senza CDN; dichiarare le dipendenze effettive.
- Visualizzare frame, punti numerati, poligono, lati etichettati e, separatamente, ROI.
- Consentire quattro clic per inizializzare, trascinamento, campi numerici alternativi, reset e annulla ultima modifica.
- Gestire i movimenti nel browser e inviare al backend uno stato coerente al rilascio; evitare inferenze ad ogni movimento.
- Usare identità video e revisione nell'identità del componente. Cambiare video deve azzerare frame, proposta e stato dell'editor non pertinente.
- Invalidare l'accettazione precedente dopo qualunque modifica a punti, distanze o ROI.
- Mostrare anteprima con griglia metrica e limite alle dimensioni del rendering, per evitare immagini enormi con distanze mal inserite.
- Consentire il salvataggio di una bozza incompleta; abilitare conferma e analisi solo con dati validi.

Completamento: quattro punti realmente trascinabili; coordinate corrette anche ridimensionando la finestra; stato coerente dopo un rerun.

### Fase 3 — Proposta automatica

- Riutilizzare maschere e primitive esistenti di OpenCV/scikit-image. Non aggiungere subito un modello pesante.
- Rilevare segmenti e contorni, raggruppare direzioni prospettiche e scartare outlier. Gestire anche punti di fuga all'infinito.
- Cercare prima riferimenti planari con due direzioni e supporto visivo sufficiente, come barre e strutture di attraversamenti; valutare corsie/tratteggi senza imporre rettangoli non supportati dalla scena.
- Restituire un numero limitato di candidati, con quattro punti, tipo di riferimento, linee di supporto, motivi di qualità/scarto e misure eventualmente suggerite.
- Separare qualità geometrica e disponibilità della scala. Una sola famiglia di linee parallele non basta per la calibrazione metrica completa.
- Non usare il valore canonico 10×30 del vecchio fallback come misura in metri.
- Implementare preset USA documentati, selezionabili e correggibili; non attivarli sulla sola etichetta `World` o su `UAE`.
- Collegare ogni misura suggerita al tratto selezionato: un rettangolo che comprende più segmenti non ha la lunghezza di un solo tratteggio.
- Lasciare vuote le dimensioni non determinabili. La conferma di un'ipotesi non la trasforma in una misura osservata.
- Popolare direttamente l'editor dal candidato scelto. Mostrare alternative quando disponibili.
- Non usare automaticamente sfondo mediano da tutto il filmato o dimensioni universali delle automobili.

Completamento: pulsante funzionante, proposta modificabile e fallback manuale comprensibile per scene senza riferimenti. Un risultato vuoto motivato è corretto quando la geometria non è supportata.

### Fase 4 — Video singolo e pipeline

- In modalità metrica eseguire ricerca/revisione della calibrazione prima dell'analisi.
- Impostare percorso e `camera_id` dalla configurazione selezionata, evitando il riuso della camera generica del YAML.
- Verificare esplicitamente `units == "m"`, accettazione e validità; le unità canoniche non abilitano i detector metrici.
- Adattare la calibrazione alla risoluzione elaborata prima delle verifiche dimensionali e dell'uso nella pipeline.
- Salvare nell'output del run calibrazione originale, calibrazione effettiva o trasformazione applicata, identità/revisione e provenienza delle distanze.
- Se la calibrazione perde validità durante il video, rendere esplicito l'esito non valutabile; non presentare assenza di allarmi come un negativo valido.
- Conservare gli attuali controlli per camera motion; non disabilitarli per far passare il flusso.
- Conservare il comportamento della modalità immagine e dei percorsi CLI preesistenti.

Completamento: video metrico con risoluzione ridotta analizzabile e artefatti sufficienti a ricostruire la calibrazione effettiva.

### Fase 5 — Preparazione persistente delle calibrazioni batch

- Tabella per video con stati mancante/bozza/confermato/incompatibile e origine delle misure.
- Apertura dello stesso editor per ogni riga; azione per passare al prossimo video da revisionare.
- Pulsante **Proponi calibrazioni mancanti** con worker persistente, progressi e stop/riprendi. Preferire un processo separato per non vincolare il lavoro ai rerun Streamlit.
- Salvare indice, stato e risultato di ogni proposta in una sessione di preparazione separata dall'esperimento di inferenza.
- Una proposta confermata non deve essere sovrascritta da una successiva generazione automatica.
- Nessuna conferma automatica in massa: la generazione produce bozze da revisionare.
- Controllare input e hash alla ripresa; il video in lavorazione può essere riproposto dall'inizio, quelli già salvati vengono saltati.
- Errori di preparazione restano nella tabella, senza creare risultati di classificazione.

Completamento: chiusura/riapertura GUI e stop/riprendi non perdono le proposte già salvate.

### Fase 6 — Batch metrico e snapshot

- Estendere `prepare_job` con una mappa esplicita video → calibrazione; conservarne l'opzionalità per il batch immagine.
- Verificare tutte le calibrazioni prima di pubblicare un nuovo esperimento pronto. Non saltare implicitamente i video senza calibrazione.
- Introdurre una nuova versione del manifest. Per ogni video salvare riferimento alla copia locale della calibrazione, hash, identità sorgente, revisione e origine delle misure.
- Copiare YAML e immagini di riferimento nella cartella del job, mantenendo risolvibili i percorsi relativi. Verificare i file copiati prima di rendere avviabile il job.
- Registrare la modalità in configurazione e identità dell'esperimento. Separare le liste GUI per modello e modalità, evitando chiavi di widget condivise accidentalmente.
- In `run_job`, prima di ciascun video, impostare `cfg.calibration.file` e `camera_id` dalla copia del manifest; non dalla cartella centrale modificabile.
- Estendere `validate_manifest` alla verifica delle copie. Una modifica successiva all'archivio centrale non invalida lo snapshot; una modifica allo snapshot sì.
- Se un video perde la calibrazione o termina con geometria invalida, interrompere con stato spiegato senza commit del risultato. La correzione della calibrazione implica un nuovo esperimento.
- Conservare il commit atomico per video, l'esportazione ricostruibile e il riuso del modello YOLO nel worker.
- Estendere output e CSV con modalità, riferimento/hash calibrazione e origine della scala. Per vecchi risultati usare valori assenti espliciti senza inventare metadati.
- Abilitare la voce batch con omografia in `apps/streamlit_app.py` solo quando il flusso è completo.

Completamento: due video con calibrazioni diverse, due modelli e due modalità producono esperimenti indipendenti; stop/riprendi non duplica risultati.

### Fase 7 — Compatibilità, valutazione e documentazione

- Aggiornare caricamento ed esportazione dei vecchi manifest, mantenendo consultabili risultati e metriche già prodotti.
- Non aggirare `code_signature()`: oggi la ripresa richiede lo stesso hash del codice. Non riscrivere gli hash di vecchi job per consentire una ripresa con logica diversa.
- Distinguere la possibilità di leggere un esperimento dalla possibilità di continuarlo. Per una vecchia esecuzione incompatibile, conservare gli artefatti e spiegare che la ripresa richiede il codice originale; un nuovo esperimento usa il codice nuovo.
- Verificare un campione di scene reali senza usare le etichette degli incidenti per scegliere punti o parametri.
- Misurare copertura delle proposte, correzioni necessarie e tempo di revisione. Misurare errore metrico solo dove esiste una misura indipendente: non dichiarare ground truth un preset stradale.
- Confrontare le due modalità sullo stesso insieme di video, area, modello e FPS; dichiarare eventuali differenze nelle regole dei detector.
- Aggiornare documentazione, esempi di configurazione e `LLM_recap.md` con stato realmente verificato e limiti rimasti.

Completamento: suite pertinente superata, verifica GUI reale documentata e istruzioni riproducibili per utente e prossimo agente.

## 7. Strategia di test

| Area | Caso significativo | Risultato atteso |
|---|---|---|
| Geometria | Rettangolo noto proiettato e punti indipendenti | Distanze recuperate entro tolleranza numerica dichiarata |
| Resize | Stessa scena a due risoluzioni, inclusi arrotondamenti | Coordinate metriche equivalenti e ROI coerente |
| Punti | Incroci, duplicati, quasi collinearità, valori non finiti | Messaggio specifico e nessuna conferma |
| Scala | Distanze mancanti o unità canoniche | Bozza salvabile, analisi metrica bloccata |
| Persistenza | Salva/carica, collisione nomi, sorgente sostituita | Nessuna associazione errata al video |
| Provenienza | Un lato misurato e uno assunto | Distinzione conservata in YAML e risultati |
| Automatico | Riferimenti noti, outlier e scena vuota | Candidato coerente o rifiuto motivato |
| Editor | Drag, resize finestra, reset e cambio video | Nessun punto obsoleto o distanza associata al lato errato |
| Pipeline | Calibrazione invalidata durante inferenza | Esito tecnico esplicito, nessun negativo valutabile |
| Batch | Due video con omografie differenti | Ogni esecuzione usa la copia corretta |
| Ripresa | Stop a metà video e riavvio worker | Video completati saltati, corrente ricominciato, nessun doppio commit |
| Snapshot | Archivio modificato e copia di job alterata | Archivio indipendente; copia alterata rifiutata |
| Crash | Dopo commit risultato, prima export CSV | CSV ricostruito senza duplicati |
| Legacy | Vecchi YAML/manifest e nuovi campi mancanti | Lettura supportata e politica di ripresa esplicita |
| Metriche | Confini temporali a ±1 s e appena oltre | Stesso comportamento del protocollo attuale |

Creare test unitari per la logica nuova e integrazioni per i passaggi a rischio. Riutilizzare fixture e factory di `tests/integration/test_batch.py`. Evitare caricamenti ripetuti dei pesi YOLO nei test unitari.

File proposti: `tests/unit/test_calibration_records.py`, `test_calibration_coordinates.py`, `test_calibration_proposals.py`; `tests/integration/test_calibration_ui.py`, `test_metric_batch.py`. Estendere i test esistenti dove verificano già il comportamento interessato.

Streamlit AppTest può verificare flusso e stato Python, ma non basta per dimostrare il trascinamento del componente JavaScript. Effettuare anche un controllo nel browser con almeno drag, resize e cambio sorgente; se gli strumenti disponibili non lo consentono, dichiarare quella verifica ancora mancante.

Comandi iniziali in PowerShell, da adattare solo ai file effettivamente presenti:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_geometry.py tests/unit/test_batch_evaluation.py tests/integration/test_batch.py tests/integration/test_batch_ui.py tests/integration/test_ui.py tests/integration/test_tracking_calibration.py
.\.venv\Scripts\python.exe -m ruff check apps src tests
```

Eseguire prima i test della fase modificata, poi le regressioni pertinenti. Non effettuare un batch completo dei 150 video come semplice verifica del codice. Un eventuale smoke test reale deve essere breve, isolato in un nuovo output e distinto dalla valutazione scientifica.

## 8. Rischi tecnici da risolvere esplicitamente

1. **Confidenza ingannevole:** quattro corrispondenze possono produrre errore di riproiezione quasi nullo pur con misure sbagliate. Non usarlo come unica prova di qualità.
2. **Risoluzioni diverse:** oggi la pipeline genera un errore; risolvere con la trasformazione corretta, non eliminando il controllo.
3. **Stato GUI:** non ricostruire o perdere la bozza ad ogni rerun; invalidare soltanto ciò che è diventato incoerente.
4. **Camere e file:** `camera_id`, stem del video e identità del contenuto hanno ruoli diversi. Non confonderli per riutilizzare calibrazioni.
5. **Calibrazione e ROI:** non restringere accidentalmente il detector al solo rettangolo dei quattro punti.
6. **Invalidazione a metà video:** un run può arrivare alla fine senza misure valide; il worker deve verificarne l'idoneità alla valutazione prima del commit.
7. **Hash del codice:** anche nuovi file Python nel package possono cambiare la firma. Conservare la protezione della riproducibilità.
8. **Standard stradali:** i valori sono ipotesi documentate, non misure certificate della scena; verificare tipo di segnaletica, luogo e fonte.
9. **Sfondo mediano:** non introdurre uso implicito di frame futuri nella calibrazione sul primo frame.
10. **Dipendenze/UI:** controllare il packaging degli asset locali del componente e l'avvio dall'ambiente di progetto, non soltanto dalla directory sorgente.

## 9. Fuori dalla prima versione

- Calibrazione completa da dimensioni medie di automobili o ricostruzione 3D.
- Addestramento di un nuovo modello di segnaletica prima di aver valutato le primitive disponibili.
- Telecamere mobili, rettifica dinamica, più piani stradali o ricostruzione di strade curve.
- Riprogettazione dei detector, modifica della tolleranza temporale o tuning sulle etichette del test.
- Ripresa esatta del tracker dal frame interrotto: rimane la ripartenza dall'inizio del video incompleto.

Se una di queste estensioni diventa necessaria per una scena, registrare il limite anziché simulare una calibrazione affidabile.

## 10. Checklist conclusiva dell'agente implementatore

- [ ] Quattro punti manuali sul primo frame e distanze in metri.
- [ ] Quattro punti trascinabili e anteprima aggiornata senza perdita dello stato.
- [ ] Pulsante automatico che popola l'editor con una proposta o spiega l'insuccesso.
- [ ] Origine delle distanze visibile e salvata, senza valori metrici inventati.
- [ ] Archivio per video, riutilizzo e rilevamento di incompatibilità.
- [ ] Pipeline metrica corretta anche con resize e controlli di validità.
- [ ] Preparazione batch con bozze persistenti, stop e ripresa.
- [ ] Batch metrico con copie immutabili per video e cartelle separate per esperimento.
- [ ] Risultati non duplicati; errori tecnici esclusi dalle confusion matrix.
- [ ] Report video/evento invariati nel protocollo ±1 secondo.
- [ ] Nessuna regressione verificata nella modalità senza omografia.
- [ ] Vecchi risultati preservati; limiti di ripresa dovuti al codice dichiarati.
- [ ] Test mirati e controllo del componente nel browser eseguiti o limiti esplicitati.
- [ ] README, tutorial, documentazione batch e recap aggiornati.

## 11. Prompt pronto per avviare il lavoro in una nuova sessione

> Leggi `PLANNING_GPT_6_1_SOL.md`, `docs/piano-calibrazione-omografia.md`, le istruzioni del repository e il codice indicato. Implementa il piano per la calibrazione manuale/automatica con quattro punti trascinabili e distanze modificabili, integrandolo nel video singolo e nel batch con omografia. Conserva modifiche esistenti, esperimenti e protocollo ±1 secondo. Procedi per fasi verificabili, aggiorna la checklist e documenta ciò che è stato realmente testato. Non riscrivere gli hash degli esperimenti per aggirare controlli di compatibilità. Usa decisioni già definite e chiedi chiarimenti solo se emerge un'ambiguità che cambia materialmente il risultato. Non dichiarare completate funzioni o verifiche non eseguite.

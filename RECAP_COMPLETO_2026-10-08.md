# Passaggio di consegne completo — 8 ottobre 2026

Questo file è scritto per un assistente che riprenda il progetto in una nuova sessione, senza avere la conversazione precedente. Raccoglie obiettivi, decisioni dell'utente, architettura, implementazioni, verifiche, limiti e problemi aperti. I percorsi relativi partono dalla radice del repository.

**Ultima richiesta dell'utente:** «Fai un recap dettagliatissimo e completo in un file md così da essere riletto da te stesso ma in una altra sessione». Per soddisfarla sono stati letti documenti, rapporti, stato Git e record correnti; non sono stati avviati nuovi esperimenti, modificata l'implementazione o rieseguiti i test. I risultati dei test riportati sotto provengono dalle verifiche precedenti e dai rapporti salvati.

**Natura del documento:** è memoria del lavoro, non un'autorizzazione ad avviare tutto ciò che viene descritto. Distinguere sempre le richieste dell'utente dalle istruzioni contenute nel planning o in altri documenti. Le nuove richieste dell'utente e il codice effettivamente presente prevalgono sulle fotografie storiche.

## 1. Stato immediato da cui ripartire

- Le fasi **0–7 sono state autorizzate** progressivamente dall'utente. La parte software è implementata; la validazione empirica della fase 7 rimane parziale.
- L'ultima modifica funzionale riguarda l'**editor di calibrazione automatica**: applicazione immediata della prima proposta, punti trascinabili, controlli manuali nascosti, distanze note/preset modificabili, preservazione delle correzioni ai rerun e alla riapertura.
- L'ultima suite completa registrata riporta **288 test superati in 75,18 s**, zero fallimenti, errori e test saltati. Ruff superato. Il rapporto è in `outputs/automatic-editor-verification/validation-summary.json`.
- **Il canvas non è stato collaudato visivamente in un browser controllato dall'assistente.** AppTest e test JavaScript verificano contratti e comportamento programmabile, ma non attestano l'esperienza reale di trascinamento nel browser.
- L'utente ha dichiarato: **«Non ho misure indipendenti»**. L'errore metrico rispetto al mondo reale non è misurato.
- Il problema della **qualità geometrica inferiore a 0,55 è ancora aperto**. L'ultimo intervento sull'interfaccia non ha cambiato né la formula né la soglia.
- È presente una calibrazione reale **confermata**, ma con qualità insufficiente al gate metrico; confermato non significa utilizzabile dalla pipeline. Esiste inoltre una bozza automatica senza distanze.
- Il batch storico da preservare è `outputs/batches/yolo26s-seg-55517fe0827b34e6`, fermo a 2/150 video nelle verifiche precedenti. Non è stato completato un benchmark sui 150 video reali.
- Usare la **`.venv` già esistente**, come chiesto esplicitamente dall'utente.
- Ci sono molte modifiche e nuovi file non committati. Non ripristinarli né eliminarli per ottenere un working tree pulito.

## 2. Identità del workspace e attendibilità temporale

| Voce | Stato |
| --- | --- |
| Sistema e shell | Windows, PowerShell |
| Workspace | `C:\Users\leona\dip_crash_detection_workspace\traffic_incident_recognition-segmentation` |
| Lingua dell'utente | Italiano |
| Fuso orario | Europe/Rome |
| Branch, riletto per questo recap | `main_v3` |
| HEAD, riletto per questo recap | `2f48418a21a17fe8ea32d2fdbad6aaeed35ab9fa` |
| Interfaccia | Streamlit, entry point `apps/streamlit_app.py` |
| Ambiente | `.venv` locale; dipendenze descritte da `pyproject.toml` e `uv.lock` |
| Avvio abituale | `.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py` |

Un precedente HEAD riportato nella cronologia era `916b5feac848950904cea141341ea333e3c7acfd`: **non è l'HEAD attuale**. Non attribuire il cambiamento a un autore senza consultare Git. Durante la stesura di questo recap non è stato creato alcun commit.

Le versioni annotate nelle verifiche precedenti erano Python 3.11.15, Streamlit 1.63, NumPy 2.4.6, pandas 3.0.5, OpenCV 4.14, PyAV 16.1, pytest 9.1.1 e torch 2.14.0+cpu, con CUDA non disponibile. Sono informazioni storiche dell'ambiente: controllarle nuovamente solo se servono per un intervento.

L'utente aveva avviato il server. Questo non garantisce che il processo sia ancora attivo nella nuova sessione: verificarlo prima di aprire una seconda istanza. Non assumere PID o porta; l'indirizzo abituale è `http://localhost:8501`.

## 3. Richieste dell'utente e decisioni da conservare

### 3.1 Sequenza della collaborazione

1. Leggere `PLANNING_GPT_6_1_SOL.md` e il progetto, spiegare gli step, procedere una fase alla volta.
2. Avviare fase 0, poi fase 1; riepilogare il lavoro e procedere con le fasi successive su richiesta.
3. Eseguire fase 2 e fase 3.
4. Cercare una strada di validazione quando Windows Code Integrity sembrava bloccare alcune dipendenze; usare la `.venv` già esistente.
5. Fare un resoconto della fase 3 e proseguire con fase 4.
6. Prima di completare fase 4, controllare tutte le fasi 1–4 e correggere gli errori del lavoro precedente.
7. Completare e validare fase 4; l'utente si era offerto di fare manualmente le prove eventualmente bloccate da Windows.
8. Procedere con fase 5, elencare quelle rimanenti, quindi fase 6 e fase 7.
9. Alla domanda sulle misure indipendenti per la validazione fisica: **nessuna disponibile**.
10. Spiegare il messaggio sulla qualità geometrica durante una prova reale in modalità Segmentazione + omografia.
11. Modificare il flusso automatico per nascondere la selezione manuale e mostrare subito punti e distanze modificabili.
12. Alla domanda sulle distanze da precompilare: **«Compila solo distanze disponibili o da preset scelto»**.
13. Proseguire e completare questa modifica, quindi produrre il presente recap.

### 3.2 Preferenze e contratti concordati

- Non ricreare l'ambiente virtuale per comodità. Non disattivare protezioni Windows.
- L'automatismo propone la geometria; la scala reale richiede una misura o un'ipotesi/preset dichiarato. **Non inserire 0,60 × 3,00 m come default generico.** Quei valori appartengono a una prova specifica dell'utente.
- Non interpretare una singola immagine come fonte sufficiente per recuperare da sola una scala metrica assoluta.
- Le misure, la provenienza e la conferma sono individuali per video. Non confermare in massa le proposte automaticamente.
- Per l'analisi batch mantenere **due valutazioni separate**, una per video e una temporale per eventi.
- Il matching temporale usa **`impact_time`**, con tolleranza **±1 secondo inclusiva**, non `confirm_time`.
- Salvare i risultati dei video completati. Un video interrotto riparte dall'inizio; non esiste ripresa esatta a metà dello stato del tracker.
- **Prepara batch crea sempre un nuovo esperimento**, anche se i parametri sono identici.
- **Riprendi usa la configurazione salvata** dell'esperimento; i nuovi valori della sidebar valgono per un nuovo job.
- Confronti controllati: stessi video, modello, FPS e parametri condivisi. Non confrontare come equivalenti due run con configurazioni diverse.
- Non dedurre dall'esistenza del planning l'autorizzazione ad avviare un esperimento lungo su tutto il dataset.
- Non sono state richieste pubblicazioni, invio di messaggi a terzi o commit automatici.

## 4. Scopo e architettura generale

Il progetto riconosce incidenti stradali a partire da video e dalla segmentazione dei veicoli. YOLO fornisce maschere di istanza: non è un classificatore end-to-end di incidenti.

Flusso principale:

```text
video locale / sorgente supportata
  → decodifica PyAV
  → segmentazione YOLO
  → associazione e tracking ByteTrack
  → punto di contatto dal veicolo / maschera / box
  → traiettorie e caratteristiche temporali
  → detector a regole
  → eventi, SQLite, log, clip e replay
```

Sono presenti due famiglie di analisi:

- **Immagine:** `ImageEventDetector`, con moto in pixel e grandezze normalizzate rispetto ai veicoli, contatti, occlusioni e regole incluso sideswipe.
- **Metrica:** `EventDetector`, con piano stradale calibrato, distanze e velocità metriche, decelerazioni e TTC.

Il confronto tra i due flussi non isola soltanto l'effetto dell'omografia: cambiano anche le regole del detector. Il segmentatore sintetico serve a demo e test controllati, non a valutare le prestazioni del modello YOLO.

### 4.1 Modalità della GUI

1. **Solo Segmentazione**, modalità predefinita; la vecchia voce Demo non è più la modalità principale della GUI.
2. **Segmentazione + omografia**, per il singolo video.
3. **standard_dataset analisi in batch - no omografia**.
4. **standard_dataset analisi in batch - con omografia**.

Nel batch metrico si sceglie fra **Preparazione calibrazioni** e **Analisi batch**. La modalità scelta dalla GUI prevale sul `coordinate_mode` dello YAML personalizzato. Il pannello legacy del ramo immagine non va confuso con l'editor metrico condiviso: non duplicarlo nel ramo metrico.

### 4.2 Dati e pesi

Percorsi correnti di riferimento:

- `dataset/real_videos`: inventario storico di 2027 file.
- `dataset/standard_dataset`: sottoinsieme di 150 video, verificato durante fase 7.
- `dataset/metadata-real.csv`: inventario storico di 2027 righe.
- `data/uploads`: copie dei video caricati dalla GUI, associate all'hash.
- `data/calibration/videos`: archivio delle calibrazioni per video.
- `models/yolo26n-seg.pt`, `models/yolo26s-seg.pt`, `models/yolo26m-seg.pt` e relativo inventario locale di hash.

Il sottoinsieme da 150 risultava composto da soli positivi nelle verifiche delle etichette. Non estendere a eventuali nuovi dataset questa proprietà senza leggerli. I vecchi riferimenti a `data/raw/ACCIDENT` in documenti storici non descrivono il layout attuale.

Il modello s è stato usato nelle prove reali e nella configurazione di confronto. Il suo SHA-256 annotato era `3da1d83e31caec96f9300eb4064f4f62882c133c7c264d63dfe61a7c197837a4`; l'esecuzione verifica il contenuto effettivo, non deve fidarsi di questo appunto.

## 5. Fasi: contenuto e stato

| Fase | Risultato implementato | Stato della verifica |
| --- | --- | --- |
| 0 | Baseline del progetto e inventario del comportamento preesistente | 27 test mirati riportati; nessuna nuova funzionalità |
| 1 | Schema di calibrazione, archivio revisionato, identità video, geometria e trasformazioni | 64 test mirati riportati |
| 2 | Editor condiviso sul primo frame, drag, ROI, misure, salvataggio e riuso | 182 test completi e 5 gruppi JS riportati; browser reale non attestato |
| 3 | Proposte automatiche geometriche, alternative, diagnostica e preset espliciti | 29 nuovi casi; prima verifica completa limitata da pandas/Code Integrity |
| Audit 1–4 | Correzione di quattro difetti riprodotti e aggiornamento di una fixture obsoleta | 224 test completi, 13 casi di regressione aggiunti |
| 4 | Uso della calibrazione nella pipeline, resize, invalidazione, snapshot e replay | 231 test completi; prova controllata di traiettoria metrica |
| 5 | Preparazione persistente delle calibrazioni di un insieme di video, worker e revisione | 248 test completi, 17 nuovi casi |
| 6 | Batch metrico con snapshot per video, controlli di integrità e stop/ripresa | 271 test completi, 23 nuovi casi |
| 7 | Compatibilità storica, export separato, campione e comparatore fra modalità | 285 test completi, 14 nuovi casi; validazione reale parziale |
| Modifica successiva | Nuovo flusso automatico immediato, preservazione delle correzioni | **288 test completi**, 3 nuovi casi rispetto a fase 7 |

I numeri sono risultati di momenti diversi, non suite indipendenti da sommare. Il dato attuale di riferimento è 288. Non usare il conteggio della fase 3 o della fase 7 come stato finale.

Documenti di dettaglio: [fase 0](docs/fase-0-baseline.md), [fase 1](docs/fase-1-calibrazione.md), [fase 2](docs/fase-2-editor.md), [fase 3](docs/fase-3-proposte.md), [audit](docs/audit-fasi-1-4.md), [fase 4](docs/fase-4-pipeline.md), [fase 5](docs/fase-5-preparazione.md), [fase 6](docs/fase-6-batch-metrico.md), [fase 7](docs/fase-7-validazione.md).

## 6. Backend di calibrazione: contratti importanti

### 6.1 Record e provenienza — `calibration/records.py`

`CalibrationRecord` è un modello Pydantic congelato con schema versione 1. Non è un generico YAML di omografia riutilizzabile indiscriminatamente su altri video. Conserva:

- ID del record, revisione, timestamp, camera e stato `draft`, `confirmed` o `invalid`.
- Identità del video: percorso, nome originale, dimensione del file, SHA-256, dimensioni native, rotazione e hash dei pixel del primo frame.
- Riferimento: indice, timestamp, PTS/time base, hash dei pixel, percorso e hash del PNG.
- Vertici con ID stabili P1–P4 e modalità geometrica.
- Rettangolo: `width` per P1→P2 e `length` per P2→P3, espresse in metri.
- Origine di ogni misura: `unknown`, `measured`, `standard`, `experimental`; fonte, eventuale preset e `user_confirmed`.
- In alternativa, corrispondenze esplicite e relativa dichiarazione di scala.
- ROI indipendente dal quadrilatero di calibrazione.
- Metodo automatico/manuale, versione dell'algoritmo, parametri, seed, punti iniziali e diagnostica.
- `geometric_quality` e, solo quando confermato, calibrazione runtime.

La conferma controlla geometria e provenienza, ma **non alza artificialmente la qualità**. Le modifiche invalidano l'accettazione/runtime. Cambiamenti di geometria o ROI azzerano la qualità salvo valore esplicitamente ricalcolato; l'editor si occupa del ricalcolo. Non aggirare questa regola nelle fixture.

### 6.2 Archivio — `calibration/repository.py`

- Estrae con PyAV il primo frame decodificabile nelle coordinate native.
- Cerca record compatibili usando l'identità del video, non soltanto il nome.
- La ricerca collettiva evita di scandire inutilmente l'archivio per ogni video; ambiguità e incompatibilità hanno motivi espliciti.
- Verifica hash, dimensioni e pixel **degli stessi byte del riferimento che decodifica**.
- Mantiene i percorsi relativi all'interno della posizione consentita.
- Salva con lock di sistema, controllo ottimistico della revisione e pubblicazione atomica: PNG di riferimento, YAML di revisione immutabile, YAML corrente.
- `only_if_missing=True` protegge dalla gara con un salvataggio manuale concorrente.
- Supporta associazione/import esplicito del formato legacy come bozza, senza conferme implicite.

Convenzione: `nomevideo.yaml`, `nomevideo.revision-000N.yaml`, `nomevideo.reference-r000N.png`, con gestione delle collisioni dei nomi. Conservare revisioni e riferimenti; non sovrascriverli a mano per risolvere un errore della GUI.

### 6.3 Coordinate — `calibration/coordinates.py`

La validazione rifiuta quadrilateri degeneri, incrociati, non finiti o fuori dall'immagine. ROI e punti di calibrazione hanno responsabilità separate. `ImageTransform` rappresenta esplicitamente resize/letterbox e inversione delle coordinate.

Se `T` porta le coordinate native in quelle dell'immagine elaborata, l'omografia operativa è **`H_operativa = H_nativa · T⁻¹`**. Usare le dimensioni effettive dopo arrotondamento, non solo il fattore nominale di resize. Non confondere coordinate CSS del canvas, pixel nativi, pixel elaborati e metri.

### 6.4 Stato editor e componente

File principali: `calibration/editor.py`, `calibration_ui.py`, `components/calibration_editor/__init__.py`.

- Sessione legata alla sorgente tramite percorso, size, mtime e root del progetto.
- Epoch e payload con identità/revisione/event ID impediscono che eventi vecchi del componente finiscano su un altro video o una nuova revisione.
- Quattro punti in pixel nativi, ID/lati stabili, trascinamento con commit al rilascio.
- Undo limitato a 50 stati; annullamento del drag e limiti verificati.
- ROI modificabile separatamente nel flusso manuale; lock del record confermato e riuso esplicito.
- Anteprima metrica limitata: dimensione predefinita 640, massimo 1024, fino a 40 linee di griglia per asse.
- Componente Streamlit CCv2 con HTML/CSS/JS inline, asset e packaging verificati nelle fasi precedenti.

Nei test AppTest è stato necessario isolare il registro dei componenti fra import; non trasferire automaticamente questa gestione di test al server reale, dove il registro è unico.

## 7. Proposte automatiche e preset

### 7.1 Proposte — `calibration/proposals.py`

Algoritmo `painted-contours-v1`, sul primo frame: maschere di bianco/giallo, contorni quadrilateri, segmenti LSD, sostegno dei bordi e famiglie robuste di punti di fuga. Scarta forme degeneri/tagliate e conserva diagnostica e alternative. Non usa etichette degli incidenti e non carica YOLO.

Parametri predefiniti registrati: `max_dimension=1280`, `max_candidates=5`, `max_segments=160`, `seed=42`, `min_edge_support=0.7`.

La qualità del candidato valuta il sostegno geometrico/visivo della proposta, con un'euristica distinta da quella dell'editor. **Non equivale a scala nota, affidabilità fisica o superamento del gate runtime.** Il generatore reale attuale produce normalmente distanze sconosciute. Il supporto UI a distanze fornite da un candidato non significa che l'algoritmo le deduca dal frame.

`candidate_changes` conserva punti iniziali/diagnostica, rimuove le misure del candidato precedente e revoca accettazione e conferme. Con l'ultima modifica imposta esplicitamente `user_confirmed=False` per entrambe le dimensioni anche quando il candidato ne contiene un valore.

### 7.2 Preset — `calibration/presets.py`

Preset opzionali basati sui riferimenti FHWA MUTCD documentati nel progetto, edizione 11 / revisione 1 dicembre 2025. La loro applicabilità richiede scelta esplicita del contesto USA, tipo di segnaletica, tratto completo e orientamento dei lati. Sono conservati fonte, edizione e identificazione del preset; controlli di sovrapposizione proteggono l'applicazione a un quadrilatero modificato.

Valori documentati durante lo sviluppo:

| Riferimento | Dimensione disponibile |
| --- | --- |
| Tratto ordinario di linea discontinua | Lunghezza 10 ft = 3,048 m; larghezza ammessa 0,1016–0,1524 m |
| Tratto puntinato di corsia | Lunghezza 3 ft = 0,9144 m |
| Prolungamento nell'intersezione | Lunghezza 2 ft = 0,6096 m |
| Barra di attraversamento | Larghezza 0,3048–0,6096 m; lunghezza sconosciuta |

Un intervallo non determina una misura esatta: occorre dichiarare l'assunzione per il valore scelto. Nessuna inferenza automatica del paese dal video. Un preset non è una misura indipendente. Le fonti e condizioni complete sono nel codice e in `docs/piano-calibrazione-omografia.md`; se una nuova richiesta riguarda l'attualità della normativa, verificarla nuovamente sulle fonti ufficiali.

## 8. Ultima implementazione: editor automatico immediato

### 8.1 Comportamento ottenuto

Premendo **Calibrazione automatica**:

1. La sessione passa al workflow automatico e genera le proposte.
2. La prima proposta è applicata direttamente e i punti appaiono nello stesso canvas, già trascinabili.
3. Non serve più un secondo pulsante «Usa proposta nell'editor».
4. Cambiare **Proposta da esaminare** applica direttamente l'alternativa selezionata.
5. I controlli di inserimento manuale, le coordinate numeriche/avanzate, i controlli della ROI e gli azzeramenti manuali sono nascosti in questo workflow. Restano i controlli condivisi di revisione, misure, conferma/salvataggio e undo pertinente.
6. Le distanze note sono precompilate e modificabili. Quelle sconosciute restano vuote. I preset restano una scelta esplicita dell'utente.
7. **Torna alla selezione manuale** ripristina i controlli conservando il lavoro.

Le correzioni dell'utente non vengono riapplicate/sovrascritte a ogni rerun. Cambiare candidato elimina la scala del riferimento precedente e revoca le conferme. Se la generazione fallisce o non trova candidati, non si presenta un canvas manuale vuoto come se fosse la proposta automatica; il ritorno al manuale conserva i punti precedenti.

Una bozza automatica salvata e riaperta torna nel workflow automatico con le correzioni e l'accesso ai preset, anche senza rigenerare tutte le proposte. Le sessioni Streamlit già vive, create prima dell'aggiunta dei nuovi campi, vengono adattate senza perdere punti e misure.

### 8.2 Dettagli per continuare il codice

`EditorSession` ha aggiunto:

```python
workflow: str = "manual"
applied_proposal: tuple[int, str] | None = None
proposal_error: str | None = None
```

`calibration_proposal_ui.py` gestisce il cambio workflow, risultato/errore, generazione, dropdown e preset. La coppia `(proposal_generation, candidate_id)` identifica la proposta già applicata. Prima di applicarla controlla `reference_pixel_sha256`. La selezione iniziale può essere recuperata dalla diagnostica salvata.

`calibration_ui.py` inizializza il workflow dal metodo `auto_assisted`, preserva il marcatore della proposta preparata, migra gli oggetti live vecchi e rende condizionali i controlli manuali. Il canvas automatico usa la modalità di calibrazione; gli errori/risultati vuoti producono una selezione non pronta all'analisi.

Test in `tests/integration/test_calibration_ui.py`: applicazione immediata e distanze note, persistenza delle modifiche, riapertura, eliminazione della vecchia scala al cambio candidato, fallimento automatico con ritorno al manuale, migrazione della sessione viva precedente. Adeguati anche il test del drag in modalità automatica e l'integrazione della preparazione.

Documentazione aggiornata durante questo intervento: `README.md`, `start_tutorial.md`, `docs/fase-3-proposte.md`, `LLM_recap.md`.

## 9. Problema aperto: qualità geometrica e soglia 0,55

### 9.1 Segnalazione reale

L'utente ha selezionato quattro punti, inserito P1→P2 e P2→P3 in metri, dichiarato origine sperimentale, scritto la fonte e spuntato la conferma. L'app ha mostrato:

> Qualità geometrica inferiore alla soglia richiesta (0.55): controlla i punti e la soglia nella configurazione.

La conferma della provenienza e il controllo geometrico sono condizioni diverse. Il salvataggio come `confirmed` non garantisce che la calibrazione soddisfi `runtime.metric_valid(min_confidence)`.

### 9.2 Formula effettiva

Implementazione riletta in `calibration/editor.py::geometric_quality`; soglia predefinita riletta in `config.py::CalibrationSettings.min_confidence`:

```text
punti_normalizzati = punti_px / [larghezza_frame, altezza_frame]
A = area dell'inviluppo convesso dei punti normalizzati
H = omografia dai punti normalizzati al quadrato unitario
K = np.linalg.cond(H)

q = 0.79 × min(1, A / 0.05) × min(1, 50 / max(K, 1))
soglia predefinita = 0.55
```

Per corrispondenze esplicite, si normalizzano anche le destinazioni sulla loro estensione e si stima H con `findHomography`; i fattori finali restano gli stessi.

Conseguenze:

- Il punteggio massimo di questa euristica è 0,79.
- Un quadrilatero che occupa meno del 5% del frame è penalizzato dal termine di area.
- Un condizionamento superiore a 50 applica un'ulteriore penalità.
- Nel rettangolo, inserire metri, fonte e origine non aumenta il punteggio.
- I numeri 0,79, 0,05, 50 e il gate 0,55 non risultano calibrati su una validazione empirica dell'errore fisico.
- Il punteggio non è una probabilità di correttezza, una percentuale di precisione o un errore in metri.

`records._build_runtime` trasferisce `record.geometric_quality` nella confidence runtime. Il gate richiede validità, accettazione, unità `m` e confidence almeno pari alla soglia. La UI può confermare/salvare, mentre la selezione pronta all'analisi resta falsa.

### 9.3 Record corrente della prova, riletto per questo recap

File: `data/calibration/videos/1O-y5PmPWdU_00.yaml`.

| Campo | Valore corrente osservato |
| --- | --- |
| Record ID | `db20641151a344c0a9f81166d6fa34aa` |
| Revisione | **3** |
| Stato / metodo | `confirmed` / `manual` |
| Frame nativo | 1280 × 720 |
| Larghezza P1→P2 | 0,6 m, `experimental`, fonte `gemini`, confermata |
| Lunghezza P2→P3 | 3,0 m, `experimental`, fonte `gemini`, confermata |
| Qualità / confidence runtime | `0.012658754616345457` |
| Runtime | `accepted=true`, `valid=true`, unità `m`; gate 0,55 non superato |
| ROI | Quadrilatero non coincidente con tutto il frame |

Punti salvati:

```text
P1 = (133.46669921875, 378.70001831054685)
P2 = (143.86669921875, 362.1666687011719)
P3 = (194.26669921875, 360.3000183105469)
P4 = (183.6, 378.4333435058594)
```

ROI della revisione 3:

```text
(261.86669921875, 123.9000244140625)
(926.4, 142.0333251953125)
(1197.3333984375, 472.7000244140625)
(98.66669921875, 620.9667236328125)
```

Nell'analisi precedente, con gli stessi quattro punti, l'area era circa **0,09332% del frame**, `K ≈ 58,23968`, fattore area `0,01866435`, fattore condizionamento `0,85852119`. Questo spiega il punteggio molto basso. Quell'analisi citava revisione 2: l'archivio attuale è arrivato a revisione 3, con la ROI sopra e la stessa qualità.

L'errore di riproiezione memorizzato è molto piccolo, circa `4,20e-7`, ma con quattro corrispondenze usate per stimare H non certifica l'accuratezza fisica altrove sul piano. La fonte testuale `gemini` non costituisce una misura indipendente; non reinterpretarla come dato misurato.

### 9.4 Secondo record corrente

`data/calibration/videos/9xwZ_urlz-k_00.yaml`: revisione 4, `draft`, metodo `auto_assisted`, 1280 × 720; P1=(849,267), P2=(861,275), P3=(859,305), P4=(846,296), ROI nulla, entrambe le dimensioni sconosciute/non confermate.

Nell'analisi precedente il punteggio della proposta era circa 0,76276, mentre il ricalcolo dell'editor dava circa 0,00210835, area 0,04232% e K≈158,56485. È un esempio concreto della differenza fra qualità della proposta e qualità usata dal gate. Non presentare queste due quantità come una sola metrica coerente.

### 9.5 Stato della correzione

È stata riconosciuta una limitazione dell'implementazione: l'euristica penalizza fortemente piccoli riferimenti dipinti che possono comunque essere utili. **Non è stata riprogettata.** L'ultima richiesta funzionale riguardava la UI automatica, non la revisione del criterio scientifico.

Possibile lavoro futuro, da concordare con la prossima richiesta: separare errori geometrici effettivi da avvertenze di incertezza, chiarire il significato del punteggio e rendere coerenti proposta/editor/runtime. Abbassare semplicemente la soglia non dimostra che la calibrazione sia accurata. Non modificare punti, misure, soglia o record dell'utente solo per ottenere un esito verde.

## 10. Audit delle fasi 1–4 e correzioni effettive

L'audit richiesto dall'utente ha riprodotto quattro difetti prima di correggerli:

| Difetto | Correzione |
| --- | --- |
| Il riferimento PNG poteva essere riletto dopo la verifica usando byte diversi | Verifica e decodifica degli stessi byte, riuso della lettura verificata in archivio/pipeline |
| In modalità immagine, dopo invalidazione il run poteva risultare ancora valutabile | Valutabilità subordinata anche alla validità della calibrazione, con motivi conservati |
| Export/replay perdevano lo stato non valutabile | Propagazione di stato, motivi e invalidazioni; avvertenza `NON VALUTABILE` sul video |
| Replay poteva usare un video sostituito ma di uguali dimensioni | Verifica SHA-256 prima del rendering e prima della pubblicazione per i nuovi run con identità |

I nuovi casi davano inizialmente 4 fallimenti e 9 successi; rapporto `outputs/audit-fasi-1-4/regressions-before.xml`. Dopo le correzioni: 224 test completi superati.

È stato inoltre corretto `test_real_yolo_video_serializes_trajectories`: avviava un test metrico con unità canoniche. Poiché la fixture non ha misure reali, ora prova YOLO e serializzazione in **modalità immagine**, con unità pixel e senza coordinate metriche. Non è stato allentato il contratto metrico per far passare il test.

Nella verifica successiva della fase 4, una fixture video controllata è stata resa lossless con FFV1 per non confondere artefatti di compressione con l'errore geometrico. Non sono state allargate arbitrariamente le tolleranze.

I replay legacy senza hash della sorgente conservano controlli dimensionali/numero di frame: non si può inventare retroattivamente un'identità che il vecchio run non ha salvato.

## 11. Pipeline, snapshot e invalidazione — fase 4

`calibration/runtime.py::load_run_calibration` verifica record o calibrazione legacy, hash dei byte, accettazione, metri, confidence, camera, record/revisione, identità video e riferimento. I nuovi record legati a un video locale non sono associati arbitrariamente a una sorgente RTSP.

Il preflight metrico precede il caricamento del modello. Conserva uno snapshot portabile del record/riferimento, più calibrazione originale ed effettiva, trasformazioni e provenienza. I byte sorgente vengono verificati nel percorso di caricamento per evitare sostituzioni durante la preparazione.

Durante la pipeline:

- Si distinguono dimensioni native ed effettive del frame elaborato.
- H viene adattata al resize reale, inclusi arrotondamenti.
- Si adatta coerentemente la soglia della guardia camera.
- Movimento della camera o cambi di risoluzione invalidano la calibrazione, anche se osservati su frame decodificati saltati dall'inferenza.
- Si svuotano le storie rilevanti, si resetta/sospende il detector e non si producono osservazioni metriche valide usando la calibrazione invalidata.
- `features.jsonl`, `run.json`, metriche e replay riportano stato, indice frame e motivi quando previsti dal rispettivo formato.
- `completed` e `evaluable` sono distinti: terminare la decodifica non rende valido un risultato.
- Il callback di progresso trasporta validità e invalidazioni; il batch può fermarsi automaticamente.
- Il segmentatore può essere passato dall'esterno per riusare un solo modello nel worker.

Errori che avvengono prima dell'inizializzazione della pipeline possono non produrre un riepilogo di run; la GUI li tratta come eccezioni normali. Non promettere che qualsiasi errore lasci sempre un run completo.

Il replay dei nuovi run verifica l'identità della sorgente prima/dopo il rendering. Le copie portabili consentono l'uso senza archivio centrale. Non perdere l'avvertenza di non valutabilità nell'esportazione.

## 12. Preparazione delle calibrazioni — fase 5

File: `calibration/preparation.py` e `calibration_preparation_ui.py`.

La preparazione è separata dall'analisi degli incidenti: usa i video e il generatore di proposte, senza YOLO né etichette. Crea sessioni in `outputs/calibration-preparations/preparation-<uuid>` con identità degli input, parametri, versione del codice e checkpoint.

- Worker persistente in subprocess, lock di lancio/esecuzione, controllo PID più tempo di creazione, stop/ripresa.
- Salvataggio atomico per video di risultato, proposta, diagnostica, maschera/anteprima e bozza non confermata nell'archivio.
- Preservazione di bozze e confermati già esistenti; incompatibilità esplicite.
- Salvataggio solo se assente per evitare di sovrascrivere una revisione manuale concorrente.
- Ripresa subordinata a input, hash, parametri, codice, risultati già completati e revisioni coerenti.
- Recupero della finestra di crash fra salvataggio della bozza e commit del risultato di preparazione.
- Errori per video registrati senza trasformarli in proposte valide. Se l'errore è già stato registrato come risultato, correggere la causa e creare una nuova sessione per riprovarlo.
- Tabella/lista di video, editor condiviso, comando per il prossimo video da revisionare e cursore persistente in `review.json`.
- Editor non attivo mentre il worker modifica la stessa preparazione.

La proposta preparata è legata alla sessione, al riferimento e all'identità del record; non applicare risultati vecchi a un video diverso. Attenzione alle dipendenze: il modulo preparazione importa utility batch; evitare un import top-level inverso che crei un ciclo.

## 13. Batch metrico — fase 6

File principali: `batch.py`, `batch_ui.py`, `calibration/batch_snapshot.py`.

`prepare_job` richiede, in modalità metrica, una mappa esplicita delle calibrazioni con chiavi corrispondenti ai percorsi relativi dei video. Le verifica tutte prima di pubblicare il job pronto; una mancante/non valida blocca la preparazione, senza esclusioni implicite.

Manifest versione 2: modalità, hash delle sorgenti, metadati/hash degli snapshot per video, camera, record/revisione, identità video e provenienza delle dimensioni. I manifest v1 restano leggibili per compatibilità.

Snapshot tipico per ogni video:

```text
calibrations/000000/
  calibration-source.yaml
  calibration-record.yaml
  calibration-original.yaml
  calibration-reference-original.png
```

La sorgente grezza serve all'audit e può contenere vecchi percorsi; il worker usa il record portabile. Dopo la preparazione, modificare l'archivio centrale non deve cambiare l'esperimento.

Ulteriori invarianti:

- Preparazione in staging temporaneo e pubblicazione solo dopo i controlli.
- Nuova directory con identificatore univoco per ogni preparazione; hash del protocollo separato dall'identità del job.
- Configurazioni conservate in `batch_config.json`, `source_config.yaml`, `resolved_config.yaml`.
- Validazione di firma del codice, modello, etichette, hash/stat dei video e integrità di tutte le copie metriche.
- Controlli prima di ogni video e dopo l'inferenza.
- Commit solo di risultati completati, valutabili, con calibrazione valida e dati metrici disponibili quando richiesti.
- `InvalidBatchCalibration`: checkpoint in errore con `requires_new_experiment=true`; ripresa vietata. Una correzione della calibrazione richiede un nuovo esperimento.
- Risultato per video atomico; esportazioni ricostruibili. Un tentativo interrotto non entra nei risultati validi.
- Un solo modello nel worker e un solo worker attivo per root tramite lock.
- Liste separate per modello/modalità e widget del batch metrico con chiavi dedicate.
- CSV con modalità, percorso/hash calibrazione, record/revisione e origini width/length/scala esplicita. Campi assenti nei risultati vecchi restano vuoti/null.

## 14. Significato delle metriche di incidente

Le etichette vengono associate per basename, con rifiuto di nomi duplicati, dati mancanti o timestamp invalidi. Un'etichetta assente non è un negativo.

Tipi positivi riconosciuti nelle verifiche: `single`, `t-bone`, `rear-end`, `sideswipe`, `head-on`. Sono supportati negativi espliciti come `normal`, `negative`, `no-accident`, `no_accident`, con timestamp vuoto e durata valida. Le etichette non vengono usate per ricavare geometria o scegliere le proposte.

### 14.1 Valutazione per video

Un video è predetto positivo se contiene almeno un evento confermato in qualunque istante. Su un insieme di soli positivi, TN e FP per video sono zero; accuracy e recall coincidono. Se ci sono predizioni positive, la precision è 1; se non ce ne sono il denominatore è nullo e la precision non è definita. Questo non dimostra assenza di falsi allarmi su video negativi.

### 14.2 Valutazione temporale per eventi

Matching uno-a-uno dell'evento temporalmente più vicino al ground truth, con `abs(impact_time - truth_time) <= 1 s`. Gli altri allarmi, anche duplicati nella finestra, sono FP; ground truth senza match sono FN. Un allarme fuori tempo può produrre contemporaneamente video TP ed evento FP+FN.

TN e accuracy per eventi non sono definiti. Denominatori nulli diventano `null` nel JSON e celle vuote nel CSV. Non sostituirli con zero per comodità. La valutazione non certifica classificazione del tipo di incidente né localizzazione spaziale.

## 15. Compatibilità e confronto controllato — fase 7

### 15.1 Consultazione e ripresa

`resume_compatibility` fornisce un controllo economico per la UI. `start_job` esegue la validazione completa **prima** di cambiare checkpoint/stop flag o avviare il subprocess; il worker la ripete.

Gli esperimenti con codice incompatibile restano consultabili, con configurazione e download, ma non avviabili/riprendibili. Anche un modello storico senza pesi locali resta selezionabile per la consultazione; la preparazione nuova richiede pesi disponibili. Non riscrivere `code_signature` nel manifest per bypassare il blocco.

`export_results(job, output_dir=...)` e la CLI consentono di ricostruire gli export in una nuova destinazione senza alterare l'esperimento originale.

### 15.2 Campionamento e revisione

`scripts/evaluate_calibration.py` espone `sample`, `review`, `compare`, `export`. `calibration/evaluation.py` gestisce campioni riproducibili, hash, immagini, proposte e revisioni. Campioni da 1 fino a min(20,N), destinazione nuova, ordinamento dei percorsi e seed fissato.

`reviews.json` parte da `pending`. Decisioni: `accept`, `correct`, `reject`, `no_reference`, `pending`. Accettazione/correzione indicano il candidato; correzione include quattro punti nativi. Sono richiesti revisore e tempo umano valido quando pertinente. Il tempo di generazione è distinto dal tempo umano. Si calcolano medie solo su dati effettivamente registrati; l'errore metrico resta assente senza misure indipendenti.

### 15.3 Comparatore

`batch_comparison.py` richiede risultati v2 completi e valutabili, stessi video, identità del modello, codice, metadati/etichette, tolleranza temporale, impostazioni condivise di video/perception/tracking/features/events, seed e guardia camera. Controlla gli snapshot metrici e le etichette salvate.

Il confronto richiede ROI a frame intero nel metrico, poiché il ramo immagine usa tutto il frame. La calibrazione reale della prova utente alla revisione 3 ha una ROI ristretta: oltre al problema di qualità, non soddisferebbe questo requisito del confronto.

In caso di mismatch vengono restituiti i motivi e `metrics: null`; non si producono numeri apparentemente comparabili. La configurazione condivisa è `configs/paired-evaluation.yaml`: 8 FPS, larghezza massima 1280, inference size 640, YOLO26s su CPU, clip evento pre 3 s/post 5 s. Usarla in entrambe le modalità; i default di altri YAML possono differire.

I test del confronto hanno eseguito pipeline reali su due video temporanei controllati con segmentatore sintetico/vuoto. Sono collaudi del software, non un confronto scientifico YOLO sui dati reali.

## 16. Prove reali effettuate e ciò che non dimostrano

### 16.1 Campione della fase 7

Directory: `outputs/phase7-verification/real-sample`. Cinque video estratti dai 150, seed 42, senza leggere le etichette e senza tarare l'algoritmo sui risultati.

| Video | Proposte | Osservazione dell'assistente sulle anteprime |
| --- | ---: | --- |
| `1ZQAChv0TEI_00.mp4` | 1 | Candidato al bordo sinistro della strada; natura del riferimento da accertare |
| `6T58bsyZz40_00.mp4` | 1 | Riferimento sottile e prospettico; rettangolarità fisica non accertata |
| `CQaoY2QypM8_00.mp4` | 1 | Candidato minuscolo, scena notturna con riflessi/watermark; non validato |
| `D1eP4Bn4hDQ_0_00.mp4` | 5 | Barre di attraversamento riconoscibili, dimensioni fisiche ignote |
| `FnahNGsmeSQ_00.mp4` | 0 | Nessuna proposta |

Copertura grezza **4/5 = 80%**, zero errori di elaborazione. Significa soltanto presenza di almeno una proposta. Non è precisione, tasso di calibrazioni valide né accuratezza metrica; cinque video non giustificano generalizzazioni al dataset.

Artefatti: `sample.json`, `report.json`, `reviews.json`, cartelle `000`–`004`, PNG di riferimento/maschera/anteprima, `assistant-inspection.json`. La revisione visiva delle immagini da parte dell'assistente non sostituisce una revisione utente cronometrata. Stato documentato: **0 revisioni umane**, tempi/correzioni non misurati.

### 16.2 Smoke test storico con YOLO reale

Directory `outputs/batch-smoke/`, script `verify.py`, job `jobs/yolo26s-seg-2bf4b4ab84a02068`. Due clip di circa 3 secondi tratte da `6oymx7wBmjw_02.mp4` e `987C4_UdnJE_00.mp4`. Stop durante il primo video a circa 12–15%, nessun risultato incompleto committato, poi ripresa e due risultati unici.

Risultati storici: video TP=1, FP=0, FN=1, TN=0, accuracy=0,5, precision=1, recall=0,5, F1≈0,6667. Eventi TP=1, FP=0, FN=1, TN/accuracy non definiti. È uno smoke test di pipeline e ripresa, non un benchmark sui 150 video.

### 16.3 Export storico

`outputs/phase7-verification/legacy-export` contiene la riesportazione del batch storico, 2/150 completati. Metriche parziali: video TP=0, FP=0, FN=2, TN=0; eventi TP=0, FP=0, FN=2, TN non definito. Non è nuova inferenza e non è una valutazione completa.

## 17. Batch storico da preservare

Directory: **`outputs/batches/yolo26s-seg-55517fe0827b34e6`**.

La baseline `outputs/audit-fasi-1-4/batch-hashes-before.json` comprende 33 file. Nel controllo finale della fase 7, tutti i 33 coincidevano, senza aggiunte/rimozioni/modifiche. Questo confronto non è stato rieseguito per scrivere il recap.

La cronologia più vecchia in `LLM_recap.md` descrive un'eccezione precedente alle fasi: durante una modifica iniziale alla configurazione degli output era stata annotata una migrazione di compatibilità del manifest, con copia `manifest.before-output-config-20261006.json` e `compatibility_updates`. È un fatto storico, **non un'autorizzazione a ripeterlo**. Le fasi successive hanno preservato il batch e non hanno indebolito la firma del codice.

Se il codice attuale non coincide con quello dell'esperimento, consultare/esportare oppure recuperare l'ambiente originale in modo isolato. Non forzare la ripresa alterando gli hash o adattando i risultati.

## 18. Verifiche automatiche, ambiente e blocchi strumenti

### 18.1 Ultime evidenze

`outputs/automatic-editor-verification/validation-summary.json`, riletto per questo recap:

```json
{
  "date": "2026-10-08",
  "environment": ".venv",
  "pytest_passed": 288,
  "pytest_failures": 0,
  "pytest_errors": 0,
  "pytest_skipped": 0,
  "pytest_duration_s": 75.18,
  "new_test_cases": 3,
  "ruff": "passed",
  "browser_visual_validation": "not performed",
  "immediate_automatic_proposal": true,
  "manual_controls_hidden_in_automatic_workflow": true,
  "generic_assumed_distances": false,
  "available_or_selected_preset_distances_only": true,
  "preserves_corrections_and_existing_live_sessions": true
}
```

XML completo: `outputs/automatic-editor-verification/pytest-full.xml`. Una verifica mirata precedente riportava 69 test in 36,23 s in `pytest-targeted.xml`; è antecedente all'ultima regressione aggiunta per la sessione live. Un risultato intermedio di 287 test non è quello finale.

Altri rapporti sono nelle directory `outputs/audit-fasi-1-4/` e `outputs/phaseN-verification/` per le fasi che le prevedono. Le guide di fase contengono i comandi e i percorsi esatti. Controlli di formato fino a 133 file e cinque gruppi JavaScript erano superati alla fase 7.

### 18.2 Windows Code Integrity

Nella prima validazione della fase 3 alcune DLL di pandas erano state bloccate: il risultato storico era 200 test passati, 7 fallimenti e 4 errori, oltre alla verifica mirata delle nuove funzioni. Il blocco **non si è riprodotto** nell'audit e nelle verifiche successive, inclusi YOLO e GUI, usando la stessa `.venv`.

Non sono state disattivate protezioni Windows, sostituite dipendenze per aggirare il blocco o ricreato l'ambiente. Non riportare oggi quel vecchio fallimento come se impedisse tuttora tutti i test. Se ricompare, raccogliere l'errore attuale prima di attribuirgli la stessa causa.

### 18.3 Browser e JavaScript

Tentativi di controllo browser precedenti: pipe nativa Windows non disponibile (`os error 2`); inventario alternativo vuoto per app/browser; IAB non disponibile. Non è stato possibile completare la prova visiva automatizzata.

Node non era nel PATH. I cinque gruppi in `tests/frontend/calibration_editor.mjs` sono stati eseguiti nel runtime V8 disponibile agli strumenti, usando il JavaScript inline effettivo del componente. Per usare Node in futuro, leggere il runner documentato nella cartella; non installarlo implicitamente solo perché manca.

### 18.4 Sandbox della sessione di recap

Il normale `exec_command` aveva restituito `helper_unknown_error: setup refresh had errors`. Le letture sono riuscite con escalation esplicita. Questo è un problema dello strumento/sandbox, distinto da pandas e Code Integrity. Non generalizzare a una nuova sessione: provare normalmente e richiedere l'escalation prevista solo se necessaria. `apply_patch` è stato disponibile per scrivere documentazione.

## 19. Stato Git e mappa dei file

Il working tree riletto per il recap conteneva modifiche precedenti, qui elencate prima dell'aggiunta del presente documento. Molti file fondamentali sono ancora **untracked**: perderli significa perdere le implementazioni. Non usare `git clean`, `reset --hard` o ripristini indiscriminati.

### 19.1 File tracked modificati

```text
.gitignore
LLM_recap.md
README.md
apps/streamlit_app.py
docs/batch-analysis.md
docs/real-videos.md
pyproject.toml
src/cctv_incident/batch.py
src/cctv_incident/batch_ui.py
src/cctv_incident/calibration/assisted.py
src/cctv_incident/calibration/homography.py
src/cctv_incident/calibration/lane_mask.py
src/cctv_incident/calibration/line_fitting.py
src/cctv_incident/config.py
src/cctv_incident/pipeline.py
src/cctv_incident/replay.py
src/cctv_incident/video.py
start_tutorial.md
tests/integration/test_batch.py
tests/integration/test_batch_ui.py
tests/integration/test_real_model.py
tests/integration/test_ui.py
tests/integration/test_video_selection.py
uv.lock
```

### 19.2 File e cartelle nuovi principali

```text
PLANNING_GPT_6_1_SOL.md
configs/paired-evaluation.yaml
docs/audit-fasi-1-4.md
docs/fase-0-baseline.md
docs/fase-1-calibrazione.md
docs/fase-2-editor.md
docs/fase-3-proposte.md
docs/fase-4-pipeline.md
docs/fase-5-preparazione.md
docs/fase-6-batch-metrico.md
docs/fase-7-validazione.md
docs/piano-calibrazione-omografia.md
scripts/evaluate_calibration.py
src/cctv_incident/batch_comparison.py
src/cctv_incident/calibration/batch_snapshot.py
src/cctv_incident/calibration/coordinates.py
src/cctv_incident/calibration/editor.py
src/cctv_incident/calibration/evaluation.py
src/cctv_incident/calibration/preparation.py
src/cctv_incident/calibration/presets.py
src/cctv_incident/calibration/proposals.py
src/cctv_incident/calibration/records.py
src/cctv_incident/calibration/repository.py
src/cctv_incident/calibration/runtime.py
src/cctv_incident/calibration_preparation_ui.py
src/cctv_incident/calibration_proposal_ui.py
src/cctv_incident/calibration_ui.py
src/cctv_incident/components/
tests/frontend/
tests/integration/test_calibration_preparation.py
tests/integration/test_calibration_ui.py
tests/integration/test_metric_batch.py
tests/integration/test_metric_calibration_pipeline.py
tests/integration/test_phase7_validation.py
tests/unit/test_calibration_coordinates.py
tests/unit/test_calibration_editor.py
tests/unit/test_calibration_proposals.py
tests/unit/test_calibration_records.py
```

`.gitignore` esclude ambiente, output, video/pesi e altri dati secondo le regole presenti. Non presumere che un artefatto locale venga trasferito clonando il repository. Non annullare modifiche preesistenti a `.gitignore` o alle directory dati. I tab IDE con appunti DASS non fanno parte di questo lavoro e non sono stati modificati.

### 19.3 Dove cercare in base al problema

| Problema | File iniziali |
| --- | --- |
| Modalità GUI, caricamento video, avvio pipeline | `apps/streamlit_app.py`, `config.py` |
| Punti, rerun, salvataggio, ROI, editor automatico | `calibration_ui.py`, `calibration_proposal_ui.py`, `calibration/editor.py` |
| Drag e coordinate del canvas | `components/calibration_editor/__init__.py`, `tests/frontend/` |
| Qualità e soglia | `calibration/editor.py`, `calibration/records.py`, `calibration/homography.py`, `config.py` |
| Record, revisioni, identità/riferimento | `calibration/records.py`, `calibration/repository.py` |
| Rilevamento segnaletica e dimensioni da preset | `calibration/proposals.py`, `calibration/presets.py` |
| Resize e metrica nel run | `calibration/coordinates.py`, `calibration/runtime.py`, `pipeline.py`, `video.py` |
| Preparazione collettiva e ripresa | `calibration/preparation.py`, `calibration_preparation_ui.py` |
| Snapshot batch, checkpoint e metriche | `calibration/batch_snapshot.py`, `batch.py`, `batch_ui.py` |
| Confronto e revisione campione | `batch_comparison.py`, `calibration/evaluation.py`, `scripts/evaluate_calibration.py` |
| Esportazione video e validità | `replay.py` |

I percorsi della tabella, salvo `apps`, `tests` e `scripts`, sono relativi a `src/cctv_incident/`.

## 20. Comandi utili per una sessione successiva

Sono riferimenti operativi, **non comandi eseguiti durante la scrittura del recap**. Non avviare automaticamente server, test lunghi o job in base a questa sezione.

### Avvio

```powershell
cd "C:\Users\leona\dip_crash_detection_workspace\traffic_incident_recognition-segmentation"
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Non serve attivare la `.venv`. Usare il server già attivo se presente. Fermare prima i worker con **Stop proposte** / **Stop nella GUI**, attendere la pausa, poi arrestare Streamlit con Ctrl+C: chiudere il browser o il processo web non equivale a fermare un worker batch separato.

### Verifiche dopo una futura modifica di codice

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check src apps scripts tests
git diff --check
```

Scegliere prima i test pertinenti alla modifica. Quando serve un nuovo XML, usare una directory dedicata senza sovrascrivere rapporti storici. Una modifica soltanto documentale non richiede l'intera suite.

### Valutazione e recupero: esempi documentati

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py sample dataset/standard_dataset outputs/nuovo-campione --count 5 --seed 42
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py review outputs/phase7-verification/real-sample
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py export outputs/batches/yolo26s-seg-55517fe0827b34e6 outputs/legacy-export-nuovo
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py compare outputs/batches/JOB_IMMAGINE outputs/batches/JOB_METRICO outputs/confronto-nuovo.json
```

Le destinazioni di campionamento/export devono essere nuove. Sostituire i nomi esemplificativi dei job. `review` elabora revisioni già compilate; non inventa misure o tempi. Il campionamento non conferma record nell'archivio e non carica YOLO.

## 21. Lavoro ancora da fare e priorità ragionevoli

Questa lista descrive attività pendenti, non un ordine automatico di esecuzione. Per riprendere seguire la nuova richiesta dell'utente.

### 21.1 Collaudo manuale dell'ultima UI

- [ ] In Segmentazione + omografia, generare proposte e verificare che la prima compaia immediatamente.
- [ ] Verificare che coordinate/istruzioni/ROI manuali siano nascoste nel workflow automatico.
- [ ] Trascinare P1–P4, ridimensionare finestra/sidebar e verificare allineamento immagine/punti.
- [ ] Modificare distanze/provenienza, provocare un rerun e verificare che le correzioni restino.
- [ ] Cambiare candidato: punti aggiornati, scala precedente rimossa, conferme revocate.
- [ ] Controllare distanze sconosciute vuote e preset espliciti compatibili.
- [ ] Salvare/riaprire una bozza automatica, tornare al manuale e verificare la conservazione del lavoro.
- [ ] Provare un video senza proposte e un errore di generazione, poi tornare al manuale.
- [ ] Cambiare video: nessuna misura/punto della sorgente precedente deve essere trasferita implicitamente.
- [ ] Provare il flusso anche dalla preparazione batch, dove l'editor è condiviso.

Registrare data, browser/versione, passi ed esito. Per i controlli ROI/azzeramento passare al manuale, poiché ora sono intenzionalmente nascosti nel workflow automatico.

### 21.2 Qualità geometrica

Il caso reale della sezione 9 è il principale problema funzionale ancora noto. Se l'utente chiede di risolverlo, partire dai dati esistenti e rendere esplicita la distinzione fra geometria invalida, instabilità/propagazione dell'errore e qualità della scala. Verificare la coerenza fra punteggi del generatore, editor, salvataggio e runtime. Non reimplementare l'editor automatico già completato.

### 21.3 Validazione scientifica

- Revisioni umane cronometrate del campione e classificazione di accettazioni/correzioni/rifiuti.
- Calibrazioni utilizzabili e confermate del sottoinsieme scelto per il confronto.
- Due esperimenti reali comparabili, stesso codice/modello/configurazione/sottoinsieme, ROI a frame intero.
- Dichiarazione delle esclusioni e delle assunzioni di scala.
- Misure indipendenti per quantificare l'errore metrico; al momento l'utente non ne dispone.
- Eventuali video negativi per studiare falsi allarmi a livello di video; il sottoinsieme positivo non basta.

Non confondere «test software passati» con «accuratezza fisica dimostrata» o «tutti i video calibrabili automaticamente».

## 22. Come leggere i documenti storici senza ripartire da premesse errate

Ordine consigliato: questo recap, richiesta più recente, eventuali `AGENTS.md` applicabili, stato Git, file della funzionalità coinvolta, rapporto di test pertinente. Per la storia completa leggere anche [LLM_recap.md](LLM_recap.md), [planning operativo](PLANNING_GPT_6_1_SOL.md), [piano tecnico](docs/piano-calibrazione-omografia.md), [README](README.md), [guida di avvio](start_tutorial.md) e [guida batch](docs/batch-analysis.md).

Correzioni temporali da tenere presenti:

- «Fase 4 in corso» nel documento di audit è uno stato storico; fase 4 è stata poi completata.
- «Batch metrico da implementare» nei primi appunti è superato dalle fasi 5–6.
- «285 test» nel planning/fase 7 precede l'ultimo intervento: ora il rapporto finale registra 288.
- «Un solo record reale, in bozza, nessun confermato» nella fase 7 descrive quella data. Ora ci sono i due record descritti sopra, uno confermato alla revisione 3 ma sotto soglia.
- «Revisione 2» nell'analisi iniziale della prova utente non è più il puntatore corrente: ora revisione 3.
- Il problema Code Integrity è stato osservato prima, ma non ha bloccato le ultime suite.
- Il successo di AppTest/JS non completa il collaudo browser.
- La copertura 80% del campione non è una misura di correttezza delle proposte.
- La vecchia migrazione di un manifest non autorizza a bypassare le incompatibilità odierne.
- Le distanze sperimentali della prova utente non sono diventate default dell'app.

Se un file è cambiato dopo il recap, rileggerlo: soprattutto record, checkpoint, configurazioni e processi possono evolvere mentre l'utente usa il server. Conservare l'incertezza invece di attribuire al presente risultati di una fotografia passata.

## 23. Istruzione di ripresa sintetica per l'assistente

> Leggi questo recap e la nuova richiesta dell'utente. Il software delle fasi 0–7 e il nuovo workflow automatico sono già implementati; non ricominciare da zero. Usa la `.venv` esistente e preserva modifiche locali, revisioni e batch storico. L'ultimo rapporto è 288 test passati, ma la prova browser e la validazione fisica restano incomplete. Non inventare distanze: precompila soltanto valori disponibili o un preset scelto. Il gate geometrico 0,55 è ancora quello precedente e blocca la calibrazione reale con q≈0,01266. Verifica lo stato corrente prima di intervenire e distingui gli esperimenti proposti da quelli effettivamente eseguiti. Il solo compito svolto per creare questo file è stato documentare lo stato.

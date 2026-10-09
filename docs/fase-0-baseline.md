# Fase 0 — Ricognizione e baseline

Data: 6 ottobre 2026 (Europe/Rome).
Stato: completata. Le fasi di implementazione 1–7 non sono state avviate.

## Repository e modifiche preesistenti

- Branch: `main_v3`.
- HEAD verificato: `2f48418a21a17fe8ea32d2fdbad6aaeed35ab9fa`.
- Nessun `AGENTS.md` applicabile trovato nella ricognizione.
- Prima della fase erano presenti 13 file modificati e due documenti non tracciati:
  `PLANNING_GPT_6_1_SOL.md` e `docs/piano-calibrazione-omografia.md`.
- Le differenze esistenti riguardano la rimozione della demo dal menu GUI,
  i nomi delle modalità, le cartelle univoche degli esperimenti e il salvataggio
  delle configurazioni, con aggiornamenti di test e documentazione.
  `.gitignore` contiene inoltre l'esclusione preesistente di `*.zip`.
- `git diff --check`: superato. Gli avvisi LF/CRLF descrivono la conversione
  configurata da Git; non sono errori di whitespace.

Il codice corrente prevale sui recap storici. Non sono stati modificati codice,
test, configurazioni, pesi o artefatti degli esperimenti. Questa fase aggiunge
il presente rapporto e un riferimento in `LLM_recap.md`.

## Ambiente verificato

Importazioni reali di Streamlit, OpenCV, PyAV, scikit-image, Pydantic, psutil,
PyTorch e Ultralytics riuscite con il Python della `.venv` del progetto.

| Componente | Versione |
|---|---|
| Python | 3.11.15 |
| Streamlit | 1.63.0 |
| PyAV | 16.1.0 |
| opencv-python | 4.14.0.94 (`cv2`: 4.14.0) |
| scikit-image | 0.26.0 |
| Pydantic | 2.13.5 |
| PyTorch | 2.14.0+cpu |
| Ultralytics | 8.4.150 |
| pytest | 9.1.1 |
| Ruff | 0.16.7 |

CUDA non disponibile. Non sono state installate o aggiornate dipendenze.
La documentazione locale di Streamlit è stata individuata tramite la skill
`developing-with-streamlit`; sono già stati consultati i riferimenti ai componenti
v2 e allo stato di sessione. Il minimo dichiarato `streamlit>=1.45` va riesaminato
quando si sceglierà l'API dell'editor.

## Esperimento esistente preservato

Esperimento: `outputs/batches/yolo26s-seg-55517fe0827b34e6`.

- Stato: `paused`, 2/150 video completati, due file di risultato.
- Worker non attivo; `stop.request` presente. Nessun processo è stato fermato o avviato
  per questo esperimento.
- Video interrotto: il terzo, `-qmYW4S3Xxo_00.mp4`, progresso salvato circa 82,08%.
- Configurazione salvata: coordinate immagine, YOLO26s-seg, CPU, **15 FPS**,
  immagine di inferenza 640, larghezza massima video 1280 e tolleranza ±1 s.
  I 15 FPS sono una scelta dell'esperimento esistente; non vanno sostituiti con
  gli 8 FPS predefiniti della GUI alla ripresa.
- Manifest schema 1: ogni video contiene `relative_path`, `signature` e `label`;
  nessuna copia della calibrazione per video. Il manifest contiene una precedente
  `compatibility_updates`, che non è stata modificata in questa fase.
- `validate_manifest` eseguita in sola lettura: superata, inclusi i controlli su
  codice, pesi, metadati e firme dei video.

Hash verificati prima e dopo i test:

```text
Firma codice: c34dc722b5ab96c044e662ee25b7def1c4222fcb3d54b34bdb4724d36f0b5939
manifest.json: 83b3f29773abc7cac1f92819eb7bb820c1fe9c9a2847db7ac382fa7ac311b45a
checkpoint.json: b3adab265f24f3db8d275784d065dead5d5dc8df2bfc37e3028c2bc2042f5e27
```

Modifiche future ai file Python del package possono cambiare la firma e impedire
la ripresa di questo esperimento con il nuovo codice. Non aggiornare il suo hash
per aggirare il controllo: conservare gli artefatti e usare il codice originale
per continuarlo, oppure un nuovo esperimento per la nuova implementazione.

## Baseline dei controlli

Comandi eseguiti dalla radice del progetto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/unit/test_geometry.py tests/unit/test_batch_evaluation.py tests/integration/test_batch.py tests/integration/test_batch_ui.py tests/integration/test_ui.py tests/integration/test_tracking_calibration.py
.\.venv\Scripts\python.exe -m ruff check apps src tests
git diff --check
```

- Test mirati: **27 passed in 10.36s**.
- Ruff: **All checks passed**.
- Controllo whitespace: superato.
- I test includono geometria, fallback delle proposte, tolleranza inclusiva ±1 s,
  matching uno-a-uno, stop/ripresa, commit atomici, recupero dei CSV, separazione
  degli esperimenti, subprocess su video sintetico, AppTest e guardia movimento camera.

Il primo tentativo nel sandbox ha prodotto 15 successi, un fallimento e 11 errori
per accesso negato alle directory temporanee di pytest e a
`AppData/Roaming/Ultralytics/settings.json`. Anche l'importazione iniziale di
Ultralytics era stata bloccata sullo stesso file. Gli stessi controlli, rieseguiti
fuori dal sandbox con autorizzazione, sono passati. Sono limiti di accesso
dell'ambiente, non fallimenti funzionali della baseline.

Non è stata eseguita l'intera suite, una valutazione sui 150 video o una verifica
visuale nel browser. AppTest non dimostra il trascinamento JavaScript.

## Punti di integrazione e limiti già presenti

| Area | Stato osservato e intervento futuro |
|---|---|
| `apps/streamlit_app.py` | Quattro modalità; il batch metrico termina con `st.stop()`. Calibrazione da JSON, primo frame su richiesta e proposta da copiare: integrare editor e revisione prima dell'analisi. |
| `calibration/homography.py` | `Calibration` richiede geometria completa; non rappresenta bozze. Caricamento YAML legacy da conservare; aggiungere archivio, identità e trasformazioni delle coordinate. |
| Geometria e ROI | L'ROI predefinita deriva dall'inviluppo dei punti di calibrazione. Separarla nel nuovo flusso; aggiungere controlli sull'ordine, sugli incroci e sulla quasi degenerazione. |
| Qualità e scala | La confidenza manuale deriva dagli inlier e dalla riproiezione; quattro punti possono dare confidenza prossima a 1 senza provare l'accuratezza metrica. Conservare distinta la provenienza di ciascuna distanza. |
| `calibration/assisted.py`, `lane_mask.py`, `line_fitting.py` | Maschere e Hough esistenti; famiglie basate sugli assi immagine e linee estreme. Il fallback 10×30 è canonico, non metrico. Servono candidati supportati dalla scena. |
| `calibration/background.py` | Campionamento sull'intero video disponibile come azione separata; non usarlo implicitamente per la nuova calibrazione sul primo frame. |
| `config.py`, `video_inputs.py`, `video.py` | Modelli con `extra=forbid`, upload persistenti, elenco video e resize PyAV riutilizzabili. L'identità e il riferimento della calibrazione richiedono dimensioni effettive, indice e timestamp. |
| `pipeline.py` | Un solo YAML, controllo `camera_id`, rifiuto di dimensioni diverse. Adattare H, punti, ROI e riferimento al resize senza alterare il record originale. |
| Invalidazione runtime | La guardia camera invalida la calibrazione e sospende le decisioni, ma `completed` descrive la fine della decodifica. Il futuro batch metrico dovrà verificare anche l'idoneità alla valutazione prima del commit. |
| `batch.py` | Estendere `prepare_job`, `run_job`, `validate_manifest` ed `export_results`: oggi è accettata solo la modalità immagine e mancano snapshot delle calibrazioni per video. Riutilizzare il worker e i commit esistenti. |
| `batch_ui.py` | Riutilizzare monitor, stop/ripresa e selezione esperimenti; aggiungere preparazione/revisione delle calibrazioni e separare le chiavi per modalità. |
| `batch_evaluation.py` | Protocollo ±1 s inclusivo già verificato dai test: conservarne la matematica. Alcuni documenti storici riportano ±2 s per altri workflow, da non applicare al batch. |

Non sono emersi fallimenti funzionali nei controlli eseguiti. I limiti sopra sono
stato preesistente del software e punti da risolvere nelle fasi successive,
non regressioni introdotte dalla Fase 0.

Prossima fase: **Fase 1 — Record, archivio e geometria**, su istruzione dell'utente.

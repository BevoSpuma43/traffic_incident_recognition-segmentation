# Audit delle fasi 1–4

Data: 6 ottobre 2026. Richiesta: controllare e correggere le fasi precedenti
e la fase 4 parzialmente implementata prima di riprenderne lo sviluppo.

## Esito

Audit e correzioni conclusi. **224 test pytest superati in 46,97 s**, senza
fallimenti o test saltati, usando esclusivamente la `.venv` esistente.
Superati Ruff, controllo del formato (99 file), `git diff --check` e i cinque
gruppi di test JavaScript dell'editor eseguiti in V8.

Il precedente blocco Code Integrity di pandas non si è riprodotto: anche i
test dell'app principale e quelli con YOLO locale passano. Non sono state
modificate protezioni Windows, dipendenze o ambiente virtuale.

## Difetti riprodotti e corretti

| Problema | Effetto prima della correzione | Correzione e prova |
| --- | --- | --- |
| Rilettura del riferimento senza verificarne i byte usati, fase 4 | Un PNG sostituito durante la verifica del video poteva essere usato dalla guardia camera e copiato nello snapshot pur essendo diverso da quello approvato. | `load_reference` verifica hash, pixel e dimensioni degli stessi byte che decodifica e restituisce. Usato sia dall'archivio sia dal caricatore della pipeline. Test con sostituzione durante il preflight. |
| Esito valutabile in modalità immagine dopo invalidazione, fase 4 | Il detector era sospeso ma il riepilogo poteva contenere `evaluable=true`, con zero eventi interpretabili come negativo valido. | La valutabilità richiede anche `calibration.valid`; i motivi di invalidazione sono conservati in entrambe le modalità. Test parametrizzato immagine/metrico. |
| Export privo dello stato di valutabilità | `replay.json` e video esportato perdevano l'avvertenza presente nel riepilogo/GUI. | Report con valutabilità, motivi e invalidazioni; sovrimpressione `NON VALUTABILE` nel video del run non valutabile. Test sulla propagazione al report e alla lista dei run. |
| Replay su sorgente sostituita con stessa risoluzione | Le vecchie traiettorie potevano essere sovrapposte a un video diverso, pur disponendo dell'identità del video nel nuovo run. | Nei run con record video, verifica SHA-256 della sorgente prima del rendering e prima della pubblicazione. Test di sostituzione con risoluzione e numero di frame invariati. |

I quattro difetti sono stati dimostrati prima delle modifiche: **4 fallimenti
e 9 successi** nel nuovo file di regressione. Il relativo XML è conservato in
`outputs/audit-fasi-1-4/regressions-before.xml`.

La prima suite completa dopo queste correzioni ha inoltre rilevato un
**test preesistente non aggiornato al contratto della fase 4**:
`test_real_yolo_video_serializes_trajectories` avviava la pipeline metrica con
unità canoniche. La fixture non dispone di misure stradali: ora verifica YOLO
e serializzazione nella modalità immagine e controlla unità `px` e assenza
di coordinate metriche. Il divieto delle unità canoniche resta attivo ed è
verificato separatamente prima della creazione del run.

Non sono stati abbassati i requisiti di validità per far passare i test.
Il controllo di camera motion resta attivo quando configurato; la sua
rilevazione effettiva tramite OpenCV è coperta anche dal test già esistente.

## Copertura rispetto al planning

| Fase | Controlli effettuati | Stato e limiti |
| --- | --- | --- |
| 1: record, archivio, geometria | Bozze/conferme, provenance, primo frame PyAV, identità video, revisioni e collisioni, pubblicazione atomica, legacy, validazione geometrica e invarianti dopo resize. | Test superati. Rafforzata la lettura verificata del riferimento per l'uso nella fase 4. |
| 2: editor | AppTest per salvataggio/riuso, modifiche e invalidazione, ROI, numerici/avanzati e integrazione nell'app. Test JavaScript per clic, ID, drag al rilascio, resize/margini, cancel, limiti, cleanup, ROI e blocco del confermato. | Test automatici superati; prova visiva del canvas in un browser reale ancora da eseguire. |
| 3: automatismo e preset | Contorni e supporto dei bordi, prospettiva e punti di fuga all'infinito, outlier, determinismo/limiti, fallback vuoto, alternative, preset USA espliciti, provenienza e associazione al singolo tratto. | Nessun ulteriore difetto riprodotto nell'audit. Le proposte richiedono revisione; i test non provano accuratezza metrica nel mondo reale. |
| 4: codice parziale | Record/revisione e video sbagliati, bozza, scala canonica e confidenza insufficiente rifiutati prima del run; resize 640×360 → 321×181; distanze invarianti; snapshot portabile senza archivio centrale; soglia camera adattata; invalidazione anche su frame saltati dall'inferenza; calibrazione legacy già ridotta; replay. | Correzioni e regressioni superate. La fase resta in corso: questo audit non ne dichiara la chiusura. |

Sono stati aggiunti **13 casi di test** in
`tests/integration/test_metric_calibration_pipeline.py`. Per isolare i contratti
di calibrazione, questi casi usano brevi video sintetici e un segmenter senza
rilevamenti; la suite completa comprende anche i test esistenti di eventi,
tracking, YOLO reale, batch e GUI.

## Evidenze e ripetibilità

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --junitxml=outputs/audit-fasi-1-4/pytest-full.xml
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check src apps scripts tests
git diff --check
```

I test JavaScript sono in `tests/frontend/calibration_editor.mjs`; in questa
sessione sono stati eseguiti nel runtime V8 disponibile agli strumenti, sul JS
inline effettivo del componente. Non equivalgono a una prova nel browser.
Con Node disponibile si può usare il runner documentato nello stesso folder.

La baseline mirata iniziale riportava 87 successi. La prima suite completa
riportava 223 successi e il test YOLO obsoleto sopra descritto; dopo il suo
aggiornamento la suite finale riporta 224 successi. XML e riepilogo sono in
`outputs/audit-fasi-1-4/`.

Confrontati i **33 file** di `outputs/batches/yolo26s-seg-55517fe0827b34e6`
con gli hash raccolti prima dell'avvio della fase 4: nessuna modifica, aggiunta
o rimozione. Nessun batch riavviato, nessun manifest adattato alla nuova firma.

## Ripresa della fase 4

Ripartire dal codice corretto, senza reimplementare record, trasformazioni
o snapshot già presenti. Completare la verifica del flusso utente da selezione
e revisione fino ad analisi e replay, includendo il canvas nel browser e una
prova metrica controllata con misure indipendenti. L'euristica geometrica e
le misure da preset non certificano da sole l'accuratezza fisica.

Il controllo SHA del replay usa l'identità disponibile nei nuovi record;
i vecchi run privi di tale identità mantengono i controlli dimensionali e di
numero di frame. Non viene inventato retroattivamente un hash della sorgente.
Il batch metrico e le fasi 5–7 restano fuori da questo audit.

# Fase 6 — Batch metrico e snapshot

Questa pagina descrive la fase originale. Il nuovo avvio richiede una preparazione
completa e mostra il riepilogo delle calibrazioni prima dell'analisi:
[batch automatico con ipotesi USA](batch-calibrazione-automatica-usa.md).

Implementazione del 7 ottobre 2026, nella `.venv` esistente.
Suite completa: **271 test superati in 89,23 s**, inclusi 23 nuovi casi;
zero errori, fallimenti o test saltati. Ruff, formato di 128 file e controllo
whitespace superati. Windows non ha bloccato la validazione.

## Uso nella GUI

1. Selezionare **standard_dataset analisi in batch - con omografia**.
2. In **Preparazione calibrazioni**, preparare e confermare ogni video della
   cartella. Le bozze, le misure senza conferma e i record incompatibili non
   consentono l'avvio del batch metrico.
3. Nel menu **Batch con omografia**, passare ad **Analisi batch**.
4. Selezionare modello YOLO di segmentazione, cartella, CSV delle etichette e FPS.
5. Premere **Prepara batch**: tutti i video devono superare le verifiche.
   Nessun video privo di calibrazione viene escluso implicitamente.
6. Premere **Avvia batch**. **Stop** interrompe l'analisi; **Riprendi** salta i
   video già pubblicati e ricomincia dall'inizio del video incompleto.

La selezione degli esperimenti è separata per modello e modalità. Il confronto
tra esperimenti riporta anche la modalità. Tornare alla stessa modalità e allo
stesso modello per ritrovare un batch già preparato.

## Contratto e artefatti

`prepare_job(..., calibration_map={"a.mp4": Path(...), ...})` richiede una mappa
esatta dei percorsi relativi restituiti da `list_dataset_videos()` per la
modalità metrica. Il batch immagine conserva la mappa opzionale e non richiede
calibrazioni. Per il batch metrico sono necessari record video confermati:
eventuali YAML legacy vanno prima importati e confermati nell'editor.

Il manifest versione 2 registra `coordinate_mode`, SHA-256 dei video e, per
ciascuna calibrazione, identità del video, camera, revisione, provenienza delle
misure, percorso e hash delle copie locali. Una preparazione fallita non appare
tra gli esperimenti pronti. Le immagini vengono elaborate una alla volta, senza
tenere in memoria i riferimenti di tutta la cartella.

Ogni `calibrations/000000/` contiene:

- `calibration-source.yaml`: byte originali, conservati per audit;
- `calibration-record.yaml`: record portabile usato dal worker;
- `calibration-original.yaml`: geometria nativa portabile;
- `calibration-reference-original.png`: immagine verificata del riferimento.

I percorsi relativi del record portabile e della geometria puntano alla copia
dell'immagine. Il file sorgente conserva invece i byte originali: per eseguire
il batch si usa sempre `calibration-record.yaml`. Le copie sono verificate prima
della pubblicazione, all'avvio/ripresa, prima di ogni video e dopo l'inferenza.
L'archivio centrale può essere aggiornato o rimosso senza invalidare il batch.
Una modifica o rimozione delle copie locali viene rifiutata.

Il worker riutilizza un solo segmentatore per esecuzione. Per ciascun video
imposta camera, record e revisione dalla copia locale; la pipeline conserva
anche i propri snapshot nativi/effettivi e la trasformazione di ridimensionamento.

## Errori, ripresa e risultati

Se la calibrazione diventa invalida durante l'analisi, il worker richiede
l'arresto tramite il callback di avanzamento e rifiuta il commit del video.
Anche un riepilogo completato deve dichiarare esplicitamente la valutabilità
e la validità metrica. Un errore tecnico non diventa un vero negativo.

La perdita di calibrazione o l'alterazione degli snapshot registra
`requires_new_experiment: true`: **Riprendi** è disabilitato. Correggere il record
nell'archivio e creare un nuovo esperimento. I risultati dei video già completati
restano disponibili come metriche parziali. Stop ordinario e arresto del processo
mantengono invece la ripresa prevista dal protocollo.

I JSON dei risultati conservano la provenienza completa. `videos.csv` ed
`events.csv` includono modalità, percorso/hash del record, identificativo,
revisione, origine di larghezza/lunghezza e dell'eventuale scala esplicita.
`metrics.csv` e `metrics.json` riportano la modalità dell'esperimento. Le
esportazioni sono ricostruibili dai risultati atomici senza duplicarli.
I metadati assenti nei risultati storici restano null nei JSON e vuoti nei CSV.

Il controllo di `code_signature()` resta attivo: gli esperimenti preparati con
codice diverso sono consultabili, ma la ripresa richiede il codice originale.
Non modificare gli hash dei vecchi manifest per continuare con la logica nuova.

## Validazione

I test della fase 6 coprono due video con calibrazioni differenti, due modelli
e due modalità, riuso del segmentatore, pipeline reale con archivio rimosso,
stop/ripresa anche in un vero subprocess, controllo delle copie, modifiche ai
video con dimensione/mtime invariati, invalidazione per movimento della camera,
assenza di commit non valutabili, provenienza CSV e separazione delle modalità
in Streamlit AppTest.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short tests/integration/test_metric_batch.py tests/integration/test_batch.py tests/integration/test_batch_ui.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --junitxml=outputs/phase6-verification/pytest-full.xml
.\.venv\Scripts\python.exe -m ruff check apps src tests
```

Rapporti in `outputs/phase6-verification/`. I test usano video temporanei e
segmentazione controllata/sintetica; il confronto fra pesi YOLO reali e la
valutazione scientifica sui video dell'utente appartengono alla fase 7.
La verifica visiva nel browser resta da eseguire: AppTest non dimostra il
funzionamento del trascinamento JavaScript. Non è stato avviato il batch dei
150 video né modificato quello storico: tutti i suoi 33 file coincidono con
gli hash salvati prima dell'audit. La suite finale disabilita la cache di pytest
per evitare l'avviso di permessi sulla cache emerso nel primo test mirato.

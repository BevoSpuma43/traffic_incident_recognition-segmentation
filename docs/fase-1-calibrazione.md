# Fase 1 — Record, archivio e geometria

Data: 6 ottobre 2026 (Europe/Rome). Stato: implementata e verificata nel backend.
L'editor Streamlit della Fase 2 e l'integrazione automatica nella pipeline della
Fase 4 restano da implementare.

## Moduli e contratti

- `calibration/records.py`: documento `CalibrationRecord` schema 1, stati
  `draft`/`confirmed`/`invalid`, identità del video, riferimento, vertici con ID,
  ROI indipendente, provenienza per distanza, metadati dell'automazione e runtime.
- `calibration/repository.py`: primo frame con PyAV, hash del video e dei pixel,
  ricerca delle calibrazioni, salvataggio atomico con revisioni, verifica dei
  riferimenti e importazione esplicita dei vecchi dati metrici.
- `calibration/coordinates.py`: rettangolo metrico, validazione dei quattro
  vertici ordinati, trasformazioni originali/elaborate/display e adattamento
  di omografia, ROI, immagine e soglia in pixel della guardia camera.
- `calibration/homography.py`: `load_calibration` legge ancora i vecchi YAML;
  riconosce anche i nuovi record, rifiutando bozze, record invalidi e riferimenti
  mancanti o alterati prima di restituire la calibrazione runtime.

Non sono state aggiunte dipendenze o cambiate GUI, detector, motore batch o
protocollo di valutazione.

## Stato dell'editor e scala

Una bozza ammette meno di quattro punti e distanze mancanti, ma non contiene
una `Calibration` runtime. I valori numerici presenti devono essere finiti;
le distanze presenti devono essere positive.

Gli ID P1–P4 determinano le corrispondenze:

```text
P1 → (0, 0)
P2 → (larghezza, 0)
P3 → (larghezza, lunghezza)
P4 → (0, lunghezza)
```

Il backend non riordina i vertici in base alla loro posizione sullo schermo.
Il trascinamento può quindi modificare la geometria senza scambiare larghezza
e lunghezza. Quattro punti incrociati, concavi, duplicati, quasi collineari,
fuori immagine o con area insufficiente vengono rifiutati alla conferma.

Per ogni distanza si conservano valore, unità `m`, origine
`measured`/`standard`/`experimental`/`unknown`, fonte, eventuale preset e conferma
dell'utente. Entrambe le distanze devono avere valore e provenienza confermata
per costruire il rettangolo metrico. Un'ipotesi confermata resta un'ipotesi.

`edit_record` elimina sempre il runtime e riporta il record a bozza; modifiche
alla geometria azzerano anche il punteggio geometrico, salvo un nuovo punteggio
esplicitamente fornito dal chiamante. `confirm_record` non aumenta il punteggio.
Il valore iniziale è zero (qualità non ancora valutata): la futura UI dovrà
gestire il punteggio geometrico prima di applicare la soglia della pipeline.
La conferma e la riproiezione dei quattro punti non dimostrano l'accuratezza
delle misure fisiche.

In assenza di ROI esplicita, il nuovo flusso utilizza tutta l'immagine, anziché
dedurre l'area valutata dal rettangolo di calibrazione. Il comportamento legacy
di `estimate_calibration` è preservato.

## Identità e primo frame

`create_draft` restituisce il record e l'immagine BGR del primo frame
decodificabile, senza cercare uno sfondo nel resto del filmato. Il decoder è
chiuso anche in caso di errore. Video vuoti, senza stream video o senza frame
decodificabili producono un errore esplicito.

Si conservano dimensioni effettive dell'array, hash SHA-256 del video,
dimensione in byte, percorso originale e relativo quando possibile, nome,
rotazione e hash dei pixel del primo frame. Il riferimento conserva indice,
PTS, time base e timestamp originale, che resta assente quando non disponibile.
L'indice è l'ordinale dei frame decodificati, non un indice ricostruito assumendo
FPS costanti. Un pacchetto corrotto scartato non permette di inventare il numero
di frame mancanti.

L'immagine mantiene l'orientamento nativo di PyAV, coerente con `VideoSource`.
I metadati di rotazione vengono registrati e confrontati; l'applicazione di
rotazioni nella pipeline non è stata introdotta in questa fase.

## Archivio e revisioni

La destinazione predefinita è `data/calibration/videos/<video-stem>.yaml`.
Collisioni fra cartelle, estensioni o identità diverse usano sottocartelle,
conservando lo stem del video nel nome YAML. I nomi vengono resi validi su
Windows, inclusi i nomi riservati e quelli che assomigliano ai file di revisione.

`find_record` restituisce `missing`, `compatible`, `incompatible` o `ambiguous`.
La compatibilità richiede hash, dimensione in byte, risoluzione, rotazione e
primo frame coincidenti. Un file rinominato con contenuto invariato può essere
riconosciuto; un video sostituito mantenendo lo stesso nome è incompatibile.
Più record compatibili richiedono una scelta esplicita. Stato della bozza e
compatibilità della sorgente rimangono concetti separati.

Un salvataggio pubblica, nell'ordine:

1. `clip.reference-r0001.png`, senza perdita dei pixel del frame;
2. `clip.revision-0001.yaml`, copia immutabile di quella revisione;
3. `clip.yaml`, puntatore alla revisione corrente.

Ogni file usa temporaneo, flush/fsync e sostituzione atomica. La coppia YAML/PNG
è riletta e verificata prima della pubblicazione del file corrente. Un lock del
sistema operativo impedisce salvataggi concorrenti; la revisione deve coincidere
con quella sul disco per evitare di sovrascrivere modifiche più recenti.

Un errore prima della pubblicazione finale conserva il record precedente.
Una revisione rimasta sul disco dopo un errore non viene sovrascritta: un nuovo
tentativo usa il numero successivo disponibile. Non vengono cancellati
automaticamente immagini o revisioni precedenti.

Dopo `save_record`, ricaricare il percorso restituito per ottenere la revisione
salvata, prima di effettuare un nuovo salvataggio.

## API per le fasi successive

```python
from cctv_incident.calibration.repository import create_draft, save_record, load_record, find_record
from cctv_incident.calibration.records import edit_record, confirm_record, to_runtime_calibration
from cctv_incident.calibration.coordinates import ImageTransform

draft, frame = create_draft(video_path, project_root)
match = find_record(draft.video, archive_root)
# L'editor popolerà vertices, width, length, ROI e qualità tramite edit_record.
draft_path = save_record(draft, archive_root, reference_image=frame)
saved = load_record(draft_path)
# confirm_record(saved) richiede geometria e distanze complete e confermate.
# to_runtime_calibration restituisce una copia solo da un record confermato.

transform = ImageTransform.resize(original_size, actual_processed_size)
# H_processed = H_original @ inverse(T)
# transform.adapt_calibration(calibration) conserva il record originale.
# transform.adapt_reference(frame) e adapt_motion_threshold(threshold_px)
# devono essere applicati dalla futura integrazione nella pipeline.
```

Per l'importazione legacy, `import_legacy_calibration` richiede associazione
visiva verificata al video e provenienza esplicita del sistema metrico. Produce
una bozza con le corrispondenze originali e una copia dei metadati legacy.
Le corrispondenze avanzate con più di quattro punti restano supportate.
Una calibrazione canonica non viene convertita implicitamente in metri.

## Verifiche effettuate

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short tests/unit/test_calibration_records.py tests/unit/test_calibration_coordinates.py tests/unit/test_geometry.py tests/unit/test_batch_evaluation.py tests/integration/test_batch.py tests/integration/test_batch_ui.py tests/integration/test_ui.py tests/integration/test_tracking_calibration.py tests/integration/test_pipeline.py tests/integration/test_image_pipeline.py
.\.venv\Scripts\python.exe -m ruff check apps src tests
.\.venv\Scripts\python.exe -m ruff format --check src/cctv_incident/calibration tests/unit/test_calibration_records.py tests/unit/test_calibration_coordinates.py
git diff --check
```

- **64 test superati in 16,30 s**, inclusi 34 nuovi casi parametrizzati nei due
  file unitari e le regressioni pertinenti su pipeline, batch e GUI.
- Ruff, formato dei file interessati e whitespace: superati.
- Geometria sintetica con punti metrici indipendenti recuperata entro `2e-6 m`;
  trasformazioni fra risoluzioni equivalenti entro `1e-10` nei test numerici.
- Verificati distanze/ROI dopo resize con arrotondamento, margini del display,
  persistenza di bozze/conferme, revisioni obsolete, collisioni, sostituzione del
  video, alterazione del riferimento, errore di pubblicazione, lock e legacy.

Il primo tentativo nel sandbox ha incontrato le restrizioni già note sulla
directory temporanea di pytest. I test sono poi stati eseguiti con i permessi
autorizzati, senza modificare le dipendenze.

Non sono stati eseguiti un batch sui 150 video, una valutazione scientifica
dell'errore reale o un controllo del trascinamento nel browser. La Fase 1
fornisce il backend; la GUI rimane quella preesistente.

## Esperimento dell'utente e prossima fase

I 33 file dell'esperimento `outputs/batches/yolo26s-seg-55517fe0827b34e6` sono
stati confrontati tramite hash prima e dopo il lavoro: nessuna aggiunta,
rimozione o modifica. Rimane in pausa a 2/150, senza worker attivo.

La nuova firma del codice è:

```text
f6d629eef4abf212a7a8901aade8aebe0d11dcb01ae07faf01bdc832ad487778
```

Differisce da quella del manifest precedente: la ripresa del vecchio batch
con questo codice verrà rifiutata. Il suo manifest non è stato aggiornato per
aggirare il controllo. I risultati restano consultabili; per continuare quel
job serve il codice precedente, mentre il nuovo codice richiede un nuovo job.

Prossima fase: **Fase 2 — Editor grafico riutilizzabile**, da avviare su istruzione
dell'utente.

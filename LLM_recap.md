# Contesto per riprendere il progetto

**Per riprendere in una nuova sessione, leggere prima il
[recap completo consolidato dell'8 ottobre 2026](RECAP_COMPLETO_2026-10-08.md).**
Include le decisioni dell'utente, lo stato corrente verificato, i contratti del
codice, le prove e i limiti aperti. Questo file conserva anche la cronologia:
le descrizioni storiche superate vanno interpretate secondo il recap consolidato.

Aggiornato il **8 ottobre 2026**, fuso dell'utente **Europe/Rome**.
Questo documento riassume la conversazione e il lavoro effettivamente svolto.
È un passaggio di consegne, non una richiesta di avviare automaticamente nuovi esperimenti.
Prima di intervenire, verificare lo stato corrente dei file: potrebbe essere cambiato dopo questa fotografia.

## 1. Stato da cui ripartire

**Aggiornamento corrente — Editor automatico, 8 ottobre 2026.** Su richiesta
dell'utente, Calibrazione automatica applica subito la prima proposta e nasconde
istruzioni/coordinate avanzate/controlli ROI della selezione manuale. Un unico
canvas mostra i punti già ottenuti, trascinabili; la scelta di un'alternativa
si applica subito, senza Usa proposta nell'editor. Le correzioni non vengono
sovrascritte dai rerun. Torna alla selezione manuale conserva il lavoro; risultati
vuoti/errori permettono lo stesso ritorno senza perdere i punti precedenti.

Preferenza esplicita dell'utente: **precompilare solo distanze disponibili nella
proposta o da un preset scelto**, senza ipotesi generiche 0,60 × 3,00 m. I valori
disponibili appaiono direttamente nei campi metrici modificabili, con origine e
conferma ancora richieste. Cambiare candidato non trasferisce le distanze del
riferimento precedente. Le bozze automatiche riaperte mantengono correzioni e
accesso ai preset; le sessioni della versione precedente vengono adattate al
nuovo flusso durante il reload, senza azzerare punti o misure.

**288 test superati in 75,18 s**, zero errori/fallimenti/saltati. Tre nuovi casi
più regressioni aggiornate per applicazione immediata, trascinamento attraverso
il callback del componente, valori precompilati, cambio candidato, errore e
ritorno manuale, riapertura e sessione live precedente. Ruff/formato superati.
Rapporti: `outputs/automatic-editor-verification/`. Usata la `.venv` esistente;
il servizio del terminale isolato non si avviava, quindi i comandi di verifica
sono stati eseguiti fuori dal sandbox tramite le autorizzazioni degli strumenti.
Nessuna verifica visiva nel browser dichiarata. Aggiornati README, tutorial e
guida fase 3. Questo intervento riguarda il flusso UI, non ricalibra l'euristica
di qualità geometrica discussa nella domanda precedente.

**Stato della fase 7 al 7 ottobre — software implementato e validazione reale parziale.**
Compatibilità versione 1/2 e consultazione anche senza pesi; GUI distingue lettura
e ripresa, disabilitando avvio/ripresa con codice incompatibile. `start_job`
valida prima di modificare checkpoint/stop flag o lanciare il worker. Hash del
codice mai aggirati. `export_results(job, output_dir=...)` permette esportazioni
separate: recuperati i 2 risultati storici su 150 senza modificare i 33 file
del batch, verificati rispetto alla baseline dell'audit.

Nuovi `calibration/evaluation.py`, `batch_comparison.py` e
`scripts/evaluate_calibration.py`: campione di proposte senza etichette,
schede di revisione con tempi/correzioni espliciti, confronto fra job completi
con controlli su video, etichette, modello, parametri, FPS, ROI e snapshot.
Il confronto dichiara le differenze dei detector; non è un'ablazione pura
dell'omografia. Esempio condiviso: `configs/paired-evaluation.yaml`.

Campione realmente elaborato: 5/150 video, seed 42, primo frame, algoritmo
predefinito, zero errori e 4 video con proposte (80% di copertura grezza, non
accuratezza). Anteprime ispezionate dall'assistente: riferimenti anche piccoli
o ambigui; nessuna conferma nell'archivio. Artefatti e osservazioni in
`outputs/phase7-verification/real-sample/`. L'utente dichiara **nessuna misura
indipendente**. L'archivio contiene un solo record corrente, bozza senza distanze:
nessun confronto metrico reale avviato, nessuna misura di errore fisico inventata.
Revisioni umane e tempo medio ancora non misurati.

**285 test superati in 109,81 s**, inclusi 14 nuovi casi. Zero errori/fallimenti/
saltati; cinque gruppi JavaScript superati in V8, Ruff/formato di 133 file e
whitespace superati. Node non è nel PATH; non installato alcun runtime aggiuntivo.
La prima suite trovava una fixture ROI che azzerava la qualità geometrica;
corretta la fixture con qualità sintetica esplicita, senza abbassare soglie.
Windows non ha bloccato pytest; usata la `.venv` già esistente.

**Verifica GUI reale non completata:** Computer Use Windows restituisce pipe
nativa indisponibile (`os error 2`); inventario alternativo vuoto (nessuna app o
browser). Nessuna modifica alle protezioni Windows. Checklist, comandi e limiti:
[Fase 7](docs/fase-7-validazione.md). Per chiudere la validazione reale servono
collaudo visivo documentato, revisioni cronometrate e calibrazioni confermate
per il confronto delle due modalità. L'errore metrico fisico richiede inoltre
misure indipendenti, attualmente assenti. Non dichiarare conclusa questa parte.

**Stato storico — Fase 6 implementata e validata:** batch metrico disponibile
dal menu laterale **Batch con omografia → Analisi batch**, dopo la preparazione
e conferma individuale delle calibrazioni. Manifest versione 2, mappa esplicita
video → record, snapshot YAML/PNG verificati e indipendenti dall'archivio centrale,
identità/revisione/origine delle misure nei risultati e nei CSV. Esperimenti e
widget separati per modalità; selezione filtrata anche per modello. Un solo
segmentatore per worker; stop/ripresa e commit atomici conservati. Invalidazioni
metriche interrompono il worker senza conteggiare il video e richiedono un nuovo
esperimento. I progressi della pipeline ora espongono la validità della calibrazione.

**271 test superati in 89,23 s**, 23 nuovi casi, zero errori/fallimenti/saltati;
inclusi pipeline reale, vero subprocess metrico con stop/ripresa e Streamlit
AppTest. Ruff superato, 128 file formattati correttamente, `git diff --check`
superato. XML e riepilogo in `outputs/phase6-verification/`. Windows non ha
bloccato i test; usata soltanto la `.venv` esistente. Cache pytest disabilitata
nella suite finale dopo un avviso di permessi sulla cache nel test mirato.
I 33 file del batch storico sono invariati rispetto agli hash dell'audit;
nessuna riscrittura di `code_signature()` e nessun batch sui 150 video avviato.
Guida: [Fase 6](docs/fase-6-batch-metrico.md); aggiornati tutorial e README.
Resta la fase 7, inclusi prova visiva nel browser e valutazione su misure reali.
La richiesta corrente autorizzava la fase 6; la fase 7 non è stata avviata.

**Stato storico — Fase 5 implementata:** preparazione persistente in
`outputs/calibration-preparations`, separata dall'inferenza. Tabella degli stati,
provenienza, editor condiviso, alternative riapribili e cursore di revisione
persistente. Worker separato per proposte mancanti, stop/ripresa, controlli hash
e recupero dopo arresto. Protegge tutte le bozze/conferme e i salvataggi manuali
concorrenti; non legge etichette e non carica YOLO. **248 test superati in 41,33 s**,
17 nuovi casi, incluso worker reale e AppTest. Ruff/formato/whitespace superati.
Windows non ha bloccato la validazione. I 33 file del batch precedente sono
invariati; nessuna generazione sui video reali dell'utente è stata avviata.
Guida: [Fase 5](docs/fase-5-preparazione.md); aggiornato `start_tutorial.md`.
Restano prova visiva nel browser, accuratezza su misure reali e fasi 6–7.
La voce batch con omografia ora apre la preparazione, ma non avvia inferenza.

**Stato storico — Fase 4 implementata e validata automaticamente:** 231 test
superati in 45,85 s nella `.venv` esistente, senza errori o test saltati.
35 test mirati, cinque gruppi JavaScript, Ruff/formato/whitespace superati.
Completato il collaudo GUI → riuso record → pipeline → replay; la modalità del
menu prevale su YAML personalizzati. Log per frame con validità, esito coerente
in run/metriche e riepilogo conservato nella GUI dopo errori di runtime.
Prova con segmentazione sintetica, ByteTrack e vera guardia camera: traiettoria
nota verificata a 640×360 e 321×181; spostamento camera invalidato correttamente.
Corretta la fixture geometrica passando da MPEG-4 a FFV1 senza perdita, senza
allargare le tolleranze. Windows non ha bloccato la validazione; nessuna modifica
alle protezioni o alle dipendenze. Resta controllo manuale del canvas nel browser
(browser integrato indisponibile) e verifica dell'accuratezza fisica su misure
reali. Procedura e contratti: [Fase 4](docs/fase-4-pipeline.md). Rapporti in
`outputs/phase4-verification/`. Non avviare le fasi 5–7 o il batch metrico
senza richiesta; nessun commit richiesto o eseguito.

**Risultato storico dell'audit precedente:** Fasi 1–3 e modifiche
parziali della Fase 4 revisionate. Corretti quattro difetti riprodotti:
rilettura del riferimento senza verifica dei byte usati, falsa valutabilità
in modalità immagine dopo invalidazione, stato non valutabile perso nell'export
e replay su un video sostituito della stessa risoluzione. Aggiornato inoltre
il test YOLO che usava unità canoniche nella modalità metrica, ora vietate.
**224 test superati in 46,97 s**, 13 nuove regressioni, Ruff/formato/whitespace
e cinque gruppi JavaScript superati. Usata la `.venv` esistente; il blocco pandas
non si è riprodotto, senza modifiche alle protezioni Windows o alle dipendenze.
I 33 file del batch sono invariati rispetto agli hash precedenti alla Fase 4.
Alla fine dell'audit la **Fase 4 era in corso**: erano presenti gate
metrico, adattamento al resize, snapshot e invalidazioni; ripartire da questi.
Restano verifica del canvas nel browser e prova metrica con misure indipendenti.
Dettagli: [audit delle fasi 1–4](docs/audit-fasi-1-4.md). Rapporti in
`outputs/audit-fasi-1-4`. Nessun batch avviato, nessun commit eseguito.

I paragrafi seguenti conservano i risultati **storici** delle singole fasi;
per lo stato attuale prevalgono l'aggiornamento sopra e il rapporto di audit.

Fase 3: codice implementato su richiesta dell'utente, **validazione finale
in sospeso**. Aggiunti generatore di candidati da contorni/linee, gestione
prospettica dei punti di fuga, scelta delle alternative nell'editor e preset
USA espliciti con provenienza. Preset verificati nel PDF FHWA corrente;
nessuna scala inventata, nessun nuovo modello. Ruff e formato superati.
Validazione ripresa nella **`.venv` già esistente**, come richiesto dall'utente:
Python 3.11.15 torna ad avviarsi; nessuna ricreazione o installazione eseguita.
**Tutti i 29 nuovi test passano** (25 unitari e 4 AppTest). Suite completa:
**200 superati, 7 falliti, 4 errori in 57,12 s**, nessun test saltato.
Gli 11 non superati dipendono dal blocco Code Integrity sulla DLL
`pandas/_libs/tslib.cp311-win_amd64.pyd` (eventi 3077/3033). Verificato nei log;
non confondere i successivi `StopIteration` della GUI con errori dei preset.
Corretto il timeout iniziale del test della bozza a 20 s. Wheel costruita offline
e provata da pacchetto estratto con la stessa `.venv`: proposta e salvataggio
bozza funzionano. Prova sui primi frame di quattro video reali completata senza
errori: candidati 0, 0, 0, 1, circa 0,45–0,52 s ciascuno; nessuna valutazione
scientifica o accettazione metrica. I 33 file del batch sono ancora invariati.
Rapporti XML/JSON e anteprime in `outputs/phase3-verification`.
Python ufficiale 3.12.10 già installato è stato trovato firmato e avviato,
ma i test finali usano la `.venv`, non un nuovo ambiente. Nessuna protezione
di Windows modificata. Restano verifica dell'app principale e controllo nel
browser: non dichiarare la suite completamente superata. Dettagli:
[Fase 3](docs/fase-3-proposte.md).

Fase 2 implementata il 6 ottobre: editor CCv2 sul primo frame, quattro clic,
trascinamento con invio al rilascio, punti numerati, dimensioni e provenienza,
ROI indipendente, coordinate numeriche/avanzate, reset, annullamento, bozze e
conferme. In **Segmentazione + omografia** l'analisi richiede un record salvato
e revisionato per il video selezionato. I 33 file dell'esperimento salvato
sono ancora invariati. **182 test superati in 25,17 s**, Ruff e formato superati,
cinque gruppi di prove JavaScript in V8 e wheel verificata fuori dai sorgenti;
controllo visivo nel
browser ancora da eseguire perché nessun browser è disponibile agli strumenti.
Dettagli e limiti: [Fase 2](docs/fase-2-editor.md). Alla fine della Fase 2 il prossimo lavoro era la Fase 3;
adattamento della pipeline alla risoluzione nella Fase 4, batch metrico più avanti.

Fase 1 del piano calibrazione/omografia completata nel backend il 6 ottobre:
record versionati, bozze/conferme, archivio per video con revisioni atomiche,
primo frame PyAV, trasformazioni geometriche e caricamento legacy.
**64 test mirati e di regressione superati**, Ruff e formato superati.
I 33 file del batch dell'utente sono invariati, ma la firma del codice è cambiata:
quel job resta leggibile e richiede il codice precedente per la ripresa.
Non aggiornare il suo manifest per aggirare il controllo. Alla fine della
Fase 1 GUI e pipeline non erano ancora integrate; l'editor è stato aggiunto
nella Fase 2, come descritto sopra. Per la Fase 3 vedere lo stato di validazione
in apertura; le Fasi 4–7 non sono implementate.
Contratti, limiti e comandi: [Fase 1](docs/fase-1-calibrazione.md).

Fase 0 del piano calibrazione/omografia completata il 6 ottobre: **27 test mirati
superati**, Ruff e `git diff --check` superati. Ambiente utilizzabile; batch
`yolo26s-seg-55517fe0827b34e6` preservato in pausa a 2/150, con manifest ancora
compatibile e hash di manifest/checkpoint invariati alla fine della Fase 0.
In quella fase non erano state apportate modifiche al codice. Risultati, versioni e punti di integrazione:
[baseline Fase 0](docs/fase-0-baseline.md).

Aggiornamento GUI del 6 ottobre: rimossa dal menu la modalità **Demo sintetica**
e il pulsante per generare video dimostrativi. L'apertura predefinita è ora
**Solo Segmentazione**. Le utility sintetiche da CLI e i relativi test
restano disponibili. **Segmentazione + omografia** analizza un singolo video
con omografia e misure reali della strada; non è il batch con omografia, tuttora
un segnaposto.

L'utente sta sviluppando un **progetto universitario di traffic incident recognition basato sulla segmentazione dei veicoli**. Comunica in italiano e lavora con VS Code su Windows.

È stata implementata e verificata una nuova modalità della GUI Streamlit che:

- analizza sequenzialmente tutti i video di una cartella;
- permette di scegliere il modello YOLO tra i pesi locali;
- confronta i risultati con le annotazioni ACCIDENT;
- esporta CSV per video, per evento e per metriche aggregate;
- conserva checkpoint ed esperimenti distinti per modello/configurazione/sottoinsieme;
- permette Stop e Riprendi, mostrando nome del video, posizione sul totale e percentuale;
- recupera il lavoro dopo riapertura della GUI o arresto del processo.

**L'implementazione del batch è conclusa, non soltanto pianificata.** Alla sua
conclusione la suite riportava **128 test superati**; dopo le fasi 1 e 2 di
calibrazione l'ultima suite completa riporta **182 test superati**. È stata
eseguita anche una prova con YOLO26s su due video reali, interrompendo e
riprendendo il primo video.

**Non è stata eseguita dall'assistente l'analisi completa dei 150 video.** Alla verifica del 6 ottobre esiste un batch dell'utente in pausa con 2/150 video completati in `outputs/batches/`; esiste anche l'esperimento di verifica separato in `outputs/batch-smoke/`. Verificare sempre lo stato attuale su disco.

Aggiornamento del 6 ottobre: l'utente ha richiesto una cartella distinta per ogni
nuovo batch e un salvataggio esplicito delle impostazioni. **Prepara batch** ora
crea sempre un nuovo ID, anche con parametri identici; **Riprendi** mantiene la
cartella originale. I nuovi esperimenti salvano `batch_config.json` con modalità,
percorso YAML, modello YOLO, directory del dataset e scelta del menu, directory
video, CSV etichette e FPS; `source_config.yaml` conserva lo YAML originale e
`resolved_config.yaml` i parametri effettivi. La GUI passa esplicitamente modalità
e percorso YAML a `render_batch_page` e mostra le impostazioni salvate.

L'ultima richiesta implementata (6 ottobre) riguarda cartelle indipendenti per ogni nuovo batch e salvataggio esplicito della configurazione. Non ci sono chiarimenti in attesa, commit o pubblicazioni richiesti, né un benchmark completo già avviato dall'assistente da continuare.

Verifica dell'aggiornamento del 6 ottobre: **16 test batch/GUI/metriche superati**.
Il batch dell'utente `yolo26s-seg-55517fe0827b34e6` è stato preservato con checkpoint
invariato (2/150, in pausa). Poiché la firma del codice comprende anche `prepare_job`,
è stata aggiornata solo la compatibilità del suo manifest dopo aver verificato che
le funzioni runtime e il codice di inferenza non fossero cambiati. Il manifest
originale è conservato in `manifest.before-output-config-20261006.json`, con
motivazione registrata in `compatibility_updates`; validazione di ripresa superata.
Questa è una migrazione puntuale per modifiche alla preparazione, non una deroga
generale ai controlli di versione. I nuovi file di configurazione completi sono
prodotti dai nuovi batch; lo YAML originale di esperimenti precedenti non viene inventato.

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

- `prepare_job`: valida cartella/metadati/pesi, congela configurazione e manifest e crea sempre un nuovo esperimento.
- ID univoco con modello, timestamp UTC e suffisso casuale; `protocol_sha256` conserva separatamente l'impronta dei parametri per confrontare esperimenti equivalenti.
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

Verifica storica prima del piano di calibrazione; per quella corrente vedere
lo stato delle fasi in apertura e [Fase 2](docs/fase-2-editor.md):

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

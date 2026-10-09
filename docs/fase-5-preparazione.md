# Fase 5 — Preparazione persistente delle calibrazioni

Implementata il **7 ottobre 2026**. Suite completa: **248 test superati in
41,33 s**, senza test saltati, errori o fallimenti, nella `.venv` esistente.
La preparazione è disponibile; l'inferenza batch metrica è stata aggiunta nella
[fase 6](fase-6-batch-metrico.md), nel menu **Batch con omografia → Analisi batch**.

## Uso

1. Avviare Streamlit come indicato in [start_tutorial.md](../start_tutorial.md).
2. Selezionare **standard_dataset analisi in batch - con omografia**.
3. Indicare la cartella dei video e premere **Crea sessione di preparazione**.
   Vengono verificati gli input e mostrati stato della calibrazione, provenienza
   delle misure, revisione, stato della preparazione ed eventuale motivo di errore.
4. Premere **Proponi calibrazioni mancanti**. Un processo separato analizza solo
   i primi fotogrammi: non carica YOLO, non legge le etichette degli incidenti
   e non produce predizioni o metriche di classificazione.
5. **Stop proposte** attende il calcolo in corso; se non ancora salvato, quel
   video ripartirà dall'inizio. **Riprendi proposte** usa la sessione salvata e
   salta le voci già pubblicate. Chiudere browser o server non ferma il worker.
6. A worker fermo, selezionare una riga oppure **Video da revisionare**. L'editor
   è quello del video singolo: alternative automatiche, punti, ROI, misure,
   provenienza, preset espliciti, bozza e conferma individuale.
7. **Prossimo video da revisionare** passa alla prossima voce non confermata.
   La selezione viene salvata su disco e ripristinata riaprendo la pagina.

Le sessioni sono elencate in **Sessione di preparazione salvata**. Il campo
cartella vale per una nuova sessione; la ripresa usa sempre gli input salvati.
**Aggiorna stato e verifica video** aggiorna l'archivio, ricontrolla integralmente
gli hash e ricarica l'editor selezionato. Salva prima eventuali modifiche in
memoria: una bozza non salvata non è persistente.

## Stati e protezione del lavoro manuale

La tabella distingue **Mancante**, **Bozza**, **Confermato**, **Incompatibile**
ed **Errore**, mostrando separatamente l'esito della generazione. Il progresso
conta i video esaminati, compresi quelli con errore: non indica quante
calibrazioni siano pronte per l'inferenza.

Il worker propone solo per video senza record compatibile e senza conflitti.
Conserva **sia le conferme sia le bozze esistenti**, comprese quelle modificate
manualmente. Nessuna conferma in massa. Una proposta senza riferimenti produce
una bozza vuota con motivo e diagnostica; larghezza e lunghezza restano vuote.

Il salvataggio automatico verifica nuovamente l'assenza di una calibrazione
sotto il lock dell'archivio: se nel frattempo vince un salvataggio manuale, lo
conserva senza creare una seconda calibrazione. Il worker non sovrascrive
calibrazioni incompatibili: è necessaria una revisione manuale.

Un record **Confermato** indica lo stato salvato, non certifica l'accuratezza
fisica né il superamento della soglia geometrica della pipeline. Le misure
da standard e le ipotesi restano distinguibili dalle misure osservate.

## Persistenza

Ogni sessione ha un ID e una cartella indipendente:

```text
outputs/calibration-preparations/preparation-<id>/
  manifest.json
  checkpoint.json
  review.json
  worker.json
  worker.log
  stop.request                 # solo dopo richiesta di stop
  items/000000/
    proposal.json              # tutti i candidati e la diagnostica, se generati
    mask.png
    preview.png
    result.json                # commit della preparazione di questo video
```

I record YAML/PNG restano nell'archivio condiviso `data/calibration/videos/`,
riutilizzabile tra modelli. `result.json` conserva riferimenti alla revisione
immutabile, hash degli artefatti e stato; non contiene TP/FP/FN/TN o eventi.

La pubblicazione del risultato è atomica. Un arresto dopo il salvataggio della
bozza ma prima del commit viene recuperato dall'archivio senza rigenerare o
incrementare la revisione. Le alternative vengono riaperte dal disco solo
quando corrispondono alla sessione, al fotogramma e alla ROI pertinente.

Prima dell'avvio/ripresa sono verificati lista e SHA-256 dei video, versione
del codice di calibrazione, algoritmo, hash dei parametri e integrità degli
artefatti completati. Modifiche manuali successive creano nuove revisioni e
non invalidano le copie storiche. Se input/codice/parametri cambiano, creare
una nuova sessione: i vecchi risultati restano consultabili.

Gli errori per singolo video restano in tabella e gli altri video proseguono.
Una voce di errore già pubblicata non viene ripetuta indefinitamente con
**Riprendi**: dopo aver corretto la causa, creare una nuova sessione; le
calibrazioni valide già archiviate saranno comunque conservate.

## Validazione

**17 nuovi casi** coprono:

- proposte multiple, risultato vuoto e provenienza non confermata;
- protezione di bozze/conferme, incompatibilità e salvataggio manuale concorrente;
- stop durante il calcolo, ripresa selettiva e recupero dopo arresto improvviso;
- controllo degli hash anche con dimensione/mtime invariati, artefatti alterati,
  variazioni di lista video, parametri o codice;
- errori di decodifica/generazione, senza risultati di classificazione;
- esclusione di worker concorrenti e vero subprocess con stop/ripresa;
- AppTest per alternative salvate, riapertura pagina, cursore persistente e
  voce del menu principale senza requisiti YOLO/etichette.

Ruff, formato di 102 file e whitespace superati. Windows non ha bloccato i test.
XML: `outputs/phase5-verification/pytest-full.xml`. I 33 file del batch
`yolo26s-seg-55517fe0827b34e6` sono invariati rispetto agli hash precedenti.
Il relativo manifest non è stato modificato per aggirare la firma del codice.

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short tests/integration/test_calibration_preparation.py
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --junitxml=outputs/phase5-verification/pytest-full.xml
```

Non è stata avviata la generazione sui 150 video dell'utente. Il collaudo usa
fixture temporanee; resta il controllo visivo nel browser del canvas e della
selezione della tabella. L'accuratezza metrica su strade reali richiede misure
indipendenti, come già indicato nella fase 4.

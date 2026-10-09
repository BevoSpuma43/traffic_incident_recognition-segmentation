# Fase 7 — Compatibilità e valutazione

Stato del 7 ottobre 2026: implementata la parte software; validazione empirica
parziale. Il collaudo visivo della GUI, le revisioni cronometrate e il confronto
fra modalità sui video reali non sono attestati come completati.
L'utente ha confermato di non avere misure indipendenti: l'errore metrico fisico
rimane **non misurato**.

Validazione automatica finale: **285 test superati in 109,81 s**, inclusi 14
nuovi casi della fase 7; zero errori, fallimenti o test saltati. Cinque gruppi
JavaScript superati nel runtime V8 disponibile (Node non è nel PATH). Ruff,
formato di 133 file e `git diff --check` superati. I 33 file del batch storico
coincidono integralmente con la baseline SHA-256 dell'audit. Windows non ha
bloccato questi test, eseguiti nella `.venv` esistente.

## Compatibilità degli esperimenti

La GUI distingue consultazione e ripresa. Se la firma del codice è diversa,
risultati, configurazione e download restano disponibili, ma **Avvia batch** e
**Riprendi** sono disabilitati con la spiegazione che serve il codice originale.
Anche senza i pesi locali, il modello del vecchio esperimento resta selezionabile
per consultarlo. Una nuova preparazione richiede invece pesi disponibili.

`start_job` verifica integralmente manifest, video, metadati, pesi e snapshot
prima di modificare checkpoint/stop flag o avviare il subprocess. Il worker
ripete i controlli. `code_signature()` non è stato indebolito né riscritto nei
manifest storici.

I manifest versione 1 e 2 sono leggibili. La nuova esportazione mantiene null
o celle vuote per provenienze assenti. Per ricostruire i CSV senza toccare i file
dell'esperimento:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py export outputs/batches/yolo26s-seg-55517fe0827b34e6 outputs/legacy-export-nuovo
```

Il comando richiede una destinazione nuova. È stata realmente eseguita una
riesportazione del batch storico in `outputs/phase7-verification/legacy-export`:
2 video completati su 150, metriche parziali coerenti con i risultati salvati.
Questo recupero non costituisce una nuova inferenza o una valutazione completa.

## Campione reale, senza etichette

Il campione comprende 5 video estratti dai 150 di `dataset/standard_dataset`,
con ordinamento dei percorsi e `random.Random(42).sample`. Sono stati letti i
primi frame e l'identità dei file, senza leggere le etichette degli incidenti.
Parametri predefiniti di `painted-contours-v1`; nessuna modifica dell'algoritmo
o scelta di parametri basata sui risultati del campione.

| Video | Candidati | Ispezione visiva delle anteprime da parte dell'assistente |
|---|---:|---|
| `1ZQAChv0TEI_00.mp4` | 1 | Candidato vicino al bordo sinistro della strada; natura del riferimento da verificare. |
| `6T58bsyZz40_00.mp4` | 1 | Riferimento molto sottile e prospettico; rettangolarità fisica non accertata. |
| `CQaoY2QypM8_00.mp4` | 1 | Candidato minuscolo in una scena notturna con riflessi e watermark; non validato. |
| `D1eP4Bn4hDQ_0_00.mp4` | 5 | Barre di attraversamento riconoscibili; distanze fisiche non disponibili. |
| `FnahNGsmeSQ_00.mp4` | 0 | Nessuna proposta; resta necessaria una calibrazione manuale se possibile. |

**Copertura grezza: 4/5 = 80%; errori di elaborazione: 0.** La copertura significa
soltanto “almeno una proposta”, non precisione, accettabilità o percentuale di
calibrazioni metriche valide. Il campione è piccolo e non giustifica conclusioni
sull'intero dataset. Le osservazioni dell'assistente non equivalgono a una
revisione utente cronometrata o a una conferma nell'archivio.

Artefatti: `outputs/phase7-verification/real-sample/`, con `sample.json`,
`report.json`, `reviews.json` e sottocartelle `000`–`004` contenenti riferimento,
maschera e anteprima. Sono registrati SHA-256, parametri, identità del video e
tempo di generazione, distinto dal tempo umano di revisione.

Per ripetere su una destinazione nuova:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py sample dataset/standard_dataset outputs/nuovo-campione --count 5 --seed 42
```

Il comando non salva né conferma record nell'archivio centrale e non carica YOLO.

## Misurare revisione e correzioni

In `reviews.json`, ogni video parte da `decision: "pending"` e tempi assenti.
Cronometrare il lavoro attivo di revisione, escludendo pause e generazione.
Compilare `reviewer`, `review_seconds` e una decisione:

- `accept`: proposta utilizzabile senza correzioni geometriche; indicare `candidate_id`;
- `correct`: indicare candidato e quattro `corrected_points_px` finali in pixel
  nativi, mantenendo l'ordine dei punti;
- `reject`: proposte presenti ma non utilizzabili;
- `no_reference`: nessuna proposta generata;
- `pending`: revisione ancora da fare, senza dichiarare tempi o correzioni.

La decisione riguarda la geometria proposta. Non conferma le dimensioni fisiche
e non abilita l'analisi: la conferma del record resta nell'editor.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py review outputs/phase7-verification/real-sample
```

Il report verifica l'integrità delle immagini, i riferimenti ai candidati, la
geometria corretta e i tempi finiti/positivi. Riporta quanti video sono stati
revisionati, quanti corretti, quanti punti modificati e la media dei soli tempi
effettivamente registrati. Attualmente **0 revisioni umane**, tempo medio e
correzioni dei punti **non misurati**. Non sono stati inventati tempi o misure.

## Confronto fra modalità

Il progetto dispone al momento di un solo record reale, in bozza e privo di
distanze. Non è stato avviato un confronto metrico sui video reali. Mancano
calibrazioni confermate; non sarebbe valido trattare errori o video saltati
come risultati negativi.

Quando le calibrazioni sono disponibili:

1. Scegliere lo stesso sottoinsieme in entrambe le modalità, prima di esaminare
   le etichette. Se alcune scene non sono calibrabili, dichiarare l'esclusione
   e usare lo stesso sottoinsieme per entrambe.
2. Usare **lo stesso** `configs/paired-evaluation.yaml`, modello, CSV, FPS e
   cartella in entrambi i batch. La modalità scelta nella GUI prevale sullo YAML.
3. Usare l'intero frame come ROI (`roi_px: null` nei record metrici): la modalità
   immagine usa il frame intero. Il comparatore rifiuta una ROI metrica ristretta.
4. Confermare misure e provenienza, dichiarando esplicitamente eventuali ipotesi.
   Un preset o una distanza assunta non sono una misura indipendente.
5. Completare entrambi gli esperimenti con la stessa versione del codice.
6. Eseguire, sostituendo i due percorsi dei job:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_calibration.py compare outputs/batches/JOB_IMMAGINE outputs/batches/JOB_METRICO outputs/confronto-nuovo.json
```

Il comparatore controlla completezza/valutabilità, hash dei video e del modello,
etichette salvate, codice, FPS, resize, parametri condivisi, seed, guardia camera
e ROI. Verifica gli snapshot metrici. In caso di differenze produce motivi
espliciti e `metrics: null`, senza proporre un confronto numerico non controllato.

Le regole differiscono: `ImageEventDetector` usa moto normalizzato rispetto ai
veicoli e regole di contatto/occlusione/sideswipe; `EventDetector` usa distanze,
velocità, decelerazioni e TTC sul piano calibrato. Il confronto riguarda due
flussi di rilevamento, non l'effetto isolato dell'omografia. La soglia della
guardia camera metrica viene adattata dal riferimento nativo al resize; quella
immagine è espressa nei pixel elaborati. Non dichiarare equivalenza fisica delle
soglie o accuratezza metrica per scale soltanto ipotizzate.

La procedura è collaudata su due video temporanei con pipeline reali e segmentatore
controllato. Questo verifica software e controlli di confronto, non prestazioni
YOLO sul campione reale.

## Collaudo browser da completare

Tentativi effettuati in questa sessione: Computer Use Windows non si collega
alla pipe nativa (`os error 2`); l'inventario alternativo non espone browser o app.
Non è stato aggirato Windows Code Integrity né modificata alcuna protezione.
Streamlit AppTest e i test JavaScript non sostituiscono questa prova visiva.

Avviare `.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py` e
annotare browser/versione, data, esito e difetti in un file di collaudo:

- [ ] In **Segmentazione + omografia**, creare P1–P4 e trascinare ogni punto.
      Il punto deve seguire il mouse e mantenere numero/lato associato.
- [ ] Ridimensionare finestra e barra laterale; punti e immagine restano allineati.
- [ ] Cambiare video: nessun punto o misura della sorgente precedente viene
      applicato implicitamente. Tornare al primo video e riaprire la bozza.
- [ ] Provare **Azzera punti**, **Nuova ROI** e **ROI: tutto il frame**.
      La ROI non deve cambiare il rettangolo di calibrazione.
- [ ] Salvare e riaprire una bozza. Una misura sconosciuta mantiene disabilitata
      la conferma metrica; un record confermato richiede riuso esplicito.
- [ ] Passare fra le due modalità batch e due modelli: liste corrette, download
      disponibili per lo storico, ripresa incompatibile disabilitata.

## Ripetere le verifiche automatiche

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --junitxml=outputs/phase7-verification/pytest-full.xml
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Rapporti in `outputs/phase7-verification/`. Il riepilogo distingue controlli
superati, campione ispezionato e verifiche ancora mancanti. Non considerare
chiusa la validazione reale finché GUI e revisioni non sono state documentate.

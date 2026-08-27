# Stato del progetto e modifiche recenti

Data di aggiornamento: 27 agosto 2026
Branch di riferimento: `main_v2`

## 1. Obiettivo del progetto

Il progetto analizza video stradali e segnala possibili collisioni tra veicoli.
La pipeline combina:

1. segmentazione delle istanze con un modello Ultralytics YOLO;
2. assegnazione degli ID temporali tramite ByteTrack;
3. stima della cinematica dei veicoli in spazio immagine;
4. analisi temporale del contatto tra coppie di veicoli;
5. rendering degli avvisi e registrazione di eventi diagnostici.

Il sistema rimane un rilevatore sperimentale e spiegabile. Non deve essere
considerato un sistema certificato per applicazioni di sicurezza.

## 2. Stato attuale

La pipeline principale, la GUI e la CLI sono operative. L'algoritmo di
collisione è stato trasformato da un controllo istantaneo basato sul solo
overlap a un rilevatore temporale con memoria per coppia. È inoltre disponibile
un sistema di benchmark che utilizza direttamente i video reali e i relativi
metadati.

Verifiche effettuate:

- compilazione dei moduli `src` e `tests` completata;
- suite automatica: 39 test superati su 39;
- benchmark end-to-end completato su un video reale;
- 2.027 righe di metadati associate a 2.027 file MP4;
- nessun path video mancante o duplicato.

## 3. Modifiche all'algoritmo di collisione

### 3.1 Cinematica dei track

La velocità non viene più stimata dal semplice spostamento del centroide tra
due frame consecutivi. Il sistema ora:

- usa il punto inferiore centrale della bounding box come ancora di movimento;
- normalizza velocità e accelerazione per il numero di frame trascorsi;
- applica un filtro esponenziale EMA alla velocità;
- conserva una cronologia limitata dei campioni cinematici;
- invalida la cinematica dopo gap di tracking troppo lunghi;
- impedisce ai track nuovi di essere immediatamente classificati come fermi;
- conserva, in caso di ID duplicati nello stesso frame, la detection con
  confidenza maggiore.

Queste modifiche riducono gli spike causati dalle deformazioni delle maschere,
dalle detection intermittenti e dai duplicati del tracker.

### 3.2 Evidenza spaziale

Il contatto tra due veicoli combina più indicatori:

- distanza tra bounding box come pre-filtro economico;
- area assoluta di intersezione tra maschere;
- overlap normalizzato rispetto alla maschera più piccola;
- contatto tra maschere leggermente dilatate, utile quando le sagome sono
  adiacenti ma non condividono pixel.

L'overlap non viene più trattato come prova sufficiente di una collisione:
occlusione prospettica e sovrapposizione nell'immagine possono infatti avvenire
anche durante una normale circolazione.

### 3.3 Evidenza temporale e dinamica

Per ogni coppia ordinata di track viene mantenuto uno stato persistente. Un
candidato richiede contatto recente, movimento precedente e almeno
un'anomalia coerente con un impatto:

- avvicinamento relativo;
- forte decelerazione;
- transizione recente da movimento ad arresto;
- arresto di entrambi i veicoli, accettato solo in presenza di moto precedente.

Contatto e anomalia possono presentarsi in frame vicini grazie a una finestra
temporale. Il candidato deve persistere per più frame prima della conferma.
Dopo l'emissione, la coppia entra in cooldown per evitare duplicati continui.

Ogni `CollisionEvent` include track coinvolti, frame di primo contatto e
conferma, overlap, distanza, velocità relativa, dati cinematici, motivazione e
un punteggio di confidenza spiegabile.

## 4. Configurazione attuale

Le soglie principali sono centralizzate in `src/config.py`:

| Parametro | Valore predefinito |
| --- | ---: |
| Overlap minimo | 120 px |
| Overlap normalizzato | 0,015 |
| Distanza massima di contatto | 8 px |
| Dilatazione maschere | 3 px |
| Velocità minima pre-impatto | 3 px/frame |
| Velocità minima di avvicinamento | 1,5 px/frame |
| Finestra temporale | 5 frame |
| Conferme richieste | 2 frame |
| Cooldown della coppia | 30 frame |
| Filtro EMA della velocità | 0,45 |

Questi valori sono iniziali e devono essere calibrati sul dataset, evitando di
ottimizzarli sulla sola sequenza usata per lo smoke test.

## 5. Dataset reale

I video sono in `dataset/real_videos`; la ground truth è in
`dataset/metadata-real.csv`.

Distribuzione corrente:

| Tipo | Video |
| --- | ---: |
| `single` | 680 |
| `t-bone` | 657 |
| `rear-end` | 328 |
| `sideswipe` | 245 |
| `head-on` | 117 |

Lo split `split_in_distribution` contiene 507 video train e 1.520 test. Lo
split `split_geo_aware` contiene 454 video train e 1.573 test.

La valutazione usa il timestamp restituito dal decoder e lo confronta con
`accident_time`. `accident_frame` viene mantenuto come metrica parallela e come
fallback per log precedenti. Questa scelta è necessaria perché
`no_frames / duration` non coincide sempre con la cadenza temporale vicino
all'incidente. Due righe collocano inoltre `accident_frame` esattamente a
`no_frames`, cioè sul bordo finale dichiarato del filmato.

Tutti i video del dataset contengono un incidente. Il dataset consente quindi
di misurare eventi mancati, fuori finestra o duplicati, ma non è sufficiente a
stimare il tasso di falsi allarmi su traffico senza incidenti.

## 6. Benchmark e diagnostica

Il modulo `src/benchmark.py` supporta:

- selezione per split e tipologia;
- limite sul numero di video;
- ripresa di esecuzioni interrotte con `--resume`;
- elaborazione senza rendering;
- log JSONL separati per video;
- precision, recall, F1 e ritardo in frame e secondi;
- throughput, tempo di inizializzazione e fattore real-time;
- metriche per tipo, rollover, scenario, meteo, fascia oraria e qualità;
- diagnostica opzionale di detection, track e coppie per ogni frame.

Esempio su 10 video del test set:

```powershell
.\.venv\Scripts\python.exe -m src.benchmark `
  --metadata dataset\metadata-real.csv `
  --dataset-root dataset `
  --model yolo26n-seg.pt `
  --split-field split_in_distribution `
  --split test `
  --limit 10 `
  --resume
```

Per un sottoinsieme ristretto destinato alla calibrazione si può aggiungere
`--diagnostics`. Questa opzione produce molti più dati e non è consigliata
sull'intero dataset.

## 7. Risultato dello smoke test

Video: `Z4kg2Ev3vhk_00.mp4`

Classe: `rear-end`
Condizioni: notte, qualità `Very_Poor`

| Metrica | Risultato |
| --- | ---: |
| Frame decodificati | 425 |
| Velocità di elaborazione | 12,53 FPS |
| Fattore real-time | 0,88× |
| Eventi rilevati | 0 |
| Falsi negativi | 1 |
| Frame con almeno una detection | 1/425 |
| Frame con almeno due track | 0/425 |
| Numero massimo di track attivi | 0 |

Il falso negativo non è causato dalle soglie di collisione: l'algoritmo non
riceve mai due track validi da confrontare. Su questo campione il collo di
bottiglia è la segmentazione/tracciatura a monte, probabilmente aggravata da
notte e qualità video molto bassa.

Il risultato riguarda un solo video e non rappresenta le prestazioni globali
del progetto.

## 8. Test automatici

La suite comprende test per:

- geometria delle maschere e contatto dilatato;
- velocità, accelerazione e filtro EMA;
- gestione dei track, gap e ID duplicati;
- conferma temporale, cooldown e rifiuto dei veicoli già fermi;
- caricamento di `metadata-real.csv`;
- matching per timestamp e frame;
- aggregazione delle metriche per gruppo;
- selezione dei campioni del benchmark.

Comando verificato:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Risultato corrente: `39 passed`.

## 9. Limiti ancora aperti

1. Il modello può non segmentare i veicoli in scene notturne o di qualità
   molto bassa; senza almeno due track il rilevatore di collisione non può
   operare.
2. Velocità e accelerazioni rimangono espresse in pixel/frame e dipendono da
   prospettiva, risoluzione e posizione nella scena.
3. Un cambio di ID durante l'occlusione dell'impatto azzera parte della storia
   cinematica.
4. Le soglie non sono ancora calibrate su un campione rappresentativo.
5. Non è disponibile un insieme negativo di video senza incidenti.
6. Il benchmark CPU completo sui 2.027 video richiede tempi significativi; lo
   smoke test ha raggiunto meno della velocità real-time.

## 10. Prossimi passi consigliati

1. Eseguire un benchmark stratificato su un piccolo campione per ogni tipo,
   fascia oraria e livello di qualità.
2. Confrontare `yolo26n-seg.pt`, `yolo26s-seg.pt` e `yolo26m-seg.pt` su recall
   delle detection e costo computazionale.
3. Calibrare prima la soglia di confidenza YOLO e i parametri ByteTrack, poi le
   soglie del rilevatore di collisione.
4. Aggiungere video negativi e misurare i falsi allarmi per ora di filmato.
5. Valutare una trasformazione prospettica o una normalizzazione della
   cinematica rispetto alla scala apparente del veicolo.
6. Analizzare separatamente i casi `single`, che non possono essere rilevati da
   una logica limitata alle collisioni tra coppie di veicoli.

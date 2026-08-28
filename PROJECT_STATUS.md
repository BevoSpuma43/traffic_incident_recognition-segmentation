# Stato del progetto e modifiche recenti

Data di aggiornamento: 28 agosto 2026
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
- suite automatica: 52 test superati su 52;
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

### 3.4 Protezione dalle code già presenti

Se due track risultano già in contatto quando la coppia viene osservata per la
prima volta, il contatto viene classificato come preesistente. La coppia resta
disarmata anche se il jitter delle maschere produce falsi picchi di velocità o
decelerazione. Il riarmo avviene dopo una separazione stabile per più frame
consecutivi; un eventuale ricontatto successivo viene analizzato con le regole
normali.

Il disarmo è però limitato in due modi, perché nella versione precedente
impediva strutturalmente di rilevare gli urti reali.

Il contatto è considerato preesistente soltanto quando **entrambi** i track
sono appena comparsi. Una coppia nuova formata da un track maturo e da un ID
appena creato non dimostra che i due veicoli fossero già accostati: nella
grande maggioranza dei casi il tracker ha riassegnato un ID mentre le sagome si
occludevano, cioè proprio durante l'impatto. Sul campione diagnostico misurato,
tutte le coppie disarmate nella finestra dell'incidente avevano un track a
`observed_frames = 1` e l'altro fra 16 e 149 frame.

Il disarmo non è inoltre più permanente: dopo `preexisting_contact_max_frames`
frame consecutivi la coppia torna alle regole normali anche se le sagome non si
sono mai separate. Senza questo limite due veicoli che restano a contatto dopo
un urto non potevano più generare alcun evento per il resto del video.

Inoltre, il movimento pre-impatto richiede più campioni consecutivi sopra
soglia. Un singolo salto della segmentazione non è quindi sufficiente a
dimostrare che un veicolo si stesse realmente muovendo.

### 3.5 Protezione dal traffico parallelo lento

L'avvicinamento relativo deve essere confermato su più frame consecutivi e
deve restare temporalmente vicino all'inizio dell'episodio di contatto. Una
coppia che procede affiancata con maschere sovrapposte non può quindi essere
trasformata in collisione da una variazione di velocità comparsa molti frame
dopo. L'overlap forte senza convergenza è ammesso soltanto insieme a una forte
decelerazione.

### 3.6 Memoria di traiettoria per urti contro veicoli fermi

La memoria non è stata aumentata indiscriminatamente. La finestra breve che
valida overlap, frenate e arresti resta invariata; è stata aggiunta una seconda
cronologia dedicata alla traiettoria e alla classificazione dei ruoli.

Quando un veicolo si muove verso un bersaglio stabilmente fermo, il sistema:

- conserva l'avvicinamento sostenuto osservato nello stesso episodio;
- stima la direzione pre-impatto su 12 frame;
- richiede una reazione del veicolo mobile su almeno due frame consecutivi;
- riconosce una deviazione di almeno 45 gradi rispetto alla direzione abituale;
- permette questa evidenza anche dopo un overlap prospettico lungo, senza
  rendere più longeve le singole anomalie cinematiche.

La reazione del bersaglio fermo non viene più attribuita automaticamente al
veicolo in movimento. Questo elimina i falsi eventi prodotti da auto che
svoltano davanti a veicoli fermi e continuano normalmente la marcia.

### 3.7 Attribuzione nelle traiettorie incrociate

Quando un veicolo attraversa orizzontalmente più flussi, la sua frenata non
può più confermare automaticamente ogni coppia con cui si sovrappone. Due
traiettorie sono considerate incrociate usando la direzione recente dei track;
in questo caso viene richiesta anche una perturbazione locale del bersaglio:

- deviazione rispetto alla sua traiettoria precedente; oppure
- accelerazione positiva improvvisa del veicolo inizialmente più lento,
  coerente con un trasferimento di moto durante l'urto.

Un contatto prodotto soltanto dalla dilatazione delle maschere non può inoltre
usare, nel secondo frame di conferma, una vecchia frenata se non esiste più né
avvicinamento né evidenza dinamica corrente.

### 3.8 Ponte temporale per occlusione durante un dual stop

In un urto tra traiettorie incrociate entrambi i veicoli possono frenare prima
del contatto, arrestarsi senza deviare e occultarsi a vicenda proprio durante
l'impatto. La normale finestra breve non viene estesa globalmente: viene
conservata separatamente, per un massimo di 12 frame, solo l'evidenza di un
avvicinamento incrociato già confermato.

Il ponte può confermare un evento esclusivamente quando:

- la coppia non è stata osservata per almeno due frame consecutivi;
- dopo il gap i track ricompaiono con un overlap reale, non solo dilatato;
- esiste movimento precedente verificato;
- entrambi risultano fermi in modo persistente secondo la regola `dual_stop`.

Il contesto incrociato resta attivo anche quando le velocità finali sono nulle,
ma senza un vero gap di tracking il solo arresto simultaneo viene rifiutato.
Questo limita il rischio di classificare come urto due veicoli che si fermano
normalmente per concedersi la precedenza.

## 4. Configurazione attuale

Le soglie principali sono centralizzate in `src/config.py`:

| Parametro | Valore predefinito |
| --- | ---: |
| Overlap minimo | 120 px |
| Overlap normalizzato | 0,015 |
| Distanza massima di contatto | 8 px |
| Dilatazione maschere | 3 px |
| Velocità minima pre-impatto | 3 px/frame |
| Conferma del movimento | 3 campioni |
| Velocità minima di avvicinamento | 1,5 px/frame |
| Conferma dell'avvicinamento | 3 frame |
| Validità dell'avvicinamento | 2 frame |
| Finestra temporale | 5 frame |
| Memoria della traiettoria | 12 frame |
| Ritardo escluso dalla direzione di base | 2 frame |
| Spostamento minimo della traiettoria | 12 px |
| Deviazione minima | 45 gradi |
| Memoria per direzioni incrociate | 6 frame |
| Angolo minimo di incrocio | 45 gradi |
| Accelerazione minima del bersaglio | 8 px/frame² |
| Memoria del ponte dual stop incrociato | 12 frame |
| Gap minimo per il ponte | 2 frame |
| Memoria per bersaglio fermo | 10 frame |
| Quota minima di campioni fermi | 0,75 |
| Conferma della reazione d'impatto | 2 frame |
| Età massima del contatto candidato | 5 frame |
| Conferme richieste | 2 frame |
| Cooldown della coppia | 30 frame |
| Separazione per riarmo | 3 frame |
| Età massima di un track "nuovo" | 10 frame |
| Scadenza del disarmo preesistente | 45 frame |
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

### Caso di regressione: traffico parallelo

Video: `8G56ILxFFNM_00.mp4`

Prima della correzione il sistema emetteva tre eventi tra i frame 190 e 219,
mentre i veicoli avanzavano lentamente su file parallele. Dopo l'introduzione
della convergenza sostenuta e del limite di età del contatto, gli eventi falsi
sono passati da 3 a 0 sullo stesso video.

Il dataset annota comunque un incidente `sideswipe` al frame 262, che non viene
ancora rilevato. Il caso rimane quindi utile anche come test di recall: il
filtro ha corretto i falsi positivi precedenti, ma il riconoscimento dell'evento
annotato richiede ulteriore lavoro su tracking e dinamica laterale.

### Caso di regressione: veicolo mobile contro veicolo fermo

Video: `95Tx-2p0O_E_00.mp4`

Con le regole precedenti venivano emessi due falsi positivi ai frame 159 e 174
durante normali svolte all'incrocio, mentre l'urto `rear-end` annotato al frame
194 non veniva rilevato. La diagnostica ha mostrato che la coppia reale `2-35`
risultava già in contatto prospettico dal frame 176: il limite di età del
contatto, corretto per il traffico parallelo, ne impediva la conferma tardiva.

Dopo l'aggiunta della memoria di traiettoria:

| Metrica | Prima | Dopo |
| --- | ---: | ---: |
| Veri positivi | 0 | 1 |
| Falsi positivi | 2 | 0 |
| Falsi negativi | 1 | 0 |
| Frame di conferma reale | - | 198 |
| Ritardo dall'annotazione | - | 4 frame / circa 0,25 s |

Il benchmark di non-regressione su `8G56ILxFFNM_00.mp4` mantiene 0 falsi
positivi. Il `sideswipe` annotato al frame 262 resta tuttavia non rilevato.

### Caso di regressione: veicolo trasversale su più corsie

Video: `987C4_UdnJE_01.mp4`

Il track `19` attraversa orizzontalmente più veicoli, ma urta fisicamente solo
il track `20`. Prima della correzione l'algoritmo emetteva cinque eventi e
riutilizzava la dinamica del track `19` per più coppie prospettiche.

| Metrica | Prima | Dopo |
| --- | ---: | ---: |
| Eventi totali | 5 | 1 |
| Veri positivi | 1 | 1 |
| Falsi positivi | 4 | 0 |
| Falsi negativi | 0 | 0 |
| Coppia attribuita | errata/ambigua | `19-20` |
| Frame di conferma | 30 | 36 |
| Scarto dall'annotazione | -9 frame | -3 frame / circa -0,12 s |

I benchmark di non-regressione mantengono `95Tx-2p0O_E_00.mp4` a 1 TP e 0 FP
e `8G56ILxFFNM_00.mp4` a 0 FP; il `sideswipe` di quest'ultimo resta non
rilevato.

### Caso di regressione: arresto simultaneo dopo occlusione

Video: `D1eP4Bn4hDQ_0_00.mp4`

Il camion dei pompieri (track `21`) e l'auto (track `22`) frenano prima
dell'urto, non cambiano traiettoria e infine si arrestano. Il track `21`
scompare tra i frame 117 e 121 e la coppia ricompare con contatto reale: la
finestra d'impatto ordinaria perdeva quindi l'avvicinamento incrociato osservato
fino al frame 116.

| Metrica | Prima | Dopo |
| --- | ---: | ---: |
| Veri positivi | 0 | 1 |
| Falsi positivi | 0 | 0 |
| Falsi negativi | 1 | 0 |
| Coppia attribuita | - | `21-22` |
| Frame di conferma | - | 126 |
| Ritardo dall'annotazione | - | 5 frame / circa 0,21 s |

L'evento viene spiegato come `stop_transition`, `dual_stop` e
`occlusion_bridge`. I benchmark di non-regressione restano invariati:
`987C4_UdnJE_01.mp4` e `95Tx-2p0O_E_00.mp4` mantengono 1 TP e 0 FP;
`8G56ILxFFNM_00.mp4` mantiene 0 FP e il falso negativo `sideswipe` già noto.

## 8. Test automatici

La suite comprende test per:

- geometria delle maschere e contatto dilatato;
- velocità, accelerazione e filtro EMA;
- gestione dei track, gap e ID duplicati;
- conferma temporale, cooldown e rifiuto dei veicoli già fermi;
- contatto preesistente delle auto in coda e riarmo dopo la separazione;
- rifiuto dei picchi isolati di movimento dovuti al jitter;
- traffico lento su file parallele e anomalie tardive durante overlap lunghi;
- conservazione di un evento con convergenza sostenuta vicino al contatto;
- rifiuto di una frenata isolata durante una svolta davanti a un veicolo fermo;
- rilevamento tardivo di un urto tramite deviazione della traiettoria;
- attribuzione dell'urto tra più traiettorie incrociate;
- conferma del dual stop incrociato dopo un breve gap di tracking;
- rifiuto del dual stop incrociato quando il gap di tracking non esiste;
- rifiuto di un contatto solo dilatato con evidenza dinamica scaduta;
- caricamento di `metadata-real.csv`;
- matching per timestamp e frame;
- aggregazione delle metriche per gruppo;
- selezione dei campioni del benchmark.

Comando verificato:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Risultato corrente: `52 passed`.

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

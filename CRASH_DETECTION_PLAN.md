# Piano di miglioramento del crash detection

Data: 29 agosto 2026 · Ultimo aggiornamento: 12 settembre 2026 (punti 2b e 3) ·
Branch: `main_v2` · Modello di riferimento: `yolo26n-seg.pt`

Questo documento raccoglie le modifiche individuate analizzando il codice e
misurando il comportamento reale della pipeline sul dataset annotato. Ogni voce
riporta il problema, l'evidenza numerica che lo dimostra, l'intervento proposto
e il criterio di verifica. Le voci sono ordinate per impatto misurato, non per
comodità implementativa.

---

## 1. Baseline misurata

Campione stratificato: **10 video per tipo** (`rear-end`, `t-bone`, `sideswipe`,
`head-on`, `single`) presi dal test set `split_in_distribution`, per un totale di
**50 video, 19.981 frame, 20,5 minuti** di filmato.

| Metrica | Valore |
| --- | ---: |
| Throughput CPU | 13,0 FPS — 0,78× realtime |
| True positives | 1 |
| False positives | 2 |
| False negatives | 49 |
| Precision / Recall / F1 | 0,33 / **0,02** / 0,04 |
| Eventi emessi | 3 su 130.854 righe coppia×frame |

Proiezione: una passata sul dataset completo (765.043 frame) costa **≈16,3 ore**
su questa macchina.

### Il funnel

Ricostruito dai log diagnostici, contando i video in cui almeno una riga nella
finestra ±1 s attorno all'incidente annotato supera ciascuno stadio:

| Stadio | Prima | Dopo il punto 1 |
| --- | ---: | ---: |
| Video annotati | 50 | 50 |
| ≥2 track nella finestra | 33 | 33 |
| Contatto spaziale | 22 | 22 |
| Non disarmato da `preexisting_contact` | 14 | **22** |
| Moto precedente (`had_recent_motion`) | 14 | 18 |
| Evidenza dinamica | 8 | 12 |
| `sustained_approach` | 6 | 6 |
| Contatto fresco (`age ≤ 5`) | 3 | 3 |
| **Evento emesso** | 1 | 1 |

Il tetto di percezione è **33/50 = 66%**: in 17 video il tracker non consegna mai
due track nella finestra dell'incidente, quindi nessuna soglia di collisione può
recuperarli.

---

## 2. Interventi

### ✅ 1 — `preexisting_contact` non deve essere permanente né armarsi sugli ID switch

**Stato: FATTO** (56 test verdi, effetto misurato)

**Problema.** Una coppia già in contatto alla prima osservazione veniva
disarmata, e il riarmo richiedeva una separazione stabile che dopo un urto reale
non arriva mai. Il flag era pensato per le auto in coda a inizio video, ma
colpiva selettivamente proprio le coppie che collidono.

**Evidenza.** Il disarmo bloccava **8 video su 22** che raggiungevano il
contatto (36%). In **tutti e 8** un track aveva `observed_frames == 1` e l'altro
fra 16 e 149: non erano veicoli già accostati, era ByteTrack che riassegnava un
ID durante l'occlusione dell'urto e la coppia rinasceva già in contatto.

**Modifica applicata.**

- *Arming ristretto* — `preexisting_contact` scatta solo se **entrambi** i track
  sono appena comparsi (`preexisting_contact_max_track_age_frames = 10`, helper
  `_both_tracks_are_new`). La protezione originale resta intatta: a inizio video
  tutti i track nascono insieme.
- *Disarmo a scadenza* — dopo `preexisting_contact_max_frames = 45` frame la
  coppia torna alle regole normali anche senza separazione
  (`_preexisting_contact_is_released`).

File toccati: `src/config.py`, `src/collision_logic.py`,
`tests/test_collision_logic.py` (3 nuovi test), `README.md`,
`PROJECT_STATUS.md`.

**Effetto misurato.**

| | Prima | Dopo |
| --- | ---: | ---: |
| Righe `preexisting_contact` | 4.170 | **911** (−78%) |
| Video bloccati dal disarmo | 8 | **0** |
| TP / FP / FN | 1 / 2 / 49 | 1 / **3** / 49 |
| Recall | 0,02 | 0,02 |

**Lettura onesta.** Il recall non si è mosso e la precision è scesa da 0,33 a
0,25. Il valore dell'intervento è che gli 8 video sono passati da
*strutturalmente impossibili* — la coppia non poteva emettere nemmeno in linea di
principio — a *bloccati da una soglia*, che è calibrabile. Dei 8 sbloccati, 4
cadono ora su `had_recent_motion` e 4 su `sustained_approach`: esattamente le
soglie del punto 2.

La soglia dei 10 frame non è delicata: fra `1` e `16` c'è un vuoto ampio nei
dati, qualsiasi valore fra 3 e 15 dà lo stesso risultato su questo campione.

---

### 🟡 2 — Normalizzare le soglie per scala apparente e fps

**Stato: IMPLEMENTATO E MISURATO — disattivato di default perche peggiora**

**Problema di partenza.** Tutte le soglie cinematiche sono in pixel assoluti e
tutte le finestre in frame, mentre l'fps del dataset va da 3,9 a 49,7 e la scala
apparente dei veicoli da 39 a 462 px di diagonale.

**Cosa e stato implementato.** Un fattore correttivo applicato a ogni confronto,
continuo con la taratura attuale (a condizioni di riferimento vale 1):

- velocita: `base × (scala / 90 px) × (15 / fps)`
- accelerazione: `base × (scala / 90 px) × (15 / fps)²`
- distanze: `base × (scala / 90 px)`; aree: `× (scala / 90 px)²`
- finestre in frame: `base × (fps / 15)`

La scala apparente e la diagonale della bbox filtrata EMA, salvata in
`VehicleState.scale_px` e in ogni `TrackSample`. `video_fps` arriva dal decoder
ed e collegato in `benchmark.py`, `app.py` e `ui_main.py`.

File: `src/config.py`, `src/models.py`, `src/kinematics.py`
(`compute_bbox_scale_px`), `src/tracker_state.py`, `src/collision_logic.py`
(~30 siti di confronto), `src/app.py`, `src/benchmark.py`, `src/ui_main.py`,
`tests/test_collision_logic.py` (4 nuovi test).

**Risultato misurato — ed e negativo.**

| | punto 1 | + normalizzazione |
| --- | ---: | ---: |
| TP / FP / FN | 1 / 3 / 49 | **0 / 5 / 50** |
| Stadio *dinamica* del funnel | 12 | **8** |
| Stadio *approach* | 6 | 3 |
| Recall | 0,02 | **0,00** |

**Perche fallisce.** Due cause distinte, entrambe misurate:

1. *La scala va nella direzione sbagliata.* Le coppie che collidono hanno scala
   apparente **sopra** la mediana del dataset (128 px contro 90): sono piu
   vicine alla camera della media. Normalizzare su una mediana globale gli
   **alza** le soglie del 40% invece di abbassarle. Verificato direttamente: la
   quota di contatti che supera `min_closing_speed_px` passa da 10% a 9%.
2. *La dilatazione delle finestre e la meta piu dannosa.* A 30 fps
   `stopped_frames_threshold` passa da 3 a 6 e `motion_confirmation_frames` da 3
   a 6. E fisicamente corretto, ma i track coinvolti in un urto vivono spesso
   pochi frame perche il tracker riassegna gli ID durante l'occlusione: chiedere
   il doppio delle conferme a chi ha meno storia disponibile azzera
   `dual_stop` e `stop_transition`, ed e questo che fa crollare lo stadio
   *dinamica* da 12 a 8.

**Stato del codice.** L'infrastruttura resta in repo, sotto due flag separati,
entrambi `False`:

- `kinematic_normalization_enabled` — magnitudini. La configurazione con le sole
  magnitudini (finestre non dilatate) **non e ancora stata misurata**: il run e
  stato interrotto a 20 video su 50.
- `normalize_time_windows` — finestre. Misurata dannosa, da lasciare spenta.

**Cosa fare prima di riattivarla.** La normalizzazione per scala ha senso solo
se il riferimento non e una mediana globale ma qualcosa di locale alla coppia:
per esempio normalizzare la velocita di avvicinamento sulla scala **della coppia
stessa** senza riferirla a una costante di dataset, cioe ragionare in
"lunghezze-veicolo al secondo" con soglie ritarate da zero su quella unita,
invece di correggere soglie tarate in pixel. Questo pero richiede di ritarare i
valori base, non di moltiplicarli per un fattore.

---

#### 2b — Il vero collo di bottiglia scoperto misurando: le coppie senza storia

**Stato: MISURATO — (b) attivo di default, (a) disattivato perche inefficace**

Questa e la scoperta piu importante emersa dal punto 2, e non era nel piano
iniziale.

**Evidenza.** Nei 6 video bloccati su `sustained_approach`, in **5 casi la
sequenza diagnostica della coppia inizia esattamente al frame del primo
contatto**: i due track non sono mai stati attivi insieme prima dell'urto. Uno
dei due nasce nell'istante dell'impatto.

Conseguenze a cascata, tutte verificate nei log:

- `approach_confirmation_frames = 3` non e soddisfacibile su una coppia che
  esiste da 1–3 frame. In `afvzz3v87cc` il closing speed vale **16,66 px/f** a un
  singolo frame — evidenza schiacciante — e viene scartato perche non si ripete.
- `closing_speed` vale **0,00 per costruzione** quando un track e appena nato,
  perche `_closing_speed` esce a zero se `kinematics_valid` e False.
- Ogni gate basato sulla storia (`had_recent_motion`, `_is_stably_stationary`,
  `_recent_trajectory_vector`) fallisce per lo stesso motivo.

**Evidenza aggiuntiva raccolta implementando.** Il campione ristretto descritto
sotto conferma la diagnosi su dati nuovi: il **12%** delle righe di contatto
nella finestra dell'incidente (22 su 182) appartiene a coppie osservate da 3
frame o meno. In `OXVXsdDT9QQ_00` la coppia reale `4-10` diventa candidata al
frame 29 e perde la candidatura al frame 30 senza mai emettere, pur avendo
mostrato closing speed fino a 14,7 px/f pochi frame prima.

Una precisazione sulla diagnosi originale del punto 2b, verificata sui log e
risultata piu stretta di come era stata scritta. `closing_speed` vale zero *per
costruzione* solo nel frame in cui il track nasce, perche `kinematics_valid` e
ancora False. Gia dal frame successivo la velocita e misurabile: in
`OXVXsdDT9QQ_00` ai frame 29-30 entrambi i track hanno cinematica valida e il
closing speed e zero perche i due veicoli si stanno realmente **allontanando**
dopo l'urto, non perche manchi la storia. Il gate che manca alla coppia giovane
e quindi `approach_confirmation_frames` — la *persistenza* — non la
misurabilita della velocita. E esattamente quello che il rimedio (b) attacca.

**Cosa e stato implementato.** Due rimedi indipendenti, sotto flag separati per
poterne attribuire l'effetto separatamente — la lezione del punto 2.

- *(a) Re-identificazione con eredita della storia* (`track_reid_enabled`).
  Quando un ID nuovo compare dove un altro e appena sparito, il nuovo track
  prende il posto del vecchio nello store invece di nascere da zero. Il
  candidato donatore deve essere assente in questo frame, sparito da al massimo
  `track_reid_max_gap_frames` (5), trovarsi entro
  `track_reid_max_distance_scale` (0,8) diagonali di bbox dalla posizione
  estrapolata col suo moto, e avere scala compatibile entro un fattore
  `track_reid_max_scale_ratio` (2,0). Lo stato viene *spostato*, non copiato:
  il frame successivo passa dal normale `_update_existing_state`, quindi
  velocita e accelerazione sono calcolate sullo spostamento reale attraverso il
  cambio di ID invece che ricostruite a mano. Attacca la radice: rimette in
  funzione `had_recent_motion`, `_closing_speed`, `_is_stably_stationary` e la
  memoria di traiettoria tutti insieme.
- *(b) Avvicinamento impulsivo* (`impulse_approach_enabled`). Su una coppia
  osservata da al massimo `young_pair_max_frames` (3) frame, un singolo frame
  con closing speed almeno `impulse_approach_speed_multiplier` (4,0) volte la
  soglia vale come avvicinamento sostenuto. E la scorciatoia minima:
  `approach_confirmation_frames = 3` non e soddisfacibile da una coppia che
  esiste da meno frame di quanti ne richiede. Sostituisce solo quel gate:
  contatto, moto precedente ed evidenza dinamica restano richiesti.

File: `src/config.py`, `src/models.py` (due campi diagnostici nuovi),
`src/tracker_state.py` (`_find_reid_donor`, `_inherit_track_history`),
`src/collision_logic.py` (`_is_impulse_approach`, `_PairState.observed_frames`),
`src/benchmark.py` (`--set`), `tests/test_tracker_state.py` (5 test nuovi),
`tests/test_collision_logic.py` (3), `tests/test_benchmark.py` (2).

**Osservabilita.** `PairDiagnostic` espone ora `pair_observed_frames` e
`impulse_approach`. Le re-identificazioni non hanno bisogno di un campo
dedicato: un track che compare nei log gia con `observed_frames > 1` ha per
definizione ereditato una storia, quindi si contano dai record `track`.

**Risultato misurato.** Matrice `base` / `reid` / `imp` sul campione di §3.1.
I due meccanismi danno esiti opposti.

| Stadio del funnel — campione (15 video) | base | reid | imp |
| --- | ---: | ---: | ---: |
| ≥2 track nella finestra | 7 | 7 | 7 |
| Contatto spaziale | 4 | 4 | 4 |
| Moto precedente | 3 | 3 | 3 |
| Evidenza dinamica | 3 | 3 | 3 |
| `sustained_approach` | 2 | 2 | **3** |
| Contatto fresco (`age ≤ 5`) | 0 | 0 | **1** |
| **Evento emesso** | 0 | 0 | **1** |
| TP / FP / FN | 0 / 0 / 15 | 0 / 0 / 15 | **1** / 0 / 14 |

Sui 4 video di regressione le tre configurazioni danno **3 TP / 2 FP / 1 FN**,
con gli stessi eventi sulle stesse coppie negli stessi frame.

La quarta configurazione, `both`, e stata eseguita come controllo e coincide con
`imp` su entrambi i set: 1 TP / 0 FP sul campione, 3 TP / 2 FP sulla
regressione. I due meccanismi non interagiscono — prevedibile, visto che uno dei
due non fa nulla, ma verificarlo costa un run e chiude la questione.

**Conferma su campione doppio.** Poiche un solo TP su 15 video non basta a
cambiare un default, il campione e stato portato a **6 video per tipo (30
video)** per le due sole configurazioni fra cui decidere. `--resume` ha reso
gratuiti i 15 gia calcolati.

| Campione esteso (30 video) | base | imp |
| --- | ---: | ---: |
| `sustained_approach` | 5 | **6** |
| Contatto fresco | 2 | **3** |
| Evento emesso | 1 | **2** |
| TP / FP / FN | 1 / 2 / 29 | **2** / 2 / 28 |
| Precision / Recall | 0,33 / 0,03 | **0,50** / **0,07** |

Gli eventi, uno per uno:

| | video | tipo | coppia | scarto | base | imp |
| --- | --- | --- | --- | ---: | :-: | :-: |
| TP | `6hJ8mYLUfE8_00` | head-on | `1-16` | −0,03 s | ✓ | ✓ |
| TP | `OXVXsdDT9QQ_00` | t-bone | `4-10` | +0,60 s | — | **✓** |
| FP | `gp_9zTMpVrs_00` | single | `1-4` | −1,58 s | ✓ | ✓ |
| FP | `M6WWLqx3z88_00` | sideswipe | `15-21` | +3,97 s | ✓ | ✓ |

I due falsi positivi sono **identici** nelle due configurazioni, stessi frame e
stesse coppie: nessuno dei due nasce dall'avvicinamento impulsivo. Uno dei due
e per giunta su un incidente `single`, dove una logica a coppie non puo avere
ragione per costruzione — e il punto 7, non un errore di soglia.

**(b) avvicinamento impulsivo: funziona, e non costa.** Converte
`OXVXsdDT9QQ_00` (t-bone notturno) da *candidato che non si conferma* a evento
corretto: coppia `4-10`, la stessa che nella baseline restava bloccata,
confermata a t = 1,60 s contro un'annotazione a t = 1,00 s, quindi dentro la
tolleranza ma con poco margine. Il closing speed di picco e 14,67 px/f contro
una soglia di 1,5.

**Decisione.** `impulse_approach_enabled` passa a `True` di default; la sua
evidenza e **2 TP su 30 video** contro 1, ed e sottile: un miglioramento reale
ma piccolo, su un campione dove il sistema emette pochissimo. Va rimisurato
quando il tetto di percezione (punto 6) si alza e i video che arrivano al
contatto diventano molti di piu. `track_reid_enabled` resta `False`.

Il costo misurato e **zero**: nessun falso positivo nuovo, ne sul campione ne
sui casi di regressione, dove pure il meccanismo si attiva 52 volte senza
cambiare un solo evento.

Vale pero la pena guardare *perche* non costa, perche il conteggio degli eventi
da solo lo nasconderebbe. Su tutte le 159 righe in cui l'avvicinamento
impulsivo scatta nei due set:

| Esito della riga impulsiva | righe | quota |
| --- | ---: | ---: |
| `status=clear` — la coppia non diventa mai candidata | 154 | 97% |
| Ha moto precedente | 53 | 33% |
| Ha evidenza dinamica | 23 | 14% |
| Ha contatto spaziale | 11 | **7%** |
| Diventa `contact_candidate` | 3 | 2% |
| Emette | 2 | 1% |

La condizione impulsiva in se **non e selettiva**: scatta ogni volta che un
track nuovo compare vicino a un veicolo in moto. A filtrare e la congiunzione —
il 93% delle righe muore sul requisito di contatto. Il meccanismo e quindi
sicuro *in prestito*: la sua precisione e interamente a carico degli altri gate.
Se un intervento futuro indebolisse il requisito di contatto — per esempio il
punto 4, che allenta l'eta del contatto candidato — questa scorciatoia andrebbe
rimisurata insieme a quello, non data per acquisita.

**(a) re-identificazione: nessun effetto, e la misura dice perche.** Scatta 16
volte su 196 track nel campione e 28 su 136 nei casi di regressione, ma non
sposta **nessuno** stadio del funnel e non cambia **nessun** evento: gli ID
adottati sono sempre track periferici, mai quelli delle coppie che collidono.

Ricostruendo dai log `track` della baseline ogni nascita di ID e cercandone il
donatore migliore:

| | tutte le nascite | nella finestra |
| --- | ---: | ---: |
| Nascite | 196 | 28 |
| Senza nessun donatore entro 5 frame | **115 (59%)** | 11 (39%) |
| Miglior donatore entro 0,4 diagonali | 15% | 6% |
| … entro 0,8 (soglia attuale) | 21% | 24% |
| … entro 1,5 | 31% | 29% |
| … entro 3,0 | 68% | 71% |
| Mediana del miglior donatore | **2,19 diagonali** | 2,02 |

Tre conclusioni, tutte misurate:

1. *Per il 59% delle nascite la re-identificazione e impossibile per
   costruzione*: nessun track e sparito nei 5 frame precedenti. Non sono ID
   riassegnati, sono veicoli che entrano in scena.
2. *Non esiste una soglia che separi.* Il punto 1 aveva un vuoto ampio fra 1 e
   16 frame che rendeva la soglia indifferente; qui la distribuzione e un
   continuo (15% → 21% → 31% → 68%). Allargare la soglia comprerebbe copertura
   tirando a indovinare a due lunghezze di veicolo di distanza.
3. *L'estrapolazione col vettore velocita non serve.* Usando semplicemente
   l'ultima posizione nota la mediana e 2,21 diagonali invece di 2,19 e la quota
   entro 0,8 e identica (21%). La complessita non si ripaga.

L'associazione su sola posizione e scala non e separabile su questi dati.
Una re-identificazione utile richiede un descrittore d'aspetto — istogramma di
colore della maschera, o un embedding — che e un intervento di categoria
diversa e va valutato come tale, non come taratura di questa soglia.

---

### ✅ 3 — Delta-V vettoriale invece della derivata del modulo

**Stato: MISURATO — evidenza generica attiva, attribuzione incrociata dannosa**

**Problema.** `compute_acceleration` è la differenza finita della *magnitudine*
della velocità. Un t-bone che devia un veicolo di 90° senza cambiarne il modulo
produce accelerazione ≈ 0 e non attiva nessuna evidenza dinamica. Inoltre l'EMA
con α = 0,45 smorza proprio il transitorio dell'impatto, rendendo
`strong_deceleration_threshold = -4.0` difficile da raggiungere.

**Evidenza.** *Evidenza dinamica assente* blocca **6 video su 18** che arrivano
allo stadio precedente. I `t-bone` sono 657 su 2027 nel dataset. Sul campione
esteso di §3.1 lo stesso stadio costa **3 video su 10** (`had_motion` 10 →
`dynamic` 7), quindi il collo di bottiglia e confermato su dati nuovi.

**Evidenza quantitativa raccolta prima di scrivere il codice.** Ricostruendo il
delta-V dai log gia disponibili — velocita filtrate EMA, quindi una stima *per
difetto* del transitorio reale — su 42.664 campioni, di cui 4.404 nella finestra
dell'incidente:

| | tutti | nella finestra | arricchimento |
| --- | ---: | ---: | ---: |
| `|Δv|` mediano | 0,31 | 0,55 | |
| `|Δv|` p99 | 7,68 | 12,88 | |
| Quota con `|Δv| ≥ 8` | 0,93% | 2,23% | 2,4× |
| Quota con `|Δv| ≥ 14` | 0,31% | 0,86% | 2,8× |

Il numero che giustifica l'intervento e pero un altro. Nella finestra
dell'incidente la decelerazione scalare `≤ −4,0` scatta su **66 campioni su
4.404** (1,5%); `|Δv| ≥ 8` ne prende **98, di cui 72 (73%) con accelerazione
scalare sopra la soglia**, cioe completamente invisibili alla grandezza usata
oggi. Il delta-V non raddoppia semplicemente l'evidenza disponibile: quasi tre
quarti di cio che aggiunge e informazione nuova, ed e esattamente la firma
prevista — deviazioni a modulo costante.

**Cosa e stato implementato.**

1. `compute_delta_v_px` in `src/kinematics.py`: `|Δv⃗| = |v⃗ₜ − v⃗ₜ₋₁|`,
   normalizzato per i frame trascorsi. Non ha segno, perche e un modulo: non
   distingue una frenata da un'accelerazione, distingue un moto perturbato da
   uno regolare.
2. **Due velocita separate** in `VehicleState`: la filtrata EMA continua a
   stimare traiettoria e direzione, la grezza alimenta il delta-V. Leggere il
   delta-V dal dato filtrato vanificherebbe l'intervento, perche l'EMA e tarato
   proprio per smorzare i picchi.
3. Il delta-V richiede **due frame entrambi misurati**. Senza questa guardia il
   primo campione di ogni track varrebbe l'intera velocita e ogni veicolo appena
   apparso sembrerebbe appena urtato — e il punto 2b ha mostrato che quella
   popolazione e numerosa.
4. Evidenza dinamica **accanto** alla decelerazione, mai al suo posto, sotto due
   flag distinti: `delta_v_evidence_enabled` per l'evidenza generica e per la
   reazione del veicolo in moto contro bersaglio fermo;
   `delta_v_crossing_disruption_enabled` per l'attribuzione fra traiettorie
   incrociate. Il secondo e l'uso piu promettente — un bersaglio colpito di lato
   ha delta-V alto ma spesso nessuna `trajectory_deflection`, che pretende
   velocita e spostamento di base minimi — ma tocca la macchina anti-falsi-
   positivi del punto 3.7, quindi va attribuito separatamente.
5. Il delta-V e registrato nei log **anche a gate spento**, nei record `track` e
   nella diagnostica di coppia: la soglia va tarata sulla distribuzione reale,
   non a occhio.

File: `src/kinematics.py`, `src/models.py`, `src/tracker_state.py`,
`src/collision_logic.py`, `src/config.py`, `src/calibration.py`,
`tests/test_kinematics.py` (3 nuovi), `tests/test_tracker_state.py` (3),
`tests/test_collision_logic.py` (4).

**Verifica di non-regressione.** Una passata a gate spento su 10 video produce
un comportamento **identico riga per riga** alla configurazione corrente: stessa
sequenza di stati di coppia, stessi eventi, 10 su 10 video. Il calcolo del
delta-V e la separazione delle due velocita non cambiano nessuna decisione
finche i flag restano `False`.

Il primo confronto era stato fatto contro i log della baseline pre-punto-2b e
segnalava tre video diversi su `sustained_approach`: non era una regressione ma
l'avvicinamento impulsivo, acceso di default dal punto 2b. Vale la pena
registrarlo perche e il modo tipico in cui un controllo di non-regressione
mente: confrontando con una baseline che nel frattempo non e piu la baseline.

**Taratura della soglia sul delta-V grezzo.** La distribuzione misurata sui log
nuovi (3.940 campioni, 445 nella finestra dell'incidente) e diversa da quella
stimata sui log filtrati, come previsto: mediana 1,00 contro 0,31, p99 14,58
contro 7,68.

| Soglia | quota su tutti | quota nella finestra | arricchimento | nuovi rispetto allo scalare |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 5,41% | 11,46% | 2,1× | 47 su 51 |
| 8 | 2,89% | 7,19% | 2,5× | 28 su 32 |
| **10** | **1,95%** | **6,74%** | **3,4×** | **26 su 30** |
| 12 | 1,55% | 5,39% | 3,5× | 20 su 24 |
| 15 | 0,94% | 3,37% | 3,6× | 12 su 15 |
| 20 | 0,53% | 1,80% | 3,4× | 5 su 8 |

La soglia scelta e **10,0 px/frame²**, e non e arbitraria: fra 8 e 10
l'arricchimento passa da 2,5× a 3,4× e da li in poi si appiattisce attorno a
3,5×. C'e un ginocchio, quindi sotto i 10 si comprano soprattutto campioni di
fondo. Sopra i 15 si perde meta del segnale senza guadagnare selettivita.

Per confronto, nella stessa finestra la decelerazione scalare `≤ −4,0` scatta su
**4 campioni su 445** (0,90%). A soglia 10 il delta-V ne intercetta 30, di cui
**26 con accelerazione scalare sopra soglia**: l'87% di quello che aggiunge e
evidenza che oggi non esiste affatto.

**Risultato misurato — `delta_v_evidence_enabled` a soglia 10.**

| Campione (30 video) | attuale | + delta-V |
| --- | ---: | ---: |
| Evidenza dinamica | 7 | **8** |
| `sustained_approach` | 6 | 6 |
| Contatto fresco | 3 | **4** |
| Coppia candidata | 3 | **5** |
| Evento emesso | 2 | **3** |
| TP / FP / FN | 2 / 2 / 28 | **3** / 2 / 27 |
| Precision / Recall | 0,50 / 0,07 | **0,60** / **0,10** |

Sui 4 casi di regressione: **3 TP / 2 FP / 1 FN**, invariati.

**Il caso recuperato e esattamente quello previsto.** `fzWY0vLAXzI_00` e un
**t-bone**, la coppia attribuita e `1-3`, lo scarto −0,67 s, e la motivazione
dell'evento e `temporal_contact_and_delta_v_impact`: il delta-V e l'**unica**
evidenza dinamica presente. Nella configurazione precedente quel video si
fermava allo stadio *approach*. Anche il TP gia esistente `OXVXsdDT9QQ_00` e un
t-bone e guadagna `delta_v_impact` fra le motivazioni. Entrambi i veri positivi
recuperati fra punto 2b e punto 3 sono t-bone, cioe la classe che il piano
indicava come cieca per costruzione alla derivata del modulo: 657 video su
2.027.

**Il rovescio della medaglia, misurato.** Il delta-V non scatta solo sugli urti:
si attiva anche sui due falsi positivi gia presenti e **ne alza la confidenza**.

| Falso positivo | confidenza prima | dopo | `delta_v_px` |
| --- | ---: | ---: | ---: |
| `M6WWLqx3z88_00`, coppia `15-21` | 0,75 | **0,90** | 32,0 |
| `8G56ILxFFNM_00`, coppia `2-54` | 1,00 | 1,00 | **100,0** |

Nessun falso positivo nuovo, ma quelli esistenti diventano *meno* distinguibili:
e una cattiva notizia specifica per il **punto 5**, perche una soglia sul
punteggio di confidenza non separerebbe questi casi, e il delta-V peggiora la
situazione invece di migliorarla.

Il valore di 100 px/frame² merita attenzione a parte: e implausibile per un
veicolo reale a queste scale — il massimo osservato sull'intera distribuzione
grezza e 87 — ed e quasi certamente un salto della maschera o una confusione di
ID, non fisica. Suggerisce che il delta-V vada usato con un **limite
superiore** oltre il quale il campione e da trattare come artefatto di
segmentazione, non come urto piu violento. Da misurare come intervento a se.

**Risultato misurato — `delta_v_crossing_disruption_enabled`: dannoso.**

Sul campione sembra il migliore dei due: recupera un terzo t-bone
(`ftk_fSc4haM_00`, coppia `29-30`, scarto −0,40 s) e porta a **4 TP / 2 FP**,
precision 0,67. Sui casi di regressione pero:

| Set di regressione | attuale | + delta-V | + delta-V incrociato |
| --- | ---: | ---: | ---: |
| TP | 3 | 3 | 3 |
| FP | 2 | 2 | **5** |
| Precision | 0,60 | 0,60 | **0,38** |

`987C4_UdnJE_01` torna a emettere **cinque eventi** — coppie `10-19`, `9-19`,
`13-19`, `19-20`, `1-19` — cioe il track 19 accoppiato con tutto cio che
attraversa visivamente. E esattamente il comportamento che la scheda 3.7 di
`PROJECT_STATUS.md` documenta come *precedente* alla correzione.

La causa e strutturale e invalida l'idea **nella forma implementata**, non la
taratura: il delta-V di un veicolo che attraversa piu flussi e alto su ogni
coppia prospettica che incontra. La perturbazione smette quindi di essere
*locale* e torna a essere condivisa, che e precisamente il difetto che il gate
del punto 3.7 esisteva per correggere. Nessuna soglia risolve, perche il
problema non e l'intensita ma l'attribuzione: **il delta-V appartiene a un
track, non a una coppia**.

*Prima ipotesi di rimedio, verificata e scartata.* Si era proposto di usare la
coincidenza temporale fra il picco di delta-V e il primo contatto della coppia.
Misurata sui log, non separa: in `987C4_UdnJE_01` tutte e cinque le coppie
hanno il picco entro ±3 frame dal proprio primo contatto, comprese le quattro
sbagliate.

*Seconda ipotesi, sostenuta dai dati.* Misurare il delta-V **dentro l'episodio
di contatto di quella specifica coppia** invece che sull'intera vita del track,
e usarlo in modo **competitivo** fra le coppie che condividono un track: solo la
coppia con il picco piu alto puo usarlo come perturbazione. Su questa statistica
i due casi si ordinano correttamente:

| Video | coppia corretta | picco nell'episodio | miglior concorrente |
| --- | --- | ---: | ---: |
| `987C4_UdnJE_01` | `19-20` | **79,8** | 60,0 (`1-19`) |
| `7Nc9tJ7R4KY_00` | `9-28` | **24,3** | 3,6 (`9-22`) |

Una soglia assoluta continuerebbe a non funzionare — a 10 passerebbero ancora
quattro coppie di `987C4` — mentre l'argmax seleziona quella giusta in entrambi
i video.

*Implementata e misurata: risolve meta del problema.* Il ciclo di
`detect_collisions` e stato diviso in due fasi — misura delle evidenze, poi
decisione — perche il confronto fra coppie e impossibile in un ciclo
per-coppia. A flag spento il comportamento resta identico riga per riga
(3 video su 3 verificati). Con il flag acceso:

| | esito |
| --- | --- |
| `7Nc9tJ7R4KY_00` | **risolto**: coppia `9-28`, frame 83, scarto **−0,00 s** |
| `987C4_UdnJE_01` | **non risolto**: sempre 5 eventi |
| Totale sui 5 video | 4 TP / **5 FP** / 1 FN, precision 0,44 |

**Perche fallisce ancora, ed e una causa diversa dalla precedente.** Le cinque
coppie di `987C4` non sono in contatto negli stessi frame: il veicolo
trasversale le spazza una dopo l'altra.

| coppia | frame di contatto |
| --- | --- |
| `10-19` | 28–29 |
| `9-19` | 33 |
| `13-19` | 34–38 |
| `19-20` | **36** (la corretta) |
| `1-19` | 44–46 |

La competizione e per-frame, quindi quasi ogni coppia e **sola** nel proprio
istante e vince per definizione. La clausola "senza concorrenti vince" — che su
`7Nc9tJ7R4KY` era corretta, perche li la coppia rivale era in contatto
simultaneo — e esattamente la scappatoia.

**Dove va risolto davvero.** Estendere la competizione dall'istante
all'episodio richiederebbe di rimandare l'emissione fino a quando l'episodio e
finito, cioe attendere il frame 46 per decidere sul frame 28. Non e una
correzione del gate per-coppia: e un **NMS temporale di scena**, cioe il
**punto 8B**, dove un unico urto fisico che genera eventi su piu coppie
adiacenti viene collassato in uno solo. Il cooldown oggi e per-coppia, non
globale, ed e questo il limite. Il delta-V competitivo e il mattone giusto, ma
ha bisogno di quella infrastruttura per funzionare.

**Stato del codice.** `delta_v_crossing_disruption_enabled` resta `False`. La
versione competitiva sostituisce quella a soglia assoluta: e strettamente
migliore — risolve un caso invece di nessuno — e non aggiunge falsi positivi
propri, ma non basta da sola.

**Decisione.** `delta_v_evidence_enabled` passa a `True` con soglia 10,0;
`delta_v_crossing_disruption_enabled` resta `False`. Il codice del secondo resta
in repo: il difetto e nell'uso come prova di attribuzione, non nella grandezza.

---

### ⬜ 4 — Rilassare `max_contact_candidate_age_frames`

**Problema.** `first_contact_frame` non si azzera finché il contatto resta
"recente", quindi un overlap prospettico persistente (un sorpasso, un veicolo
che ne occlude un altro) invecchia la coppia oltre i 5 frame e la esclude, salvo
le tre eccezioni cablate (`late_stationary_impact`, i due `bridge`).

**Evidenza.** Blocca **3 video su 6** che arrivano allo stadio precedente. Il
**15%** delle righe di contatto nella finestra ha `contact_age > 5`.

**Intervento.** Legare l'età non all'inizio del contatto ma all'inizio
dell'*episodio di avvicinamento*, oppure resettare l'episodio quando compare una
nuova evidenza dinamica forte. Da tarare insieme al punto 2, perché la finestra
è in frame e soffre dello stesso problema di fps.

---

### ⬜ 5 — Esporre una soglia sul `confidence` dell'evento

**Problema.** `CollisionEvent.confidence` è calcolato in `_build_event` ma non è
mai usato come soglia. Il sistema produce **un solo punto operativo**: non si può
tracciare una curva precision/recall né scegliere il compromesso.

**Evidenza.** 2 eventi fuori finestra su 20,5 minuti ≈ **5,8/ora**. Il punto
operativo attuale è estremamente conservativo e c'è ampio margine per scambiare
precisione con recall — margine che oggi non è nemmeno esplorabile.

**Intervento.** Aggiungere `min_event_confidence` a `AppConfig` e filtrare in
`detect_collisions`. Costo: poche righe. Rende possibile disegnare la curva PR
sul dataset rieseguendo solo `src.calibration` sui log già prodotti.

---

### ⬜ 6 — Recuperare il tetto di percezione

**Problema.** In 17 video su 50 (34%) il tracker non consegna mai due track nella
finestra dell'incidente. Nessuna modifica alla logica di collisione può
recuperarli: il recall massimo teorico è **66%**.

**Intervento.**

1. Abbassare `confidence` da 0,35: è alta per le scene notturne e `Very_Poor`
   (651 video notturni e 1.075 fra `Poor` e `Very_Poor` nel dataset).
2. Confrontare `yolo26n-seg` / `yolo26s-seg` / `yolo26m-seg` su recall delle
   detection e costo, non su F1 finale.
3. Tarare i parametri ByteTrack (soglie di match e `track_buffer`) per ridurre
   gli ID switch, che il punto 1 mitiga ma non elimina.

**Verifica.** Misurare direttamente `frames_with_multiple_tracks` e lo stadio 1
del funnel, prima di guardare gli eventi.

---

### ⬜ 7 — Trattare separatamente gli incidenti `single`

**Problema.** 680 video su 2027 (**34%**) sono a veicolo singolo: uscita di
strada, ribaltamento. Una logica basata su coppie non può rilevarli per
costruzione, e sul campione hanno dato 0 TP e 0 FP come previsto.

**Intervento.** Escluderli dalla metrica principale (dichiarandolo) oppure
aggiungere un rilevatore mono-veicolo: deviazione brusca della traiettoria,
variazione dell'aspect ratio o dell'orientamento della maschera, arresto
anomalo. Il CSV ha già la colonna `rollover` (231 video) come segnale di
supporto.

---

### ⬜ 8 — Rendere la valutazione più severa e più informativa

**Problema A — matching solo temporale.** Un evento emesso sulla coppia
sbagliata al momento giusto conta come TP. Il CSV contiene
`center_x, center_y, x1, y1, x2, y2` (bbox normalizzata dell'incidente) che
`load_real_metadata_csv` **non legge**: è ground truth spaziale già disponibile.

**Problema B — nessun NMS temporale di scena.** Un urto reale può generare
eventi su più coppie adiacenti; `evaluate_records` matcha 1:1 e conta le altre
come FP. Il cooldown è per-coppia, non globale.

**Problema C — nessun set negativo.** Tutti i 2027 video contengono un
incidente, quindi il tasso reale di falsi allarmi su traffico pulito non è
misurabile. Serve un insieme di video senza incidenti.

**Extra.** `region` (20 valori) è caricato in `Annotation` ma non è fra i campi
di `by_group` in `src/benchmark.py`: aggiungerlo è gratis.

---

### ⬜ 9 — Disaccoppiare inferenza e logica per poter calibrare

**Problema.** A 13 FPS una passata completa costa **16,3 ore**. Con questo ciclo
di iterazione non è praticabile nessuno sweep di soglie, ed è il motivo per cui
le soglie attuali sono tarate su 4 video.

**Intervento.** Una passata YOLO+ByteTrack che scrive su disco i track per frame
(bbox, poligono o maschera RLE, classe, confidenza); poi la logica di collisione
gira offline a migliaia di fps e uno sweep costa secondi. Oggi `--diagnostics`
salva i record `track` **senza geometria**, quindi l'evidenza spaziale non è
ricalcolabile a posteriori: aggiungere il poligono al dump è il prerequisito.

**Nota.** Questo abilita anche la strada più promettente a medio termine:
mantenere il layer di estrazione delle evidenze — che è buono e spiegabile — e
sostituire il predicato booleano finale (~25 iperparametri accoppiati, nati da 4
casi di regressione) con uno **scorer calibrato**. `PairDiagnostic` è già
esattamente il feature vector, ed è già serializzato in JSONL.

---

## 3. Come rieseguire il benchmark

Il dataset atteso è `dataset/metadata-real.csv` + `dataset/real_videos/*.mp4`,
con la colonna `path` relativa a `--dataset-root`.

```powershell
# 1. run stratificata: una invocazione per tipo, --limit non stratifica da solo
foreach ($t in "rear-end","t-bone","sideswipe","head-on","single") {
  .\.venv\Scripts\python.exe -m src.benchmark `
    --metadata dataset/metadata-real.csv --dataset-root dataset `
    --model yolo26n-seg.pt --output-dir calibration/real `
    --split-field split_in_distribution --split test `
    --type $t --limit 10 --resume --diagnostics --tolerance-seconds 1.0
}

# 2. aggregazione su tutti i log (report.json copre solo l'ultima selezione)
$logs = Get-ChildItem calibration/real/*.jsonl | ForEach-Object { "--log", $_.FullName }
.\.venv\Scripts\python.exe -m src.calibration @logs `
  --annotations dataset/metadata-real.csv --tolerance-seconds 1.0
```

Tre dettagli non ovvi:

- **`--limit N` prende i primi N in ordine CSV**, non un campione casuale, e si
  applica alla lista filtrata nel suo insieme: con più `--type` dà N totali, non
  N per tipo. Per stratificare serve una run per tipo.
- **`--diagnostics` è indispensabile**: senza, i record `frame`/`track`/
  `diagnostic` non vengono scritti e `summarize_diagnostics` restituisce zeri.
  Costo misurato ≈ 150 MB per 50 video.
- **Il matching è puramente temporale a ±1 s** quando `--tolerance-frames` è 0
  (default) e i log contengono `timestamp_s`. È la scelta giusta: nel **5,6%**
  delle righe `accident_frame/fps` diverge da `accident_time` di oltre 1 s (fino
  a 8,1 s).

---

### 3.1 Campione ristretto e confronto fra varianti

Una passata da 50 video costa 27 minuti e va moltiplicata per ogni
configurazione da confrontare. Per il punto 2b si e usato un campione piu
piccolo: **3 video per tipo** (15 video) piu i **4 casi di regressione**
documentati in `PROJECT_STATUS.md`, tenuti in una cartella separata perche sono
l'unico sottoinsieme su cui il sistema oggi emette davvero eventi.

Le configurazioni non si confrontano piu modificando `src/config.py` fra un run
e l'altro: `--set nome=valore` sovrascrive qualsiasi campo di `AppConfig` per
la singola esecuzione, e l'override finisce nel record `metadata` di ogni JSONL
e nella sezione `selection` di `report.json`. Una modifica al sorgente non
lascia invece nessuna traccia nel log, ed e cosi che si perde la tracciabilita
di cosa ha prodotto quali numeri.

```powershell
foreach ($t in "rear-end","t-bone","sideswipe","head-on","single") {
  .\.venv\Scripts\python.exe -m src.benchmark `
    --metadata dataset/metadata-real.csv --dataset-root dataset `
    --model yolo26n-seg.pt --output-dir calibration/sample_reid `
    --split-field split_in_distribution --split test `
    --type $t --limit 3 --resume --diagnostics --tolerance-seconds 1.0 `
    --set track_reid_enabled=true
}
```

Serve una `--output-dir` diversa per variante: con la stessa cartella `--resume`
considera gia completati i log dell'altra configurazione. In compenso `--resume`
rende gratuito allargare il campione, perche `--limit 6` sulla stessa cartella
ricalcola solo i tre video nuovi per tipo.

Il campione e stato poi portato a **6 video per tipo (30 video)** per le due
sole configurazioni fra cui andava presa una decisione, riusando i 15 gia
calcolati. Il confronto a quattro configurazioni resta sui 15 video.

**Baseline del campione a 15 video** (entrambi i flag del punto 2b `False`):

| Stadio del funnel | sample (15) | regressione (4) |
| --- | ---: | ---: |
| Video annotati | 15 | 4 |
| ≥2 track nella finestra | 7 | 4 |
| Contatto spaziale | 4 | 4 |
| Non disarmato da `preexisting_contact` | 4 | 4 |
| Moto precedente | 3 | 4 |
| Evidenza dinamica | 3 | 4 |
| `sustained_approach` | 2 | 2 |
| Contatto fresco (`age ≤ 5`) | 0 | 1 |
| Coppia candidata | 1 | 3 |
| **Evento emesso** | 0 | 3 |
| TP / FP / FN | 0 / 0 / 15 | 3 / 2 / 1 |

Due letture immediate, indipendenti dal punto 2b:

- Il campione da 15 video parte da **0 TP**: il recall non puo scendere e puo
  salire al massimo di 1-2 punti. Su un campione cosi il giudizio sta nel
  funnel e nei casi di regressione, non nella F1.
- I 4 video di regressione danno oggi **3 TP / 2 FP**, non i 3 TP / 0 FP
  documentati caso per caso in `PROJECT_STATUS.md`. I due falsi positivi sono:
  la coppia `9-19` al frame 34 in `987C4_UdnJE_01` — proprio una delle quattro
  sovrapposizioni prospettiche che la scheda del punto 3.7 dichiara soppresse —
  e la coppia `2-54` al frame 416 in `8G56ILxFFNM_00`, che la scheda dichiara a
  zero falsi positivi. Le altre tre attribuzioni reggono: `2-35` al frame 199,
  `19-20` al frame 36, `21-22` al frame 126. Le due schede vanno riallineate.

  Il confronto e stato fatto con il codice attuale e i flag del punto 2b
  spenti, quindi la regressione e anteriore a questo lavoro; non e stato
  possibile attribuirla a un intervento preciso, perche i log JSONL da cui le
  schede erano state scritte non sono piu in repo.

## 4. Ordine consigliato

Riordinato dopo le misure dei punti 1 e 2.

1. ✅ **Punto 1** — sblocco strutturale (fatto, effetto verificato)
2. 🟡 **Punto 2** — implementato ma disattivato: misurato come regressione
3. ✅ **Punto 2b** — coppie senza storia: l'avvicinamento impulsivo e attivo e
   vale +1 TP senza falsi positivi su 30 video; la re-identificazione su
   posizione e misurata inefficace e resta spenta
4. **Punto 4** — eta del contatto candidato. **Salito di priorita**: nel
   campione ristretto lo stadio *contatto fresco* e l'unico che azzera il
   funnel (2 video raggiungono `sustained_approach`, 0 superano l'eta), e in
   `OXVXsdDT9QQ_00` la coppia reale perde la candidatura per un solo frame di
   eta. Costa poco e sblocca uno stadio oggi chiuso.
5. ✅ **Punto 3** — delta-V vettoriale: attivo a soglia 10, +1 TP senza falsi
   positivi. L'uso nell'attribuzione fra traiettorie incrociate e invece
   misurato dannoso e resta spento
6. **Punto 5** — soglia di confidenza, per poter scegliere il punto operativo su
   una curva invece che a caso
7. **Punto 9** — dump offline: a 27 minuti per ciclo di misura su soli 50 video,
   e ormai il vincolo principale sulla velocita di iterazione. Il campione
   ristretto di §3.1 e `--set` sono una mitigazione, non una soluzione: una
   matrice di 4 configurazioni su 19 video costa comunque circa un'ora, e ogni
   riga della matrice ripaga da capo l'inferenza YOLO senza che nessuna delle
   configurazioni confrontate la cambi.
8. **Punto 6** — percezione. Resta il tetto, e resta il vincolo dominante: sui
   30 video del campione esteso solo **19** hanno due track nella finestra
   dell'incidente (63%), e appena **14** arrivano al contatto spaziale (47%).
   Metà dei falsi negativi non e raggiungibile da nessuna modifica alla logica
   di collisione.
9. **Punto 8B** — NMS temporale di scena. **Salito di priorita**: e la
   condizione mancante perche il delta-V competitivo possa attribuire un urto
   fra coppie che si toccano in sequenza, e vale di per se sui falsi positivi
   di `987C4_UdnJE_01`, dove un solo urto fisico produce cinque eventi
10. **Punti 7 e 8** — metrica onesta e casi `single`

## 5. Lezioni di metodo

Due cicli su tre hanno prodotto un risultato diverso da quello atteso. Vale la
pena registrarne il motivo.

- **Misurare il meccanismo prima di scrivere il fix.** Al punto 1 l'ispezione
  dei log ha mostrato che il problema non era il disarmo permanente ma la
  condizione di arming: senza quella misura il fix sarebbe stato incompleto.
- **Una modifica teoricamente corretta puo peggiorare.** Il punto 2 e giusto in
  linea di principio e sbagliato su questi dati, perche la popolazione che conta
  (le coppie che collidono) non ha le stesse statistiche della popolazione
  generale su cui si calcola il riferimento.
- **Tenere separabili i meccanismi.** Aver diviso magnitudini e finestre in due
  flag ha permesso di attribuire il danno alla meta giusta senza rifare tutto.
- **Il recall non e l'unica metrica di progresso.** Il punto 1 non ha mosso il
  recall ma ha spostato 8 video da "impossibili" a "bloccati da una soglia": e
  quello spostamento nel funnel a dire se un intervento e servito.
- **La configurazione di un run deve stare nel log, non nel working tree.**
  Fino al punto 2b le varianti si confrontavano modificando `src/config.py` fra
  un run e l'altro: i numeri sopravvivono, ma la configurazione che li ha
  prodotti no. Con `--set` l'override finisce nel record `metadata` di ogni
  JSONL, quindi un log vecchio resta interpretabile anche mesi dopo. La prova
  del contrario e in repo: i 50 log del 28 agosto in `calibration/real` hanno un
  `config` privo dei campi introdotti dopo, ed e l'unico motivo per cui si sa
  che sono anteriori al punto 1.
- **Una soglia non e trasferibile fra un segnale filtrato e uno grezzo.** La
  distribuzione del delta-V ricostruita dai log EMA dice se la grandezza *serve*
  — e lo dice bene, con il 73% di evidenza nuova — ma non dove tagliarla. Sul
  dato grezzo la mediana passa da 0,31 a 1,00 e il p99 da 7,68 a 14,58: la
  soglia stimata sul filtrato sarebbe stata quasi il doppio troppo bassa. Sono
  due domande distinte e vanno misurate su due dati distinti.
- **Una grandezza per-track non puo fare da prova per-coppia.** Il delta-V
  funziona come evidenza che *qualcosa* e successo a un veicolo e fallisce come
  prova di *con chi* e successo: un veicolo che attraversa piu flussi ha delta-V
  alto su ogni coppia prospettica, quindi riporta il difetto che il gate delle
  traiettorie incrociate esisteva per correggere. Prima di usare una misura in
  un ruolo di attribuzione, chiedersi a quale entita appartiene.
- **Un controllo di non-regressione puo mentire per colpa della baseline.** Il
  confronto a gate spenti segnalava tre video cambiati; la causa non era il
  codice nuovo ma il fatto che la cartella di riferimento era stata prodotta
  prima di accendere l'avvicinamento impulsivo. Un log di configurazione
  incorporato nel run — vedi la lezione precedente — e cio che ha permesso di
  capirlo in un minuto invece che di inseguire un bug inesistente.
- **Un campione piccolo cambia quale metrica puo rispondere.** Sui 15 video di
  §3.1 il sistema parte da 0 TP: precision e recall non hanno risoluzione
  sufficiente per distinguere due varianti, e l'unica misura informativa e di
  quanti video si sposta ciascuno stadio del funnel. Sceglierlo sapendolo e
  legittimo; leggerne poi la F1 come se fosse un risultato, no.

# Piano di miglioramento del crash detection

Data: 29 agosto 2026 · Branch: `main_v2` · Modello di riferimento: `yolo26n-seg.pt`

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

**Intervento proposto.** Ammettere un percorso di conferma alternativo per le
coppie giovani, in cui l'intensita sostituisce la persistenza: un singolo frame
con closing speed molto elevato, oppure un overlap che compare bruscamente fra
due track di cui almeno uno era in moto confermato *sotto il suo ID precedente*.
In alternativa, attaccare il problema alla radice con una re-identificazione che
erediti la storia cinematica quando un nuovo ID compare dove un altro e appena
sparito.

Questo e ora il candidato numero uno, davanti al punto 3.

---

### ⬜ 3 — Delta-V vettoriale invece della derivata del modulo

**Problema.** `compute_acceleration` è la differenza finita della *magnitudine*
della velocità. Un t-bone che devia un veicolo di 90° senza cambiarne il modulo
produce accelerazione ≈ 0 e non attiva nessuna evidenza dinamica. Inoltre l'EMA
con α = 0,45 smorza proprio il transitorio dell'impatto, rendendo
`strong_deceleration_threshold = -4.0` difficile da raggiungere.

**Evidenza.** *Evidenza dinamica assente* blocca **6 video su 18** che arrivano
allo stadio precedente. I `t-bone` sono 657 su 2027 nel dataset.

**Intervento.**

1. Calcolare `|Δv⃗| = |v⃗ₜ − v⃗ₜ₋₁|` (delta-V, la firma fisica dell'urto) e usarlo
   come evidenza dinamica accanto alla decelerazione scalare.
2. Mantenere **due velocità**: quella filtrata EMA per stimare traiettoria e
   direzione, quella raw per rilevare il transitorio.

File: `src/kinematics.py`, `src/models.py`, `src/tracker_state.py`,
`src/collision_logic.py`, `src/config.py`.

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

## 4. Ordine consigliato

Riordinato dopo le misure dei punti 1 e 2.

1. ✅ **Punto 1** — sblocco strutturale (fatto, effetto verificato)
2. 🟡 **Punto 2** — implementato ma disattivato: misurato come regressione
3. **Punto 2b** — coppie senza storia: e il collo di bottiglia reale emerso
   misurando, ed e ora il candidato con il rapporto impatto/rischio migliore
4. **Punto 3** — delta-V vettoriale
5. **Punto 5** — soglia di confidenza, per poter scegliere il punto operativo su
   una curva invece che a caso
6. **Punto 9** — dump offline: a 27 minuti per ciclo di misura su soli 50 video,
   e ormai il vincolo principale sulla velocita di iterazione
7. **Punto 4** — eta del contatto candidato
8. **Punto 6** — percezione
9. **Punti 7 e 8** — metrica onesta e casi `single`

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

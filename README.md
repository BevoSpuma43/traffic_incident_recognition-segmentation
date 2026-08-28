# Traffic Accident Segmentation System

Questo progetto è un sistema per il rilevamento di incidenti stradali a partire da flussi video. Sfrutta un modello **Ultralytics YOLO segmentation** per segmentazione e tracking (ByteTrack) dei veicoli, insieme a un motore cinematico temporale per la rilevazione delle collisioni.

---

## 📂 Struttura del Progetto

Il progetto è suddiviso in un modulo principale `src` che contiene la logica di business, l'interfaccia utente (GUI) e la pipeline video, e una suite di test `tests` per validare la geometria e la cinematica. Nella root del progetto è presente anche uno script standalone `test_segmentation.py` per un test rapido del modello.

---

## 🔍 Analisi Dettagliata dei File Sorgente

### 1. `test_segmentation.py`
È uno script standalone di base per verificare il funzionamento del modello YOLO con il tracking attivato, applicando una trasparenza (alpha blending) alle maschere.
**Pezzo di codice chiave:**
```python
# Inferenza del modello sul singolo frame con ByteTrack persistente
results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", classes=vehicle_classes, verbose=False)
```
Questo script serve principalmente come demo o test di sanità prima di usare la pipeline completa.

---

### 2. `src/config.py`
Contiene la configurazione centralizzata dell'intera pipeline tramite una singola dataclass `AppConfig`. Questo modulo rappresenta la singola "source of truth" per le soglie di rilevamento, semplificando la calibrazione del sistema senza modificare il codice.
**Pezzi di codice chiave:**
```python
# Soglie fondamentali per la cinematica e l'overlap
mask_overlap_threshold: int = 120
stopped_speed_threshold: float = 2.5
stopped_frames_threshold: int = 3
strong_deceleration_threshold: float = -4.0
```

---

### 3. `src/models.py`
Questo modulo definisce le strutture dati di dominio usate in tutta l'applicazione, garantendo type hinting e coerenza. Le classi principali sono `Point2D`, `DetectionResult`, `VehicleState` e `CollisionEvent`.
**Pezzi di codice chiave:**
```python
@dataclass(slots=True)
class CollisionEvent:
    frame_index: int
    track_id_a: int
    track_id_b: int
    overlap_area: int
    reason: str
```
La dataclass `VehicleState` memorizza l'intero stato del veicolo, dalla cinematica (velocità, accelerazione) alla geometria (bbox, maschera) frame-per-frame.

---

### 4. `src/detector.py`
Contiene la classe `VehicleSegmenter`, il wrapper che incapsula il modello YOLO. Per ogni frame video esegue l'inferenza e normalizza i dati grezzi in una lista di oggetti `DetectionResult`.
**Pezzi di codice chiave:**
```python
# Creazione sicura della maschera binaria OpenCV dai poligoni estratti
polygon = safe_polygon_array(polygons_raw[index])
mask = polygon_to_binary_mask(polygon, frame.shape[:2])
bbox = compute_bbox_from_polygon(polygon)
```
Gestisce anche le anomalie e i casi degeneri per evitare crash in caso YOLO restituisca forme invalide o mancanti.

---

### 5. `src/geometry.py`
Fornisce funzioni matematiche pure che operano sui poligoni e sulle maschere binarie tramite le librerie NumPy e OpenCV. Qui non esistono dipendenze verso altri file del progetto.
**Pezzi di codice chiave:**
```python
def mask_intersection_area(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> int:
    # bitwise_AND: 255 se entrambe le maschere sono attive
    intersection = cv2.bitwise_and(mask_a, mask_b)
    return int(np.count_nonzero(intersection > 0))
```
Calcola inoltre i centroidi usando i momenti geometrici via `cv2.moments(polygon_array)`.

---

### 6. `src/kinematics.py`
Funzioni pure per la cinematica in spazio pixel. Calcolano il vettore velocità
normalizzandolo per i frame trascorsi e applicano un filtro EMA per ridurre il
jitter del tracker.
**Pezzi di codice chiave:**
```python
vx, vy = compute_velocity_px(previous_anchor, motion_anchor, frame_delta)
speed_px = math.hypot(vx, vy)
```
Espone inoltre accelerazione e contatore di stop consecutivo.

---

### 7. `src/tracker_state.py`
Implementa lo store `VehicleStateStore` che mantiene lo storico frame-to-frame di ogni traccia assegnata da ByteTrack.
Usa il punto inferiore centrale della bbox come ancora di moto, conserva una
storia limitata e invalida la cinematica dopo gap troppo lunghi.
**Pezzi di codice chiave:**
```python
kinematics_valid = 0 < frame_delta <= max_kinematic_gap_frames
state.history.append(track_sample)
```
Questo modulo fa da ponte tra il layer di YOLO/ByteTrack e il layer della pura logica di collisione.

---

### 8. `src/collision_logic.py`
Core temporale del sistema. Per ogni coppia mantiene uno stato persistente e
combina due famiglie di evidenze:

- contatto spaziale: overlap assoluto o normalizzato, oppure contatto tra
  maschere dilatate per riconoscere sagome adiacenti senza pixel condivisi;
- dinamica: decelerazione brusca, transizione movimento-arresto e dual stop
  solo quando esiste movimento precedente.

Le evidenze possono cadere in frame vicini grazie a una finestra temporale.
Un candidato deve essere confermato più volte e, dopo l'emissione, la coppia
entra in cooldown per evitare un evento duplicato a ogni frame.

Una coppia le cui maschere risultano già in contatto alla prima osservazione
viene marcata come `preexisting_contact`: non può generare eventi finché le
maschere non restano separate per più frame consecutivi. Questo impedisce alle
auto già in coda all'inizio del video di essere classificate come incidente a
causa della prospettiva. Anche il movimento pre-impatto deve persistere per più
campioni, così un singolo salto della segmentazione non riattiva il candidato.

Il marchio scatta però solo se entrambi i track sono appena comparsi: una
coppia nuova fra un track maturo e un ID appena creato nasce quasi sempre da
una riassegnazione del tracker durante l'occlusione dell'urto, non da due
veicoli già accostati. Il disarmo scade inoltre dopo un numero massimo di
frame, così due veicoli che restano a contatto dopo un impatto non restano
esclusi per il resto del video.

Nel traffico parallelo lento, anche l'avvicinamento relativo deve essere
confermato per più frame ed essere temporalmente vicino all'inizio del contatto.
Se le maschere rimangono sovrapposte a lungo, una variazione cinematica tardiva
non viene più attribuita retroattivamente a quel contatto. Un overlap forte
senza convergenza può superare il filtro soltanto insieme a una decelerazione
brusca.

Per gli urti contro un veicolo stabilmente fermo viene usata una memoria di
traiettoria separata dalla finestra d'impatto. Lo storico più lungo stima la
direzione abituale e conserva l'avvicinamento osservato durante un contatto
prospettico persistente; l'evento richiede però una reazione del veicolo mobile
su almeno due frame (frenata/arresto o deviazione netta). In questo modo una
singola frenata apparente durante una svolta non resta valida più a lungo, ma
una deviazione successiva a un vero impatto può essere riconosciuta.

Quando due traiettorie si incrociano, la sola anomalia del veicolo trasversale
non viene più riutilizzata per tutte le coppie le cui maschere incontra. Il
bersaglio deve mostrare una perturbazione locale, come una deviazione oppure
un'accelerazione positiva improvvisa coerente con un trasferimento di moto.
Per i contatti dovuti alla sola dilatazione delle maschere, ogni frame di
conferma richiede inoltre evidenza dinamica o avvicinamento corrente.

Un impatto fronto-laterale può però arrestare entrambi i veicoli senza
deviazioni visibili e nascondere temporaneamente uno dei track proprio durante
l'urto. Per questo il rilevatore conserva per 12 frame il solo contesto di
avvicinamento incrociato. Il contesto può confermare un `dual_stop` soltanto se
la coppia è mancata per almeno due frame, ricompare con overlap reale e i due
track risultano fermi in modo persistente. L'arresto simultaneo senza gap di
tracking resta escluso, così una normale precedenza all'incrocio non sfrutta
questa eccezione.

---

### 9. `src/video_pipeline.py`
Il direttore d'orchestra (`TrafficAccidentPipeline`). Si occupa di chiamare sequenzialmente per ogni frame: il segmenter, l'update dello state store, il collision detector e infine il renderer.
**Pezzi di codice chiave:**
```python
detections = self.segmenter.segment_and_track(frame=frame)
active_states = self.state_store.update(detections, frame_index)
self.state_store.remove_stale_tracks(frame_index)
collisions = self.collision_detector.detect_collisions(active_states, frame_index)
annotated = self._render(frame, active_states, collisions)
```

---

### 10. `src/renderer.py`
Si occupa solo del lato visivo. Costruisce le sovrapposizioni colorate per i veicoli, disegna centroidi, bounding box e segna in rosso (`CRASH?`) i veicoli considerati in collisione.
**Pezzi di codice chiave:**
```python
# Evidenziatore di incidente
cv2.rectangle(output, (x1, y1), (x2, y2), _CRASH_COLOR, 3, lineType=cv2.LINE_AA)
cv2.putText(output, "CRASH?", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, _CRASH_COLOR, 2, cv2.LINE_AA)
```
Usa un hash LCG pseudo-casuale deterministico per colorare con costanza i veicoli tra i frame.

---

### 11. `src/app.py`
Questo è l'entry point in formato Command Line Interface (headless) per eseguire la pipeline usando la `TrafficAccidentPipeline` creata ed evidenziare gli incidenti in console.
**Pezzi di codice chiave:**
```python
# Stampa a terminale in modo leggibile
print(f"[COLLISION] frame={frame_number} tracks={tracks} overlap={overlap_area}px reason={reason}")
```

---

### 12. `src/ui_main.py`
La GUI principale basata su `customtkinter`. Consente all'utente di caricare un video e un modello via path, premere "Start" per avviare il processing e "Stop" per terminarlo. Separa il layer di visualizzazione in un thread worker demonizzato.
**Pezzi di codice chiave:**
```python
self._worker_thread = threading.Thread(
    target=self._worker_loop,
    name="traffic-processing-thread",
    daemon=True,
)
self._worker_thread.start()
```

---

### 13. I File di Test (`tests/`)
- `tests/test_geometry.py`: Verifica che funzioni cruciali come `compute_centroid_from_polygon` e `mask_intersection_area` siano matematicamente ineccepibili.
- `tests/test_kinematics.py`: Testa l'accuratezza del calcolo distanza euclidea (`compute_speed_px`) e l'accelerazione per prevenire falsi positivi nel modulo cinematico.
- `tests/test_collision_logic.py`: Testa la business logic del collision detector fornendo falsi stati simulati ed accertando che le soglie restituiscano le corrette collisioni limitando l'overlap insufficiente o comportamenti cinematici falsati.

---

## Benchmark sul dataset reale

Il progetto legge direttamente `dataset/metadata-real.csv`: ogni riga viene
associata al video omonimo in `dataset/real_videos`. Il timestamp effettivo
restituito dal decoder viene confrontato con `accident_time`; `accident_frame`
rimane il riferimento parallelo e il fallback per i log che non contengono il
tempo decodificato.

Il dataset attuale è stato verificato: contiene 2.027 righe e altrettanti MP4,
senza path mancanti o duplicati. Le classi sono 680 `single`, 657 `t-bone`,
328 `rear-end`, 245 `sideswipe` e 117 `head-on`; lo split in-distribution è
composto da 507 video train e 1.520 test. Il riepilogo `ground_truth` del report
segnala inoltre automaticamente timestamp incoerenti e frame fuori intervallo.

Per una prova rapida su 10 video del test set:

```powershell
python -m src.benchmark `
  --metadata dataset/metadata-real.csv `
  --dataset-root dataset `
  --split-field split_in_distribution `
  --split test `
  --limit 10 `
  --resume
```

Il comando elabora i video senza rendering, salva un JSONL per video in
`calibration/real` e genera `calibration/real/report.json`. Il report contiene
precision, recall, F1, ritardo di rilevamento, throughput, fattore real-time e
metriche separate per tipo di incidente, rollover, scenario, meteo, fascia
oraria e qualità. La tolleranza predefinita è di 1 secondo; può essere cambiata con
`--tolerance-seconds` o `--tolerance-frames`.

Un singolo caso di regressione può essere selezionato con, ad esempio,
`--video 8G56ILxFFNM_00.mp4`.

Per calibrare le soglie su un sottoinsieme ristretto si può aggiungere
`--diagnostics`: vengono registrate anche le evidenze di ogni coppia per ogni
frame e lo stato cinematico di ogni track (bbox, velocità, accelerazione,
direzione e anzianità). Questa opzione è volutamente disattivata nel benchmark normale perché
può produrre file molto grandi. Per valutare log già esistenti:

```powershell
python -m src.calibration `
  --log calibration/real/NOME_VIDEO.jsonl `
  --annotations dataset/metadata-real.csv `
  --tolerance-seconds 1
```

Il vecchio formato a intervalli JSON (`calibration/annotations.example.json`)
rimane supportato per annotazioni manuali con inizio e fine dell'impatto.

Il dataset reale contiene un incidente annotato per ciascun video. Permette
quindi di misurare rilevamenti mancati, eventi fuori finestra e duplicati, ma
non stima da solo il tasso di falsi allarmi su video senza incidenti: per una
precisione operativa completa serve anche un insieme negativo.

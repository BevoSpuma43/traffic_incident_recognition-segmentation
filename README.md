# Traffic Accident Segmentation System

Questo progetto è un sistema avanzato per il rilevamento di incidenti stradali a partire da flussi video. Sfrutta **YOLOv8** per la segmentazione e il tracking (ByteTrack) dei veicoli, accoppiato ad un solido motore cinematico per l'analisi del movimento e la rilevazione di collisioni.

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
Funzioni pure per la valutazione della cinematica in spazio pixel. Usa un protocollo `_HasXY` per ottenere le coordinate spaziali senza legarsi ad una implementazione concreta.
**Pezzi di codice chiave:**
```python
def compute_speed_px(prev_centroid: _HasXY | None, curr_centroid: _HasXY | None) -> float:
    # Calcolo della norma L2 stabile
    dx = float(curr_centroid.x) - float(prev_centroid.x)
    dy = float(curr_centroid.y) - float(prev_centroid.y)
    return math.hypot(dx, dy)
```
Calcola la velocità (pixel/frame), l'accelerazione (differenza delle velocità) ed espone un contatore per capire da quanti frame un veicolo è fermo.

---

### 7. `src/tracker_state.py`
Implementa lo store `VehicleStateStore` che mantiene lo storico frame-to-frame di ogni traccia assegnata da ByteTrack.
Esegue l'aggiornamento dei centroidi, il ricalcolo della cinematica e la pulizia dei track vecchi (stale).
**Pezzi di codice chiave:**
```python
speed_px: float = compute_speed_px(previous_centroid, centroid)
acceleration_px: float = compute_acceleration(speed_px, previous_speed_px)
state.speed_px = speed_px
state.acceleration_px = acceleration_px
```
Questo modulo fa da ponte tra il layer di YOLO/ByteTrack e il layer della pura logica di collisione.

---

### 8. `src/collision_logic.py`
Core del sistema che rileva gli incidenti. Verifica le coppie di veicoli applicando l'euristica basata su: sovrapposizione maschere + anomalia cinematica (fermata simultanea o decelerazione anomala).
**Pezzi di codice chiave:**
```python
# Anomalia cinematica 1: Dual Stop (entrambi fermi)
dual_stop = self._both_stopped(vehicle_a, vehicle_b)

# Anomalia cinematica 2: Hard Deceleration (frenata brusca)
hard_deceleration = self._hard_deceleration(vehicle_a, vehicle_b)

# Controllo sovrapposizione maschere
has_overlap, overlap_area = self._has_overlap(vehicle_a, vehicle_b)
```
Se unisce un overlap positivo e una di queste due anomalie, l'evento di collisione `CollisionEvent` viene emesso.

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

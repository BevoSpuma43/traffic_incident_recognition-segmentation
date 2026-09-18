# Stato dell'implementazione

## Software disponibile

- Package Python installabile, ambiente Python 3.11, uv.lock, configurazioni validate.
- Video MP4 con PTS, campionamento dell'inferenza e decoder RTSP con riconnessioni limitate.
- YOLO26n-seg con pesi locali, filtro classi, maschere alla risoluzione originale.
- ByteTrack reale, associazione maschera–ID, scadenza temporale delle tracce.
- Calibrazione manuale interattiva o da JSON, serializzazione, ROI e controlli geometrici.
- Sfondo mediano, strisce, linee, punti di fuga e proposta assistita con revisione.
- Controllo spostamento camera e sospensione delle decisioni metriche.
- Traiettorie limitate, EMA temporale, derivate reali, coppie via griglia e TTC.
- Detector temporale, post-impatto, ragioni, rifiuto e cooldown.
- Ring buffer JPEG, scrittura H.264 VFR in thread, clip parziali segnalate, retention.
- SQLite, JSON, CSV, log JSONL e report runtime/hardware.
- UI Streamlit con video, bird's-eye, calibrazione, eventi e riproduzione clip.
- Split per sorgente/camera, training segmentazione e classificatore, export e ablation.
- Test unitari e integrazione, demo sintetica positiva e negativa.

## Video reali

Disponibili la modalita in coordinate immagine, il preset Streamlit e
il benchmark riproducibile su soli 10 video reali ACCIDENT.
Sono salvati video annotati, feature, tracce, metriche e confronto con le
annotazioni temporali. Procedura e limiti: [real-videos.md](real-videos.md).
Report locale: [campione reale](../outputs/accident-sample/report.md).

La calibrazione metrica rimane necessaria per velocita in km/h e distanze
sul piano stradale. La baseline in pixel ha regole proprie e richiede
una valutazione dell'accuratezza distinta dalla verifica del software.

## Limiti e attività dipendenti da dati/hardware

Questa è una baseline software funzionante, non il completamento degli esperimenti di sei
settimane del piano. ACCIDENT v9 è ora disponibile localmente (vedere [accident-dataset.md](accident-dataset.md)); mancano ancora le misure di calibrazione delle scene.
Rimangono quindi aperti: fine-tuning, valutazione su sorgenti mai viste, un'ora di negativi,
IDF1/HOTA, confronto automatico/manuale, ablation reali e selezione delle soglie.

La proposta automatica richiede sempre revisione. La metrica completa da sole strisce
longitudinali non viene inferita. Il modello learned per strisce, la piccola GRU/TCN e
l'auto-calibrazione metrica completamente autonoma non sono implementati.

Il detector conferma soprattutto arresti post-contatto e applica una regola conservativa
per urti di un solo veicolo; non copre tutte le dinamiche d'incidente. Il punteggio non è
calibrato come probabilità. INT8/FP16 e gli altri backend hanno script di export, ma non
sono stati confrontati su questa macchina. RTSP è implementato ma non provato con una
camera fisica. La perdita completa degli ID può aggirare la deduplicazione per traccia.

Lo smoke test YOLO e la demo verificano l'esecuzione. I risultati misurati di questa sessione
sono in outputs/environment.json, outputs/demo/runs/ e nel report sintetico generato alla
fine della verifica. Non costituiscono risultati di generalizzazione su CCTV reali.

Risultati misurati e limiti del campione: [real-video-validation.md](real-video-validation.md).

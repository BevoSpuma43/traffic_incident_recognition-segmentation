# Verifica dell'implementazione — 13 settembre 2026

- Python 3.11.15, Windows, 12 CPU logiche, 31.6 GiB RAM; PyTorch CPU.
- uv.lock generato e ambiente installato con extra inference, ui e ml.
- Pesi locali YOLO26n-seg: SHA256 361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f.
- pytest: **33 passed in 6.90s** nell'ultima esecuzione.
- Ruff: check superato, 51 file Python formattati.
- YOLO sull'immagine bus inclusa nella libreria: un veicolo segmentato, maschera 1080 × 810.
- Pipeline con video di immagine bus ripetuta: 16 osservazioni, un ID stabile, log JSONL validi.
  Questo smoke test breve, con avvio incluso, ha misurato circa 4.52 FPS.
  Non verifica il target su CCTV reali né la velocità a regime.
- Demo sintetica positiva: un evento, una clip H.264 con contesto prima/dopo.
- Demo sintetica negativa: zero eventi.
- Valutazione dei due video sintetici: TP=1, FP=0, FN=0; ritardo=1.1 s.
  Totale video 15.9 s: campione di test software, non misura significativa di falsi allarmi/ora.
- Comandi export-events, evaluate ed extract_features eseguiti con successo.
- Streamlit verificato mediante AppTest, senza eccezioni all'avvio.
- Test specifici: timestamp VFR, scadenza/riapparizione tracce, omografie degeneri,
  movimento camera, fallback strisce, proposta assistita, conferma/cooldown,
  negativi difficili, memoria del buffer, limite della coda e fine video.
- Nessun training su dataset reale, nessuna validazione di RTSP fisico o export accelerati.

Report dettagliati generati in outputs/environment.json, outputs/demo/evaluation.json
e nelle cartelle outputs/demo/runs/ e outputs/model-smoke/runs/.
I file generati e i pesi sono esclusi da Git; è conservato l'hash dei pesi.

Aggiornamento del 15 settembre 2026: disponibile la [verifica del campione reale ACCIDENT](real-video-validation.md).

# Dipendenze, pesi e dati

Il codice applicativo è stato scritto per questo progetto. Non è stata assegnata una licenza
di pubblicazione all'intero repository: la scelta compete al titolare.

Ultralytics e i modelli YOLO sono soggetti alle condizioni pubblicate dal fornitore:
https://www.ultralytics.com/license. Verificare l'applicabilità di AGPL-3.0 oppure di una
licenza enterprise alla distribuzione prevista. L'integrazione del tracker usa
l'implementazione nel pacchetto Ultralytics. Conservare i relativi avvisi.

Il file uv.lock documenta esattamente le versioni delle dipendenze. Le licenze dei pacchetti
rimangono quelle delle rispettive distribuzioni. Non copiare questa nota come parere legale.

I pesi vengono scaricati esplicitamente dallo script download_model.py e accompagnati da
un SHA256. Non sono inclusi nel controllo versione. L'immagine bus per lo smoke test resta
nell'installazione Ultralytics; non è stata copiata nel repository.

Nessun dataset CCTV di terzi è incluso o redistribuito. I video generati dalla demo
sono artificiali e vanno identificati come tali nei report.

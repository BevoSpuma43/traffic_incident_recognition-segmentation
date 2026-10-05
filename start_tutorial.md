# Avviare il web server

Le istruzioni sono per **Windows e PowerShell**. Nel progetto sono già presenti
l'ambiente `.venv` e i modelli YOLO: per il normale avvio basta seguire i primi tre passaggi.

## 1. Apri il terminale nella cartella del progetto

In VS Code scegli **Terminale → Nuovo terminale** e assicurati di usare PowerShell.
Esegui:

```powershell
cd "C:\Users\leona\dip_crash_detection_workspace\traffic_incident_recognition-segmentation"
```

## 2. Avvia Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Il comando usa direttamente Python dell'ambiente del progetto: **non serve attivare
manualmente `.venv`**. Lascia aperto il terminale mentre utilizzi l'interfaccia.

## 3. Apri la web GUI

Apri nel browser l'indirizzo indicato nel terminale come `Local URL`, normalmente:

[http://localhost:8501](http://localhost:8501)

Se il browser non si apre automaticamente, incolla l'indirizzo nella barra degli indirizzi.

Per analizzare il sottoinsieme del dataset:

1. Nella barra laterale scegli **Modalita → standard_dataset analisi in batch - no omografia**.
2. Seleziona il **modello YOLO** e la cartella **standard_dataset**.
3. Lascia `dataset/metadata-real.csv` come file delle etichette.
4. Premi **Prepara batch**, poi **Avvia batch**.

Per continuare un esperimento già iniziato, seleziona lo stesso modello e
l'esperimento salvato, quindi premi **Riprendi**.

La voce **standard_dataset analisi in batch - con omografia** è un segnaposto
per una futura implementazione e non avvia alcuna analisi.

I dettagli su risultati, metriche e checkpoint sono nella [guida batch](docs/batch-analysis.md).

## Arrestare o riavviare il server

Se è in corso un'analisi batch, premi prima **Stop nella GUI** e attendi che compaia
lo stato **In pausa**. L'analisi batch usa un processo separato: chiudere il browser
o arrestare il server web non è il comando per fermare quel processo.

Poi torna al terminale del server e premi:

```text
Ctrl+C
```

Per riavviare il server, esegui nuovamente:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

## Se la porta 8501 è già occupata

Controlla se il server è già aperto in un altro terminale. Puoi usare quello
esistente oppure avviarne uno su una porta diversa:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py --server.port 8502
```

In questo caso apri [http://localhost:8502](http://localhost:8502).

## Solo per una nuova installazione o dipendenze mancanti

Se `.venv` non esiste o compare un errore come `No module named streamlit`,
con `uv` disponibile esegui dalla radice del progetto:

```powershell
uv sync --frozen --extra inference --extra ui --extra ml --dev
```

Attendi il completamento, poi ripeti il comando di avvio del server.
L'installazione delle dipendenze richiede accesso a Internet.

Se durante un'analisi compare `Local model missing`, scarica esplicitamente il
modello richiesto. Per YOLO26s-seg, usato dalla modalità Rapida:

```powershell
.\.venv\Scripts\python.exe scripts/download_model.py --output models/yolo26s-seg.pt
```

Terminato il download, premi nuovamente **Avvia analisi** per il singolo video,
oppure **Riprendi** per il batch interrotto dall'errore. Non serve ripetere i download
a ogni avvio del server.

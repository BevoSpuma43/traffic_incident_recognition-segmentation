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

Se compare `DLL load failed while importing _ssl` con un blocco del controllo
applicazioni, consultare il [ripristino Python su Windows](docs/ripristino-python-windows.md).
L'ambiente del progetto usa Python 3.12 installato sul sistema.

## 3. Apri la web GUI

Apri nel browser l'indirizzo indicato nel terminale come `Local URL`, normalmente:

[http://localhost:8501](http://localhost:8501)

Se il browser non si apre automaticamente, incolla l'indirizzo nella barra degli indirizzi.

Per analizzare il sottoinsieme del dataset:

1. Nella barra laterale scegli **Modalita → standard_dataset analisi in batch - no omografia**.
2. Seleziona il **modello YOLO** e la cartella **standard_dataset**.
3. Lascia `dataset/metadata-real.csv` come file delle etichette.
4. Premi **Prepara batch**, poi **Avvia batch**.

**Prepara batch** crea sempre una nuova cartella di esperimento, anche se le
impostazioni coincidono con un batch precedente. Nella cartella sono salvati
`batch_config.json`, lo YAML originale (`source_config.yaml`) e i parametri
effettivi (`resolved_config.yaml`).

Per continuare un esperimento già iniziato, seleziona lo stesso modello e
l'esperimento salvato, quindi premi **Riprendi**.

Per preparare le calibrazioni, scegli **standard_dataset analisi in batch - con
omografia**, indica la cartella e premi **Crea sessione di preparazione**, poi
**Proponi calibrazioni mancanti**. Il worker salva bozze, alternative e diagnostica;
usa **Stop proposte** e **Riprendi proposte** per interrompere e continuare.
Seleziona un video dalla tabella o dal menu per revisionarlo nello stesso editor
del video singolo. **Prossimo video da revisionare** conserva la posizione anche
dopo la riapertura. Le misure e la conferma restano individuali.

Dopo aver confermato tutti i video, nel menu laterale **Batch con omografia**
seleziona **Analisi batch**. Scegli modello, cartella, etichette e FPS, poi premi
**Prepara batch** e **Avvia batch**. Ogni esperimento conserva copie indipendenti
delle calibrazioni; una calibrazione mancante o non valida blocca la preparazione.
Gli esperimenti salvati sono separati per modello e modalità.
Procedura e stati: [preparazione](docs/fase-5-preparazione.md) e
[batch metrico](docs/fase-6-batch-metrico.md).

I dettagli su risultati, metriche e checkpoint sono nella [guida batch](docs/batch-analysis.md).

Nel video singolo **Segmentazione + omografia**, **Calibrazione automatica**
mostra direttamente la prima proposta e nasconde i controlli di inserimento
manuale. Trascina i punti per correggerli oppure cambia **Proposta da esaminare**.
Le distanze disponibili sono già nei campi modificabili; quelle sconosciute
restano vuote. Per un riferimento compatibile puoi scegliere e applicare un
preset: compariranno i valori disponibili con la loro origine da confermare.
**Torna alla selezione manuale** riapre i controlli manuali conservando i punti.

Per un confronto fra modalità, usa `configs/paired-evaluation.yaml` in entrambe,
con stessi video, modello, FPS e ROI sull'intero frame. Completa prima la
conferma delle calibrazioni. La [guida della fase 7](docs/fase-7-validazione.md)
contiene comandi di confronto/esportazione, campione reale e checklist browser.
Un esperimento creato con codice diverso resta consultabile, ma richiede il
codice originale per la ripresa: non modificare l'hash nel manifest.

## Arrestare o riavviare il server

Se è in corso una preparazione, premi **Stop proposte** e attendi **In pausa**.
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

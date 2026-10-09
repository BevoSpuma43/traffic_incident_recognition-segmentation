# Ripristino dell'avvio Streamlit su Windows

Il 9 ottobre 2026 l'avvio si interrompeva con:

```text
ImportError: DLL load failed while importing _ssl:
Un criterio di controllo dell'applicazione ha bloccato il file.
```

Il registro `Microsoft-Windows-CodeIntegrity/Operational` (eventi 3033 e 3077)
ha identificato il blocco di `_ssl.pyd` e `_lzma.pyd` nel Python 3.11.15 gestito
da `uv`. Il problema si verificava prima dell'esecuzione dell'applicazione.
`Set-ExecutionPolicy` regola gli script PowerShell e non risolve questo blocco
delle librerie native.

## Ambiente ripristinato

- `.venv` ricreata con Python 3.12.10 già installato sul computer:
  `C:\Users\leona\AppData\Local\Programs\Python\Python312\python.exe`.
- Dipendenze reinstallate da `uv.lock`, inclusi gli extra `inference`, `ui`,
  `ml` e gli strumenti di sviluppo.
- Vincoli specifici per Windows: SciPy **1.17.1**, PyTorch **2.9.1+cpu** e
  Torchvision **0.24.1+cpu**. Windows bloccava anche `_linalg_pythran` di
  SciPy 1.18.1 e alcune estensioni native delle coppie PyTorch/Torchvision
  2.14/0.29 e 2.13/0.28. La coppia selezionata è pubblicata nelle
  [istruzioni ufficiali PyTorch](https://pytorch.org/get-started/previous-versions/#v291)
  ed è stata verificata con NMS e inferenza YOLO sul PC.
- `.python-version` aggiornata a `3.12`.
- `pyproject.toml` configurato con `python-preference = "only-system"`,
  secondo la [documentazione di uv](https://docs.astral.sh/uv/concepts/python-versions/#adjusting-python-version-preferences).
- Il vecchio ambiente e l'inventario dei suoi pacchetti sono conservati in
  `outputs/runtime-repair-20261009/`. La cartella `venv-python311-backup`
  è una copia di recupero; non va usata come ambiente attivo dopo lo spostamento.

La configurazione e il lock mantengono le versioni verificate su Windows.
La riparazione non modifica le policy di sicurezza del sistema.

## Avvio normale

Aprire un nuovo terminale nella radice del progetto ed eseguire:

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

Non serve attivare l'ambiente. Per verificare quale interprete viene usato:

```powershell
.\.venv\Scripts\python.exe -c "import sys, ssl; print(sys.version); print(sys.base_prefix); print(ssl.OPENSSL_VERSION)"
```

Per sincronizzare le dipendenze in futuro:

```powershell
uv sync --frozen --extra inference --extra ui --extra ml --dev
```

Il Python di sistema deve essere installato e individuabile da `uv`. Se serve
indicare esplicitamente il percorso, usare:

```powershell
uv sync --frozen --extra inference --extra ui --extra ml --dev --python "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
```

Il percorso dell'ultimo esempio corrisponde all'installazione standard per
utente di Python 3.12 e va adattato su altri computer.

## Verifica finale

- Import di SSL, LZMA, SciPy e moduli di calibrazione riuscito.
- Avvio del vero comando Streamlit: health check `200 ok` e pagina iniziale
  HTTP 200. Il server di prova è stato arrestato al termine.
- Operazione nativa Torchvision NMS riuscita e inferenza YOLO26n-seg su un
  fotogramma reale con cinque rilevamenti.
- Suite completa: **348 test superati in 129,40 secondi**, inclusi AppTest
  dell'interfaccia e test del modello reale.
- Il comando `uv sync --frozen --extra inference --extra ui --extra ml --dev
  --dry-run` verifica 84 pacchetti senza proporre modifiche.

Rapporti in `outputs/runtime-repair-20261009/`: `verification.json`,
`streamlit-server.log`, `server-check.json` e `pytest-final.xml`.

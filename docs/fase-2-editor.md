# Fase 2 — Editor grafico di calibrazione

Implementazione del 6 ottobre 2026, autorizzata dall'utente con «Esegui la fase 2».
La parte software e le verifiche automatiche sono concluse; resta la verifica
visiva del trascinamento in un browser reale, non disponibile nella sessione.

## Uso nella GUI

1. Selezionare **Segmentazione + omografia** e un video locale.
2. Aprire **Calibrazione**: viene letto il primo fotogramma decodificabile,
   senza ridimensionare le coordinate originali.
3. Cliccare P1, P2, P3 e P4 lungo il perimetro di un rettangolo stradale;
   trascinare i punti per correggerli. P1→P2 indica la larghezza e P2→P3
   la lunghezza. Le etichette rimangono associate agli stessi vertici.
4. Inserire larghezza e lunghezza in metri, indicare l'origine e confermare
   ciascuna misura. Valori da standard o ipotesi richiedono una fonte.
5. Se necessario scegliere **ROI stradale**. **Nuova ROI** avvia un poligono
   indipendente; **ROI: tutto il frame** ripristina l'intera immagine.
6. Controllare l'anteprima con griglia, spuntare la verifica del fotogramma
   e premere **Conferma e salva calibrazione**.

**Salva bozza** funziona anche senza punti o misure complete. Coordinate
numeriche, corrispondenze avanzate, reset e annullamento sono disponibili.
Le corrispondenze avanzate conservano l'ordine delle due liste e richiedono
una fonte e una conferma specifica della scala metrica.

Alla riapertura viene cercato il record compatibile con il contenuto del video:
una calibrazione confermata è visualizzata bloccata. **Riutilizza calibrazione**
richiede la revisione del fotogramma; **Modifica calibrazione** torna alla bozza.
L'analisi si abilita solo per un record salvato, confermato, revisionato nella
sessione e compatibile con la soglia di qualità configurata.

## Implementazione e limiti

- `calibration_ui.render_calibration_editor` è riutilizzabile con un namespace
  separato e restituisce percorso e camera del record selezionato.
- Il componente è CCv2 nativo con HTML/CSS/JavaScript inline nel pacchetto Python.
  Non richiede npm, CDN o componenti canvas obsoleti. Il requisito UI è ora
  `streamlit>=1.63,<2`, coerente con la versione installata e bloccata nel lockfile.
- Clic e trascinamento usano pixel originali ricavati dal rettangolo effettivo
  del canvas. Il browser disegna durante il movimento; invia un evento solo
  al rilascio. Ridimensionamenti aggiornano il disegno senza inviare modifiche.
- Identità del componente: video, record e revisione. Gli eventi includono
  inoltre una versione dello stato e un ID univoco: messaggi vecchi o duplicati
  non possono sovrascrivere una modifica successiva o un altro video.
- Modifiche a punti, distanze e ROI annullano l'accettazione. Annullare una
  modifica ripristina i dati, ma richiede nuovamente la conferma complessiva.
- Cambiare percorso, dimensione del file o data di modifica ricrea la sessione;
  salvataggio e riuso ricontrollano anche hash, risoluzione e primo frame.
- Sono mantenuti salvataggi atomici e revisioni ottimistiche della fase 1.
  La ROI può essere concava, ma deve avere area e vertici validi, senza incroci.
- L'anteprima è limitata a 640×640 pixel e a 40 linee per asse. Le dimensioni
  del rettangolo in metri non determinano le dimensioni dell'immagine renderizzata.
- La qualità geometrica è una semplice euristica di area e condizionamento
  normalizzato, limitata a 0,79. La conferma dell'utente non la porta a 1;
  questa euristica non stima l'errore metrico reale.

L'app principale seleziona il record e blocca l'analisi incompleta, come richiesto
dalla fase 2. L'adattamento alla risoluzione della pipeline e i metadati dei run
restano nella fase 4. La proposta automatica nel nuovo editor resta nella fase 3;
il batch metrico nelle fasi successive. La modalità immagine mantiene il flusso
preesistente. Gli URL RTSP non sono associabili all'archivio per video locale.

## Verifiche

- Pytest: **182 test superati in 25,17 s** nella suite completa del progetto,
  inclusi stato editor, persistenza,
  geometria, Streamlit AppTest e collegamento con l'app principale.
- AppTest: bozza incompleta, conferma numerica, riuso dopo revisione, modifica
  delle misure, annullamento, reset, ROI indipendente, sostituzione del file,
  sorgente mancante, scala avanzata e messaggio nativo CCv2 di rilascio.
  La pipeline è simulata nella prova del pulsante Analisi: nessuna inferenza.
- JavaScript: cinque gruppi di prove in V8 con canvas/DOM simulati: quattro clic,
  ID stabili, rilascio, resize e margini, cancellazione, limiti, pulizia dei
  listener, ROI e blocco. Queste prove non sono un controllo visivo del browser.
  Il test è in `tests/frontend/calibration_editor.mjs`; il runner Node opzionale
  è `tests/frontend/run_calibration_editor.mjs` (Node non installato qui).
- Lockfile verificato offline; wheel costruita offline e aperta fuori dall'albero
  sorgente con Python in modalità isolata. AppTest riceve canvas, CSS, JavaScript
  e immagine dal pacchetto estratto, senza dipendere da asset nel repository.
- Ruff superato, **93 file** già formattati e `git diff --check` superato.

Non è stato possibile eseguire il controllo visivo: l'inventario computer-use
non espone browser e i tentativi con IAB e Chrome restituiscono browser non
disponibile. AppTest non esegue JavaScript. Restano da osservare in un browser
reale trascinamento, ridimensionamento della finestra e cambio di video.

I 33 file del batch `outputs/batches/yolo26s-seg-55517fe0827b34e6` hanno gli
stessi hash prima e dopo questa fase. Nessun esperimento sui 150 video è stato
avviato, né è stato modificato il manifest per consentire una ripresa incompatibile.

Comandi dalla radice del progetto:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short
.\.venv\Scripts\ruff.exe check src apps scripts tests
.\.venv\Scripts\ruff.exe format --check src apps scripts tests
git diff --check
uv lock --check --offline
uv build --wheel --offline --out-dir outputs/phase2-verification/dist
# Opzionale, se Node è disponibile:
node tests/frontend/run_calibration_editor.mjs
```

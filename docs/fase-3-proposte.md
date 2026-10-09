# Fase 3 — Proposta automatica

**Aggiornamento dopo l'audit delle fasi 1–4:** suite completa con **224 test
superati**, inclusa l'app principale, nella `.venv` esistente. Il blocco pandas
descritto sotto è un risultato storico e non si è riprodotto nell'audit.
Resta la verifica visiva nel browser. Vedere [audit](audit-fasi-1-4.md).

## Resoconto storico della validazione della fase 3

Implementazione del 6 ottobre 2026, autorizzata con «fai la fase 3».
La validazione è stata ripresa il 6 ottobre usando, su richiesta dell'utente,
**la `.venv` già esistente**: il suo Python si avvia di nuovo.
**Tutti i 29 nuovi test della fase 3 passano**. La suite completa termina con
200 test superati, 7 fallimenti e 4 errori: Windows Code Integrity blocca ancora
la DLL nativa `pandas/_libs/tslib`, impedendo l'avvio dell'app principale nei
restanti test di integrazione. La verifica completa dell'app resta in sospeso.

## Flusso aggiunto

Nella calibrazione del singolo video, **Calibrazione automatica** analizza solo
il primo fotogramma già associato al record. Le proposte rimangono nello stato
della sessione; cambiare la scelta o inserire misure non ripete il rilevamento.
Non vengono chiamati YOLO, lo sfondo mediano o un worker di inferenza.

Aggiornamento dell'8 ottobre 2026: **Calibrazione automatica** applica subito la
prima proposta e nasconde i controlli per l'inserimento manuale. I punti sono
già disegnati sul fotogramma e si correggono trascinandoli. **Proposta da
esaminare** permette di cambiare direttamente fra le alternative, senza un
secondo pulsante di applicazione. Le correzioni restano conservate nei rerun.

L'applicazione conserva la proposta iniziale nella diagnostica, annulla
l'accettazione e sostituisce le dimensioni del riferimento precedente con
quelle disponibili nel nuovo candidato. I campi metrici sono precompilati
solo se esiste una distanza disponibile o si applica un preset scelto; i valori
sconosciuti restano vuoti. L'utente ha escluso valori sperimentali generici
precompilati. Origine e conferma delle misure restano esplicite.

**Torna alla selezione manuale** riapre coordinate numeriche e controlli ROI,
conservando il lavoro corrente. La ROI rimane indipendente; modificarla scarta
le proposte cercate nella regione precedente. Una bozza automatica riaperta
mostra direttamente i punti e le distanze salvati e rende disponibili i preset.

Se il riferimento non è visibile o la geometria è insufficiente, viene mostrato
un motivo e rimane disponibile il percorso manuale. Un risultato vuoto non
azzera i punti che l'utente aveva già inserito. L'applicazione del candidato
controlla anche l'hash dei pixel del fotogramma di riferimento.

## Rilevamento e diagnostica

- Maschera bianca/gialla con le primitive OpenCV/scikit-image già installate.
  La variante usata dal generatore conserva l'interno delle barre larghe;
  la maschera legacy mantiene il comportamento preesistente.
- Elaborazione ridotta a massimo 1280 pixel sul lato maggiore, configurabile
  entro limiti espliciti; punti e segmenti vengono restituiti nei pixel originali.
- Contorni chiusi, convessi e con quattro lati, verificati contro i bordi
  osservati. Vengono scartati lati troppo corti, forme non quadrangolari,
  bordi poco supportati e riferimenti tagliati dal frame o dalla ROI di ricerca.
- Segmenti LSD deduplicati e limitati a 160; raggruppamento robusto mediante
  ipotesi di punti di fuga omogenei e residui angolari. La procedura gestisce
  direzioni ruotate, outlier e punti di fuga finiti o all'infinito.
- Le due famiglie di lati opposti del contorno sono registrate separatamente.
  Le linee di sfondo non vengono combinate tramite estremi per inventare un
  rettangolo tra componenti differenti.
- Sono conservati versione `painted-contours-v1`, parametri, seed, hash del
  fotogramma, dimensioni di elaborazione, famiglie, motivi di scarto, supporto
  dei lati e geometria iniziale del candidato scelto.

La ricerca predefinita considera i due terzi inferiori del fotogramma;
una ROI esplicita permette di scegliere un'altra regione. Le coordinate dei
punti di fuga nella diagnostica sono riferite all'immagine elaborata, mentre
punti dell'editor e segmenti restituiti sono in pixel originali.

Una forma quadrangolare non dimostra che sia un rettangolo sul piano stradale.
Oggetti, superfici rialzate, ombre e occlusioni possono produrre falsi candidati.
Il tipo di segnaletica e la completezza del tratto richiedono la verifica umana.
La qualità della proposta valuta il supporto dei bordi; la qualità dell'editor
considera anche area e condizionamento della regione. Entrambe sono euristiche,
non stime dell'errore metrico reale. Un riferimento piccolo può non superare
la soglia richiesta per l'analisi.

## Preset USA espliciti

Verificati nel PDF ufficiale dell'[11ª edizione MUTCD con revisione 1,
dicembre 2025](https://mutcd.fhwa.dot.gov/pdfs/11th_Editionr1/mutcd11theditionr1hl.pdf),
indicato da [FHWA come edizione corrente](https://mutcd.fhwa.dot.gov/kno_11th_Editionr1.htm).
La sezione 3A.04 è a pagina PDF 579, numerazione interna 538; la sezione 3C.06
a pagina PDF 633, numerazione interna 592.

| Preset | Suggerimento sul singolo riferimento | Condizione |
| --- | --- | --- |
| Discontinuo ordinario | Lunghezza 3,048 m; larghezza normale nell'intervallo 0,1016–0,1524 m | Un solo tratto completo, senza l'intervallo né altri tratti; la guida ammette proporzioni differenti |
| Punteggiato di corsia normale | Lunghezza 0,9144 m; intervallo di larghezza normale | Distinto dalle estensioni nell'intersezione e dalle linee larghe |
| Estensione punteggiata di linea normale | Lunghezza 0,6096 m; intervallo di larghezza normale | Un solo tratto nell'intersezione, raccordo o rampa |
| Barra longitudinale di attraversamento | Larghezza nell'intervallo 0,3048–0,6096 m; lunghezza vuota | Una sola barra bianca, non l'intero attraversamento o una coppia di barre |

Nessun preset è preselezionato. Occorre verificare esplicitamente provenienza
USA, tipo di segnaletica, completezza del tratto e corrispondenza dei lati.
`World` e `UAE` non attivano preset; i metadati del dataset non vengono letti
per scegliere automaticamente misure o punti. La larghezza nell'intervallo
non riceve un valore medio automatico: può rimanere vuota o essere indicata
dall'utente. Il minimo di larghezza dell'attraversamento non è assunto come
lunghezza effettiva della barra.

Le misure suggerite hanno origine `standard`, fonte, ID del preset e conferma
inizialmente falsa. Correggerle annulla la conferma; confermarle non cambia
l'origine in `measured`. Cambiare preset cancella ipotesi incompatibili del
preset precedente. Le misure per cui il nuovo preset non propone un valore
rimangono disponibili, incluse le larghezze già misurate. Un rettangolo
allargato a più tratti non può ricevere la lunghezza di un singolo tratteggio.
Il riferimento deve mantenere sovrapposizione sufficiente con il candidato
originario. Per altri riferimenti resta disponibile l'inserimento manuale.

La fonte va confrontata con specifiche locali e data della ripresa. Il PDF
scaricato e il testo estratto sono in `outputs/phase3-verification`; hash SHA-256:
`f508594285e5fccc45714660cc6aba1f895ae88226449345582fbbc1ceb7ba07`.
Il parser PDF è stato usato in un ambiente temporaneo di uv, senza aggiungere
dipendenze al progetto.

## Compatibilità

L'API legacy `propose_calibration` mantiene la tupla di ritorno, usando il nuovo
generatore. In assenza di entrambe le dimensioni esplicite restituisce solo una
geometria canonica non accettata su quadrato unitario: eliminato il fallback
10×30. I candidati del nuovo editor non contengono una scala canonica.

La pipeline e il batch metrico non vengono estesi in questa fase. La voce batch
con omografia rimane disabilitata. La firma del codice cambia; il manifest del
batch precedente non va modificato per aggirare il controllo di ripresa.

## Verifiche eseguite e ancora necessarie

- Prove dirette sintetiche del generatore: vuoto e sole linee parallele danno
  zero candidati; barre complete, rettangolo delimitato, figura ruotata,
  prospettiva e segnaletica gialla producono candidati; barra tagliata dal frame
  scartata. Dopo la correzione delle famiglie globali, la prova diretta dell'API
  legacy sul rettangolo del test preesistente torna a produrre una proposta.
- Suite completa nella `.venv` esistente: **200 superati, 7 falliti, 4 errori in
  57,12 s**, nessun test saltato. Rapporto: `outputs/phase3-verification/pytest-full.xml`.
  Gli 11 test non superati dipendono dall'importazione bloccata di pandas
  nell'app principale o nella UI batch; gli errori `StopIteration` seguono
  l'assenza dei widget dopo il mancato avvio dell'app.
- Tutti i **25 nuovi test unitari delle proposte/preset e 4 nuovi AppTest**
  sono superati. Le regressioni mirate di coordinate, record, editor e GUI
  riportano 81 superati su 82; resta il test che carica l'app principale.
  Anche il test legacy del rettangolo passa nella suite completa.
- Corretto il timeout iniziale dell'AppTest della bozza a 20 secondi per
  consentire gli import al primo avvio; il test passa nella suite completa.
- Ruff superato, 97 file già formattati; `git diff --check` superato.
- I 33 file del batch esistente sono invariati, verificati confrontando gli hash
  SHA-256 prima e dopo le modifiche della fase 3.
- Aggiunti test parametrizzati per geometria, scala assente, resize, limiti,
  determinismo, punti di fuga, outlier, preset e cambiamento del preset.
- Aggiunti AppTest per alternative, modifica/salvataggio, risultato vuoto,
  conferma USA e provenienza, modifiche delle misure e invalidazione dopo ROI.
- Prova breve sui primi fotogrammi di quattro video locali: nessun errore,
  circa 0,45–0,52 s per video. Risultati e anteprime salvati in
  `outputs/phase3-verification/real-first-frames.json` e file JPEG associati.
  Candidati: `-2UPLUV7JLg_00.mp4` 0, `-6SQSDj8cYU_00.mp4` 0,
  `-7-vQ4obVwQ_00.mp4` 0, `-9oifpjUxxM_00.mp4` 1.
  Il campione è scelto per ordine dei nomi: non è una valutazione scientifica.
  Nessuna inferenza, calibrazione accettata o misura metrica generata.
  Il candidato nell'ultimo video è piccolo e richiede la verifica dell'utente;
  la sua presenza non attesta idoneità alla calibrazione metrica.
- Wheel costruita offline con il Python della `.venv`. Estratta in una cartella
  separata dai sorgenti e verificata tramite AppTest: due candidati, applicazione
  nell'editor e salvataggio della bozza con misure vuote e origine automatica.
  Le dipendenze restano quelle della `.venv` esistente, senza creare un nuovo
  ambiente di validazione. Rapporto: `outputs/phase3-verification/wheel-smoke.json`.
- Nessuna nuova verifica visiva nel browser è stata eseguita.

Ambiente effettivo: Python 3.11.15, Streamlit 1.63.0, pandas 3.0.5, pytest 9.1.1.
Nessuna ricreazione o installazione nell'ambiente durante questa ripresa.
Il blocco attuale è confermato negli eventi Code Integrity 3077/3033 sul file
`.venv/Lib/site-packages/pandas/_libs/tslib.cp311-win_amd64.pyd`, con policy
`{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. Il blocco precedente sull'avvio di Python
non si è ripresentato; non è stata determinata la ragione del cambiamento e non
sono state modificate le protezioni di Windows. Python ufficiale 3.12.10 già
installato è stato trovato con firma valida e provato, ma non è stato usato
per questi test dopo la richiesta di mantenere la `.venv` esistente.

Per completare la verifica serve un ambiente autorizzato a caricare pandas,
poi ripetere la suite completa e controllare l'editor nel browser. Comandi:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --tb=short --junitxml=outputs/phase3-verification/pytest-full.xml
.\.venv\Scripts\ruff.exe check src apps scripts tests
.\.venv\Scripts\ruff.exe format --check src apps scripts tests
git diff --check
uv build --wheel --offline --python .venv/Scripts/python.exe --no-managed-python --out-dir outputs/phase3-verification/dist
```

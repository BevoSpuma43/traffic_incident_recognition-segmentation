# Analisi sequenziale con checkpoint

Avviare l'interfaccia dalla radice del progetto:

```powershell
.venv\Scripts\python.exe -m streamlit run apps/streamlit_app.py
```

1. Selezionare **Modalita → standard_dataset analisi in batch - no omografia**.
2. Selezionare un modello locale `.pt` dalla cartella `models/`. Deve essere un
   modello YOLO di segmentazione; i pesi non vengono scaricati automaticamente.
3. Scegliere **standard_dataset** nel menu delle cartelle, oppure indicare il
   percorso di un'altra cartella. La ricerca include le sottocartelle.
4. Lasciare `dataset/metadata-real.csv` come CSV delle etichette e scegliere gli FPS.
   Tutti i modelli partono da 8 FPS, immagine 640 e configurazione
   `configs/accident-image.yaml`. Per confrontarli mantenere gli stessi parametri.
5. Premere **Prepara batch**, poi **Avvia batch**.

La voce **standard_dataset analisi in batch - con omografia** è per ora un
segnaposto: mostra un messaggio e non avvia elaborazioni. È riservata al futuro
confronto sullo stesso sottoinsieme con e senza omografia.

La GUI mostra nome e posizione del video (es. 12/150), percentuale del video,
numero di video completati, metriche parziali e download CSV. Il progresso del video
usa i timestamp del decoder e la durata del container; se questa non è disponibile
usa la durata dichiarata nei metadati. Il 100% viene assegnato dopo il salvataggio.

## Stop e ripresa

**Stop** viene controllato tra i fotogrammi: attende il termine dell'inferenza in
corso e il salvataggio delle clip. Non occorre attendere la fine del video.
**Riprendi** salta i video completati e riparte dall'inizio di quello interrotto.
Lo stato interno di ByteTrack e del detector temporale non viene serializzato a
metà video: ricominciare quel video evita di alterare i risultati con storie troncate.

L'analisi gira in un processo locale separato. Chiudere o aggiornare il browser
non la interrompe. Usare Stop per fermarla intenzionalmente. Dopo un arresto del
computer, riaprire la GUI, scegliere lo stesso modello, selezionare l'esperimento
salvato e premere Riprendi. Gli esperimenti sono scoperti su disco, senza dipendere
dalla sessione del browser. Un solo batch per volta viene eseguito nella cartella
`outputs/batches`; è possibile alternare i modelli con Stop e Riprendi.

Ogni risultato completo viene scritto atomicamente prima di aggiornare i CSV.
Tentativi interrotti o falliti restano nei log della pipeline ma non vengono
conteggiati. Un errore di decodifica o inferenza ferma il batch sul video interessato;
non viene trasformato in un falso negativo. I CSV si rigenerano dai risultati
completi alla ripresa, anche dopo un'interruzione durante l'esportazione.

L'esperimento conserva un manifest con elenco ordinato dei video, dimensioni e
date di modifica, hash dei pesi e dei metadati, configurazione e hash del codice.
Ogni risultato include anche SHA256 del video. Se cambiano input, modello,
parametri o codice, preparare un nuovo batch: i risultati precedenti restano
separati. Riprendi usa sempre la configurazione salvata, non i valori attuali
della barra laterale. I video aggiunti successivamente alla cartella entrano solo
in un nuovo batch.

## Metadati e protocollo di confronto

`metadata-real.csv` contiene 2.027 righe, una per video, con:

- `path`: ad esempio `real_videos/50uYv-SxT-o_00.mp4`;
- `type`: `rear-end`, `t-bone`, `single`, `head-on`, `sideswipe`;
- `accident_time`: istante dell'incidente in secondi dall'inizio;
- `accident_frame`, `no_frames`, `duration`, `height`, `width`;
- posizione normalizzata dell'incidente (`center_x`, `center_y`, `x1`…`y2`);
- attributi della scena e split `split_in_distribution` e `split_geo_aware`.

Il sottoinsieme viene associato per **nome file invariato**, perché la directory
di copia differisce da `real_videos/`. Nomi duplicati, video senza etichetta,
timestamp non validi ed etichette ambigue vengono rifiutati prima dell'inferenza.
La verifica locale del 5 ottobre 2026 ha trovato 150/150 corrispondenze in
`dataset/standard_dataset`, tutte positive.

Il detector non riceve tipo, timestamp o posizione dell'incidente annotato.
Le etichette servono solo per il confronto dopo l'analisi; la durata può servire
come fallback della barra di progresso. Non vengono valutate classificazione del
tipo di urto o localizzazione spaziale: la pipeline predice eventi di incidente.

### Report per video

Unità: video completo. Verità positiva se esiste un incidente annotato;
predizione positiva se il detector conferma almeno un evento, a qualsiasi istante.
Si calcolano TP, FP, FN, TN, accuracy, precision, recall e F1.

Tutti i video ACCIDENT reali forniti sono positivi: TN=0 e FP=0 in questo report.
L'accuracy coincide quindi con la recall, e la precision è 1 se esiste almeno
una predizione positiva. Questo report da solo non misura i falsi allarmi.
Per eventuali video negativi aggiuntivi occorrono righe esplicite con `type`
`normal`, `negative`, `no-accident` o `no_accident`, `accident_time` vuoto e durata
positiva. Un'etichetta assente non viene mai interpretata come negativa.

### Report temporale degli eventi

Unità: evento. Un rilevamento è TP solo se il suo `impact_time_s` differisce da
`accident_time` di **al massimo 1 secondo**, estremi inclusi, nello stesso video.
Ogni annotazione può corrispondere a un solo rilevamento: si sceglie quello con
minore errore temporale. Ulteriori allarmi, anche nella finestra, sono FP.
Un incidente senza corrispondenza è FN. Un allarme fuori finestra in un video
positivo genera quindi un FP e un FN nel report eventi, pur essendo TP per video.

Si calcolano TP, FP, FN, precision, recall e F1. TN e accuracy **non sono definiti**
per eventi puntuali senza un protocollo di finestre negative; rimangono vuoti,
non vengono sostituiti con numeri inventati. Il timestamp usato è quello stimato
dell'impatto, non il successivo istante di conferma.

Formule: accuracy=(TP+TN)/(TP+TN+FP+FN), precision=TP/(TP+FP),
recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Qualsiasi denominatore zero produce un
valore non definito (`null` nel JSON, cella vuota nel CSV).

## File prodotti

Ogni modello/configurazione/sottoinsieme ha una directory distinta:

```text
outputs/batches/<modello>-<identificatore>/
  manifest.json       configurazione e protocollo congelati
  checkpoint.json     stato e progresso corrente
  results/*.json      un risultato definitivo per video completato
  videos.csv          etichetta, predizione binaria, esito, conteggi eventi, timestamp
  events.csv          una riga per incidente rilevato: timestamp, score, TP/FP, errore
  metrics.csv         due righe: unit=video e unit=event
  metrics.json        metriche e numero di video positivi/negativi completati
  worker.log          diagnostica del processo
  pipeline/          run, traiettorie, feature, eventi e clip della pipeline
```

I CSV sono UTF-8, separati da virgole, con punto decimale. `videos.csv` include
anche video senza rilevamenti; `events.csv` ha solo la riga di intestazione se
non viene rilevato alcun evento. Gli elenchi di timestamp e ID sono stringhe JSON.
`metrics.csv` indica esplicitamente se il risultato è completo o parziale.
Il replay completo non viene generato automaticamente nel batch: rimangono log
e clip degli eventi, evitando una seconda decodifica di tutti i video.

Per confrontare un altro software usare esattamente gli stessi video e lo stesso
protocollo (presenza per video e matching uno-a-uno entro ±1 s). La GUI permette
anche di affiancare le metriche degli esperimenti salvati; confrontare soltanto
esperimenti completi sul medesimo sottoinsieme e con parametri equivalenti.

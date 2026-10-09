# Calibrazione preliminare automatica del batch — modalità sperimentale

La modalità **standard_dataset analisi in batch - con omografia** prepara le
calibrazioni prima dell'analisi degli incidenti. Si ipotizzano riferimenti USA
anche nei video girati altrove. Il risultato è una scala approssimativa per il
progetto universitario, non una misura certificata.

## Utilizzo

1. In **Preparazione calibrazioni**, scegli cartella e pesi locali per le auto.
2. Premi **Crea sessione di preparazione**, poi **Calibra tutti i video**.
3. Il worker esamina la cartella senza avviare il riconoscimento degli incidenti.
   Un riferimento mancante o un errore di calibrazione viene registrato e il
   worker passa al video successivo. Stop/ripresa conservano i risultati completati.
4. Il riepilogo mostra accettazione automatica o manuale, riferimento, dimensioni,
   qualità geometrica e motivazione. Una seconda tabella elenca i video da
   calibrare manualmente. Usa l'editor sottostante; per le proposte vuote premi
   **Torna alla selezione manuale**. In alternativa premi
   **Usa solo i N video calibrati automaticamente** per escludere esplicitamente
   quelli che richiedono una correzione.
5. Quando tutti i video hanno una calibrazione utilizzabile, premi
   **Passa all'avvio dell'analisi**. In **Analisi batch** puoi scegliere modello,
   FPS e CSV delle etichette, controllare di nuovo il riepilogo e premere
   **Avvia analisi con queste calibrazioni**.

La cartella dell'analisi coincide con quella della preparazione. Non si saltano
silenziosamente video non calibrati: l'esclusione avviene attraverso il pulsante
del campione. Non si passa automaticamente alle misure
in pixel. Le calibrazioni già confermate e compatibili vengono conservate; le
bozze vengono riprovate automaticamente, conservando le revisioni precedenti.
Un salvataggio manuale concorrente prevale sulla proposta del worker.

## Stesso campione con e senza omografia

Il pulsante **Usa solo i N video calibrati automaticamente** salva l'elenco esatto
dei video, gli hash dei contenuti, i motivi di esclusione e copie indipendenti delle
calibrazioni in `outputs/batch-selections/`. L'elenco comprende soltanto record
con accettazione automatica e qualità sufficiente. Ad esempio, su una preparazione
con 87 video automatici utilizzabili e 63 falliti, il campione contiene 87 video.

La selezione diventa comune alle modalità batch **con omografia** e **no omografia**.
Rimane attiva anche cambiando pagina, modello o riavviando Streamlit. La cartella
è vincolata al campione e compare l'elenco dei video selezionati, scaricabile in JSON.
La sola segmentazione utilizza gli stessi file, senza caricare l'omografia.
Il pulsante di selezione non avvia nessuna analisi.

Nuovi video aggiunti alla cartella o successive calibrazioni manuali non modificano
il campione salvato. Una nuova pressione del pulsante crea una nuova selezione
esplicita; gli esperimenti esistenti conservano il proprio elenco. Le metriche
usano solo i video del campione: gli esclusi non vengono contati come negativi o
errori. Manifest, impostazioni e CSV registrano `selection_id`; la lista degli
esperimenti in ciascuna modalità mostra quelli del campione attivo, evitando di
riavviare involontariamente un vecchio batch sull'intera cartella.

Le preparazioni già **completate** possono essere utilizzate dopo l'aggiornamento
del codice, senza ripetere la calibrazione, verificandone input e artefatti salvati.
La ripresa di una preparazione incompleta continua a richiedere il codice originale.

## Riferimenti e ipotesi

| Riferimento | Dimensioni adottate | Condizione / limite |
|---|---|---|
| Tratto discontinuo ordinario | 3,048 m × 0,1524 m (10 ft × 6 in) | Barra completa; il riconoscimento del tipo è geometrico e può sbagliare |
| Gruppo di tratti consecutivi | Dimensioni del tratto precedente; intervalli misurati nel piano rettificato | Almeno 3 tratti allineati; non si assegna 3,048 m all'intero gruppo |
| Attraversamento pedonale | Barra larga 0,4572 m (18 in), lunga **3 m per ipotesi del progetto** | Almeno 3 barre affiancate; la lunghezza non è fissata dal MUTCD |
| Corsia | 3,6576 m (12 ft), combinati con un tratto lungo 3,048 m | Due strisce tratteggiate adiacenti e allineate; la larghezza da sola non determina un'omografia |
| Auto tipo unica | Lunghezza 4,7 m; larghezza 1,8 m; altezza 1,5 m | Valori scelti, non una media statistica verificata del parco auto americano |

Fonti: [MUTCD 11th Edition, Revision 1, dicembre 2025](https://mutcd.fhwa.dot.gov/pdfs/11th_Editionr1/mutcd11theditionr1hl.pdf),
§§3A.04 e 3C.06; [FHWA, Designing a Road Diet](https://highways.dot.gov/safety/other/road-diets/road-diet-informational-guide/4-designing-road-diet).
Le larghezze delle corsie variano (ad esempio 10–12 ft); il rapporto guida
10/30 ft dei tratti e intervalli non è universale. La larghezza dell'intera
carreggiata non viene assunta uguale a quella di una corsia.

Tutte le distanze assegnate automaticamente hanno origine `experimental`, perché
sia la provenienza USA sia l'identificazione del riferimento sono ipotesi.
Non vengono applicate le conferme manuali dei preset dell'editor.

## Ricerca e scelta

La ricerca usa fino a 6 campioni nei **primi 10 secondi**, oppure la parte
disponibile di un video più corto. La decodifica è limitata a 900 fotogrammi per
passaggio; a frame rate estremamente elevati la finestra effettivamente letta
può quindi essere più corta. Le ricerche della segnaletica e delle auto possono
comportare due passaggi di decodifica. Sono conservati hash e diagnostica dei
fotogrammi. Le osservazioni successive vengono utilizzate solo se il confronto
con il riferimento iniziale verifica una camera sufficientemente stabile.

Ordine delle ipotesi: gruppi di segnaletica, corsia con tratto, automobili,
singolo tratto ambiguo. All'interno della stessa categoria prevale la qualità
geometrica. Le automobili vengono cercate come ripiego quando manca un gruppo
stradale o una corsia utilizzabile. YOLO viene caricato solo quando serve e
riutilizzato nel worker. Non si scaricano pesi durante la preparazione.

La dimensione metrica dei gruppi viene ricavata rettificando un elemento e
misurando i lati del rettangolo complessivo: resta corretta anche se i vertici
del gruppo hanno un orientamento diverso da quelli della singola barra.
La corsia richiede due tratti che, rettificati, presentino estremi allineati e
una distanza laterale compatibile con una corsia, non con barre pedonali vicine.
Le auto utilizzano il modello prospettico 3D già descritto nella
[guida dedicata](calibrazione-dalle-automobili.md): gli angoli dei bounding box
non vengono trattati come punti sulla strada.

La soglia geometrica della modalità sperimentale è **0,20**. Un singolo tratto
ha qualità limitata a 0,35; corsia a 0,50; gruppi e auto al massimo a 0,60.
Questi valori non sono probabilità di correttezza né stime dell'errore in metri.
La configurazione generale dell'analisi singola non cambia: la soglia specifica
viene salvata nella preparazione e nell'esperimento batch.

Non vengono mediate omografie discordanti né si afferma un consenso metrico tra
metodi. Se un candidato non supera la validazione geometrica, il worker prova
i successivi; in assenza di candidati utilizzabili richiede l'intervento manuale.

## Tracciabilità e ripresa

Il campo `automatic_acceptance: experimental_usa_v1` distingue l'accettazione
automatica dalla conferma manuale. Per compatibilità con la pipeline, un record
utilizzabile mantiene `status: confirmed`, ma le distanze **non** ricevono
`user_confirmed: true`. La validazione lega l'accettazione ai punti e alle misure
del candidato scelto. Qualsiasi modifica nell'editor revoca l'accettazione e
torna a una bozza. La conferma manuale richiede ancora entrambe le misure confermate.

Le sessioni salvano parametri, hash dei video, codice, pesi, proposte e immagini
diagnostiche in `outputs/calibration-preparations/`. Le revisioni delle
calibrazioni rimangono in `data/calibration/videos/`. L'analisi copia i record e
i riferimenti nel proprio esperimento; modifiche successive all'archivio non
ne cambiano la scala. La provenienza automatica è presente anche nei CSV.
Le vecchie sessioni completate restano utilizzabili; solo per ripetere o riprendere
la calibrazione con codice diverso occorre creare una nuova sessione.

## Verifica e limiti

Suite completa dopo l'aggiunta del campione condiviso: **368 test superati in 113,39 secondi**,
senza fallimenti o test saltati. Controlli Ruff e formattazione superati.
Report locale: `outputs/automatic-batch-tests/pytest-selection-full.xml`.

Verificati anche selezione persistente nelle due modalità, recupero in una nuova
sessione Streamlit, uguale elenco e conteggio nelle metriche, esclusione delle
conferme manuali, input modificati, indipendenza della sola segmentazione dagli
snapshot omografici e uso delle preparazioni completate con codice precedente.
La sessione locale da 150 video è stata verificata in sola lettura: **87** record
automatici utilizzabili, senza rieseguire la calibrazione o avviare l'analisi.

I test coprono conversione dei gruppi anche sotto trasformazioni prospettiche,
corsie, finestra di 10 secondi, ripiego sulle auto, errori per video, tabella dei
fallimenti, correzione manuale, blocco dell'avvio, stop/ripresa, conservazione
delle conferme, revoca dopo modifica e snapshot metrici indipendenti.

Prova locale del 9 ottobre 2026, con `yolo26s-seg.pt`, sui video
`_Etcfb7cUH8_00.mp4`, `-AztVDZ6cEE_00.mp4` e `-SNFUobKjoM_00.mp4`:
3/3 proposte dalle automobili hanno superato la validazione numerica.
Durate locali circa 18, 6 e 5 secondi, incluse decodifica e prima inizializzazione.
Artefatti locali: `outputs/automatic-batch-tests/smoke/`.
Non sono stati modificati gli archivi reali né avviate analisi degli incidenti.

Questa prova **non** verifica l'accuratezza fisica. Una scena con cavalcavia e
strada sottostante contiene piani differenti, per i quali una sola omografia
non è fisicamente valida. Sono possibili errori anche per curve, pendenze,
auto fuori sagoma, segnaletica diversa dalle ipotesi, ombre e superfici rialzate.
La proposta numericamente valida resta quindi una stima da controllare nel
riepilogo; il progetto non garantisce la calibrazione automatica di ogni video.

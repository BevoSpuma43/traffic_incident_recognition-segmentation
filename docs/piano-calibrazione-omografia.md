# Piano: calibrazione manuale e automatica per omografia

Data: 6 ottobre 2026. Stato: proposta di implementazione; le funzionalità descritte non sono ancora implementate. Questo documento integra le richieste per il video singolo e prepara la modalità batch con omografia.

## Obiettivo e flusso utente

In **Segmentazione + omografia**, prima dell'analisi:

1. Caricare automaticamente il primo frame decodificabile del video e cercare una calibrazione salvata compatibile.
2. Se presente, mostrarla sovrapposta al frame, con le distanze e le opzioni **Riutilizza calibrazione** e **Modifica calibrazione**.
3. Altrimenti consentire **Seleziona 4 punti** oppure **Calibrazione automatica**. Il secondo pulsante cerca riferimenti stradali e propone quattro punti, senza richiedere di copiare coordinate JSON.
4. Mostrare quattro maniglie numerate trascinabili sull'immagine. Consentire anche correzioni numeriche, annulla ultima modifica e ripristino della proposta.
5. Mostrare i lati associati ai campi **Larghezza (m)** e **Lunghezza (m)**. Ogni distanza proposta deve indicare origine e natura: misura conosciuta, ipotesi da standard o stima sperimentale.
6. Aggiornare l'anteprima dall'alto dopo il rilascio di un punto o la modifica di una distanza. Segnalare problemi geometrici e riferimenti insufficienti.
7. Con **Conferma e salva** salvare la configurazione; con **Avvia analisi** utilizzare esattamente quella configurazione.

L'automatismo è una proposta correggibile: se non trova riferimenti sufficienti, deve spiegare il motivo e lasciare disponibile la selezione manuale. Non promettere quattro punti affidabili in ogni scena.

## Geometria e limiti della misura

La prima versione utilizza i quattro vertici di un **rettangolo sul piano stradale**, che può apparire trapezoidale nel video. Le corrispondenze metriche sono `(0,0)`, `(L,0)`, `(L,P)`, `(0,P)`. Non basta selezionare un quadrilatero arbitrario e assegnargli due dimensioni: occorre verificare l'ipotesi rettangolare. Il modulo avanzato con coordinate metriche esplicite può restare disponibile.

L'omografia descrive un piano: curve verticali, strade su piani differenti, forti distorsioni dell'obiettivo e movimenti della telecamera richiedono gestione aggiuntiva. Per la prima versione, dichiarare il supporto a telecamera fissa e zona stradale approssimativamente planare. Cambi di inquadratura rendono la calibrazione non più applicabile.

Separare la ricerca della geometria dalla determinazione delle distanze. Due bordi paralleli di corsia, da soli, non determinano una calibrazione metrica completa. Anche una buona rettifica prospettica non rivela automaticamente la scala in metri. Le primitive OpenCV stimano la trasformazione a partire da corrispondenze, non forniscono la misura fisica della strada. [OpenCV: omografia](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html).

## Situazione del progetto

- `apps/streamlit_app.py`: estrae il primo frame, contiene moduli JSON e il pulsante **Proponi calibrazione dalle strisce**. La proposta richiede attualmente copia manuale dei punti.
- `calibration/assisted.py`, `lane_mask.py`, `line_fitting.py`: esistono maschere di segnaletica, skeleton, Hough e intersezioni. Il raggruppamento orizzontale/verticale nell'immagine e la scelta delle linee estreme vanno sostituiti con controlli adatti alla prospettiva e ai riferimenti riconosciuti.
- `calibration/homography.py`: contiene stima, validazione, caricamento e salvataggio. La proposta senza misure usa unità canoniche: queste non devono diventare metri implicitamente.
- `calibration/background.py`: lo sfondo mediano usa campioni distribuiti nel video. Il nuovo flusso deve usare il primo frame, evitando di usare automaticamente scene successive o successive all'incidente.
- `batch.py`: il worker attuale accetta soltanto coordinate immagine; servirà un'integrazione esplicita per caricare una calibrazione diversa per ciascun video.

## Dataset e riferimenti statunitensi

Verifica locale tramite associazione del nome video al campo `path` di `dataset/metadata-real.csv`: nel sottoinsieme attuale di 150 video, 96 hanno una località statunitense, 46 hanno `region=World` e 8 `region=UAE`. `World` non identifica un Paese. Il preset USA deve quindi dipendere dalla provenienza confermata e restare modificabile.

Esempi utili per suggerire misure, ricavati dal MUTCD 2023, Parte 3:

| Riferimento | Dimensione di riferimento | Uso proposto |
|---|---|---|
| Linea discontinua ordinaria | Tratto 10 ft = 3,048 m; intervallo 30 ft = 9,144 m; previste anche proporzioni analoghe | Suggerimento da confermare dopo aver riconosciuto il tipo di linea |
| Linea longitudinale normale | 4–6 in = 0,1016–0,1524 m | Intervallo, non dimensione esatta |
| Barra longitudinale di attraversamento | 12–24 in = 0,3048–0,6096 m | Riconoscimento del disegno e controllo di plausibilità |
| Larghezza attraversamento | Minimo 6 ft = 1,8288 m, con eccezioni | Non assumere che il minimo sia la larghezza effettiva |

Fonte: [FHWA, MUTCD 2023 Parte 3, sezioni 3A.04, 3C.03 e 3C.06](https://mutcd.fhwa.dot.gov/pdfs/11th_Edition/part3.pdf). Sono riferimenti per ipotesi, non misure dei video. Distinguere strisce discontinue ordinarie da quelle più corte nelle intersezioni; non fissare una larghezza universale di corsia.

FHWA indica come edizione corrente l'11ª con revisione 1, dicembre 2025. Prima di codificare i preset, verificarli nell'edizione e nelle specifiche locali pertinenti e registrare la fonte utilizzata. La data di ripresa può precedere la norma corrente. [Edizione corrente FHWA](https://mutcd.fhwa.dot.gov/kno_11th_Editionr1.htm).

## Librerie verificate e scelta proposta

Verifica dell'ambiente locale: OpenCV **4.14.0**, scikit-image **0.26.0**; Kornia non installata. OpenCV e scikit-image sono già dichiarate in `pyproject.toml`. Verificata la presenza delle API OpenCV elencate; non è ancora stato eseguito un benchmark del rilevamento automatico sul dataset.

| Libreria | Funzione nel progetto | Decisione |
|---|---|---|
| OpenCV | `createLineSegmentDetector`, `HoughLinesP`, contorni; `getPerspectiveTransform`, `findHomography`, `warpPerspective` | Base della prima versione, già disponibile |
| scikit-image | Componenti connesse, proprietà delle regioni e `ransac` per eliminare osservazioni incoerenti | Supporto alla geometria, già disponibile |
| Kornia | Stima robusta di omografie con tensori e integrazione PyTorch | Opzionale per sviluppi successivi; non necessaria per quattro punti su CPU |

Fonti: [OpenCV: rilevamento di segmenti](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html), [scikit-image: measure e RANSAC](https://scikit-image.org/docs/stable/api/skimage.measure.html), [Kornia: RANSAC](https://kornia.readthedocs.io/en/latest/geometry.ransac.html).

Queste librerie forniscono i componenti matematici e di visione; il riconoscimento dei riferimenti stradali e la loro associazione a distanze reali richiedono logica applicativa. Non è stato verificato un pacchetto che risolva in modo affidabile l'intero problema con una sola chiamata.

## Algoritmo di proposta automatica

1. **Preparazione:** primo frame, trasformazione nota tra coordinate originali e immagine di lavoro, contrasto locale, maschere bianche/gialle, possibile ROI stradale modificabile. Evitare assunzioni rigide sulla posizione della strada.
2. **Riferimenti candidati:** rilevare segmenti, contorni allungati e ripetizioni. Dare priorità a strisce pedonali e rettangoli stradali riconoscibili; usare bordi di corsia e tratteggi quando offrono vincoli sufficienti nelle due direzioni.
3. **Geometria robusta:** raggruppare linee per coerenza prospettica e punti di fuga, anche all'infinito, scartando outlier. Non usare semplicemente gli assi orizzontale/verticale dell'immagine. Non imporre intervalli regolari in pixel: la prospettiva li modifica.
4. **Quattro vertici:** costruire e ordinare candidati sullo stesso piano, associati allo stesso riferimento. Preferire riferimenti con buona estensione e supporto visivo; non ricavare un grande rettangolo da quattro linee senza relazione fisica.
5. **Distanze:** proporre solo dimensioni legate al riferimento effettivamente selezionato. Se un lato copre più tratti e intervalli, esplicitare la somma. Se una dimensione non è determinabile, lasciare il campo da compilare. Salvare fonte, ipotesi e valori originali dei suggerimenti.
6. **Valutazione:** separare qualità geometrica, disponibilità della scala e revisione umana. Un punteggio euristico non è una probabilità di correttezza. Mostrare i riferimenti usati e consentire di scegliere una proposta alternativa.
7. **Revisione:** trasferire i punti direttamente nell'editor trascinabile; ogni modifica invalida l'anteprima precedente e richiede ricalcolo prima del salvataggio.

Le dimensioni delle automobili sono una possibile estensione sperimentale, non il riferimento principale: variano tra modelli e le sagome visibili contengono punti sopra il piano stradale. Non usare gli angoli della bounding box come quattro punti a terra. Una futura stima da veicoli richiederà geometria 3D o punti di contatto appropriati, più osservazioni e una valutazione indipendente dell'errore.

Un eventuale modello dedicato alla segnaletica si valuterà solo dopo il benchmark del metodo classico, con dati annotati e pesi adeguati. Per confrontare modelli YOLO, mantenere la calibrazione uguale per tutti: anche eventuali maschere dei veicoli usate durante la calibrazione devono provenire da una procedura fissata.

## Editor e persistenza

Realizzare un componente grafico Streamlit riutilizzabile da video singolo e batch. Conservare punti, distanze e identità del video nello stato della sessione; gestire i trascinamenti nel browser e sincronizzare al rilascio per evitare continui ricalcoli. Verificare l'API del componente rispetto alla versione Streamlit installata prima dell'implementazione.

Salvare in `data/calibration/videos/<nome-video>.yaml`, ad esempio `clip_01.yaml` per `clip_01.mp4`, insieme a `clip_01.reference.jpg`. Per nomi ripetuti usare sottocartelle per dataset/percorso e, dove necessario, un'identità della sorgente: non sovrascrivere silenziosamente una calibrazione omonima.

Lo schema versionato deve includere:

- Identità e hash del video, percorso relativo, dimensioni originali, frame di riferimento e suo timestamp.
- Punti in pixel originali, corrispondenze metriche, distanze, omografia, ROI e convenzione degli assi.
- Metodo manuale/assistito, riferimento riconosciuto, preset e fonte, provenienza delle misure, proposta iniziale e correzioni confermate.
- Stato bozza/confermato/non valido, versione dell'algoritmo e parametri, diagnostica e data di salvataggio.

Prevedere salvataggio atomico e rilettura di verifica. Una modifica crea una nuova revisione; le copie già usate negli esperimenti restano immutabili. Una calibrazione con scala ipotizzata va esplicitamente identificata anche nei risultati, pur essendo espressa in metri.

Validare punti distinti, ordine coerente, quadrilatero convesso senza incroci, dimensioni positive, area sufficiente e matrice non degenere. Tenere coerenti punti, ROI e omografia quando il video viene ridimensionato. L'errore di riproiezione sui soli quattro punti può risultare quasi nullo anche con distanze sbagliate: aggiungere, quando disponibili, riferimenti indipendenti di controllo.

## Compatibilità batch e confronto degli esperimenti

1. Aggiungere un controllo preliminare con tabella: video, calibrazione mancante/bozza/confermata/incompatibile, fonte della scala e azione di modifica.
2. Aggiungere **Proponi calibrazioni mancanti**, con avanzamento, stop e ripresa. Salvare le proposte come bozze per video; l'editor consente di revisionarle una alla volta. La fase di preparazione deve essere riprendibile anch'essa.
3. Avviare il batch metricamente calibrato quando tutti i video selezionati hanno una configurazione confermata e compatibile. Non contare una calibrazione mancante come FN e non saltare video in modo implicito.
4. Copiare le calibrazioni confermate nella cartella unica dell'esperimento e registrarne gli hash nel manifest insieme a YAML, modello, dataset, CSV e FPS.
5. Caricare la copia corretta prima di ogni video. La ripresa riutilizza le copie dell'esperimento; modificare successivamente l'archivio centrale non cambia un batch già avviato.
6. Mantenere progressi, pulsanti stop/riprendi, checkpoint per modello e ripartenza dall'inizio del video incompleto, come nel batch attuale. Nuove calibrazioni per video già elaborati richiedono un nuovo esperimento.
7. Conservare i report per video e per evento, con tolleranza temporale **±1 secondo inclusivo**. Salvare anche stato e provenienza della calibrazione, errori e copertura del dataset.

Per confronti corretti utilizzare gli stessi video, FPS, pesi YOLO e area valutata. Evitare che il rettangolo di calibrazione restringa implicitamente la ROI solo nella modalità con omografia. Se un gruppo di video non è calibrabile, dichiararlo e confrontare anche il medesimo sottoinsieme nei due sistemi. Il confronto misura le due pipeline complete se cambiano anche regole o soglie dei detector.

## Fasi di realizzazione e verifica

| Fase | Lavoro | Criterio di completamento |
|---|---|---|
| 1 | Schema, archivio per video, editor manuale sul primo frame | Punti trascinabili, distanze, salvataggio/ricaricamento e identità video corretti |
| 2 | Pulsante automatico, candidati stradali e preset documentati | Proposte correggibili, nessuna falsa scala automatica nei casi privi di riferimenti |
| 3 | Revisione nel video singolo e collegamento alla pipeline metrica | Analisi usa la configurazione mostrata e confermata, con resize coerente |
| 4 | Preparazione batch, copie immutabili e ripresa | Due video con calibrazioni diverse usano le rispettive copie anche dopo stop/riprendi |
| 5 | Valutazione e documentazione | Report riproducibili e indicazione dei limiti osservati |

Test mirati: omografia nota e distanze indipendenti; coordinate dopo resize; trascinamento e cambio video; punti degeneri e scala assente; nomi duplicati e video modificati; proposta scartata su immagini senza segnaletica; interruzione durante preparazione e inferenza; isolamento degli esperimenti; regressione del batch senza omografia.

Prima di estendere l'automatismo a tutti i video, valutarlo su un campione rappresentativo di scene, qualità, illuminazione e provenienze. Misurare percentuale di proposte utilizzabili, tempo di correzione, errore geometrico e, dove esiste una misura indipendente, errore metrico. Non promettere in anticipo una percentuale di successo.

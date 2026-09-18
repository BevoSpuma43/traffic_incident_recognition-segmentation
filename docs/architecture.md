# Architettura e protocollo

Obiettivo: localizzare temporalmente possibili impatti stradali osservati da una sola camera
fissa. Non si tratta di previsione preventiva né di un sistema di allerta d'emergenza.

## Flusso

PyAV → istanze YOLO → ByteTrack → punto inferiore robusto della maschera → omografia
→ EMA dipendente dal delta temporale → velocità/accelerazione → griglia spaziale delle coppie
→ NORMAL/CANDIDATE/CONFIRMED/COOLDOWN o REJECTED → SQLite/JSON e clip H.264.

L'interfaccia riceve soltanto frame già elaborati. Per MP4 il decode è sequenziale e ogni
frame alimenta il ring buffer; l'inferenza viene campionata per timestamp. Per RTSP un thread
riempie una coda di dimensione 1–4 e aggiorna il buffer prima di scartare frame vecchi.
La scrittura MP4 avviene in un worker dedicato con numero massimo di lavori.

## Geometria e affidabilità

La calibrazione memorizza dimensioni immagine, punti sorgente/destinazione, H, ROI, unità,
errore di riproiezione, confidenza, accettazione e riferimento visuale. Gli errori
su punti usati per stimare H non costituiscono una misura indipendente: utilizzare
`held_out_error` con punti misurati non usati nella stima.

La proposta dalle strisce è una baseline classica: LAB/HSV, top-hat, ROI, skeleton,
Hough, due famiglie orientative, stima robusta del punto di fuga e intersezioni estreme.
La confidenza è deliberatamente limitata a 0.79: la corrispondenza fisica delle linee deve
essere controllata. Non è implementata l'accettazione metrica completamente autonoma.

L'estrazione del punto usa la componente connessa più grande, quantile inferiore robusto
e mediana della fascia bassa. Maschere tagliate hanno qualità ridotta. Le osservazioni
mancanti non diventano misure reali: non vengono emesse accelerazioni da tracce predette.
Una riapparizione dopo un gap o salto implausibile azzera la storia affidabile.

Il controllo movimento usa feature ORB e registrazione affine RANSAC escludendo le
maschere dei veicoli; due registrazioni coerenti oltre soglia invalidano H in memoria.
La pipeline continua la visualizzazione, ma sospende gli eventi. Occorre poi ricalibrare.
Scene prive di punti statici sufficienti possono impedire il rilevamento dello spostamento.

## Coordinate immagine senza calibrazione

Con `events.coordinate_mode: image` si usa un riferimento identita in pixel,
con confidenza metrica zero. Il detector dedicato normalizza le soglie rispetto
alla diagonale dei box: non riutilizza soglie espresse in metri. La modalita
metrica mantiene i propri controlli e il classificatore metrico non accetta
questi dati. Il ridimensionamento conserva i timestamp e viene registrato nel run.

Il benchmark ACCIDENT usa un campione limitato e congelato prima dell'inferenza:
vedere [real-videos.md](real-videos.md).

## Decisione

Una coppia candidata deve avere traiettorie affidabili e convergenza con distanza
minima prevista plausibile oppure prossimità con forte decelerazione. L'attivazione
richiede almeno tre osservazioni e una durata minima. La conferma richiede contatto
plausibile e arresto post-impatto mantenuto nel tempo. La sola prossimità non basta.
La regola singolo veicolo richiede cambiamento brusco di direzione, decelerazione e arresto;
è conservativa e non copre tutti gli impatti reali.

Il cooldown è per ID coinvolti: blocca coppie sovrapposte relative allo stesso evento.
ID switch completi dopo un urto possono comunque generare duplicati; questo limite va
misurato con annotazioni di tracking. Il classificatore opzionale può modificare lo score
dei candidati, ma non rimuove il vincolo post-impatto.

## Misure predefinite

| Area | Misura | Obiettivo/protocollo |
| --- | --- | --- |
| Runtime | FPS elaborati / secondi reali | 5–10 sul video e macchina target |
| Evento | precision, recall, F1 | Matching uno-a-uno entro ±2 s dall'impatto |
| Evento | falsi allarmi/ora | Durata totale, inclusi video normali |
| Ritardo | conferma − impatto annotato | Media e p95, valori negativi conservati |
| Memoria | RSS massimo campionato | History, frame e job esplicitamente limitati |
| Geometria | errore su punti indipendenti | Unità della calibrazione |
| Segmentazione | mAP50–95 delle maschere | Validazione Ultralytics su split per camera |
| Tracking | IDF1/HOTA e ID switch | Richiede annotazioni; non ancora valutato |

Le distribuzioni di latenza conservano gli ultimi 10.000 campioni; la media usa il totale
di tutta l'esecuzione. I tempi di rendering comprendono il callback UI. La scrittura clip
è asincrona e il tempo totale comprende il flush finale. La metrica end_to_end misura il
frame elaborato dalla segmentazione al callback, con decode e buffer riportati separatamente.

Il piano sperimentale finale richiede telecamere/sorgenti separate per train, validation,
test. Nessuna soglia viene scelta sul test. Hardware e versioni sono registrati per run.

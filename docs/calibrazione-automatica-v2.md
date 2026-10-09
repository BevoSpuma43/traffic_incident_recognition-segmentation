# Calibrazione automatica: barre incomplete, gruppi e fotogrammi iniziali

Implementazione dell'8 ottobre 2026. Algoritmo: `painted-patterns-v2`.

## Uso nell'editor

**Calibrazione automatica** continua a cercare sul primo fotogramma, ma ora
include barre ricostruite e gruppi di segnaletica. Il menu **Proposta da esaminare**
distingue barra dipinta, rettangolo delimitato, barra ricostruita, gruppo di barre
affiancate e gruppo di tratteggi allineati. Le proposte restano trascinabili.

Se un veicolo copre la segnaletica, aprire **Ricerca su più fotogrammi**, impostare
**Secondi iniziali da esaminare** e premere **Cerca nei primi secondi**. Il valore
iniziale è 5 secondi, con massimo 15. Scegliere un intervallo prima dell'incidente;
il programma non legge etichette di incidente per determinare questo intervallo.

La ricerca campiona fino a sei fotogrammi, compreso quello iniziale. Un candidato
trovato successivamente mostra il tempo e il fotogramma in cui è visibile, con
i suoi quattro punti sovrapposti. Il fotogramma principale dell'editor, la ROI e
l'identità del video restano quelli originali. La geometria viene mantenuta nel
sistema di coordinate del primo fotogramma.

La diagnostica mostra gli scarti del primo frame, i campioni aggiuntivi verificati
e le osservazioni prive di conferma temporale. Punti, misure e correzioni rimangono
nello stato della sessione: i normali rerun non ripetono il rilevamento.

## Metodi aggiunti

### Recupero della vernice incompleta

- Approssimazione del contorno convesso per barre con bordi consumati o piccole
  lacune; possibile unione di due frammenti allineati e di larghezza simile.
- Evidenza su tutti e quattro i bordi: supporto minimo 50%, medio 70%, lacuna
  massima 35% del lato. Riempimento della maschera almeno 70%.
- Il colore originale deve restare compatibile con vernice bianca/gialla:
  il solo aumento del contrasto non basta a trasformare un oggetto grigio in
  una barra ricostruita.
- Restano esclusi riferimenti tagliati dalla ROI o dal frame e geometrie troppo
  piccole o degeneri. Budget limitati per contorni, unioni e proposte.
- I vertici ricostruiti sono esplicitamente indicati; qualità euristica massima
  0,65. Il punteggio non misura l'accuratezza metrica.

### Gruppi di segnaletica

Si cercano almeno tre tratti compatibili. Prima si rettifica la prospettiva
rispetto a un tratto, poi si controllano allineamento degli estremi, dimensioni,
orientamento e regolarità della spaziatura. Le distanze in pixel nell'immagine
originale non vengono assunte costanti.

Barre affiancate e tratteggi consecutivi producono candidati distinti, con i bordi
dei membri nella diagnostica. Quando ci sono molti tratti completi, uno dei
posti disponibili viene riservato a una proposta di gruppo. Questa è un'ipotesi
geometrica: non costituisce riconoscimento semantico certo dell'attraversamento
o del tipo normativo di linea.

### Ricerca temporale

I fotogrammi aggiuntivi devono avere dimensioni coerenti e corrispondenze ORB
distribuite nello sfondo. La stima affine robusta deve indicare spostamento
massimo della camera entro 1,5 pixel originali, almeno 16 inlier, frazione inlier
60% e copertura dello sfondo 12%. Mancanza di dettagli, tagli, pan e zoom
impediscono di usare il campione. Non viene trasferita la calibrazione tramite
un'omografia arbitraria fra fotogrammi.

Una nuova proposta deve essere osservata nello stesso punto in almeno due
fotogrammi verificati. Questo riduce gli agganci a veicoli in movimento, ma
non distingue tutti gli oggetti statici rialzati dalla segnaletica a terra.
Se nessun campione aggiuntivo è verificabile, rimangono disponibili le proposte
del primo frame, indicate come prive di conferma temporale.

Il backend consente da 2 a 12 campioni e legge al massimo 900 frame decodificati.
Vengono verificati il primo frame e l'assenza di modifiche al file durante la
ricerca. La scansione si ferma al termine dell'intervallo, del video o del budget.
Le immagini di evidenza rimangono nella sessione; nel record sono salvati tempi,
indici, hash e controlli di stabilità, senza sostituire il riferimento originale.

## Misure e compatibilità

Nessuno dei nuovi metodi assegna metri ai pixel. Larghezza e lunghezza restano
da specificare e confermare. I preset di un singolo tratto sono bloccati sui
gruppi e sulle barre ricostruite. Un gruppo non riceve la misura di una singola
barra. I preset preesistenti rimangono disponibili sulle barre complete.

L'API legacy, lo script di calibrazione e la preparazione batch ricevono i nuovi
metodi geometrici tramite `generate_proposals`. La ricerca temporale è una scelta
esplicita nell'editor condiviso: non viene attivata silenziosamente nei worker
batch o nel campionamento scientifico dei primi frame. Il formato dei record
resta compatibile. Applicare una proposta archiviata conserva la versione
dell'algoritmo che l'ha generata.

## Verifica e limiti osservati

I test coprono vernice incompleta, frammenti, curve, tagli, lacune troppo ampie,
gruppi in prospettiva, allineamenti incompatibili, assenza di scala, divieto dei
preset, camera mossa, zoom, tagli, superfici in movimento, intervallo temporale,
ROI, identità del video, salvataggio metrico, trascinamento e riapertura nell'editor.

Il confronto diretto sui quattro video del precedente controllo locale è in
`outputs/calibration-v2/real-comparison.json`, con anteprime JPEG. Il confronto
disattiva recupero e gruppi per la baseline e usa 3 secondi / 4 campioni per
la ricerca temporale. Non modifica archivi di calibrazione o risultati di analisi.

| Video | Baseline, primo frame | V2, primo frame | V2, ricerca temporale |
|---|---:|---:|---:|
| `-2UPLUV7JLg_00.mp4` | 0 | 0 | 0 |
| `-6SQSDj8cYU_00.mp4` | 0 | 0 | 0 |
| `-7-vQ4obVwQ_00.mp4` | 0 | 2 | 2 |
| `-9oifpjUxxM_00.mp4` | 1 | 5 | 5 |

Verifica finale: **326 test superati**, Ruff e formattazione dei sette file Python
interessati superati, `git diff --check` superato. Rapporto della suite:
`outputs/calibration-v2/pytest-final.xml`. La UI è verificata tramite AppTest,
inclusi ricerca temporale, immagine di evidenza, correzione dei punti e salvataggio;
le anteprime dei video reali sono state ispezionate visivamente. Non è stata
eseguita una nuova verifica dell'interfaccia in un browser.

La verifica visiva ha evidenziato che cartelli, scritte sovrapposte e guardrail
possono ancora generare candidati: persistenza e geometria non dimostrano che
un oggetto sia sul piano stradale. Una ROI corretta e la revisione dei punti sono
ancora necessarie. Il conteggio delle proposte non è una misura di calibrazioni
valide; manca una valutazione metrica con distanze indipendenti sui video reali.

È disponibile un ulteriore metodo sperimentale da sagome di automobili e
dimensioni ipotizzate, descritto in [calibrazione dalle automobili](calibrazione-dalle-automobili.md).
Restano da sviluppare un filtro semantico del piano stradale, la calibrazione
da planimetrie e la gestione di telecamere mobili o piani diversi.

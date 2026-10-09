# Calibrazione sperimentale dalle automobili

Implementazione dell'8 ottobre 2026, algoritmo `mean-vehicle-cuboid-v1`.
Disponibile nell'editor condiviso per video singoli e batch. Si avvia
esplicitamente e non sostituisce la ricerca della segnaletica.

## Uso

1. Aprire **Calibrazione dalle automobili · sperimentale** nella scheda
   **Calibrazione**. Se possibile, delimitare prima la ROI sul tratto stradale
   interessato, escludendo parcheggi e strade con orientamenti diversi.
2. Impostare lunghezza, larghezza e altezza dell'auto tipo. I valori iniziali
   **4,7 × 1,8 × 1,5 m** sono un'ipotesi modificabile per questo progetto,
   **non una media statistica documentata del parco auto americano**.
3. Scegliere un modello di segmentazione locale COCO, per esempio
   `models/yolo26n-seg.pt`, e l'intervallo iniziale da osservare: 5 secondi
   predefiniti, massimo 15; 0 usa solo il primo fotogramma. Scegliere un
   intervallo prima dell'incidente.
4. Premere **Usa automobili (sperimentale)**. Se disponibile, la proposta
   inserisce quattro punti e le dimensioni della porzione stradale stimata.
   Queste dimensioni **non sono la lunghezza e larghezza di una singola auto**.
5. Controllare le sagome 3D azzurre e il rettangolo sul piano stradale.
   Confermare valore e origine sperimentale delle due distanze e il controllo
   visivo, quindi salvare con il normale flusso dell'editor.

La ricerca viene eseguita solo al clic; modificare i valori richiede una nuova
ricerca. Un normale rerun non ripete l'inferenza. Parametri personalizzati e
intervallo sono ripristinati alla riapertura di un record prodotto dal metodo.
In assenza di una proposta rimangono disponibili riferimenti stradali e
selezione manuale; la diagnostica spiega lo scarto e mostra le sagome valutate.

## Metodo

Il rilevatore usa esclusivamente pesi già presenti sul computer. Cerca auto,
bus e camion; solo la classe COCO `car` (ID 2) può fornire la scala dell'auto
tipo. Le altre classi servono anche per evitare sagome sovrapposte. Si escludono
auto piccole, tagliate, poco confidenti, con maschera debole o punto di contatto
fuori dalla ROI. Senza ROI si usa come filtro grossolano la zona sotto il primo
terzo dell'immagine: questo non identifica semanticamente la strada.

Si campionano fino a sei fotogrammi e si decodificano al massimo 900 frame.
Il primo deve coincidere con il riferimento dell'editor; i successivi devono
superare il controllo dello sfondo e della camera fissa già utilizzato dalla
ricerca temporale. Le rilevazioni sono ricondotte ai pixel originali.
Si selezionano fino a 12 osservazioni distribuite nella scena; rilevazioni
statiche praticamente coincidenti vengono deduplicate. Una stessa auto in
movimento può fornire più osservazioni: il conteggio non indica veicoli distinti.

Ogni auto viene modellata come un parallelepipedo delle dimensioni ipotizzate.
Un'ottimizzazione robusta da sei inizializzazioni stima focale, inclinazione e
altezza della camera, orientamento comune delle auto e posizioni sul piano.
Le proiezioni dei parallelepipedi vengono confrontate con le sagome rilevate.
**I quattro angoli del bounding box non vengono trattati come punti a terra**:
il modello considera esplicitamente l'altezza dell'auto. Dal piano `Z = 0`
si ricavano l'omografia e un rettangolo virtuale con dimensioni metriche.

Il metodo richiede almeno cinque osservazioni utili con diverse dimensioni
apparenti e sufficiente distribuzione spaziale. Rifiuta dati concentrati o
allineati, mancata convergenza, scarsa compatibilità delle sagome, soluzioni
forzate ai limiti della camera, attraversamento dell'orizzonte e alternative
con scale troppo discordanti. Il rettangolo deve avere area utile, rientrare
nel fotogramma e superare i controlli della ROI.

L'impostazione prende spunto dall'uso di modelli 3D per la scala in
[Sochor, Juránek e Herout, *Traffic Surveillance Camera Calibration by 3D Model Bounding Box Alignment for Accurate Vehicle Speed Measurement*](https://arxiv.org/abs/1702.06451).
Questa implementazione usa un parallelepipedo semplificato e proprie ipotesi
di camera; non riproduce l'algoritmo completo o le prestazioni dello studio.

## Origine e limiti delle misure

La scala è archiviata come `experimental`, con dimensioni ipotizzate, fonte,
versione del metodo, modello e hash dei pesi, osservazioni, tempi, hash dei
fotogrammi, controlli della camera e risultati della stima. Le distanze
richiedono conferma dell'utente. La qualità euristica resta al massimo 0,60
anche dopo la revisione nell'editor: non è una probabilità di correttezza o
un intervallo di errore metrico. Il formato dei record resta compatibile e
il riferimento rimane il primo fotogramma originale.

Ipotesi principali: strada piana, camera fissa senza rollio, pixel quadrati,
punto principale al centro dell'immagine, auto con orientamenti paralleli
e dimensioni simili al modello. Curve, rampe, zoom, prospettiva debole,
distorsione ottica, SUV/pickup di dimensioni diverse, parcheggi e svolte
possono far fallire il metodo o produrre una proposta sbagliata.

Una buona corrispondenza delle sagome non dimostra che la camera sia corretta.
La verifica visiva sui video reali ha evidenziato rettangoli che comprendono
spartitraffico o aree laterali: il sistema non possiede ancora un filtro
semantico del piano stradale. Usare la ROI e verificare la planimetria proposta.
Cambiare proporzionalmente tutte le dimensioni del modello cambia
proporzionalmente la scala ricavata. L'errore della stima delle sagome non va
interpretato come errore sulle distanze o sulle velocità.

## Verifica locale

I test geometrici usano una proiezione 3D indipendente del modello di fitting
e distanze a terra tenute fuori dalla stima. Verificano recupero della scala
nelle condizioni ideali, sensibilità alle dimensioni ipotizzate e rumore
delle rilevazioni. Sono inoltre coperti scarti geometrici, ROI, identità del
video, esclusione di bus/camion, assenza dei pesi, diagnostica di prospettive
forzate, mancata convergenza, conferma, salvataggio e riapertura nell'editor.

Controllo con YOLO26n-seg locale, intervallo di 5 secondi, dimensioni
predefinite, nessuna ROI aggiuntiva:

| Video | Osservazioni selezionate | Proposte |
|---|---:|---:|
| `-2UPLUV7JLg_00.mp4` | 2 | 0 |
| `-6SQSDj8cYU_00.mp4` | 4 | 0 |
| `-7-vQ4obVwQ_00.mp4` | 11 | 0 |
| `-9oifpjUxxM_00.mp4` | 3 | 0 |
| `-AztVDZ6cEE_00.mp4` | 12 | 0 |
| `-RrDtLjWsT4_00.mp4` | 8 | 0 |
| `-RE3XseZINA_00.mp4` | 0 | 0 |
| `-Qt5bDJNT84_00.mp4` | 10 | 1 |
| `-qmYW4S3Xxo_00.mp4` | 12 | 1 |
| `-PpjzmhI_PE_00.mp4` | 6 | 1 |
| `-PpBteU0p3Q_00.mp4` | 8 | 0 |
| `-oreKuCoQnI_00.mp4` | 3 | 0 |
| `-NgnSm_oEB4_00.mp4` | 7 | 1 |
| `-jZ5Rlhzzi4_00.mp4` | 2 | 0 |
| `-i9bRJWMtTo_00.mp4` | 9 | 0 |
| `-gbcp_G6NAE_00.mp4` | 1 | 0 |

**4 proposte su 16 video**, non 4 calibrazioni metricamente validate.
I primi sei video provati su 3 secondi non avevano prodotto proposte.
Rapporti completi: `outputs/vehicle-calibration/real-checks.json`,
`default-checks.json`, `extended-checks.json`, `default-16-summary.json`;
anteprime JPEG e script di riproduzione nella stessa cartella. Questi controlli
non hanno scritto archivi di calibrazione o avviato analisi degli incidenti.
Le quattro anteprime con proposte sono state ispezionate visivamente.
La UI è verificata tramite AppTest; non è stata eseguita una prova in browser.
Manca una valutazione metrica con distanze indipendenti sui video reali.

Verifica finale: **348 test superati** nella suite completa (105,49 secondi),
Ruff e formattazione dei sei file Python interessati superati,
`git diff --check` superato. Rapporto:
`outputs/vehicle-calibration/pytest-final.xml`.

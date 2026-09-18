# Verifica su video reali ACCIDENT - 15 settembre 2026

Pipeline eseguita integralmente su 10 video CCTV reali (due per classe), senza training o tuning sul test. Il campione e stato congelato prima dell'inferenza.

- YOLO26n-seg preaddestrato, PyTorch CPU, ByteTrack, regole temporali in coordinate immagine.
- 2.117 frame analizzati; 264,68 secondi di video secondo i container.
- Durata nei metadati: 264,26 secondi; differenza dovuta alla rappresentazione della durata.
- Tempo pipeline: 195,30 s; caricamento modello condiviso: 1,38 s.
- FPS aggregati: 10,84; per video: 6,37-18,87; RAM massima: 771,94 MiB.
- TP=0, FP=2, FN=10; precision=0, recall=0, F1=0, con matching entro +/-2 s.
- Nessun tratto PAUSED, nessun errore runtime. Calibrazione metrica disponibile: zero scene.
- Nessun ritardo di rilevamento calcolabile: non ci sono veri positivi.

**Il software elabora i video reali, ma la baseline di rilevamento e insufficiente su questo campione.** Il test non dimostra affidabilita operativa.

| Clip | Tipo | TP/FP/FN | FPS |
|---|---|---|---:|
| JTIfOGQql0A_00 | head-on | 0/1/1 | 12.10 |
| orZsH8JuEE4_00 | head-on | 0/0/1 | 7.69 |
| JC52fZUaMio_00 | rear-end | 0/0/1 | 18.87 |
| afVZZ3v87cc_00 | rear-end | 0/1/1 | 11.00 |
| 7ORiNi_LtxY_00 | sideswipe | 0/0/1 | 6.37 |
| sxonKsim9qE_00 | sideswipe | 0/0/1 | 9.14 |
| OjosPgV4Lss_00 | single | 0/0/1 | 17.05 |
| FOPldt8txP4_00 | single | 0/0/1 | 11.54 |
| n77TaXyADls_00 | t-bone | 0/0/1 | 12.78 |
| 50uYv-SxT-o_00 | t-bone | 0/0/1 | 16.38 |

## Diagnostica

In tre clip lo score raggiunge 0,95-1,00 entro un secondo dall'impatto annotato, ma il detector non completa la conferma temporale. Nella clip notturna JC52fZUaMio_00 non ci sono traiettorie che superano i filtri di eta e qualita nella stessa finestra. Questi conteggi riguardano la scena intera, non identita annotate dei veicoli incidentati.

I due allarmi sono temporalmente fuori bersaglio: JTIfOGQql0A_00 a 27,53 s (annotato 8,267 s) e afVZZ3v87cc_00 a 15,79 s (annotato 3,167 s). Non vengono promossi a veri positivi per la sola presenza di un incidente nella clip.

## Verifiche e limiti

- Suite completa precedente: 46 test passati; due nuovi test aggiuntivi passati (preset reale Streamlit e rifiuto del classificatore metrico in modalita immagine).
- Tutte le 10 anteprime e le 2 clip evento decodificate; 2.117 frame anteprima e ordine dei PTS verificati. Una clip evento e troncata dalla fine del video, come segnalato nei metadati.
- Metriche ricalcolate dalle predizioni e confrontate con il report; hash del codice di inferenza verificati.
- Campione bilanciato e selezionato per contesto, non rappresentativo del traffico ordinario.
- Nessun video negativo: il tasso di falsi allarmi operativo non e disponibile.
- Nomi sorgente approssimativi, senza camera_id fisici verificati; overlap del pretraining con video online ignoto.
- Nessuna misura metrica della strada, annotazione di tracking o segmentazione di istanza usata per la valutazione.

Occorrono training e selezione delle soglie su dati separati, negativi realistici e misure del tracking prima di una nuova valutazione su test non utilizzato per il tuning.

## Riproduzione e artefatti

Istruzioni: [real-videos.md](real-videos.md). Il benchmark e limitato per default a 10 video, con un massimo esplicito di 20.

- [Report dettagliato locale](../outputs/accident-sample/report.md)
- [Diagnostica e grafici](../outputs/accident-sample/diagnostics.md)
- [Anteprima del campione](../outputs/accident-sample/contact-sheet.jpg)
- [CSV per video](../outputs/accident-sample/per-video.csv)
- [Manifest con checksum](../outputs/accident-sample/sample.json)

Pesi SHA256: 361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f.
I dati, i pesi e gli output generati restano esclusi da Git; questo riepilogo conserva i risultati essenziali.


## Correzione del 18 settembre 2026: JTIfOGQql0A_00

La segnalazione dell'utente ha confermato un falso positivo della baseline a
27,533 s; il CSV annota l'urto reale a 8,267 s. La correzione del detector:

- richiede un avvicinamento relativo significativo per usare il TTC, escludendo
  piccoli movimenti delle bbox di mezzi fermi;
- considera anche la frazione di sovrapposizione sul veicolo piu piccolo,
  evitando che la grande bbox del camion nasconda il contatto col pickup;
- fa partire dall'impatto una finestra limitata per la conferma successiva.

Nella nuova esecuzione completa, senza usare etichette o timestamp del dataset
come input del detector: un evento a **8,2667 s**, confermato a **9,9333 s**, ID 4 e
16. Nessun evento a 27 s. Rigenerati clip evento e replay completo di 446 frame.
Suite: **58 test superati**, inclusi jitter da fermi, dimensioni diverse e scadenza
della conferma. [Report della correzione](../outputs/impact-diagnosis/correction.json).

La rivalutazione del detector sulle traiettorie gia salvate dei dieci video
produce TP=1, FP=1, FN=9; il falso allarme di afVZZ3v87cc_00 rimane. Questo e un
controllo di regressione successivo a una correzione suggerita dal test set, non
una nuova valutazione indipendente. I risultati del 15 settembre sopra restano
il riferimento storico della versione precedente.


## Correzione del 18 settembre 2026: n77TaXyADls_00

Il video contiene un urto annotato a **4,471133 s**, ma il profilo rapido
YOLO26n-seg a 8 FPS non conserva il camion nel tracking vicino all'impatto e
produce zero eventi. Dopo l'urto i veicoli continuano a muoversi: la sola conferma
basata sull'arresto non e sufficiente.

E stata aggiunta la modalita **Qualita analisi -> Accurata** nell'interfaccia:
YOLO26m-seg, immagine di inferenza 640, campionamento richiesto 15 FPS. Il detector
puo conservare per un secondo un contatto osservato quando rileva frenata e
deviazione su due veicoli distinti. La conferma richiede un rallentamento persistente
di un veicolo ancora osservato nella regione del contatto. Osservazioni predette,
assenza di entrambi i mezzi e rallentamenti transitori non bastano.

Nuova esecuzione completa `9ed089f4c8de4e59`, senza timestamp o etichette del
dataset come input dell'analisi:

- un evento a **4,6046 s**, confermato a **5,338667 s**, ID 2 e 11;
- errore temporale dell'impatto **+0,133467 s** rispetto all'annotazione;
- 141 frame analizzati, 155 decodificati, 57,89 s su CPU (2,44 FPS effettivi);
- replay completo di 155 frame, PTS crescenti e segnale d'impatto verificati;
- clip evento salvata, con finestra successiva troncata dalla fine del sorgente;
- **69 test superati** e Ruff senza errori.

[Report della correzione](../outputs/impact-diagnosis/n77TaXyADls_00/correction.json).
[Replay annotato](../outputs/real-videos/runs/9ed089f4c8de4e59/annotated.mp4).

Il controllo del nuovo detector sulle traiettorie nano gia salvate dei dieci video
mantiene i risultati della correzione precedente: TP=1, FP=1, FN=9. JTIfOGQql0A_00
resta a 8,2667 s; il falso allarme di afVZZ3v87cc_00 rimane. Per n77TaXyADls_00 e
necessario il profilo Accurata: il profilo rapido continua a perdere il camion.
Non e stata eseguita una nuova valutazione indipendente del modello medio su
tutto il campione: questi sono controlli di regressione su video segnalati.

Pesi YOLO26m-seg SHA256:
`16b636f04e8fb6a325b3370f22dc5e5535ff473e384f4d041fd28d788f6ee9f5`.


## Correzione del 18 settembre 2026: 7ORiNi_LtxY_00

Anche il profilo Accurata precedente produceva zero eventi: YOLO riconosceva
l'auto bianca, ma il criterio di associazione basato sulla sovrapposizione
frammentava la traccia durante l'ingresso veloce nella scena. Inoltre, l'urto
laterale non produce un arresto immediato.

Il tracker ora tenta un'associazione basata sul movimento misurato quando la
normale associazione fallisce. La ricerca e limitata a osservazioni recenti,
alla stessa classe e a vicini reciprocamente non ambigui. Le posizioni predette
servono solo all'associazione: vengono esportate esclusivamente rilevazioni reali.

La nuova conferma laterale richiede contatti osservati ripetuti, avvicinamento,
una reazione nel secondo veicolo e variazioni persistenti di forma della bbox,
direzione e velocita del primo. Il contatto di una traccia giovane puo essere
conservato, ma la conferma attende che la traccia maturi. La finestra di conferma
e limitata; assenza di osservazioni, calibrazione invalidata e discontinuita
interrompono la raccolta delle prove.

Nuove esecuzioni complete, senza usare le annotazioni del dataset nell'inferenza:

| Video | Profilo | Impatto annotato | Impatto rilevato | Conferma | Run |
|---|---|---:|---:|---:|---|
| 7ORiNi_LtxY_00 | Accurata, YOLO26m, 15 FPS richiesti | 3,792 s | 3,791667 s | 4,958333 s | 8dd284465bf647d0 |
| JTIfOGQql0A_00 | Rapida, YOLO26n, 8 FPS | 8,267 s | 8,266667 s | 9,933333 s | a85599ddf6694f64 |
| n77TaXyADls_00 | Accurata, YOLO26m, 15 FPS richiesti | 4,471133 s | 4,604600 s | 5,271933 s | 038a37b522034e3a |

Un solo evento per ciascuno dei tre video. Per 7ORiNi_LtxY_00: ID 16 e 22,
241 frame analizzati e decodificati, 141,22 s su CPU. Il replay contiene tutti
i 241 frame, con timestamp crescenti e segnale del contatto verificato visivamente.
Anche i replay degli altri due video sono stati rigenerati e decodificati.

Suite: **83 test superati**, incluso un caso riproducibile con osservazioni
salvate; controlli negativi per assenza di contatto/rotazione/reazione dell'altro
veicolo, perdita delle osservazioni, previsioni, riferimenti invalidi e scadenza.
Ruff senza errori.

- [Report delle tre nuove analisi](../outputs/impact-diagnosis/7ORiNi_LtxY_00/correction.json)
- [Replay corretto](../outputs/real-videos/runs/8dd284465bf647d0/annotated.mp4)
- [Fotogramma del replay](../outputs/impact-diagnosis/7ORiNi_LtxY_00/corrected-impact.jpg)

Le vecchie traiettorie dei dieci video mantengono i risultati precedenti usando
il nuovo detector. Questo controllo non rivaluta il nuovo tracker sui dieci
video: sono state rieseguite integralmente le tre clip indicate nella tabella.
La correzione e verificata su casi segnalati dall'utente, non su un nuovo test
indipendente; non dimostra affidabilita generale.

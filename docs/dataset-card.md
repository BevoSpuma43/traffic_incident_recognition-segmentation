# Dataset card

## Stato dei dati

Il 14 settembre 2026 è stato scaricato ACCIDENT v9 da Kaggle in `data/raw/ACCIDENT/`:
2.027 video CCTV reali, 2.211 video sintetici e annotazioni ufficiali.
Tutti i file hanno superato il controllo CRC; 30 video campione sono stati decodificati.
Dettagli, licenza, split e particolarità dei percorsi in [accident-dataset.md](accident-dataset.md).
Fine-tuning non svolto. La prima verifica della baseline in coordinate immagine su 10 video reali ha prodotto TP=0, FP=2, FN=10: [risultati e limiti](real-video-validation.md). Il campione non dimostra generalizzazione o affidabilita nel traffico ordinario.

ACCIDENT è già disponibile localmente; [CADP](https://arxiv.org/abs/1809.05782) rimane una
possibile sorgente secondaria da esaminare. Registrare separatamente licenze, condizioni
d'accesso, privacy e redistribuzione di ogni sorgente. I video locali sono esclusi da Git.

## Manifest di preparazione: JSONL

Una riga per clip; path relativo al manifest. Esempio di struttura:

```json
{"clip_id":"clip001","camera_id":"camera01","source_id":"sorgente01","path":"raw/clip001.mp4","license":"specificare condizioni e riferimento","duration_s":60.0,"fps":25,"width":1920,"height":1080,"weather":"rain","day_night":"night","hard_negative":false,"duplicate_group":"famiglia001"}
```

Servono almeno tre gruppi indipendenti. Lo script unisce telecamere, sorgenti e gruppi di
duplicati prima dello split; elimina dagli elenchi i duplicati byte-identici tramite SHA256.
La ricerca automatica di quasi duplicati non è implementata: assegnare `duplicate_group`
dopo controllo manuale. I video originali non vengono ricodificati.

Le annotazioni della segmentazione devono seguire il formato YOLO con poligoni per istanza.
Le liste train/val/test devono riferirsi a frame estratti solo dalle clip del rispettivo split.

## Annotazioni eventi: JSON

```json
[
  {"camera_id":"camera01","clip_id":"clip001","impact_time_s":12.4}
]
```

Usare una lista vuota per una clip negativa. Timestamp in secondi dal primo frame.
La pipeline esporta `clip_id` dal config o dal nome del video. Non usare il run_id
al posto dell'identificatore stabile della clip.

Per valutazione multipla `evaluation.json` contiene record:

```json
[
  {"config":"../configs/camera01-test.yaml","clip_id":"clip001","source_id":"sorgente01","camera_id":"camera01","split":"test","truth":"annotations/clip001.json"}
]
```

La durata di valutazione è quella di tutti i video, compresi hard negatives e code.
Annotare meteo, illuminazione, occlusione e dimensione apparente per analisi stratificate.

## Revisione richiesta sui dati reali

- Provenienza e condizioni d'uso tracciabili.
- Nessuna sovrapposizione fra sorgenti/camere dei diversi split.
- Copertura di incidenti, quasi incidenti, traffico fermo, notte, pioggia e occlusioni.
- Controllo umano dei poligoni e della definizione temporale dell'impatto.
- Distanze di calibrazione misurate e punti indipendenti per validarle.

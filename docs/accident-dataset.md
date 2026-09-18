# ACCIDENT scaricato — 14 settembre 2026

Acquisita la versione **9** del dataset ufficiale [picekl/accident su Kaggle](https://www.kaggle.com/datasets/picekl/accident),
collegata dal [sito degli autori](https://accidentbench.github.io/).
Licenza indicata dal catalogo Kaggle: **CC BY-NC-SA 4.0**.

## Percorsi locali

Tutto il dataset è in `data/raw/ACCIDENT/`, escluso da Git.

| Contenuto | Percorso | Quantità |
| --- | --- | --- |
| Video CCTV reali | `real_videos/` | 2.027 MP4 |
| Etichette e split reali | `metadata-real.csv` | 2.027 righe |
| Video sintetici | `synthetic_videos/videos/` | 2.211 MP4 |
| Metadati sintetici | `metadata-synthetic.csv` | 2.211 righe |
| Annotazioni sintetiche | `synthetic_videos/annotations/` | 2.211 JSON |
| Classi semantiche sintetiche | `annotation_classes.yaml` | 1 file |

Archivio conservato: `accident-v9.zip`, **16.758.864.720 byte** (16,76 GB).
Dati estratti: **20.158.641.093 byte** (20,16 GB), **6.452 file**.
Occupazione complessiva archivio + dati: circa **36,92 GB**.

SHA256 dell'archivio:

```text
420d79ff1b5c5e82e93f4d8f8be7f43509943d5844218ccbbbd977c34c26aae6
```

## Verifiche eseguite

- Download completo con dimensione corrispondente al server.
- CRC verificato per tutti i file durante l'estrazione.
- SHA256 individuale registrato in `file-manifest.jsonl`.
- Tutti i video indicati nei CSV sono presenti.
- Timestamp degli incidenti compresi nelle durate dichiarate.
- Primo frame decodificato correttamente per 15 video reali e 15 sintetici.
- JSON delle annotazioni letto correttamente per i 15 campioni sintetici.

Report riproducibili: `download-report.json`, `kaggle-metadata.json`,
`inspection-report.json` e `file-manifest.jsonl`, nella cartella del dataset.
Queste verifiche riguardano acquisizione e leggibilità, non accuratezza del detector.

## Particolarità del pacchetto sintetico

Il CSV originale usa `rgb_path` per i video e riporta annotazioni come
`synthetic_videos/annotations/NOME.json.gz`. Nel pacchetto Kaggle v9 i file corrispondenti
sono già decompressi in `synthetic_videos/annotations/NOME.json/NOME.json`.

Gli originali sono conservati. `annotation-path-map.json` registra la corrispondenza esatta
per tutte le 2.211 annotazioni, senza duplicare o rinominare i dati.

## Protocollo reale disponibile

Durata totale dichiarata dei video reali: circa **12,30 ore**, su cinque categorie:
rear-end (328), t-bone (657), single (680), head-on (117), sideswipe (245).

| Split ufficiale | Train | Test |
| --- | --- | --- |
| `split_in_distribution` | 507 | 1.520 |
| `split_geo_aware` | 454 | 1.573 |

Non è presente una partizione validation esplicita. L'eventuale validation va ricavata
dal train, senza usare il test per scegliere soglie o iperparametri.
I metadati reali non forniscono un camera_id verificato né un'omografia metrica.
Per la pipeline geometrica del progetto servono ancora calibrazioni della scena e
controllo dei raggruppamenti per sorgente/camera.

Le annotazioni reali comprendono tipo, tempo/frame dell'incidente, localizzazione spaziale
e attributi della scena. Non sono maschere per ogni veicolo in ogni frame.
Le clip sono etichettate per tipo di incidente: non costituiscono da sole il corpus
di traffico normale necessario per misurare i falsi allarmi/ora.

## Ripetere l'acquisizione o l'ispezione

```powershell
.venv\Scripts\python.exe scripts/download_accident.py --output data/raw/ACCIDENT --version 9
.venv\Scripts\python.exe scripts/inspect_accident.py --root data/raw/ACCIDENT --samples 15
```

Il download riprende un eventuale archivio parziale. Se l'archivio completo è presente,
lo script ripete hash ed estrazione. Le regole di estrazione rifiutano percorsi esterni
alla directory destinazione e collisioni fra nomi su Windows.

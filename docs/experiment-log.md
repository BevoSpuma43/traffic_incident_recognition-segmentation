# Registro esperimenti

Ogni esecuzione produce una cartella identificata da run_id con configurazione, hash pesi,
hardware, geometria, traiettorie, feature e metriche. Conservare i manifest di split accanto
ai report per poter ripetere l'esperimento.

## Esperimenti implementati nello script evaluate_suite.py

| Variante | Modifica |
| --- | --- |
| baseline | Configurazione originale |
| box_ground_point | Centro inferiore del box al posto del punto della maschera |
| no_smoothing | EMA alpha = 1 |
| long_confirmation | Conferma persistente per 0.8 s |

Lo script supporta validation e test, rifiuta sovrapposizioni di camera/sorgente/clip tra split
e registra le metriche di ogni run. Le varianti non sono state confrontate su dati reali.
Gli esperimenti E0/E1 in coordinate immagine richiedono soglie in pixel indipendenti dalle
soglie metriche: non vengono simulati riutilizzando impropriamente le soglie in metri.

## Error analysis

Ispezionare features.jsonl per stato, score e ragioni. Collegare l'errore al modulo:
segmentazione, tracking, punto a terra, calibrazione, occlusione, soglia o annotazione.
Le maschere vengono renderizzate durante la demo, ma non archiviate per ogni frame nei log
per contenere spazio e I/O; rieseguire la clip con la stessa configurazione per esaminarle.

Non presentare i risultati sintetici come prestazioni CCTV o evidenza della superiorità
di un algoritmo. L'ora di video normale, il test per sorgente e gli esperimenti di
quantizzazione devono ancora essere eseguiti dopo l'acquisizione dei dati.

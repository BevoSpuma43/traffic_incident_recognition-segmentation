"""Evaluate only a frozen, stratified sample of ACCIDENT real test videos."""

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from cctv_incident.accident_sample import select_sample, source_id
from cctv_incident.config import load_config
from cctv_incident.metrics import evaluate_events
from cctv_incident.pipeline import Pipeline, hash_file
from cctv_incident.preview import PreviewWriter
from cctv_incident.segmenter import create_segmenter
from cctv_incident.storage import EventStorage


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def write_report(output, sample, results, metrics, model_load_s):
    total_frames = sum(row["processed_frames"] for row in results)
    processing_s = sum(row["elapsed_s"] for row in results)
    summary = {
        "status": "complete" if len(results) == len(sample["videos"]) else "partial",
        "dataset": "ACCIDENT v9",
        "coordinate_mode": "image",
        "sample_policy": sample["policy"],
        "completed_clips": len(results),
        "model_sha256": sample["model_sha256"],
        "model_load_s": model_load_s,
        "processed_frames": total_frames,
        "processing_elapsed_s": processing_s,
        "effective_fps": total_frames / processing_s if processing_s else 0,
        "peak_rss_mb": max((row["peak_rss_mb"] for row in results), default=0),
        "metrics": metrics,
        "false_alarms_per_hour_on_normal_video": None,
        "limitations": [
            "Small balanced sample of incident clips; not representative of normal traffic.",
            "False alarms/hour uses incident-clip duration; normal-traffic FAR is unavailable.",
            "No metric calibration, vehicle speed ground truth or instance tracking annotations.",
            "Image rules are an untrained baseline; thresholds were frozen before this test.",
            "Filename source groups are proxies, not verified camera identities.",
            "Model pretraining overlap with online source videos is unknown.",
        ],
        "videos": results,
    }
    save_json(output / "summary.json", summary)
    if not results:
        return
    columns = [
        "clip_id",
        "type",
        "day_time",
        "weather",
        "duration_s",
        "impact_time_s",
        "processed_frames",
        "segmented_instances",
        "track_observations",
        "unique_tracks",
        "events",
        "true_positives",
        "false_positives",
        "false_negatives",
        "effective_fps",
        "frame_p95_ms",
        "peak_rss_mb",
        "paused_fraction",
        "max_candidate_score",
        "run_id",
        "preview_path",
    ]
    with (output / "per-video.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    report = [
        "# ACCIDENT: verifica su un campione reale",
        "",
        f"Stato: {summary['status']}. Video completati: {len(results)}/{len(sample['videos'])}.",
        "YOLO26n-seg preaddestrato + ByteTrack + regole temporali in coordinate immagine.",
        "Nessuna calibrazione metrica disponibile: posizioni in pixel, velocita in px/s.",
        "",
        f"- Campione: {sample['policy']['per_class']} clip per classe, seed {sample['policy']['seed']}, split test ufficiale.",
        f"- Durata elaborata: {metrics['duration_s']:.2f} s; frame analizzati: {total_frames}.",
        f"- TP / FP / FN: {metrics['true_positives']} / {metrics['false_positives']} / {metrics['false_negatives']}.",
        f"- Precision: {metrics['precision']:.3f}; recall: {metrics['recall']:.3f}; F1: {metrics['f1']:.3f}.",
        f"- Matching uno-a-uno per clip entro +/- {metrics['tolerance_s']:g} s dall'impatto annotato.",
        f"- FPS effettivi aggregati: {summary['effective_fps']:.2f}; tempo elaborazione: {processing_s:.2f} s.",
        f"- Caricamento modello separato: {model_load_s:.2f} s; RAM massima: {summary['peak_rss_mb']:.0f} MiB.",
        "",
        "I tempi includono decodifica, segmentazione, tracking, geometria, logging, rendering ed encoding",
        "delle anteprime durante la pipeline. Sono esclusi il flush finale delle anteprime e il report.",
        "I percentili per frame escludono decode e buffer, riportati separatamente nei run.",
        "Precision vale convenzionalmente zero se non ci sono predizioni. Il ritardo e disponibile solo per i TP.",
        "",
        "| Video | Tipo | Condizioni | TP/FP/FN | FPS | PAUSED | Anteprima |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in results:
        report.append(
            f"| {row['clip_id']} | {row['type']} | {row['day_time']}, {row['weather']} | "
            f"{row['true_positives']}/{row['false_positives']}/{row['false_negatives']} | "
            f"{row['effective_fps']:.2f} | {row['paused_fraction']:.1%} | "
            f"[MP4]({row['preview_path']}) |"
        )
    report += [
        "",
        "## Interpretazione e limiti",
        "",
        "Il campione contiene incidenti, non un corpus di traffico normale: non permette di stimare",
        "il tasso di falsi allarmi operativo. Gli eventuali FP sono allarmi non associati all'impatto annotato.",
        "La stratificazione per classe e condizioni e i vincoli sul contesto prima/dopo impatto",
        "introducono una selezione esplicita: questi numeri non stimano l'accuratezza sull'intero dataset.",
        "Il punto da maschera, la prospettiva, le occlusioni e i cambi ID limitano le regole in pixel.",
        "La protezione contro i movimenti della camera resta attiva; le porzioni PAUSED rimangono nel denominatore.",
        "Nessun tuning sul campione test e nessun uso delle annotazioni durante l'inferenza.",
        "Servono un validation set separato e video negativi prima di modificare soglie e misurare l'affidabilita.",
        "",
        "## Artefatti",
        "",
        "- [Manifest congelato](sample.json), [configurazione](config.json), [risultati JSON](summary.json), [CSV](per-video.csv).",
        "- [Annotazioni per la sola valutazione](ground-truth.json), [predizioni](predictions.json).",
        "- Anteprime MP4 e fotogrammi scelti dal massimo punteggio in previews/ e snapshots/.",
        "- Configurazione, hash, hardware, traiettorie e feature di ogni esecuzione in pipeline/runs/.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/ACCIDENT"))
    parser.add_argument("--config", type=Path, default=Path("configs/accident-image.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/accident-sample"))
    parser.add_argument("--per-class", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=45)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.max_duration <= 60:
        parser.error("--max-duration must be between 10 and 60 seconds")
    output, root = args.output.resolve(), args.root.resolve()
    if (output / "sample.json").exists():
        parser.error("Output already contains a frozen sample; use a new --output directory")
    cfg = load_config(args.config)
    if cfg.events.coordinate_mode != "image" or cfg.perception.backend == "synthetic":
        parser.error("Use image coordinate mode and a real segmentation model")
    if cfg.video.max_frames is not None:
        parser.error("max_frames must be unset: selected clips must be processed completely")
    selected, policy = select_sample(
        root / "metadata-real.csv", args.per_class, args.seed, "test", args.max_duration
    )
    for row in selected:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            parser.error(f"Missing or invalid video: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for directory in ("previews", "snapshots", "configs"):
        (output / directory).mkdir(exist_ok=True)
    cfg.project.seed = args.seed
    cfg.project.output_dir = output / "pipeline"
    sample = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "ACCIDENT v9",
        "dataset_root": str(root),
        "metadata_sha256": hash_file(root / "metadata-real.csv"),
        "model_sha256": hash_file(cfg.perception.model),
        "policy": policy,
        "code_sha256": {
            str(path.relative_to(cfg.project.root_dir)): hash_file(path)
            for path in sorted((cfg.project.root_dir / "src/cctv_incident").rglob("*.py"))
        },
        "videos": [
            {
                **row,
                "clip_id": Path(row["path"]).stem,
                "camera_id": "source_" + source_id(row["path"]),
                "source_group": source_id(row["path"]),
                "sha256": hash_file(root / row["path"]),
            }
            for row in selected
        ],
    }
    save_json(output / "sample.json", sample)
    save_json(output / "config.json", cfg.model_dump(mode="json"))
    truth = [
        {
            "camera_id": row["camera_id"],
            "clip_id": row["clip_id"],
            "impact_time_s": float(row["accident_time"]),
            "type": row["type"],
        }
        for row in sample["videos"]
    ]
    save_json(output / "ground-truth.json", truth)
    print(f"Frozen sample: {len(selected)} real videos, {policy['duration_s']:.2f} s", flush=True)
    if args.prepare_only:
        return
    loaded = time.perf_counter()
    segmenter = create_segmenter(cfg.perception)
    model_load_s = time.perf_counter() - loaded
    results, predictions, completed_truth = [], [], []
    for index, row in enumerate(sample["videos"]):
        clip_id = row["clip_id"]
        run_cfg = cfg.model_copy(deep=True)
        # Only path and identity enter inference. Labels/times are used after Pipeline.run().
        run_cfg.video.source = str(root / row["path"])
        run_cfg.video.clip_id = clip_id
        run_cfg.calibration.camera_id = row["camera_id"]
        (output / "configs" / f"{clip_id}.yaml").write_text(
            yaml.safe_dump(run_cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
        )
        preview_rel = f"previews/{clip_id}.mp4"
        writer = PreviewWriter(output / preview_rel, cfg.video.target_fps)
        best, last_progress = {"score": -1, "frame": None}, -10
        print(f"[{index + 1}/{len(selected)}] {clip_id} ({row['type']})", flush=True)

        def callback(data, writer=writer, best=best):
            nonlocal last_progress
            writer.append(data["frame"], data["timestamp_s"])
            score = data["decision"].score
            if score > best["score"]:
                best.update(score=score, frame=data["frame"].copy())
            if data["timestamp_s"] - last_progress >= 10:
                print(
                    f"  video={data['timestamp_s']:.1f}s fps={data['fps']:.2f} "
                    f"state={data['decision'].state}",
                    flush=True,
                )
                last_progress = data["timestamp_s"]

        try:
            metrics = Pipeline(run_cfg, segmenter=segmenter).run(callback)
        except Exception as exc:
            save_json(output / "failure.json", {"clip_id": clip_id, "error": repr(exc)})
            raise
        finally:
            writer.close()
        if best["frame"] is not None:
            cv2.imwrite(str(output / "snapshots" / f"{clip_id}.jpg"), best["frame"])
        with (Path(metrics["run_dir"]) / "trajectories.jsonl").open(encoding="utf-8") as handle:
            tracks = {json.loads(line)["track_id"] for line in handle}
        storage = EventStorage(cfg.project.output_dir)
        try:
            events = storage.list_events(metrics["run_id"])
        finally:
            storage.close()
        duration = metrics["source_duration_s"] or float(row["duration"])
        if abs(metrics["media_duration_s"] - duration) > max(0.5, 2 / cfg.video.target_fps):
            raise ValueError(f"Video ended prematurely: {clip_id}")
        target = truth[index]
        per_clip = evaluate_events(events, [target], duration)
        predictions.extend(events)
        completed_truth.append(target)
        results.append(
            {
                **metrics,
                **per_clip,
                "clip_id": clip_id,
                "type": row["type"],
                "day_time": row["day_time"],
                "weather": row["weather"],
                "impact_time_s": target["impact_time_s"],
                "unique_tracks": len(tracks),
                "frame_p95_ms": metrics["timings"]["end_to_end"]["p95_ms"],
                "paused_fraction": metrics["state_counts"].get("PAUSED", 0)
                / metrics["processed_frames"],
                "max_candidate_score": best["score"],
                "preview_path": preview_rel,
            }
        )
        save_json(output / "predictions.json", predictions)
        combined = evaluate_events(
            predictions, completed_truth, sum(r["duration_s"] for r in results)
        )
        write_report(output, sample, results, combined, model_load_s)
        print(
            f"  done: frames={metrics['processed_frames']} events={len(events)} "
            f"TP/FP/FN={per_clip['true_positives']}/{per_clip['false_positives']}/{per_clip['false_negatives']} "
            f"fps={metrics['effective_fps']:.2f}",
            flush=True,
        )
    # A compact visual index of the saved annotated frames, selected independently of ground truth.
    tiles = []
    for row in results:
        image = cv2.imread(str(output / "snapshots" / f"{row['clip_id']}.jpg"))
        if image is None:
            continue
        tile = np.zeros((224, 400, 3), np.uint8)
        ratio = min(400 / image.shape[1], 194 / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * ratio), round(image.shape[0] * ratio)))
        tile[: resized.shape[0], : resized.shape[1]] = resized
        cv2.putText(
            tile,
            row["clip_id"] + " / " + row["type"],
            (4, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    if tiles:
        cv2.imwrite(
            str(output / "contact-sheet.jpg"),
            np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]),
        )
    print(f"Report: {output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()

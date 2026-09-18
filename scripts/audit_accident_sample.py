"""Audit saved sample outputs and plot traces; never runs model inference."""

import argparse
import json
from pathlib import Path

import av
import matplotlib

from cctv_incident.metrics import evaluate_events
from cctv_incident.pipeline import hash_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_json(path):
    return path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/accident-sample"))
    args = parser.parse_args()
    output = args.output.resolve()
    summary = json.loads(read_json(output / "summary.json"))
    sample = json.loads(read_json(output / "sample.json"))
    truth = json.loads(read_json(output / "ground-truth.json"))
    predictions = json.loads(read_json(output / "predictions.json"))
    if summary["status"] != "complete":
        raise ValueError("Finish the sample before auditing it")
    metrics = evaluate_events(predictions, truth, summary["metrics"]["duration_s"])
    if metrics != summary["metrics"]:
        raise ValueError("Saved aggregate event metrics do not match predictions")
    config = json.loads(read_json(output / "config.json"))
    repository = Path(config["project"]["root_dir"])
    for name, expected in sample["code_sha256"].items():
        if hash_file(repository / name) != expected:
            raise ValueError(f"Inference code changed since the benchmark: {name}")
    diagnostics, videos = [], []
    figure, axes = plt.subplots(len(summary["videos"]), 1, figsize=(11, 18), squeeze=False)
    for row, target, axis in zip(summary["videos"], truth, axes[:, 0], strict=True):
        run = Path(row["run_dir"])
        trace = [json.loads(line) for line in read_json(run / "features.jsonl").splitlines()]
        context = [
            frame for frame in trace if abs(frame["timestamp_s"] - target["impact_time_s"]) <= 1
        ]
        reliable_counts = [
            sum(
                m["quality"] >= config["events"]["image"]["min_quality"]
                and m["age_s"] >= config["events"]["min_track_age_s"]
                for m in frame["motions"].values()
            )
            for frame in context
        ]
        diagnostics.append(
            {
                "clip_id": row["clip_id"],
                "context_window_s": 1,
                "context_frames": len(context),
                "frames_with_reliable_motion_anywhere": sum(count > 0 for count in reliable_counts),
                "max_candidate_score_near_impact": max((f["score"] for f in context), default=0),
                "candidate_frames_near_impact": sum(f["state"] == "CANDIDATE" for f in context),
            }
        )
        preview = output / row["preview_path"]
        with av.open(str(preview)) as container:
            previous, count = None, 0
            for frame in container.decode(video=0):
                if frame.time is None or (previous is not None and frame.time <= previous):
                    raise ValueError(f"Non-increasing preview timestamp: {preview}")
                previous = frame.time
                count += 1
            if count != row["processed_frames"]:
                raise ValueError(f"Frame count mismatch: {preview}")
        videos.append(
            {
                "path": str(preview.relative_to(output)),
                "frames": count,
                "last_pts_s": previous,
                "monotonic_pts": True,
            }
        )
        times = [frame["timestamp_s"] for frame in trace]
        scores = [frame["score"] for frame in trace]
        axis.plot(times, scores, color="#176d99", linewidth=1, label="Score euristico")
        axis.axvline(
            target["impact_time_s"], color="#25803b", linestyle="--", label="Impatto annotato"
        )
        for event in predictions:
            if event["clip_id"] == row["clip_id"]:
                axis.axvline(event["impact_time_s"], color="#c83d25", label="Impatto predetto")
        axis.set(
            xlim=(0, row["duration_s"]),
            ylim=(-0.05, 1.05),
            ylabel="Score",
            title=f"{row['clip_id']} / {row['type']}",
        )
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    axes[-1, 0].set_xlabel("Tempo video (s)")
    figure.tight_layout()
    figure.savefig(output / "score-timelines.png", dpi=150)
    plt.close(figure)
    for event in predictions:
        path = Path(event["clip_path"])
        with av.open(str(path)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        if not decoded:
            raise ValueError(f"Empty event clip: {path}")
        videos.append(
            {
                "path": str(path.relative_to(output)),
                "frames": decoded,
                "event_id": event["event_id"],
                "truncated": event["clip_truncated"],
            }
        )
    audit = {
        "inference_rerun": False,
        "metrics_reproduced": True,
        "inference_code_hashes_match": True,
        "videos_decoded": videos,
        "diagnostics": diagnostics,
        "diagnostic_limit": "Reliable motion counts refer to the whole scene, not annotated vehicle identities.",
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Diagnostica del campione",
        "",
        "Analisi dei log salvati, senza rieseguire inferenza o modificare soglie.",
        f"Decodificate {len(summary['videos'])} anteprime e {len(predictions)} clip degli allarmi.",
        "Conteggi frame e timestamp delle anteprime verificati; metriche riprodotte dalle predizioni.",
        "",
        "![Punteggio e istanti di impatto](score-timelines.png)",
        "",
        "| Clip | Frame entro +/-1 s | Frame con moto affidabile nella scena | Score massimo vicino all'impatto |",
        "|---|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['clip_id']} | {row['context_frames']} | "
            f"{row['frames_with_reliable_motion_anywhere']} | "
            f"{row['max_candidate_score_near_impact']:.2f} |"
        )
    lines += [
        "",
        "Un punteggio alto non implica conferma: servono contatto plausibile e "
        "persistenza dell'arresto. I conteggi riguardano tutti i veicoli della scena; "
        "non misurano il tracking dei soli veicoli coinvolti nell'incidente.",
        "",
        "L'esecuzione verifica la pipeline reale ma evidenzia una baseline di rilevamento "
        f"con recall {metrics['recall']:.3f} su questo campione. Le scene notturne, i veicoli piccoli, "
        "la prospettiva e la frammentazione delle tracce richiedono un'analisi su validation "
        "separata e dati di training appropriati. Nessun risultato di mAP, IDF1 o HOTA e deducibile da questi log.",
    ]
    (output / "diagnostics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verified_videos": len(videos),
                "verified_preview_frames": sum(
                    row["processed_frames"] for row in summary["videos"]
                ),
                "metrics": metrics,
                "max_rss_mb": summary["peak_rss_mb"],
                "aggregate_fps": summary["effective_fps"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

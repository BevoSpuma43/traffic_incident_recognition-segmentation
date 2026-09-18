import argparse
import json
from pathlib import Path

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="Local CCTV incident detection")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "demo", "benchmark"):
        sub = commands.add_parser(name)
        sub.add_argument(
            "--config", default="configs/demo.yaml" if name == "demo" else "configs/default.yaml"
        )
        sub.add_argument("--source")
        sub.add_argument("--show", action="store_true")
        sub.add_argument("--max-frames", type=int)
        if name == "demo":
            sub.add_argument("--negative", action="store_true")
    export = commands.add_parser("export-events")
    export.add_argument("--config", default="configs/default.yaml")
    export.add_argument("--output", required=True)
    export.add_argument("--run-id")
    environment = commands.add_parser("check")
    environment.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "check":
        from .pipeline import hardware_info

        print(json.dumps(hardware_info(), indent=2))
        return
    if args.command == "export-events":
        from .storage import EventStorage

        storage = EventStorage(cfg.project.output_dir)
        try:
            storage.export(args.output, args.run_id)
        finally:
            storage.close()
        return
    if args.command == "demo":
        from .demo import generate_demo

        cfg.video.source = str(generate_demo(cfg.project.root_dir, args.negative))
        cfg.perception.backend = "synthetic"
    if args.source:
        cfg.video.source = (
            args.source
            if args.source.startswith(("rtsp://", "rtsps://"))
            else str(Path(args.source).resolve())
        )
    if args.max_frames:
        cfg.video.max_frames = args.max_frames
    from .pipeline import Pipeline

    stopped = [False]

    def preview(data):
        import cv2

        cv2.imshow("CCTV Incident", data["frame"])
        if data["bird_eye"] is not None:
            cv2.imshow("Bird's-eye", data["bird_eye"])
        stopped[0] = cv2.waitKey(1) & 0xFF == ord("q")

    try:
        summary = Pipeline(cfg).run(preview if args.show else None, lambda: stopped[0])
        print(json.dumps(summary, indent=2))
    finally:
        if args.show:
            import cv2

            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

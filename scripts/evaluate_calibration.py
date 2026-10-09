"""Phase 7: reproducible proposal sample, review report, paired comparison or legacy export."""

import argparse
import json
from pathlib import Path

from cctv_incident.batch import export_results, write_json
from cctv_incident.batch_comparison import compare_jobs
from cctv_incident.calibration.evaluation import review_report, sample_proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("sample")
    sample.add_argument("folder", type=Path)
    sample.add_argument("output", type=Path)
    sample.add_argument("--count", type=int, default=5)
    sample.add_argument("--seed", type=int, default=42)
    review = sub.add_parser("review")
    review.add_argument("directory", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("image_job", type=Path)
    compare.add_argument("metric_job", type=Path)
    compare.add_argument("output", type=Path)
    export = sub.add_parser("export")
    export.add_argument("job", type=Path)
    export.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "sample":
        report = sample_proposals(
            args.folder, args.output, count=args.count, seed=args.seed, project_root=Path.cwd()
        )
    elif args.command == "review":
        report = review_report(args.directory)
        write_json(args.directory / "report.json", report)
    elif args.command == "compare":
        report = compare_jobs(args.image_job, args.metric_job)
        if args.output.exists():
            parser.error("Choose a new output file to preserve existing reports")
        write_json(args.output, report)
    else:
        if args.output.resolve() == args.job.resolve() or args.output.exists():
            parser.error("Choose a new output directory; historical experiments remain unchanged")
        report = export_results(args.job, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

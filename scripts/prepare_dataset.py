import argparse

from cctv_incident.dataset import prepare_splits

parser = argparse.ArgumentParser()
parser.add_argument("manifest")
parser.add_argument("--output", default="data/splits")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rows = prepare_splits(args.manifest, args.output, args.seed)
print({split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")})

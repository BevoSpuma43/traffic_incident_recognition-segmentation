import argparse
import hashlib
import json
from pathlib import Path

from ultralytics.utils.downloads import attempt_download_asset

parser = argparse.ArgumentParser(
    description="Explicit preparation step; inference never downloads weights"
)
parser.add_argument("--output", default="models/yolo26n-seg.pt")
args = parser.parse_args()
output = Path(args.output).resolve()
output.parent.mkdir(parents=True, exist_ok=True)
downloaded = Path(attempt_download_asset(str(output)))
if not downloaded.exists():
    raise FileNotFoundError(f"Model download failed: {downloaded}")
digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
downloaded.with_suffix(".sha256.json").write_text(
    json.dumps({"file": downloaded.name, "sha256": digest}, indent=2), encoding="utf-8"
)
print(f"{downloaded}\nSHA256 {digest}")

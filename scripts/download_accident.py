"""Download the official ACCIDENT release, resume transfers and verify extracted files."""

import argparse
import hashlib
import json
import shutil
import stat
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DATASET = "picekl/accident"
PAGE = "https://www.kaggle.com/datasets/picekl/accident"
OFFICIAL = "https://accidentbench.github.io/"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected):
    partial = path.with_suffix(path.suffix + ".part")
    if path.exists():
        if path.stat().st_size != expected:
            raise ValueError("Existing archive size differs; refusing to overwrite it")
        print("Archive already present; verifying and extracting.", flush=True)
        return
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected:
            partial.replace(path)
            return
        if offset > expected:
            raise ValueError("Partial archive is larger than the expected release")
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={offset}-"} if offset else {}
        )
        started, reported, initial = time.monotonic(), time.monotonic(), offset
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.headers.get_content_type() not in (
                    "application/zip",
                    "application/octet-stream",
                    "application/x-zip-compressed",
                ):
                    raise ValueError("Download did not return a ZIP archive")
                if offset and response.status == 200:
                    offset = initial = 0
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an unexpected byte range")
                length = response.headers.get("Content-Length")
                if length and int(length) + offset != expected:
                    raise ValueError("Server archive size differs from dataset metadata")
                with partial.open("ab" if offset else "wb") as handle:
                    while block := response.read(8 * 1024**2):
                        handle.write(block)
                        offset += len(block)
                        now = time.monotonic()
                        if now - reported >= 10:
                            speed = (offset - initial) / max(1e-6, now - started)
                            print(
                                json.dumps(
                                    {
                                        "stage": "download",
                                        "bytes": offset,
                                        "total_bytes": expected,
                                        "percent": round(offset / expected * 100, 1),
                                        "mb_per_s": round(speed / 1e6, 1),
                                    }
                                ),
                                flush=True,
                            )
                            reported = now
            if partial.stat().st_size != expected:
                raise OSError("Transfer ended before the expected size")
            partial.replace(path)
            return
        except (OSError, urllib.error.URLError) as exc:
            print(f"Transfer interrupted ({type(exc).__name__}); retry {attempt + 1}/6", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))


def extract_verified(archive, root):
    records = []
    with zipfile.ZipFile(archive) as package:
        members = package.infolist()
        required = sum(member.file_size for member in members)
        if shutil.disk_usage(root).free < required + 1024**3:
            raise OSError("Insufficient free space for extraction")
        names = set()
        for member in members:
            name = PurePosixPath(member.filename)
            target = (root / str(name)).resolve()
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.filename
                or ":" in member.filename
                or not target.is_relative_to(root)
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe archive member: {member.filename}")
            normalized = str(name).casefold()
            if normalized in names:
                raise ValueError(f"Duplicate archive member: {member.filename}")
            names.add(normalized)
        print(
            json.dumps({"stage": "extract", "files": len(members), "uncompressed_bytes": required}),
            flush=True,
        )
        for index, member in enumerate(members):
            target = root / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".extracting")
            digest = hashlib.sha256()
            # Reading the entire member validates its ZIP CRC.
            with package.open(member) as source, temporary.open("wb") as destination:
                while block := source.read(4 * 1024**2):
                    destination.write(block)
                    digest.update(block)
            if temporary.stat().st_size != member.file_size:
                raise ValueError(f"Extracted size mismatch: {member.filename}")
            temporary.replace(target)
            records.append(
                {
                    "path": member.filename,
                    "bytes": member.file_size,
                    "zip_crc32": f"{member.CRC:08x}",
                    "sha256": digest.hexdigest(),
                }
            )
            if (index + 1) % 250 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "extract",
                            "files_completed": index + 1,
                            "files_total": len(members),
                        }
                    ),
                    flush=True,
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/ACCIDENT")
    parser.add_argument("--version", type=int, default=9)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata = fetch_json("https://www.kaggle.com/api/v1/datasets/list?search=picekl%2Faccident")
    entry = next(item for item in metadata if item["ref"] == DATASET)
    if entry["currentVersionNumber"] != args.version:
        raise ValueError(
            "The catalog version changed; inspect the release before updating --version"
        )
    expected = int(entry["totalBytes"])
    archive = root / f"accident-v{args.version}.zip"
    existing = sum(
        p.stat().st_size for p in (archive, archive.with_suffix(".zip.part")) if p.exists()
    )
    if shutil.disk_usage(root).free < max(0, 2.2 * expected - existing) + 1024**3:
        raise OSError("Insufficient free space for archive and extracted dataset")
    (root / "kaggle-metadata.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}?datasetVersionNumber={args.version}"
    download(url, archive, expected)
    print("Download complete. Computing archive SHA256.", flush=True)
    archive_hash = sha256(archive)
    records = extract_verified(archive, root)
    with (root / "file-manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    report = {
        "dataset": DATASET,
        "dataset_version": args.version,
        "official_page": OFFICIAL,
        "kaggle_page": PAGE,
        "download_url": url,
        "license": entry["licenseName"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive_path": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_hash,
        "files": len(records),
        "extracted_bytes": sum(record["bytes"] for record in records),
        "real_videos": sum(
            record["path"].startswith("real_videos/") and record["path"].endswith(".mp4")
            for record in records
        ),
        "synthetic_videos": sum(
            record["path"].startswith("synthetic_videos/") and record["path"].endswith(".mp4")
            for record in records
        ),
        "integrity": "All extracted members passed ZIP CRC and received a local SHA256",
    }
    (root / "download-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

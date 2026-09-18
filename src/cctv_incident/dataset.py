import hashlib
import json
import random
from pathlib import Path


def prepare_splits(manifest_path, output_dir, seed=42):
    manifest_path = Path(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required = {"clip_id", "camera_id", "source_id", "path", "license", "duration_s"}
    seen_ids, seen_hashes = set(), {}
    # Union-find over camera, source and optional near-duplicate groups.
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        parent[find(b)] = find(a)

    for row in rows:
        if not required <= row.keys() or not row["license"] or row["duration_s"] <= 0:
            raise ValueError(f"Invalid manifest record: {row.get('clip_id')}")
        if row["clip_id"] in seen_ids:
            raise ValueError("Duplicate clip_id")
        seen_ids.add(row["clip_id"])
        path = (manifest_path.parent / row["path"]).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
        row["sha256"], row["absolute_path"] = sha, str(path)
        camera, source = "camera:" + row["camera_id"], "source:" + row["source_id"]
        union(camera, source)
        if row.get("duplicate_group"):
            union(camera, "duplicate:" + row["duplicate_group"])
        if sha in seen_hashes:
            union(camera, seen_hashes[sha])
            row["duplicate_of"] = next(
                r["clip_id"] for r in rows if r.get("sha256") == sha and r is not row
            )
        seen_hashes[sha] = camera
    groups = sorted({find("camera:" + row["camera_id"]) for row in rows})
    if len(groups) < 3:
        raise ValueError(
            "At least three independent camera/source groups are needed for train/val/test"
        )
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * 0.15))
    n_val = max(1, round(len(groups) * 0.15))
    mapping = {
        group: "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for i, group in enumerate(groups)
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["split"] = mapping[find("camera:" + row["camera_id"])]
    for split in ("train", "val", "test"):
        paths = [
            row["absolute_path"]
            for row in rows
            if row["split"] == split and "duplicate_of" not in row
        ]
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows

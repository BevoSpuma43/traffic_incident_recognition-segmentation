import json

from cctv_incident.dataset import prepare_splits


def test_camera_source_and_duplicates_stay_together(tmp_path):
    rows = []
    for i in range(8):
        path = tmp_path / f"{i}.mp4"
        path.write_bytes(bytes([i % 7]) * 50)
        rows.append(
            {
                "clip_id": str(i),
                "camera_id": f"camera_{i}",
                "source_id": f"source_{i // 2}",
                "path": path.name,
                "duration_s": 5,
                "license": "synthetic",
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows))
    output = prepare_splits(manifest, tmp_path / "splits")
    mapping = {row["clip_id"]: row["split"] for row in output}
    assert mapping["0"] == mapping["1"] == mapping["7"] == mapping["6"]
    assert set(mapping.values()) == {"train", "val", "test"}

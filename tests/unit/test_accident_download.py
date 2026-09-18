import hashlib
import zipfile

import pytest

from scripts.download_accident import extract_verified


def test_extract_verifies_and_inventories_original_bytes(tmp_path):
    archive = tmp_path / "source.zip"
    payload = b"example video payload"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("real_videos/example.mp4", payload)
        package.writestr("metadata-real.csv", "path,type\nreal_videos/example.mp4,single\n")
    root = tmp_path / "dataset"
    root.mkdir()
    records = extract_verified(archive, root)
    assert (root / "real_videos/example.mp4").read_bytes() == payload
    assert len(records) == 2
    assert records[0]["sha256"] == hashlib.sha256(payload).hexdigest()
    assert not list(root.rglob("*.extracting"))


@pytest.mark.parametrize("unsafe", ["../escape.txt", "/absolute.txt", "C:/escape.txt"])
def test_archive_cannot_write_outside_dataset(tmp_path, unsafe):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("safe.txt", "safe")
        package.writestr(unsafe, "unsafe")
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_verified(archive, root)
    assert not list(root.iterdir())


def test_windows_case_collision_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("real_videos/Example.mp4", "first")
        package.writestr("real_videos/example.mp4", "second")
    root = tmp_path / "dataset"
    root.mkdir()
    with pytest.raises(ValueError, match="Duplicate archive member"):
        extract_verified(archive, root)
    assert not list(root.iterdir())


def test_resolve_kaggle_decompressed_annotation_layout(tmp_path):
    from scripts.inspect_accident import resolve_annotation

    actual = tmp_path / "synthetic_videos/annotations/example.json/example.json"
    actual.parent.mkdir(parents=True)
    actual.write_text("{}", encoding="utf-8")
    assert resolve_annotation(tmp_path, "synthetic_videos/annotations/example.json.gz") == actual
    assert resolve_annotation(tmp_path, "synthetic_videos/annotations/missing.json.gz") is None
    with pytest.raises(ValueError, match="escapes"):
        resolve_annotation(tmp_path, "../outside.json.gz")

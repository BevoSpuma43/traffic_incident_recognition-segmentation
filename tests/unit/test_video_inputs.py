import pytest

from cctv_incident.video_inputs import list_dataset_videos, save_uploaded_video


def test_same_name_uploads_keep_originals_and_reuse_identical_content(tmp_path):
    first = save_uploaded_video("incidente.mp4", b"first video", tmp_path)
    again = save_uploaded_video("incidente.mp4", b"first video", tmp_path)
    second = save_uploaded_video("incidente.mp4", b"second video", tmp_path)
    assert again == first
    assert second != first
    assert first.read_bytes() == b"first video"
    assert second.read_bytes() == b"second video"


@pytest.mark.parametrize(
    "name", ["../../clip.mp4", r"C:\Videos\clip.MP4", "NUL.mp4", "clip:uno.mp4"]
)
def test_uploaded_filename_stays_inside_storage_and_is_valid_on_windows(tmp_path, name):
    saved = save_uploaded_video(name, b"video", tmp_path)
    assert saved.is_relative_to(tmp_path)
    assert saved.read_bytes() == b"video"
    assert not any(char in saved.name for char in '<>:"/\\|?*')
    assert saved.stem.upper() != "NUL"


@pytest.mark.parametrize("name,content", [("video.mp4", b""), ("document.txt", b"text")])
def test_empty_or_nonvideo_uploads_are_rejected(tmp_path, name, content):
    with pytest.raises(ValueError):
        save_uploaded_video(name, content, tmp_path)
    assert not list(tmp_path.iterdir())


def test_dataset_menu_includes_all_nested_videos_not_only_sample(tmp_path):
    for name in ["real/extra.MP4", "other/second.mkv", "real/metadata.csv"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "fake.mp4").mkdir()
    assert list_dataset_videos(tmp_path) == ["other/second.mkv", "real/extra.MP4"]
    assert list_dataset_videos(tmp_path / "missing") == []

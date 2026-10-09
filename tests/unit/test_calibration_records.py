from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import av
import numpy as np
import pytest
from pydantic import ValidationError

from cctv_incident.calibration import estimate_calibration, load_calibration
from cctv_incident.calibration.records import (
    CalibrationRecord,
    Distance,
    Vertex,
    confirm_record,
    edit_record,
    invalidate_record,
    to_runtime_calibration,
)
from cctv_incident.calibration.repository import (
    create_draft,
    find_record,
    import_legacy_calibration,
    inspect_video,
    load_record,
    read_first_frame,
    save_record,
)


def write_video(path, value=50, size=(160, 96)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(3):
            image = np.full((size[1], size[0], 3), value + index, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture
def draft(tmp_path):
    return create_draft(write_video(tmp_path / "clip.mp4"), tmp_path)


def ready_record(record):
    return edit_record(
        record,
        vertices=tuple(
            Vertex(id=f"P{i + 1}", x=x, y=y)
            for i, (x, y) in enumerate([[10, 10], [140, 10], [140, 80], [10, 80]])
        ),
        width=Distance(value=5, origin="measured", user_confirmed=True),
        length=Distance(
            value=12,
            origin="standard",
            source="Documented local road hypothesis",
            preset="test",
            user_confirmed=True,
        ),
        geometric_quality=0.63,
    )


def test_incomplete_draft_roundtrip_and_original_first_frame(draft, tmp_path):
    record, image = draft
    record = edit_record(record, vertices=(Vertex(id="P1", x=10, y=10),))
    path = save_record(record, tmp_path / "archive", reference_image=image)
    loaded = load_record(path)
    assert path.name == "clip.yaml"
    assert loaded.revision == 1 and loaded.status == "draft" and loaded.runtime is None
    assert loaded.vertices == record.vertices and loaded.width.value is None
    assert loaded.video.relative_path == "clip.mp4"
    assert loaded.reference.frame_index == 0 and loaded.reference.timestamp_s == 0
    assert loaded.reference.pts is not None and loaded.reference.time_base
    assert loaded.video.image_size == (160, 96)
    assert find_record(loaded.video, tmp_path / "archive").status == "compatible"
    with pytest.raises(ValueError, match="confirmed"):
        to_runtime_calibration(loaded)
    with pytest.raises(ValueError):
        confirm_record(loaded)
    with pytest.raises(ValueError, match="confirmed"):
        load_calibration(path)


def test_confirmed_roundtrip_provenance_quality_independent_roi_and_revision(draft, tmp_path):
    record, image = draft
    confirmed = confirm_record(ready_record(record))
    assert confirmed.geometric_quality == confirmed.runtime.confidence == 0.63
    assert confirmed.runtime.roi_px == [[0, 0], [159, 0], [159, 95], [0, 95]]
    path = save_record(confirmed, tmp_path / "archive", reference_image=image)
    first = load_record(path)
    assert first.width.origin == "measured" and first.length.origin == "standard"
    assert first.length.source == "Documented local road hypothesis"
    runtime = load_calibration(path)
    np.testing.assert_allclose(runtime.transform((75, 45)), [2.5, 6], atol=1e-6)
    runtime.valid = False
    assert to_runtime_calibration(first).valid  # No mutable runtime object is leaked.
    edited = edit_record(first, length=Distance(value=14, origin="measured", user_confirmed=True))
    assert edited.status == "draft" and edited.runtime is None
    save_record(edited, tmp_path / "archive")
    second = load_record(path)
    assert second.revision == 2 and second.created_at == first.created_at
    assert second.reference.image_path != first.reference.image_path
    assert load_record(path.with_name("clip.revision-0001.yaml")) == first
    assert find_record(first.video, tmp_path / "archive").record == second
    with pytest.raises(ValueError, match="Stale"):
        save_record(edited, tmp_path / "archive")


def test_vertex_ids_survive_drag_and_array_reordering(draft):
    record, _ = draft
    ready = ready_record(record)
    scrambled = edit_record(ready, vertices=tuple(reversed(ready.vertices)))
    confirmed = confirm_record(scrambled)
    assert confirmed.runtime.destination_points == [[0, 0], [5, 0], [5, 12], [0, 12]]
    for vertex in confirmed.vertices:
        expected = confirmed.runtime.destination_points[int(vertex.id[1:]) - 1]
        np.testing.assert_allclose(
            confirmed.runtime.transform((vertex.x, vertex.y)), expected, atol=1e-6
        )
    moved = tuple(
        Vertex(id=v.id, x=v.x + (1 if v.id == "P1" else 0), y=v.y) for v in confirmed.vertices
    )
    assert edit_record(confirmed, vertices=moved).runtime is None
    assert edit_record(confirmed, vertices=moved).geometric_quality == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"width": Distance(value=5)},
        {"length": Distance()},
        {"width": Distance(value=5, origin="measured", user_confirmed=False)},
    ],
)
def test_missing_or_unconfirmed_scale_cannot_become_runtime(draft, changes):
    with pytest.raises(ValueError, match="distances"):
        confirm_record(edit_record(ready_record(draft[0]), **changes))


def test_mutation_tampering_invalid_and_unknown_schema_are_rejected(draft):
    record = confirm_record(ready_record(draft[0]))
    data = record.model_dump()
    data["runtime"]["homography"][0][0] *= 2
    with pytest.raises(ValueError, match="homography"):
        CalibrationRecord.model_validate(data)
    data = record.model_dump()
    data["schema_version"] = 2
    with pytest.raises(ValidationError):
        CalibrationRecord.model_validate(data)
    with pytest.raises(ValueError, match="confirmed"):
        to_runtime_calibration(invalidate_record(record, "Camera moved"))
    with pytest.raises(ValueError, match="corrected"):
        confirm_record(invalidate_record(record, "Camera moved"))
    for value in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValidationError):
            Distance(value=value)
    with pytest.raises(ValidationError):
        Distance(value=5, origin="standard")


def test_same_name_replaced_video_is_incompatible_but_content_can_move(draft, tmp_path):
    record, image = draft
    archive = tmp_path / "archive"
    save_record(record, archive, reference_image=image)
    moved = tmp_path / "renamed.mp4"
    moved.write_bytes(Path(record.video.source_path).read_bytes())
    assert find_record(inspect_video(moved), archive).status == "compatible"
    write_video(Path(record.video.source_path), value=90)
    replaced = inspect_video(record.video.source_path)
    result = find_record(replaced, archive)
    assert result.status == "incompatible" and "Video content hash differs" in result.reasons


def test_collisions_extensions_and_windows_names_do_not_overwrite(tmp_path):
    archive = tmp_path / "archive"
    paths = []
    for directory, name, value in [
        ("a", "clip.mp4", 10),
        ("b", "clip.mp4", 40),
        ("c", "clip.avi", 70),
    ]:
        record, image = create_draft(write_video(tmp_path / directory / name, value))
        paths.append(save_record(record, archive, reference_image=image))
    assert len(set(paths)) == 3 and all(p.name == "clip.yaml" for p in paths)
    assert paths[0] == archive / "clip.yaml"
    for path in paths:
        record = load_record(path)
        assert find_record(record.video, archive).path == path.resolve()
    record, image = create_draft(write_video(tmp_path / "safe.mp4"))
    data = record.model_dump()
    data["video"]["original_name"] = "CON.mp4"
    safe = save_record(CalibrationRecord.model_validate(data), archive, reference_image=image)
    assert safe.name == "video_CON.yaml"
    data["record_id"] = uuid4().hex
    data["video"]["original_name"] = "clip.revision-0001.mp4"
    safe = save_record(CalibrationRecord.model_validate(data), archive, reference_image=image)
    assert safe.name == "clip.revision-0001_video.yaml"


def test_reference_tampering_and_wrong_image_are_rejected(draft, tmp_path):
    record, image = draft
    archive = tmp_path / "archive"
    with pytest.raises(ValueError, match="first frame"):
        save_record(record, archive, reference_image=np.zeros_like(image))
    path = save_record(record, archive, reference_image=image)
    loaded = load_record(path)
    (path.parent / loaded.reference.image_path).write_bytes(b"wrong image")
    with pytest.raises(ValueError, match="hash"):
        load_record(path)
    assert find_record(record.video, archive).status == "incompatible"
    other, other_image = create_draft(write_video(tmp_path / "other.mp4", 120))
    assert (
        load_record(save_record(other, archive, reference_image=other_image)).video == other.video
    )


def test_failed_publish_preserves_previous_yaml_and_reference(draft, tmp_path, monkeypatch):
    from cctv_incident.calibration import repository

    record, image = draft
    archive = tmp_path / "archive"
    path = save_record(record, archive, reference_image=image)
    previous = load_record(path)
    before = path.read_bytes()
    atomic = repository._atomic_bytes

    def fail_current(target, content):
        if target == path:
            raise OSError("Injected publication failure")
        atomic(target, content)

    monkeypatch.setattr(repository, "_atomic_bytes", fail_current)
    with pytest.raises(OSError, match="publication"):
        save_record(edit_record(previous, geometric_quality=0.4), archive)
    assert path.read_bytes() == before and load_record(path) == previous
    monkeypatch.setattr(repository, "_atomic_bytes", atomic)
    save_record(edit_record(previous, geometric_quality=0.4), archive)
    assert load_record(path).geometric_quality == 0.4


def test_legacy_loading_and_explicit_video_association(draft, tmp_path):
    record, image = draft
    ready = ready_record(record)
    legacy = estimate_calibration(
        "generic_camera",
        record.video.image_size,
        ready.points_px,
        [[0, 0], [5, 0], [5, 12], [0, 12]],
    )
    path = tmp_path / "legacy.yaml"
    legacy.save(path)
    assert load_calibration(path).model_dump() == legacy.model_dump()
    with pytest.raises(ValueError, match="Legacy"):
        load_record(path)
    with pytest.raises(ValueError, match="associated"):
        import_legacy_calibration(
            legacy,
            record.video,
            record.reference,
            association_verified=False,
            scale_origin="measured",
            scale_source="Survey",
        )
    imported = import_legacy_calibration(
        legacy,
        record.video,
        record.reference,
        association_verified=True,
        scale_origin="measured",
        scale_source="Survey",
    )
    assert imported.status == "draft" and imported.camera_id != "generic_camera"
    imported_path = save_record(
        confirm_record(imported), tmp_path / "archive", reference_image=image
    )
    np.testing.assert_allclose(
        load_calibration(imported_path).transform((75, 45)), [2.5, 6], atol=1e-6
    )
    legacy.units = "canonical"
    with pytest.raises(ValueError, match="metric"):
        import_legacy_calibration(
            legacy,
            record.video,
            record.reference,
            association_verified=True,
            scale_origin="measured",
            scale_source="Survey",
        )


def test_empty_corrupt_video_and_no_stream_fail_cleanly(tmp_path, monkeypatch):
    from cctv_incident.calibration import repository

    for content in (b"", b"not a video"):
        path = tmp_path / "broken.mp4"
        path.write_bytes(content)
        with pytest.raises(ValueError):
            read_first_frame(path)
    closed = []

    class Container:
        streams = SimpleNamespace(video=[])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

    monkeypatch.setattr(repository.av, "open", lambda source: Container())
    with pytest.raises(ValueError, match="no video stream"):
        read_first_frame(path)
    assert closed == [True]
    with pytest.raises(ValueError, match="missing"):
        create_draft(tmp_path / "missing.mp4")


def test_first_decodable_frame_skips_bad_packet_and_closes_decoder(tmp_path, monkeypatch):
    from cctv_incident.calibration import repository

    closed = []
    frame = av.VideoFrame.from_ndarray(np.zeros((8, 12, 3), np.uint8), format="bgr24")

    class BadPacket:
        def decode(self):
            raise av.InvalidDataError(1, "corrupt packet")

    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(metadata={})])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def demux(self, stream):
            return iter([BadPacket(), SimpleNamespace(decode=lambda: [frame])])

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(repository.av, "open", lambda source: Container())
    result = read_first_frame(path)
    assert result.timestamp_s is None and result.pts is None and result.image.shape == (8, 12, 3)
    assert closed == [True]
    frame.pts, frame.time_base = 120, Fraction(1, 10)
    assert read_first_frame(path).timestamp_s == 12.0
    identity = create_draft(path)[0].video
    changed_orientation = identity.model_copy(update={"rotation_degrees": 90})
    archive = tmp_path / "archive"
    record, image = create_draft(path)
    save_record(record, archive, reference_image=image)
    assert "Video rotation metadata differs" in find_record(changed_orientation, archive).reasons


def test_first_frame_matches_native_pipeline_decoder(draft):
    from cctv_incident.config import Video
    from cctv_incident.video import VideoSource

    record, image = draft
    source = VideoSource(Video(source=record.video.source_path))
    try:
        first = next(iter(source))
        np.testing.assert_array_equal(image, first.image)
        assert first.frame_index == record.reference.frame_index
    finally:
        source.close()


def test_singular_runtime_and_reference_path_escape_rejected(draft, tmp_path):
    import yaml

    confirmed = confirm_record(ready_record(draft[0]))
    data = confirmed.model_dump()
    data["runtime"]["homography"] = [[0, 0, 0]] * 3
    with pytest.raises(ValueError, match="Degenerate"):
        CalibrationRecord.model_validate(data)
    archive = tmp_path / "archive"
    path = save_record(confirmed, archive, reference_image=draft[1])
    data = load_record(path).model_dump(mode="json")
    data["reference"]["image_path"] = "../outside.png"
    data["runtime"]["reference_image"] = "../outside.png"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        load_record(path)


def test_ambiguous_content_requires_explicit_choice_and_archive_lock(draft, tmp_path):
    from cctv_incident.calibration import repository

    record, image = draft
    archive = tmp_path / "archive"
    save_record(record, archive, reference_image=image)
    duplicate = tmp_path / "renamed.mp4"
    duplicate.write_bytes(Path(record.video.source_path).read_bytes())
    other, other_image = create_draft(duplicate)
    save_record(other, archive, reference_image=other_image)
    assert find_record(record.video, archive).status == "ambiguous"
    with repository._save_lock(archive), pytest.raises(RuntimeError, match="another process"):
        save_record(other, archive, reference_image=other_image)


def test_legacy_advanced_correspondences_and_original_metadata_survive_import(draft, tmp_path):
    record, image = draft
    points = ready_record(record).points_px + [[50, 30], [100, 60]]
    destination = [[(x - 10) / 26, (y - 10) * 12 / 70] for x, y in points]
    legacy = estimate_calibration("legacy", record.video.image_size, points, destination)
    imported = import_legacy_calibration(
        legacy,
        record.video,
        record.reference,
        association_verified=True,
        scale_origin="experimental",
        scale_source="Original survey approximation",
    )
    path = save_record(confirm_record(imported), tmp_path / "archive", reference_image=image)
    loaded = load_record(path)
    assert len(loaded.vertices) == 6
    assert loaded.automation.parameters["legacy_calibration"] == legacy.model_dump(mode="json")
    assert loaded.explicit_scale.origin == "experimental"
    np.testing.assert_allclose(
        to_runtime_calibration(loaded).transform((75, 45)), [2.5, 6], atol=1e-6
    )

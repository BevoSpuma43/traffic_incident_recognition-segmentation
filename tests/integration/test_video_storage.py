import av
import cv2
import numpy as np

from cctv_incident.clip_buffer import ClipBuffer, EncodedFrame, write_clip
from cctv_incident.config import Storage, Video
from cctv_incident.types import Event
from cctv_incident.video import VideoSource


def test_vfr_timestamp_roundtrip(tmp_path):
    image = np.zeros((64, 64, 3), np.uint8)
    ok, jpeg = cv2.imencode(".jpg", image)
    assert ok
    times = [12, 12.07, 12.21, 12.5, 12.9]
    path = tmp_path / "vfr.mp4"
    write_clip(path, [EncodedFrame(t, jpeg.tobytes()) for t in times])
    source = VideoSource(Video(source=str(path)))
    try:
        actual = [frame.timestamp_s for frame in source]
    finally:
        source.close()
    np.testing.assert_allclose(actual, np.array(times) - 12, atol=0.002)


def test_buffer_eof_and_byte_bound(tmp_path, config):
    buffer = ClipBuffer(config.events, Storage(buffer_mb=1), tmp_path)
    image = np.zeros((64, 64, 3), np.uint8)
    event = Event("event_1", "camera", 1, 2, 3, [1, 2], 0.9, 1, ["stop"])
    for timestamp in np.arange(0, 3.1, 0.1):
        buffer.append(image, timestamp)
        assert buffer.bytes <= buffer.max_bytes
    buffer.schedule(event)
    for timestamp in np.arange(3.1, 3.6, 0.1):
        buffer.append(image, timestamp)
    completed = buffer.close()
    assert len(completed) == 1 and event.clip_truncated
    assert event.clip_status == "saved"
    with av.open(event.clip_path) as container:
        assert sum(1 for _ in container.decode(video=0)) > 20


def test_buffer_applies_actual_memory_limit(tmp_path, config):
    buffer = ClipBuffer(config.events, Storage(buffer_mb=1), tmp_path)
    image = np.random.default_rng(42).integers(0, 255, (512, 512, 3), dtype=np.uint8)
    try:
        for index in range(30):
            buffer.append(image, index / 20)
        assert buffer.bytes <= 1024**2
        assert len(buffer.frames) < 10
    finally:
        buffer.close()


def test_zero_post_context_and_capacity(tmp_path, config):
    config.events.post_event_s = 0
    buffer = ClipBuffer(config.events, Storage(max_pending_clips=1), tmp_path)
    image = np.zeros((64, 64, 3), np.uint8)
    for index in range(21):
        buffer.append(image, index / 10)
    first = Event("first", "camera", 1, 1.5, 2, [1], 0.9, 1, ["stop"])
    second = Event("second", "camera", 1, 1.5, 2, [2], 0.9, 1, ["stop"])
    buffer.schedule(first)
    buffer.schedule(second)
    assert second.clip_status == "skipped_capacity"
    buffer.close()
    assert first.clip_status == "saved"
    assert not first.clip_truncated

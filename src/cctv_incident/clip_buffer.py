from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from functools import wraps
from pathlib import Path
from threading import RLock

import av
import cv2
import numpy as np


@dataclass
class EncodedFrame:
    timestamp: float
    jpeg: bytes


@dataclass
class PendingClip:
    event: object
    end: float
    frames: list = field(default_factory=list)
    byte_count: int = 0


def write_clip(path, frames):
    if not frames:
        raise ValueError("Cannot encode an empty event clip")
    with av.open(str(path), "w") as container:
        first = cv2.imdecode(np.frombuffer(frames[0].jpeg, np.uint8), cv2.IMREAD_COLOR)
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = first.shape[1] // 2 * 2, first.shape[0] // 2 * 2
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {"crf": "23", "preset": "veryfast"}
        last_pts = -1
        for item in frames:
            pixels = cv2.imdecode(np.frombuffer(item.jpeg, np.uint8), cv2.IMREAD_COLOR)
            pixels = pixels[: stream.height, : stream.width]
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            pts = max(last_pts + 1, round((item.timestamp - frames[0].timestamp) * 90000))
            frame.pts, frame.time_base = pts, Fraction(1, 90000)
            last_pts = pts
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return str(path)


def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return wrapped


class ClipBuffer:
    def __init__(self, events_config, storage_config, directory):
        self.lock = RLock()
        self.config, self.storage = events_config, storage_config
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = deque()
        self.bytes = 0
        self.max_bytes = storage_config.buffer_mb * 1024**2
        self.pending = []
        self.jobs = []
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clip-writer")
        self.last_timestamp = -float("inf")

    @property
    def history_seconds(self):
        return self.config.pre_event_s + self.config.candidate_timeout_s + 1

    @synchronized
    def append(self, frame, timestamp):
        if timestamp <= self.last_timestamp:
            return
        self.last_timestamp = timestamp
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.storage.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("JPEG compression failed")
        item = EncodedFrame(timestamp, encoded.tobytes())
        self.frames.append(item)
        self.bytes += len(item.jpeg)
        while self.frames and (
            timestamp - self.frames[0].timestamp > self.history_seconds
            or self.bytes > self.max_bytes
        ):
            self.bytes -= len(self.frames.popleft().jpeg)
        for pending in list(self.pending):
            if timestamp <= pending.end:
                pending.frames.append(item)
                pending.byte_count += len(item.jpeg)
            if pending.byte_count > self.max_bytes:
                pending.event.clip_truncated = True
                self._submit(pending)
            elif timestamp >= pending.end:
                self._submit(pending)

    @synchronized
    def schedule(self, event):
        if len(self.pending) + len(self.jobs) >= self.storage.max_pending_clips:
            event.clip_status = "skipped_capacity"
            return
        start = max(0, event.impact_time_s - self.config.pre_event_s)
        end = event.confirm_time_s + self.config.post_event_s
        frames = [frame for frame in self.frames if start <= frame.timestamp <= end]
        event.clip_truncated = not frames or frames[0].timestamp > start + 0.25
        self.pending.append(
            PendingClip(
                event,
                end,
                frames,
                sum(len(f.jpeg) for f in frames),
            )
        )

        if self.last_timestamp >= end:
            self._submit(self.pending[-1])

    def _submit(self, pending):
        self.pending.remove(pending)
        event = pending.event
        if not pending.frames:
            event.clip_status = "failed_empty"
            self.jobs.append((event, None))
            return
        event.clip_start_s, event.clip_end_s = (
            pending.frames[0].timestamp,
            pending.frames[-1].timestamp,
        )
        path = self.directory / f"{event.event_id}.mp4"
        future = self.executor.submit(write_clip, path, pending.frames)
        self.jobs.append((event, future))

    @synchronized
    def collect(self):
        completed = []
        for event, future in list(self.jobs):
            if future is not None and not future.done():
                continue
            if future is not None:
                try:
                    event.clip_path = future.result()
                    event.clip_status = "saved"
                except Exception as exc:
                    event.clip_status = f"failed: {type(exc).__name__}: {exc}"
            completed.append(event)
            self.jobs.remove((event, future))
        return completed

    @synchronized
    def enforce_retention(self):
        files = sorted(self.directory.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        deleted = []
        protected = {event.event_id for event, _ in self.jobs}
        root = self.directory.resolve()
        for path in files:
            if total <= self.storage.retention_gb * 1024**3:
                break
            if path.stem in protected:
                continue
            resolved = path.resolve()
            if resolved.parent != root:
                continue
            total -= resolved.stat().st_size
            resolved.unlink()
            deleted.append(path.stem)
        return deleted

    @synchronized
    def close(self):
        for pending in list(self.pending):
            pending.event.clip_truncated = True
            self._submit(pending)
        self.executor.shutdown(wait=True)
        return self.collect()

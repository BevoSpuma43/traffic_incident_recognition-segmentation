import queue
import threading
import time
from dataclasses import dataclass

import av
import cv2
import numpy as np


@dataclass
class VideoFrame:
    image: np.ndarray
    timestamp_s: float
    frame_index: int
    discontinuity: bool = False


class VideoSource:
    """MP4 keeps decoded PTS; RTSP uses a bounded latest-frame queue."""

    def __init__(self, config, on_frame=None):
        self.config = config
        self.on_frame = on_frame
        self.live = config.source.lower().startswith(("rtsp://", "rtsps://"))
        self.stop = threading.Event()
        self.thread = None
        self.container = None
        self.error = None
        self.dropped_frames = 0
        self.timestamp_fallbacks = 0
        self.duration_s = None
        self.original_image_size = None

    def _decode(self):
        reconnects, index, last = 0, 0, -1.0
        origin = None
        last_arrival = time.monotonic()
        while not self.stop.is_set():
            try:
                options = {"rtsp_transport": "tcp"} if self.live else {}
                with av.open(
                    self.config.source,
                    options=options,
                    timeout=(self.config.timeout_s, self.config.timeout_s),
                ) as container:
                    self.container = container
                    stream = container.streams.video[0]
                    self.original_image_size = (stream.width, stream.height)
                    if not self.live and stream.duration is not None:
                        self.duration_s = float(stream.duration * stream.time_base)
                    offset = None
                    first_after_reconnect = reconnects > 0
                    for frame in container.decode(stream):
                        if self.stop.is_set():
                            return
                        raw = float(frame.pts * frame.time_base) if frame.pts is not None else None
                        if raw is None:
                            self.timestamp_fallbacks += 1
                            if self.live:
                                raw = time.monotonic()
                            elif stream.average_rate:
                                raw = (origin or 0) + index / float(stream.average_rate)
                            else:
                                raise ValueError(
                                    "Video has neither timestamps nor a fallback frame rate"
                                )
                        if origin is None:
                            origin = raw
                        if offset is None:
                            gap = max(1 / self.config.target_fps, time.monotonic() - last_arrival)
                            offset = (last + gap - raw) if reconnects else -origin
                        timestamp = raw + offset
                        if timestamp <= last:
                            continue
                        pixels = frame.to_ndarray(format="bgr24")
                        if self.config.max_width and pixels.shape[1] > self.config.max_width:
                            factor = self.config.max_width / pixels.shape[1]
                            pixels = cv2.resize(
                                pixels,
                                (self.config.max_width, round(pixels.shape[0] * factor)),
                                interpolation=cv2.INTER_AREA,
                            )
                        packet = VideoFrame(
                            pixels,
                            timestamp,
                            index,
                            first_after_reconnect,
                        )
                        last_arrival = time.monotonic()
                        first_after_reconnect = False
                        if self.live and self.on_frame:
                            self.on_frame(packet)
                        yield packet
                        last, index = timestamp, index + 1
                        if self.config.max_frames and index >= self.config.max_frames:
                            return
                if not self.live:
                    return
            except (av.FFmpegError, OSError):
                if not self.live:
                    raise
            finally:
                self.container = None
            if not self.config.reconnect_rtsp or reconnects >= self.config.max_reconnects:
                raise RuntimeError("RTSP ended or reconnection limit reached")
            reconnects += 1
            self.stop.wait(min(reconnects, 3))

    def __iter__(self):
        if not self.live:
            yield from self._decode()
            return
        frames = queue.Queue(maxsize=self.config.queue_size)

        def produce():
            try:
                for frame in self._decode():
                    try:
                        frames.put_nowait(frame)
                    except queue.Full:
                        try:
                            frames.get_nowait()
                        except queue.Empty:
                            pass
                        self.dropped_frames += 1
                        frames.put_nowait(frame)
            except Exception as exc:
                self.error = exc
            finally:
                self.stop.set()

        self.thread = threading.Thread(target=produce, name="rtsp-decoder", daemon=True)
        self.thread.start()
        while not self.stop.is_set() or not frames.empty():
            try:
                yield frames.get(timeout=0.25)
            except queue.Empty:
                continue
        if self.error:
            raise self.error

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=self.config.timeout_s + 1)
        elif self.container is not None:
            self.container.close()

"""Stream an annotated preview without retaining the video in memory."""

from fractions import Fraction
from pathlib import Path

import av
import cv2


class PreviewWriter:
    def __init__(self, path, fps=8):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.container = self.stream = None
        self.first_timestamp = None
        self.last_pts = -1

    def append(self, image, timestamp_s):
        if self.container is None:
            self.container = av.open(str(self.path), "w", options={"movflags": "+faststart"})
            self.stream = self.container.add_stream(
                "libx264", rate=Fraction(str(self.fps)).limit_denominator(1_000_000)
            )
            self.stream.width = image.shape[1] // 2 * 2
            self.stream.height = image.shape[0] // 2 * 2
            self.stream.pix_fmt = "yuv420p"
            self.stream.time_base = Fraction(1, 90000)
            self.stream.codec_context.time_base = Fraction(1, 90000)
            self.stream.options = {"preset": "veryfast", "crf": "24"}
            self.first_timestamp = timestamp_s
        size = (self.stream.width, self.stream.height)
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.time_base = Fraction(1, 90000)
        frame.pts = max(self.last_pts + 1, round((timestamp_s - self.first_timestamp) * 90000))
        self.last_pts = frame.pts
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        if self.container is not None:
            try:
                if self.stream is not None:
                    for packet in self.stream.encode():
                        self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None

from pathlib import Path

import av
import cv2
import numpy as np


def sample_background(source, count=100, max_width=640):
    if not 4 <= count <= 200:
        raise ValueError("Sample count must be between 4 and 200")
    frames = []
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if duration is None and container.duration:
            duration = container.duration / av.time_base
        if not duration:
            raise ValueError("Background sampling requires a finite prerecorded video")
        start = stream.start_time or 0
        image_size = None
        for timestamp in np.linspace(0, max(0, duration - 0.1), count):
            container.seek(start + int(timestamp / float(stream.time_base)), stream=stream)
            target = timestamp + float(start * stream.time_base)
            for frame in container.decode(stream):
                if frame.time is not None and frame.time + 0.05 < target:
                    continue
                pixels = frame.to_ndarray(format="bgr24")
                image_size = (pixels.shape[1], pixels.shape[0])
                scale = min(1, max_width / pixels.shape[1])
                pixels = cv2.resize(pixels, None, fx=scale, fy=scale)
                frames.append(pixels)
                break
    if len(frames) < 4:
        raise ValueError("Not enough frames for a median background")
    exposure = np.array([image.mean() for image in frames])
    median = np.median(exposure)
    frames = [
        frame
        for frame, mean in zip(frames, exposure, strict=True)
        if abs(mean - median) <= max(15, 0.35 * median)
    ]
    stack = np.stack(frames)
    background = np.median(stack, axis=0).astype(np.uint8)
    unstable = np.median(np.abs(stack.astype(np.float32) - background), axis=0).mean(axis=2) > 20
    return background, unstable.astype(np.uint8) * 255, image_size


def save_background(source, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    background, unstable, size = sample_background(source)
    cv2.imwrite(str(directory / "background.jpg"), background)
    cv2.imwrite(str(directory / "unstable.png"), unstable)
    return background, size

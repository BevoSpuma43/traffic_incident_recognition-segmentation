import inspect
from types import SimpleNamespace

import numpy as np

from .track_association import motion_costs
from .types import Track


class VehicleTracker:
    """ByteTrack adapter; only real matched detections are exported as observations."""

    def __init__(self, config, target_fps):
        from .segmenter import configure_offline_runtime

        configure_offline_runtime()
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.config = config
        args = SimpleNamespace(
            track_high_thresh=config.track_high_thresh,
            track_low_thresh=config.track_low_thresh,
            new_track_thresh=config.new_track_thresh,
            track_buffer=max(1, round(config.lost_seconds * target_fps)),
            match_thresh=config.match_thresh,
            fuse_score=config.fuse_score,
        )
        kwargs = (
            {"frame_rate": target_fps}
            if "frame_rate" in inspect.signature(BYTETracker).parameters
            else {}
        )
        owner = self

        class MotionByteTracker(BYTETracker):
            def get_dists(self, tracks, detections):
                costs = super().get_dists(tracks, detections)
                if not owner.config.motion_matching:
                    return costs
                return motion_costs(
                    costs,
                    tracks,
                    detections,
                    owner.observed_boxes,
                    owner.last_timestamp,
                    self.frame_id,
                    self.args.match_thresh,
                )

        self.observed_boxes = {}
        self.tracker = MotionByteTracker(args, **kwargs)
        self.seen = {}
        self.last_timestamp = -float("inf")
        self.target_fps = target_fps

    def update(self, instances, timestamp, image_shape):
        from ultralytics.engine.results import Boxes

        if timestamp <= self.last_timestamp:
            return []
        # Explicit wall/media-time expiry, independent of frame skipping.
        cutoff = timestamp - self.config.lost_seconds
        for name in ("tracked_stracks", "lost_stracks"):
            tracks = getattr(self.tracker, name)
            setattr(
                self.tracker, name, [t for t in tracks if self.seen.get(t.track_id, -1) >= cutoff]
            )
        self.seen = {key: value for key, value in self.seen.items() if value >= cutoff}
        self.observed_boxes = {
            key: rows for key, rows in self.observed_boxes.items() if rows[-1][0] >= cutoff
        }
        self.last_timestamp = timestamp
        rows = np.array(
            [[*x.bbox, x.confidence, x.class_id] for x in instances], np.float32
        ).reshape(-1, 6)
        result = self.tracker.update(Boxes(rows, image_shape))
        self.seen.update({track.track_id: timestamp for track in self.tracker.tracked_stracks})
        for track in self.tracker.tracked_stracks:
            index = int(track.idx)
            if track.frame_id == self.tracker.frame_id and 0 <= index < len(instances):
                samples = self.observed_boxes.setdefault(track.track_id, [])
                samples.append((timestamp, instances[index].bbox.copy()))
                del samples[:-2]
        output = []
        for row in result:
            track_id, detection_index = int(row[4]), int(row[-1])
            if 0 <= detection_index < len(instances):
                self.seen[track_id] = timestamp
                output.append(Track(track_id, instances[detection_index]))
        # The dependency keeps an audit list of removed tracks; our JSONL is the audit log.
        self.tracker.removed_stracks = self.tracker.removed_stracks[-self.config.max_tracks :]
        return output

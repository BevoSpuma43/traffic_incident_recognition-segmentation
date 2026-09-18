# Traffic Accident Detection System: A Kinematic and Instance Segmentation Approach

## Abstract
This report presents a comprehensive, detailed analysis of a custom traffic accident detection system. The project leverages advanced computer vision techniques, specifically real-time instance segmentation and multi-object tracking, combined with deterministic kinematic physics to identify potential collisions between vehicles in video streams. By analyzing the pixel-level overlap of vehicle masks alongside behavioral anomalies—such as sudden decelerations or simultaneous stops—the system aims to accurately detect crashes while mitigating false positives caused by simple visual occlusions (e.g., vehicles passing each other in different lanes). This document extensively reviews the architecture, the methodologies, the data flow pipeline, the underlying detection logic, the current physical and computational limitations, and potential future improvements.

## Used Techniques & Methods
The project is built entirely in Python and relies on a combination of state-of-the-art computer vision libraries and robust mathematical modeling. The core components include:
*   **Ultralytics YOLO (You Only Look Once):** Utilized for real-time instance segmentation. Unlike standard object detection, this model extracts precise pixel-level binary masks and normalized polygons for each detected vehicle, which is crucial for determining true physical proximity.
*   **ByteTrack:** Employed for robust, persistent multi-object tracking across sequential frames. It associates YOLO detections frame-by-frame, allowing the system to maintain consistent vehicle identities (track IDs) and calculate continuous, stable trajectories over time.
*   **OpenCV & NumPy:** Used extensively for low-level image processing, geometric transformations, matrix operations (e.g., bitwise intersections of masks), and bounding box/polygon calculations.
*   **CustomTkinter & Pillow:** Used to build a modern, hardware-accelerated Graphical User Interface (GUI) to interact with the underlying video processing pipeline.

**Hardware Specifications:**
The software architecture is explicitly designed and optimized to run on standard consumer hardware without the need for specialized tensor accelerators. Specifically, the system is configured to execute efficiently on a laptop with the following specifications:
*   **CPU:** AMD Ryzen 5 7530U
*   **RAM:** 16GB
*   **GPU:** No dedicated video card (CPU-only inference).

## Pipeline Data Flow

The system processes video files frame-by-frame through a strictly coordinated, 4-step deterministic pipeline. The orchestrator for this flow is defined in `src/video_pipeline.py`. Below is the architectural diagram illustrating the exact flow of data through the system components.

```mermaid
flowchart TD
    classDef model fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef data fill:#dfd,stroke:#333,stroke-width:2px;
    
    A[Raw BGR Frame]:::data --> B[1. VehicleSegmenter <br> YOLO + ByteTrack]:::model
    B -->|List of DetectionResult<br>mask, polygon, bbox, track_id| C[2. VehicleStateStore]:::process
    C -.->|Stale Tracks Cleanup| C
    C -->|List of Active VehicleState<br>speed, acceleration, centroid| D[3. CollisionDetector]:::process
    A --> E[4. Renderer]:::process
    C -->|VehicleState| E
    D -->|List of CollisionEvent| E
    E --> F[Annotated Frame Output]:::data
```

## Core Modules Description

The application is structured into highly cohesive, specialized modules that guarantee separation of concerns. Below is a detailed breakdown of the primary Python files handling the logic:

### `video_pipeline.py`
This module acts as the **Orchestrator**. It initializes all other subsystems and defines the `TrafficAccidentPipeline` class. For every frame of the video, it executes the pipeline in an exact deterministic order: it calls the detector, updates the kinematic states, purges stale tracks, runs the collision logic, and finally overlays the graphical annotations using the renderer. It is the central nervous system connecting the data flow.

### `detector.py`
This file implements the **Vision Layer**. It contains the `VehicleSegmenter` class, which wraps the YOLO model and the ByteTrack algorithm. Its primary responsibility is to accept a raw BGR frame and return a list of `DetectionResult` objects. It abstracts away PyTorch tensors by converting them into standard Python lists and OpenCV-compatible NumPy arrays (polygons, bounding boxes, binary masks), ensuring that the rest of the application does not tightly couple to the Ultralytics API.

### `tracker_state.py`
This module acts as the **Memory Store**. The `VehicleStateStore` maintains active vehicles and a bounded history of motion samples. Duplicate detections for one track ID are collapsed by confidence. Motion is measured from the bottom-center bbox anchor, normalized by the observed frame gap, and invalidated after long gaps. Stale tracks are removed to prevent memory leaks and ghost collisions.

### `kinematics.py`
A pure math module defining the **Physical Rules**. It operates on generic 2D coordinates without importing internal project models. It exports vector velocity normalized by elapsed frames, scalar speed and acceleration, an exponential moving average, and the consecutive stopped-frame counter.

### `geometry.py`
A low-level spatial module defining the **Shape Computations**. It relies heavily on OpenCV and NumPy to process geometric entities. It includes functions like `safe_polygon_array` to normalize raw YOLO shapes into int32 OpenCV arrays, `polygon_to_binary_mask` to render polygons into uint8 full-frame pixel masks, `compute_centroid_from_polygon` which uses image moments ($m_{10}/m_{00}$) to find the precise center of mass, and `mask_intersection_area` which uses bitwise AND operations to calculate the exact overlapping pixel count between two vehicles.

### `collision_logic.py`
This module embodies the **Business Logic**. The `CollisionDetector` maintains persistent state for every ordered pair of tracks. It combines normalized spatial contact with recent motion, hard deceleration, and movement-to-stop transitions. Candidate evidence is confirmed over time and followed by a cooldown, so a sustained crash produces one event rather than one event per frame.

## Crash Detection Logic
The detector no longer requires exact mask overlap in one frame. Its current flow is:

1. **Track kinematics:** a bottom-center motion anchor is measured over the actual frame gap and its velocity is filtered with an exponential moving average. New tracks and tracks returning after a long gap do not provide valid impact evidence. Pre-impact motion must persist over multiple samples, rejecting isolated segmentation jumps.
2. **Spatial candidate:** bbox distance is used as an inexpensive pre-filter. Exact overlap is evaluated both in pixels and as `intersection/min(mask areas)`. Slightly dilated masks also detect adjacent silhouettes that touch without sharing pixels.
3. **Pre-existing contact guard:** if a pair is already touching when first observed, it is treated as an existing queue/occlusion and remains disarmed until a stable multi-frame separation occurs. A later re-contact can then become a collision candidate.
4. **Sustained convergence:** relative closing motion must persist across multiple consecutive frames and remain close to the beginning of the contact episode. This rejects adjacent lanes moving slowly in the same direction and prevents a late speed fluctuation from turning a long visual overlap into a crash. Persistence, however, cannot be demanded from a pair that has existed for fewer frames than the rule requires — precisely the situation created when the tracker reassigns an ID during the occlusion of an impact. For pairs observed over only a few frames, intensity therefore substitutes for persistence: a single frame whose closing speed is several times the threshold counts as sustained convergence. It replaces that one gate only; contact, prior motion and dynamic evidence remain required, and the mechanism borrows its precision from them.
5. **Role and trajectory memory:** for a moving vehicle approaching a stably stationary target, a separate longer history estimates the pre-impact heading. A late anomaly can override the normal contact-age limit only when the same contact episode contained sustained convergence and the moving vehicle shows a multi-frame reaction, including a significant trajectory deflection. This history does not extend the lifetime of isolated overlap or braking evidence.
6. **Delta-V as impact evidence:** scalar acceleration is the derivative of the velocity *magnitude*, so it is blind to an impact that deflects a vehicle without slowing it — the characteristic signature of a t-bone, 657 of the 2,027 annotated videos. The detector therefore also computes the vector velocity change, measured on the unfiltered velocity because the EMA is tuned to damp exactly the transient of interest, and uses it alongside hard deceleration rather than in its place. It is deliberately *not* used to attribute an impact between crossing trajectories: a vehicle cutting across several streams shows a large delta-V against every pair it visually overlaps, so the perturbation stops being local and the multi-pair attribution defect returns.
7. **Temporal evidence:** contact and a dynamic anomaly may occur within `impact_window_frames`. A stop is accepted only if recent history proves that at least one vehicle was moving. Strong overlap without confirmed convergence requires hard deceleration.
8. **Crossing-trajectory attribution:** when one vehicle crosses several traffic streams, its own braking cannot validate every pair it visually overlaps. Crossing pairs require a local disruption of the counterpart, such as trajectory deflection or a strong positive acceleration consistent with transferred momentum. Dilated-only contacts also require current evidence on every confirmation frame.
9. **Occlusion bridge for crossing dual stops:** a confirmed crossing approach is retained for a short, separate window when one of the tracks disappears at the impact. The bridge can emit only after a gap of at least two pair observations, real mask overlap on reappearance, prior motion, and a persistent stop of both vehicles. A simultaneous stop without a tracking gap is still rejected as normal yielding.
10. **Confirmation and cooldown:** evidence must persist for `collision_confirmation_frames`; after confirmation, the pair enters cooldown and cannot emit duplicate events. The high-confidence crossing impulse and occlusion-bridge branches already contain multi-frame evidence and can therefore complete confirmation immediately.

`CollisionEvent` records the strongest overlap, relative closing speed, kinematics, first contact frame, confirmation frame, and an explainable confidence score.

### Calibration and Evaluation
When `calibration_log_path` is enabled, the pipeline writes JSON Lines records
for pair-level diagnostics, per-track kinematics, and unique collision events. The `src.calibration`
module reads `dataset/metadata-real.csv` directly. It records the decoder's
timestamp for each prediction and compares it with `accident_time`, while
`accident_frame` remains a parallel metric and a fallback for older logs.
Predictions are matched one-to-one within a temporal tolerance and delay is
reported in frames and seconds. The `src.benchmark` runner supports
dataset splits, accident-type filters, resumable execution, throughput and
real-time-factor measurements, plus breakdowns by accident type and
environmental metadata. Pair diagnostics are optional during batch runs
because their size grows with every vehicle pair and frame.

On `95Tx-2p0O_E_00.mp4`, the previous rules emitted two false positives at
frames 159 and 174 and missed the annotated rear-end collision at frame 194.
The trajectory-reaction rule suppresses both turning-traffic events and emits
one event for tracks 2-35 at frame 198 (about 0.25 seconds after the annotation).
The regression result therefore changed from 0 TP / 2 FP / 1 FN to
1 TP / 0 FP / 0 FN for this video. This is a single-case result, not a global
dataset performance claim.

On `987C4_UdnJE_01.mp4`, the previous detector emitted five events while track
19 crossed several vehicles and did not correctly attribute the physical
impact. Pair-local crossing evidence now selects tracks 19-20 at frame 36,
three frames before the frame-39 annotation, and suppresses the four projected
overlaps plus one later dilated-only false contact. The case changes from
1 TP / 4 FP to 1 TP / 0 FP; as above, this is a regression-case result only.

On `D1eP4Bn4hDQ_0_00.mp4`, the fire truck and car brake before impact and stop
without a measurable trajectory change. Track 21 is absent for five frames at
the collision, so the ordinary short impact window previously lost the
confirmed crossing approach. The occlusion bridge reconnects that approach to
the real overlap and persistent dual stop after tracking resumes. It emits for
tracks 21-22 at frame 126, five frames (about 0.21 seconds) after the frame-121
annotation, changing this case from 0 TP / 0 FP / 1 FN to 1 TP / 0 FP / 0 FN.
The three earlier regression videos retain their previous results.

A re-measurement of these four cases on 12 September 2026 no longer reproduces
two of them. `987C4_UdnJE_01.mp4` emits a second event on pair 19-9 at frame 34
alongside the correct 19-20 attribution, and `8G56ILxFFNM_00.mp4` emits one
event on pair 2-54 at frame 416, outside the annotated window. The set as a
whole now scores 3 TP / 2 FP / 1 FN rather than the 3 TP / 0 FP the individual
case descriptions imply. The regression predates the work on pairs without
history — the mechanisms described above change none of these four videos — but
it could not be attributed to a specific change, because the JSONL logs the
original case descriptions were written from are no longer in the repository.
Each case above therefore reports a result measured at the time it was written,
not the current behaviour of the system.

The current real dataset contains 2,027 metadata rows and 2,027 corresponding
MP4 files, with no missing or duplicate paths. Its accident distribution is
680 single-vehicle, 657 t-bone, 328 rear-end, 245 sideswipe, and 117 head-on
videos. Metadata validation is included in the generated report; two current
rows place `accident_frame` exactly at `no_frames`, so timestamp-based matching
is especially important for those end-of-video annotations.

Every real-dataset video contains an annotated accident. Consequently, this
benchmark measures missed, mistimed, and duplicate detections, but cannot by
itself estimate the false-alarm rate on accident-free traffic. A representative
negative-video set is still required for deployment-level precision claims.

## Issues
While the current deterministic approach is computationally efficient, explainable, and capable of running on low-end hardware, it is subject to several physical and technical limitations:
*   **Lack of Perspective Awareness:** The kinematic calculations (speed, acceleration) are currently performed in the 2D pixel space rather than real-world Euclidean coordinates. Consequently, the apparent speed of a vehicle varies drastically depending on its distance from the camera focal point (perspective distortion), requiring very generic and potentially inaccurate thresholding.
*   **Hardware Constraints:** Running an instance segmentation model (YOLO) and a tracker entirely on a CPU (AMD Ryzen 5 7530U) heavily limits the processing framerate. A low framerate can cause the system to miss rapid kinematic changes (the spike in deceleration) that occur between the processed frames.
*   **Environmental Sensitivity:** The accuracy of the YOLO segmenter heavily degrades in adverse weather conditions (rain, fog, snow) or poor lighting (nighttime). If the masks are poorly segmented, the pixel-level overlap condition becomes unreliable.
*   **Tracker ID Switching:** If ByteTrack loses a vehicle during a severe occlusion (common during a crash) and reassigns a new ID when the vehicle reappears, the kinematic history (speed, previous centroids) is reset. This prevents the detection of "Hard Deceleration" since the velocity differentials cannot be computed across the ID switch. Repairing the switch by re-identification was implemented and measured on the real dataset, and does not work on position alone: 59% of new IDs have no track that disappeared in the preceding frames — they are vehicles entering the scene, not reassignments — and among the remainder the best available donor lies a median of two vehicle-lengths away from the extrapolated position, with no threshold separating the two populations. Extrapolating with the velocity vector performs no better than using the last known position. A useful re-identification needs an appearance descriptor, which is a different class of intervention.

## Future Improvement
To address the aforementioned issues and push the system towards enterprise-grade robustness, the following architectural and mathematical improvements are recommended:
1.  **Bird's Eye View (BEV) Transformation:** Implementing an Inverse Perspective Mapping (IPM) using a homography matrix. By manually selecting 4 ground-plane points, the system could project the 2D image coordinates into a top-down 3D metric space. Speeds and accelerations would then be calculated in real-world metric units (meters per second) rather than pixels, standardizing the detection thresholds across the entire video frame.
2.  **Hardware Acceleration via Optimization:** Integrating inference engines like ONNX Runtime, OpenVINO, or NCNN could significantly boost the YOLO model's execution speed on CPU-only environments, enabling near-real-time processing and capturing finer kinematic details.
3.  **Hybrid Deep Learning Approach:** Instead of relying entirely on deterministic rules, a temporal neural network (such as an LSTM or an action-recognition Transformer) could be trained on the extracted kinematic features (trajectories, speed vectors, overlap ratios) to classify crash patterns more dynamically, learning to ignore complex false positives.
4.  **Kalman Filter Enhancements:** Tuning the ByteTrack's internal Kalman filter state estimation to better predict a vehicle's position during severe occlusions, maintaining the track ID even when a vehicle is temporarily hidden behind a larger vehicle during a pile-up.

## Conclusions
The implemented traffic accident detection system represents a modular and explainable approach to road safety monitoring. Instance segmentation now supplies spatial candidates while filtered track history and persistent pair state provide temporal confirmation. This reduces the most obvious false positives from stationary traffic and duplicate per-frame alerts, but it does not make mask contact proof of a physical crash. Perspective calibration, video-level validation, and robust ID reassociation remain necessary before deployment.

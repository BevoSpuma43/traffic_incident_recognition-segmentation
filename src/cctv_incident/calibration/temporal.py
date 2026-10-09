"""Opt-in, bounded initial-window search; all points stay in the first-frame basis."""

from pathlib import Path

import av
import cv2
import numpy as np

from .proposals import ProposalParameters, ProposalResult, generate_proposals
from .repository import pixel_sha256


def stationary_scene(reference, image):
    """Require distributed static feature matches and <=1.5 px camera displacement.

    Do not transfer calibration through a pan, zoom, cut, or unconstrained warp.
    Insufficient visual evidence is a rejection, not evidence of a fixed camera.
    """
    if image.shape != reference.shape:
        return {"accepted": False, "reason": "dimensioni_diverse"}
    size = (
        min(960, reference.shape[1]),
        min(960, reference.shape[1]) * reference.shape[0] // reference.shape[1],
    )
    a, b = [cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_BGR2GRAY) for im in (reference, image)]
    # Exact frames are stationary even in textureless synthetic footage.
    if np.array_equal(a, b):
        return {"accepted": True, "reason": "frame_identico", "max_displacement_px": 0.0}
    orb = cv2.ORB_create(nfeatures=1600)
    keys_a, desc_a = orb.detectAndCompute(a, None)
    keys_b, desc_b = orb.detectAndCompute(b, None)
    if desc_a is None or desc_b is None or min(len(desc_a), len(desc_b)) < 16:
        return {"accepted": False, "reason": "sfondo_non_verificabile"}
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(desc_a, desc_b, k=2)
    matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.7 * p[1].distance]
    # A train feature must not supply several independent-looking correspondences.
    matches = sorted(matches, key=lambda m: m.distance)
    unique, used = [], set()
    for match in matches:
        if match.trainIdx not in used:
            unique.append(match)
            used.add(match.trainIdx)
    if len(unique) < 16:
        return {"accepted": False, "reason": "sfondo_non_verificabile"}
    src = np.float32([keys_a[m.queryIdx].pt for m in unique])
    dst = np.float32([keys_b[m.trainIdx].pt for m in unique])
    affine, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5
    )
    if affine is None or mask is None:
        return {"accepted": False, "reason": "sfondo_non_verificabile"}
    inliers = mask.ravel().astype(bool)
    coverage = abs(cv2.contourArea(cv2.convexHull(src[inliers]))) / (size[0] * size[1])
    corners = np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], float)
    displacement = float(
        np.max(np.linalg.norm(np.c_[corners, np.ones(4)] @ affine.T - corners, axis=1))
        * reference.shape[1]
        / size[0]
    )
    accepted = bool(
        inliers.sum() >= 16 and inliers.mean() >= 0.6 and coverage >= 0.12 and displacement <= 1.5
    )
    return {
        "accepted": accepted,
        "reason": "camera_stabile" if accepted else "camera_mossa_o_sfondo_insufficiente",
        "inliers": int(inliers.sum()),
        "background_coverage": float(coverage),
        "max_displacement_px": displacement,
    }


def search_initial_frames(
    source, reference, roi=None, *, seconds=5.0, sample_count=6, parameters=None
):
    """Search <=12 frames / <=15 seconds / <=900 decoded frames, without labels.

    Keep the original reference, ROI and pixel hash. Additional images provide
    evidence only; no synthetic background or new reference identity is substituted.
    """
    if (
        not np.isfinite(seconds)
        or not 0 < seconds <= 15
        or not isinstance(sample_count, int)
        or not 2 <= sample_count <= 12
    ):
        raise ValueError("Usa un intervallo di 0–15 secondi e da 2 a 12 campioni")
    parameters = ProposalParameters.model_validate(parameters or {})
    path = Path(source)
    before = path.stat()
    result = generate_proposals(reference, roi, parameters=parameters)
    candidates = list(result.candidates)
    evidence = {}
    targets = np.linspace(0, seconds, sample_count)
    sampled, target, reference_verified, decoded_count = [], 1, False, 0
    combined_mask = result.mask.copy()
    reference_hash = pixel_sha256(reference)
    try:
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index >= 900:
                    break
                decoded_count += 1
                if index == 0:
                    image = frame.to_ndarray(format="bgr24")
                    if pixel_sha256(image) != reference_hash:
                        raise ValueError(
                            "Il video è cambiato rispetto al fotogramma di riferimento"
                        )
                    reference_verified = True
                    start = frame.time
                    if start is None:
                        sampled.append(
                            {"frame_index": index, "accepted": False, "reason": "timestamp_assente"}
                        )
                        break
                    continue
                if frame.time is None:
                    continue
                elapsed = float(frame.time - start)
                if elapsed > seconds or target >= len(targets):
                    break
                if elapsed < targets[target]:
                    continue
                while target < len(targets) and targets[target] <= elapsed:
                    target += 1
                image = frame.to_ndarray(format="bgr24")
                stability = stationary_scene(reference, image)
                row = {"frame_index": index, "timestamp_s": float(frame.time), **stability}
                sampled.append(row)
                if not stability["accepted"]:
                    continue
                proposal = generate_proposals(image, roi, parameters=parameters)
                row["candidate_count"] = len(proposal.candidates)
                combined_mask = cv2.bitwise_or(combined_mask, proposal.mask)
                image_hash = pixel_sha256(image)
                if proposal.candidates:
                    evidence[image_hash] = cv2.resize(
                        image, (result.preview.shape[1], result.preview.shape[0])
                    )
                for candidate in proposal.candidates:
                    data = {
                        **candidate.diagnostics,
                        "source_frame": {
                            "frame_index": index,
                            "timestamp_s": float(frame.time),
                            "pixel_sha256": image_hash,
                        },
                        "stationarity": stability,
                    }
                    candidates.append(candidate.model_copy(update={"diagnostics": data}))
    except av.FFmpegError as exc:
        raise ValueError("Impossibile leggere i fotogrammi iniziali del video") from exc
    if not reference_verified:
        raise ValueError("Il video non contiene un fotogramma di riferimento leggibile")
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Il video è cambiato durante la ricerca")
    candidates.sort(
        key=lambda c: (-c.geometric_quality, -abs(cv2.contourArea(np.float32(c.points_px))))
    )
    persistent, unconfirmed = [], 0
    for candidate in candidates:
        observations = {}
        for other in candidates:
            if _same_reference(candidate, other):
                source_frame = other.diagnostics.get("source_frame", {})
                observations[source_frame.get("frame_index", 0)] = source_frame.get("timestamp_s")
        # A single bright moving surface must not become a new temporal reference.
        # Retain the first-frame fallback only if no other frame could be verified.
        if len(observations) < 2 and (
            candidate.diagnostics.get("source_frame") or any(row["accepted"] for row in sampled)
        ):
            unconfirmed += 1
            continue
        persistent.append(
            candidate.model_copy(
                update={
                    "diagnostics": {
                        **candidate.diagnostics,
                        "supporting_frames": [
                            {"frame_index": i, "timestamp_s": t}
                            for i, t in sorted(observations.items())
                        ],
                        "temporal_consistency_verified": len(observations) >= 2,
                    }
                }
            )
        )
    unique = []
    for candidate in persistent:
        if any(_same_reference(candidate, c) for c in unique):
            continue
        unique.append(candidate)
    chosen = unique[: parameters.max_candidates]
    groups = [c for c in unique if c.reference_type.startswith("repeated_")]
    if (
        groups
        and parameters.max_candidates >= 3
        and not any(c.reference_type.startswith("repeated_") for c in chosen)
    ):
        chosen[-1] = groups[0]
    chosen = tuple(
        c.model_copy(update={"candidate_id": f"candidate_{i}"}) for i, c in enumerate(chosen, 1)
    )
    preview = cv2.resize(reference, (result.preview.shape[1], result.preview.shape[0]))
    ratios = np.array(
        [reference.shape[1] / preview.shape[1], reference.shape[0] / preview.shape[0]]
    )
    for candidate in chosen:
        quad = np.rint(np.asarray(candidate.points_px) / ratios).astype(np.int32)
        cv2.polylines(preview, [quad], True, (0, 255, 0), 2)
        cv2.putText(
            preview,
            candidate.candidate_id,
            tuple(quad[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    hashes = {c.diagnostics.get("source_frame", {}).get("pixel_sha256") for c in chosen}
    diagnostics = {
        **result.diagnostics,
        "temporal_search": {
            "seconds": seconds,
            "sample_count": sample_count,
            "max_decoded_frames": 900,
            "decoded_frames": decoded_count,
            "unconfirmed_observations": unconfirmed,
            "frames": sampled,
        },
        "candidate_types": {
            kind: sum(c.reference_type == kind for c in chosen)
            for kind in sorted({c.reference_type for c in chosen})
        },
        "reason": "Scegli una proposta e controlla punti e misure."
        if chosen
        else "Nessun riferimento sufficientemente stabile nei fotogrammi esaminati. Usa i clic o le coordinate numeriche.",
    }
    return ProposalResult(
        chosen,
        diagnostics,
        combined_mask,
        preview,
        {key: im for key, im in evidence.items() if key in hashes},
    )


def _same_reference(a, b):
    if a.reference_type.startswith("repeated_") != b.reference_type.startswith("repeated_"):
        return False
    qa, qb = np.float32(a.points_px), np.float32(b.points_px)
    overlap, _ = cv2.intersectConvexConvex(qa, qb)
    return overlap / max(abs(cv2.contourArea(qa)), abs(cv2.contourArea(qb)), 1) > 0.8

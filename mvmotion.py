"""Camera motion read out of the encoder's own motion vectors, instead of recomputed from pixels.

Every P/B frame in an H.264 stream already carries, per 16x16 block, "this block came from THERE
in the previous frame". That is a dense motion field the encoder was paid to compute — and it
sits unused in every file we open. motion.py currently throws it away and recomputes the same
thing with Lucas-Kanade, which is slower and, more importantly, falls apart on exactly the
footage that matters: Julian's reference whips into near-blackness, and there is nothing in a
black smeared frame for a corner detector to grab. The encoder had no such trouble — it tracked
50-125px of motion straight through it.

Same output shape as motion.track_shot(), so it drops in as a replacement.

Two things to know about the vectors:
  - `source` is negative for blocks predicted from a PAST frame and positive for a FUTURE one
    (B-frames predict both ways). Only past-referencing vectors describe forward motion.
  - motion_x/motion_y are in quarter-pixels; divide by motion_scale.
"""
import av
import cv2
import numpy as np

MIN_BLOCKS = 20

# Subject-vs-camera separation. RANSAC gives us the camera; the blocks it REJECTS are the
# thing moving independently of the camera. A block whose residual (its real displacement minus
# what the camera move predicts) exceeds the RANSAC reprojection tolerance was not explained by
# the camera — it is subject motion. Same 3px tolerance so the two agree on what "the camera
# explains".
SUBJ_MIN_PX = 3.0        # residual above this = not explained by the camera move
SUBJ_MIN_BLOCKS = 4      # fewer moving blocks than this is encoder noise, not an object


def _subject_stats(M, src, dst, w, h):
    """Independent (non-camera) motion in one frame, from the blocks the camera model rejects.

    Subtract the fitted camera transform from every block's displacement; whatever is left is
    that block moving on its own. A moving subject leaves a cluster of blocks with a large,
    COHERENT residual; encoder noise leaves a scatter of small residuals pointing every way.
    Returns (coverage, speed, dx, dy, coherence) or None when nothing moves independently.
      coverage  — fraction of blocks moving independently of the camera
      speed     — median residual magnitude, as a fraction of frame width
      dx, dy    — mean residual direction, as fractions of frame width/height
      coherence — how aligned those residuals are (1 = one moving object, ~0 = noise)
    """
    pred = src @ M[:, :2].T + M[:, 2]        # where the camera move alone would put each block
    resid = dst - pred                       # ~0 for background; the subject's own motion for a mover
    rmag = np.hypot(resid[:, 0], resid[:, 1])
    moving = rmag > SUBJ_MIN_PX
    n = int(moving.sum())
    if n < SUBJ_MIN_BLOCKS:
        return None

    rv = resid[moving]
    coverage = n / len(rmag)
    speed = float(np.median(rmag[moving])) / w
    mean_vec = rv.mean(0)
    units = rv / (np.hypot(rv[:, 0], rv[:, 1])[:, None] + 1e-6)
    coherence = float(np.hypot(*units.mean(0)))
    return (float(coverage), speed,
            float(mean_vec[0]) / w, float(mean_vec[1]) / h, coherence)


def _frame_transforms(video_path):
    """Per-frame (camera transform, intra ratio, subject stats) from the encoder's vectors.

    The intra ratio is a free bonus: it is the fraction of the frame the encoder could NOT
    predict at all. At a cut it goes to 1.0 — that is what an I-frame IS. The subject stats are
    the other free bonus — the blocks RANSAC threw away as not-the-camera (see _subject_stats).
    """
    c = av.open(str(video_path))
    s = c.streams.video[0]
    s.codec_context.options = {"flags2": "+export_mvs"}

    out = []
    for frame in c.decode(s):
        blocks = (frame.width // 16) * (frame.height // 16)
        mv = frame.side_data.get("MOTION_VECTORS")
        if mv is None:
            out.append((None, 1.0, None))    # nothing predicted: a keyframe
            continue

        a = mv.to_ndarray()
        past = a[a["source"] < 0]
        intra = max(0.0, 1.0 - len(past) / max(blocks, 1))
        if len(past) < MIN_BLOCKS:
            out.append((None, intra, None))
            continue

        src = np.stack([past["src_x"], past["src_y"]], 1).astype(np.float32)
        dst = np.stack([past["dst_x"], past["dst_y"]], 1).astype(np.float32)

        # RANSAC for the same reason motion.py uses it: the DOMINANT motion is the camera.
        # A subject moving through frame is a minority of blocks and gets rejected.
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if M is None or inl is None or int(inl.sum()) < MIN_BLOCKS:
            out.append((None, intra, None))
            continue
        subj = _subject_stats(M, src, dst, frame.width, frame.height)
        out.append((M, intra, subj))

    c.close()
    return out, float(s.average_rate or 30.0), (s.codec_context.width, s.codec_context.height)


def track_shot(video_path, start, end, max_keys=90, cache=None):
    """Cumulative camera transform across one shot — same contract as motion.track_shot()."""
    tf, fps, (w, h) = cache if cache else _frame_transforms(video_path)

    lo = int(round(start * fps))
    hi = min(len(tf), int(round(end * fps)))
    n = hi - lo
    if n < 2:
        return []

    step = max(1, n // max_keys)
    cum = np.eye(3)
    keys = []
    for i in range(n):
        M = tf[lo + i][0] if lo + i < len(tf) else None
        if M is not None and i > 0:
            # new step on the LEFT — same composition order as motion.py. Getting this backwards
            # silently rotates the accumulated translation and the pan comes out wrong.
            cum = np.vstack([M, [0, 0, 1]]) @ cum

        if i % step == 0 or i == n - 1:
            keys.append([
                round(i / fps, 4),
                round(float(cum[0, 2]) / w, 5),
                round(float(cum[1, 2]) / h, 5),
                round(float(np.hypot(cum[0, 0], cum[1, 0])), 5),
                round(float(np.degrees(np.arctan2(cum[1, 0], cum[0, 0]))), 3),
            ])
    return keys if len(keys) >= 2 else []


def intra_ratios(video_path):
    """Per-frame fraction of the picture the encoder could not predict. 1.0 == a cut."""
    tf, fps, _ = _frame_transforms(video_path)
    return [r for _, r, _ in tf], fps


# A shot has independent subject motion only if it PERSISTS. One coherent frame is a fluke; a
# subject moving across a shot leaves coherent residuals frame after frame. And we only trust
# frames the encoder could actually predict — a frame mid-transition (high intra) is smeared
# wreckage and its "subject motion" is meaningless.
SUBJ_COV_MIN = 0.02      # a real subject occupies at least this fraction of the frame
SUBJ_COH_MIN = 0.45      # its residuals must point roughly one way (an object, not noise)
SUBJ_INTRA_MAX = 0.5     # ignore transition-wreckage frames
SUBJ_PERSIST_MIN = 0.25  # active in at least this fraction of the shot's usable frames


def subject_shot(cache, start, end):
    """Subject motion across one shot, distinct from the camera. None if the shot is camera-only.

    Aggregates the per-frame residuals (_subject_stats) over the shot, keeping only frames the
    encoder could predict and where the independent motion is both present (coverage) and
    coherent (points one way). This mirrors the two-gates-must-agree discipline that fixed the
    dissolve and speed-ramp detectors: coverage alone fires on noise; coverage AND coherence
    together mean an object.

    Returns {coverage, speed, direction_deg, persistence} or None.
    """
    tf, fps, (w, h) = cache
    lo = int(round(start * fps))
    hi = min(len(tf), int(round(end * fps)))

    usable, active, vecs, covs, spds = 0, 0, [], [], []
    for i in range(lo, hi):
        M, intra, subj = tf[i]
        if M is None or intra > SUBJ_INTRA_MAX:
            continue
        usable += 1
        if subj is None:
            continue
        cov, spd, dx, dy, coh = subj
        if cov >= SUBJ_COV_MIN and coh >= SUBJ_COH_MIN:
            active += 1
            vecs.append((dx, dy))
            covs.append(cov)
            spds.append(spd)

    if usable < 2 or active < max(2, SUBJ_PERSIST_MIN * usable):
        return None

    v = np.array(vecs)
    mean_dir = v.mean(0)
    return {
        "coverage": round(float(np.median(covs)), 4),
        "speed": round(float(np.median(spds)), 5),
        "direction_deg": round(float(np.degrees(np.arctan2(mean_dir[1], mean_dir[0]))), 1),
        "persistence": round(active / usable, 3),
    }


def step_vectors_from_cache(cache):
    """Per-frame camera step (vx, vy) in pixels, from an existing _frame_transforms cache.

    This is the instantaneous camera motion — the frame-to-frame translation the encoder saw —
    which is what tells a cut ON motion apart from a cut on stillness, and which way the motion
    points across the seam. Frames the encoder couldn't fit (cuts, no vectors) come back NaN so
    the caller skips them rather than reading them as "not moving".
    """
    tf, fps, (w, h) = cache
    v = np.array([[float(M[0, 2]), float(M[1, 2])] if M is not None else [np.nan, np.nan]
                  for M, _, _ in tf], dtype=float)
    return v, fps, (w, h)


def step_series(video_path):
    """Per-frame camera step (vx, vy) in pixels for one of the user's clips. Decodes once."""
    return step_vectors_from_cache(_frame_transforms(video_path))


def speed_series(video_path):
    """Per-frame apparent motion magnitude, straight from the encoder's vectors.

    This is what velocity.py needs, and asking the bitstream for it rather than recomputing it
    with dense optical flow matters: on real, dark, motion-blurred footage the optical-flow speed
    curve is noise, and it invented a speed ramp on 15 of 21 shots — every one with the same
    0.3x -> 2.3x shape, which is not an edit, it is a systematic artefact.

    Returns (speeds, intra, fps). Frames where intra is high are transition wreckage and carry no
    usable motion — the caller should drop them rather than believe them.
    """
    c = av.open(str(video_path))
    s = c.streams.video[0]
    s.codec_context.options = {"flags2": "+export_mvs"}
    w = s.codec_context.width or 1

    speeds, intra = [], []
    for frame in c.decode(s):
        blocks = (frame.width // 16) * (frame.height // 16)
        mv = frame.side_data.get("MOTION_VECTORS")
        if mv is None:
            speeds.append(0.0)
            intra.append(1.0)
            continue
        a = mv.to_ndarray()
        past = a[a["source"] < 0]
        intra.append(max(0.0, 1.0 - len(past) / max(blocks, 1)))
        if len(past) == 0:
            speeds.append(0.0)
            continue
        sc = past["motion_scale"].astype(float)
        dx = past["motion_x"] / sc
        dy = past["motion_y"] / sc
        mag = np.hypot(dx, dy)
        # p90, NOT the mean. The mean answers "how much of the frame is moving", which RISES as a
        # subject walks into shot at a perfectly constant speed — and that fake acceleration put a
        # false speed ramp on steady footage. The p90 answers "how fast is the moving stuff
        # moving", which is what a retime actually changes.
        speeds.append(float(np.percentile(mag, 90)) / w)

    fps = float(s.average_rate or 30.0)
    c.close()
    return np.array(speeds), np.array(intra), fps

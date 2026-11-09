"""Recover the reference's speed ramps — the defining move of the velocity/phonk edit.

The genre rushes through the dull part of a shot and slams into slow motion on the beat.
That is a RETIME, and it is measurable.

The insight that makes it possible: we never need the reference's ABSOLUTE speed. We are
retiming the user's own clip, so all we need is the reference's RELATIVE speed profile —
fast here, slow there — and relative speed is exactly what apparent motion gives us.
(Absolute speed genuinely is unrecoverable: 60fps played at 30 and 30fps slowed to 0.5x are
pixel-identical. That impossibility is real but NARROW, and it does not touch ramps.)

Method: dense optical flow per frame gives apparent speed. Divide by the shot's own median
to get relative speed r(t). Then source time advances in proportion to r, so

    src(t) = src_start + span * ( integral of r up to t ) / ( integral of r over the shot )

which redistributes time inside the shot without needing any extra footage — the same
number of source seconds get consumed, just unevenly. That curve goes straight into After
Effects as Time Remap keyframes.

Known ambiguity, accepted deliberately: a subject genuinely accelerating in frame looks the
same as a retime. We cannot separate them, and for copying an edit's speed RHYTHM we do not
need to.
"""
import cv2
import mvmotion
import numpy as np

# Dense flow is the expensive part, but do NOT shrink this further: optical flow
# systematically UNDER-measures sub-pixel displacement, so a slow section shot at low
# resolution reads as slower than it is and the recovered ramp comes out exaggerated.
# Bigger frames keep the slow end honest.
FLOW_W = 320
SMOOTH = 5            # frames of moving average — raw flow is noisy
# Biased toward ABSTAINING (Julian: if it can't clearly read the edit, add nothing). A borderline
# ramp now reads as steady rather than getting a Time Remap it might not deserve. Raise further to
# be stricter; lower only if genuine ramps are being missed.
MIN_SPREAD = 3.3      # p90/p20 of relative speed. A real slam guts the motion; drift doesn't.
MIN_BEND = 0.40       # how badly a straight line fails to explain the speed curve
CLAMP = (0.30, 3.5)   # never ask AE for a wilder retime than this


def speed_curve(video_path, start, end):
    """Apparent motion per frame across one shot, with the TRANSITION FRAMES CUT OFF.

    That trimming is the whole difference between a working detector and a broken one.

    Every shot in a whip-heavy edit BEGINS in the wreckage of the transition that brought it in:
    dark, motion-blurred frames where apparent motion reads low, which then "recovers" as the
    image settles. Measured naively that is a textbook slow-to-fast ramp, and the detector duly
    found one on 15 of 21 shots — every single one ramping 0.2x to ~3x, which is not an edit,
    it is an artefact. In After Effects those became Time Remap curves and the footage stuttered.

    So: drop the frames at each end of the shot that are markedly blurrier than the shot's own
    core, and measure the retime on what is left.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    n = max(2, int(round((end - start) * fps)))

    grays, sharp, lum = [], [], []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        g = cv2.cvtColor(cv2.resize(f, (FLOW_W, max(2, int(h * FLOW_W / w)))), cv2.COLOR_BGR2GRAY)
        grays.append(g)
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        lum.append(float(g.mean()))
    cap.release()
    if len(grays) < 8:
        return np.array([]), fps

    # A whip/flash transition frame is DARK *AND* SOFT. Both conditions are needed: keying on
    # softness alone also throws away legitimately low-detail footage — it ate the entire
    # slow-motion half of the ground-truth clip, where the moving object had simply left frame.
    ks, kl = float(np.median(sharp)), float(np.median(lum))
    trash = [s < 0.5 * ks and l < 0.7 * kl for s, l in zip(sharp, lum)]
    cap_trim = max(1, int(0.30 * len(grays)))       # never eat more than 30% off an end

    lo, hi = 0, len(grays) - 1
    while lo < hi and trash[lo] and lo < cap_trim:
        lo += 1
    while hi > lo and trash[hi] and hi > len(grays) - 1 - cap_trim:
        hi -= 1
    if hi - lo < 6:                      # nothing left but transition: no honest ramp to read
        return np.array([]), fps

    speeds = []
    for i in range(lo + 1, hi + 1):
        flow = cv2.calcOpticalFlowFarneback(grays[i - 1], grays[i], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        speeds.append(float(np.hypot(flow[..., 0], flow[..., 1]).mean()))
    return np.array(speeds), fps


def _smooth(x, k=SMOOTH):
    if len(x) < k or k < 2:
        return x
    pad = np.pad(x, (k // 2, k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")[:len(x)]


def bitstream_speed(video_path, start, end, cache=None):
    """Apparent speed per frame from the encoder's motion vectors, transition frames removed.

    Preferred over the optical-flow version above for the same reason the camera tracker moved to
    the bitstream: on real dark, motion-blurred footage the flow-based curve is noise. Frames the
    encoder could barely predict (high intra) are transition wreckage and get dropped rather than
    believed.
    """
    speeds, intra, fps = cache if cache else mvmotion.speed_series(video_path)
    lo = int(round(start * fps))
    hi = min(len(speeds), int(round(end * fps)))
    if hi - lo < 8:
        return np.array([]), fps

    seg, seg_intra = speeds[lo:hi], intra[lo:hi]
    keep = seg_intra < 0.5                 # drop the frames the encoder had to give up on
    if keep.sum() < 8:
        return np.array([]), fps
    return seg[keep], fps


def profile(video_path, start, end, max_keys=60, cache=None):
    """The shot's retime as a NORMALISED curve: [[out_frac, src_frac], ...], both in 0..1.

    Normalised because the reference shot and the user's replacement clip have different
    durations and start points — the builder maps this onto whatever footage goes in the slot.

    Returns [] when the shot plays at a steady speed. A flat shot must not be littered with a
    time remap that does nothing but risk breaking it.
    """
    speeds, fps = np.array([]), 30.0
    try:
        speeds, fps = bitstream_speed(video_path, start, end, cache=cache)
    except Exception:
        pass
    if len(speeds) < 6:                    # no usable vectors: fall back to pixels
        speeds, fps = speed_curve(video_path, start, end)
    if len(speeds) < 6:
        return []

    s = _smooth(speeds)
    base = float(np.median(s))
    if base <= 1e-4:                       # a static shot: no motion to measure speed from
        return []

    raw_r = s / base

    # Is this actually a retime, or just content? Apparent motion also changes when a subject
    # walks off frame at a perfectly constant playback speed — so a spread in speed alone is
    # NOT enough, and treating it as enough put a false ramp on steady footage. Two
    # independent things have to agree, and they separate cleanly:
    #
    #   spread — a velocity edit SLAMS: its slow section has almost no motion left, so the
    #            p90/p20 ratio explodes (millions in testing). Content drift reached 2.25.
    #   bend   — content drift is smooth, so a straight line explains it well (residual 0.28).
    #            A slam is a step, and a line cannot fit it (0.48).
    p90, p20 = np.percentile(raw_r, 90), max(float(np.percentile(raw_r, 20)), 0.02)
    t = np.arange(len(raw_r))
    line = np.polyval(np.polyfit(t, raw_r, 1), t)
    bend = float(np.abs(raw_r - line).mean() / max(raw_r.mean(), 1e-6))

    if p90 / p20 < MIN_SPREAD or bend < MIN_BEND:
        return []

    r = np.clip(raw_r, *CLAMP)

    # Source time advances in proportion to relative speed. Normalising the integral to the
    # shot's own duration means the same total source footage is consumed — the retime
    # redistributes time WITHIN the shot rather than demanding extra frames.
    cum = np.concatenate([[0.0], np.cumsum(r)])
    cum /= cum[-1]

    step = max(1, len(cum) // max_keys)
    keys = [[round(i / (len(cum) - 1), 4), round(float(cum[i]), 4)]
            for i in range(0, len(cum), step)]
    if keys[-1][0] < 1.0:
        keys.append([1.0, 1.0])
    return keys if len(keys) > 2 else []


def describe(keys):
    """Slowest and fastest playback rate in the curve, relative to normal. For reporting."""
    if len(keys) < 3:
        return None
    rates = [(b[1] - a[1]) / (b[0] - a[0])
             for a, b in zip(keys, keys[1:]) if b[0] - a[0] > 1e-6]
    if not rates:
        return None
    return {"slowest": round(float(min(rates)), 2), "fastest": round(float(max(rates)), 2)}

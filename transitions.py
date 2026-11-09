"""Classify what actually happens at each cut, and find freeze frames inside shots.

Until now every cut was treated as a hard cut, which silently threw away every dissolve,
fade and whip in the reference. Each of those leaves a distinct, measurable signature in
the frames around the boundary:

  dissolve  — the frames between A and B are literal linear blends of A and B. That is a
              very strong signal: solve for the blend weight per frame and see if the
              residual collapses. Nothing else in video looks like that.
  fade      — luminance dives to (near) black and comes back.
  flash     — luminance spikes far above the shot's own median.
  whip      — the frames at the boundary go simultaneously high-motion and *blurry*;
              a hard cut is high-motion but stays sharp.
  hard      — none of the above.

Freeze frames are the same trick applied inside a shot: consecutive frames that are
near-identical mean the editor held a frame.

Honest limit: a slow dissolve changes content gradually, so the cut detector can miss the
boundary entirely — we can only classify boundaries it found. Dissolves longer than about
a second are the ones at risk.
"""
import cv2
import numpy as np

SMALL = (80, 45)     # everything below is computed on this; detail buys nothing here

# A flash is a brief near-WHITE spike. These gates (absolute near-white peak + big jump above the
# local level) stop a plain dark->bright cut being mistaken for a flash — the over-firing Julian saw.
FLASH_MIN_LUM = 200  # near-white, on 0-255 mean luminance
FLASH_MIN_JUMP = 55  # peak must be this far above the local median


def frame_features(video_path):
    """One pass over the video: small grayscale frame, luminance, blurriness, displacement."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, lum, sharp, shift = [], [], [], []
    win = cv2.createHanningWindow(SMALL, cv2.CV_32F)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, SMALL), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lum.append(float(g.mean()))
        # variance of the Laplacian: the standard sharpness proxy. A whip is blurred, so
        # this collapses — but so does a dissolve, which is why it can't be the whole test.
        sharp.append(float(cv2.Laplacian(g, cv2.CV_32F).var()))
        # how far the frame moved since the last one, as a fraction of frame width. This
        # is what actually separates a whip (the frame flies across) from a dissolve (the
        # frame doesn't move, it fades).
        if frames:
            (dx, dy), _ = cv2.phaseCorrelate(frames[-1], g, win)
            shift.append(float(np.hypot(dx, dy)) / SMALL[0])
        else:
            shift.append(0.0)
        frames.append(g)
    cap.release()
    return fps, np.array(frames), np.array(lum), np.array(sharp), np.array(shift)


def _blend_weights(frames, cut, half):
    """Solve the A->B blend weight for each frame around the cut.

    A frame in a dissolve is literally (1-alpha)*A + alpha*B, so projecting it onto the
    A->B axis recovers alpha, and the residual says how much of the frame that blend fails
    to explain. Returns (alphas, residual_ratio).

    The catch that matters: a HARD cut also fits perfectly — every frame before it is pure
    A (alpha 0) and every frame after is pure B (alpha 1), residual zero. So a low residual
    alone means nothing. What separates a dissolve is that alpha *dwells in the middle*;
    a hard cut steps 0 -> 1 with nothing in between. The caller checks for that.
    """
    A, B = frames[cut - half - 1], frames[cut + half]
    d = B - A
    denom = float((d * d).sum())
    if denom < 1e-6:                      # A and B identical: nothing to dissolve between
        return [], 1.0

    alphas, resid = [], []
    for i in range(cut - half, cut + half):
        F = frames[i] - A
        alpha = float((F * d).sum() / denom)
        err = F - alpha * d
        alphas.append(alpha)
        resid.append(float(np.sqrt((err * err).mean())))

    spread = float(np.sqrt((d * d).mean()))
    return alphas, [r / max(spread, 1e-6) for r in resid]


def classify_cut(frames, lum, sharp, shift, cut_frame, fps, lo_bound, hi_bound, max_len=30):
    """What kind of transition happens at this cut? Returns (type, duration_seconds).

    lo_bound/hi_bound are the neighbouring cut frames — the window never crosses them, or
    we'd be measuring a blend across three different shots.
    """
    n = len(frames)
    if not (1 <= cut_frame < n - 1):
        return "hard", 0.0

    max_half = min(max_len // 2,
                   (cut_frame - lo_bound) // 2,
                   (hi_bound - cut_frame) // 2,
                   cut_frame - 1,
                   n - cut_frame - 1)
    if max_half < 2:
        return "hard", 0.0

    window = slice(cut_frame - max_half, cut_frame + max_half)
    base_lum = float(np.median(lum))

    # fade through black — luminance collapses across the boundary
    if lum[window].min() < max(0.14 * base_lum, 6.0):
        dark = int((lum[window] < max(0.35 * base_lum, 14.0)).sum())
        return "fade", round(max(dark, 1) / fps, 3)

    # flash — a brief near-WHITE spike, not merely "brighter than the neighbour". Requiring an
    # absolute near-white peak (not just 1.5x the local median) stops a plain dark->bright cut, or
    # a cut into a brighter-lit shot, from being mis-read as a flash.
    local = float(np.median(lum[window]))
    peak = float(lum[window].max())
    if peak >= FLASH_MIN_LUM and (peak - local) >= FLASH_MIN_JUMP and peak > 1.5 * max(local, 1.0):
        hot = int((lum[window] > 1.25 * local).sum())
        return "flash", round(max(hot, 1) / fps, 3)

    # Whip — the frame physically flies across, and blurs while doing it. This is checked
    # BEFORE dissolve because a whip's smeared frames also project onto the A->B axis and
    # will happily masquerade as a blend.
    near = slice(max(0, cut_frame - max_half), min(n, cut_frame + max_half))
    med_sharp = float(np.median(sharp))
    # Stricter than before (0.10/0.7 -> 0.12/0.62): a whip has to CLEARLY fly and blur, or it reads
    # as a plain cut. Prefer a missed whip over stamping a Directional Blur on a normal cut.
    if float(shift[near].max()) > 0.12 and float(sharp[near].min()) < 0.62 * med_sharp:
        fast = int((shift[near] > 0.05).sum())
        return "whip", round(max(fast, 1) / fps, 3)

    # Dissolve. Two traps here, both learned the hard way:
    #  1. The window must be wide enough that its endpoints sit OUTSIDE the dissolve. Take
    #     a half-blended frame as the "pure B" endpoint and the blend axis is wrong.
    #  2. The residual cannot be tight. Footage MOVES during a dissolve, and a pure A->B
    #     blend cannot explain that motion, so a perfectly real dissolve still leaves a
    #     residual around 0.5. The ramp shape is the signal; the residual is only a sanity
    #     bound.
    # Dissolve needs two independent things to agree, because either one alone lies:
    #
    #  (a) the blend weight ramps monotonically through the middle. Necessary but NOT
    #      sufficient — camera drift smears a hard cut's weights into fake mid values.
    #  (b) image sharpness DIPS. This is the one that actually works. A dissolve's middle
    #      frames are two uncorrelated shots averaged together, and averaging uncorrelated
    #      detail destroys it — measured sharpness collapses to about half. A hard cut has
    #      no such frame and its sharpness never dips (>0.94 of median in testing, versus
    #      0.54 for a real dissolve). Residual-based tests were tried first and do NOT
    #      separate the two: drift dominates the residual.
    if float(sharp[near].min()) >= 0.80 * med_sharp:
        return "hard", 0.0

    for half in range(3, max_half + 1):        # smallest window that fits = tightest duration
        alphas, _ = _blend_weights(frames, cut_frame, half)
        if not alphas:
            continue
        mid = sum(1 for a in alphas if 0.10 < a < 0.90)
        monotonic = all(alphas[i + 1] >= alphas[i] - 0.15 for i in range(len(alphas) - 1))
        if mid >= 3 and monotonic:
            return "dissolve", round(mid / fps, 3)

    return "hard", 0.0


def find_freezes(frames, fps, start_f, end_f, still=0.55, min_frames=3):
    """Runs of near-identical frames inside a shot — the editor held a frame."""
    freezes, run_start = [], None
    for i in range(start_f + 1, min(end_f, len(frames))):
        d = float(np.abs(frames[i] - frames[i - 1]).mean())
        if d < still:
            run_start = i - 1 if run_start is None else run_start
        else:
            if run_start is not None and i - run_start >= min_frames:
                freezes.append([round((run_start - start_f) / fps, 3),
                                round((i - start_f) / fps, 3)])
            run_start = None
    if run_start is not None and min(end_f, len(frames)) - run_start >= min_frames:
        freezes.append([round((run_start - start_f) / fps, 3),
                        round((min(end_f, len(frames)) - start_f) / fps, 3)])
    return freezes


def analyze_transitions(video_path, shots):
    """Per shot: what transition brings it in, and does it hold any frames?"""
    fps, frames, lum, sharp, shift = frame_features(video_path)
    if len(frames) < 3:
        return [{"index": s["index"], "type": "hard", "duration": 0.0, "freezes": []}
                for s in shots]

    out = []
    for i, s in enumerate(shots):
        start_f = int(round(s["start"] * fps))
        end_f = int(round(s["end"] * fps))
        if s["index"] == 0:
            kind, dur = "hard", 0.0          # nothing precedes the first shot
        else:
            lo = int(round(shots[i - 1]["start"] * fps))
            hi = int(round(shots[i + 1]["start"] * fps)) if i + 1 < len(shots) else len(frames) - 1
            kind, dur = classify_cut(frames, lum, sharp, shift, start_f, fps, lo, hi)
        out.append({
            "index": s["index"],
            "type": kind,
            "duration": dur,
            "freezes": find_freezes(frames, fps, start_f, end_f),
        })
    return out

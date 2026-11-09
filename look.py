"""Frame-geometry and channel effects: letterbox, vignette, RGB split.

These are the cheap, high-visibility things the tool was blind to. All three leave signals
that are trivially separable from content — which is what makes them worth doing before any
of the hard stuff.

Note on what is NOT here: beat-driven shake and zoom punches need no detector. motion.py
already measures the reference's real per-frame camera movement, so a jolt on the beat is
already captured and already lands on the footage as keyframes. Adding a "shake on beat"
effect on top would be inventing motion we have already copied.
"""
import cv2
import numpy as np

SAMPLES = 14


def _sample(video_path, n=SAMPLES):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def detect_letterbox(video_path, dark=18, cover=0.98):
    """Black bars at the frame edges.

    A bar has to be dark in EVERY sampled frame — a single dark shot is not a letterbox.
    Returns edge thicknesses as fractions of the frame, or None if there are no bars.
    """
    frames = _sample(video_path)
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    def run(get_line, count):
        n = 0
        for i in range(count // 3):          # a bar can never be a third of the frame
            if all(get_line(g, i).mean() < dark for g in grays):
                n += 1
            else:
                break
        return n

    top = run(lambda g, i: g[i, :], h)
    bottom = run(lambda g, i: g[h - 1 - i, :], h)
    left = run(lambda g, i: g[:, i], w)
    right = run(lambda g, i: g[:, w - 1 - i], w)

    if max(top, bottom, left, right) < 2:
        return None
    return {"top": round(top / h, 4), "bottom": round(bottom / h, 4),
            "left": round(left / w, 4), "right": round(right / w, 4)}


def detect_vignette(video_path, letterbox=None, min_strength=0.30):
    """Corners consistently darker than the centre.

    Content makes this noisy (a dark sky, a bright face), so we take the MEDIAN corner/centre
    ratio across many frames. A real vignette is a constant optical property; a dark corner in
    one shot is not.

    letterbox must be passed in and cropped away first. Black bars ARE dark corners — measure
    without cropping and every letterboxed video reports a vignette it does not have.

    ⚠️ HARD LIMITATION, not a tuning problem: a bright subject on a dark background is, to any
    measurement, indistinguishable from a vignette — and that describes most creator footage,
    including Julian's. The threshold is deliberately high (0.30) so only an unmistakable
    vignette fires. Painting fake dark corners onto an edit is much worse than missing a subtle
    real one, so this errs toward missing it.
    """
    frames = _sample(video_path)
    if len(frames) < 4:
        return 0.0
    ratios = []
    for f in frames:
        if letterbox:
            fh, fw = f.shape[:2]
            t, b = int(letterbox["top"] * fh), int(letterbox["bottom"] * fh)
            l, r_ = int(letterbox["left"] * fw), int(letterbox["right"] * fw)
            f = f[t:fh - b or fh, l:fw - r_ or fw]
            if f.size == 0:
                continue
        g = cv2.cvtColor(cv2.resize(f, (160, 284)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
        centre = g[r < 0.35].mean()
        corner = g[r > 0.95].mean()
        if centre > 8:
            ratios.append(float(corner / centre))
    if not ratios:
        return 0.0
    strength = 1.0 - float(np.median(ratios))
    return round(strength, 4) if strength > min_strength else 0.0


def detect_rgb_split(video_path, shots, min_px=1.2, min_resp=0.40):
    """Red/blue channel offset — the 'chromatic aberration' / RGB-split look.

    Measure each channel against GREEN, not against each other. Two reasons:

      1. Video is encoded 4:2:0 — the colour channels are stored at half resolution and
         rebuilt from shared luma. Red and blue therefore come back dominated by unshifted
         luma structure, and correlating them against each other finds almost nothing (the
         codec crushes it: a true 10px offset measured as -1.2px, response 0.04).
      2. Green IS essentially luma, so it survives the codec intact and gives a clean lock.
         Same footage, measured against green: 4.82px against a true 5px, response 0.78.

    A split usually pushes red and blue in OPPOSITE directions, so returning both offsets
    separately is also what After Effects needs to rebuild it.

    Returns {shot_index: {"r": [dx, dy], "b": [dx, dy]}} for shots where the split is real.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    win = cv2.createHanningWindow((360, 640), cv2.CV_32F)
    out = {}

    def settle(samples, spread=1.2):
        """A channel counts as measured only if the correlation peak is strong AND the
        frames AGREE. Whichever of red/blue happens to carry the contrast in a given video
        locks solidly (peak ~1.0); the other can wander and even flip sign. Returning a
        confident 0.0 for a channel we could not read would be a silent lie — so an
        unreadable channel comes back None and simply does not get applied."""
        good = [(dx, dy) for dx, dy, resp in samples if resp > min_resp]
        if len(good) < 3:
            return None
        xs = np.array([g[0] for g in good])
        if xs.std() > spread:
            return None
        return [round(float(np.median(xs)), 2),
                round(float(np.median([g[1] for g in good])), 2)]

    for s in shots:
        rs, bs = [], []
        for t in np.linspace(s["start"], max(s["start"], s["end"] - 0.05), 6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, f = cap.read()
            if not ok:
                continue
            b, g, r = cv2.split(cv2.resize(f, (360, 640)).astype(np.float32))
            (rdx, rdy), rresp = cv2.phaseCorrelate(g, r, win)
            (bdx, bdy), bresp = cv2.phaseCorrelate(g, b, win)
            rs.append((rdx, rdy, rresp))
            bs.append((bdx, bdy, bresp))

        r_off, b_off = settle(rs), settle(bs)
        mag = max(np.hypot(*r_off) if r_off else 0.0,
                  np.hypot(*b_off) if b_off else 0.0)
        if mag >= min_px:
            out[s["index"]] = {"r": r_off, "b": b_off}

    cap.release()
    return out


def probe_look(video_path, shots):
    bars = detect_letterbox(video_path)
    return {
        "letterbox": bars,
        "vignette": detect_vignette(video_path, letterbox=bars),
        "rgb_split": detect_rgb_split(video_path, shots),
    }

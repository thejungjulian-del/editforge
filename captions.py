"""Find burned-in captions and text overlays — near-universal in short-form, and until now
completely invisible to us.

We detect the caption SLOT, not the words: where text sits, how big it is, and when it comes
and goes. Those become real, editable AE text layers with placeholder copy.

That is deliberate. Reading the reference's actual words with OCR would be worse on both
counts — it copies someone else's script into Julian's video, and it is the fragile part of
the pipeline. What transfers is the caption RHYTHM and PLACEMENT, which is the part that
makes an edit feel like the reference.

Method: a morphological TOP-HAT isolates bright thin structures sitting on a darker
surround — which is exactly what a caption is, and what makes it legible over any footage.
Letters are then closed together horizontally into lines, and the resulting boxes are
filtered on text-line geometry.

Detecting lines rather than letters is deliberate: burned-in captions carry a heavy stroke
that fuses the glyphs together, so letter-level segmentation (MSER) collapses each word into
one blob and finds "2 glyphs" where there are thirteen. The line IS the thing we want anyway.

The bright-on-dark assumption is a real limitation: a dark caption on a light background will
be missed. That is the right way to be wrong — a phantom caption stamped across every shot is
far worse than a missed one.
"""
import cv2
import numpy as np

SAMPLE_FPS = 6.0
MIN_SPAN = 0.15         # seconds; a caption that flickers for one sample is a false positive


def _lines(gray, h, w):
    """Text-line boxes: bright, stroked, horizontally extended structures."""
    # top-hat: whatever is brighter than its local surroundings, at glyph scale
    k = max(3, int(h * 0.045) | 1)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    # Require BOTH "brighter than its surroundings" AND "actually near-white". The second
    # condition is what stops a caption from fusing with mid-bright background clutter during
    # the closing step below — when that happened, the merged blob came out too tall and the
    # real caption was thrown away as "not a text line".
    mask = ((tophat > 40) & (gray > 185)).astype(np.uint8) * 255

    # fuse the letters of a word/line into one run
    join = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(w * 0.05)), 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, join)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bh < 0.015 * h or bh > 0.14 * h:      # too small to read / too big to be a line
            continue
        if bw < 0.20 * w or bw < 1.4 * bh:       # a caption line is WIDE
            continue
        if area < 0.15 * bw * bh:                # text is sparse in its box, but not empty
            continue
        patch = gray[y:y + bh, x:x + bw]
        if patch.size == 0:
            continue
        # Bimodal: bright glyph over a dark stroke. Measured on real vs decoy content, a true
        # caption hits ~250 peak with a ~200 spread; the busiest background blob managed 178
        # and 130. These thresholds sit in that gap.
        hi, lo = float(np.percentile(patch, 92)), float(np.percentile(patch, 12))
        if hi < 200 or hi - lo < 150:
            continue
        out.append([int(x), int(y), int(bw), int(bh)])
    return out


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def detect_captions(video_path):
    """Caption spans: [{t0, t1, x, y, w, h, height}] with all geometry as fractions of frame."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if not total:
        cap.release()
        return []

    step = max(1, int(round(fps / SAMPLE_FPS)))
    frames = list(range(0, total, step))

    live, done = [], []          # live = spans still being extended
    W = H = 1
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        rows = _lines(gray, H, W)
        t = fi / fps

        matched = set()
        for row in rows:
            hit = None
            for span in live:
                if _iou(span["box"], row) > 0.35:
                    hit = span
                    break
            if hit:
                hit["t1"] = t
                hit["box"] = row[:4]
                matched.add(id(hit))
            else:
                s = {"t0": t, "t1": t, "box": row[:4]}
                live.append(s)
                matched.add(id(s))

        # a span that missed this sample has ended
        still, ended = [], []
        for span in live:
            (still if id(span) in matched else ended).append(span)
        live, _ = still, done.extend(ended)

    done.extend(live)
    cap.release()

    out = []
    for s in done:
        if s["t1"] - s["t0"] < MIN_SPAN:
            continue
        x, y, w, h = s["box"]
        out.append({
            "t0": round(s["t0"], 3),
            "t1": round(s["t1"] + 1.0 / SAMPLE_FPS, 3),   # it was still there at the last sample
            "x": round((x + w / 2) / W, 4),               # centre, as a fraction
            "y": round((y + h / 2) / H, 4),
            "w": round(w / W, 4),
            "h": round(h / H, 4),
        })
    return sorted(out, key=lambda s: s["t0"])

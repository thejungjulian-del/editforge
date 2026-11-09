"""Find runs of reference shots that are the same source video, spliced.

Editors constantly chop ONE clip into several quick cuts — a continuous take becomes four shots in
a row. When we rebuild with Julian's footage we want to do the same: use ONE of his clips across
those slots, not four different ones. But a rendered video keeps no record of its source files, so
"same original video" can only be INFERRED from appearance: consecutive shots that look near-
identical (same scene, background, colour, framing) almost certainly came from one source.

Conservative by Julian's choice: only CONSECUTIVE, near-identical shots are grouped. This nails the
common case (a continuous take chopped into jump cuts) with almost no false grouping. It will miss
same-source shots separated by a big visual jump — that's the deliberate trade for not over-grouping
an edit that's all one person in one room.

The signature per shot is robust to the moving subject on purpose: an AVERAGE of several downscaled
frames blurs whatever moves and leaves the static scene, plus a colour histogram for palette/lighting.
Two shots are the same source only if BOTH agree (the two-gates discipline used across this codebase).
"""
import cv2
import numpy as np

SIG_SIZE = 24        # averaged-thumbnail resolution
N_SAMPLES = 6        # frames sampled per shot
SIG_T = 0.85         # thumbnail-correlation gate (structure/scene)
HIST_T = 0.70        # colour-histogram-correlation gate (palette/lighting)


def _shot_signature(cap, start, end, fps):
    """(averaged downscaled frame, colour histogram) for one shot — or None if unreadable."""
    times = np.linspace(start, max(start + 0.01, end - 0.01), N_SAMPLES)
    acc, hist, cnt = None, None, 0
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            continue
        small = cv2.resize(f, (SIG_SIZE, SIG_SIZE)).astype(np.float32)
        acc = small if acc is None else acc + small
        hsv = cv2.cvtColor(cv2.resize(f, (64, 64)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        hist = h if hist is None else hist + h
        cnt += 1
    if cnt == 0:
        return None
    avg = (acc / cnt).flatten()
    cv2.normalize(hist, hist)
    return avg, hist


def _similarity(a, b):
    """(thumbnail correlation, histogram correlation) between two shot signatures."""
    (avg_a, hist_a), (avg_b, hist_b) = a, b
    ta, tb = avg_a - avg_a.mean(), avg_b - avg_b.mean()
    denom = (np.linalg.norm(ta) * np.linalg.norm(tb)) or 1e-6
    sig_corr = float(ta @ tb / denom)
    hist_corr = float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))
    return sig_corr, hist_corr


def detect_source_groups(video_path, shots):
    """Runs of consecutive shots that look like one spliced source.

    Returns a list of groups, each a list of shot indices (only runs of length >= 2). A shot not in
    any returned group is a distinct source of its own.
    """
    if len(shots) < 2:
        return []
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sigs = [_shot_signature(cap, s["start"], s["end"], fps) for s in shots]
    cap.release()

    groups, cur = [], [0]
    for i in range(1, len(shots)):
        same = False
        if sigs[i] is not None and sigs[i - 1] is not None:
            sc, hc = _similarity(sigs[i - 1], sigs[i])
            same = sc > SIG_T and hc > HIST_T
        if same:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return [g for g in groups if len(g) >= 2]


def annotate_shots(video_path, shots):
    """Tag each shot with source_group = the index of its group (or None if it stands alone)."""
    groups = detect_source_groups(video_path, shots)
    member = {}
    for gi, g in enumerate(groups):
        for idx in g:
            member[idx] = gi
    for sh in shots:
        sh["source_group"] = member.get(sh["index"])
    return groups

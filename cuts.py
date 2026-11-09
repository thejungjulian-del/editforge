"""Tell a real cut apart from the camera merely moving fast.

PySceneDetect's ContentDetector fires on HOW MUCH the frame changed. A whip pan changes the
frame enormously while staying in the same shot, so it gets reported as a cut — and every
false cut shortens the surrounding shots and corrupts the cut rhythm the whole tool exists to
copy. Julian hit exactly this: the opening pans of his reference came back as transitions.

The test that separates them:

    at a CUT      the new frame is a different scene. No camera move explains it.
    at a PAN      the new frame IS the old frame, moved. A global transform explains it almost
                  perfectly.

So estimate the global motion across the boundary, warp the earlier frame forward by it, and
measure how well it now matches. A pan lines up. A cut does not. This also, for free, keeps
whip-pan CUTS (huge motion AND unrelated content) while dropping whip-pan MOVES.
"""
import cv2
import numpy as np

W = 320
MIN_ALIGN = 0.55       # normalised correlation after warping: above this the move explains it

# Inlier ratio is NOT evidence of a cut — it is evidence of how easy the footage is to track,
# and it must never veto the alignment score. A dark, motion-blurred whip pan on Julian's real
# reference tracked only 14% inliers yet aligned at 0.72: unmistakably a camera move, thrown
# away by a 30% inlier gate. This floor now exists only to reject a degenerate transform fitted
# to almost nothing.
MIN_INLIERS = 0.06

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def _gray(frame):
    h, w = frame.shape[:2]
    g = cv2.cvtColor(cv2.resize(frame, (W, max(2, int(h * W / w)))), cv2.COLOR_BGR2GRAY)
    # Whip pans go dark AND blurry — luminance fell 69 -> 12 and sharpness 13 -> 1 across the
    # boundary that failed. Equalising local contrast gives the tracker something to hold on to
    # instead of finding no features at all and defaulting to "must be a cut".
    return _clahe.apply(g)


def _global_motion(a, b):
    """Best global transform taking frame a to frame b, plus how many features agreed."""
    pts = cv2.goodFeaturesToTrack(a, maxCorners=300, qualityLevel=0.01, minDistance=7)
    if pts is None or len(pts) < 12:
        return None, 0.0

    # A fast pan can move the frame further than plain LK will follow, so give it a coarse
    # head start from phase correlation and let LK refine. Without this a genuine whip pan
    # looks untrackable and gets mistaken for a cut — the very error we are fixing.
    (sx, sy), _ = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))
    guess = pts + np.array([[[sx, sy]]], np.float32)

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(
        a, b, pts, guess.astype(np.float32),
        winSize=(31, 31), maxLevel=4,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    if nxt is None:
        return None, 0.0
    ok = status.ravel() == 1
    src, dst = pts[ok], nxt[ok]
    if len(src) < 12:
        return None, 0.0

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
    if M is None or inliers is None:
        return None, 0.0
    return M, float(inliers.sum()) / len(src)


def _alignment(a, b, M):
    """How well does warping a by M reproduce b? Normalised correlation over the overlap."""
    h, w = a.shape
    warped = cv2.warpAffine(a, M, (w, h), borderValue=-1, flags=cv2.INTER_LINEAR)
    valid = warped >= 0
    if valid.sum() < 0.25 * h * w:            # the move pushed almost everything off-frame
        return 0.0
    x = warped[valid].astype(np.float32)
    y = b[valid].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    denom = float(np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    return float((x * y).sum() / denom) if denom > 1e-6 else 0.0


MIN_CONTENT = 0.45     # sharp frames either side must actually show the same stuff


def _usable(cap, f):
    """A frame bright and sharp enough to judge CONTENT from. Skips the black, smeared frames
    at a whip boundary — those are the ones that lie."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f))
    ok, fr = cap.read()
    if not ok:
        return None
    g = cv2.cvtColor(cv2.resize(fr, (160, 284)), cv2.COLOR_BGR2GRAY)
    if g.mean() < 25 or cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var() < 20:
        return None
    return fr


def _same_content(cap, f, span=20):
    """Do the nearest SHARP frames either side of the boundary show the same scene?

    ORB feature matching, because after a whip the camera has moved a long way and small-motion
    optical flow cannot follow it — but ORB matches regardless of displacement.
    """
    before = next((_usable(cap, f - d) for d in range(2, span) if _usable(cap, f - d) is not None), None)
    after = next((_usable(cap, f + d) for d in range(1, span) if _usable(cap, f + d) is not None), None)
    if before is None or after is None:
        return 0.0

    ga, gb = _gray(before), _gray(after)
    orb = cv2.ORB_create(1500)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(matches) < 10:
        return 0.0
    src = np.float32([ka[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if M is None or inl is None:
        return 0.0
    return _alignment(ga, gb, M)


def is_camera_move(video_path, t, fps=None, cap=None):
    """Is the change at time t explained by the camera moving, rather than by a cut?

    TWO tests must agree, and the second one is the important one:

      1. the frames across the boundary line up under a global motion, AND
      2. the nearest SHARP frames either side still show the SAME CONTENT.

    Test 1 alone is not enough and merged real cuts on Julian's reference. His edit whips into
    near-blackness at every boundary, and two blurred, near-black frames correlate at 0.97
    whether or not they are the same shot — the score was measuring mush. Test 2 skips the
    smeared frames entirely and asks whether the actual scene survived the boundary. On a cut
    it has not: he is wearing sunglasses on the far side.
    """
    own = cap is None
    if own:
        cap = cv2.VideoCapture(str(video_path))
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    f = int(round(t * fps))
    best = 0.0
    for lead in (1, 2, 3):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f - lead))
        ok1, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, f + lead - 1)
        ok2, b = cap.read()
        if not (ok1 and ok2):
            continue
        ga, gb = _gray(a), _gray(b)
        M, inl = _global_motion(ga, gb)
        if M is None or inl < MIN_INLIERS:
            continue
        best = max(best, _alignment(ga, gb, M))

    content = _same_content(cap, f) if best >= MIN_ALIGN else 0.0
    if own:
        cap.release()
    return (best >= MIN_ALIGN and content >= MIN_CONTENT), round(min(best, content), 3)


def filter_cuts(video_path, cuts):
    """Drop boundaries that are just the camera moving, merging the shots either side.

    Returns (cuts, dropped) — dropped is a list of (time, alignment_score) for reporting.
    """
    if len(cuts) < 2:
        return cuts, []

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    merged = [list(cuts[0])]
    dropped = []
    for start, end in cuts[1:]:
        moved, score = is_camera_move(video_path, start, fps, cap)
        if moved:
            merged[-1][1] = end                 # same shot really — extend the previous one
            dropped.append((round(start, 3), score))
        else:
            merged.append([start, end])
    cap.release()
    return [tuple(c) for c in merged], dropped

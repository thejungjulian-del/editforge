"""Measure the reference's camera motion, shot by shot.

The point of this module is to stop *inventing* motion. Everything else about a shot
(a zoom punch, a shake) used to be a preset we applied by rule; here we actually
measure what the reference camera did — pan, push, roll, handheld jitter — frame by
frame, and hand it to After Effects as keyframes.

Method: track corner features between consecutive frames with Lucas-Kanade optical
flow, then fit a similarity transform (translate + uniform scale + rotation) with
RANSAC. RANSAC matters: it locks onto the *dominant* motion, which is the background
moving under the camera. A subject moving through frame is the minority of the
features and gets rejected as outliers.

What this cannot do: separate camera push-in from a subject walking toward the lens.
Both move the pixels the same way. That ambiguity is in the pixels, not in the code.
"""
import cv2
import numpy as np

TRACK_W = 480          # work at this width; full res buys nothing and costs a lot
MIN_FEATURES = 12      # below this the fit is noise, so hold the previous transform


def _similarity(prev_gray, gray):
    """Frame-to-frame similarity transform, as (dx, dy, scale, rot_rad) in TRACK_W pixels."""
    pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=220, qualityLevel=0.01, minDistance=8, blockSize=7)
    if pts is None or len(pts) < MIN_FEATURES:
        return None

    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None)
    if nxt is None:
        return None
    ok = status.ravel() == 1
    src, dst = pts[ok], nxt[ok]
    if len(src) < MIN_FEATURES:
        return None

    m, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if m is None or inliers is None or int(inliers.sum()) < MIN_FEATURES:
        return None

    # a partial affine is [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    rot = float(np.arctan2(m[1, 0], m[0, 0]))
    return float(m[0, 2]), float(m[1, 2]), scale, rot


def track_shot(video_path, start, end, max_keys=90):
    """Cumulative camera transform across one shot, relative to its first frame.

    Returns a list of [t, dx, dy, scale, rot_deg] where t is seconds from the shot's
    start, dx/dy are fractions of frame width/height (so they survive any resolution
    change between the reference and your footage), scale is a multiplier, and rot is
    degrees. The first key is always the identity.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))

    n_frames = max(2, int(round((end - start) * fps)))
    step = max(1, n_frames // max_keys)   # long shots get thinned, not truncated

    # Cumulative transform, kept as a 3x3 so composing frame deltas is just a product.
    cum = np.eye(3)
    keys, prev_gray, scale_to_track = [], None, None

    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if scale_to_track is None:
            scale_to_track = TRACK_W / float(w)
        gray = cv2.cvtColor(
            cv2.resize(frame, (TRACK_W, max(2, int(h * scale_to_track)))), cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            step_tf = _similarity(prev_gray, gray)
            if step_tf is not None:
                dx, dy, s, r = step_tf
                c, sn = np.cos(r) * s, np.sin(r) * s
                # Frame k's transform maps frame k-1 into frame k, so the cumulative
                # transform from frame 0 is  S_k @ ... @ S_1 — the new step goes on the
                # LEFT. Composing the other way silently rotates/scales the accumulated
                # translation by the wrong side and the pan comes out wrong.
                cum = np.array([[c, -sn, dx], [sn, c, dy], [0, 0, 1]]) @ cum
        prev_gray = gray

        if i % step == 0 or i == n_frames - 1:
            th, tw = gray.shape
            keys.append([
                round(i / fps, 4),
                round(float(cum[0, 2]) / tw, 5),          # fraction of frame width
                round(float(cum[1, 2]) / th, 5),          # fraction of frame height
                round(float(np.hypot(cum[0, 0], cum[1, 0])), 5),
                round(float(np.degrees(np.arctan2(cum[1, 0], cum[0, 0]))), 3),
            ])

    cap.release()
    if len(keys) < 2:
        return []
    return keys


def overscale(keys):
    """How far the layer must be scaled up so the camera move never exposes an edge.

    Panning a layer by 20% of the frame means 20% of the frame would be empty on one
    side unless the layer is bigger than the frame to start with. Same for a zoom-out
    or a roll. This returns the multiplier that keeps the frame covered for the whole
    shot.
    """
    if not keys:
        return 1.0
    max_shift = max(max(abs(k[1]), abs(k[2])) for k in keys)
    min_scale = min(k[3] for k in keys)
    max_rot = max(abs(np.radians(k[4])) for k in keys)
    need = (1.0 + 2.0 * max_shift) / max(min_scale, 0.05)
    need *= 1.0 + abs(np.sin(max_rot))       # a rotated rectangle needs more cover
    return round(min(max(1.0, need), 3.0), 4)  # cap: past 3x the footage is mush anyway


def is_static(keys, shift_tol=0.004, scale_tol=0.01, rot_tol=0.3):
    """A locked-off shot. Don't litter the comp with keyframes that do nothing."""
    if not keys:
        return True
    return (max(max(abs(k[1]), abs(k[2])) for k in keys) < shift_tol
            and max(abs(k[3] - 1.0) for k in keys) < scale_tol
            and max(abs(k[4]) for k in keys) < rot_tol)

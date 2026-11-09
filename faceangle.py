"""Score how flattering the camera ANGLE on a face is, frame by frame.

Julian's spec: prefer frames that are front-facing or shot from slightly above (top-down), and
avoid low-angle "up-the-nose" frames that show a double chin. We can't detect a double chin
directly — that's a hard vision problem — but we CAN detect the head pose that causes one:

  - camera BELOW the face (looking up)  -> you see under the jaw / up the nose -> double chin.  BAD
  - camera at eye level (front-facing)                                                           GOOD
  - camera slightly ABOVE (top-down), face angled up to it                                       GOOD (best)
  - face turned away (large yaw, profile) -> not front-facing                                    BAD

Pose comes from the 5 landmarks YuNet already returns (both eyes, nose tip, both mouth corners)
via solvePnP against a generic 3D face. Five points make pose NOISY in absolute terms, so we use
it RELATIVELY: rank the frames of one clip against each other (same face, same lens — systematic
error cancels) and prefer that clip's better-angled moments. It targets the angle Julian named;
it is not a beauty score and makes no claim to be one.
"""
from pathlib import Path

import cv2
import numpy as np

import face as facemod

# Generic 3D face, face-centred. X right, Y up, Z toward the camera (nose is the closest point).
# Order matches YuNet's landmarks: right eye, left eye, nose tip, right mouth, left mouth.
MODEL_3D = np.array([
    [-31.0,  36.0, -35.0],   # right eye
    [ 31.0,  36.0, -35.0],   # left eye
    [  0.0,   0.0,   0.0],   # nose tip (forward-most)
    [-28.0, -28.0, -35.0],   # right mouth corner
    [ 28.0, -28.0, -35.0],   # left mouth corner
], dtype=np.float64)

# Flattering-angle policy, in degrees of pitch. Positive pitch = camera looking DOWN at the face
# (top-down, the good side); negative = camera below looking up (the double-chin side).
PITCH_GOOD_LO = 0.0      # level and above is good
PITCH_GOOD_HI = 22.0     # ...up to a comfortable top-down
PITCH_BAD = -18.0        # by here we're squarely up-the-nose
PITCH_TOO_HIGH = 45.0    # past here it's an extreme overhead, not flattering either
YAW_FRONTAL = 18.0       # within this of straight-on reads as front-facing
YAW_PROFILE = 55.0       # by here it's a profile, not front-facing


def _camera(w, h):
    f = float(w)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)


def estimate_pose(landmarks, w, h):
    """(pitch, yaw, roll) in degrees from the 5 image landmarks, or None if solvePnP fails.

    pitch > 0 == camera above the face looking down (top-down); pitch < 0 == camera below (up-nose).
    """
    pts = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts, _camera(w, h), None,
                               flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # Euler from R. sy near 0 is gimbal lock; rare for faces, guard anyway.
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy < 1e-6:
        return None
    pitch = float(np.degrees(np.arctan2(-R[2, 1], R[2, 2])))
    yaw = float(np.degrees(np.arctan2(R[2, 0], sy)))
    roll = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    # A face upright to the camera should read ~0 pitch; the model above puts it near +180/-180
    # because Z points toward the camera, so fold pitch into [-90, 90] around level.
    if pitch > 90:
        pitch -= 180
    elif pitch < -90:
        pitch += 180
    return pitch, yaw, roll


def _ramp(x, lo, hi):
    """0 below lo, 1 above hi, linear between — a soft gate."""
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


def flattering_score(pitch, yaw):
    """0..1 — how flattering this head angle is under Julian's front-facing / top-down / no-double-chin rule."""
    # pitch: reward level-to-top-down, punish up-the-nose, ease off an extreme overhead.
    if pitch >= PITCH_GOOD_LO:
        pscore = 1.0 - _ramp(pitch, PITCH_GOOD_HI, PITCH_TOO_HIGH) * 0.6
    else:
        pscore = _ramp(pitch, PITCH_BAD, PITCH_GOOD_LO)   # 0 at/under the double-chin angle
    # yaw: reward front-facing, fade toward profile.
    yscore = 1.0 - _ramp(abs(yaw), YAW_FRONTAL, YAW_PROFILE)
    return round(float(pscore * yscore), 4)


def clip_curve(clip_path, samples=48):
    """Per-sample (time, flattering_score) across the clip. Frames with no clear face score None.

    One detector pass over the clip; the caller averages this over any window it is considering.
    """
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if not total:
        cap.release()
        return []
    dur = total / fps
    times = np.linspace(0, max(0.0, dur - 0.03), samples)

    det, curve = None, []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            curve.append((round(float(t), 3), None))
            continue
        h, w = f.shape[:2]
        sw = facemod.DET_W
        sh = max(2, int(h * facemod.DET_W / w))
        if det is None:
            det = facemod._detector(sw, sh)
            if det is None:
                cap.release()
                return []
        _, faces = det.detect(cv2.resize(f, (sw, sh)))
        if faces is None or len(faces) == 0:
            curve.append((round(float(t), 3), None))
            continue
        best = max(faces, key=lambda r: r[2] * r[3])
        pose = estimate_pose(best[4:14], sw, sh)
        curve.append((round(float(t), 3), flattering_score(pose[0], pose[1]) if pose else None))

    cap.release()
    return curve


def window_score(curve, start, dur):
    """Mean flattering score over [start, start+dur], or None if the window has no scored face."""
    vals = [s for t, s in curve if s is not None and start <= t < start + dur]
    return float(np.mean(vals)) if vals else None

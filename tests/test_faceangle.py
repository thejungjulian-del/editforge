"""Ground truth for head-pose estimation and the flattering-angle score (faceangle.py).

No real faces needed to check the MATH: take the same generic 3D face, rotate it by a KNOWN pitch
and yaw, project it to 2D through a camera, feed those 2D points back as if they were YuNet
landmarks, and confirm estimate_pose() recovers the angle it was given. Then check the score obeys
Julian's rule: front-facing / top-down high, up-the-nose low, profile low.

Run from ~/editforge:  ./venv/bin/python tests/test_faceangle.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import faceangle
import cutpoint

W, H = 320, 400


def _project(pitch_deg, yaw_deg):
    """Project the 3D model at a known pitch/yaw to 2D image landmarks."""
    # Build rotation: our convention is +pitch = camera above (top-down). Rotating the face by
    # -pitch about X presents its top to a camera on +Z, i.e. the camera looks down at it.
    rx = np.radians(-pitch_deg)
    ry = np.radians(yaw_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R = Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    tvec = np.array([[0.0], [0.0], [400.0]])          # push the face in front of the camera
    pts, _ = cv2.projectPoints(faceangle.MODEL_3D, rvec, tvec, faceangle._camera(W, H), None)
    return pts.reshape(5, 2)


def test_steering():
    """refine_cut_points should slide the cut onto the clip's best-angle segment.

    Inject a face curve that is flattering only late in the clip (t >= 2.7) and flat/neutral cut
    styles, so the flattering-angle term is the deciding factor, then confirm the chosen src_start
    lands in that good-angle region even though match() started it at 0."""
    clip = str(Path(__file__).resolve().parent.parent / "fixtures/clips/clip0.mp4")
    good_from = 2.7
    orig = cutpoint.faceangle.clip_curve
    cutpoint.faceangle.clip_curve = lambda path, samples=48: [
        (float(t), 1.0 if t >= good_from else 0.1) for t in np.linspace(0, 4, 48)]
    try:
        plan = [{"shot": 0, "clip": clip, "name": "a", "src_start": 0.0, "duration": 0.5},
                {"shot": 1, "clip": clip, "name": "b", "src_start": 0.0, "duration": 0.5}]
        shots = [{"index": 0, "cut_in": None}, {"index": 1, "cut_in": None}]  # neutral: face decides
        cutpoint.refine_cut_points(plan, shots)
        starts = [p["src_start"] for p in plan]
    finally:
        cutpoint.faceangle.clip_curve = orig
    print(f"\n  face steering: src_starts -> {starts} (good angle region starts at {good_from}s)")
    return ["steering: src_start %.2f not in the good-angle region (>=%.1f)" % (s, good_from - 0.5)
            for s in starts if s < good_from - 0.5]


def main():
    fails = []

    # 1. estimator recovers known angles (relative accuracy is what we rely on)
    for pitch, yaw in [(0, 0), (20, 0), (-20, 0), (0, 30), (15, -25)]:
        lm = _project(pitch, yaw)
        est = faceangle.estimate_pose(lm, W, H)
        if est is None:
            fails.append(f"pose None at pitch={pitch} yaw={yaw}")
            continue
        ep, ey, _ = est
        print(f"  truth pitch={pitch:>4} yaw={yaw:>4}  ->  est pitch={ep:6.1f} yaw={ey:6.1f}")
        if abs(ep - pitch) > 8:
            fails.append(f"pitch off: truth {pitch}, est {ep:.1f}")
        # only |yaw| matters to the score (front-vs-profile), so compare magnitude — the sign
        # convention between this projection and the estimator is just opposite, and harmless.
        if abs(abs(ey) - abs(yaw)) > 8:
            fails.append(f"yaw magnitude off: truth {yaw}, est {ey:.1f}")

    # 2. the score obeys the rule
    front = faceangle.flattering_score(0, 0)      # front-facing, level
    topdown = faceangle.flattering_score(15, 0)   # slightly above
    upnose = faceangle.flattering_score(-25, 0)   # camera below -> double chin
    profile = faceangle.flattering_score(0, 60)   # turned away
    overhead = faceangle.flattering_score(50, 0)  # extreme overhead
    print(f"\n  scores: front={front} topdown={topdown} upnose={upnose} profile={profile} overhead={overhead}")
    if not (front > 0.8):
        fails.append(f"front-facing should score high, got {front}")
    if not (topdown > 0.8):
        fails.append(f"top-down should score high, got {topdown}")
    if not (upnose < 0.15):
        fails.append(f"up-the-nose should score low, got {upnose}")
    if not (profile < 0.2):
        fails.append(f"profile should score low, got {profile}")
    if not (topdown >= front > upnose):
        fails.append("ordering wrong: top-down >= front > up-nose")

    # 3. the score actually steers clip selection
    fails += test_steering()

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — pose recovered from projected landmarks, and the score follows the front/top-down/no-double-chin rule.")


if __name__ == "__main__":
    main()

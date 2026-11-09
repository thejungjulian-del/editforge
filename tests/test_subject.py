"""Ground truth for subject-vs-camera separation (mvmotion.subject_shot).

Synthesises H.264 clips where BOTH motions are known: the whole scene translates by C (the
"camera"), and a textured blob translates by an extra S on top (the "subject"). The encoder's
motion vectors then carry C everywhere and C+S on the blob; subject_shot subtracts the fitted
camera (~C) and should recover S — the subject's own motion — distinct from the camera.

Run from ~/editforge:  ./venv/bin/python tests/test_subject.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mvmotion

W, H, FPS, N = 320, 480, 30, 24


def _texture(h, w, seed, cell=10):
    """Blocky high-contrast texture the encoder can actually track (per-pixel noise won't do)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // cell + 2, w // cell + 2, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, cell, 0), cell, 1)[:h, :w]


def _render(path, cam, subj):
    """cam=(cx,cy) px/frame camera pan (a moving crop over a big static background — a REAL pan,
    not a wrapped one); subj=(sx,sy) px/frame is the blob's motion relative to the scene (or None).

    A background point then moves by -cam in image space, so the encoder's global motion is -cam;
    the blob is drawn at -cam+subj so its residual after removing the camera is exactly subj.
    """
    win0 = (100, 100)                          # crop window top-left at t=0 (x, y)
    bgw = W + win0[0] + abs(cam[0]) * N + 40
    bgh = H + win0[1] + abs(cam[1]) * N + 40
    bg = _texture(int(bgh), int(bgw), seed=1)
    blob = _texture(100, 100, seed=2, cell=6)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-g", "48", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for t in range(N):
        ox, oy = win0[0] + round(cam[0] * t), win0[1] + round(cam[1] * t)
        frame = bg[oy:oy + H, ox:ox + W].copy()
        if subj is not None:
            x = 90 + round((-cam[0] + subj[0]) * t)   # image motion = camera(-cam) + subject
            y = 60 + round((-cam[1] + subj[1]) * t)
            if 0 <= y < H - 100 and 0 <= x < W - 100:
                frame[y:y + 100, x:x + 100] = blob
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()


def _analyze(cam, subj):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name
    _render(path, cam, subj)
    cache = mvmotion._frame_transforms(path)
    track = mvmotion.track_shot(path, 0, N / FPS, cache=cache)
    result = mvmotion.subject_shot(cache, 0, N / FPS)
    Path(path).unlink(missing_ok=True)
    # camera pan magnitude actually recovered, in px (dx is a fraction of width)
    cam_px = max((abs(k[1]) * W for k in track), default=0.0)
    return result, cam_px


def _ang(v):
    return np.degrees(np.arctan2(v[1], v[0]))


def main():
    fails = []

    # 1. Camera pan, NO subject — the false-positive control. Must NOT invent a subject.
    subj, cam_px = _analyze((3, 0), None)
    print(f"1 pan / no subject      -> camera {cam_px:5.1f}px  subject={subj}")
    if subj is not None:
        fails.append("1: invented a subject on a camera-only pan")

    # 2. Static camera, blob moving RIGHT (+x). Camera must stay ~still; subject points at 0deg.
    subj, cam_px = _analyze((0, 0), (6, 0))
    print(f"2 static / blob right   -> camera {cam_px:5.1f}px  subject={subj}")
    if subj is None:
        fails.append("2: missed a moving subject on a static camera")
    else:
        if cam_px > 3.0:
            fails.append(f"2: camera fooled by the subject ({cam_px:.1f}px)")
        if abs(subj["direction_deg"]) > 25:
            fails.append(f"2: subject direction {subj['direction_deg']} != ~0deg (right)")

    # 3. Camera pans RIGHT, blob additionally moves DOWN (+y). The hard case: recover BOTH,
    #    and the subject's direction (~90deg, down) must be distinct from the camera's (right).
    subj, cam_px = _analyze((3, 0), (0, 6))
    print(f"3 pan right / blob down -> camera {cam_px:5.1f}px  subject={subj}")
    if subj is None:
        fails.append("3: missed the subject while the camera panned")
    else:
        if cam_px < 8.0:
            fails.append(f"3: camera pan not recovered ({cam_px:.1f}px)")
        d = subj["direction_deg"]
        if not (60 <= d <= 120):
            fails.append(f"3: subject direction {d} != ~90deg (down)")

    # 4. Fully static. Nothing moves; subject must be None and camera ~0.
    subj, cam_px = _analyze((0, 0), None)
    print(f"4 fully static          -> camera {cam_px:5.1f}px  subject={subj}")
    if subj is not None:
        fails.append("4: invented a subject on a static frame")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — subject motion is separated from camera motion on all four cases.")


if __name__ == "__main__":
    main()

"""Learned shot-boundary detection (TransNetV2), to replace PySceneDetect as the candidate source.

PySceneDetect thresholds a frame-difference score, which is why it both MISSED real cuts in
Julian's reference and fired on camera pans: a whip pan changes the frame more than some cuts do.
A learned detector looks at a window of frames and asks "is this a boundary", which is a
different — and much better-posed — question.

Verified numbers (adversarial research pass, 3-0): TransNetV2 scores F1 0.799 on SHOT, the only
public benchmark built from short-video-platform content; AutoShot scores 0.841 there. Both are
far above what a threshold heuristic manages on this material.

But it is NOT a silver bullet, and the TransNetV2 authors say so themselves: fast abrupt changes
from camera motion still cause "many false hits", and 0.841 on short-form vs 0.97 on broadcast
says short-form shot detection is nowhere near solved. So this produces CANDIDATES, and the
encoder's keyframes still adjudicate them (see bitstream.py). Two independent sources of truth,
neither trusted alone — the same discipline that fixed dissolve and velocity detection.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).parent / "models"
WEIGHTS = MODELS / "transnetv2-pytorch-weights.pth"

_model = None


def available():
    return WEIGHTS.exists()


def _load():
    global _model
    if _model is not None:
        return _model
    import torch
    sys.path.insert(0, str(MODELS))
    from transnetv2_pytorch import TransNetV2

    m = TransNetV2()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    m.eval()
    _model = m
    return m


def _frames(video_path):
    """TransNetV2 wants 48x27 RGB. It really is that small — it reads motion, not detail."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, (48, 27)), cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(out, dtype=np.uint8), fps


def predictions(video_path):
    """Per-frame probability that a shot boundary falls on that frame."""
    import torch

    frames, fps = _frames(video_path)
    if len(frames) < 4:
        return np.array([]), fps
    model = _load()

    # The model takes 100-frame windows and only its middle 50 frames are trusted, so the clip is
    # padded by 25 at each end and stepped by 50. Getting this padding wrong silently shifts every
    # predicted boundary.
    pad = np.concatenate([
        np.repeat(frames[:1], 25, axis=0),
        frames,
        np.repeat(frames[-1:], 25 + 50, axis=0),
    ])

    preds = []
    with torch.no_grad():
        for i in range(0, len(frames), 50):
            window = pad[i:i + 100]
            if len(window) < 100:
                window = np.concatenate(
                    [window, np.repeat(window[-1:], 100 - len(window), axis=0)])
            x = torch.from_numpy(window).unsqueeze(0)          # 1 x 100 x 27 x 48 x 3
            single, _ = model(x)
            p = torch.sigmoid(single).numpy().reshape(-1)[25:75]
            preds.append(p)

    return np.concatenate(preds)[:len(frames)], fps


def cuts(video_path, threshold=0.5, min_len=0.1):
    """Shot boundaries as (start, end) seconds."""
    p, fps = predictions(video_path)
    if len(p) == 0:
        return []

    # a boundary is a run of high-probability frames; take the peak of each run
    hits, run = [], []
    for i, v in enumerate(p):
        if v >= threshold:
            run.append(i)
        elif run:
            hits.append(run[int(np.argmax(p[run]))])
            run = []
    if run:
        hits.append(run[int(np.argmax(p[run]))])

    duration = len(p) / fps
    times = [0.0] + [h / fps for h in hits]
    out = []
    for i, s in enumerate(times):
        e = times[i + 1] if i + 1 < len(times) else duration
        if e - s >= min_len:
            out.append((round(s, 3), round(e, 3)))
    return out or [(0.0, duration)]

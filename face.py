"""Find the subject's face in a clip, so the framing can be anchored to it.

Why an anchor and not a crop: After Effects scales and rotates a layer AROUND its anchor
point. Put the anchor on the face and every zoom and roll we copy from the reference pivots
on the face instead of sliding it out of frame. The face stays put for free — no tracking
expression, no per-frame correction.

We do face DETECTION, not face recognition. If a clip has two people in it we take the
biggest, most consistently-present face, which is the subject in practically all creator
footage. Telling Julian's face from someone else's would need an identity model and a
reference photo, and it is not worth it until a clip actually breaks this assumption.

No face (b-roll, hands, scenery) is a normal outcome, not a failure: we fall back to the
centre of frame, which is what the tool did before any of this existed.
"""
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
DET_W = 320
MIN_CONF = 0.75


def _detector(w, h):
    if not MODEL.exists():
        return None
    return cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), MIN_CONF, 0.3, 5000)


def find_face(clip_path, start=0.0, duration=None, samples=6):
    """Where the subject's face sits in this clip, as fractions of frame width/height.

    Returns {"x", "y", "size"} (size = face height as a fraction of frame height), or None.
    Sampling several frames and taking the median keeps one bad detection from throwing the
    anchor across the frame.
    """
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if not total:
        cap.release()
        return None

    end = start + duration if duration else total / fps
    times = np.linspace(start, max(start, end - 0.05), samples)

    det, hits = None, []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if not ok:
            continue
        h, w = f.shape[:2]
        sw = DET_W
        sh = max(2, int(h * DET_W / w))
        if det is None:
            det = _detector(sw, sh)
            if det is None:
                cap.release()
                return None
        small = cv2.resize(f, (sw, sh))
        _, faces = det.detect(small)
        if faces is None or len(faces) == 0:
            continue
        # biggest face in the frame = the subject
        best = max(faces, key=lambda r: r[2] * r[3])
        fx, fy, fw_, fh_ = best[:4]
        hits.append(((fx + fw_ / 2) / sw, (fy + fh_ / 2) / sh, fh_ / sh))

    cap.release()
    if len(hits) < max(2, samples // 3):        # a single hit is noise, not a subject
        return None

    return {
        "x": round(float(np.median([h[0] for h in hits])), 4),
        "y": round(float(np.median([h[1] for h in hits])), 4),
        "size": round(float(np.median([h[2] for h in hits])), 4),
        "found": len(hits),
        "of": samples,
    }

"""Read the cuts out of the compressed video itself, instead of guessing from pixels.

H.264 does not store most frames as pictures. It stores them as PREDICTIONS from the previous
frame — motion vectors plus a small residual. So the encoder has already had to answer, for
every single frame, the exact question we keep failing to answer:

    "can I explain this frame as the previous one, MOVED?"

At a camera pan the answer is yes: same content, displaced, cheap to describe with motion
vectors. At a cut the answer is no: nothing predicts, so the encoder gives up and spends a
keyframe (an I-frame). x264 calls this scenecut detection, and it has been running inside every
video we have ever opened.

This is strictly better than the pixel approach for one reason: it is immune to the things that
were defeating us. Julian's reference whips into near-blackness at every boundary, so the frames
either side are dark, smeared mush — and mush correlates with mush. The encoder does not care;
it only cares whether the bits predict.

Measured on his real reference: the encoder put a keyframe on 17 of the 18 cuts we had found,
AND on 3 more cuts that PySceneDetect had missed entirely.

Caveat that decides whether any of this is usable: some encoders emit I-frames on a fixed timer
(a constant GOP) rather than at scene changes. Those tell you nothing. So check the spacing
first — irregular means content-driven, regular means a metronome.
"""
import subprocess

import numpy as np


def keyframes(video_path):
    """Timestamps of every I-frame in the stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pict_type,best_effort_timestamp_time",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    times = []
    for line in r.stdout.splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2 or parts[1] != "I":
            continue
        try:
            times.append(float(parts[0]))
        except ValueError:
            continue
    return sorted(times)


def is_scenecut_encoded(times, min_std=0.25, min_count=4):
    """Did the encoder place keyframes at CONTENT changes, or just on a timer?

    A fixed GOP gives evenly spaced keyframes and carries no information about the edit.
    Irregular spacing means the encoder was reacting to the content — which is the signal.
    """
    if len(times) < min_count:
        return False
    gaps = np.diff(times)
    return len(gaps) > 0 and float(gaps.std()) > min_std


def cuts(video_path, scene_cuts, duration, tol=0.15):
    """Merge what the encoder knows with what the pixel detector found.

    The encoder is treated as the authority, because it is:
      - a scene cut it flagged that the pixel detector missed  -> ADD it
      - a "cut" the pixel detector found with no keyframe near  -> it is a camera move, DROP it

    That second rule is the whole pan-vs-cut problem, solved by asking the encoder rather than
    trying to out-argue motion blur.

    Returns (cuts, added, dropped) as lists of (start, end) and times.
    """
    kf = keyframes(video_path)
    if not is_scenecut_encoded(kf):
        return scene_cuts, [], []          # fixed GOP: no information, keep the pixel result

    starts = [s for s, _ in scene_cuts]
    kf = [t for t in kf if 0.25 < t < duration - 0.25]      # ignore the opening keyframe

    kept, dropped = [], []
    for s in starts:
        if s <= 0.01 or any(abs(s - t) <= tol for t in kf):
            kept.append(s)
        else:
            dropped.append(round(s, 3))    # the encoder could predict it: the camera just moved

    added = [round(t, 3) for t in kf if not any(abs(t - s) <= tol for s in starts)]
    boundaries = sorted(set([0.0] + kept + added))

    out = []
    for i, s in enumerate(boundaries):
        e = boundaries[i + 1] if i + 1 < len(boundaries) else duration
        if e - s > 0.08:                   # drop absurdly short slivers
            out.append((s, e))
    return out, added, dropped

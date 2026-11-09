"""Extract the 'edit DNA' from a reference video: cut rhythm + beat grid + per-shot energy."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import librosa
import numpy as np
from scenedetect import ContentDetector, SceneManager, open_video

import bitstream
import cutpoint
import mvmotion
import sourcegroup
import captions as caps
import motion
import transitions as trans
import velocity
import warp as warpmod


def detect_cuts(video_path, threshold=27.0, min_scene_frames=4):
    video = open_video(str(video_path))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_frames))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    if not scenes:
        return [(0.0, video.duration.get_seconds())]
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def detect_beats(video_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "22050", wav],
        check=True, capture_output=True,
    )
    y, sr = librosa.load(wav, sr=22050)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beats = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    Path(wav).unlink(missing_ok=True)
    if len(beats) < 2:
        return 0.0, beats

    # librosa reliably misses beats before its first onset. Extrapolate backwards
    # to t=0 so cuts near the start of the edit still have a grid to snap to.
    interval = float(np.median(np.diff(beats)))
    while beats[0] - interval >= -0.05:
        beats.insert(0, beats[0] - interval)
    beats = [b for b in beats if b >= 0]

    # librosa quantizes beat times to its ~23ms analysis grid, so any single gap
    # (and librosa's own tempo scalar) is off by up to a few percent. Least-squares
    # fitting the whole beat sequence averages the quantization out.
    idx = np.arange(len(beats))
    slope = np.polyfit(idx, beats, 1)[0]
    bpm = 60.0 / slope if slope > 0 else 0.0
    return bpm, beats


def shot_energy(video_path, start, end, samples=5):
    """Mean frame-to-frame pixel change across the shot, 0-1ish."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    times = np.linspace(start, max(start + 0.01, end - 0.01), samples)
    frames = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54))
            frames.append(small.astype(np.float32))
    cap.release()
    if len(frames) < 2:
        return 0.0
    diffs = [np.abs(frames[i + 1] - frames[i]).mean() for i in range(len(frames) - 1)]
    return round(float(np.mean(diffs)) / 255.0, 4)


def snap_to_grid(cuts, beats, bpm):
    """How does the cut rhythm relate to the beat grid? Returns the dominant subdivision."""
    if not beats or bpm <= 0:
        return None
    beat_len = 60.0 / bpm
    durations = [e - s for s, e in cuts]
    subdivisions = [1 / 4, 1 / 2, 1, 2, 4]
    best, best_err = None, 1e9
    for sub in subdivisions:
        target = beat_len * sub
        err = np.mean([min(abs(d - target), target) for d in durations])
        if err < best_err:
            best, best_err = sub, err
    return {"beat_length_s": round(beat_len, 4), "dominant_subdivision": best}


def analyze(video_path):
    video_path = Path(video_path)
    cuts = detect_cuts(video_path)
    duration = cuts[-1][1] if cuts else 0.0

    # Camera motion from the encoder's OWN motion vectors, not recomputed from pixels.
    # On Julian's real reference the optical-flow tracker returned nonsense — zoom factors of
    # 0.23 and 0.58, rolls of -16 degrees — because it loses its grip in the dark, motion-blurred
    # frames and the error accumulates. We were feeding that garbage straight into AE as
    # keyframes. The encoder tracked the same footage fine (it had to, to compress it), and its
    # vectors give sane values. Optical flow stays as the fallback for anything without usable
    # vectors.
    try:
        mv_cache = mvmotion._frame_transforms(video_path)
        sp_cache = mvmotion.speed_series(video_path)
    except Exception:
        mv_cache = sp_cache = None

    # Ask the ENCODER where the cuts are. It already had to decide, for every frame, whether the
    # frame could be predicted from the last one — which is exactly the pan-vs-cut question. A
    # pan predicts cheaply (motion vectors); a cut does not, so it spends a keyframe.
    #
    # This replaced a pixel-based pan filter that was actively harmful: it merged real cuts away
    # because this reference whips into near-blackness at every boundary, and two dark, smeared
    # frames correlate at 0.97 whether or not they are the same shot. The encoder is not fooled
    # by blur, and on Julian's reference it also caught 3 cuts PySceneDetect missed outright.
    cuts, added_cuts, dropped_cuts = bitstream.cuts(video_path, cuts, duration)

    bpm, beats = detect_beats(video_path)
    shots = []
    for i, (s, e) in enumerate(cuts):
        track = []
        if mv_cache:
            track = mvmotion.track_shot(video_path, s, e, cache=mv_cache)
        if not track:                       # no vectors (or not H.264): fall back to pixels
            track = motion.track_shot(video_path, s, e)
        static = motion.is_static(track)
        shots.append({
            "index": i,
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "energy": shot_energy(video_path, s, e),
            # the reference camera's actual move across this shot — not a preset
            "motion": [] if static else track,
            "overscale": 1.0 if static else motion.overscale(track),
            # the shot's speed ramp, normalised — [] if it plays at a steady rate
            "speed": velocity.profile(video_path, s, e, cache=sp_cache),
            # something moving in-frame INDEPENDENTLY of the camera (None = camera-only shot).
            # This is the camera's RANSAC leftovers: blocks the camera move can't explain.
            "subject": mvmotion.subject_shot(mv_cache, s, e) if mv_cache else None,
        })
    for sh, tr in zip(shots, trans.analyze_transitions(video_path, shots)):
        sh["transition"] = tr["type"]
        sh["transition_duration"] = tr["duration"]
        sh["freezes"] = tr["freezes"]

    warps = warpmod.detect_warp(video_path, shots)
    for sh in shots:
        sh["warp"] = warps.get(sh["index"])

    # How each cut sits on the motion around it — cut on action, motion-matched, on stillness.
    # Stored as shot['cut_in'] (the cut leading into each shot) so the builder can reproduce it.
    cut_styles = cutpoint.annotate_shots(mv_cache, shots) if mv_cache else []

    # Runs of consecutive shots that are the same source video, spliced (one take chopped into
    # several cuts). Stored as shot['source_group'] so the builder can splice ONE of the user's
    # clips across those slots instead of spreading different clips.
    source_groups = sourcegroup.annotate_shots(video_path, shots)

    text = caps.detect_captions(video_path)
    durations = [sh["duration"] for sh in shots]
    return {
        "source": video_path.name,
        "total_duration": round(cuts[-1][1], 3),
        "bpm": round(bpm, 1),
        "beat_count": len(beats),
        "beats": [round(b, 3) for b in beats],
        "shot_count": len(shots),
        "cuts_added_by_encoder": len(added_cuts),
        "camera_moves_not_cuts": len(dropped_cuts),
        "moving_shots": sum(1 for sh in shots if sh["motion"]),
        "subject_shots": sum(1 for sh in shots if sh["subject"]),
        "ramped_shots": sum(1 for sh in shots if sh["speed"]),
        "warped_shots": sum(1 for sh in shots if sh["warp"]),
        "captions": text,
        "transitions": {
            k: sum(1 for sh in shots if sh["transition"] == k)
            for k in ("hard", "dissolve", "flash", "fade", "whip")
        },
        "freeze_shots": sum(1 for sh in shots if sh["freezes"]),
        "cut_styles": {k: cut_styles.count(k) for k in ("action", "match", "still", "whip", "hard")},
        "source_groups": source_groups,
        "spliced_shots": sum(len(g) for g in source_groups),
        "median_shot_duration": round(float(np.median(durations)), 3),
        "grid": snap_to_grid(cuts, beats, bpm),
        "shots": shots,
    }


if __name__ == "__main__":
    result = analyze(sys.argv[1])
    summary = {k: v for k, v in result.items() if k not in ("beats", "shots")}
    print(json.dumps(summary, indent=2))
    print(f"\nfirst 12 shots (duration s / energy):")
    for sh in result["shots"][:12]:
        bar = "#" * int(sh["energy"] * 120)
        print(f"  {sh['index']:>3}  {sh['duration']:>5.2f}s  {sh['energy']:.3f}  {bar}")
    Path("output/reference.json").write_text(json.dumps(result, indent=2))
    print("\n-> output/reference.json")

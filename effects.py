"""Effects for EditForge.

Two of these are DETECTED from the reference (real measurements):
  - color grade: LAB colour statistics -> a 3D LUT that maps your footage onto the
    reference's look (Reinhard colour transfer)
  - flash cuts: luminance spikes at the reference's cut points

Two are PRESETS applied by rule, because they can't be reliably recovered from a
finished video (you cannot tell 0.5x slow-mo from footage shot at 60fps):
  - zoom punch on the beat
  - slow-mo on the low-energy shots
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

LUT_GRID = 17


def _sample_frames(video_path, n=24):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames = []
    for i in np.linspace(0, max(0, total - 1), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def lab_stats(video_path, n=24):
    """Mean and std of the video's colour in LAB space — its 'look' as numbers.

    Frames are downscaled first: colour statistics don't need full resolution, and
    summing tens of millions of float32 pixels silently loses precision (float32 only
    represents integers exactly up to 2^24, and the running sum blows past that).
    float64 accumulation keeps it honest.
    """
    frames = _sample_frames(video_path, n)
    if not frames:
        return None
    labs = [
        cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        for f in frames
    ]
    stacked = np.concatenate(labs, axis=0).astype(np.float64)
    return {
        "mean": stacked.mean(axis=0).tolist(),
        "std": (stacked.std(axis=0) + 1e-6).tolist(),
    }


def make_lut(src_stats, ref_stats, out_path, grid=LUT_GRID, strength=1.0):
    """Build a .cube 3D LUT that shifts src colours onto the reference's distribution.

    Reinhard transfer: for each LAB channel, renormalise the source to zero-mean /
    unit-std, then re-scale onto the reference's mean and std.
    """
    src_mean = np.array(src_stats["mean"], np.float32)
    src_std = np.array(src_stats["std"], np.float32)
    ref_mean = np.array(ref_stats["mean"], np.float32)
    ref_std = np.array(ref_stats["std"], np.float32)

    # every RGB grid point, as a 1-pixel-tall image so cv2 can convert it
    axis = np.linspace(0, 255, grid, dtype=np.float32)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    # .cube iterates red fastest, then green, then blue
    rgb = np.stack([r, g, b], axis=-1).reshape(-1, 1, 3)
    bgr = rgb[..., ::-1].astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    shifted = (lab - src_mean) / src_std * ref_std + ref_mean
    lab_out = lab + (shifted - lab) * strength
    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)

    out_bgr = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR).astype(np.float32) / 255.0
    out_rgb = out_bgr[..., ::-1].reshape(-1, 3)

    lines = [f"LUT_3D_SIZE {grid}", ""]
    lines += [f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in out_rgb]
    Path(out_path).write_text("\n".join(lines) + "\n")
    return out_path


# A flash flashes WHITE and briefly. Both gates must hold, or a shot that merely STARTS brighter
# (auto-exposure settling, a bright opening element, a cut to a brighter-lit shot) reads as a flash
# when it isn't — that was the over-firing. Relative brightness alone (head/body >= 1.35) is not
# enough; the spike has to actually be near-white AND a big jump from the shot's own body.
FLASH_MIN_LUM = 200    # near-white, on 0-255 mean luminance
FLASH_MIN_JUMP = 55    # the spike must be this far above the shot's normal level


def detect_flashes(video_path, shots, spike=1.35):
    """A flash transition shows up as a brief near-WHITE luminance spike right at the cut.

    Compares the brightness of the first few frames of each shot against that shot's own median
    brightness, and requires the spike to be genuinely white. Returns the shot indices that start
    on a flash.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    flashes = set()

    for shot in shots:
        lums = []
        start_f = int(shot["start"] * fps)
        end_f = int(shot["end"] * fps)
        if end_f - start_f < 3:
            continue
        for f in range(start_f, min(end_f, start_f + 20)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            lums.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        if len(lums) < 4:
            continue
        head = max(lums[:2])
        body = float(np.median(lums[2:]))
        if (head >= FLASH_MIN_LUM
                and (head - body) >= FLASH_MIN_JUMP
                and head / max(body, 1.0) >= spike):
            flashes.add(shot["index"])

    cap.release()
    return flashes


def probe_reference(reference_video, shots):
    """Everything we can actually measure about the reference's look."""
    return {
        "color": lab_stats(reference_video),
        "flash_shots": sorted(detect_flashes(reference_video, shots)),
    }

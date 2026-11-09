"""Ground truth for look-alike matching (contentmatch.match_clips_to_shots).

Render a reference with three visually-distinct shots (red / green / blue scenes), plus two of
"his" clips: one that RESEMBLES the red shot, one (yellow) that resembles none. The matcher must
lock the red-looking clip into the red shot's slot and lock the yellow clip nowhere.

Run from ~/editforge:  ./venv/bin/python tests/test_contentmatch.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import contentmatch

W, H, FPS, SHOT = 200, 280, 30, 15
TINTS = {"red": (0.35, 0.35, 1.0), "green": (0.35, 1.0, 0.35), "blue": (1.0, 0.35, 0.35),
         "yellow": (0.35, 1.0, 1.0)}
SEEDS = {"red": 1, "green": 2, "blue": 3, "yellow": 4}


def _texture(seed, tint):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, (H // 8 + 1, W // 8 + 1, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(base, 8, 0), 8, 1)[:H, :W].astype(np.float32)
    return np.clip(big * np.array(tint), 0, 255).astype(np.uint8)


def _render(path, scene_names, frames=SHOT):
    scenes = [_texture(SEEDS[n], TINTS[n]) for n in scene_names]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-g", "300", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    f = 0
    for scene in scenes:
        for _ in range(frames):
            frame = scene.copy()
            x = 20 + (f * 5) % (W - 50)          # a moving blob: same scene, not identical frames
            frame[H // 2 - 15:H // 2 + 15, x:x + 30] = 255
            proc.stdin.write(frame.tobytes())
            f += 1
    proc.stdin.close()
    proc.wait()


def _tmp(suffix=".mp4"):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        return f.name


def main():
    ref = _tmp()
    _render(ref, ["red", "green", "blue"])            # reference: 3 distinct shots
    shots = [{"index": i, "start": round(i * SHOT / FPS, 3),
              "end": round((i + 1) * SHOT / FPS, 3)} for i in range(3)]

    my_red = _tmp()
    _render(my_red, ["red"], frames=40)               # my clip that LOOKS like the red shot
    my_yellow = _tmp()
    _render(my_yellow, ["yellow"], frames=40)         # my clip that resembles nothing

    locks = contentmatch.match_clips_to_shots(ref, shots, [my_red, my_yellow])
    for p in (ref, my_red, my_yellow):
        Path(p).unlink(missing_ok=True)

    print("locks (shot -> clip):", {k: Path(v).name for k, v in locks.items()})
    fails = []
    if locks.get(0) != my_red:
        fails.append(f"the red-looking clip should lock into shot 0, got {locks.get(0)}")
    if my_yellow in locks.values():
        fails.append("the yellow clip resembles nothing and must not lock anywhere")
    if 1 in locks or 2 in locks:
        fails.append(f"green/blue shots have no look-alike and should stay unlocked, got {locks}")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — a look-alike clip locks into the shot it resembles; dissimilar clips don't lock.")


if __name__ == "__main__":
    main()

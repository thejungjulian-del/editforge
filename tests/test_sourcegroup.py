"""Ground truth for same-source detection (sourcegroup.py).

Render a reference whose FIRST FOUR shots are one source (a distinctive red scene with a moving
blob — the blob is at a different place in each shot, so the shots are NOT identical frames, which
is the realistic 'spliced continuous take' case), followed by two shots from clearly different
sources (a green scene, a blue scene). The detector must group shots 0-3 and leave 4 and 5 alone.

Run from ~/editforge:  ./venv/bin/python tests/test_sourcegroup.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sourcegroup

W, H, FPS, SHOT = 240, 320, 30, 15   # 6 shots x 15 frames = 90 frames


def _texture(seed, tint):
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 210, (H // 8 + 1, W // 8 + 1, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(base, 8, 0), 8, 1)[:H, :W].astype(np.float32)
    return np.clip(big * np.array(tint), 0, 255).astype(np.uint8)


def _scene(name):
    tints = {"red": (0.35, 0.35, 1.0), "green": (0.35, 1.0, 0.35), "blue": (1.0, 0.35, 0.35),
             "yellow": (0.35, 1.0, 1.0), "cyan": (1.0, 1.0, 0.35), "magenta": (1.0, 0.35, 1.0)}
    seeds = {"red": 1, "green": 2, "blue": 3, "yellow": 4, "cyan": 5, "magenta": 6}
    return _texture(seeds[name], tints[name])


def _render(path, scene_names):
    scenes = [_scene(n) for n in scene_names]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-g", "300", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    f = 0
    for si, scene in enumerate(scenes):
        for k in range(SHOT):
            frame = scene.copy()
            # a moving white blob, so the four RED shots are the same scene at different moments
            x = 20 + (f * 4) % (W - 50)
            frame[H // 2 - 20:H // 2 + 20, x:x + 30] = 255
            proc.stdin.write(frame.tobytes())
            f += 1
    proc.stdin.close()
    proc.wait()


def _groups_for(scene_names):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    _render(path, scene_names)
    shots = [{"index": i, "start": round(i * SHOT / FPS, 3),
              "end": round((i + 1) * SHOT / FPS, 3)} for i in range(len(scene_names))]
    groups = sourcegroup.detect_source_groups(path, shots)
    Path(path).unlink(missing_ok=True)
    return groups


def main():
    fails = []

    # spliced: 4 red shots (same source) + green + blue -> one group [0,1,2,3]
    g = _groups_for(["red", "red", "red", "red", "green", "blue"])
    print("spliced       ->", g)
    if g != [[0, 1, 2, 3]]:
        fails.append(f"spliced: expected [[0,1,2,3]], got {g}")

    # all distinct sources -> NO groups (the over-grouping control)
    g = _groups_for(["red", "green", "blue", "yellow", "cyan", "magenta"])
    print("all distinct  ->", g)
    if g != []:
        fails.append(f"all-distinct: expected no groups, got {g}")

    # two separate spliced runs -> two groups
    g = _groups_for(["red", "red", "green", "blue", "blue", "blue"])
    print("two runs      ->", g)
    if g != [[0, 1], [3, 4, 5]]:
        fails.append(f"two-runs: expected [[0,1],[3,4,5]], got {g}")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — the four spliced shots grouped as one source; the two distinct shots stayed apart.")


if __name__ == "__main__":
    main()

"""Ground truth for flash detection (effects.detect_flashes) — the over-firing fix.

Render shots with KNOWN brightness and confirm only a genuine near-white flash is flagged:
  shot 0  normal throughout                        -> no flash
  shot 1  two near-WHITE frames then normal        -> FLASH (real)
  shot 2  merely STARTS brighter (grey), not white -> no flash  (the old false positive)
  shot 3  uniformly bright the whole shot          -> no flash  (a bright shot, not a flash)

Run from ~/editforge:  ./venv/bin/python tests/test_flash.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import effects

W, H, FPS, SHOT = 160, 240, 30, 15


def _render(path, profiles):
    """profiles[i] = a list of per-frame target brightnesses for shot i."""
    rng = np.random.default_rng(0)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
         "-g", "300", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for prof in profiles:
        for target in prof:
            noise = rng.integers(-8, 8, (H, W, 3))
            frame = np.clip(noise + target, 0, 255).astype(np.uint8)
            proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def main():
    normal = [90] * SHOT
    flash = [245, 245] + [90] * (SHOT - 2)         # real white flash at the cut
    brighter = [140, 140] + [90] * (SHOT - 2)      # starts brighter but NOT white
    bright = [205] * SHOT                           # a bright shot, no flash
    profiles = [normal, flash, brighter, bright]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        path = tf.name
    _render(path, profiles)
    shots = [{"index": i, "start": round(i * SHOT / FPS, 3),
              "end": round((i + 1) * SHOT / FPS, 3)} for i in range(4)]
    found = effects.detect_flashes(path, shots)
    Path(path).unlink(missing_ok=True)

    print("flashes detected:", sorted(found), " (expected just {1} — the real white flash)")
    fails = []
    if 1 not in found:
        fails.append("missed the real white flash (shot 1)")
    for bad in (0, 2, 3):
        if bad in found:
            reason = {0: "normal shot", 2: "shot that merely starts brighter (not white)",
                      3: "uniformly bright shot"}[bad]
            fails.append(f"false positive on shot {bad}: {reason}")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — only a genuine near-white flash fires; brighter starts and bright shots don't.")


if __name__ == "__main__":
    main()

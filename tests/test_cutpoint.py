"""Ground truth for cut-point analysis (cutpoint.py).

Part 1 — reference_cut_styles: render one continuous clip whose camera motion changes on a known
schedule, hand it boundaries at the segment edges, and check each boundary is classified by the
motion around it: two segments moving the same way -> match; both moving, different ways ->
action; both calm -> still; one moving one calm -> hard; and a whip-labelled cut -> whip.

Part 2 — refine_cut_points: give the builder clips that each contain a fast half and a calm half,
and check the DP slides the cut points to land on fast frames for an 'action' boundary and calm
frames for a 'still' boundary.

Run from ~/editforge:  ./venv/bin/python tests/test_cutpoint.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cutpoint
import mvmotion

W, H, FPS = 320, 480, 30


def _texture(h, w, seed, cell=10):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // cell + 2, w // cell + 2, 3), dtype=np.uint8)
    return np.repeat(np.repeat(small, cell, 0), cell, 1)[:h, :w]


def _render(path, segments):
    """segments = [(cam_vec, nframes), ...]; camera pans by cam_vec px/frame within each segment
    (a moving crop over a big static background, so it's a real pan with no wrap seam)."""
    total = sum(n for _, n in segments)
    peak = sum(abs(np.array(c)) * n for c, n in segments)
    bg = _texture(int(H + 200 + peak[1]), int(W + 200 + peak[0]), seed=1)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
         "-g", "300", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ox, oy = 100.0, 100.0
    for cam, n in segments:
        for _ in range(n):
            x, y = int(round(ox)), int(round(oy))
            proc.stdin.write(bg[y:y + H, x:x + W].copy().astype(np.uint8).tobytes())
            ox += cam[0]
            oy += cam[1]
    proc.stdin.close()
    proc.wait()
    return total


def _tmp():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        return f.name


def test_reference():
    fails = []
    seg = 24
    # right, right, down, still, still -> boundaries: match, action, hard, still
    segs = [((4, 0), seg), ((4, 0), seg), ((0, 4), seg), ((0, 0), seg), ((0, 0), seg)]
    path = _tmp()
    _render(path, segs)
    shots = [{"index": i, "start": round(i * seg / FPS, 3),
              "end": round((i + 1) * seg / FPS, 3), "transition": "hard"} for i in range(5)]
    cache = mvmotion._frame_transforms(path)
    styles = cutpoint.reference_cut_styles(cache, shots)
    print("reference styles:", styles)
    expect = ["match", "action", "hard", "still"]
    for i, (got, exp) in enumerate(zip(styles, expect)):
        if got != exp:
            fails.append(f"boundary {i}: got {got}, expected {exp}")

    # a whip-labelled cut is a whip regardless of the motion around it
    shots[2]["transition"] = "whip"
    styles = cutpoint.reference_cut_styles(cache, shots)
    if styles[1] != "whip":
        fails.append(f"whip boundary: got {styles[1]}, expected whip")
    Path(path).unlink(missing_ok=True)
    return fails


def _boundary_motion(plan, motion):
    """|out| of slot0 + |in| of slot1 at the chosen offsets — how 'hot' the seam ended up."""
    p0, p1 = plan[0], plan[1]
    v0, f0, _ = motion[p0["clip"]]
    v1, f1, _ = motion[p1["clip"]]
    out0 = cutpoint._endpoint_vec(v0, f0, p0["src_start"] + p0["duration"] - cutpoint.WIN_S,
                                  p0["src_start"] + p0["duration"])
    in1 = cutpoint._endpoint_vec(v1, f1, p1["src_start"], p1["src_start"] + cutpoint.WIN_S)
    return np.hypot(*out0) + np.hypot(*in1)


def test_refine():
    fails = []
    # two clips, each: 30 calm frames then 30 fast frames. Slot length 0.5s (15 frames) leaves
    # plenty of room to slide the cut onto either the calm or the fast half.
    clips = [_tmp(), _tmp()]
    for c in clips:
        _render(c, [((0, 0), 30), ((6, 0), 30)])

    def make_plan():
        return [{"shot": 0, "clip": clips[0], "name": "a", "src_start": 0.0, "duration": 0.5},
                {"shot": 1, "clip": clips[1], "name": "b", "src_start": 0.0, "duration": 0.5}]

    motion = {}
    for c in clips:
        v, fps, _ = mvmotion.step_series(c)
        motion[c] = (v, fps, len(v) / fps)

    # 'action' boundary -> both endpoints should land on the fast half
    p_action = make_plan()
    cutpoint.refine_cut_points(p_action, [{"index": 0, "cut_in": None},
                                          {"index": 1, "cut_in": "action"}])
    hot = _boundary_motion(p_action, motion)

    # 'still' boundary -> both endpoints should land on the calm half
    p_still = make_plan()
    cutpoint.refine_cut_points(p_still, [{"index": 0, "cut_in": None},
                                         {"index": 1, "cut_in": "still"}])
    calm = _boundary_motion(p_still, motion)

    print(f"refine: action seam motion={hot:.2f}px  still seam motion={calm:.2f}px")
    if not hot > calm + 2.0:
        fails.append(f"action seam ({hot:.2f}) should be much hotter than still seam ({calm:.2f})")
    if hot < 3.0:
        fails.append(f"action seam motion {hot:.2f}px too low — didn't find the fast half")
    for c in clips:
        Path(c).unlink(missing_ok=True)
    return fails


def main():
    fails = test_reference() + test_refine()
    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — cut-point styles classified, and cut points refined to match them.")


if __name__ == "__main__":
    main()

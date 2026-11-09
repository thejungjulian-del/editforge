"""Score source clips, match them to the reference's shot slots, render MP4 + AE script."""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

import effects
import face
from analyzer import shot_energy

OUT_W, OUT_H, FPS = 1080, 1920, 30


def probe_size(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


# Where the subject's face should sit in a vertical frame. Slightly above centre reads better
# than dead centre and leaves room for captions underneath.
FACE_TARGET_Y = 0.42
SAFE_X = (0.16, 0.84)     # the face is never allowed outside this box...
SAFE_Y = (0.12, 0.86)     # ...no matter what the reference camera does
MAX_SCALE = 4.0           # past this the footage is mush; pull the anchor in instead

# COVERAGE IS ABSOLUTE. The clip fills the frame, always — no blurred backdrop, no bars, no
# gaps. Julian's call, and he is right that a drifting edge looks sloppy.
#
# That makes zoom and camera motion the things that have to give, and the ORDER matters:
#
#   1. Zoom only as far as coverage actually requires. The old code zoomed in EXTRA to
#      manufacture spare pixels for the reference's pan and roll to move into — that extra
#      zoom is what was eating his face.
#   2. Instead, spend the slack the footage ALREADY has. Landscape footage in a vertical frame
#      is ~3400px wide behind a 1080px window: horizontal pan is nearly free. Vertical slack is
#      zero, so vertical pan is not.
#   3. Whatever motion does not fit in that slack gets damped. Motion is the soft constraint
#      now; framing is not.
MIN_FACE = 0.13           # if the face lands smaller than this, zoom in until it doesn't
                          # (the ONLY thing allowed to zoom past bare coverage — and it can
                          #  never open a gap, since zooming in only ever covers more)


def plan_framing(fw, fh, face, motion, comp_w, comp_h, margin=0.0):
    """Anchor on the face, guarantee the frame is always covered, and clamp the pan.

    Returns (anchor_px, target_pos, base_scale, motion_gain).

    The anchor is the whole trick. AE scales and rotates a layer about its anchor point, so
    putting the anchor ON THE FACE makes every zoom and roll we copied from the reference
    pivot around the face rather than sliding it out of shot.

    base_scale is then SOLVED, not guessed. For every frame of the motion track we ask how far
    the layer must be scaled up for the footage to still cover all four corners of the comp,
    and take the worst case. The old code used a capped heuristic, and when the cap bit, the
    footage no longer covered the frame — that is exactly where the black edges came from.
    """
    if face:
        # keep the anchor off the extreme edge, or covering the frame demands absurd zoom
        ax = min(max(face["x"], 0.18), 0.82) * fw
        ay = min(max(face["y"], 0.18), 0.82) * fh
    else:
        ax, ay = fw / 2.0, fh / 2.0
    anchor = np.array([ax, ay])

    # Dead centre is the CHEAPEST place to put the anchor — any other position needs extra zoom
    # to keep the far edge covered. So start here, and only drift toward the nicer,
    # slightly-high face placement later if the slack pays for it. Buying composition with zoom
    # is what was eating the face.
    centre_t = np.array([comp_w / 2.0, comp_h / 2.0])
    ideal_t = np.array([comp_w / 2.0, comp_h * FACE_TARGET_Y]) if face else centre_t
    target = centre_t

    keys = motion or [[0.0, 0.0, 0.0, 1.0, 0.0]]
    corners = [np.array([x, y]) for x in (-margin, comp_w + margin)
               for y in (-margin, comp_h + margin)]
    centre = np.array([fw / 2.0, fh / 2.0])

    def needed(a, t=None, gx=0.0, gy=0.0, gr=0.0, track=None):
        """Smallest scale that covers the comp for the whole track, at the given motion gains.

        A pull-out (motion scale below 1) would shrink the layer under coverage, so it is
        clamped: the reference's zoom-outs become holds. Losing a pull-out is a smaller price
        than a visible edge. The jsx applies the SAME clamp — if the two ever disagree, edges
        reappear.
        """
        k = 1.0
        tt = target if t is None else t
        for _, dx, dy, s, rot in (track if track is not None else keys):
            pos = tt + np.array([dx * comp_w * gx, dy * comp_h * gy])
            th = np.radians(rot * gr)
            R = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
            for c in corners:
                u = R @ (c - pos)
                # p_layer = a + u/k must land inside [0, fw] x [0, fh]
                for uu, ai, extent in ((u[0], a[0], fw), (u[1], a[1], fh)):
                    room = ai if uu < 0 else (extent - ai)
                    if room <= 1e-6:
                        return float("inf")
                    k = max(k, abs(uu) / room / max(s, 1.0))
        return k

    # An anchor near the edge of the footage forces a huge zoom to keep the frame covered, so
    # slide it back toward centre until the zoom is sane. Never cap the scale itself — capping
    # coverage is what let black edges through in the first place.
    still = [[0.0, 0.0, 0.0, 1.0, 0.0]]
    anchor_used = anchor

    # THE CHEAPEST POSSIBLE ZOOM: the crop that just covers the frame with the SOURCE centred.
    # Nothing beats this — it is the scale at which the footage first reaches all four edges.
    base = needed(centre, centre_t, track=still)

    # The only thing allowed to zoom past that is a face too small to see — and zooming IN can
    # never open a gap. Capped, because pushing a tiny face up to a readable size can demand 5x+
    # on a wide shot, and at that point the footage is mush.
    if face and face.get("size", 0) > 0.02:
        base = max(base, min(MIN_FACE * comp_h / (face["size"] * fh), MAX_SCALE))

    # Where the face LANDS if we do nothing at all — i.e. the framing that costs zero zoom.
    # (Anchor and position are interchangeable for a static layer, so putting the anchor on the
    # face costs nothing by itself; it is only *insisting the face sit at frame centre* that
    # costs zoom, by shoving the crop window off-centre in the source.)
    neutral_t = centre_t + base * (anchor_used - centre)

    # Now drift the face toward the nicer composition — but only as far as the slack we already
    # have will carry it. Composition is never bought with zoom.
    target = neutral_t
    for f in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0):
        t = neutral_t + (ideal_t - neutral_t) * f
        if needed(anchor_used, t, track=still) <= base + 1e-6:
            target = t
            break

    # THE POINT. The zoom is now FIXED at bare coverage. We do not zoom in one pixel further to
    # make room for the reference's camera move — that extra zoom was what ate the face. Instead
    # the move is only allowed to spend slack the footage ALREADY has.
    #
    # Per-axis, because the slack is wildly different per axis. Landscape footage in a vertical
    # frame has ~1200px of horizontal room and ZERO vertical: a sideways pan is free, a vertical
    # one is impossible. A single shared gain would let the impossible vertical move drag the
    # free horizontal one down with it, throwing away motion for nothing.
    grid = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
    tol = base + 1e-6

    def solve(axis):
        for g in grid:
            gains = {"x": 0.0, "y": 0.0, "r": 0.0}
            gains[axis] = g
            if needed(anchor_used, target,
                      gx=gains["x"], gy=gains["y"], gr=gains["r"]) <= tol:
                return g
        return 0.0

    gx, gy, gr = solve("x"), solve("y"), solve("r")

    # Each axis fits on its own; together they can still overrun (a roll eats into the room a
    # pan was counting on). Back all three off together until the combination fits.
    for _ in range(12):
        if needed(anchor_used, target, gx=gx, gy=gy, gr=gr) <= tol:
            break
        gx, gy, gr = gx * 0.8, gy * 0.8, gr * 0.8

    # Keep the face inside the safe box regardless of what survived above.
    for _, dx, dy, _, _ in keys:
        for need, tgt, lo, hi, which in (
            (dx * comp_w, target[0], SAFE_X[0] * comp_w, SAFE_X[1] * comp_w, "x"),
            (dy * comp_h, target[1], SAFE_Y[0] * comp_h, SAFE_Y[1] * comp_h, "y"),
        ):
            if abs(need) < 1e-6:
                continue
            room = (hi - tgt) if need > 0 else (tgt - lo)
            cap = max(0.0, room / abs(need))
            if which == "x":
                gx = min(gx, cap)
            else:
                gy = min(gy, cap)

    gains = [round(float(gx), 4), round(float(gy), 4), round(float(gr), 4)]
    return anchor_used, target, round(float(base), 5), gains, None


def probe_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def find_moments(clip_path, window=0.6, stride=0.4):
    """Slide a window over the clip and score each position by motion energy."""
    duration = probe_duration(clip_path)
    moments = []
    t = 0.0
    while t + window <= duration:
        moments.append({
            "clip": str(clip_path),
            "name": Path(clip_path).name,
            "start": round(t, 3),
            "energy": shot_energy(clip_path, t, t + window),
        })
        t += stride
    if not moments:  # clip shorter than one window
        moments.append({
            "clip": str(clip_path), "name": Path(clip_path).name,
            "start": 0.0, "energy": shot_energy(clip_path, 0, duration),
        })
    for m in moments:
        m["max_len"] = round(duration - m["start"], 3)
    return moments


def match(shots, moments, locks=None):
    """Assign a source moment to each reference shot, matching energy and avoiding back-to-back
    repeats. Reuses moments only when there are fewer than shots.

    `locks` = {shot_index: clip_path} from contentmatch: a shot the user has a look-alike clip for.
    A locked shot is filled from THAT clip (best-energy moment of it), overriding energy/anti-repeat
    — a genuine visual match should be used the same way the reference used its clip.
    """
    if not moments:
        return []
    locks = locks or {}
    used, plan, last_clip = set(), [], None

    for shot in shots:
        need = shot["energy"]

        def cost(m, i):
            c = abs(m["energy"] - need)
            if i in used:
                c += 0.5                       # strongly prefer unused footage
            if m["name"] == last_clip:
                c += 0.25                      # avoid two cuts from the same clip in a row
            if m["max_len"] < shot["duration"]:
                c += 10.0                      # can't fill the slot
            return c

        lock = locks.get(shot["index"])
        cand = [i for i in range(len(moments)) if str(moments[i]["clip"]) == str(lock)] if lock else []
        cand = cand or range(len(moments))     # no lock (or its clip has no moments) -> all moments

        best_i = min(cand, key=lambda i: cost(moments[i], i))
        m = moments[best_i]
        used.add(best_i)
        if len(used) == len(moments):
            used.clear()                       # ran out of fresh footage, allow reuse
        last_clip = m["name"]
        plan.append({
            "shot": shot["index"],
            "clip": m["clip"],
            "name": m["name"],
            "src_start": m["start"],
            "duration": shot["duration"],
            "max_len": m["max_len"],
            "target_energy": need,
            "actual_energy": m["energy"],
            "locked": bool(lock),              # a look-alike match; source-groups won't override it
        })
    return plan


def apply_source_groups(plan, groups):
    """Reproduce the reference's splicing: where it cut ONE source video into several consecutive
    shots, use ONE of the user's clips across those slots too — sequential segments of it — so his
    one clip is spliced the way their one video was. Mutates and returns plan.

    match() already picked a clip per slot by energy; for a group we keep the LONGEST of those
    (most room to splice) and lay its segments back to back. Grouped slots are marked so the
    cut-point refiner leaves their splice layout alone.
    """
    if not groups:
        return plan
    by_shot = {p["shot"]: p for p in plan}
    for g in groups:
        slots = [by_shot[i] for i in g if i in by_shot]
        if len(slots) < 2:
            continue
        if any(p.get("locked") for p in slots):
            continue                           # a look-alike match wins; don't overwrite its clip
        best = max(slots, key=lambda p: probe_duration(p["clip"]))
        clip, name, dur = best["clip"], best["name"], probe_duration(best["clip"])
        total = sum(p["duration"] for p in slots)
        if dur >= total:
            # back-to-back consecutive segments — a literal splice of one continuous take
            starts, acc = [], 0.0
            for p in slots:
                starts.append(acc)
                acc += p["duration"]
        else:
            # not enough footage for non-overlapping pieces: spread the in-points across the clip
            # so each segment still fits (they may overlap — the best a short clip allows)
            span = max(0.0, dur - max(p["duration"] for p in slots))
            starts = list(np.linspace(0, span, len(slots)))
        for p, s in zip(slots, starts):
            p["clip"], p["name"] = clip, name
            p["src_start"] = round(float(s), 3)
            p["max_len"] = round(dur - float(s), 3)
            p["grouped"] = True
    return plan


def make_seamless_loop(plan):
    """Make the edit loop seamlessly: the LAST shot becomes the first clip's run-up, ending exactly
    where the first shot begins — so when the video replays, its last frame flows straight into its
    first frame with no visible jump (Julian's signature style). Mutates and returns plan.

    Needs `first_clip` to have `last_shot_duration` of footage sitting BEFORE the first shot's
    in-point. If it doesn't, fall back to the clip's own opening (a soft bookend). If even that
    won't fit, leave the edit as-is. The loop shot is flagged so it plays at normal speed (no
    slow-mo / ramp) into the opening.
    """
    if len(plan) < 2:
        return plan
    first, last = plan[0], plan[-1]
    dur = last["duration"]
    s0 = first["src_start"]
    clip_dur = probe_duration(first["clip"])
    if s0 >= dur:
        s_loop = s0 - dur                         # ends exactly at the first shot's in-point
    elif clip_dur >= dur:
        s_loop = 0.0                              # not enough run-up: soft bookend on the opening
    else:
        return plan                              # clip too short to loop cleanly
    last["clip"], last["name"] = first["clip"], first["name"]
    last["src_start"] = round(s_loop, 3)
    last["max_len"] = round(clip_dur - s_loop, 3)
    last["loop"] = True
    return plan


SLOWMO_FACTOR = 0.5  # low-energy shots play at half speed

# A speed ramp must never slow all the way to a frozen frame (Julian: "the slowdown should never
# be zero"). Blending the retime curve toward a constant-speed line lifts its slowest sections off
# a dead stop while keeping the ramp's shape and its EXACT endpoints (the line shares them). At
# a=0.18 the slowest speed is at least ~18% of real time — dramatic, but never a freeze.
RETIME_FLOOR_BLEND = 0.18


def _retime_keys(prof, src_start, dur, src_cap):
    """Normalised speed-ramp profile [[out_frac, src_frac], ...] -> AE Time Remap keys
    [[comp_time, source_time], ...]. Floors the speed so it never freezes, caps source times just
    inside the clip. Returns [] for a profile too short to be a ramp."""
    if len(prof) < 2:
        return []
    raw = [(u * dur, src_start + v * dur) for u, v in prof]
    u0, uN, s0, sN = raw[0][0], raw[-1][0], raw[0][1], raw[-1][1]
    span = uN - u0
    a = RETIME_FLOOR_BLEND
    keys = []
    for u, s in raw:
        lin = s0 + (sN - s0) * ((u - u0) / span if span else 0.0)
        s_floored = (1 - a) * s + a * lin
        keys.append([round(u, 4), min(round(s_floored, 4), src_cap)])
    return keys


def _build_luts(plan, ref_color, work):
    """One LUT per source clip: each clip has its own colour cast to correct from."""
    luts = {}
    for name in {p["name"] for p in plan}:
        clip = next(p["clip"] for p in plan if p["name"] == name)
        src = effects.lab_stats(clip)
        if not src:
            continue
        luts[name] = effects.make_lut(src, ref_color, work / f"lut_{name}.cube")
    return luts


def _filter_chain(p, i, fx, luts, flash_shots, beat_shots, median_energy):
    """Filters for one shot, in the order they have to happen."""
    chain = [
        f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase",
        f"crop={OUT_W}:{OUT_H}",
        "setsar=1",
    ]

    if fx.get("slowmo") and p.get("slowed"):
        chain.append(f"setpts={1 / SLOWMO_FACTOR:.4f}*PTS")

    chain.append(f"fps={FPS}")

    if fx.get("color") and p["name"] in luts:
        lut = str(luts[p["name"]]).replace("\\", "/").replace(":", "\\:")
        chain.append(f"lut3d=file='{lut}'")

    if fx.get("zoom") and p["shot"] in beat_shots:
        frames = max(2, round(p["duration"] * FPS))
        # punch in on the cut, ease back out across the shot
        chain.append(
            f"zoompan=z='max(1.0,1.09-0.09*on/{frames - 1})'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={OUT_W}x{OUT_H}:fps={FPS}"
        )

    if fx.get("flash") and p["shot"] in flash_shots:
        chain.append("fade=t=in:st=0:d=0.06:color=white")

    return ",".join(chain)


def render(plan, reference_video, out_path, work_dir, use_audio=False,
           fx=None, ref_color=None, flash_shots=(), beat_shots=()):
    """Cut each slot, normalize to vertical, apply effects, concat. Optionally lay the
    reference's audio over the result — off by default, that track isn't yours."""
    fx = fx or {}
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    for f in list(work.glob("piece_*.mp4")) + list(work.glob("lut_*.cube")):
        f.unlink()

    luts = _build_luts(plan, ref_color, work) if (fx.get("color") and ref_color) else {}
    flash_shots, beat_shots = set(flash_shots), set(beat_shots)

    energies = [p["target_energy"] for p in plan]
    median_energy = float(np.median(energies)) if energies else 0.0

    # decide slow-mo up front so the trim length can account for it
    for p in plan:
        p["slowed"] = bool(
            fx.get("slowmo")
            and p["target_energy"] < median_energy
            and p["max_len"] >= p["duration"] * SLOWMO_FACTOR
        )

    pieces = []
    for i, p in enumerate(plan):
        piece = work / f"piece_{i:03d}.mp4"
        # a slowed shot only needs half as much source to fill the same slot
        take = p["duration"] * SLOWMO_FACTOR if p["slowed"] else p["duration"]
        vf = _filter_chain(p, i, fx, luts, flash_shots, beat_shots, median_energy)

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-ss", str(p["src_start"]), "-t", str(round(take, 3)), "-i", p["clip"],
             "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-t", str(p["duration"]),
             "-pix_fmt", "yuv420p", str(piece)],
            check=True, capture_output=True,
        )
        pieces.append(piece)

    # concat demuxer resolves paths relative to the list file, so these must be absolute
    listfile = work / "concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in pieces))
    silent = work / "silent.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-c", "copy", str(silent)],
        check=True, capture_output=True,
    )

    if not use_audio:
        silent.replace(out_path)
        return out_path

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(reference_video),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
         "-shortest", str(out_path)],
        check=True, capture_output=True,
    )
    return out_path


# Quality polish, applied to every clip when enhance is on. COLOUR-NEUTRAL by Julian's rule
# ("don't change the coloring of anything"): only detail, never tone or colour.
#   hqdn3d  — light denoise, cleans phone/compression noise before we sharpen it
#   unsharp — moderate LUMA sharpen (no chroma sharpen; chroma_amount 0.0 leaves colour untouched)
# No contrast/saturation/brightness change of any kind. It makes footage crisper and cleaner; it
# cannot add real resolution (that needs AI upscaling).
ENHANCE_VF = "hqdn3d=2:1.5:4:4,unsharp=5:5:0.6:5:5:0.0"

# Soft "heaven" glow (an Orton-style bloom): a heavily-blurred copy screened back over the frame,
# so highlights bloom and the image goes dreamy/ethereal. Screen mode lifts the brights most,
# which is what makes it read as a glow rather than a flat haze; low opacity keeps it SLIGHT.
# Must be appended LAST — it splits the stream and recombines, so nothing chains after it.
GLOW_VF = ("split[glowbase][glowsrc];[glowsrc]gblur=sigma=9[glowblur];"
           "[glowbase][glowblur]blend=all_mode=screen:all_opacity=0.20")


def prepare_ae_footage(plan, ref_color, work, color=True, enhance=False, glow=False):
    """Transcode each source clip to an AE-friendly intermediate, baking in the colour grade
    and/or the quality-enhancement polish.

    After Effects will not reliably decode ffmpeg-muxed H.264: it imports the file happily
    and then renders Media Offline bars. So the footage AE sees is always ProRes in a .mov,
    grade or no grade.

    AE's own LUT effect is scriptable in theory but fiddly and I can't test it headlessly,
    so the grade is applied to the footage before AE ever sees it. You can still grade
    on top inside AE.
    """
    work = Path(work)
    luts = _build_luts(plan, ref_color, work) if color and ref_color else {}
    prepared = {}
    for p in plan:
        name = p["name"]
        if name in prepared:
            continue
        out = work / f"ae_{Path(name).stem}.mov"
        vf = []
        if name in luts:
            lut_arg = str(luts[name]).replace("\\", "/").replace(":", "\\:")
            vf.append(f"lut3d=file='{lut_arg}'")
        if enhance:
            vf.append(ENHANCE_VF)                 # denoise/sharpen AFTER the grade
        if glow:
            vf.append(GLOW_VF)                    # glow LAST — it splits and recombines the stream
        vf_arg = ["-vf", ",".join(vf)] if vf else []
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", p["clip"], *vf_arg, "-an",
             "-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", str(out)],
            check=True, capture_output=True,
        )
        prepared[name] = str(out.resolve())
    return prepared


def _extract_audio(reference_video, work):
    """Hand AE a plain WAV — importing the reference .mp4 just for its audio hits the same
    decoder problem as the video does."""
    out = Path(work) / "reference_audio.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(reference_video),
         "-vn", "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True,
    )
    return str(out.resolve())


def write_jsx(plan, reference_video, jsx_path, comp_name="EditForge",
              fx=None, ref_color=None, flash_shots=(), motion_tracks=None, trans=None,
              rgb_split=None, letterbox=None, vignette=0.0, speeds=None,
              warps=None, caption_slots=(), corrections=None,
              work_dir="work", use_audio=False):
    """Emit an After Effects ExtendScript that rebuilds the edit as a real, editable comp.

    Cuts become layers, flashes become white solids, and the reference's measured camera
    move becomes Position/Scale/Rotation keyframes on your footage — everything lands as
    keyframes you can drag, not as pixels baked into a render.
    """
    fx = fx or {}
    flash_shots = set(flash_shots)
    motion_tracks = motion_tracks or {}
    trans = trans or {}

    graded = prepare_ae_footage(plan, ref_color, work_dir, color=bool(fx.get("color")),
                                enhance=bool(fx.get("enhance")), glow=bool(fx.get("glow")))

    energies = [p["target_energy"] for p in plan]
    median_energy = float(np.median(energies)) if energies else 0.0

    items, t = [], 0.0
    for p in plan:
        # ROOT FIX for the recurring "gap between clips": clamp this slot to the REAL footage AE
        # will import, up front, before anything reads src_start/max_len. src_start/max_len can be
        # set from a frame-count duration ESTIMATE (mvmotion, in refine) that OVERSHOOTS the real
        # file; unclamped, the slot's out-point lands past the last frame and the layer goes
        # transparent = a black gap. probe_duration is the authoritative length AE sees, so make
        # every later decision (slow-mo, coverage, ramp) use it. This is why the gap kept coming
        # back: each past fix patched one code path while others still trusted the estimate.
        src = graded.get(p["name"], str(Path(p["clip"]).resolve()))
        fw, fh = probe_size(src)
        real_dur = round(probe_duration(src), 4)
        src_cap = round(real_dur - 1.5 / FPS, 4)
        p["src_start"] = round(min(p["src_start"], max(0.0, src_cap - p["duration"])), 4)
        p["max_len"] = round(real_dur - p["src_start"], 4)

        slowed = bool(
            fx.get("slowmo")
            and not p.get("loop")                 # the loop shot plays clean into the opening
            and p["target_energy"] < median_energy
            and p["max_len"] >= p["duration"] * SLOWMO_FACTOR
        )
        track = motion_tracks.get(p["shot"], {}) if fx.get("motion") else {}

        tr = trans.get(p["shot"], {}) if fx.get("transitions") else {}
        kind = tr.get("type", "hard")
        tdur = float(tr.get("duration", 0.0))
        if kind == "dissolve":
            # the incoming layer has to start EARLY to overlap the outgoing one, so it can
            # only dissolve for as long as it has footage sitting before its in-point
            tdur = min(tdur, p["src_start"], p["duration"])
            if tdur < 1 / FPS:
                kind = "hard"
        elif kind in ("flash", "fade", "whip"):
            tdur = min(tdur, p["duration"])

        # Ripple. The DETECTION is trustworthy (conservative: 2 of 21 shots on the real
        # reference). The measured PARAMETERS are not, so they get clamped into a range that
        # actually looks like a ripple:
        #
        #   amplitude — the detector undershoots by ~2x (optical flow reports the change BETWEEN
        #               frames, not the wave's displacement from rest), and raw it lands at 2-5px
        #               in a 1080 frame, i.e. invisible. Doubled, then floored so it reads.
        #   speed     — measured 5-10 waves/sec, which is not a ripple, it is a seizure.
        #   direction — from a noisy fit, so snapped to the nearest axis. An odd angle guessed
        #               from noise looks like a mistake; a clean horizontal ripple looks intended.
        w = (warps or {}).get(p["shot"]) if fx.get("warp") else None
        warp_fx = None
        if w:
            height = min(max(w["wave_amp"] * 2.0 * OUT_W, 9.0), 34.0)
            width = min(max(w["wavelength"] * OUT_W, 90.0), 520.0)
            speed = min(max(w["cycles_per_frame"] * FPS, 0.6), 2.5)
            direction = round(w["direction"] / 90.0) * 90.0
            warp_fx = {"height": round(height, 1), "width": round(width, 1),
                       "speed": round(speed, 2), "direction": direction}

        # RGB split: one extra layer per channel we could actually MEASURE. An unmeasurable
        # channel comes back None from the detector and is simply not applied — better a
        # partial split than an invented one.
        split = (rgb_split or {}).get(p["shot"]) if fx.get("rgb_split") else None
        split_layers = []
        if split:
            for ch in ("r", "b"):
                off = split.get(ch)
                if off and (abs(off[0]) > 0.5 or abs(off[1]) > 0.5):
                    split_layers.append({"ch": ch, "dx": off[0], "dy": off[1]})

        # Speed ramp -> AE Time Remap. The profile is normalised (out_frac -> src_frac), so it maps
        # onto whatever clip landed in this slot, redistributing time WITHIN the shot (no extra
        # footage). Source times are capped at src_cap (just inside the real file) so a Time Remap
        # value never lands at/past the source duration — AE rejects that ("setValueAtTime out of
        # range"). src_start/max_len/src_cap were all pinned to the real duration at the loop top.
        # Only ramp a clip that actually has the footage the ramp needs (src_start..+duration inside
        # the clip); a too-short clip would clamp its tail to a frozen frame — exactly the freeze
        # Julian wants gone. The loop shot plays clean into the opening, so no ramp on it either.
        has_footage = p["src_start"] + p["duration"] <= src_cap
        prof = ((speeds or {}).get(p["shot"], [])
                if fx.get("velocity") and has_footage and not p.get("loop") else [])
        remap = _retime_keys(prof, p["src_start"], p["duration"], src_cap)

        # How much to stretch the layer. A clip SHORTER than its slot would otherwise run out of
        # footage mid-shot and show black — the "gap between clips". Stretch it just enough to fill
        # the slot instead. This subsumes slow-mo (a 2x stretch on calm shots) and only slows a
        # too-short clip as far as needed. A speed ramp drives its own timing via Time Remap, so
        # leave stretch at 100 there and let the clamp above guard its footage.
        avail = max(p["max_len"], 1e-3)
        if len(remap) > 1:
            stretch = 100.0
        else:
            base_stretch = 100.0 / SLOWMO_FACTOR if slowed else 100.0
            cover_stretch = float(np.ceil(p["duration"] * 100.0 / avail))
            stretch = max(base_stretch, cover_stretch)

        if fx.get("native"):
            # Julian's rule: the ONLY time zooming in is allowed is when a clip is not big enough to
            # match the rest. FILL the comp (max of the two ratios = cover), which does EXACTLY that:
            #   - scale > 1 (a zoom-in) happens iff the clip is smaller than the comp in some
            #     dimension — i.e. it isn't big enough to fill/match, the one case he permits.
            #   - a clip big enough (comp-sized or larger) scales DOWN to fit, never zoomed in.
            # This also keeps every clip filling the frame so they match (no black bars), and does
            # not reintroduce the old 4K "still zooming" bug (a 4K clip scales down, not in).
            # No camera move, no face-framing; centred.
            anchor = [fw / 2.0, fh / 2.0]
            target = [OUT_W / 2.0, OUT_H / 2.0]
            base = max(OUT_W / fw, OUT_H / fh)
            gain, bg, fc = [0.0, 0.0, 0.0], None, None
            item_motion = []
        else:
            # Anchor the framing on the subject's face, solve the scale that always covers the
            # frame, and clamp the pan so the face cannot be dragged out of shot.
            fc = face.find_face(src, p["src_start"], p["duration"]) if fx.get("face") else None
            split_margin = max([abs(v) for s in split_layers for v in (s["dx"], s["dy"])] + [0.0])
            # A wave DISPLACES pixels, so the footage has to cover the frame plus the wave's swing —
            # otherwise the ripple drags an empty edge into shot, which is exactly what Julian said
            # must never happen. Feed the amplitude into the coverage solver as margin.
            warp_margin = warp_fx["height"] if warp_fx else 0.0
            anchor, target, base, gain, bg = plan_framing(
                fw, fh, fc, track.get("motion", []), OUT_W, OUT_H,
                margin=split_margin + warp_margin + 2)
            item_motion = track.get("motion", [])

        # Auto-correction overrides for this shot, derived from the last render-vs-reference score
        # (correct.derive_corrections). They only nudge flashes and transitions toward the reference.
        corr = corrections or {}
        sh = p["shot"]
        trans_kind = kind if kind != "hard" else (
            "flash" if (fx.get("flash") and sh in flash_shots) else "hard")
        eff_tdur = tdur if tdur else 0.08
        fov = corr.get("flash", {}).get(sh)
        if fov == "off" and trans_kind == "flash":
            trans_kind = "hard"                          # kill a flash the reference doesn't have
        elif fov == "boost":
            trans_kind = "flash"                         # force + lengthen a flash that didn't register
            eff_tdur *= corr.get("flash_gain", {}).get(sh, 1.0)
        tov = corr.get("transition", {}).get(sh)
        if tov:
            trans_kind = tov                             # a transition that came out as a hard cut
            eff_tdur = max(eff_tdur, 0.12) * corr.get("trans_gain", {}).get(sh, 1.0)
        eff_tdur = round(min(eff_tdur, p["duration"]), 4)

        items.append({
            "file": src,
            "label": p["name"],
            "remap": remap,
            "anchor": [round(float(anchor[0]), 2), round(float(anchor[1]), 2)],
            "target": [round(float(target[0]), 2), round(float(target[1]), 2)],
            "base": base,
            "bg": bg,
            "gain": gain,
            "face": fc,
            "src_start": round(p["src_start"], 4),
            "at": round(t, 4),
            "dur": round(p["duration"], 4),
            "stretch": round(stretch, 2),
            # flash is one of the transition types; both may be nudged by auto-corrections above
            "trans": trans_kind,
            "tdur": eff_tdur,
            # [t, dx, dy, scale, rot_deg] per key — the reference camera's actual move ([] = static)
            "motion": item_motion,
            "overscale": track.get("overscale", 1.0),
            "split": split_layers,
            "warp": warp_fx,
        })
        t += p["duration"]

    payload = json.dumps({
        "comp": comp_name, "w": OUT_W, "h": OUT_H, "fps": FPS,
        "total": round(t, 3),
        "audio": _extract_audio(reference_video, work_dir) if use_audio else None,
        "letterbox": letterbox if fx.get("letterbox") else None,
        "vignette": round(float(vignette), 4) if fx.get("vignette") else 0.0,
        "captions": list(caption_slots) if fx.get("captions") else [],
        "items": items,
    }, indent=1)

    jsx = """// Generated by EditForge.
// Run in After Effects:  File > Scripts > Run Script File...
var DATA = __EDITFORGE_DATA__;

// Import each distinct file once and remember it, so a clip reused across several
// shot slots becomes one footage item rather than N duplicates in the project panel.
var CACHE = {};
var NOT_FOUND = [];   // AE never saw the file on disk
var OFFLINE = [];     // AE imported it but cannot decode it -> colour bars

function importFile(path) {
    if (CACHE[path] !== undefined) { return CACHE[path]; }
    var f = new File(path);
    if (!f.exists) { NOT_FOUND.push(path); CACHE[path] = null; return null; }
    var item = app.project.importFile(new ImportOptions(f));
    if (item.footageMissing) { OFFLINE.push(path); }
    CACHE[path] = item;
    return item;
}

// Channel Mixer takes NUMERIC sliders (0-200%), unlike Shift Channels which takes enums —
// so isolating a channel this way is safe to script without guessing enum indices.
function mixer(layer) {
    return layer.property("ADBE Effect Parade").addProperty("ADBE Channel Mixer");
}
function keepChannel(layer, ch) {
    var m = mixer(layer);
    if (ch !== "r") { m.property("Red-Red").setValue(0); }
    if (ch !== "g") { m.property("Green-Green").setValue(0); }
    if (ch !== "b") { m.property("Blue-Blue").setValue(0); }
}
function killChannel(layer, ch) {
    var m = mixer(layer);
    if (ch === "r") { m.property("Red-Red").setValue(0); }
    if (ch === "g") { m.property("Green-Green").setValue(0); }
    if (ch === "b") { m.property("Blue-Blue").setValue(0); }
}

// One footage layer for shot `it`, shifted by (ox, oy) comp pixels. The offset is baked
// into the position keys rather than applied as a separate effect, so a split copy inherits
// the camera move exactly instead of fighting it.
function buildLayer(comp, footage, it, i, ox, oy) {
    var layer = comp.layers.add(footage);
    layer.name = it.label + " [" + i + "]";

    // A stretched layer plays its source at (100/stretch) speed, so source time
    // src_start lands at layer time startTime + src_start * (stretch/100).
    var rate = it.stretch / 100;
    if (it.stretch !== 100) { layer.stretch = it.stretch; }

    if (it.remap.length > 1) {
        // Speed ramp. Time Remap drives source time directly, so startTime must be 0 —
        // otherwise the layer's own time offset fights the remap curve.
        layer.startTime = 0;
        layer.timeRemapEnabled = true;
        var tr = layer.property("Time Remap");

        // EXTEND THE LAYER FIRST. Enabling Time Remap leaves the layer only as long as its
        // SOURCE clip, and a keyframe cannot be set outside the layer's own time range. A
        // shot sitting later in the comp than the clip is long therefore threw
        // "Unable to set value at time". Time Remap is precisely what makes a layer
        // extendable past its source, so widen it, write the keys, then trim back.
        layer.outPoint = it.at + it.dur + 1;

        for (var q = 0; q < it.remap.length; q++) {
            tr.setValueAtTime(it.at + it.remap[q][0], it.remap[q][1]);
        }
        // Drop AE's own default remap keys (at the source's start and end). They sit outside
        // our window and would drag the curve toward the wrong source times.
        for (var kk = tr.numKeys; kk >= 1; kk--) {
            var kt = tr.keyTime(kk);
            if (kt < it.at - 0.0001 || kt > it.at + it.dur + 0.0001) { tr.removeKey(kk); }
        }
    } else {
        // A stretched layer plays its source at (100/stretch) speed, so source time
        // src_start lands at layer time startTime + src_start * (stretch/100).
        layer.startTime = it.at - it.src_start * rate;
    }

    // in/out come last: enabling Time Remap rewrites them to the source's full length
    layer.inPoint = (it.trans === "dissolve") ? it.at - it.tdur : it.at;
    layer.outPoint = it.at + it.dur;

    // ANCHOR ON THE FACE. AE scales and rotates a layer about its anchor point, so putting
    // the anchor on the subject's face makes every zoom and roll copied from the reference
    // pivot AROUND the face instead of sliding it out of shot. it.base is the scale that was
    // SOLVED (in plan_framing) to always cover the frame — no black edges, ever — and it.gain
    // clamps the pan so the face cannot leave the safe box.
    layer.anchorPoint.setValue([it.anchor[0], it.anchor[1]]);
    var base = it.base * 100;
    var pos = layer.property("Position");
    var scale = layer.property("Scale");
    var rot = layer.property("Rotation");

    if (it.motion.length > 1) {
        // Replay the reference camera: each key is [t, dx, dy, scale, rot] where dx/dy are
        // fractions of the frame, so the move transfers to footage of any size.
        //
        // These three lines MUST match plan_framing()'s coverage solver exactly: pan and
        // rotation damped by the same gain, and a zoom-out clamped at 1 so the layer can never
        // shrink below coverage. If the jsx and the solver disagree, the frame stops being
        // covered and the edge shows.
        for (var m = 0; m < it.motion.length; m++) {
            var k = it.motion[m], kt = it.at + k[0];
            if (kt > it.at + it.dur) { break; }
            var ks = Math.max(k[3], 1.0);
            pos.setValueAtTime(kt, [it.target[0] + k[1] * DATA.w * it.gain[0] + ox,
                                    it.target[1] + k[2] * DATA.h * it.gain[1] + oy]);
            scale.setValueAtTime(kt, [base * ks, base * ks]);
            rot.setValueAtTime(kt, k[4] * it.gain[2]);
        }
    } else {
        pos.setValue([it.target[0] + ox, it.target[1] + oy]);
        scale.setValue([base, base]);
    }
    return layer;
}

app.beginUndoGroup("EditForge Build");

// Stamp the comp so re-running never leaves two identically-named comps to confuse us.
var stamp = new Date();
var name = DATA.comp + " " + stamp.getHours() + ":"
         + (stamp.getMinutes() < 10 ? "0" : "") + stamp.getMinutes() + ":"
         + (stamp.getSeconds() < 10 ? "0" : "") + stamp.getSeconds();
var comp = app.project.items.addComp(name, DATA.w, DATA.h, 1.0, DATA.total, DATA.fps);
var missing = 0;

// Build FRONT TO BACK, so each later shot lands above the one before it. That ordering
// is load-bearing: a dissolve overlaps two shots in time, and the incoming shot has to be
// the one on top for its opacity ramp to reveal it over the outgoing shot.
for (var i = 0; i < DATA.items.length; i++) {
    var it = DATA.items[i];
    var footage = importFile(it.file);
    if (footage === null) { missing++; continue; }

    // ox/oy shift the whole layer — used to build an RGB split, where the same footage is
    // stacked with one channel isolated and nudged.
    var layer = buildLayer(comp, footage, it, i, 0, 0);

    // RGB split: keep only the split channel on a copy, offset it, and add it back over a
    // base layer that has had that channel removed. Channels the detector could not measure
    // are absent from it.split and simply do not get one.
    for (var sp = 0; sp < it.split.length; sp++) {
        var s = it.split[sp];
        killChannel(layer, s.ch);                                   // base loses the channel
        var copy = buildLayer(comp, footage, it, i, s.dx, s.dy);    // copy carries it, shifted
        copy.name = it.label + " [" + i + "] " + s.ch.toUpperCase();
        keepChannel(copy, s.ch);
        copy.blendingMode = BlendingMode.ADD;
    }

    // --- the transition that brings this shot in, as measured from the reference ---

    if (it.trans === "dissolve") {
        // cross-fade up over the outgoing shot, which is still playing underneath
        var op = layer.property("Opacity");
        op.setValueAtTime(it.at - it.tdur, 0);
        op.setValueAtTime(it.at, 100);
    }

    if (it.warp) {
        // Ripple. Values arrive already clamped to a range that reads as a ripple (see
        // write_jsx) — the raw measurement is invisible at one end and a seizure at the other.
        // The layer is scaled with extra margin for the wave's swing, so it cannot pull an empty
        // edge into shot.
        var ww = layer.property("ADBE Effect Parade").addProperty("ADBE Wave Warp");
        ww.property("Wave Height").setValue(it.warp.height);
        ww.property("Wave Width").setValue(it.warp.width);
        ww.property("Direction").setValue(it.warp.direction);
        ww.property("Wave Speed").setValue(it.warp.speed);
    }

    if (it.trans === "whip") {
        // a whip is motion + smear. The motion is already on the layer (the camera track
        // measured it); this adds the smear that motion causes.
        var db = layer.property("ADBE Effect Parade").addProperty("ADBE Motion Blur");
        var len = db.property("Blur Length");
        len.setValueAtTime(it.at, 180);
        len.setValueAtTime(it.at + it.tdur, 0);
        db.property("Direction").setValue(90);
    }
}

// Overlay solids go on last so they sit ABOVE every footage layer — a flash under the
// footage would be invisible.
for (var i = 0; i < DATA.items.length; i++) {
    var it = DATA.items[i];

    if (it.trans === "flash") {
        var flash = comp.layers.addSolid([1, 1, 1], "flash " + i, DATA.w, DATA.h, 1.0);
        flash.startTime = it.at;
        flash.inPoint = it.at;
        flash.outPoint = it.at + it.tdur;
        var fo = flash.property("Opacity");
        fo.setValueAtTime(it.at, 100);
        fo.setValueAtTime(it.at + it.tdur, 0);
    }

    if (it.trans === "fade") {
        // dip through black: up into black across the cut, then back out
        var half = it.tdur / 2;
        var blk = comp.layers.addSolid([0, 0, 0], "fade " + i, DATA.w, DATA.h, 1.0);
        blk.startTime = it.at - half;
        blk.inPoint = it.at - half;
        blk.outPoint = it.at + half;
        var bo = blk.property("Opacity");
        bo.setValueAtTime(it.at - half, 0);
        bo.setValueAtTime(it.at, 100);
        bo.setValueAtTime(it.at + half, 0);
    }
}

// Caption SLOTS. We copy where the reference put its text and when — not the words. Its
// script is not yours, and its words would be useless in your video. Each slot is a real,
// editable text layer: type over the placeholder.
for (var ci = 0; ci < DATA.captions.length; ci++) {
    var c = DATA.captions[ci];
    var txt = comp.layers.addText("YOUR CAPTION " + (ci + 1));
    var st = txt.property("Source Text");
    var doc = st.value;
    doc.fontSize = Math.max(12, c.h * DATA.h * 1.25);
    doc.fillColor = [1, 1, 1];
    doc.applyStroke = true;
    doc.strokeColor = [0, 0, 0];
    doc.strokeWidth = Math.max(2, c.h * DATA.h * 0.12);
    doc.strokeOverFill = false;
    doc.justification = ParagraphJustification.CENTER_JUSTIFY;
    st.setValue(doc);
    txt.property("Position").setValue([c.x * DATA.w, c.y * DATA.h]);
    txt.startTime = 0;
    txt.inPoint = c.t0;
    txt.outPoint = c.t1;
    txt.name = "caption " + (ci + 1);
}

// Vignette: a black solid with an inverted, heavily feathered ellipse punched out of it, so
// only the corners darken. Opacity carries the measured strength.
if (DATA.vignette > 0) {
    var vig = comp.layers.addSolid([0, 0, 0], "vignette", DATA.w, DATA.h, 1.0);
    var vm = vig.property("Masks").addProperty("Mask");
    var shape = new Shape();
    var rx = DATA.w * 0.62, ry = DATA.h * 0.62, cx = DATA.w / 2, cy = DATA.h / 2;
    var kx = rx * 0.5523, ky = ry * 0.5523;   // circle-to-bezier constant
    shape.vertices = [[cx, cy - ry], [cx + rx, cy], [cx, cy + ry], [cx - rx, cy]];
    shape.inTangents = [[-kx, 0], [0, -ky], [kx, 0], [0, ky]];
    shape.outTangents = [[kx, 0], [0, ky], [-kx, 0], [0, -ky]];
    shape.closed = true;
    vm.property("Mask Path").setValue(shape);
    vm.maskMode = MaskMode.SUBTRACT;
    vm.maskFeather.setValue([DATA.w * 0.45, DATA.w * 0.45]);
    vig.property("Opacity").setValue(Math.min(100, DATA.vignette * 140));
}

// Letterbox bars last of all — they mask everything, including the vignette.
if (DATA.letterbox) {
    var lb = DATA.letterbox, bars = [
        ["top",    0,                          0,                          DATA.w, DATA.h * lb.top],
        ["bottom", 0,                          DATA.h * (1 - lb.bottom),   DATA.w, DATA.h * lb.bottom],
        ["left",   0,                          0,                          DATA.w * lb.left,  DATA.h],
        ["right",  DATA.w * (1 - lb.right),    0,                          DATA.w * lb.right, DATA.h]
    ];
    for (var bi = 0; bi < bars.length; bi++) {
        var b = bars[bi];
        if (b[3] < 1 || b[4] < 1) { continue; }
        var bar = comp.layers.addSolid([0, 0, 0], "bar " + b[0], Math.ceil(b[3]), Math.ceil(b[4]), 1.0);
        bar.property("Position").setValue([b[1] + b[3] / 2, b[2] + b[4] / 2]);
    }
}

if (DATA.audio) {
    var audio = importFile(DATA.audio);
    if (audio !== null) {
        var track = comp.layers.add(audio);
        track.startTime = 0;
        track.moveToEnd();
    }
}

comp.openInViewer();
app.endUndoGroup();

var report = "EditForge built \\"" + name + "\\"\\n"
           + comp.numLayers + " layers, " + DATA.items.length + " shots, "
           + DATA.total + "s\\n";
if (NOT_FOUND.length) {
    report += "\\nNOT ON DISK (" + NOT_FOUND.length + "):\\n" + NOT_FOUND.join("\\n") + "\\n";
}
if (OFFLINE.length) {
    report += "\\nAE CANNOT DECODE (" + OFFLINE.length + ") - these show colour bars:\\n"
            + OFFLINE.join("\\n") + "\\n";
}
if (!NOT_FOUND.length && !OFFLINE.length) {
    report += "\\nAll footage imported and readable.";
}
alert(report);
""".replace("__EDITFORGE_DATA__", payload)   # literal token, not %-formatting: a stray '%'
                                            # anywhere in the ExtendScript used to break the build

    Path(jsx_path).write_text(jsx)
    return jsx_path

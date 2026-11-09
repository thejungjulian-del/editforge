"""Where a cut lands, not just where it is.

analyzer.py already knows WHERE the reference cuts (bitstream.py) and WHAT the transition is
(transitions.py). This module adds the third thing an editor actually chooses: the exact frame a
cut sits on, relative to the motion around it. A good cut is hidden by motion — you cut ON an
action so the eye, already moving, never registers the seam; or you match the motion across the
seam so it flows; or, on a calm shot, you cut on stillness on purpose. Those are different
choices and they read differently.

Two halves:

  reference_cut_styles() — reads, per cut in the reference, HOW it was cut, from the camera
    motion just before and just after the boundary (mvmotion's bitstream vectors, not pixels):
      whip    — the transition itself is a whip (motion + blur); handled elsewhere, flagged here.
      match   — both sides moving, and moving the SAME way: the motion continues across the cut.
      action  — both sides moving, but not aligned: the cut is hidden by motion, not matched.
      still   — both sides calm: a deliberate cut on stillness.
      hard    — one side moving, one still: a plain, exposed cut.

  refine_cut_points() — the reproduction. match() in builder.py already picked WHICH of the
    user's clips fills each slot; this slides each clip's in/out point (keeping its length, so the
    rhythm the whole tool copies is untouched) so consecutive clips realise the reference's cut
    style at each boundary: cut on action -> land both endpoints on the clip's fastest frames;
    still -> its calmest; match -> frames whose motion points the same way.

Honest limit: we can only steer toward what a given clip actually contains. A clip with no fast
frame can't be made to "cut on action"; we get as close as its footage allows.
"""
import subprocess

import numpy as np

import faceangle
import mvmotion


def _clip_duration(path):
    """Authoritative clip duration (ffprobe), so offsets never place a segment past the real file.
    len(motion_vectors)/fps OVERSHOOTS on some files, which pushed src_start past the footage and
    left a black gap at the shot's tail — the recurring gap bug. This is the real length AE sees."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", str(path)], capture_output=True, text=True)
        return float(r.stdout.strip())
    except Exception:
        return 0.0

FACE_W = 1.0     # weight of the flattering-angle preference against the cut-motion match

# --- reference side -------------------------------------------------------------------------
WIN = 3          # frames each side of a boundary to characterise the motion
# "Moving" vs "calm" is judged against the reference's OWN motion spread (p20=calm, p80=active),
# not a fixed number and not the median — a median breaks when most shots move at one rate. A
# side counts as moving above the halfway mark of that spread, calm below the quarter mark.
STILL_FLOOR = 0.002   # if the whole reference's active level is below this (fraction of width
                      # per frame), it is a static reference: every cut is a cut on stillness.
MOVE_FRAC = 0.5
CALM_FRAC = 0.25
ALIGN = 0.5      # cosine above this = the two sides point the same way = a motion match


def _cos(a, b):
    na, nb = np.hypot(*a), np.hypot(*b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a[0] * b[0] + a[1] * b[1]) / (na * nb))


def _win_mean(v, start, step, n):
    """Mean of up to n VALID (non-NaN) vectors from `start`, walking by `step` (+1 fwd / -1 back)."""
    acc, cnt, i = np.zeros(2), 0, start
    while 0 <= i < len(v) and cnt < n:
        if not np.isnan(v[i, 0]):
            acc += v[i]
            cnt += 1
        i += step
    return acc / cnt if cnt else np.zeros(2)


def _classify(out_v, in_v, w, move_thr, calm_thr, trans):
    mout, min_ = np.hypot(*out_v) / w, np.hypot(*in_v) / w
    if trans == "whip":
        return "whip"
    hi = lambda m: m > move_thr
    lo = lambda m: m < calm_thr
    if hi(mout) and hi(min_):
        return "match" if _cos(out_v, in_v) > ALIGN else "action"
    if lo(mout) and lo(min_):
        return "still"
    return "hard"


def _thresholds(mag):
    """Reference-relative move/calm cutoffs from its own motion spread (p20 calm .. p80 active)."""
    valid = mag[~np.isnan(mag)]
    if len(valid) == 0:
        return 0.0, 0.0
    lo_ref, hi_ref = np.percentile(valid, 20), np.percentile(valid, 80)
    if hi_ref < STILL_FLOOR:                   # the whole reference is basically locked off
        return 1e9, 1e9                        # -> nothing is "moving"; every cut is "still"
    return lo_ref + MOVE_FRAC * (hi_ref - lo_ref), lo_ref + CALM_FRAC * (hi_ref - lo_ref)


def reference_cut_styles(cache, shots):
    """Style of every cut in the reference. Returns a list, one per boundary (len shots-1)."""
    if len(shots) < 2:
        return []
    v, fps, (w, h) = mvmotion.step_vectors_from_cache(cache)
    move_thr, calm_thr = _thresholds(np.hypot(v[:, 0], v[:, 1]) / w)

    styles = []
    for b in range(len(shots) - 1):
        f = int(round(shots[b + 1]["start"] * fps))
        out_v = _win_mean(v, f - 1, -1, WIN)      # frames before the boundary (skip the cut frame)
        in_v = _win_mean(v, f + 1, +1, WIN)       # frames after
        styles.append(_classify(out_v, in_v, w, move_thr, calm_thr,
                                shots[b + 1].get("transition", "hard")))
    return styles


# --- builder side ---------------------------------------------------------------------------
WIN_S = 0.12     # seconds each side of a boundary to characterise a clip's motion there
K = 7            # candidate in/out offsets to try per slot
TIE = 0.05       # gentle pull toward match()'s energy-chosen offset; only breaks style ties


def _endpoint_vec(v, fps, t0, t1):
    """Mean camera-step vector over the clip window [t0, t1] seconds (skips NaN cut frames)."""
    lo, hi = int(round(t0 * fps)), max(int(round(t0 * fps)) + 1, int(round(t1 * fps)))
    seg = v[max(0, lo):min(len(v), hi)]
    seg = seg[~np.isnan(seg[:, 0])] if len(seg) else seg
    return seg.mean(0) if len(seg) else np.zeros(2)


def _pair_cost(out_a, na, in_b, nb, style):
    """Cost (minimised) of putting slot A's out-frame next to slot B's in-frame under `style`.

    na/nb are the two endpoints' motion magnitudes, already normalised to [0,1] within their own
    slot — so "fast" means fast FOR THAT CLIP and no clip's absolute speed dominates the chain.
    """
    if style in ("action", "whip"):
        return -(na + nb)                     # both endpoints as fast as the clips allow
    if style == "still":
        return na + nb                        # both as calm as the clips allow
    if style == "match":
        return -(0.5 * (na + nb) + max(0.0, _cos(out_a, in_b)) * min(na, nb))
    return 0.0                                # hard / unknown: no preference


def refine_cut_points(plan, shots):
    """Slide each slot's src_start so consecutive clips realise the reference's cut style.

    Mutates and returns `plan`. The segment LENGTH is never changed (rhythm is sacred); only
    where in the clip it is taken from. A shortest-path DP over per-slot offset choices, because
    each boundary couples one slot's out-frame with the next slot's in-frame.
    """
    if len(plan) < 2:
        return plan

    # Each cut's style was stored on the shot it leads into (annotate_shots, at analyze time).
    style_of = {sh["index"]: sh.get("cut_in") for sh in shots}

    motion = {}                               # clip path -> (vectors, fps, duration_s)
    faces = {}                                # clip path -> flattering-angle curve
    for path in {p["clip"] for p in plan}:
        v, fps, _ = mvmotion.step_series(path)
        motion[path] = (v, fps, _clip_duration(path))   # real duration, not len(v)/fps (overshoots)
        faces[path] = faceangle.clip_curve(path)

    # Per slot: candidate offsets, each candidate's in/out endpoint vector + normalised speed, and
    # a small "stay put" penalty toward the offset match() already chose (by energy). That penalty
    # is tiny — it only decides boundaries where the style has NO preference (hard/unknown), so a
    # neutral cut keeps its energy-matched position instead of collapsing to the clip's start.
    cand = []
    for p in plan:
        v, fps, dur = motion[p["clip"]]
        L = p["duration"]
        if p.get("grouped"):
            # part of a spliced same-source group: its src_start is the splice layout, leave it
            span, offs = 0.0, [float(p["src_start"])]
        else:
            span = max(0.0, dur - L)
            offs = list(np.linspace(0, span, K)) if span > 1e-3 else [0.0]
        ins = [_endpoint_vec(v, fps, o, o + WIN_S) for o in offs]
        outs = [_endpoint_vec(v, fps, o + L - WIN_S, o + L) for o in offs]
        mags = [np.hypot(*x) for x in ins + outs]
        peak = max(mags) if mags else 0.0
        norm = (lambda m: m / peak) if peak > 1e-9 else (lambda m: 0.0)
        orig = float(p["src_start"])
        stay = [TIE * abs(o - orig) / span if span > 1e-3 else 0.0 for o in offs]
        # Flattering-angle preference: reward each candidate segment for showing a front-facing /
        # top-down face (no double chin). A clip with no face (b-roll) scores None -> zero pull, so
        # it just falls through to the motion match. Reward = lower cost, hence the minus.
        fcurve = faces[p["clip"]]
        face = [-FACE_W * (faceangle.window_score(fcurve, o, L) or 0.0) for o in offs]
        cand.append({
            "offs": offs, "ins": ins, "outs": outs, "stay": stay, "face": face,
            "nin": [norm(np.hypot(*x)) for x in ins],
            "nout": [norm(np.hypot(*x)) for x in outs],
        })

    # DP: dp[a] = best total cost of choosing candidate a for the current slot. Each node carries
    # its own unary costs (stay + face); each edge carries the boundary cut-motion match.
    dp = [s + f for s, f in zip(cand[0]["stay"], cand[0]["face"])]
    back = []
    for i in range(1, len(plan)):
        style = style_of.get(plan[i]["shot"])
        ci, cprev = cand[i], cand[i - 1]
        new_dp, bk = [], []
        for b in range(len(ci["offs"])):
            best, best_j = float("inf"), 0
            for a in range(len(cprev["offs"])):
                c = dp[a] + _pair_cost(cprev["outs"][a], cprev["nout"][a],
                                       ci["ins"][b], ci["nin"][b], style)
                if c < best:
                    best, best_j = c, a
            new_dp.append(best + ci["stay"][b] + ci["face"][b])
            bk.append(best_j)
        dp = new_dp
        back.append(bk)

    # Backtrack the cheapest chain and write the chosen offsets back into the plan.
    j = int(np.argmin(dp))
    chosen = [0] * len(plan)
    chosen[-1] = j
    for i in range(len(plan) - 1, 0, -1):
        chosen[i - 1] = back[i - 1][chosen[i]]
    for i, p in enumerate(plan):
        o = round(float(cand[i]["offs"][chosen[i]]), 3)
        p["src_start"] = o
        _, _, dur = motion[p["clip"]]
        p["max_len"] = round(dur - o, 3)
    return plan


def annotate_shots(cache, shots):
    """Attach each cut's style to the shot it leads into (shot['cut_in']); first shot is None."""
    styles = reference_cut_styles(cache, shots)
    if shots:
        shots[0]["cut_in"] = None
    for sh, st in zip(shots[1:], styles):
        sh["cut_in"] = st
    return styles

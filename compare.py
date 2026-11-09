"""Score how closely a finished edit reproduces the reference's edit mechanics.

The flow is human-in-the-loop by design: EditForge's output is an After Effects project, so the
finished video only exists once Julian renders it in AE. He drops that render back in, we run it
through the SAME analyzer as the reference, and this module compares the two edit-DNA dicts.

What "match" can and cannot mean, stated honestly so the scorecard never lies:
  - It CANNOT mean the pictures match. His clips are different footage; the content will never be
    the reference's content, and he never wanted it to be.
  - It CAN mean the edit MECHANICS match: cut rhythm, transition types, where cuts land on motion,
    effect levels. Those are measurable, and each gets a score below.
  - Some dimensions are matched to zero ON PURPOSE (camera motion, colour, slow-mo — his calls).
    Those are listed under 'not_matched' so a low number there is never read as a failure.

Everything keys off aligning the edit's cuts to the reference's cuts by TIME (seconds, so it is
independent of frame rate). The edit is built to the reference's exact shot durations, so the
boundaries should line up tightly; where one is missing or extra, that itself is the rhythm error.
"""
import numpy as np

TOL = 0.10   # seconds; two cut boundaries within this are the same cut (~3 frames at 30fps)


def _align(ref_shots, edit_shots, tol=TOL):
    """Pair reference shots with edit shots by start time. Two-pointer greedy over sorted starts.

    Returns [(ref_shot|None, edit_shot|None)]: both set = a cut that lines up; ref-only = a cut
    the edit missed; edit-only = a cut the edit added that the reference doesn't have.
    """
    pairs, i, j = [], 0, 0
    while i < len(ref_shots) and j < len(edit_shots):
        dr, de = ref_shots[i]["start"], edit_shots[j]["start"]
        if abs(dr - de) <= tol:
            pairs.append((ref_shots[i], edit_shots[j])); i += 1; j += 1
        elif dr < de:
            pairs.append((ref_shots[i], None)); i += 1
        else:
            pairs.append((None, edit_shots[j])); j += 1
    pairs += [(r, None) for r in ref_shots[i:]]
    pairs += [(None, e) for e in edit_shots[j:]]
    return pairs


def _f1(tp, fp, fn):
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return round(2 * prec * rec / (prec + rec), 3)


def _rhythm(pairs):
    """Do the cuts land in the same places? F1 over cut boundaries + timing drift on the matches."""
    # boundaries only — the first shot starts at 0 in both and isn't a cut
    cut_pairs = [(r, e) for r, e in pairs if not (r and r["start"] == 0 and e and e["start"] == 0)]
    matched = [(r, e) for r, e in cut_pairs if r and e]
    missing = sum(1 for r, e in cut_pairs if r and not e)
    extra = sum(1 for r, e in cut_pairs if e and not r)
    drifts = [abs(r["start"] - e["start"]) for r, e in matched if r["start"] > 0]
    return {
        "score": _f1(len(matched), extra, missing),
        "ref_cuts": sum(1 for r, e in cut_pairs if r),
        "edit_cuts": sum(1 for r, e in cut_pairs if e),
        "aligned": len(matched),
        "missing_in_edit": missing,
        "extra_in_edit": extra,
        "mean_drift_ms": round(float(np.mean(drifts)) * 1000, 1) if drifts else 0.0,
    }


def _per_cut_field(pairs, field):
    """Match rate of a per-cut field (transition type / cut style) over aligned, non-first cuts."""
    both = [(r, e) for r, e in pairs if r and e and r["start"] > 0]
    if not both:
        return {"score": 1.0, "matched": 0, "of": 0, "mismatches": []}
    hits, misses = 0, []
    for r, e in both:
        if r.get(field) == e.get(field):
            hits += 1
        else:
            misses.append({"at_s": round(r["start"], 2),
                           "ref": r.get(field), "edit": e.get(field)})
    return {"score": round(hits / len(both), 3), "matched": hits, "of": len(both),
            "mismatches": misses[:12]}


def _warp(pairs):
    """Did ripple/warp show up on the shots it should, at a similar strength?"""
    both = [(r, e) for r, e in pairs if r and e]
    if not both:
        return {"score": 1.0, "note": "no aligned shots"}
    agree, ref_amp, edit_amp = 0, [], []
    for r, e in both:
        rw, ew = r.get("warp"), e.get("warp")
        if bool(rw) == bool(ew):
            agree += 1
        if rw and ew:
            ref_amp.append(rw.get("amplitude", 0.0))
            edit_amp.append(ew.get("amplitude", 0.0))
    out = {"score": round(agree / len(both), 3),
           "presence_agree": agree, "of": len(both),
           "ref_warp_shots": sum(1 for r, e in both if r.get("warp")),
           "edit_warp_shots": sum(1 for r, e in both if e.get("warp"))}
    if ref_amp:
        out["ref_mean_amplitude"] = round(float(np.mean(ref_amp)), 4)
        out["edit_mean_amplitude"] = round(float(np.mean(edit_amp)), 4)
    return out


def _level(ref_val, edit_val):
    """Closeness of two scalar effect levels, 1.0 = identical, decaying with relative error."""
    hi = max(abs(ref_val), abs(edit_val), 1e-6)
    return round(1.0 - min(1.0, abs(ref_val - edit_val) / hi), 3)


def _effects(ref, edit, pairs):
    rl, el = ref.get("look", {}), edit.get("look", {})
    rv, ev = rl.get("vignette", 0.0) or 0.0, el.get("vignette", 0.0) or 0.0
    rlb = (rl.get("letterbox") or {}).get("top", 0.0)
    elb = (el.get("letterbox") or {}).get("top", 0.0)
    return {
        "warp": _warp(pairs),
        "vignette": {"ref": round(rv, 4), "edit": round(ev, 4), "score": _level(rv, ev)},
        "letterbox": {"ref": round(rlb, 4), "edit": round(elb, 4), "score": _level(rlb, elb)},
        "rgb_split": {"ref": len(rl.get("rgb_split", {})), "edit": len(el.get("rgb_split", {}))},
    }


# Dimensions turned off on purpose — so a low similarity here is by design, not a failure.
NOT_MATCHED = [
    ("Content", "Your clips are different footage — never meant to match theirs."),
    ("Camera motion", "Your clip is a locked, static layer by your choice; the reference's camera move isn't replayed."),
    ("Colour grade", "Off — the LAB match looked bad on real footage."),
    ("Slow-mo", "Can't be recovered from a finished video, so never invented."),
    ("Subject motion", "Measured for understanding only, not applied to your footage."),
]


def compare_edits(ref, edit):
    """Full scorecard comparing an analyzed edit to the analyzed reference."""
    pairs = _align(ref["shots"], edit["shots"])
    rhythm = _rhythm(pairs)
    transitions = _per_cut_field(pairs, "transition")
    cut_points = _per_cut_field(pairs, "cut_in")
    effects = _effects(ref, edit, pairs)

    # Overall = the mechanics we actually try to match. Rhythm is the backbone, so it leads.
    core = [("rhythm", rhythm["score"], 0.4),
            ("transitions", transitions["score"], 0.25),
            ("cut_points", cut_points["score"], 0.25),
            ("warp", effects["warp"]["score"], 0.1)]
    overall = round(sum(s * w for _, s, w in core) / sum(w for _, _, w in core), 3)

    return {
        "overall": overall,
        "rhythm": rhythm,
        "transitions": transitions,
        "cut_points": cut_points,
        "effects": effects,
        "not_matched": [{"dimension": d, "why": w} for d, w in NOT_MATCHED],
    }

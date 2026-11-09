"""Turn a render-vs-reference scorecard into concrete build corrections.

This is the brain of the auto-correction loop. compare.py MEASURES how far a render is from the
reference, dimension by dimension, and lists the specific cuts that don't match. This module reads
that and decides what to CHANGE in the next build so the render moves toward the reference:

  - a flash the reference has but the render lost -> make that flash stronger (longer) so it
    registers, or turn it on if it was off
  - a flash the render has but the reference doesn't -> turn it off
  - a transition (dissolve/fade/whip) that came out as a plain cut -> lengthen it so it reads
  - an effect level (vignette, letterbox, ...) measured off from the reference -> gain it toward
    the reference's measured value

It ONLY touches what the build actually controls. It does not try to "fix" content, colour, or
anything that differs because the footage is Julian's, not the reference's — those aren't errors.
Corrections are cumulative: each pass nudges further, so a flash that's still missing next round
gets boosted again.
"""

FLASH_STEP = 1.6      # each correction pass lengthens a missing flash by this factor
TRANS_STEP = 1.5      # ...and a transition that came out as a hard cut
LEVEL_LO, LEVEL_HI = 0.4, 2.5   # clamp effect-level gains to something sane


def _shot_at(shots, t):
    """Index of the reference shot whose start is nearest time t (mismatches are keyed by time)."""
    return min(shots, key=lambda s: abs(s["start"] - t))["index"]


def derive_corrections(reference, scorecard, prev=None):
    """Build overrides that push the next build toward the reference. `prev` is last pass's
    corrections so fixes accumulate (a still-missing flash boosts further, not resets).

    Returns {"flash": {idx: "boost"|"off"}, "flash_gain": {idx: factor},
             "transition": {idx: factor}, "levels": {effect: gain}, "summary": [...]}.
    """
    prev = prev or {}
    shots = reference["shots"]
    corr = {"flash": {}, "flash_gain": dict(prev.get("flash_gain", {})),
            "transition": {}, "trans_gain": dict(prev.get("trans_gain", {})),
            "levels": {}, "summary": []}

    for m in scorecard.get("transitions", {}).get("mismatches", []):
        idx = _shot_at(shots, m["at_s"])
        ref, edit = m.get("ref"), m.get("edit")
        if ref == "flash" and edit != "flash":
            corr["flash"][idx] = "boost"
            corr["flash_gain"][idx] = round(corr["flash_gain"].get(idx, 1.0) * FLASH_STEP, 3)
            corr["summary"].append(f"shot {idx}: flash didn't register — stronger flash "
                                   f"(x{corr['flash_gain'][idx]})")
        elif edit == "flash" and ref != "flash":
            corr["flash"][idx] = "off"
            corr["summary"].append(f"shot {idx}: unwanted flash — removed")
        elif ref in ("dissolve", "fade", "whip"):
            corr["transition"][idx] = ref
            corr["trans_gain"][idx] = round(corr["trans_gain"].get(idx, 1.0) * TRANS_STEP, 3)
            corr["summary"].append(f"shot {idx}: came out '{edit}', want '{ref}' — "
                                   f"longer transition (x{corr['trans_gain'][idx]})")

    for eff in ("vignette", "letterbox"):
        e = scorecard.get("effects", {}).get(eff, {})
        rr, ee, sc = e.get("ref", 0.0), e.get("edit", 0.0), e.get("score", 1.0)
        if rr and sc < 0.8:
            gain = rr / ee if ee > 1e-3 else LEVEL_HI
            gain = round(min(max(gain, LEVEL_LO), LEVEL_HI), 3)
            corr["levels"][eff] = gain
            corr["summary"].append(f"{eff}: {ee} vs target {rr} — gain x{gain}")

    return corr

"""Place a user clip in the reference slot it RESEMBLES — used the same way as the reference's clip.

Julian: "if there is a clip that is really similar it should be used in the same manner as the
reference clip." This is a TARGETED content match, not a blanket one: only a STRONG visual
resemblance locks a clip to a slot. Everything else still fills by motion energy (the default),
because in general his footage doesn't look like the reference's and forcing a match would look
worse. So this only fires when there's a genuine look-alike — the case he actually wants.

Reuses the visual fingerprint from sourcegroup: an AVERAGE of several downscaled frames (which
blurs whatever moves and leaves the scene/composition) plus a colour histogram. A reference SHOT is
a time range in the reference video; one of his CLIPS is a whole file — same fingerprint over each.
"""
import cv2

import sourcegroup

# A clip must clearly resemble the reference shot (composition + palette) to lock into its slot.
# Different people's footage of the same trend won't be pixel-identical, so these sit below the
# same-SOURCE thresholds, but high enough not to lock on a vague resemblance. Tunable on real
# footage — raise if it over-matches, lower if genuine look-alikes slip through.
MATCH_SIG_T = 0.72     # averaged-thumbnail correlation (scene / composition)
MATCH_HIST_T = 0.60    # colour-histogram correlation (palette / lighting)


def _clip_signature(path):
    """Visual fingerprint of a whole user clip, or None if unreadable."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if not total:
        cap.release()
        return None
    sig = sourcegroup._shot_signature(cap, 0.0, total / fps, fps)
    cap.release()
    return sig


def match_clips_to_shots(reference_video, shots, clip_paths):
    """{shot_index: clip_path} for every reference shot a user clip clearly resembles.

    A shot with no strong look-alike is simply omitted (it falls through to energy matching). One
    clip may lock into several shots if it resembles all of them; the best-scoring clip wins a slot.
    """
    cap = cv2.VideoCapture(str(reference_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ref_sigs = {sh["index"]: sourcegroup._shot_signature(cap, sh["start"], sh["end"], fps)
                for sh in shots}
    cap.release()

    clip_sigs = {str(p): _clip_signature(p) for p in clip_paths}
    locks = {}
    for sh in shots:
        rs = ref_sigs.get(sh["index"])
        if rs is None:
            continue
        best_score, best_clip = 0.0, None
        for path, cs in clip_sigs.items():
            if cs is None:
                continue
            sig_corr, hist_corr = sourcegroup._similarity(rs, cs)
            if sig_corr >= MATCH_SIG_T and hist_corr >= MATCH_HIST_T:
                score = min(sig_corr, hist_corr)
                if score > best_score:
                    best_score, best_clip = score, path
        if best_clip:
            locks[sh["index"]] = best_clip
    return locks

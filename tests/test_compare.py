"""Ground truth for the edit-vs-reference scorecard (compare.py).

Pure dict math — no video. Build a reference edit-DNA, then perturb copies in ONE known way each
(change a transition, shift a cut, drop a cut, remove a warp, change a cut style) and assert the
scorecard moves in exactly that dimension and nowhere else.

Run from ~/editforge:  ./venv/bin/python tests/test_compare.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare

TRANS = ["hard", "dissolve", "whip", "flash", "hard"]
STYLE = ["match", "action", "still", "hard", "match"]


def make_dna(n=5, dur=0.5):
    shots = []
    for i in range(n):
        shots.append({
            "index": i,
            "start": round(i * dur, 3),
            "end": round((i + 1) * dur, 3),
            "transition": "hard" if i == 0 else TRANS[i % 5],
            "cut_in": None if i == 0 else STYLE[i % 5],
            "warp": {"amplitude": 0.02} if i == 2 else None,
        })
    return {"shots": shots,
            "look": {"vignette": 0.4, "letterbox": {"top": 0.08, "bottom": 0.08},
                     "rgb_split": {"3": [5, 0]}}}


def main():
    fails = []
    ref = make_dna()

    # 1. Identical edit -> everything perfect.
    sc = compare.compare_edits(ref, copy.deepcopy(ref))
    print(f"1 identical         overall={sc['overall']}  rhythm={sc['rhythm']['score']} "
          f"trans={sc['transitions']['score']} cuts={sc['cut_points']['score']} "
          f"warp={sc['effects']['warp']['score']}")
    if sc["overall"] < 0.999:
        fails.append(f"1: identical edit scored {sc['overall']}, expected 1.0")
    for dim in ("rhythm", "transitions", "cut_points"):
        if sc[dim]["score"] != 1.0:
            fails.append(f"1: {dim} {sc[dim]['score']} != 1.0 on identical")

    # 2. One transition changed -> transitions 3/4, rhythm & cut_points untouched.
    e = copy.deepcopy(ref); e["shots"][1]["transition"] = "flash"
    sc = compare.compare_edits(ref, e)
    print(f"2 one transition    trans={sc['transitions']['score']} (mismatches "
          f"{len(sc['transitions']['mismatches'])})  rhythm={sc['rhythm']['score']}")
    if sc["transitions"]["score"] != 0.75:
        fails.append(f"2: transitions {sc['transitions']['score']} != 0.75")
    if sc["rhythm"]["score"] != 1.0 or sc["cut_points"]["score"] != 1.0:
        fails.append("2: a transition change bled into rhythm/cut_points")

    # 3. One cut shifted past tolerance -> one missing + one extra, rhythm drops.
    e = copy.deepcopy(ref); e["shots"][2]["start"] = 1.3
    sc = compare.compare_edits(ref, e)
    print(f"3 cut shifted       rhythm={sc['rhythm']['score']} missing={sc['rhythm']['missing_in_edit']} "
          f"extra={sc['rhythm']['extra_in_edit']}")
    if sc["rhythm"]["score"] >= 1.0:
        fails.append("3: shifted cut didn't lower rhythm")
    if sc["rhythm"]["missing_in_edit"] != 1 or sc["rhythm"]["extra_in_edit"] != 1:
        fails.append(f"3: expected 1 missing + 1 extra, got {sc['rhythm']['missing_in_edit']}/"
                     f"{sc['rhythm']['extra_in_edit']}")

    # 4. Edit dropped a cut entirely -> one missing, fewer edit cuts.
    e = copy.deepcopy(ref); del e["shots"][3]
    sc = compare.compare_edits(ref, e)
    print(f"4 dropped a cut     rhythm={sc['rhythm']['score']} missing={sc['rhythm']['missing_in_edit']} "
          f"ref_cuts={sc['rhythm']['ref_cuts']} edit_cuts={sc['rhythm']['edit_cuts']}")
    if sc["rhythm"]["missing_in_edit"] != 1:
        fails.append(f"4: expected 1 missing cut, got {sc['rhythm']['missing_in_edit']}")

    # 5. Warp removed from the shot that had it -> warp presence disagreement.
    e = copy.deepcopy(ref); e["shots"][2]["warp"] = None
    sc = compare.compare_edits(ref, e)
    print(f"5 warp removed      warp={sc['effects']['warp']['score']} "
          f"ref_warp={sc['effects']['warp']['ref_warp_shots']} "
          f"edit_warp={sc['effects']['warp']['edit_warp_shots']}")
    if sc["effects"]["warp"]["score"] >= 1.0:
        fails.append("5: removing a warp didn't lower the warp score")

    # 6. One cut style changed -> cut_points 3/4, transitions untouched.
    e = copy.deepcopy(ref); e["shots"][1]["cut_in"] = "still"
    sc = compare.compare_edits(ref, e)
    print(f"6 cut style changed cuts={sc['cut_points']['score']} trans={sc['transitions']['score']}")
    if sc["cut_points"]["score"] != 0.75:
        fails.append(f"6: cut_points {sc['cut_points']['score']} != 0.75")
    if sc["transitions"]["score"] != 1.0:
        fails.append("6: a cut-style change bled into transitions")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — the scorecard moves in exactly the perturbed dimension, and nowhere else.")


if __name__ == "__main__":
    main()

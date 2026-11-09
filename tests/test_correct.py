"""Ground truth for the correction engine (correct.py). Pure dict math, no video.

Feed a scorecard with a KNOWN mismatch and confirm the derived correction is the right fix, that
it only touches what the build controls, and that repeated passes accumulate (a still-missing flash
boosts further rather than resetting).

Run from ~/editforge:  ./venv/bin/python tests/test_correct.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import correct

REF = {"shots": [{"index": i, "start": round(i * 0.5, 3)} for i in range(8)]}


def card(trans_mismatches=(), effects=None):
    return {"transitions": {"mismatches": list(trans_mismatches)},
            "effects": effects or {}}


def main():
    fails = []

    # 1. reference flash the render lost -> boost that shot's flash
    sc = card([{"at_s": 1.5, "ref": "flash", "edit": "hard"}])   # 1.5s == shot 3
    c = correct.derive_corrections(REF, sc)
    print("1 missing flash   ->", c["flash"], c["flash_gain"])
    if c["flash"].get(3) != "boost" or c["flash_gain"].get(3) != 1.6:
        fails.append(f"1: expected shot 3 flash boost x1.6, got {c['flash']} {c['flash_gain']}")

    # 2. render has a flash the reference doesn't -> turn it off
    c = correct.derive_corrections(REF, card([{"at_s": 2.0, "ref": "hard", "edit": "flash"}]))
    print("2 extra flash     ->", c["flash"])
    if c["flash"].get(4) != "off":
        fails.append(f"2: expected shot 4 flash off, got {c['flash']}")

    # 3. dissolve came out as a hard cut -> lengthen the dissolve
    c = correct.derive_corrections(REF, card([{"at_s": 1.0, "ref": "dissolve", "edit": "hard"}]))
    print("3 lost dissolve   ->", c["transition"], c["trans_gain"])
    if c["transition"].get(2) != "dissolve" or c["trans_gain"].get(2) != 1.5:
        fails.append(f"3: expected shot 2 dissolve x1.5, got {c['transition']} {c['trans_gain']}")

    # 4. cumulative: a flash still missing next pass boosts FURTHER, not resets
    sc = card([{"at_s": 1.5, "ref": "flash", "edit": "hard"}])
    c1 = correct.derive_corrections(REF, sc)
    c2 = correct.derive_corrections(REF, sc, prev=c1)
    print("4 cumulative      -> pass1 x%.2f  pass2 x%.2f" % (c1["flash_gain"][3], c2["flash_gain"][3]))
    if not (c2["flash_gain"][3] > c1["flash_gain"][3]):
        fails.append(f"4: second pass should boost further, got {c1['flash_gain'][3]} -> {c2['flash_gain'][3]}")

    # 5. an effect level measured off from the reference -> gain toward it (clamped)
    c = correct.derive_corrections(REF, card(effects={"vignette": {"ref": 0.4, "edit": 0.1, "score": 0.3}}))
    print("5 level gain      ->", c["levels"])
    if not (correct.LEVEL_LO <= c["levels"].get("vignette", 0) <= correct.LEVEL_HI):
        fails.append(f"5: vignette gain out of range: {c['levels']}")

    # 6. nothing off -> no corrections (don't invent work)
    c = correct.derive_corrections(REF, card())
    print("6 clean render    ->", c["flash"], c["transition"], c["levels"])
    if c["flash"] or c["transition"] or c["levels"]:
        fails.append(f"6: clean render should yield no corrections, got {c}")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS — corrections target the right fix, only touch build-controlled things, and accumulate.")


if __name__ == "__main__":
    main()

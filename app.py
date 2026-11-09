"""EditForge — drop a reference edit, drop your clips, get the edit rebuilt with your footage."""
import json
import os
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import builder
import compare
import contentmatch
import correct
import cutpoint
import effects
import fetch
import look
from analyzer import analyze

# Where uploads/outputs live. /build deletes the previous batch of clips, so ANY second
# instance sharing this directory destroys the first one's data — which is exactly what
# happened when tests were run against the live app. Set EDITFORGE_DATA to give a test run
# its own directory and keep it away from real footage.
ROOT = Path(os.environ.get("EDITFORGE_DATA", Path(__file__).parent))
UPLOADS = ROOT / "uploads"
OUTPUT = ROOT / "output"
WORK = ROOT / "work"
for d in (UPLOADS, OUTPUT, WORK):
    d.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB


def _load_corrections():
    """Auto-corrections from the last render score, or None. JSON stringifies the int shot-keys,
    so restore them for the build's per-shot lookups."""
    p = OUTPUT / "corrections.json"
    if not p.exists():
        return None
    c = json.loads(p.read_text())
    for k in ("flash", "flash_gain", "transition", "trans_gain"):
        if k in c:
            c[k] = {int(i): v for i, v in c[k].items()}
    return c


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_reference():
    """Accepts either an uploaded file ('reference') or a link ('url')."""
    ref = UPLOADS / "reference.mp4"
    source = None

    url = (request.form.get("url") or "").strip()
    f = request.files.get("reference")

    if url:
        if not fetch.looks_supported(url):
            return jsonify({"error": "Give me a TikTok, Instagram, or YouTube link."}), 400
        try:
            meta = fetch.fetch(url, ref)
        except fetch.FetchError as e:
            return jsonify({"error": str(e)}), 400
        source = f"{meta['uploader']} — {meta['title']}".strip(" —") or url
    elif f:
        f.save(ref)
        source = f.filename
    else:
        return jsonify({"error": "no reference video"}), 400

    result = analyze(ref)
    shots = result["shots"]
    result["look"] = effects.probe_reference(ref, shots)
    result["look"].update(look.probe_look(ref, shots))
    (OUTPUT / "reference.json").write_text(json.dumps(result))

    # cuts that land on a detected beat — where a zoom punch belongs
    beats = result["beats"]
    beat_shots = [
        s["index"] for s in shots
        if beats and min(abs(b - s["start"]) for b in beats) < 0.08
    ]
    result["beat_shots"] = beat_shots
    (OUTPUT / "reference.json").write_text(json.dumps(result))
    # a new reference starts a fresh correction loop — drop any overrides from the last one
    (OUTPUT / "corrections.json").unlink(missing_ok=True)

    return jsonify({
        "source": source,
        "flash_cuts": len(result["look"]["flash_shots"]),
        "camera_moves": result["moving_shots"],
        "subject_shots": result["subject_shots"],
        "false_cuts_dropped": result["camera_moves_not_cuts"],
        "speed_ramps": result["ramped_shots"],
        "warped_shots": result["warped_shots"],
        "caption_count": len(result["captions"]),
        "transitions": result["transitions"],
        "cut_styles": result["cut_styles"],
        "source_groups": result["source_groups"],
        "freeze_shots": result["freeze_shots"],
        "letterbox": result["look"]["letterbox"],
        "vignette": result["look"]["vignette"],
        "rgb_split_shots": len(result["look"]["rgb_split"]),
        "beat_cuts": len(beat_shots),
        "bpm": result["bpm"],
        "duration": result["total_duration"],
        "shot_count": result["shot_count"],
        "median_shot": result["median_shot_duration"],
        "subdivision": result["grid"]["dominant_subdivision"] if result["grid"] else None,
        # Distinct sources, not raw shots: a spliced run of N shots reuses ONE clip, so it counts
        # once. "ideal" = one clip per distinct source (nothing repeats); "few" = a soft floor you
        # can go below, the build just reuses clips to fill every shot. Neither is enforced.
        "clips_ideal": result["shot_count"] - result["spliced_shots"] + len(result["source_groups"]),
        "clips_few": max(3, round(
            (result["shot_count"] - result["spliced_shots"] + len(result["source_groups"])) / 3)),
        "energy_curve": [s["energy"] for s in shots],
    })


@app.route("/build", methods=["POST"])
def build():
    ref_path = OUTPUT / "reference.json"
    if not ref_path.exists():
        return jsonify({"error": "analyze a reference video first"}), 400
    reference = json.loads(ref_path.read_text())

    files = request.files.getlist("clips")
    clip_dir = UPLOADS / "clips"
    if files:
        # a fresh upload: clear the previous batch (only once we know this one is valid) and save
        if clip_dir.exists():
            shutil.rmtree(clip_dir)
        clip_dir.mkdir()
        saved = []
        for f in files:
            p = clip_dir / Path(f.filename).name
            f.save(p)
            saved.append(p)
    else:
        # no upload = a REBUILD (e.g. applying corrections): reuse the clips already on disk
        saved = sorted(p for p in clip_dir.glob("*") if p.is_file()) if clip_dir.exists() else []
        if not saved:
            return jsonify({"error": "no clips"}), 400

    moments = []
    for p in saved:
        moments.extend(builder.find_moments(p))

    # Look-alike matching: if one of his clips genuinely RESEMBLES the clip the reference used in a
    # slot, lock it into that slot (used the same way). Only strong matches lock; the rest fill by
    # energy. This is the targeted content-match he asked for.
    locks = contentmatch.match_clips_to_shots(UPLOADS / "reference.mp4", reference["shots"], saved)
    plan = builder.match(reference["shots"], moments, locks=locks)

    # Where the reference spliced ONE source video across several consecutive shots, use ONE of
    # his clips across those slots too (sequential segments of it) instead of spreading different
    # clips. This overrides match()'s clip choice for those slots and marks them so the cut-point
    # refiner leaves the splice layout alone.
    builder.apply_source_groups(plan, reference.get("source_groups", []))

    # match() chose WHICH clip fills each slot (by energy); now choose WHERE in each clip to cut.
    # Two pulls, blended: (1) land the cut on the motion the reference cut on (action/still/match),
    # and (2) show the most flattering face angle — front-facing / top-down, no double-chin low
    # angle (Julian's ask). Only slides each segment within its clip (length fixed) so the cut
    # rhythm is untouched. Always runs now: even with no cut styles, the face preference applies.
    cutpoint.refine_cut_points(plan, reference["shots"])

    # Julian's signature: make the edit loop seamlessly. Runs LAST so the first shot's in-point is
    # already final — the last shot becomes the run-up to it, ending exactly where it starts.
    builder.make_seamless_loop(plan)

    # No toggles: everything we can honestly recover from the reference is always applied.
    # Slow-mo is deliberately absent — it cannot be detected from a finished video (0.5x
    # footage and 60fps capture are pixel-identical), and inventing it would corrupt the
    # very shot timing we're copying.
    # Colour grade is off: the Reinhard LAB match looked bad on real footage. Clips still
    # get transcoded to ProRes for AE, just ungraded.
    motion_tracks = {
        sh["index"]: {"motion": sh.get("motion", []), "overscale": sh.get("overscale", 1.0)}
        for sh in reference["shots"]
    }
    trans = {
        sh["index"]: {"type": sh.get("transition", "hard"),
                      "duration": sh.get("transition_duration", 0.0)}
        for sh in reference["shots"]
    }
    look_data = reference.get("look", {})
    builder.write_jsx(
        plan, UPLOADS / "reference.mp4", OUTPUT / "edit.jsx",
        # CLEAN MODE + slow-mo + flashes (Julian's calls, 2026-07-24). Footage stays untouched:
        # "native" places each clip at 100% scale, centred, no zoom/crop/camera-move/face-framing
        # (his clips are vertical, so 1:1 fills the frame) — this is the fix for the "weird zoom /
        # wrong size" error. Still OFF (the "glitchy effects"): warp, RGB split, vignette,
        # letterbox, captions, speed ramps.
        #
        # ON: transitions between clips; FLASH cuts (the reference's real luminance-spike cuts);
        # SLOW-MO (a preset — below-median-energy shots play at 0.5x); VELOCITY (per-shot speed
        # ramps → Time Remap). Slow-mo and velocity keep the SAME slot length, so the cut rhythm is
        # preserved. The cut-point picker (cutpoint.refine_cut_points, above) also stays — invisible,
        # only chooses WHERE in each clip to cut. Everything is one flag from the full stack.
        fx={"color": False, "motion": False, "warp": False,
            "transitions": True, "flash": True, "velocity": True,
            "rgb_split": False, "letterbox": False, "vignette": False,
            "captions": False, "face": False, "slowmo": True,
            "native": True, "enhance": True, "glow": False},
        ref_color=look_data.get("color"),
        flash_shots=look_data.get("flash_shots", []),
        motion_tracks=motion_tracks,
        trans=trans,
        rgb_split={int(k): v for k, v in look_data.get("rgb_split", {}).items()},
        letterbox=look_data.get("letterbox"),
        vignette=look_data.get("vignette", 0.0),
        speeds={sh["index"]: sh.get("speed", []) for sh in reference["shots"]},
        warps={sh["index"]: sh["warp"] for sh in reference["shots"] if sh.get("warp")},
        caption_slots=reference.get("captions", []),
        # auto-corrections from the last render-vs-reference score (if any), so each rebuild moves
        # closer to the original. Written by /compare; converted to int keys (JSON stringifies them).
        corrections=_load_corrections(),
        work_dir=WORK,
        use_audio=True,  # guide track: you can't cut to a beat you can't hear
    )
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2))

    look = reference.get("look", {})
    # Slots whose clip is shorter than the shot it fills — those used to run out of footage and
    # show black (the "gap between clips"). They now get slowed to fill; surface them so the user
    # knows to bring longer clips if the slow-down is too strong.
    short = [{"shot": p["shot"], "clip": p["name"],
              "have": round(p["max_len"], 2), "need": round(p["duration"], 2)}
             for p in plan if p["max_len"] < p["duration"] - 0.02]
    return jsonify({
        "jsx": "/output/edit.jsx",
        "jsx_path": str((OUTPUT / "edit.jsx").resolve()),
        "shots": len(plan),
        "clips_used": len({p["name"] for p in plan}),
        "clips_given": len(saved),
        "short_clips": short,
        "lookalikes": len(locks),   # shots filled by a clip that resembles the reference's clip
        "applied": {
            "flashes": len(look.get("flash_shots", [])),
            "camera_moves": sum(1 for sh in reference["shots"] if sh.get("motion")),
        },
        "plan": [
            {"shot": p["shot"], "name": p["name"], "start": p["src_start"],
             "duration": p["duration"], "target": p["target_energy"],
             "actual": p["actual_energy"]}
            for p in plan
        ],
    })


@app.route("/compare", methods=["POST"])
def compare_render():
    """Score a finished AE render against the reference. Julian renders edit.jsx in After Effects,
    then drops the rendered video here; we run it through the SAME analyzer as the reference and
    report, dimension by dimension, how close the edit MECHANICS came — plus what's off by design."""
    ref_path = OUTPUT / "reference.json"
    if not ref_path.exists():
        return jsonify({"error": "analyze a reference first — there's nothing to compare against"}), 400
    reference = json.loads(ref_path.read_text())

    f = request.files.get("edit")
    if not f:
        return jsonify({"error": "drop your rendered edit (export edit.jsx from After Effects first)"}), 400
    render_path = UPLOADS / "edit_render.mp4"
    f.save(render_path)

    # analyse the render exactly like a reference, so its DNA is directly comparable (look included)
    edit = analyze(render_path)
    edit["look"] = effects.probe_reference(render_path, edit["shots"])
    edit["look"].update(look.probe_look(render_path, edit["shots"]))

    scorecard = compare.compare_edits(reference, edit)
    scorecard["source"] = f.filename
    (OUTPUT / "scorecard.json").write_text(json.dumps(scorecard, indent=2))

    # Auto-correction: turn the mismatches into build overrides that the NEXT build applies, so the
    # render moves toward the reference. Accumulate onto the previous pass (a still-missing flash
    # boosts further). The user rebuilds edit.jsx, re-renders, drops it back — the loop, closed.
    corrections = correct.derive_corrections(reference, scorecard, prev=_load_corrections())
    (OUTPUT / "corrections.json").write_text(json.dumps(corrections, indent=2))
    scorecard["corrections"] = corrections
    return jsonify(scorecard)


@app.route("/output/<path:name>")
def output_file(name):
    return send_from_directory(OUTPUT, name)


if __name__ == "__main__":
    app.run(port=8781, debug=True)

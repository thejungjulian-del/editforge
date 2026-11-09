# EditForge

Paste a TikTok edit and drop in your own clips. EditForge reads the edit's cut rhythm, beat grid, energy and colour, then rebuilds it with your footage as an After Effects project. It finds which of your clips look most like each shot in the reference, moves every cut point to a good moment in your clip (including a front-facing or slightly top-down face angle), notices when the reference splices one source shot several times and does the same with one of your clips, and makes the end loop back into the start. The output is an After Effects script (`edit.jsx`) that builds the whole comp with your footage, transitions, flashes, speed ramps and slow motion.

## Run it

Needs Python 3.9+, ffmpeg on your PATH, and After Effects to open the result.

```
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py
```

Open http://127.0.0.1:8781, paste a reference link (TikTok, Instagram Reels or YouTube Shorts) and drop in your clips. In After Effects, run the generated `edit.jsx` from File > Scripts > Run Script File.

Two optional models go in a `models/` folder. Without them the app falls back to simpler methods:

- `face_detection_yunet_2023mar.onnx` (YuNet face detector, from the OpenCV model zoo) for face-aware cut points.
- `transnetv2_pytorch.py` and `transnetv2-pytorch-weights.pth` (the PyTorch version of TransNetV2, from github.com/soCzech/TransNetV2) for learned shot detection. This one also needs `torch`.

Tests run one file at a time, for example `./venv/bin/python tests/test_cutpoint.py`. `make_test_clips.sh` and `make_test_video.sh` generate synthetic clips and a synthetic reference with ffmpeg.

## Status

Works locally. Most of the testing used synthetic fixtures rather than real edits. Things to know:

- After Effects showed every clip as Media Offline because it can't reliably decode ffmpeg's H.264. The app now transcodes each clip to ProRes 422 before writing the script, which fixed it. That makes the build slower and the files bigger.
- Right now it runs in a minimal mode: cuts, transitions, flashes, speed ramps and slow motion. Colour, zoom, warp and caption effects are built but switched off in `app.py`.
- Detection is tuned to skip an effect it isn't sure about rather than guess, so it can miss things.
- Grouping shots by source looks at appearance only, so an edit filmed in one room with one person can get over-grouped.
- The Flask server runs in debug mode and each build clears the previous clips. Keep it on your own machine.

Next on the list: CapCut draft output and a chat front end, so you can describe an edit and get a finished video back.

Portfolio: https://thejungjulian-del.github.io/portfolio/

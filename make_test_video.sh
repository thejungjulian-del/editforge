#!/bin/bash
# Synthetic reference: 16 visually-distinct MOVING shots x 0.5s = 8s, 120 BPM click.
# Sources have real motion (like real footage) rather than static noise, which is
# what a content-based cut detector actually keys on.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/uploads"
D=/tmp/ef_test
rm -rf "$D" && mkdir -p "$D"

SRC=(
  "testsrc2=s=640x360:r=30"
  "mandelbrot=s=640x360:r=30"
  "life=s=640x360:r=30:mold=10"
  "cellauto=s=640x360:r=30"
  "rgbtestsrc=s=640x360:r=30"
  "smptebars=s=640x360:r=30"
  "testsrc=s=640x360:r=30"
  "gradients=s=640x360:r=30"
  "mandelbrot=s=640x360:r=30:start_scale=0.1"
  "life=s=640x360:r=30:ratio=0.3"
  "testsrc2=s=640x360:r=30:alpha=200"
  "cellauto=s=640x360:r=30:rule=110"
  "gradients=s=640x360:r=30:speed=0.05"
  "smptehdbars=s=640x360:r=30"
  "rgbtestsrc=s=640x360:r=30"
  "testsrc=s=640x360:r=30"
)
HUE=(0 40 80 120 160 200 240 280 320 20 60 100 140 180 220 260)

for i in $(seq 0 15); do
  ffmpeg -y -loglevel error -f lavfi -i "${SRC[$i]}" -t 0.5 \
    -vf "hue=h=${HUE[$i]},noise=alls=4:allf=t" -c:v libx264 -pix_fmt yuv420p "$D/seg$i.mp4"
  echo "file '$D/seg$i.mp4'" >> "$D/list.txt"
done

ffmpeg -y -loglevel error -f concat -safe 0 -i "$D/list.txt" -c copy "$D/vid.mp4"
python3 "$HERE/make_click.py" "$D/click.wav"
ffmpeg -y -loglevel error -i "$D/vid.mp4" -i "$D/click.wav" -c:v copy -c:a aac -shortest \
  "$HERE/uploads/test_reference.mp4"

echo "GROUND TRUTH -> 120 BPM | 16 shots | 0.50s each | 8.0s total | beats at 0.0,0.5,1.0,..."

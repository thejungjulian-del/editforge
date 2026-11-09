#!/bin/bash
# Eight synthetic source clips (4s each) for testing the matcher and renderer.
# Kept in fixtures/ because uploads/clips/ is wiped on every real build.
set -e
D="$(cd "$(dirname "$0")" && pwd)/fixtures/clips"
rm -rf "$D" && mkdir -p "$D"

SRC=(
  "testsrc2=s=1280x720:r=30" "mandelbrot=s=1280x720:r=30"
  "life=s=1280x720:r=30" "cellauto=s=1280x720:r=30"
  "gradients=s=1280x720:r=30" "smptebars=s=1280x720:r=30"
  "rgbtestsrc=s=1280x720:r=30" "testsrc=s=1280x720:r=30"
)
for i in $(seq 0 7); do
  ffmpeg -y -loglevel error -f lavfi -i "${SRC[$i]}" -t 4 \
    -vf "hue=h=$((i * 45))" -c:v libx264 -pix_fmt yuv420p "$D/clip$i.mp4"
done
echo "$(ls "$D" | wc -l | tr -d ' ') test clips -> $D"

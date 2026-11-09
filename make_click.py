"""120 BPM click track for analyzer ground-truth testing."""
import sys
import wave

import numpy as np

SR, DUR, BPM = 22050, 8.0, 120
t = np.arange(int(SR * DUR)) / SR
audio = 0.15 * np.sin(2 * np.pi * 110 * t)  # bass bed

click = np.sin(2 * np.pi * 1000 * np.arange(2500) / SR) * np.exp(-np.arange(2500) / 250.0)
for beat in np.arange(0, DUR, 60 / BPM):
    i = int(beat * SR)
    audio[i:i + len(click)] += click[: len(audio) - i]

pcm = (audio / np.abs(audio).max() * 0.9 * 32767).astype(np.int16)
with wave.open(sys.argv[1], "w") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print(f"click track: {BPM} BPM, {DUR}s -> {sys.argv[1]}")

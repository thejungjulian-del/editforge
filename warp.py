"""Ripple / wave warp — the non-rigid distortion our motion model deliberately throws away.

motion.py fits a RIGID transform to each shot (pan, zoom, roll) and lets RANSAC discard
everything that does not fit. A ripple is exactly what gets discarded: a smooth, wavy
displacement that no rigid transform can express. The signal has been flowing through our
code all along and going straight in the bin.

So: take the dense flow field, subtract the best rigid fit, and look at what is left.

  a ripple      -> residual is LARGE, SMOOTH and covers the WHOLE frame
  a moving subject -> residual is large but LOCAL (one blob, not the frame)
  nothing       -> residual is small and high-frequency, i.e. noise

Smoothness is the load-bearing test. Noise and subject motion both leave residual; only a
warp leaves a residual that still looks like itself after heavy blurring.
"""
import cv2
import numpy as np

FLOW_W = 192
# Amplitude does NOT separate a real ripple from a blur artefact — measured on real footage the
# false positives ranged 0.0045-0.041 and the true ripple sat at 0.0058, right in the middle.
# Raising this threshold only killed the real detection. The WAVELENGTH check below is what
# actually works, because it asks a physical question the artefact cannot answer.
MIN_AMP = 0.004        # residual RMS, as a fraction of frame width
MIN_SMOOTH = 0.75      # how much of the residual survives blurring
MIN_COVER = 0.45       # fraction of the frame the residual has to touch

# A ripple has to actually LOOK like a wave. On Julian's real reference this detector fired on
# 10 of 21 shots reporting wavelengths of 1.15 and 0.76 of the frame — a "wave" longer than the
# picture it is in is not a wave, it is the flow residual of a motion-blurred frame being fitted
# by a sine. Requiring a physically meaningful wavelength kills that whole class of nonsense.
MIN_WAVELENGTH = 0.05
MAX_WAVELENGTH = 0.45


def _residual_flow(prev, cur):
    """Dense flow with the best-fit rigid (affine) motion removed."""
    flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Least-squares affine fit to the flow field: [dx, dy] ~ A @ [x, y, 1].
    # This is the same rigid model motion.py already reproduces, so removing it leaves
    # precisely the part of the motion that AE's Position/Scale/Rotation CANNOT express.
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, np.float32)], 1)
    coef, *_ = np.linalg.lstsq(A, flow.reshape(-1, 2), rcond=None)
    resid = (flow.reshape(-1, 2) - A @ coef).reshape(h, w, 2)
    return resid


def _score(resid, w):
    mag = np.hypot(resid[..., 0], resid[..., 1])
    amp = float(np.sqrt((mag ** 2).mean())) / w          # fraction of frame width

    # Smoothness: a warp is low-frequency, so blurring barely changes it. Noise vanishes.
    blur = cv2.GaussianBlur(mag, (0, 0), 6)
    denom = float(np.sqrt((mag ** 2).mean()))
    smooth = float(np.sqrt((blur ** 2).mean())) / denom if denom > 1e-9 else 0.0

    # Coverage: a warp moves the whole frame. A walking subject moves one blob of it.
    cover = float((mag > max(0.3 * mag.mean(), 1e-6)).mean())
    return amp, smooth, cover, mag


def _wave(resid):
    """Rough wave geometry: amplitude in px, wavelength in px, direction in degrees."""
    dx, dy = resid[..., 0], resid[..., 1]
    # the axis the ripple displaces along
    direction = float(np.degrees(np.arctan2(dy.std(), dx.std())))

    # Wavelength: profile the dominant displacement along the axis it varies across, and
    # take the strongest spatial frequency.
    prof = dx.mean(axis=1) if abs(dy.std()) < abs(dx.std()) else dy.mean(axis=0)
    prof = prof - prof.mean()
    if len(prof) < 8 or np.allclose(prof, 0):
        return 0.0, 0.0, direction
    spec = np.abs(np.fft.rfft(prof))
    spec[0] = 0
    k = int(np.argmax(spec))
    wavelength = float(len(prof) / k) if k else 0.0
    amp = float(np.sqrt(2) * prof.std())
    return amp, wavelength, direction


def _temporal_freq(cap, fps, start, n=24):
    """How fast the ripple oscillates, in cycles per frame.

    This is needed to recover the wave's real AMPLITUDE. Optical flow reports the change in
    displacement BETWEEN consecutive frames, but a wave's amplitude is its displacement from
    rest. For a wave of amplitude A advancing by phase p per frame, the frame-to-frame change
    is A * 2*sin(p/2) — so a slow wave shows almost no per-frame flow despite being large.
    Measure p, and the amplitude follows.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    sig, prev = [], None
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        size = (FLOW_W, max(2, int(h * FLOW_W / w)))
        g = cv2.cvtColor(cv2.resize(f, size), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            r = _residual_flow(prev, g)
            sig.append(float(r[..., 0].mean() + r[..., 1].mean()))
        prev = g
    if len(sig) < 8:
        return 0.0
    x = np.array(sig) - np.mean(sig)
    spec = np.abs(np.fft.rfft(x))
    spec[0] = 0
    k = int(np.argmax(spec))
    return float(k / len(x)) if k else 0.0


def detect_warp(video_path, shots, samples=4):
    """Per shot: {"amp": frac_of_width, "wavelength": frac_of_height, "direction": deg} or absent."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = {}

    for s in shots:
        amps, smooths, covers, waves = [], [], [], []
        for t in np.linspace(s["start"], max(s["start"], s["end"] - 0.1), samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok1, f1 = cap.read()
            ok2, f2 = cap.read()
            if not (ok1 and ok2):
                continue
            h, w = f1.shape[:2]
            size = (FLOW_W, max(2, int(h * FLOW_W / w)))
            g1 = cv2.cvtColor(cv2.resize(f1, size), cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(cv2.resize(f2, size), cv2.COLOR_BGR2GRAY)

            resid = _residual_flow(g1, g2)
            amp, smooth, cover, _ = _score(resid, size[0])
            amps.append(amp); smooths.append(smooth); covers.append(cover)
            waves.append(_wave(resid))

        if len(amps) < 2:
            continue
        amp = float(np.median(amps))
        smooth = float(np.median(smooths))
        cover = float(np.median(covers))
        if amp < MIN_AMP or smooth < MIN_SMOOTH or cover < MIN_COVER:
            continue

        wa = float(np.median([w[0] for w in waves]))
        wl = float(np.median([w[1] for w in waves]))
        dr = float(np.median([w[2] for w in waves]))

        # a wave longer than the frame (or shorter than a few pixels) is not a wave
        if not (MIN_WAVELENGTH <= wl / FLOW_W <= MAX_WAVELENGTH):
            continue

        # Convert per-frame flow amplitude into the wave's actual amplitude (see
        # _temporal_freq). Without this the ripple comes out several times too weak.
        f_t = _temporal_freq(cap, fps, s["start"])
        gain = 1.0 / max(2 * np.sin(np.pi * f_t), 0.15) if f_t > 0 else 1.0
        gain = min(gain, 8.0)

        out[s["index"]] = {
            "amp": round(amp, 5),                            # residual RMS, frac of width
            "wave_amp": round(wa * gain / FLOW_W, 5),        # frac of width
            "wavelength": round(wl / FLOW_W, 5),             # frac of width
            "direction": round(dr, 1),
            "cycles_per_frame": round(f_t, 4),
        }

    cap.release()
    return out

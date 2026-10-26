"""Rendering of stored films to the 8-bit image every system receives (one rule for all systems).

Films stored with more than 8 bits (TAIX-Ray: 12-bit values in 16-bit PNG) are scaled linearly from the
image's own minimum and maximum to 0-255. A naive mode conversion clips them instead: on the R004 smoke prior
98.6 % of the pixels became 255, so every system saw a white image."""
from __future__ import annotations

from typing import Optional

EIGHT_BIT = ("L", "RGB")


def to_uint8(img):
    """8-bit grayscale ("L") from any single-channel film; 8-bit images are returned unchanged."""
    import numpy as np
    from PIL import Image

    if img.mode in EIGHT_BIT:
        return img
    if img.mode not in ("I;16", "I;16B", "I;16L", "I", "F"):
        return img.convert("L")
    a = np.asarray(img, dtype=np.float64)
    lo, hi = float(a.min()), float(a.max())
    scaled = np.zeros_like(a) if hi <= lo else (a - lo) * (255.0 / (hi - lo))
    return Image.fromarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8), mode="L")


def to_eval(img, size: int = 512, via: Optional[int] = None):
    """The image an automated evaluator reads, for real and generated films alike: 8-bit grayscale (to_uint8;
    RGB outputs by luminance), padded with black to a centred square, resized to size x size (Lanczos).

    Padding instead of a centre crop keeps the whole film: squaring a 512 x 421 TAIX-Ray film by a centre crop
    removes both lateral chest walls with the costophrenic angles, where effusion is read. via: the film is
    first reduced to via x via (the resolution-matched real-film control of systems with a smaller output)."""
    from PIL import Image

    g = to_uint8(img)
    if g.mode != "L":
        g = g.convert("L")
    w, h = g.size
    s = max(w, h)
    if w != h:
        canvas = Image.new("L", (s, s), 0)
        canvas.paste(g, ((s - w) // 2, (s - h) // 2))
        g = canvas
    if via is not None and via != s:
        g = g.resize((via, via), Image.Resampling.LANCZOS)
    if g.size != (size, size):
        g = g.resize((size, size), Image.Resampling.LANCZOS)
    return g

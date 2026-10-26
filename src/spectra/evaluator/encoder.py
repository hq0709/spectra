"""Frozen RAD-DINO encoder of the automated evaluator.

Every film, real or generated, is read through spectra.images.to_eval at the evaluation resolution (512^2) and then
RAD-DINO's own image processor on the CPU (bicubic resize to 518^2, rescale, normalisation; its centre crop is a no-op on
a square image). `pixel_values_u8` takes the to_eval films as one uint8 batch, so DataLoader workers send 512^2 uint8
films (0.26 MB) instead of 518^2 float32 tensors (3.2 MB); its output equals `pixel_values` exactly. The processor is
not run on the GPU: its CUDA resize differs from the CPU one by up to 0.076 after normalisation (2026-10-21).
A film keeps 1 + G*G tokens: the CLS token and the 37 x 37 patch-token grid average-pooled to
G x G, so a head can read side-specific findings. Tokens are the encoder's final layer-normed outputs.
"""
from __future__ import annotations

from typing import Optional, Sequence

DIM = 768


def processor(repo: Optional[str] = None):
    from transformers import AutoImageProcessor

    from ..config import constants

    return AutoImageProcessor.from_pretrained(repo or constants()["evaluator"]["encoder"])


def pixel_values(images: Sequence, proc, size: Optional[int] = None, via: Optional[int] = None):
    """[B, 3, 518, 518] float tensor of films from RAD-DINO's processor (reference path)."""
    from ..config import constants
    from ..images import to_eval

    size = size or constants()["generation"]["evaluation_resolution"]
    ims = [to_eval(im, size, via).convert("RGB") for im in images]
    return proc(images=ims, return_tensors="pt")["pixel_values"]


def eval_u8(img, size: Optional[int] = None, via: Optional[int] = None):
    """[size, size] uint8 array of the evaluation film (to_eval)."""
    import numpy as np

    from ..config import constants
    from ..images import to_eval

    return np.array(to_eval(img, size or constants()["generation"]["evaluation_resolution"], via), dtype=np.uint8)


def pixel_values_u8(u8, proc):
    """[B, 3, 518, 518] float tensor from a [B, H, W] uint8 tensor of evaluation films (grey replicated to RGB)."""
    imgs = [x.unsqueeze(0).expand(3, -1, -1).contiguous() for x in u8]
    return proc(images=imgs, return_tensors="pt")["pixel_values"]


class RadDino:
    def __init__(self, repo: Optional[str] = None, device: str = "cuda", dtype: str = "bfloat16",
                 grid: Optional[int] = None):
        import torch
        from transformers import AutoModel

        from ..config import constants

        ev = constants()["evaluator"]
        self.repo = repo or ev["encoder"]
        self.grid = grid or ev["token_grid"]
        self.device = device
        self.model = AutoModel.from_pretrained(self.repo, dtype=getattr(torch, dtype)).to(device).eval()

    def tokens(self, pixel_values):
        """[B, 1 + G*G, 768] float32: CLS, then the pooled grid in row-major order."""
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            h = self.model(pixel_values=pixel_values.to(self.device, self.model.dtype)).last_hidden_state
            cls, patches = h[:, :1].float(), h[:, 1:].float()
            n = int(round(patches.shape[1] ** 0.5))
            if n * n != patches.shape[1]:
                raise ValueError(f"{patches.shape[1]} patch tokens do not form a square grid")
            g = patches.reshape(-1, n, n, h.shape[-1]).permute(0, 3, 1, 2)
            g = F.adaptive_avg_pool2d(g, self.grid).flatten(2).transpose(1, 2)
        return torch.cat([cls, g], 1)

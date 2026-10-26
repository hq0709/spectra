"""Reference models (FINAL_PROPOSAL.md, Training Plan; constants.yaml `reference`): one architecture for the reference
family (MIMIC server: prior-only, prior+Delta, prior+Delta+actions) and the external reference pair (this cluster:
prior-only, prior+Delta on TAIX-Ray).

InstructPix2Pix-style latent diffusion initialized from RoentGen-v2: the prior's latent is concatenated to the noisy
latent (conv_in widened from 4 to 8 channels, the new input weights zero, so training starts from RoentGen-v2), and the
UNet's cross-attention reads a short token sequence instead of text: a learned null token (prior-only) or a Delta token
(prior+Delta). Sampling is DDIM without guidance; the starting noise is seeded by (prior, seed), so a model that ignores
Delta returns the same image for every query of one prior (prior-only: E1 C = 1/2 exactly).
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

CONFIGS = ("prior_only", "prior_delta")


def sinusoidal(x: torch.Tensor, dim: int) -> torch.Tensor:
    """[B] -> [B, dim] sinusoidal features of a scalar (here log(1 + hours))."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    a = x.float()[:, None] * freqs[None] * 100.0          # log-hours span ~0-6: stretch so low frequencies matter
    return torch.cat([a.sin(), a.cos()], dim=-1)


class Conditioner(nn.Module):
    """Cross-attention context: [B, 1, context_dim]. prior_only: a learned null token; prior_delta: MLP(sinusoidal(log1p h))."""

    def __init__(self, config: str, context_dim: int = 1024, delta_dim: int = 256):
        super().__init__()
        if config not in CONFIGS:
            raise ValueError(f"unknown reference configuration {config!r}")
        self.config, self.delta_dim = config, delta_dim
        if config == "prior_only":            # only the parameters a configuration uses (DDP reduces every parameter)
            self.null = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)
        else:
            self.mlp = nn.Sequential(nn.Linear(delta_dim, context_dim), nn.SiLU(), nn.Linear(context_dim, context_dim))

    def forward(self, delta_h: torch.Tensor) -> torch.Tensor:
        if self.config == "prior_only":
            return self.null.expand(delta_h.shape[0], -1, -1)
        return self.mlp(sinusoidal(torch.log1p(delta_h.clamp(min=0)), self.delta_dim))[:, None, :]


class PriorConvIn(nn.Module):
    """conv_in over [noisy latent | prior latent]: the pretrained 4-channel conv plus a separate prior conv that starts
    at zero (equivalent to widening conv_in to 8 channels with zero new weights; kept as its own tensor so the
    optimizer can give it its own learning rate, constants reference.prior_lr_mult)."""

    def __init__(self, old: nn.Conv2d):
        super().__init__()
        self.noisy = old
        self.prior = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
        nn.init.zeros_(self.prior.weight)
        self.in_channels, self.out_channels = 8, old.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # one 8-channel convolution: same numerics as the widened conv
        c = self.noisy
        return nn.functional.conv2d(x, torch.cat([c.weight, self.prior.weight], 1), c.bias, c.stride, c.padding,
                                    c.dilation, c.groups)


def widen_conv_in(unet) -> None:
    """conv_in 4 -> 8 input channels; channels 4-7 (the prior latent) start at zero."""
    unet.conv_in = PriorConvIn(unet.conv_in)
    unet.register_to_config(in_channels=8)


def upgrade_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Checkpoints written with a single 8-channel conv_in -> PriorConvIn keys (same function)."""
    if "conv_in.weight" in sd and sd["conv_in.weight"].shape[1] == 8:
        sd = dict(sd)
        w = sd.pop("conv_in.weight")
        sd["conv_in.noisy.weight"], sd["conv_in.prior.weight"] = w[:, :4].contiguous(), w[:, 4:].contiguous()
        if "conv_in.bias" in sd:
            sd["conv_in.noisy.bias"] = sd.pop("conv_in.bias")
    return sd


def upgrade_optimizer_state(opt_sd: dict, unet, cond, lr: float, mult: float) -> dict:
    """AdamW state of a single-group run with an 8-channel conv_in -> the two-group layout (all parameters but the prior
    conv; the prior conv): moments split along input channels, step counters copied, everything else unchanged."""
    import copy

    names = [n for n, _ in unet.named_parameters()] + [f"cond.{n}" for n, _ in cond.named_parameters()]
    rename = {"conv_in.noisy.weight": "conv_in.weight", "conv_in.noisy.bias": "conv_in.bias"}
    old_names = [rename.get(n, n) for n in names if n != "conv_in.prior.weight"]
    (g,) = opt_sd["param_groups"]
    old = {old_names[i]: st for i, st in opt_sd["state"].items()}
    by_name = {}
    for n in names:
        if n in ("conv_in.noisy.weight", "conv_in.prior.weight"):
            st, sl = old.get("conv_in.weight"), slice(0, 4) if n == "conv_in.noisy.weight" else slice(4, 8)
            if st is not None:
                by_name[n] = {k: (v[:, sl].clone() if torch.is_tensor(v) and v.dim() == 4 else copy.deepcopy(v))
                              for k, v in st.items()}
        elif rename.get(n, n) in old:
            by_name[n] = old[rename.get(n, n)]
    order = [n for n in names if n != "conv_in.prior.weight"] + ["conv_in.prior.weight"]
    groups = [dict(g, params=list(range(len(order) - 1))),
              dict(g, params=[len(order) - 1], lr=g["lr"] * mult, initial_lr=lr * mult)]
    return {"state": {i: by_name[n] for i, n in enumerate(order) if n in by_name}, "param_groups": groups}


def prior_parameters(unet) -> List[nn.Parameter]:
    return [unet.conv_in.prior.weight]


def load_reference(init: str, config: str, context_dim: int = 1024, delta_dim: int = 256):
    """(unet, conditioner, noise scheduler) from the RoentGen-v2 snapshot."""
    from diffusers import DDPMScheduler, UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(init, subfolder="unet")
    widen_conv_in(unet)
    sched = DDPMScheduler.from_pretrained(init, subfolder="scheduler")
    return unet, Conditioner(config, context_dim, delta_dim), sched


def step_noise(seed: int, step: int, batch: int, num_timesteps: int, shape=(4, 64, 64)):
    """Noise and timesteps of one global batch, a function of (training seed, step) only: identical across restarts
    and GPU counts (ranks slice it like the pair indices)."""
    g = torch.Generator(device="cpu").manual_seed(int(hashlib.sha256(f"{seed}|{step}".encode()).hexdigest()[:12], 16))
    return torch.randn((batch, *shape), generator=g), torch.randint(0, num_timesteps, (batch,), generator=g)


def diffusion_loss(unet, cond, sched, target_lat, prior_lat, delta_h, generator=None, noise=None, t=None):
    """Epsilon-prediction MSE (RoentGen-v2's parameterization); noise / t given, or drawn from `generator`."""
    if noise is None:
        noise = torch.randn(target_lat.shape, generator=generator, device=target_lat.device, dtype=target_lat.dtype)
        t = torch.randint(0, sched.config.num_train_timesteps, (target_lat.shape[0],), device=target_lat.device,
                          generator=generator)
    noisy = sched.add_noise(target_lat, noise, t)
    pred = unet(torch.cat([noisy, prior_lat], 1), t, encoder_hidden_states=cond(delta_h)).sample
    return torch.nn.functional.mse_loss(pred.float(), noise.float())


def pair_schedule(pairs: Sequence[Tuple[int, int, float]], steps: int, batch: int, seed: str) -> List[List[int]]:
    """The shared pair-draw schedule: for every step, `batch` pair indices drawn uniformly with a fixed seed (identical
    for all configurations, so they see the same pairs in the same order)."""
    import numpy as np

    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    return rng.integers(0, len(pairs), size=(steps, batch)).tolist()


def prior_noise(prior_id: str, seed: int, shape, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(int(hashlib.sha256(f"{prior_id}|{seed}".encode()).hexdigest()[:12], 16))
    return torch.randn(shape, generator=g).to(device)


@torch.no_grad()
def sample(unet, cond, init: str, prior_lat: torch.Tensor, delta_h: torch.Tensor, noise: torch.Tensor,
           steps: int = 50, start: str = "prior") -> torch.Tensor:
    """DDIM (eta 0), no guidance; returns latents. start="prior": the first latent is the prior diffused to the first
    DDIM timestep, sqrt(abar_t) prior + sqrt(1 - abar_t) noise. RoentGen-v2's schedule keeps signal at its last
    timestep (abar_999 = 0.0047), so every training input carries the target there; the prior is the inference-time
    stand-in for that target. start="noise": pure noise (the diffusers default)."""
    from diffusers import DDIMScheduler

    sch = DDIMScheduler.from_pretrained(init, subfolder="scheduler")
    sch.set_timesteps(steps, device=prior_lat.device)
    if start == "prior":
        x = sch.add_noise(prior_lat, noise, sch.timesteps[:1].repeat(prior_lat.shape[0]))
    elif start == "noise":
        x = noise * sch.init_noise_sigma
    else:
        raise ValueError(start)
    ctx = cond(delta_h)
    for t in sch.timesteps:
        eps = unet(torch.cat([sch.scale_model_input(x, t), prior_lat], 1), t, encoder_hidden_states=ctx).sample
        x = sch.step(eps, t, x, eta=0.0).prev_sample
    return x


def encode(vae, u8: torch.Tensor) -> torch.Tensor:
    """[B, H, W] uint8 evaluation films -> scaled latents (posterior mean)."""
    x = u8.float().div(127.5).sub(1.0)[:, None].expand(-1, 3, -1, -1)
    return vae.encode(x.to(vae.dtype)).latent_dist.mean * vae.config.scaling_factor


def decode(vae, lat: torch.Tensor):
    """Scaled latents -> list of 8-bit RGB PIL images (clipped and quantized). Like every system output, they are read
    through spectra.images.to_eval (grayscale by luminance)."""
    from PIL import Image

    x = vae.decode((lat / vae.config.scaling_factor).to(vae.dtype)).sample.float()
    a = x.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(im, mode="RGB") for im in a]


def taix_pairs(films, days: Tuple[int, int], max_per_patient: int, seed: str) -> List[Tuple[str, str, float]]:
    """(prior uid, target uid, Delta hours) of TAIX-Ray films (one per patient-day) within `days`, <= max_per_patient
    per patient, chosen by sha256(seed|patient|prior|target) order."""
    from collections import defaultdict

    by = defaultdict(list)
    for f in films:
        by[f.patient_id].append(f)
    out = []
    for pid, fs in by.items():
        fs = sorted(fs, key=lambda f: f.t_h)
        cand = [(a, b) for i, a in enumerate(fs) for b in fs[i + 1:] if days[0] * 24 <= b.t_h - a.t_h <= days[1] * 24]
        cand.sort(key=lambda ab: hashlib.sha256(f"{seed}|{pid}|{ab[0].study_id}|{ab[1].study_id}".encode()).hexdigest())
        out += [(a.study_id, b.study_id, b.t_h - a.t_h) for a, b in cand[:max_per_patient]]
    return sorted(out)


def trainable_parameters(unet, cond) -> Dict[str, int]:
    return {"unet": sum(p.numel() for p in unet.parameters()), "conditioner": sum(p.numel() for p in cond.parameters())}

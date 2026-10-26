"""Adapters for systems released as standard diffusers / transformers checkpoints. Sampling settings are
the model cards' recommended ones; the prompt is the certified templated input, unchanged."""
from __future__ import annotations

from .base import Generator


def _aspect_size(img, area: int, multiple: int):
    """(height, width) with the image's aspect ratio, about `area` pixels, both multiples of `multiple`."""
    w, h = img.size
    r = (area / (w * h)) ** 0.5
    return (max(multiple, int(round(h * r / multiple)) * multiple), max(multiple, int(round(w * r / multiple)) * multiple))


def _generator(device: str, seed: int):
    import torch

    return torch.Generator(device=device).manual_seed(int(seed))


class QwenImageEdit(Generator):
    """Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline): the prior film is the single input image."""

    def load(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(self.spec["weights"], torch_dtype=torch.bfloat16).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def _run(self, prior, text, seed):
        import torch

        with torch.inference_mode():
            return self.pipe(image=[prior], prompt=text, negative_prompt=" ", true_cfg_scale=4.0, guidance_scale=1.0,
                             num_inference_steps=40, num_images_per_prompt=1, generator=_generator(self.device, seed)).images[0]


class Flux2Klein(Generator):
    """FLUX.2 [klein] 4B (distilled: 4 steps, no guidance), the prior film as the reference image."""

    def load(self):
        import torch
        from diffusers import Flux2KleinPipeline

        self.pipe = Flux2KleinPipeline.from_pretrained(self.spec["weights"], torch_dtype=torch.bfloat16).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def _run(self, prior, text, seed):
        import torch

        h, w = _aspect_size(prior, 1024 * 1024, 16)   # the card's 1024² budget at the prior's aspect ratio
        with torch.inference_mode():
            return self.pipe(prompt=text, image=[prior], height=h, width=w, guidance_scale=1.0,
                             num_inference_steps=4, generator=_generator(self.device, seed)).images[0]


class StatePredictor(Generator):
    """Medical VLM as a state predictor (Lingshu-32B: Qwen2.5-VL; MedGemma 1.5: Gemma 3): prior film +
    templated input, greedy decoding, one response per query."""

    kind = "state"
    max_new_tokens = 256

    def load(self):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model = AutoModelForImageTextToText.from_pretrained(self.spec["weights"], torch_dtype=torch.bfloat16,
                                                                 device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(self.spec["weights"])

    def _run(self, prior, text, seed):
        import torch

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], images=[prior], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


# RadEdit (E2 positive control, "target state given"): an impression-style prompt for the target device state and an
# edit mask over the region the device occupies; the keep mask is its inverse. Regions are boxes (x0, y0, x1, y1) as
# fractions of the film itself (image left = patient right), placed on the 512² evaluation canvas (spectra.images.to_eval)
# inside the film, never on its black padding. The mask rule and the prompts are fixed on development data before R036
# (at most three templates; Gate 3); these are the smoke-test ones.
DEVICE_REGIONS = {
    "ETT": {"none": [(0.40, 0.00, 0.60, 0.45)]},
    "enteric": {"none": [(0.40, 0.00, 0.60, 0.70), (0.50, 0.60, 0.85, 0.95)]},
    "CVC": {"R": [(0.15, 0.00, 0.55, 0.50)], "L": [(0.45, 0.00, 0.85, 0.50)]},
    "chest_tube": {"R": [(0.05, 0.10, 0.50, 0.90)], "L": [(0.50, 0.10, 0.95, 0.90)]},
}
TARGET_PROMPTS = {
    ("ETT", True): "Endotracheal tube in standard position.", ("ETT", False): "No endotracheal tube.",
    ("enteric", True): "Enteric tube courses below the diaphragm into the stomach.", ("enteric", False): "No enteric tube.",
    ("CVC", True): "{side} internal jugular central venous catheter with tip in the superior vena cava.",
    ("CVC", False): "No central venous catheter.",
    ("chest_tube", True): "{side} chest tube in place.", ("chest_tube", False): "No chest tube.",
}


def device_mask(device: str, side: str = "none", film_size=(512, 512), size: int = 512):
    """Edit mask on the size x size canvas of a film of film_size (w, h) padded to a centred square (to_eval)."""
    from PIL import Image, ImageDraw

    w, h = film_size
    k = size / max(w, h)
    ox, oy, fw, fh = (max(w, h) - w) / 2 * k, (max(w, h) - h) / 2 * k, w * k, h * k
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in DEVICE_REGIONS[device][side]:
        d.rectangle([ox + x0 * fw, oy + y0 * fh, ox + x1 * fw - 1, oy + y1 * fh - 1], fill=255)
    return m


class RadEdit(Generator):
    """RadEdit (UNet from microsoft/radedit; SDXL-VAE; BioViL-T text encoder and tokenizer; DDIM) with the model
    card's editing pipeline (`pipeline.py`, DDPM-inversion edit with edit and keep masks) and its settings: 200
    steps, skip ratio 0.3, guidance 7.5, empty inversion prompt. Input: the prior at the 512² evaluation canvas."""

    def load(self):
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        unet = UNet2DConditionModel.from_pretrained(self.spec["weights"], subfolder="unet")
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        text_encoder = AutoModel.from_pretrained("microsoft/BiomedVLP-BioViL-T", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedVLP-BioViL-T", model_max_length=128, trust_remote_code=True)
        scheduler = DDIMScheduler(beta_schedule="linear", clip_sample=False, prediction_type="epsilon",
                                  timestep_spacing="trailing", steps_offset=1)
        gen = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                                      safety_checker=None, requires_safety_checker=False, feature_extractor=None)
        # remote code: pipeline.py (and BioViL-T's modeling_cxrbert.py / configuration_cxrbert.py) were read before
        # the first run (2026-10-21; model code only, no I/O, network or subprocess calls)
        local = snapshot_download(self.spec["weights"], allow_patterns=["pipeline.py"])
        self.pipe = DiffusionPipeline.from_pipe(gen, custom_pipeline=local, trust_remote_code=True).to(self.device, torch.float32)

    def target(self, prior, device: str, present: bool, side: str = "none", seed: int = 0):
        """The edited follow-up for a given target device state (Output, as generate())."""
        reg_side = side if device in ("CVC", "chest_tube") else "none"
        self._mask = device_mask(device, reg_side, prior.size)
        word = {"R": "Right", "L": "Left"}.get(side, "")
        return self.generate(prior, TARGET_PROMPTS[(device, present)].format(side=word).strip(), seed)

    def _run(self, prior, text, seed):
        import torch
        from PIL import ImageOps

        from ..images import to_eval

        img = to_eval(prior).convert("RGB")
        torch.manual_seed(int(seed))
        with torch.inference_mode():
            out = self.pipe(text, weights=[7.5], image=img, edit_mask=self._mask, keep_mask=ImageOps.invert(self._mask),
                            num_inference_steps=200, invert_prompt="", skip_ratio=0.3, prog_bar=False, output_type="pil")
        return out[0] if isinstance(out, (list, tuple)) else out.images[0]

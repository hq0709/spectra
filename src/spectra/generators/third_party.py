"""Adapters for systems that run through their official code (systems.yaml `code`, checked out under
SPECTRA_THIRD_PARTY, default cxr-wm-bench/third_party/), each in its own env (`env`). Every adapter follows
the repository's own inference path and settings; departures needed to run it are named in the class
docstring and in the system's adaptation record."""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

from .base import Generator, NoImage


def third_party(name: str) -> Path:
    root = Path(os.environ.get("SPECTRA_THIRD_PARTY", Path(__file__).resolve().parents[4] / "third_party"))
    p = root / name
    if not p.is_dir():
        raise FileNotFoundError(f"{p}: check out the commit named in systems.yaml (`code`)")
    return p


def snapshot(repo: str, probe: str = "config.json") -> str:
    """Local snapshot directory of a staged repository (scripts/fetch_weights.py); never downloads."""
    from huggingface_hub import try_to_load_from_cache

    f = try_to_load_from_cache(repo, probe)
    if not isinstance(f, str):
        raise FileNotFoundError(f"{repo}/{probe} is not staged (scripts/fetch_weights.py)")
    return str(Path(f).parents[len(Path(probe).parts) - 1])


class ProgEmu(Generator):
    """ProgEmu (Emu3, 8.5B) through inference/infer_progemu.py: the prior prepared as its data were
    (register_chexpertplus.py: extract_foreground, then cv2.resize to 512 x 512; the decoder assumes the
    64 x 64 token grid of a 512² film, and a non-square prior scrambles the output), VQ-encoded with the Emu3
    VisionTokenizer, condition text after it without template, prefix-constrained sampling with top-k 2048
    (scripts/infer.sh: --use_pc_proc --do_sample), no classifier-free guidance, flash attention 2. The
    tokenizer is Emu3-Stage1 (the script's TOK_HUB)."""

    def load(self):
        import torch

        sys.path.insert(0, str(third_party("progemu")))
        from emu3.mllm import Emu3Config, Emu3Tokenizer
        from emu3.mllm.modeling_ti2ti import Emu3ForCausalLM
        from emu3.mllm.processing_ti2ti import Emu3Processor
        from emu3.visionvq import Emu3VisionVQImageProcessor, Emu3VisionVQModel

        mp, tp = snapshot(self.spec["weights"]), snapshot(self.spec["tokenizer"], "tokenizer_config.json")
        vq = snapshot("BAAI/Emu3-VisionTokenizer")
        self.model = Emu3ForCausalLM.from_pretrained(mp, config=Emu3Config.from_pretrained(mp), device_map=self.device,
                                                     torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").eval()
        tok = Emu3Tokenizer.from_pretrained(tp, model_max_length=self.model.config.max_position_embeddings, padding_side="left")
        self.processor = Emu3Processor(Emu3VisionVQImageProcessor.from_pretrained(vq, trust_remote_code=True),
                                       Emu3VisionVQModel.from_pretrained(vq, device_map=self.device).eval(), tok,
                                       only_output_vision=False)

    def _run(self, prior, text, seed):
        import torch
        from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor
        from transformers.generation.configuration_utils import GenerationConfig

        import cv2
        import numpy as np
        from PIL import Image
        from register_chexpertplus import extract_foreground   # ProgEmu's own data preparation

        p = self.processor
        square = Image.fromarray(cv2.resize(extract_foreground(np.array(prior.convert("L"))), (512, 512))).convert("RGB")
        with torch.no_grad():
            px = p.image_processor(images=square, return_tensors="pt")["pixel_values"]
            tokens = p.vision_tokenizer.encode(px.to(self.device)).squeeze().cpu()
        pos = p(text=text, image=[tokens], return_tensors="pt", padding="longest")
        h, w = pos.image_size[:, 0], pos.image_size[:, 1]
        procs = LogitsProcessorList([PrefixConstrainedLogitsProcessor(p.build_prefix_constrained_fn(h, w), num_beams=1)])
        cfg = GenerationConfig(use_cache=True, eos_token_id=self.model.config.eos_token_id,
                               pad_token_id=self.model.config.pad_token_id, max_new_tokens=9216, do_sample=True, top_k=2048)
        torch.manual_seed(int(seed))
        out = self.model.generate(pos.input_ids.to(self.device), cfg, logits_processor=procs,
                                  attention_mask=pos.attention_mask.to(self.device))
        seq = p.tokenizer.decode(out.squeeze()[pos.input_ids.shape[1]:])
        rows, _ = p.split_visual_textual_tokens(seq, keep_text_as_is=False)
        if not rows:
            raise NoImage("ProgEmu generated no visual tokens")
        with torch.no_grad():
            img = p.vision_tokenizer.decode(p.get_tensor_from_visual_token_rows(rows, fit_type="global").to(self.device)).float()
        return p.image_processor.postprocess(img)["pixel_values"][0]


class UniMedVL(Generator):
    """UniMedVL through codes/interactive_image_generator.py (its default configuration): EMA bf16 weights,
    understanding mode (a greedy description of the film under the instruction, appended to it), then
    interleaved image editing with cfg_text 4.0, cfg_img 2.0, cfg interval [0, 1], text-channel renorm,
    timestep shift 3.0 and 50 steps at the aspect-preserving target size."""

    VAE_SIZE, VIT_SIZE, MAX_MEM = (1024, 32, 16), (980, 387, 14), "40GiB"

    def load(self):
        import torch
        from accelerate import dispatch_model, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

        sys.path.insert(0, str(third_party("unimedvl") / "codes"))
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.qwen2 import Qwen2Tokenizer
        from modeling.unimedvl import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel

        mp = snapshot(self.spec["weights"], "llm_config.json")
        llm = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
        llm.qk_norm, llm.tie_word_embeddings, llm.layer_module = True, False, "Qwen2MoTDecoderLayer"
        vit = SiglipVisionConfig.from_json_file(os.path.join(mp, "vit_config.json"))
        vit.rope = False
        vae, vae_cfg = load_ae(local_path=os.path.join(mp, "ae.safetensors"))
        cfg = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm, vit_config=vit, vae_config=vae_cfg,
                          vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh", latent_patch_size=2,
                          max_latent_size=64)
        with init_empty_weights():
            model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), cfg, vae_model=vae)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
        tok, new_ids, _ = add_special_tokens(Qwen2Tokenizer.from_pretrained(mp))
        cpu = {n: "cpu" for n, _ in model.named_parameters()}
        cpu.update({k: "cpu" for k in ("language_model", "vit_model", "time_embedder", "latent_pos_embed", "vae2llm",
                                       "llm2vae", "connector", "vit_pos_embed", "vae_model")})
        model = load_checkpoint_and_dispatch(model, checkpoint=os.path.join(mp, "ema_bf16.safetensors"), device_map=cpu,
                                             offload_buffers=False, dtype=torch.bfloat16, force_hooks=False)
        dm = infer_auto_device_map(model, max_memory={0: self.MAX_MEM}, no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"])
        first = dm.get("language_model.model.embed_tokens", "cuda:0")
        for k in ("language_model.model.embed_tokens", "time_embedder", "latent_pos_embed", "vae2llm", "llm2vae",
                  "connector", "vit_pos_embed"):
            if k in dm:
                dm[k] = first
        model = dispatch_model(model, device_map=dm).eval()
        vae = vae.to(device=first, dtype=torch.bfloat16)
        self.inferencer = InterleaveInferencer(model=model, vae_model=vae, tokenizer=tok,
                                               vae_transform=ImageTransform(*self.VAE_SIZE),
                                               vit_transform=ImageTransform(*self.VIT_SIZE), new_token_ids=new_ids)

    @staticmethod
    def _seed(seed: int):
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False

    def _run(self, prior, text, seed):
        from PIL import Image

        self._seed(int(seed))
        inf = self.inferencer
        und = inf(image=prior, text=text, think=False, understanding_output=True, do_sample=False, text_temperature=0.3,
                  max_think_token_n=800).get("text", "")
        final = f"{text}\n\n{und}" if und else text
        size = inf._calculate_target_size_with_aspect_ratio(*prior.size)
        out = inf.interleave_inference(input_lists=[prior, final], think=False, understanding_output=False,
                                       cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0],
                                       cfg_renorm_type="text_channel", timestep_shift=3.0, num_timesteps=50, image_shapes=size)
        img = next((x for x in out if isinstance(x, Image.Image)), None)
        if img is None:
            raise NoImage("UniMedVL returned no image")
        self.last_understanding = und
        return img


class Prism(Generator):
    """PRISM (SD v1.5 with its CheXpert-fine-tuned UNet) through generate_cf_images.py: null-text inversion
    of the prior (50 DDIM steps, 10 inner steps) with a fixed source caption (systems.yaml `source_caption`,
    identical for every test and arm), then a prompt-to-prompt refine edit to the certified 77-token compact
    input with self-replace 0.5 and cross-replace 0.8 (the script's __main__ values); the edit is deterministic
    given the inversion, so the seed has no effect. Departures, both needed to run the released code:
    (1) the repository omits the LANCE utilities it imports (utils/edit_utils, misc_utils), taken from LANCE
    at the commit in systems.yaml; (2) LANCE registers attention control on diffusers' `CrossAttention`, which
    diffusers >= 0.15 (PRISM pins 0.24.0) renamed `Attention`: registration is extended to `Attention`
    (prompt-to-prompt issue #44), otherwise no attention layer is controlled and the edit silently degrades.
    LANCE's CLIP filtering of candidate edits (OpenAI CLIP, imported at module level) is not used: one edit per
    input, so a stub module stands in for `clip` and raises if anything calls it."""

    def load(self):
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
        from safetensors.torch import load_file

        # only LANCE's editing utilities: a package stub keeps lance/__init__.py (captioning models) unimported;
        # misc_utils imports OpenAI CLIP at module level for its CLIP filtering, which a single edit never runs
        if "clip" not in sys.modules:
            def absent(*args, **kwargs):
                raise RuntimeError("OpenAI CLIP is not installed; LANCE's CLIP filtering is not part of the edit")

            clip = types.ModuleType("clip")
            clip.load = clip.tokenize = absent   # the two functions misc_utils.ClipSimilarity calls
            sys.modules["clip"] = clip
        if "lance" not in sys.modules:
            stub = types.ModuleType("lance")
            stub.__path__ = [str(third_party("lance") / "lance")]
            sys.modules["lance"] = stub
        from lance.utils import edit_utils, ptp_utils

        base = snapshot(self.spec["base"], "model_index.json")
        unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(base, subfolder="unet"))
        unet.load_state_dict(load_file(os.path.join(snapshot(self.spec["weights"], "model.safetensors"), "model.safetensors")))
        sched = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                              set_alpha_to_one=False, steps_offset=1)
        self.pipe = StableDiffusionPipeline.from_pretrained(base, unet=unet, scheduler=sched, safety_checker=None,
                                                            feature_extractor=None, requires_safety_checker=False).to(self.device)
        self._patch_registration(ptp_utils)
        self.edit_utils = edit_utils
        self.inverter = edit_utils.NullInversion(self.pipe, torch.device(self.device))
        probe = edit_utils.AttentionStore()
        ptp_utils.register_attention_control(self.pipe, probe)
        if probe.num_att_layers == 0:
            raise RuntimeError("prompt-to-prompt attention control registered on no attention layer")
        self.att_layers = probe.num_att_layers

    @staticmethod
    def _patch_registration(*modules):
        """Make LANCE's register_attention_control match diffusers' `Attention` class as well."""
        import inspect

        ptp = modules[0]
        src = inspect.getsource(ptp.register_attention_control)
        old = 'net_.__class__.__name__ == "CrossAttention"'
        if old not in src:
            raise RuntimeError("LANCE register_attention_control changed; re-check the Attention patch")
        ns = dict(ptp.__dict__)
        exec(compile(src.replace(old, 'net_.__class__.__name__ in ("CrossAttention", "Attention")'), ptp.__file__, "exec"), ns)
        ptp.register_attention_control = ns["register_attention_control"]   # edit_utils calls ptp_utils.<name>

    def _run(self, prior, text, seed):
        from PIL import Image

        eu = self.edit_utils
        source = str(self.spec["source_caption"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "prior.png")
            prior.save(path)
            (_, _), x_t, uncond = self.inverter.invert(path, source, offsets=(0, 0, 0, 0), num_inner_steps=10)
        prompts = [source, text]
        controller = eu.make_controller(prompts, self.device, self.pipe.tokenizer, False, {"default_": 0.8}, 0.5)
        images, _ = eu.run_and_display(self.pipe, prompts, controller, latent=x_t, uncond_embeddings=uncond, verbose=False)
        return Image.fromarray(images[1])

"""CogVideoX-5B-I2V adapter (experimental wrapper, weights not shipped).

This adapter exposes the paper's interior-concat condition channel
through a documented, reproducible hook:

1. The anchor image is encoded with the pipeline's VAE to a condition
   latent.
2. The native image-to-video pipeline is invoked with the anchor as its
   first-frame condition, which fixes the conditioning site.
3. At the end of each denoising step ``s`` (execution order), the
   latent slot that carries the condition is blended as
   ``z_slot <- gamma_s * z_anchor + (1 - gamma_s) * z_slot``.
   ``gamma_s = 1`` therefore reproduces the native pipeline exactly
   (the slot is left untouched), and ``gamma_s = 0`` removes the
   condition at that step.

The blend is applied through ``callback_on_step_end`` so no diffusers
internals are monkey-patched. Exact parity with the authors' original
harness is *not* claimed: the original run manifest and the released
per-step gamma vectors are the reference for the reported numbers.

Requires: ``diffusers>=0.31``, ``transformers``, ``torch`` and a local
copy of ``THUDM/CogVideoX-5b-I2V`` (not redistributed).
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from .base import BackboneAdapter


class CogVideoXAdapter(BackboneAdapter):
    name = "cogvideox-5b-i2v-concat"
    T = 30
    video_frames = 49
    resolution = "480x720"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        steps: int = 30,
        guidance: float = 6.0,
    ):
        import torch
        from diffusers import CogVideoXImageToVideoPipeline

        model_dir = model_dir or os.environ.get("COGVIDEOX_I2V_DIR")
        if not model_dir:
            raise RuntimeError(
                "Set COGVIDEOX_I2V_DIR to a local THUDM/CogVideoX-5b-I2V copy"
            )
        self.torch = torch
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_dir, torch_dtype=getattr(torch, dtype)
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.steps = steps
        self.guidance = guidance
        self._gamma: Optional[np.ndarray] = None
        self._step = 0
        self._last_frames: Optional[List[np.ndarray]] = None
        self._install_condition_hook()

    def condition_site(self) -> str:
        return (
            "CogVideoX I2V condition channels: per-step scaling "
            "c_s = gamma_s * c(a) on the concatenated image latents"
        )

    # ------------------------------------------------------------------
    def _install_condition_hook(self):
        """Scale the native condition channels per denoising step.

        CogVideoX I2V concatenates the encoded anchor ``image_latents``
        to the noisy latents along the channel axis before every
        transformer call. The hook halves that concatenated tensor and
        multiplies the condition half by ``gamma_s`` for the current
        step: ``gamma=1`` is exactly the native pipeline, ``gamma=0``
        removes the condition (zero-latent null path).
        """
        orig_forward = self.pipe.transformer.forward

        def forward(hidden_states, *args, **kwargs):
            g = 1.0 if self._gamma is None else float(
                self._gamma[min(self._step, len(self._gamma) - 1)]
            )
            self._step += 1
            if g < 0.999:
                c = hidden_states.shape[2] // 2
                cond = hidden_states[:, :, c:] * g
                hidden_states = self.torch.cat(
                    [hidden_states[:, :, :c], cond], dim=2
                )
            return orig_forward(hidden_states, *args, **kwargs)

        self.pipe.transformer.forward = forward

    # ------------------------------------------------------------------
    def generate(
        self,
        anchor: np.ndarray,
        prompt: str,
        p: int,
        seed: int,
        gamma: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Run one generation; returns ``num_frames`` RGB uint8 frames."""
        from PIL import Image

        torch = self.torch
        a = np.asarray(anchor)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(a).resize((self.width, self.height))

        self._gamma = None if gamma is None else np.asarray(gamma, float)
        self._step = 0
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        out = self.pipe(
            prompt=prompt,
            image=image,
            height=self.height,
            width=self.width,
            num_frames=self.num_frames,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=gen,
        )
        frames = out.frames[0]
        arrs = []
        for f in frames:
            x = np.asarray(f)
            if x.dtype != np.uint8:
                x = np.clip(x, 0.0, 1.0) * 255.0
                x = x.astype(np.uint8)
            arrs.append(x)
        self._last_frames = arrs
        return arrs

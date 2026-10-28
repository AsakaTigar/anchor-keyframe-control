"""Backbone adapter protocol.

ANCHOR never ships third-party weights. A backbone adapter exposes one
method::

    generate(anchor, prompt, p, seed, gamma) -> frames

where ``frames`` is a list/array of ``T_video`` RGB frames, ``p`` is the
requested interior frame index, and ``gamma`` is the per-step schedule
in execution order. Boundary-first-frame adapters are also supported
(``p`` is then ignored by the model and only used for readout).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class BackboneAdapter(ABC):
    name: str = "abstract"
    T: int = 30            # reverse steps
    video_frames: int = 49
    resolution: str = "480x720"

    @abstractmethod
    def generate(
        self,
        anchor: np.ndarray,
        prompt: str,
        p: int,
        seed: int,
        gamma: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Return the generated frames for one request."""

    def condition_site(self) -> str:
        """Human-readable description used in the run manifest."""
        return "unspecified"


class SyntheticBackbone(BackboneAdapter):
    """Deterministic toy backbone used by the unit tests and demos.

    It emulates a linear response: the anchor's pixels diffuse into the
    video according to ``gamma`` and a smooth per-step kernel, which is
    enough to exercise probing, classification, and the guard without a
    GPU.
    """

    name = "synthetic"
    T = 12
    video_frames = 13

    def __init__(self, seed: int = 2027, response: float = 0.6,
                 leak: float = 0.2, noise: float = 0.01,
                 T: int = 12):
        self.seed = seed
        self.response = response
        self.leak = leak
        self.noise = noise
        self.T = int(T)

    def condition_site(self) -> str:
        return "synthetic interior slot"

    def generate(self, anchor, prompt, p, seed, gamma=None):
        rng = np.random.default_rng(seed)
        a = np.asarray(anchor, dtype=np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        T = self.video_frames
        H, W, _ = a.shape
        g = np.ones(self.T) if gamma is None else np.asarray(gamma, float)
        g = np.clip(g, 0, 1)
        # video prior: smooth gradient background + slow pan
        frames = []
        for t in range(T):
            bg = np.zeros((H, W, 3), np.float32)
            bg[:] = (0.15 + 0.5 * t / max(T - 1, 1))
            shift = int(4 * t / max(T - 1, 1))
            if shift:
                bg = np.roll(bg, shift, axis=1)
            frames.append(bg)
        # anchor response at p, leakage elsewhere, both scaled by the
        # schedule mass at early vs late steps
        early = g[: max(1, self.T // 3)].mean()
        late = g[self.T // 3:].mean() if len(g) > self.T // 3 else 0.0
        for t in range(T):
            w_p = self.response * early * np.exp(-0.5 * ((t - p) / 1.5) ** 2)
            w_l = self.leak * late / T
            frames[t] = (1 - w_p - w_l) * frames[t] + w_p * a + w_l * a
            frames[t] += rng.normal(0, self.noise, frames[t].shape)
        return [np.clip(f, 0, 1) for f in frames]

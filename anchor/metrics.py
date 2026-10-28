"""Metrics used by ANCHOR and the revision's sanity checks.

* ``anchor_lpips``: LPIPS between the requested anchor image and the
  generated frame at the same time index (the paper's primary metric).
* ``temporal_consistency``: adjacent-frame cosine similarity under a
  caller-provided encoder (CLIP or DINOv2), a VBench-style proxy that
  catches flicker/frame leakage that anchor LPIPS cannot see.
* ``motion_magnitude``: mean Farneback flow magnitude, for motion
  collapse checks.

The torch-dependent parts are optional; the module imports cleanly
without torch so that the statistics can be tested anywhere.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

try:  # optional
    import torch  # noqa: F401

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class LPIPSMetric:
    """Thin wrapper around the official ``lpips`` package.

    The package and its AlexNet/VGG weights are not redistributed; see
    ``docs/PARAMETERS.md`` for the exact variant used in the paper
    (LPIPS-Alex, net='alex', version 0.1).
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        if not _HAS_TORCH:
            raise RuntimeError("LPIPS needs torch; install requirements-gpu.txt")
        import lpips  # type: ignore

        self.device = device
        self.model = lpips.LPIPS(net=net).to(device).eval()

    def __call__(self, anchor: np.ndarray, frame: np.ndarray) -> float:
        """Both images are HxWx3 uint8 or float in [0, 1]."""
        import torch as _t

        def prep(x: np.ndarray):
            a = np.asarray(x)
            if a.dtype == np.uint8:
                a = a.astype(np.float32) / 255.0
            ten = _t.from_numpy(a).permute(2, 0, 1).unsqueeze(0)
            return (ten * 2.0 - 1.0).to(self.device)

        with _t.no_grad():
            return float(self.model(prep(anchor), prep(frame)).item())


def temporal_consistency(
    frames: Sequence[np.ndarray],
    encode: Callable[[Sequence[np.ndarray]], np.ndarray],
) -> float:
    """Mean adjacent-frame cosine similarity.

    ``encode`` maps a list of HxWx3 frames to an (n, d) embedding matrix
    (already normalized or not; normalization happens here).
    """
    if len(frames) < 2:
        return float("nan")
    E = np.asarray(encode(list(frames)), dtype=float)
    if E.ndim != 2 or len(E) != len(frames):
        raise ValueError("encoder must return (n, d) embeddings")
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    En = E / np.clip(norms, 1e-12, None)
    return float(np.mean(np.sum(En[:-1] * En[1:], axis=1)))


def motion_magnitude(frames_gray: Sequence[np.ndarray]) -> float:
    """Mean Farneback flow magnitude between consecutive gray frames."""
    import cv2

    mags = []
    for a, b in zip(frames_gray[:-1], frames_gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        mags.append(float(np.sqrt((flow ** 2).sum(-1)).mean()))
    return float(np.mean(mags)) if mags else float("nan")


def frame_leakage(anchor: np.ndarray, frames: Sequence[np.ndarray],
                  p: int, tol: float = 0.02) -> float:
    """Fraction of non-anchor frames that are near-copies of the anchor.

    A simple pixel-space check for the failure mode discussed in the
    paper (the condition frame being copied elsewhere). ``tol`` is the
    mean absolute difference (0-1 scale) under which two frames are
    considered copies.
    """
    a = np.asarray(anchor, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    copied = 0
    other = 0
    for i, f in enumerate(frames):
        if i == p:
            continue
        ff = np.asarray(f, dtype=np.float32)
        if ff.max() > 1.5:
            ff = ff / 255.0
        if np.abs(ff - a).mean() <= tol:
            copied += 1
        other += 1
    return float(copied / max(other, 1))

"""ANCHOR reference implementation (probe -> classify -> gate).

Released with the IEEE TCE revision of
"ANCHOR: Probe, Classify, Gate -- Interface- and Domain-Adaptive
Keyframe Control for Video Diffusion".

The package implements the controller and its statistics. It does not
ship any third-party video-diffusion weights; backbone access goes
through the adapter protocol in ``anchor.backbones``.
"""

from .probe import ProbeDesign, ridge_operator, summarize_response, stability_bound
from .bootstrap import bca_interval, bca_upper_bound
from .classify import (
    Regime,
    ClassificationResult,
    classify_regime,
    classify_from_statistics,
)
from .guard import GuardDecision, guard_decision
from .candidates import native_full, skip_k, logsnr_sigmoid, micro_skip

__all__ = [
    "ProbeDesign",
    "ridge_operator",
    "summarize_response",
    "stability_bound",
    "bca_interval",
    "bca_upper_bound",
    "Regime",
    "ClassificationResult",
    "classify_regime",
    "classify_from_statistics",
    "GuardDecision",
    "guard_decision",
    "native_full",
    "skip_k",
    "logsnr_sigmoid",
    "micro_skip",
]

__version__ = "1.0.0"

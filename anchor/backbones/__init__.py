"""Backbone adapters. Third-party weights are never redistributed."""

from .base import BackboneAdapter, SyntheticBackbone

__all__ = ["BackboneAdapter", "SyntheticBackbone"]

try:  # requires diffusers + a local CogVideoX copy
    from .cogvideox import CogVideoXAdapter

    __all__.append("CogVideoXAdapter")
except Exception:  # pragma: no cover - optional dependency
    CogVideoXAdapter = None

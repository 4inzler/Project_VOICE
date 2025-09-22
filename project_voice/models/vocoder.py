"""Vocoder modules (NSF-HiFiGAN and optional BigVGAN)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


try:
    from nsfhifigan.models import NsfHifiGANGenerator  # type: ignore
except Exception:  # pragma: no cover
    NsfHifiGANGenerator = None  # type: ignore

try:
    from bigvgan.generator import BigVGAN  # type: ignore
except Exception:  # pragma: no cover
    BigVGAN = None  # type: ignore


@dataclass
class VocoderConfig:
    use_ns_fhifigan: bool = True
    use_bigvgan: bool = False
    checkpoint: Optional[Path] = None


class Vocoder(nn.Module):
    def __init__(self, cfg: VocoderConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        if cfg.use_ns_fhifigan and NsfHifiGANGenerator is not None:
            self.model = NsfHifiGANGenerator().to(device)
        elif cfg.use_bigvgan and BigVGAN is not None:
            self.model = BigVGAN().to(device)
        else:  # pragma: no cover
            raise RuntimeError("No supported vocoder backend is available")

        if cfg.checkpoint and cfg.checkpoint.exists():
            state = torch.load(cfg.checkpoint, map_location=device)
            self.model.load_state_dict(state["model"] if "model" in state else state)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.model(mel)


__all__ = ["Vocoder", "VocoderConfig"]

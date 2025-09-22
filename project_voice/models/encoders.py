"""Content encoder wrappers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio


@dataclass
class HuBERTSoftEncoder:
    """HuBERT soft content encoder wrapper."""

    device: torch.device
    layer: int = 6

    def __post_init__(self) -> None:
        bundle = torchaudio.pipelines.HUBERT_BASE
        self.model = bundle.get_model().to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        waveform = waveform.to(self.device)
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform)
        return features[self.layer]


__all__ = ["HuBERTSoftEncoder"]

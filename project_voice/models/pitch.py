"""Pitch extraction wrappers for RMVPE and CREPE."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import crepe
except Exception as exc:  # pragma: no cover
    crepe = None  # type: ignore

try:
    from rmvpe import RMVPE
except Exception:  # pragma: no cover
    RMVPE = None  # type: ignore


@dataclass
class PitchExtractor:
    """Combine RMVPE with a CREPE fallback."""

    rmvpe_checkpoint: Optional[Path]
    crepe_model_capacity: str = "tiny"
    use_gpu: bool = True
    fallback_confidence: float = 0.6

    def __post_init__(self) -> None:
        device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        self.device = torch.device(device)
        self.rmvpe = None
        if RMVPE is not None and self.rmvpe_checkpoint:
            self.rmvpe = RMVPE(self.rmvpe_checkpoint, device=device)

    def _rmvpe_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.rmvpe is None:
            raise RuntimeError("RMVPE checkpoint not available")
        return self.rmvpe.infer_from_audio(audio, sr)

    def _crepe_f0(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if crepe is None:
            raise RuntimeError("CREPE is not installed")
        _, f0, confidence, _ = crepe.predict(
            audio,
            sr,
            viterbi=True,
            model_capacity=self.crepe_model_capacity,
            center=True,
        )
        f0[confidence < self.fallback_confidence] = 0.0
        return f0

    def __call__(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if self.rmvpe is not None:
            try:
                return self._rmvpe_f0(audio, sample_rate)
            except Exception:
                pass
        return self._crepe_f0(audio, sample_rate)


__all__ = ["PitchExtractor"]

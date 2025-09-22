"""ECAPA-TDNN retrieval bank for similarity guidance."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

try:
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN  # type: ignore
except Exception:  # pragma: no cover
    ECAPA_TDNN = None  # type: ignore


@dataclass
class RetrievalBankConfig:
    embedding_dim: int = 192
    checkpoint: Path | None = None


class RetrievalBank:
    def __init__(self, cfg: RetrievalBankConfig, device: torch.device, max_entries: int = 2048) -> None:
        if ECAPA_TDNN is None:  # pragma: no cover
            raise RuntimeError("speechbrain is required for ECAPA-TDNN retrieval")
        self.model = ECAPA_TDNN(input_size=80, lin_neurons=cfg.embedding_dim)
        if cfg.checkpoint and cfg.checkpoint.exists():
            state = torch.load(cfg.checkpoint, map_location="cpu")
            self.model.load_state_dict(state)
        self.model.eval().to(device)
        self.device = device
        self.bank: List[torch.Tensor] = []
        self.max_entries = max_entries

    @torch.inference_mode()
    def add(self, mel: torch.Tensor) -> None:
        embeddings = self.model(mel.to(self.device))
        embeddings = F.normalize(embeddings, dim=-1)
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        for emb in embeddings:
            self.bank.append(emb)
            if len(self.bank) > self.max_entries:
                self.bank.pop(0)

    @torch.inference_mode()
    def similarity(self, mel: torch.Tensor) -> torch.Tensor:
        if not self.bank:
            return torch.zeros(1, device=self.device)
        emb = self.model(mel.to(self.device))
        emb = F.normalize(emb, dim=-1)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        bank = torch.stack(self.bank)
        return torch.matmul(bank, emb.T).mean(dim=0)


__all__ = ["RetrievalBank", "RetrievalBankConfig"]

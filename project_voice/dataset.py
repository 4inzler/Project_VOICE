"""Dataset utilities for Project VOICE."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .config import DatasetConfig


@dataclass
class SegmentMetadata:
    """Metadata describing a segmented utterance."""

    file_path: Path
    speaker: str
    duration: float
    text: Optional[str]


class VoiceDataset(Dataset):
    """Simple dataset returning waveform segments and metadata."""

    def __init__(
        self,
        segments: List[SegmentMetadata],
        sample_rate: int,
        segment_seconds: float = 4.0,
    ) -> None:
        self.segments = segments
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds

    def __len__(self) -> int:  # noqa: D401
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, SegmentMetadata]:
        meta = self.segments[idx]
        waveform, sr = sf.read(meta.file_path)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)
        if sr != self.sample_rate:
            raise ValueError(f"Expected sample rate {self.sample_rate}, got {sr}")

        target_length = int(self.sample_rate * self.segment_seconds)
        if len(waveform) > target_length:
            start = random.randint(0, len(waveform) - target_length)
            waveform = waveform[start : start + target_length]
        elif len(waveform) < target_length:
            pad = target_length - len(waveform)
            waveform = np.pad(waveform, (0, pad))

        tensor = torch.from_numpy(waveform).float()
        return tensor, self.sample_rate, meta


def load_segments(metadata_path: Path) -> List[SegmentMetadata]:
    """Load the dataset metadata file."""

    segments: List[SegmentMetadata] = []
    with metadata_path.open("r", encoding="utf8") as handle:
        for line in handle:
            parts = line.strip().split("|")
            if len(parts) < 3:
                continue
            file_path = Path(parts[0])
            speaker = parts[1]
            duration = float(parts[2])
            text = parts[3] if len(parts) > 3 else None
            segments.append(SegmentMetadata(file_path, speaker, duration, text))
    return segments


def filter_segments(
    segments: Iterable[SegmentMetadata],
    cfg: DatasetConfig,
) -> List[SegmentMetadata]:
    """Filter the segments to meet duration and dataset size targets."""

    filtered: List[SegmentMetadata] = []
    total_duration = 0.0
    for segment in segments:
        if not (cfg.min_segment_duration <= segment.duration <= cfg.max_segment_duration):
            continue
        filtered.append(segment)
        total_duration += segment.duration
        if cfg.max_files and len(filtered) >= cfg.max_files:
            break
        if 3600.0 <= total_duration <= 7200.0:
            continue
    return filtered


__all__ = ["VoiceDataset", "SegmentMetadata", "load_segments", "filter_segments"]

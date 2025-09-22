"""Loss functions for Project VOICE."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


def stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: Iterable[int] = (1024, 2048, 512),
    hop_sizes: Iterable[int] = (120, 240, 50),
    win_lengths: Iterable[int] = (600, 1200, 240),
) -> torch.Tensor:
    total = 0.0
    for fft, hop, win in zip(fft_sizes, hop_sizes, win_lengths):
        pred_spec = torch.stft(
            prediction,
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            return_complex=True,
        )
        target_spec = torch.stft(
            target,
            n_fft=fft,
            hop_length=hop,
            win_length=win,
            return_complex=True,
        )
        total = total + F.l1_loss(pred_spec.abs(), target_spec.abs())
    return total / len(tuple(fft_sizes))


def mel_loss(prediction: torch.Tensor, target: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=48_000,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=n_mels,
    ).to(prediction.device)
    mel_pred = mel_transform(prediction)
    mel_target = mel_transform(target)
    return F.l1_loss(mel_pred, mel_target)


class FeatureMatchingLoss(nn.Module):
    def forward(self, features_fake: List[torch.Tensor], features_real: List[torch.Tensor]) -> torch.Tensor:
        losses = []
        for fake_layer, real_layer in zip(features_fake, features_real):
            losses.append(F.l1_loss(fake_layer, real_layer))
        return sum(losses) / len(losses)


class HingeGanLoss(nn.Module):
    def generator_loss(self, logits_fake: torch.Tensor) -> torch.Tensor:
        return -logits_fake.mean()

    def discriminator_loss(self, logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
        loss_real = torch.mean(F.relu(1.0 - logits_real))
        loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        return loss_real + loss_fake


class F0RmseLoss(nn.Module):
    def forward(self, f0_pred: torch.Tensor, f0_target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean((f0_pred - f0_target) ** 2))


class FormantConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=48_000,
            n_fft=2048,
            hop_length=256,
            n_mels=40,
            f_min=90.0,
            f_max=11_000.0,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mel_pred = self.mel(prediction)
        mel_target = self.mel(target)
        return F.mse_loss(mel_pred, mel_target)


__all__ = [
    "stft_loss",
    "mel_loss",
    "FeatureMatchingLoss",
    "HingeGanLoss",
    "F0RmseLoss",
    "FormantConsistencyLoss",
]

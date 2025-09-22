"""Training loop for Project VOICE."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch import optim
from torch.cuda import amp
from torch.utils.data import DataLoader

from .config import ProjectVoiceConfig
from .dataset import SegmentMetadata, VoiceDataset
from .losses import (
    FeatureMatchingLoss,
    F0RmseLoss,
    FormantConsistencyLoss,
    HingeGanLoss,
    mel_loss,
    stft_loss,
)
from .models.encoders import HuBERTSoftEncoder
from .models.pitch import PitchExtractor
from .models.rvc import RVCGenerator
from .models.vocoder import Vocoder, VocoderConfig
from .models.retrieval import RetrievalBank, RetrievalBankConfig


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stacks = torch.nn.ModuleList()
        for _ in range(3):
            stack = torch.nn.ModuleDict(
                {
                    "block": torch.nn.Sequential(
                        torch.nn.Conv1d(1, 64, 15, stride=1, padding=7),
                        torch.nn.LeakyReLU(0.2),
                        torch.nn.Conv1d(64, 128, 41, stride=4, groups=4, padding=20),
                        torch.nn.LeakyReLU(0.2),
                        torch.nn.Conv1d(128, 256, 41, stride=4, groups=16, padding=20),
                        torch.nn.LeakyReLU(0.2),
                        torch.nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20),
                        torch.nn.LeakyReLU(0.2),
                        torch.nn.Conv1d(512, 512, 5, stride=1, padding=2),
                        torch.nn.LeakyReLU(0.2),
                    ),
                    "final": torch.nn.Conv1d(512, 1, 3, stride=1, padding=1),
                }
            )
            self.stacks.append(stack)

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        logits: List[torch.Tensor] = []
        features: List[torch.Tensor] = []
        x_in = x
        for stack in self.stacks:
            feature = stack["block"](x_in)
            logit = stack["final"](feature)
            logits.append(logit)
            features.append(feature)
            x_in = F.avg_pool1d(x_in, kernel_size=4, stride=2, padding=1)
        return logits, features


def _utmos_placeholder(waveform: torch.Tensor) -> torch.Tensor:
    """Placeholder for UTMOS scoring."""

    return torch.rand(1, device=waveform.device) * 5.0


class ProjectVoiceTrainer:
    def __init__(
        self,
        cfg: ProjectVoiceConfig,
        segments: List[SegmentMetadata],
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.device = device

        dataset = VoiceDataset(segments, sample_rate=cfg.dataset.sample_rate)
        self.loader = DataLoader(
            dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            drop_last=True,
        )

        self.content_encoder = HuBERTSoftEncoder(device)
        self.pitch_extractor = PitchExtractor(
            cfg.pitch.rmvpe_checkpoint,
            crepe_model_capacity=cfg.pitch.crepe_model_capacity,
            use_gpu=cfg.pitch.use_gpu,
        )
        self.generator = RVCGenerator(
            content_dim=cfg.model.content_dim,
            channels=cfg.model.generator_channels,
            strides=cfg.model.generator_strides,
            upsample_initial_channel=cfg.model.upsample_initial_channel,
            mel_channels=cfg.model.mel_channels,
        ).to(device)
        self.vocoder = Vocoder(
            VocoderConfig(
                use_ns_fhifigan=cfg.model.use_ns_fhifigan,
                use_bigvgan=cfg.model.use_bigvgan_vocoder,
            ),
            device,
        )
        self.retrieval = RetrievalBank(RetrievalBankConfig(), device)

        self.discriminator = MultiScaleDiscriminator().to(device)

        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=cfg.training.learning_rate,
            betas=cfg.training.betas,
            weight_decay=cfg.training.weight_decay,
        )
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=cfg.training.learning_rate,
            betas=cfg.training.betas,
            weight_decay=cfg.training.weight_decay,
        )

        self.gan_loss = HingeGanLoss()
        self.feature_matching = FeatureMatchingLoss()
        self.f0_loss = F0RmseLoss()
        self.formant_loss = FormantConsistencyLoss().to(device)
        self.scaler = amp.GradScaler(enabled=cfg.training.mixed_precision)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.dataset.sample_rate,
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            n_mels=80,
        ).to(device)

    def train_step(self, batch: Tuple[torch.Tensor, Sequence[int], Sequence[SegmentMetadata]]) -> dict:
        waveforms, _, _ = batch
        waveforms = waveforms.to(self.device)
        content = self.content_encoder(waveforms).transpose(1, 2)
        f0 = []
        for waveform in waveforms:
            waveform_np = waveform.cpu().numpy()
            f0.append(torch.from_numpy(self.pitch_extractor(waveform_np, self.cfg.dataset.sample_rate)))
        f0 = torch.stack([item.float().to(self.device) for item in f0])
        f0 = F.interpolate(f0.unsqueeze(1), size=content.shape[-1], mode="linear", align_corners=True).squeeze(1)

        waveforms_chn = waveforms.unsqueeze(1)

        with amp.autocast(enabled=self.cfg.training.mixed_precision):
            generated_mel = self.generator(content, f0)
            audio_wave = self.vocoder(generated_mel).squeeze(1)
            stft = stft_loss(audio_wave, waveforms)
            mel = mel_loss(audio_wave, waveforms)
            mel_target = self.mel_transform(waveforms)
            mel_generated = generated_mel
            self.retrieval.add(mel_target.detach())
            retrieval_similarity = self.retrieval.similarity(mel_generated.detach())
            fake_logits, fake_features = self.discriminator(audio_wave.unsqueeze(1))
            real_logits, real_features = self.discriminator(waveforms_chn)

            gan = self.gan_loss.generator_loss(torch.cat([log.mean(dim=-1) for log in fake_logits], dim=0))
            fm = self.feature_matching(fake_features, real_features)
            generated_audio = audio_wave.detach().cpu().numpy()
            f0_pred = [
                torch.from_numpy(self.pitch_extractor(sample.squeeze(), self.cfg.dataset.sample_rate)).float()
                for sample in generated_audio
            ]
            f0_pred = torch.stack([item.to(self.device) for item in f0_pred])
            f0_pred = F.interpolate(
                f0_pred.unsqueeze(1), size=f0.shape[-1], mode="linear", align_corners=True
            ).squeeze(1)
            f0_rmse = self.f0_loss(f0_pred, f0)
            formant = self.formant_loss(audio_wave, waveforms)
            loss_g = (
                self.cfg.losses.stft_weight * stft
                + self.cfg.losses.mel_weight * mel
                + self.cfg.losses.gan_weight * gan
                + self.cfg.losses.feature_matching_weight * fm
                + self.cfg.losses.f0_weight * f0_rmse
                + self.cfg.losses.formant_weight * formant
                + 0.1 * (1.0 - retrieval_similarity.mean())
            )

        self.optimizer_g.zero_grad(set_to_none=True)
        self.scaler.scale(loss_g).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        # Discriminator update
        with torch.no_grad():
            generated_detached = audio_wave.unsqueeze(1).detach()
        fake_logits, _ = self.discriminator(generated_detached)
        real_logits, _ = self.discriminator(waveforms_chn)
        loss_d = self.gan_loss.discriminator_loss(
            torch.cat([log.mean(dim=-1) for log in real_logits], dim=0),
            torch.cat([log.mean(dim=-1) for log in fake_logits], dim=0),
        )
        self.optimizer_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.optimizer_d.step()

        utmos_score = _utmos_placeholder(generated_detached)

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "stft": stft.item(),
            "mel": mel.item(),
            "gan": gan.item(),
            "fm": fm.item(),
            "f0_rmse": f0_rmse.item(),
            "formant": formant.item(),
            "retrieval": retrieval_similarity.mean().item(),
            "utmos": float(utmos_score.item()),
        }

    def train(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        step = 0
        for epoch in range(1000):
            for batch in self.loader:
                metrics = self.train_step(batch)
                step += 1
                if step % self.cfg.training.eval_interval == 0:
                    if metrics["utmos"] >= self.cfg.training.utmos_threshold and metrics["f0_rmse"] <= self.cfg.training.f0_rmse_threshold:
                        self._save_checkpoint(output_dir, step)
                        return
                if step >= self.cfg.training.max_steps:
                    self._save_checkpoint(output_dir, step)
                    return

    def _save_checkpoint(self, output_dir: Path, step: int) -> None:
        checkpoint = {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
        }
        torch.save(checkpoint, output_dir / f"checkpoint_{step}.pt")


__all__ = ["ProjectVoiceTrainer"]

"""Real-time inference utilities for Project VOICE."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

from .config import InferenceConfig, Preset, ProjectVoiceConfig
from .models.encoders import HuBERTSoftEncoder
from .models.pitch import PitchExtractor
from .models.rvc import RVCGenerator
from .models.vocoder import Vocoder, VocoderConfig
from .utils.audio import (
    add_breathiness,
    brickwall_limiter,
    de_ess,
    formant_shift,
    harmonic_tilt,
    pitch_shift,
)


@dataclass
class InferenceState:
    generator: RVCGenerator
    encoder: HuBERTSoftEncoder
    pitch: PitchExtractor
    vocoder: Vocoder
    device: torch.device


class RealTimeEngine:
    def __init__(self, cfg: ProjectVoiceConfig, checkpoint: Path) -> None:
        device = torch.device(cfg.inference.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device
        self.cfg = cfg
        self.generator = RVCGenerator(
            content_dim=cfg.model.content_dim,
            channels=cfg.model.generator_channels,
            strides=cfg.model.generator_strides,
            upsample_initial_channel=cfg.model.upsample_initial_channel,
            mel_channels=cfg.model.mel_channels,
        ).to(device)
        state = torch.load(checkpoint, map_location=device)
        self.generator.load_state_dict(state["generator"])
        self.generator.eval()
        self.encoder = HuBERTSoftEncoder(device)
        self.pitch = PitchExtractor(cfg.pitch.rmvpe_checkpoint)
        self.vocoder = Vocoder(
            VocoderConfig(
                use_ns_fhifigan=cfg.model.use_ns_fhifigan,
                use_bigvgan=cfg.model.use_bigvgan_vocoder,
            ),
            device,
        )

    def process_buffer(self, audio: np.ndarray, preset: Preset) -> np.ndarray:
        waveform = torch.from_numpy(audio).float().to(self.device)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.cfg.inference.fp16 and self.device.type == "cuda"):
            content = self.encoder(waveform).transpose(1, 2)
            f0 = torch.from_numpy(self.pitch(audio, self.cfg.inference.sample_rate)).float().to(self.device)
            f0 = F.interpolate(
                f0.unsqueeze(0).unsqueeze(0), size=content.shape[-1], mode="linear", align_corners=True
            ).squeeze(1)
            generated_mel = self.generator(content, f0)
            output_wave = self.vocoder(generated_mel).squeeze()
        output = output_wave.detach().cpu().numpy()
        output = pitch_shift(output, self.cfg.inference.sample_rate, preset.pitch_shift)
        output = formant_shift(output, self.cfg.inference.sample_rate, preset.formant_scale)
        output = de_ess(output, self.cfg.inference.sample_rate, preset.de_ess)
        output = add_breathiness(output, preset.breathiness)
        output = harmonic_tilt(output, self.cfg.inference.sample_rate, preset.harmonic_tilt)
        output = brickwall_limiter(output, preset.limiter_ceiling)
        return output

    def stream(self, preset: Preset) -> None:
        buffer_size = int(self.cfg.inference.sample_rate * (self.cfg.inference.frame_ms / 1000.0))

        def callback(indata, outdata, frames, time, status):  # type: ignore[override]
            if status:
                print(status)
            audio = indata[:, 0]
            processed = self.process_buffer(audio, preset)
            if len(processed) < frames:
                processed = np.pad(processed, (0, frames - len(processed)))
            outdata[:, 0] = processed[:frames]

        with sd.Stream(
            samplerate=self.cfg.inference.sample_rate,
            blocksize=buffer_size,
            dtype="float32",
            channels=1,
            callback=callback,
        ):
            threading.Event().wait()


__all__ = ["RealTimeEngine"]

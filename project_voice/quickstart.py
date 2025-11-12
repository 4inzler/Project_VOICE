"""Self-contained quickstart pipeline for Project VOICE."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import librosa
import numpy as np
import soundfile as sf

from .config import ProjectVoiceConfig
from .preprocess import preprocess_dataset
from .utils.audio import (
    add_breathiness,
    brickwall_limiter,
    de_ess,
    formant_shift,
    harmonic_tilt,
    pitch_shift,
)


def _synthesize_voice(sample_rate: int, seconds: float = 6.0) -> np.ndarray:
    """Generate a synthetic female-leaning vocal timbre for demos."""

    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    base = 0.55 * np.sin(2 * np.pi * 220.0 * t)
    harmonics = 0.25 * np.sin(2 * np.pi * 440.0 * t) + 0.12 * np.sin(2 * np.pi * 660.0 * t)
    vibrato = 0.03 * np.sin(2 * np.pi * 5.5 * t)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 2.5 * t)
    noise = 0.015 * np.random.randn(len(t))
    voice = (base + harmonics) * (1.0 + vibrato) * envelope + noise
    peak = np.max(np.abs(voice)) or 1.0
    return voice / peak


def _render_preset_effects(audio: np.ndarray, sample_rate: int, cfg: ProjectVoiceConfig) -> np.ndarray:
    """Apply the default preset DSP chain to a waveform."""

    preset = cfg.presets[0]
    processed = pitch_shift(audio, sample_rate, preset.pitch_shift)
    processed = formant_shift(processed, sample_rate, preset.formant_scale)
    processed = de_ess(processed, sample_rate, preset.de_ess)
    processed = add_breathiness(processed, preset.breathiness)
    processed = harmonic_tilt(processed, sample_rate, preset.harmonic_tilt)
    processed = brickwall_limiter(processed, preset.limiter_ceiling)
    return processed


def run_quickstart(output_dir: Path, sample_rate: int = 48_000) -> Dict[str, str]:
    """Produce a toy dataset, preprocess it, and export a converted sample."""

    output_dir = output_dir.expanduser().resolve()
    raw_dir = output_dir / "raw"
    workspace_dir = output_dir / "workspace"
    artifacts_dir = output_dir / "artifacts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    synthetic_audio = _synthesize_voice(sample_rate)
    raw_path = raw_dir / "synthetic_voice.wav"
    sf.write(raw_path, synthetic_audio, sample_rate)

    cfg = ProjectVoiceConfig()
    cfg.dataset.raw_audio_dir = raw_dir
    cfg.dataset.workspace_dir = workspace_dir
    cfg.dataset.sample_rate = sample_rate
    cfg.dataset.denoise = False
    cfg.dataset.watermark_ultrasonic = False

    segments = preprocess_dataset(cfg.dataset)
    if not segments:
        raise RuntimeError("Preprocessing did not yield any voice segments")

    segment_audio, _ = librosa.load(segments[0].file_path, sr=sample_rate)
    converted = _render_preset_effects(segment_audio, sample_rate, cfg)
    converted_path = artifacts_dir / "quickstart_converted.wav"
    sf.write(converted_path, converted, sample_rate)

    summary = {
        "raw_audio": str(raw_path),
        "workspace": str(workspace_dir),
        "metadata": str(workspace_dir / "metadata.json"),
        "converted_audio": str(converted_path),
        "preset": asdict(cfg.presets[0]),
    }
    (output_dir / "quickstart_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    return summary


__all__ = ["run_quickstart"]

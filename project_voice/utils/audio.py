"""Audio DSP helpers for Project VOICE."""
from __future__ import annotations

import numpy as np
import pyrubberband as pyrb
from scipy.signal import butter, lfilter


def pitch_shift(audio: np.ndarray, sample_rate: int, semitones: float) -> np.ndarray:
    """Pitch-shift the waveform using Rubber Band Library."""

    if np.isclose(semitones, 0.0):
        return audio
    return pyrb.pitch_shift(audio, sample_rate, semitones)


def formant_shift(audio: np.ndarray, sample_rate: int, ratio: float) -> np.ndarray:
    """Approximate formant lift using simple vocal-tract length perturbation."""

    if np.isclose(ratio, 1.0):
        return audio
    stretched = pyrb.time_stretch(audio, sample_rate, 1.0 / ratio)
    stretched = pyrb.pitch_shift(stretched, sample_rate, 12 * np.log2(ratio))
    min_len = min(len(audio), len(stretched))
    return stretched[:min_len]


def de_ess(audio: np.ndarray, sample_rate: int, amount: float) -> np.ndarray:
    cutoff = 6000.0
    nyquist = sample_rate / 2.0
    b, a = butter(2, cutoff / nyquist, btype="low")
    low = lfilter(b, a, audio)
    high = audio - low
    high *= 1.0 - amount
    return low + high


def add_breathiness(audio: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return audio
    noise = np.random.randn(len(audio)) * amount * 0.02
    return audio + noise


def harmonic_tilt(audio: np.ndarray, sample_rate: int, tilt: float) -> np.ndarray:
    if np.isclose(tilt, 0.0):
        return audio
    cutoff = 2000.0
    nyquist = sample_rate / 2.0
    b, a = butter(1, cutoff / nyquist, btype="high")
    high = lfilter(b, a, audio)
    return audio + tilt * high


def brickwall_limiter(audio: np.ndarray, ceiling_db: float) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20)
    return np.clip(audio, -ceiling, ceiling)


__all__ = [
    "pitch_shift",
    "formant_shift",
    "de_ess",
    "add_breathiness",
    "harmonic_tilt",
    "brickwall_limiter",
]

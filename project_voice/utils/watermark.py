"""Optional ultrasonic watermark injection."""
from __future__ import annotations

import numpy as np


def inject_ultrasonic_watermark(
    audio: np.ndarray,
    sample_rate: int,
    ultrasonic_hz: float,
    level_db: float,
) -> np.ndarray:
    """Inject a low-amplitude ultrasonic tone as a watermark."""

    duration = len(audio) / sample_rate
    t = np.linspace(0, duration, num=len(audio), endpoint=False)
    tone = np.sin(2 * np.pi * ultrasonic_hz * t)
    amplitude = 10 ** (level_db / 20)
    return audio + amplitude * tone


__all__ = ["inject_ultrasonic_watermark"]

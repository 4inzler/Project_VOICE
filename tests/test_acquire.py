import math
from pathlib import Path

import pytest

from project_voice.data.acquire import _normalise_audio


def test_normalise_audio_resamples_and_normalises(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")

    sample_rate = 22_050
    duration_seconds = 0.1
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    wave = np.stack([
        np.sin(2 * math.pi * 440 * t),
        np.sin(2 * math.pi * 660 * t),
    ], axis=1)

    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    sf.write(source, wave, sample_rate)

    _normalise_audio(source, target, 48_000)

    audio, sr = sf.read(target)

    assert sr == 48_000
    assert audio.ndim == 1
    assert audio.size > wave.shape[0]
    peak = float(np.max(np.abs(audio)))
    assert 0.9 <= peak <= 1.0 + 1e-6

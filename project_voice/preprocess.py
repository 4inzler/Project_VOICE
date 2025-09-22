"""Audio preprocessing pipeline for Project VOICE."""
from __future__ import annotations

import json
import random
from typing import Iterable, List

import librosa
import noisereduce as nr
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import webrtcvad
from tqdm import tqdm

from .config import DatasetConfig
from .dataset import SegmentMetadata
from .utils.audio import formant_shift, pitch_shift
from .utils.guardrails import enforce_guardrails
from .utils.watermark import inject_ultrasonic_watermark


def _normalize_lufs(audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(audio)
    gain = target_lufs - loudness
    factor = 10 ** (gain / 20)
    return audio * factor


def _denoise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return nr.reduce_noise(y=audio, sr=sample_rate)


def _vad_segment(
    audio: np.ndarray,
    sample_rate: int,
    cfg: DatasetConfig,
) -> Iterable[np.ndarray]:
    vad = webrtcvad.Vad(cfg.vad_aggressiveness)
    frame_length = int(sample_rate * (cfg.vad_window_ms / 1000.0))
    frame_length = max(frame_length, 160)
    hop = frame_length // 2
    voiced_frames: List[np.ndarray] = []
    current_segment: List[np.ndarray] = []
    for start in range(0, len(audio) - frame_length, hop):
        frame = audio[start : start + frame_length]
        frame_bytes = (frame * 32768.0).astype(np.int16).tobytes()
        is_voiced = vad.is_speech(frame_bytes, sample_rate)
        if is_voiced:
            current_segment.append(frame)
        elif current_segment:
            voiced_frames.append(np.concatenate(current_segment))
            current_segment = []
    if current_segment:
        voiced_frames.append(np.concatenate(current_segment))
    for voiced in voiced_frames:
        duration = len(voiced) / sample_rate
        if cfg.min_segment_duration <= duration <= cfg.max_segment_duration:
            yield voiced


def preprocess_dataset(cfg: DatasetConfig) -> List[SegmentMetadata]:
    enforce_guardrails(cfg.workspace_dir)
    cfg.workspace_dir.mkdir(parents=True, exist_ok=True)
    metadata: List[SegmentMetadata] = []

    audio_files = list(cfg.raw_audio_dir.glob("**/*.wav"))
    random.shuffle(audio_files)
    if cfg.max_files:
        audio_files = audio_files[: cfg.max_files]

    for audio_path in tqdm(audio_files, desc="Preprocessing"):
        audio, sr = librosa.load(audio_path, sr=cfg.sample_rate)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        audio = _normalize_lufs(audio, cfg.sample_rate, cfg.target_lufs)
        if cfg.denoise:
            audio = _denoise(audio, cfg.sample_rate)

        segments = list(_vad_segment(audio, cfg.sample_rate, cfg))
        for idx, segment in enumerate(segments):
            duration = len(segment) / cfg.sample_rate
            shifted = pitch_shift(segment, cfg.sample_rate, random.uniform(*cfg.pitch_shift_range))
            lifted = formant_shift(shifted, cfg.sample_rate, random.uniform(*cfg.formant_lift_range))
            if cfg.watermark_ultrasonic:
                lifted = inject_ultrasonic_watermark(
                    lifted,
                    cfg.sample_rate,
                    ultrasonic_hz=19_500.0,
                    level_db=-35.0,
                )
            out_dir = cfg.workspace_dir / "segments"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{audio_path.stem}_{idx:04d}.wav"
            sf.write(out_path, lifted, cfg.sample_rate)
            metadata.append(
                SegmentMetadata(
                    file_path=out_path,
                    speaker="target",
                    duration=duration,
                    text=None,
                )
            )
    metadata_path = cfg.workspace_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf8") as handle:
        json.dump(
            [
                {
                    "path": str(item.file_path),
                    "speaker": item.speaker,
                    "duration": item.duration,
                    "text": item.text,
                }
                for item in metadata
            ],
            handle,
            indent=2,
        )
    return metadata


__all__ = ["preprocess_dataset"]

"""Configuration dataclasses for Project VOICE training and inference."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DatasetConfig:
    """Configuration for preparing the single-speaker dataset."""

    raw_audio_dir: Path = Path("data/raw")
    workspace_dir: Path = Path("data/workspace")
    sample_rate: int = 48_000
    target_lufs: float = -16.0
    vad_window_ms: int = 30
    vad_energy_threshold: float = 0.6
    min_segment_duration: float = 1.0
    max_segment_duration: float = 12.0
    denoise: bool = True
    vad_aggressiveness: int = 2
    max_files: Optional[int] = None
    formant_lift_range: Tuple[float, float] = (1.06, 1.10)
    pitch_shift_range: Tuple[float, float] = (2.0, 3.0)
    watermark_ultrasonic: bool = False


@dataclass
class ModelConfig:
    """Top-level model hyper-parameters for the RVC stack."""

    content_encoder_name: str = "hubert-soft"
    content_dim: int = 768
    use_f0_guidance: bool = True
    generator_channels: int = 256
    generator_n_layers: int = 4
    generator_kernel_size: int = 7
    generator_strides: Tuple[int, ...] = (2, 2, 2, 2)
    upsample_initial_channel: int = 512
    mel_channels: int = 80
    use_ns_fhifigan: bool = True
    use_bigvgan_vocoder: bool = False
    spectral_subbands: int = 4
    noise_scale: float = 0.1


@dataclass
class PitchConfig:
    """Pitch extraction configuration."""

    rmvpe_checkpoint: Optional[Path] = None
    crepe_model_capacity: str = "tiny"
    f0_fallback_confidence: float = 0.6
    use_gpu: bool = True


@dataclass
class LossConfig:
    """Loss weights used during training."""

    stft_weight: float = 1.0
    mel_weight: float = 1.0
    gan_weight: float = 1.0
    feature_matching_weight: float = 2.0
    f0_weight: float = 1.0
    formant_weight: float = 0.5


@dataclass
class TrainingConfig:
    """Optimizer, scheduler and runtime configuration."""

    batch_size: int = 8
    num_workers: int = 8
    learning_rate: float = 2e-4
    betas: Tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 5000
    max_steps: int = 200_000
    gradient_accumulation: int = 1
    mixed_precision: bool = True
    seed: int = 1234
    utmos_threshold: float = 4.0
    f0_rmse_threshold: float = 30.0
    eval_interval: int = 1000
    save_interval: int = 5000
    max_checkpoints: int = 5


@dataclass
class InferenceConfig:
    """Real-time inference configuration."""

    sample_rate: int = 48_000
    frame_ms: int = 20
    look_ahead_ms: int = 6
    hop_length: int = 960
    block_size: int = 1024
    device: Optional[str] = None
    fp16: bool = True
    preset_pitch_shift: float = 2.5
    preset_formant_scale: float = 1.08
    preset_breathiness: float = 0.15
    preset_de_ess: float = 0.35
    preset_harmonic_tilt: float = 0.1
    preset_limiter: float = -1.0


@dataclass
class Preset:
    """User preset for inference."""

    name: str
    pitch_shift: float
    formant_scale: float
    breathiness: float
    de_ess: float
    harmonic_tilt: float
    limiter_ceiling: float


@dataclass
class GuardrailConfig:
    """Ethical guardrail options."""

    block_celebrities: bool = True
    consent_required: bool = True
    warning_message: str = (
        "This system is intended only for consensual voice conversion. "
        "Do not attempt to recreate voices without explicit permission."
    )
    ultrasonic_watermark_hz: float = 19_500.0
    ultrasonic_level_db: float = -35.0


@dataclass
class ProjectVoiceConfig:
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            raw_audio_dir=Path("data/raw"),
            workspace_dir=Path("data/workspace"),
        )
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    presets: List[Preset] = field(
        default_factory=lambda: [
            Preset(
                name="Default Feminine",
                pitch_shift=2.5,
                formant_scale=1.08,
                breathiness=0.15,
                de_ess=0.35,
                harmonic_tilt=0.1,
                limiter_ceiling=-1.0,
            )
        ]
    )
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)


__all__ = [
    "DatasetConfig",
    "ModelConfig",
    "PitchConfig",
    "LossConfig",
    "TrainingConfig",
    "InferenceConfig",
    "Preset",
    "GuardrailConfig",
    "ProjectVoiceConfig",
]

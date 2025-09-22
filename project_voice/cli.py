"""Typer-based CLI for Project VOICE."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import ProjectVoiceConfig
from .data import acquire_dataset, list_datasets
from .dataset import SegmentMetadata, filter_segments
from .preprocess import preprocess_dataset
from .quickstart import run_quickstart
from .utils.guardrails import validate_prompt

app = typer.Typer(help="Project VOICE command line interface")
dataset_app = typer.Typer(help="Utilities for sourcing training audio")
app.add_typer(dataset_app, name="dataset")


@dataset_app.command("list")
def dataset_list() -> None:
    """List available openly licensed datasets."""

    for dataset in list_datasets():
        typer.echo(
            f"{dataset.name}: {dataset.speaker} (~{dataset.minutes:.0f} min, {dataset.license})\n"
            f"  {dataset.description}"
        )


@dataset_app.command("acquire")
def dataset_acquire(
    name: str = typer.Option("cmu_arctic_slt", help="Dataset identifier"),
    raw_audio: Path = typer.Option(..., help="Destination directory for raw WAVs"),
    cache_dir: Optional[Path] = typer.Option(None, help="Cache directory for archives"),
    sample_rate: int = typer.Option(48_000, help="Target sample rate"),
    skip_existing: bool = typer.Option(True, help="Skip WAVs that already exist"),
) -> None:
    """Download and normalise an open female voice dataset."""

    acquire_dataset(name, raw_audio, cache_dir=cache_dir, sample_rate=sample_rate, skip_existing=skip_existing)
    typer.echo(f"Dataset '{name}' prepared in {raw_audio}")


@app.command()
def preprocess(
    raw_audio: Path = typer.Option(..., exists=True, file_okay=False),
    workspace: Path = typer.Option(..., file_okay=False),
) -> None:
    """Run dataset preprocessing with VAD, LUFS normalization and augmentation."""

    cfg = ProjectVoiceConfig()
    cfg.dataset.raw_audio_dir = raw_audio
    cfg.dataset.workspace_dir = workspace
    metadata = preprocess_dataset(cfg.dataset)
    typer.echo(f"Preprocessed {len(metadata)} segments into {workspace}")


@app.command()
def train(
    metadata_path: Path = typer.Option(..., exists=True),
    output_dir: Path = typer.Option(...),
    device_str: Optional[str] = typer.Option(None, help="Torch device, e.g., cuda:0"),
) -> None:
    """Train the Project VOICE model."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in runtime environments
        raise typer.BadParameter(
            "Training requires PyTorch and related extras. Install with `pip install .[full]`."
        ) from exc

    from .trainer import ProjectVoiceTrainer

    cfg = ProjectVoiceConfig()
    segments_json = json.loads(metadata_path.read_text(encoding="utf8"))
    segments = [
        SegmentMetadata(Path(item["path"]), item["speaker"], item["duration"], item.get("text"))
        for item in segments_json
    ]
    filtered = filter_segments(segments, cfg.dataset)
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    trainer = ProjectVoiceTrainer(cfg, filtered, device)
    trainer.train(output_dir)


@app.command()
def realtime(
    checkpoint: Path = typer.Option(..., exists=True),
    preset_name: str = typer.Option("Default Feminine"),
    device_str: Optional[str] = typer.Option(None, help="Torch device"),
) -> None:
    """Launch the real-time inference engine."""

    try:
        import torch  # noqa: F401  # imported for side effects / availability
    except ImportError as exc:  # pragma: no cover
        raise typer.BadParameter(
            "Real-time inference requires the 'full' extras. Install with `pip install .[full]`."
        ) from exc

    from .inference import RealTimeEngine

    cfg = ProjectVoiceConfig()
    preset = next((preset for preset in cfg.presets if preset.name == preset_name), None)
    if preset is None:
        raise typer.BadParameter(f"Unknown preset {preset_name}")
    if device_str:
        cfg.inference.device = device_str
    engine = RealTimeEngine(cfg, checkpoint)
    engine.stream(preset)


@app.command()
def quickstart(
    output_dir: Path = typer.Option(Path("quickstart"), file_okay=False, help="Where to place demo artifacts"),
    sample_rate: int = typer.Option(48_000, help="Sample rate for the synthetic demo"),
) -> None:
    """Generate a synthetic dataset and export a converted demo clip."""

    summary = run_quickstart(output_dir, sample_rate=sample_rate)
    typer.echo("Quickstart assets created:")
    for key, value in summary.items():
        typer.echo(f"  {key}: {value}")


@app.command()
def guardrails(prompt: str) -> None:
    """Validate that a requested prompt passes celebrity filters."""

    if validate_prompt(prompt):
        typer.echo("Prompt accepted")
    else:
        typer.echo("Prompt rejected: references disallowed individual")


if __name__ == "__main__":
    app()

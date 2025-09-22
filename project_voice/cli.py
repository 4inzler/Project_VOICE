"""Typer-based CLI for Project VOICE."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import typer

from .config import ProjectVoiceConfig
from .dataset import SegmentMetadata, filter_segments
from .inference import RealTimeEngine
from .preprocess import preprocess_dataset
from .trainer import ProjectVoiceTrainer
from .utils.guardrails import validate_prompt

app = typer.Typer(help="Project VOICE command line interface")


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

    cfg = ProjectVoiceConfig()
    preset = next((preset for preset in cfg.presets if preset.name == preset_name), None)
    if preset is None:
        raise typer.BadParameter(f"Unknown preset {preset_name}")
    if device_str:
        cfg.inference.device = device_str
    engine = RealTimeEngine(cfg, checkpoint)
    engine.stream(preset)


@app.command()
def guardrails(prompt: str) -> None:
    """Validate that a requested prompt passes celebrity filters."""

    if validate_prompt(prompt):
        typer.echo("Prompt accepted")
    else:
        typer.echo("Prompt rejected: references disallowed individual")


if __name__ == "__main__":
    app()

"""Utilities for downloading consented open female voice datasets."""
from __future__ import annotations

import json
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import librosa
import requests
import soundfile as sf
from tqdm import tqdm

CHUNK_SIZE = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DatasetSource:
    """Metadata for an openly licensed dataset."""

    name: str
    url: str
    archive_type: str
    speaker: str
    minutes: float
    license: str
    description: str
    citation: Optional[str] = None


_DATASETS: Dict[str, DatasetSource] = {
    "cmu_arctic_slt": DatasetSource(
        name="cmu_arctic_slt",
        url="http://festvox.org/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.tar.bz2",
        archive_type="tar",
        speaker="CMU Arctic SLT (female, US English)",
        minutes=70.0,
        license="BSD-like (CMU Arctic)",
        description=(
            "~1 hour of studio-quality female speech from the CMU Arctic dataset. "
            "This speaker (SLT) aligns with the Project VOICE consent requirements."
        ),
        citation="@inproceedings{kominek2004cmu, title={The CMU Arctic speech databases}, year={2004}}",
    ),
    "ljspeech": DatasetSource(
        name="ljspeech",
        url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        archive_type="tar",
        speaker="LJ Speech (female, US English)",
        minutes=840.0,
        license="CC BY 4.0",
        description=(
            "13 hours of audiobook narration from a single consenting speaker. "
            "Suitable for downsampling and selective use (60–120 minutes recommended)."
        ),
        citation="@misc{ljspeech17, author={Ito, Keith}, title={The LJ Speech Dataset}, year={2017}}",
    ),
}


def list_datasets() -> Iterable[DatasetSource]:
    """Return available dataset metadata."""

    return _DATASETS.values()


def _download(url: str, destination: Path) -> None:
    """Download a URL with progress reporting."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {destination.name}")
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
        progress.close()


def _extract(archive_path: Path, destination: Path, archive_type: str) -> Path:
    """Extract a compressed archive and return the root directory."""

    destination.mkdir(parents=True, exist_ok=True)
    extract_root = destination / archive_path.stem
    if extract_root.exists():
        return extract_root

    if archive_type == "tar":
        mode = "r:*"
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(path=extract_root)
    elif archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(path=extract_root)
    else:
        raise ValueError(f"Unsupported archive type: {archive_type}")
    return extract_root


def _iter_wavs(directory: Path) -> Iterator[Path]:
    """Yield WAV files within a directory tree."""

    for path in directory.rglob("*.wav"):
        if path.is_file():
            yield path


def _normalise_audio(source_path: Path, target_path: Path, sample_rate: int) -> None:
    """Load, resample, and save an audio file as mono 48 kHz WAV."""

    audio, _ = librosa.load(source_path, sr=sample_rate, mono=True)
    sf.write(target_path, audio, sample_rate)


def _write_metadata(raw_audio_dir: Path, dataset: DatasetSource) -> None:
    """Record dataset provenance and consent reminder."""

    metadata = {
        "dataset": dataset.name,
        "speaker": dataset.speaker,
        "minutes": dataset.minutes,
        "license": dataset.license,
        "description": dataset.description,
        "citation": dataset.citation,
    }
    (raw_audio_dir / "SOURCE.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    (raw_audio_dir / "CONSENT.txt").write_text(
        (
            "Audio sourced from an openly licensed dataset with consent for research use.\n"
            "Refer to SOURCE.json for licensing terms."
        ),
        encoding="utf8",
    )


def acquire_dataset(
    name: str,
    raw_audio_dir: Path,
    cache_dir: Optional[Path] = None,
    sample_rate: int = 48_000,
    skip_existing: bool = True,
) -> None:
    """Download and prepare a dataset into the raw audio directory."""

    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {', '.join(_DATASETS)}")

    dataset = _DATASETS[name]
    cache_dir = cache_dir or Path.home() / ".cache" / "project_voice"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_audio_dir.mkdir(parents=True, exist_ok=True)

    archive_name = Path(dataset.url).name
    archive_path = cache_dir / archive_name
    if not archive_path.exists():
        _download(dataset.url, archive_path)

    extract_root = _extract(archive_path, cache_dir, dataset.archive_type)

    wav_files = list(_iter_wavs(extract_root))
    if not wav_files:
        raise RuntimeError(f"No WAV files found after extracting {dataset.name}")

    for wav_path in tqdm(wav_files, desc="Preparing audio", unit="file"):
        target_path = raw_audio_dir / wav_path.name
        if skip_existing and target_path.exists():
            continue
        _normalise_audio(wav_path, target_path, sample_rate)

    _write_metadata(raw_audio_dir, dataset)


__all__ = ["DatasetSource", "acquire_dataset", "list_datasets"]

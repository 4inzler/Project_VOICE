# Project VOICE

Project VOICE delivers an Arch Linux-oriented Real-Time Voice Conversion (RVC) pipeline tuned for an NVIDIA RTX 4070 SUPER (12 GB) GPU paired with an AMD Ryzen 7 7700 CPU. It targets high-quality feminine voice conversion from a single consenting speaker using 60–120 minutes of 48 kHz audio.

## Features

- **HuBERT-soft content encoder** with RMVPE pitch estimation and CREPE-tiny fallback.
- **F0-guided RVC generator** coupled with an NSF-HiFiGAN vocoder (BigVGAN optional).
- **ECAPA-TDNN retrieval bank** to improve speaker similarity.
- **Loss cocktail**: multi-scale STFT, mel, GAN hinge, feature matching, F0 RMSE, and formant consistency.
- **Data preprocessing**: VAD segmentation, LUFS normalization, light denoise, stochastic formant lift (1.06–1.10×) and pitch prior (+2 to +3 semitones), optional ultrasonic watermark.
- **Early stopping** guided by UTMOS and F0 RMSE estimates.
- **ONNX/TensorRT export hooks** (via `pip install project-voice[deploy]`).
- **Real-time inference** defaults: 48 kHz, 20–32 ms frames, FP16, ~6 ms look-ahead. Preset: +2.5 st pitch, 1.08 formant scale, breathiness 0.15, de-ess 0.35, harmonic tilt +0.1, limiter −1 dBTP.
- **Typer CLI and Gradio GUI** with device selection, guardrails, and consent acknowledgement.

## Hardware assumptions

- Arch Linux or derivative with the latest NVIDIA drivers and CUDA toolkit.
- GPU: RTX 4070 SUPER 12 GB (FP16 acceleration recommended).
- CPU: Ryzen 7 7700 with AVX2/FMA support.
- Minimum 32 GB system RAM and fast NVMe storage.

## Installation

```bash
pacman -Syu python python-pip sox rubberband
pip install -e .
```

For ONNX/TensorRT export support:

```bash
pip install -e .[deploy]
```

## Dataset preparation

### Use an openly licensed baseline voice

If you do not have a consenting dataset prepared, Project VOICE can fetch one
for you. The CMU Arctic SLT speaker (~70 minutes of female US English speech)
is a good starting point:

```bash
project-voice dataset acquire --name cmu_arctic_slt --raw-audio data/raw/cmu_slt
```

To browse other options (e.g., the much longer LJ Speech corpus), run
`project-voice dataset list`. All downloads are cached in
`~/.cache/project_voice` and resampled to 48 kHz in the destination folder. The
command also writes `SOURCE.json` and `CONSENT.txt` to document provenance.

1. Gather 60–120 minutes of clean, single-speaker, **consensual** female audio at 48 kHz.
2. Place WAV files inside `data/raw/`.
3. Run preprocessing:

```bash
project-voice preprocess --raw-audio data/raw --workspace data/workspace
```

This performs LUFS normalization to −16 LUFS, optional denoise, VAD segmentation, pitch & formant augmentation, and optional ultrasonic watermarking. Processed clips live in `data/workspace/segments` with metadata in `metadata.json`.

## Training

```bash
project-voice train --metadata data/workspace/metadata.json --output-dir runs/exp1 --device-str cuda:0
```

Training uses mixed precision, RMVPE+CREPE F0 extraction, and the configured loss cocktail. Checkpoints are saved periodically; training stops when UTMOS ≥ 4.0 and F0 RMSE ≤ 30 Hz. Override the compute device with `--device-str` if needed.

## Real-time inference

```bash
project-voice realtime --checkpoint runs/exp1/checkpoint_5000.pt --preset-name "Default Feminine" --device-str cuda:0
```

The engine streams 20 ms frames with ~6 ms look-ahead, FP16 on GPU, and preset shaping (pitch +2.5 st, formant 1.08, etc.). Set `--device-str cpu` to force CPU inference.

## GUI

```bash
python -m project_voice.gui
```

The Gradio interface enforces consent acknowledgement and celebrity filtering before launching the stream.

## Ethical safeguards

- **Consent file**: preprocessing writes `CONSENT.txt` to the workspace.
- **Celebrity filter**: CLI & GUI reject prompts referencing pre-defined celebrity names.
- **Optional ultrasonic watermark**: embeds a −35 dB 19.5 kHz tone for provenance.

## Export

After installing the deploy extras, export to ONNX:

```python
import torch
from project_voice.inference import RealTimeEngine
from project_voice.config import ProjectVoiceConfig

cfg = ProjectVoiceConfig()
engine = RealTimeEngine(cfg, Path("runs/exp1/checkpoint_5000.pt"))
dummy = torch.randn(1, cfg.inference.sample_rate * cfg.inference.frame_ms // 1000)
torch.onnx.export(engine.generator, (dummy.to(engine.device), dummy.to(engine.device)), "generator.onnx")
```

TensorRT conversion can be handled with `trtexec --onnx=generator.onnx --fp16`.

## License & Use

This project is intended for ethical, consent-based voice conversion research. Do not use it to imitate non-consenting individuals or celebrities.

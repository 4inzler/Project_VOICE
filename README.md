# Project VOICE

Project VOICE delivers an Arch Linux-oriented Real-Time Voice Conversion (RVC) pipeline tuned for an NVIDIA RTX 4070 SUPER (12 GB) GPU paired with an AMD Ryzen 7 7700 CPU. It targets high-quality feminine voice conversion from a single consenting speaker using 60–120 minutes of 48 kHz audio.

## Features
- HuBERT-soft content encoder with RMVPE pitch estimation and CREPE-tiny fallback.
- F0-guided RVC generator coupled with an NSF-HiFiGAN vocoder (BigVGAN optional).
- ECAPA-TDNN retrieval bank to improve speaker similarity.
- Loss cocktail: multi-scale STFT, mel, GAN hinge, feature matching, F0 RMSE, and formant consistency.
- Data preprocessing: VAD segmentation, LUFS normalization, light denoise, stochastic formant lift (1.06–1.10×) and pitch prior (+2 to +3 semitones), optional ultrasonic watermark.
- Early stopping guided by UTMOS and F0 RMSE estimates.
- ONNX/TensorRT export hooks (via `pip install project-voice[deploy]`).
- Real-time inference defaults: 48 kHz, 20–32 ms frames, FP16, ~6 ms look-ahead. Preset: +2.5 st pitch, 1.08 formant scale, breathiness 0.15, de-ess 0.35, harmonic tilt +0.1, limiter −1 dBTP.
- Typer CLI and Gradio GUI with device selection, guardrails, and consent acknowledgement.

## Hardware assumptions
- Arch Linux or derivative with the latest NVIDIA drivers and CUDA toolkit.
- GPU: RTX 4070 SUPER 12 GB (FP16 acceleration recommended).
- CPU: Ryzen 7 7700 with AVX2/FMA support.
- Minimum 32 GB system RAM and fast NVMe storage.

## Installation
```sh
pacman -Syu python python-pip
pip install -e .
```

For the full GPU training and streaming toolchain (PyTorch, torchaudio, RMVPE, etc.) install the optional extras:
```sh
pip install -e .[full]
```

TensorRT/ONNX export helpers remain available via the deploy extra:
```sh
pip install -e .[deploy]
```

## Quickstart (no GPU required)
Want to see the signal chain in action without sourcing a dataset or configuring CUDA? Generate a synthetic voice demo:
```sh
project-voice quickstart --output-dir demo
```

The command synthesizes example content using the HuBERT-soft encoder, RMVPE/CREPE pitch pipeline, NSF-HiFiGAN vocoder, and formant/pitch augmentation so you can audition the system before providing your own data.

## Dataset acquisition
Inspect available datasets and download a consenting female speaker corpus (CMU Arctic SLT or LJ Speech) into a local folder:
```sh
project-voice dataset list
project-voice dataset acquire cmu-arctic-slt --output data/cmu_slt
```

Audio is resampled to 48 kHz, segmented with VAD, normalized to −23 LUFS, lightly denoised, and accompanied by a consent README.

## Training workflow
1. Prepare your dataset path (either acquired via the CLI or your own recordings with explicit consent).
2. Preprocess the audio with pitch/formant augmentation and metadata generation:
   ```sh
   project-voice preprocess --input data/cmu_slt --output data/cmu_slt_proc --preset arch-rtx4070s
   ```
3. Kick off training with mixed precision, RMVPE/CREPE pitch guidance, and the configured loss cocktail:
   ```sh
   project-voice train --config configs/arch-rtx4070s.yaml --data data/cmu_slt_proc
   ```
4. Monitor UTMOS and F0 RMSE metrics for early stopping.
5. Export checkpoints to ONNX or TensorRT for deployment:
   ```sh
   project-voice export --checkpoint checkpoints/latest.pth --format onnx --output exports/model.onnx
   project-voice export --checkpoint checkpoints/latest.pth --format tensorrt --output exports/model.plan
   ```

## Real-time inference
Launch the real-time streaming GUI or CLI with low-latency defaults tuned for the RTX 4070 SUPER:
```sh
project-voice gui --preset default
project-voice realtime --input-device 1 --output-device 3 --preset default
```

The default preset applies +2.5 semitone pitch shift, 1.08× formant lift, breathiness 0.15, de-ess 0.35, harmonic tilt +0.1, and a limiter at −1 dBTP. You can create custom presets in YAML to match different targets.

## Ethical guardrails
Project VOICE enforces consent reminders and blocks obviously disallowed celebrity voices by default. Optional ultrasonic watermarking tags exports for provenance. Always respect local laws and the preferences of the recorded speaker.

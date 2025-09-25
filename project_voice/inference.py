"""Real-time inference utilities for Project VOICE."""
from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F

from .config import Preset, ProjectVoiceConfig
from .models.encoders import HuBERTSoftEncoder
from .models.pitch import PitchExtractor
from .models.rvc import RVCGenerator
from .models.vocoder import Vocoder, VocoderConfig
from .utils.audio import (
    add_breathiness,
    brickwall_limiter,
    de_ess,
    formant_shift,
    harmonic_tilt,
    pitch_shift,
)


class RealTimeEngine:
    def __init__(self, cfg: ProjectVoiceConfig, checkpoint: Path) -> None:
        device = torch.device(cfg.inference.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = device
        self.cfg = cfg
        self.generator = RVCGenerator(
            content_dim=cfg.model.content_dim,
            channels=cfg.model.generator_channels,
            strides=cfg.model.generator_strides,
            upsample_initial_channel=cfg.model.upsample_initial_channel,
            mel_channels=cfg.model.mel_channels,
        ).to(device)
        state = torch.load(checkpoint, map_location=device)
        self.generator.load_state_dict(state["generator"])
        self.generator.eval()
        self.encoder = HuBERTSoftEncoder(device)
        self.pitch = PitchExtractor(cfg.pitch.rmvpe_checkpoint)
        self.vocoder = Vocoder(
            VocoderConfig(
                use_ns_fhifigan=cfg.model.use_ns_fhifigan,
                use_bigvgan=cfg.model.use_bigvgan_vocoder,
            ),
            device,
        )
        self._executor = ThreadPoolExecutor(max_workers=1)

    def process_buffer(self, audio: np.ndarray, preset: Preset) -> np.ndarray:
        waveform = torch.from_numpy(audio).float().to(self.device)
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=self.cfg.inference.fp16 and self.device.type == "cuda"):
            content = self.encoder(waveform).transpose(1, 2)
            f0 = torch.from_numpy(self.pitch(audio, self.cfg.inference.sample_rate)).float().to(self.device)
            f0 = F.interpolate(
                f0.unsqueeze(0).unsqueeze(0), size=content.shape[-1], mode="linear", align_corners=True
            ).squeeze(1)
            generated_mel = self.generator(content, f0)
            output_wave = self.vocoder(generated_mel).squeeze()
        output = output_wave.detach().cpu().numpy()
        output = pitch_shift(output, self.cfg.inference.sample_rate, preset.pitch_shift)
        output = formant_shift(output, self.cfg.inference.sample_rate, preset.formant_scale)
        output = de_ess(output, self.cfg.inference.sample_rate, preset.de_ess)
        output = add_breathiness(output, preset.breathiness)
        output = harmonic_tilt(output, self.cfg.inference.sample_rate, preset.harmonic_tilt)
        output = brickwall_limiter(output, preset.limiter_ceiling)
        return output.astype(np.float32)

    async def process_buffer_async(self, audio: np.ndarray, preset: Preset) -> np.ndarray:
        """Asynchronously run the conversion stack for a mono buffer."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.process_buffer, audio, preset)

    async def stream_async(self, preset: Preset) -> None:
        """Stream audio asynchronously with background inference workers."""

        loop = asyncio.get_running_loop()
        buffer_size = int(self.cfg.inference.sample_rate * (self.cfg.inference.frame_ms / 1000.0))
        input_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        output_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=8)
        stop_signal = threading.Event()

        def callback(indata, outdata, frames, time, status):  # type: ignore[override]
            if status:
                print(status)
            try:
                input_queue.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass
            try:
                processed = output_queue.get_nowait()
            except queue.Empty:
                processed = np.zeros(frames, dtype=np.float32)
            if len(processed) < frames:
                processed = np.pad(processed, (0, frames - len(processed)))
            outdata[:, 0] = processed[:frames]

        async def worker() -> None:
            try:
                while not stop_signal.is_set():
                    audio = await loop.run_in_executor(None, input_queue.get)
                    processed = await self.process_buffer_async(audio, preset)
                    await loop.run_in_executor(None, output_queue.put, processed)
            except asyncio.CancelledError:
                stop_signal.set()
                raise

        def audio_loop() -> None:
            with sd.Stream(
                samplerate=self.cfg.inference.sample_rate,
                blocksize=buffer_size,
                dtype="float32",
                channels=1,
                callback=callback,
            ):
                stop_signal.wait()

        worker_task = asyncio.create_task(worker())
        stream_task = asyncio.create_task(asyncio.to_thread(audio_loop))

        wait_forever = loop.create_future()
        try:
            await wait_forever
        except asyncio.CancelledError:
            pass
        finally:
            stop_signal.set()
            with suppress(queue.Full):
                input_queue.put_nowait(np.zeros(buffer_size, dtype=np.float32))
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task

    def stream(self, preset: Preset) -> None:
        """Compatibility wrapper around :meth:`stream_async` for sync callers."""

        try:
            asyncio.run(self.stream_async(preset))
        except KeyboardInterrupt:
            pass
        finally:
            self._executor.shutdown(wait=False)

    def close(self) -> None:
        """Explicitly release background resources."""

        self._executor.shutdown(wait=False)


__all__ = ["RealTimeEngine"]

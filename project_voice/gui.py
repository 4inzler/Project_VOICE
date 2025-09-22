"""Gradio GUI exposing presets and device selection."""
from __future__ import annotations

from pathlib import Path
from typing import List

import gradio as gr
import torch

from .config import ProjectVoiceConfig
from .inference import RealTimeEngine
from .utils.guardrails import validate_prompt


def build_interface() -> gr.Blocks:
    cfg = ProjectVoiceConfig()

    with gr.Blocks(title="Project VOICE") as demo:
        gr.Markdown(
            """# Project VOICE
            **Consent required.** Do not upload or request non-consensual or celebrity voices.
            """
        )
        checkpoint = gr.File(label="Checkpoint", file_types=[".pt"])
        preset = gr.Dropdown(choices=[preset.name for preset in cfg.presets], value=cfg.presets[0].name)
        device = gr.Dropdown(
            choices=["auto", "cpu", "cuda", "cuda:0"],
            value="auto",
            label="Device",
        )
        prompt = gr.Textbox(label="Usage Confirmation", value="I have consent from the speaker.")
        status = gr.Textbox(label="Status", interactive=False)

        def run_inference(file, preset_name, device_choice, prompt_text):
            if not validate_prompt(prompt_text):
                return "Prompt rejected by celebrity filter"
            if "consent" not in prompt_text.lower():
                return "Please confirm consent in the prompt"
            cfg.inference.device = None if device_choice == "auto" else device_choice
            engine = RealTimeEngine(cfg, Path(file.name))
            engine.stream(next(p for p in cfg.presets if p.name == preset_name))
            return "Streaming started"

        run = gr.Button("Start Real-Time Streaming")
        run.click(run_inference, inputs=[checkpoint, preset, device, prompt], outputs=status)
    return demo


if __name__ == "__main__":
    build_interface().launch()

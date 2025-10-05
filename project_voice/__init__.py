"""Project VOICE package.

The package exposes a small public surface area, but importing every module
eagerly pulls in optional dependencies such as :mod:`librosa`, :mod:`torch`, or
Gradio.  Many commands (for example ``project-voice dataset list``) only need
configuration data and should not fail when those heavier extras are missing.

To keep imports lightweight we expose the main entry points lazily via
``__getattr__``.  Optional dependencies are therefore only imported when the
corresponding attribute is accessed, giving users clearer error messages and a
snappier CLI startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ProjectVoiceConfig",
    "acquire_dataset",
    "list_datasets",
    "preprocess_dataset",
    "ProjectVoiceTrainer",
    "RealTimeEngine",
]


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .config import ProjectVoiceConfig
    from .data import acquire_dataset, list_datasets
    from .inference import RealTimeEngine
    from .preprocess import preprocess_dataset
    from .trainer import ProjectVoiceTrainer


def __getattr__(name: str) -> Any:  # pragma: no cover - exercised indirectly
    if name == "ProjectVoiceConfig":
        from .config import ProjectVoiceConfig as attr

        return attr
    if name in {"acquire_dataset", "list_datasets"}:
        from .data import acquire_dataset, list_datasets

        return {"acquire_dataset": acquire_dataset, "list_datasets": list_datasets}[name]
    if name == "preprocess_dataset":
        from .preprocess import preprocess_dataset as attr

        return attr
    if name == "ProjectVoiceTrainer":
        from .trainer import ProjectVoiceTrainer as attr

        return attr
    if name == "RealTimeEngine":
        from .inference import RealTimeEngine as attr

        return attr
    raise AttributeError(name)

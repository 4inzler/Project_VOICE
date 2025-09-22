"""Ethical guardrail helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

CELEBRITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"taylor swift",
        r"scarlett johansson",
        r"ariana grande",
        r"emma stone",
        r"angelina jolie",
    ]
]


def enforce_guardrails(workspace: Path) -> None:
    """Ensure that the workspace includes consent documentation."""

    consent_file = workspace / "CONSENT.txt"
    if not consent_file.exists():
        consent_file.write_text(
            "By using this dataset you confirm you have explicit consent "
            "from the voice talent and will not recreate non-consenting or "
            "celebrity voices.",
            encoding="utf8",
        )


def validate_prompt(text: str) -> bool:
    """Return False if the request appears to reference a celebrity."""

    return not any(pattern.search(text) for pattern in CELEBRITY_PATTERNS)


__all__ = ["enforce_guardrails", "validate_prompt"]

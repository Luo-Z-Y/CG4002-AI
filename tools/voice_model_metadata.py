#!/usr/bin/env python3
"""Shared voice-model metadata for training and runtime wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


# Training currently derives label ids by sorting dataset folder names. Keep the
# canonical runtime fallback in that same order so dashboards and board-side
# inference agree even before metadata files are exported.
CANONICAL_VOICE_LABELS = [
    "Bulbasaur",
    "Charizard",
    "Greninja",
    "Lugia",
    "Mewtwo",
    "Pikachu",
]

VOICE_LABELS_FILENAME = "voice_labels.json"
VOICE_PREPROCESS_CONFIG_FILENAME = "voice_preprocess_config.json"

# Future voice artefacts should export the preprocessor knobs they were trained
# with so runtime does not silently drift when defaults change.
DEFAULT_TRAINING_VOICE_PREPROCESS_KWARGS: dict[str, Any] = {
    "pre_emphasis": 0.97,
    "cepstral_mean_norm": True,
    "trim_threshold_db": -24.0,
    "trim_pad_ms": 40.0,
    "pitch_focus_half_window_s": 0.7,
    "pitch_focus_energy_threshold_db": -14.0,
}


def normalise_voice_labels(labels: Sequence[str]) -> list[str]:
    canonical = {label.lower(): label for label in CANONICAL_VOICE_LABELS}
    normalized: list[str] = []
    for label in labels:
        clean = str(label).strip()
        normalized.append(canonical.get(clean.lower(), clean))
    return normalized


def labels_from_class_map(class_map: dict[str, int]) -> list[str]:
    ordered = [label for label, _ in sorted(class_map.items(), key=lambda item: item[1])]
    return normalise_voice_labels(ordered)


def save_voice_labels(labels: Sequence[str], output_path: str | Path) -> Path:
    path = Path(output_path)
    payload = {"labels": normalise_voice_labels(labels)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_voice_labels(input_path: str | Path) -> list[str] | None:
    path = Path(input_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        return None
    return normalise_voice_labels([str(label) for label in labels])


def save_voice_preprocess_config(config: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_voice_preprocess_config(input_path: str | Path) -> dict[str, Any] | None:
    path = Path(input_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload

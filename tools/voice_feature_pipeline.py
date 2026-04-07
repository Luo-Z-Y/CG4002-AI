#!/usr/bin/env python3
"""Shared voice feature extraction for training and deployment parity.

This module makes the training pipeline call the exact same deployment
preprocessor used by the dashboard and Ultra96-side software path.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
import sys
import re

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultra96.deployment.audio import VoicePreprocessor, decode_m4a_to_waveform


AUDIO_SUFFIXES = {".wav", ".m4a"}


def _supported_preprocess_kwargs(preprocess_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    """Drop unknown kwargs so cached older preprocessors do not break notebooks."""

    if not preprocess_kwargs:
        return {}
    signature = inspect.signature(VoicePreprocessor.__init__)
    supported = set(signature.parameters)
    supported.discard("self")
    return {key: value for key, value in preprocess_kwargs.items() if key in supported}


def decode_audio_file(
    audio_path: str | Path,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> np.ndarray:
    """Decode a supported audio file into mono float32 PCM."""

    path = Path(audio_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"Unsupported audio suffix for deployment feature build: {path.suffix}")
    return decode_m4a_to_waveform(path, sample_rate=sample_rate, ffmpeg_path=ffmpeg_path).astype(np.float32)


def feature_from_audio_path(
    audio_path: str | Path,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
    preprocess_kwargs: dict[str, Any] | None = None,
    preprocessor: VoicePreprocessor | None = None,
) -> np.ndarray:
    """Build one [40, 50] feature matrix using the deployment preprocessor."""

    processor = preprocessor or VoicePreprocessor(
        sample_rate=sample_rate,
        **_supported_preprocess_kwargs(preprocess_kwargs),
    )
    waveform = decode_audio_file(audio_path, sample_rate=sample_rate, ffmpeg_path=ffmpeg_path)
    feat = processor.process_waveform(waveform, sample_rate)
    return feat.astype(np.float32)


def build_feature_set_from_manifest(
    manifest_df: Any,
    path_column: str = "path",
    label_column: str = "label_id",
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
    preprocess_kwargs: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert every manifest row into deployment-aligned MFCC features."""

    preprocess_kwargs = _supported_preprocess_kwargs(preprocess_kwargs)
    processor = VoicePreprocessor(sample_rate=sample_rate, **preprocess_kwargs)
    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    for _, row in manifest_df.iterrows():
        all_X.append(
            feature_from_audio_path(
                row[path_column],
                sample_rate=sample_rate,
                ffmpeg_path=ffmpeg_path,
                preprocess_kwargs=preprocess_kwargs,
                preprocessor=processor,
            )
        )
        all_y.append(int(row[label_column]))
    return np.stack(all_X, axis=0), np.asarray(all_y, dtype=np.int64)


def natural_sort_key(path_like: str | Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path_like).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def discover_dashboard_voice_roots(
    dashboard_data_root: str | Path,
    required_class_names: set[str] | None = None,
) -> list[Path]:
    """Find reviewed dashboard voice folders shaped like `<session>/voice/<label>/*.wav`."""

    root = Path(dashboard_data_root).resolve()
    if not root.exists():
        return []

    discovered: list[Path] = []
    for session_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=natural_sort_key):
        voice_dir = session_dir / "voice"
        if not voice_dir.is_dir():
            continue
        class_dirs = [path for path in voice_dir.iterdir() if path.is_dir()]
        if required_class_names is not None:
            class_dir_names = {path.name for path in class_dirs}
            if not (class_dir_names & required_class_names):
                continue
        has_audio = any(
            audio_path.is_file() and audio_path.suffix.lower() in AUDIO_SUFFIXES
            for class_dir in class_dirs
            for audio_path in class_dir.rglob("*")
        )
        if has_audio:
            discovered.append(voice_dir)
    return discovered


def normalize_speaker_name(name: str) -> str:
    name = re.sub(r"\([^)]*\)", "", name).strip()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name or "speaker"


def infer_speaker_id(audio_path: Path, class_dir: Path, ordinal: int, clips_per_speaker: int = 5) -> str:
    rel_parts = audio_path.relative_to(class_dir).parts
    if len(rel_parts) > 1:
        return normalize_speaker_name(rel_parts[0])
    speaker_idx = ordinal // clips_per_speaker
    return f"speaker_{speaker_idx + 1:03d}"


def scan_audio_roots(
    roots: list[str | Path] | tuple[str | Path, ...],
    clips_per_speaker: int = 5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one manifest across multiple dated audio roots."""

    root_paths = [Path(root).resolve() for root in roots]
    if not root_paths:
        raise RuntimeError("At least one audio root is required")

    class_names = sorted(
        {
            class_dir.name
            for root in root_paths
            for class_dir in root.iterdir()
            if class_dir.is_dir()
        }
    )
    if not class_names:
        raise RuntimeError("No class folders found across audio roots")

    class_map = {name: idx for idx, name in enumerate(class_names)}
    rows: list[dict[str, object]] = []
    for root in root_paths:
        source_name = root.name
        for class_name in class_names:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            audio_files = sorted(
                [
                    path
                    for path in class_dir.rglob("*")
                    if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
                ],
                key=natural_sort_key,
            )
            for ordinal, audio_path in enumerate(audio_files):
                speaker_local_id = infer_speaker_id(
                    audio_path,
                    class_dir,
                    ordinal,
                    clips_per_speaker=clips_per_speaker,
                )
                rows.append(
                    {
                        "path": str(audio_path),
                        "label": class_name,
                        "label_id": class_map[class_name],
                        "source": source_name,
                        "speaker_local_id": speaker_local_id,
                        "speaker_id": f"{source_name}_{speaker_local_id}",
                        "utterance_id": (ordinal % clips_per_speaker) + 1,
                    }
                )

    if not rows:
        raise RuntimeError("No .wav/.m4a files found across audio roots")

    manifest = pd.DataFrame(rows)
    manifest = manifest.sort_values(
        ["label_id", "source", "speaker_local_id", "utterance_id", "path"]
    ).reset_index(drop=True)
    return manifest, class_map


def subset_has_all_labels(df_subset: pd.DataFrame, class_map: dict[str, int]) -> bool:
    return set(df_subset["label_id"].tolist()) == set(class_map.values())


def split_manifest_source_heldout(
    manifest_df: pd.DataFrame,
    class_map: dict[str, int],
    test_source: str,
    test_speakers_from_source: int = 5,
    val_size: float = 0.2,
    min_val_speakers: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, set[str], set[str], set[str]]:
    """Hold out speakers from one source for test, split the rest into train/val."""

    source_speakers = np.asarray(
        sorted(manifest_df.loc[manifest_df["source"] == test_source, "speaker_id"].unique()),
        dtype=object,
    )
    if len(source_speakers) < test_speakers_from_source:
        raise RuntimeError(
            f"Requested {test_speakers_from_source} test speakers from {test_source}, "
            f"but found only {len(source_speakers)}"
        )

    # Deterministic, easy-to-audit held-out speakers.
    test_speakers = set(source_speakers[:test_speakers_from_source].tolist())
    test_df = manifest_df[manifest_df["speaker_id"].isin(test_speakers)].copy()
    if not subset_has_all_labels(test_df, class_map):
        raise RuntimeError(f"Held-out source split for {test_source} does not cover all labels")

    remaining_df = manifest_df[~manifest_df["speaker_id"].isin(test_speakers)].copy()
    remaining_speakers = np.asarray(sorted(remaining_df["speaker_id"].unique()), dtype=object)
    if len(remaining_speakers) <= min_val_speakers:
        raise RuntimeError("Not enough non-test speakers left for train/val split")

    rng = np.random.default_rng(seed)
    for attempt in range(256):
        shuffled = remaining_speakers.copy()
        rng = np.random.default_rng(seed + attempt)
        rng.shuffle(shuffled)
        n_val = max(min_val_speakers, int(round(len(shuffled) * val_size)))
        n_val = min(n_val, len(shuffled) - 1)
        val_speakers = set(shuffled[:n_val].tolist())
        train_speakers = set(shuffled[n_val:].tolist())
        train_df = remaining_df[remaining_df["speaker_id"].isin(train_speakers)].copy()
        val_df = remaining_df[remaining_df["speaker_id"].isin(val_speakers)].copy()
        if subset_has_all_labels(train_df, class_map) and subset_has_all_labels(val_df, class_map):
            return train_df, val_df, test_df, train_speakers, val_speakers, test_speakers

    raise RuntimeError("Failed to find a train/val split with all labels after holding out the test source")

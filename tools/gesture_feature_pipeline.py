#!/usr/bin/env python3
"""Helpers for combining reviewed dashboard gesture samples with notebook data."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


GESTURE_LABEL_ORDER = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
GESTURE_LABEL_SLUGS = ("raise", "shake", "chop", "stir", "swing", "punch")
GESTURE_SLUG_TO_ID = {slug: idx for idx, slug in enumerate(GESTURE_LABEL_SLUGS)}


def natural_sort_key(path_like: str | Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path_like).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def discover_dashboard_gesture_roots(
    dashboard_data_root: str | Path,
    required_class_names: set[str] | None = None,
) -> list[Path]:
    """Find reviewed dashboard gesture folders shaped like `<session>/gesture/<label>/*.txt`."""

    root = Path(dashboard_data_root).resolve()
    if not root.exists():
        return []

    discovered: list[Path] = []
    for session_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=natural_sort_key):
        gesture_dir = session_dir / "gesture"
        if not gesture_dir.is_dir():
            continue
        class_dirs = [path for path in gesture_dir.iterdir() if path.is_dir()]
        if required_class_names is not None:
            class_dir_names = {path.name for path in class_dirs}
            if not (class_dir_names & required_class_names):
                continue
        has_samples = any(sample_path.is_file() and sample_path.suffix.lower() == ".txt" for class_dir in class_dirs for sample_path in class_dir.glob("*.txt"))
        if has_samples:
            discovered.append(gesture_dir)
    return discovered


def _load_plain_window(path: Path) -> np.ndarray | None:
    arr = None
    for delim in (",", None):
        try:
            arr = np.loadtxt(path, dtype=np.float32, delimiter=delim)
            break
        except Exception:
            arr = None
    if arr is None:
        return None
    if arr.ndim == 1:
        if arr.size % 6 != 0:
            return None
        arr = arr.reshape(-1, 6)
    if arr.ndim != 2 or arr.shape[1] != 6:
        return None
    if np.isnan(arr).any():
        return None
    return arr.astype(np.float32)


def _load_dashboard_packet_window(path: Path) -> np.ndarray | None:
    text = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    rows: list[list[float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("====") or line.startswith("Sample count:") or line.startswith("Idx"):
            continue
        match = re.match(r"\d+\s*\|\s*(.+)$", line)
        if not match:
            continue
        parts = [part.strip() for part in match.group(1).split("|")]
        if len(parts) != 6:
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            return None
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def load_gesture_window_from_txt(path: str | Path) -> np.ndarray | None:
    sample_path = Path(path).resolve()
    plain = _load_plain_window(sample_path)
    if plain is not None:
        return plain
    return _load_dashboard_packet_window(sample_path)


def build_dashboard_gesture_dataframe(
    roots: list[str | Path] | tuple[str | Path, ...],
    measurement_id_start: int = 0,
) -> pd.DataFrame:
    """Convert reviewed dashboard gesture samples into the notebook's canonical dataframe."""

    root_paths = [Path(root).resolve() for root in roots]
    rows: list[dict[str, object]] = []
    measurement_id = int(measurement_id_start)

    for root in root_paths:
        source_name = root.parent.name
        for class_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=natural_sort_key):
            slug = class_dir.name.strip().lower()
            if slug not in GESTURE_SLUG_TO_ID:
                continue
            label_id = GESTURE_SLUG_TO_ID[slug]
            label_name = GESTURE_LABEL_ORDER[label_id]
            for sample_path in sorted(class_dir.glob("*.txt"), key=natural_sort_key):
                sample = load_gesture_window_from_txt(sample_path)
                if sample is None or len(sample) < 10:
                    continue
                for sequence_id, values in enumerate(sample):
                    rows.append(
                        {
                            "measurement_id": measurement_id,
                            "sequence_id": sequence_id,
                            "label_id": label_id,
                            "label": label_name,
                            "gyro_x": float(values[0]),
                            "gyro_y": float(values[1]),
                            "gyro_z": float(values[2]),
                            "acc_x": float(values[3]),
                            "acc_y": float(values[4]),
                            "acc_z": float(values[5]),
                            "source": f"dashboard:{source_name}",
                            "path": str(sample_path),
                        }
                    )
                measurement_id += 1

    return pd.DataFrame(rows)

#!/usr/bin/env python3
"""Offline cleanup for the voice dataset.

This script is intended for training-data preparation, not live deployment.
It trims leading/trailing silence, applies consistent loudness normalization,
and writes cleaned `.wav` files into a separate output folder while preserving
the class subfolder layout.

Example:
    python tools/preprocess_voice_audio.py \
        --input data/audio/20260321 \
        --output data/audio/20260321_clean
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultra96.deployment.audio import decode_m4a_to_waveform


AUDIO_SUFFIXES = {".wav", ".m4a"}
EPS = 1e-8


@dataclass
class AudioStats:
    input_path: str
    output_path: str
    duration_before_s: float
    duration_after_s: float
    trimmed_start_ms: float
    trimmed_end_ms: float
    gain_db: float
    rms_before_dbfs: float
    rms_after_dbfs: float
    peak_before_dbfs: float
    peak_after_dbfs: float


def dbfs_from_amplitude(value: float) -> float:
    if value <= EPS:
        return float("-inf")
    return 20.0 * math.log10(value)


def rms_dbfs(waveform: np.ndarray) -> float:
    return dbfs_from_amplitude(float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64))))


def peak_dbfs(waveform: np.ndarray) -> float:
    return dbfs_from_amplitude(float(np.max(np.abs(waveform), initial=0.0)))


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES:
            yield path


def compute_frame_rms(
    waveform: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, int, int]:
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if waveform.size <= frame_len:
        frame = np.pad(waveform, (0, max(0, frame_len - waveform.size)))
        return np.asarray([np.sqrt(np.mean(np.square(frame), dtype=np.float64))], dtype=np.float32), frame_len, hop_len

    rms_values = []
    for start in range(0, waveform.size - frame_len + 1, hop_len):
        frame = waveform[start : start + frame_len]
        rms_values.append(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))
    if not rms_values:
        rms_values.append(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    return np.asarray(rms_values, dtype=np.float32), frame_len, hop_len


def trim_silence(
    waveform: np.ndarray,
    sample_rate: int,
    trim_threshold_db: float,
    min_floor_dbfs: float,
    pad_ms: float,
    frame_ms: float,
    hop_ms: float,
) -> tuple[np.ndarray, int, int]:
    frame_rms, frame_len, hop_len = compute_frame_rms(waveform, sample_rate, frame_ms, hop_ms)
    peak_rms = float(frame_rms.max(initial=0.0))
    if peak_rms <= EPS:
        return waveform.copy(), 0, 0

    relative_threshold = peak_rms * (10.0 ** (trim_threshold_db / 20.0))
    absolute_threshold = 10.0 ** (min_floor_dbfs / 20.0)
    threshold = max(relative_threshold, absolute_threshold)
    active = np.flatnonzero(frame_rms >= threshold)
    if active.size == 0:
        return waveform.copy(), 0, 0

    pad = int(round(sample_rate * pad_ms / 1000.0))
    start = max(0, int(active[0]) * hop_len - pad)
    end = min(waveform.size, int(active[-1]) * hop_len + frame_len + pad)
    trimmed = waveform[start:end].copy()
    return trimmed, start, waveform.size - end


def normalize_loudness(
    waveform: np.ndarray,
    target_rms_dbfs: float,
    peak_limit_dbfs: float,
) -> tuple[np.ndarray, float]:
    current_rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    if current_rms <= EPS:
        return waveform.copy(), 0.0

    target_rms = 10.0 ** (target_rms_dbfs / 20.0)
    gain = target_rms / current_rms

    peak_limit = 10.0 ** (peak_limit_dbfs / 20.0)
    predicted_peak = float(np.max(np.abs(waveform), initial=0.0)) * gain
    if predicted_peak > peak_limit and predicted_peak > EPS:
        gain *= peak_limit / predicted_peak

    normalized = np.clip(waveform * gain, -1.0, 1.0).astype(np.float32)
    gain_db = 20.0 * math.log10(max(gain, EPS))
    return normalized, gain_db


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = np.clip(waveform, -1.0, 1.0)
    pcm16 = np.round(pcm16 * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(pcm16.tobytes())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trim silence and normalize loudness for the voice dataset.")
    parser.add_argument("--input", required=True, help="Input audio dataset root, e.g. data/audio/20260321")
    parser.add_argument("--output", required=True, help="Output root for cleaned .wav files")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--trim-threshold-db", type=float, default=-28.0, help="Relative frame-energy trim threshold")
    parser.add_argument("--min-floor-dbfs", type=float, default=-45.0, help="Absolute floor to avoid trimming noise only")
    parser.add_argument("--pad-ms", type=float, default=80.0, help="Context to keep before/after detected speech")
    parser.add_argument("--frame-ms", type=float, default=20.0)
    parser.add_argument("--hop-ms", type=float, default=10.0)
    parser.add_argument("--target-rms-dbfs", type=float, default=-18.0, help="Target loudness after trimming")
    parser.add_argument("--peak-limit-dbfs", type=float, default=-1.0, help="Hard peak cap after gain")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--report-csv", default=None, help="Optional CSV report path. Defaults to <output>/preprocess_report.csv")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")

    report_path = Path(args.report_csv).resolve() if args.report_csv else output_root / "preprocess_report.csv"
    rows: list[AudioStats] = []
    audio_files = list(iter_audio_files(input_root))
    if not audio_files:
        raise RuntimeError(f"No audio files found under: {input_root}")

    for source in audio_files:
        waveform = decode_m4a_to_waveform(source, sample_rate=args.sample_rate, ffmpeg_path=args.ffmpeg_path).astype(np.float32)
        trimmed, start_trim, end_trim = trim_silence(
            waveform,
            sample_rate=args.sample_rate,
            trim_threshold_db=args.trim_threshold_db,
            min_floor_dbfs=args.min_floor_dbfs,
            pad_ms=args.pad_ms,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
        )
        normalized, gain_db = normalize_loudness(
            trimmed,
            target_rms_dbfs=args.target_rms_dbfs,
            peak_limit_dbfs=args.peak_limit_dbfs,
        )

        rel = source.relative_to(input_root)
        dest = (output_root / rel).with_suffix(".wav")
        write_wav(dest, normalized, sample_rate=args.sample_rate)

        rows.append(
            AudioStats(
                input_path=str(source),
                output_path=str(dest),
                duration_before_s=waveform.size / float(args.sample_rate),
                duration_after_s=normalized.size / float(args.sample_rate),
                trimmed_start_ms=1000.0 * start_trim / float(args.sample_rate),
                trimmed_end_ms=1000.0 * end_trim / float(args.sample_rate),
                gain_db=gain_db,
                rms_before_dbfs=rms_dbfs(waveform),
                rms_after_dbfs=rms_dbfs(normalized),
                peak_before_dbfs=peak_dbfs(waveform),
                peak_after_dbfs=peak_dbfs(normalized),
            )
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(AudioStats.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    durations_before = [row.duration_before_s for row in rows]
    durations_after = [row.duration_after_s for row in rows]
    print(f"Processed {len(rows)} files")
    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Report: {report_path}")
    print(
        "Duration mean before/after: "
        f"{np.mean(durations_before):.3f}s -> {np.mean(durations_after):.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Segment long repeated voice recordings into per-utterance WAV clips.

This is intended for bulk recordings where one file contains many repeated
utterances for a single class. The output is written directly into the raw
training tree using the existing `<class>/<speaker>/clip.wav` layout so the
notebook manifest scan can pick the new data up without further changes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.preprocess_voice_audio import write_wav
from tools.voice_feature_pipeline import decode_audio_file


RAW_LABEL_ALIASES = {
    "balbausaur": "bulbasaur",
    "bulbarsaur": "bulbasaur",
    "bulbasaur": "bulbasaur",
    "charizard": "charizard",
    "grininja": "greninja",
    "greninja": "greninja",
    "lugia": "lugia",
    "mewtwo": "mewtwo",
    "pikachu": "pikachu",
}

THRESHOLD_CANDIDATES_DB = [-26.0, -27.0, -28.0, -29.0, -30.0, -31.0, -32.0, -33.0, -34.0, -35.0, -36.0, -37.0, -38.0]
GAP_CANDIDATES_S = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
PAD_CANDIDATES_S = [0.00, 0.05, 0.10, 0.15, 0.20]


@dataclass
class SegmentStats:
    start_s: float
    end_s: float
    duration_s: float
    score: float
    output_path: str | None = None


def canonical_label_from_path(path: Path) -> str:
    token = re.sub(r"[^a-z0-9]+", "", path.stem.lower())
    if token not in RAW_LABEL_ALIASES:
        raise ValueError(f"Cannot infer label from filename: {path.name}")
    return RAW_LABEL_ALIASES[token]


def compute_frame_rms(
    waveform: np.ndarray,
    sample_rate: int,
    frame_s: float = 0.10,
    hop_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, int]:
    frame_len = max(1, int(round(sample_rate * frame_s)))
    hop_len = max(1, int(round(sample_rate * hop_s)))
    if waveform.size <= frame_len:
        frame = np.pad(waveform, (0, max(0, frame_len - waveform.size)))
        rms = np.asarray([np.sqrt(np.mean(np.square(frame), dtype=np.float64))], dtype=np.float32)
        return rms, np.asarray([0], dtype=np.int64), frame_len
    starts = np.arange(0, waveform.size - frame_len + 1, hop_len, dtype=np.int64)
    rms = np.asarray(
        [np.sqrt(np.mean(np.square(waveform[start : start + frame_len]), dtype=np.float64)) for start in starts],
        dtype=np.float32,
    )
    return rms, starts, frame_len


def detect_segments(
    waveform: np.ndarray,
    sample_rate: int,
    rms: np.ndarray,
    starts: np.ndarray,
    frame_len: int,
    threshold_db: float,
    max_gap_s: float,
    pad_s: float = 0.20,
    min_duration_s: float = 0.35,
) -> list[SegmentStats]:
    peak_rms = float(rms.max(initial=0.0))
    if peak_rms <= 1e-8:
        return []
    threshold = peak_rms * (10.0 ** (threshold_db / 20.0))
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return []

    max_gap = int(round(sample_rate * max_gap_s))
    pad = int(round(sample_rate * pad_s))
    min_duration = int(round(sample_rate * min_duration_s))
    merged: list[tuple[int, int]] = []

    current_start = int(starts[active[0]])
    current_end = int(starts[active[0]] + frame_len)
    for idx in active[1:]:
        frame_start = int(starts[idx])
        frame_end = int(starts[idx] + frame_len)
        if frame_start - current_end <= max_gap:
            current_end = frame_end
        else:
            merged.append((current_start, current_end))
            current_start = frame_start
            current_end = frame_end
    merged.append((current_start, current_end))

    segments: list[SegmentStats] = []
    for raw_start, raw_end in merged:
        start = max(0, raw_start - pad)
        end = min(waveform.size, raw_end + pad)
        if end - start < min_duration:
            continue
        segment_wave = waveform[start:end]
        score = float((end - start) / sample_rate) * float(np.mean(np.abs(segment_wave), dtype=np.float64))
        segments.append(
            SegmentStats(
                start_s=start / sample_rate,
                end_s=end / sample_rate,
                duration_s=(end - start) / sample_rate,
                score=score,
            )
        )
    return segments


def choose_segment_plan(
    waveform: np.ndarray,
    sample_rate: int,
    target_count: int,
) -> tuple[list[SegmentStats], float, float, float]:
    rms, starts, frame_len = compute_frame_rms(waveform, sample_rate=sample_rate)
    candidates: list[tuple[tuple[int, int, float, float], list[SegmentStats], float, float, float]] = []
    for threshold_db in THRESHOLD_CANDIDATES_DB:
        for gap_s in GAP_CANDIDATES_S:
            for pad_s in PAD_CANDIDATES_S:
                segments = detect_segments(
                    waveform,
                    sample_rate,
                    rms,
                    starts,
                    frame_len,
                    threshold_db=threshold_db,
                    max_gap_s=gap_s,
                    pad_s=pad_s,
                )
                count = len(segments)
                if count == 0:
                    continue
                short_penalty = sum(segment.duration_s < 0.60 for segment in segments)
                long_penalty = sum(segment.duration_s > 2.50 for segment in segments)
                max_duration_penalty = max((segment.duration_s for segment in segments), default=0.0)
                key = (
                    long_penalty,
                    short_penalty,
                    0 if count >= target_count else 1,
                    abs(count - target_count),
                    max_duration_penalty,
                    abs(threshold_db + 32.0),
                    abs(gap_s - 0.12),
                    pad_s,
                )
                candidates.append((key, segments, threshold_db, gap_s, pad_s))

    if not candidates:
        raise RuntimeError("No segmentation candidates produced")

    candidates.sort(key=lambda item: item[0])
    best_segments, best_threshold, best_gap, best_pad = candidates[0][1], candidates[0][2], candidates[0][3], candidates[0][4]
    adjusted = adjust_segment_count(best_segments, waveform, sample_rate, target_count)
    return adjusted, best_threshold, best_gap, best_pad


def adjust_segment_count(
    segments: list[SegmentStats],
    waveform: np.ndarray,
    sample_rate: int,
    target_count: int,
) -> list[SegmentStats]:
    if len(segments) > target_count:
        kept = sorted(sorted(segments, key=lambda seg: seg.score, reverse=True)[:target_count], key=lambda seg: seg.start_s)
        return kept

    adjusted = list(sorted(segments, key=lambda seg: seg.start_s))
    while len(adjusted) < target_count and adjusted:
        split_idx = max(range(len(adjusted)), key=lambda idx: adjusted[idx].duration_s)
        segment = adjusted.pop(split_idx)
        replacement = split_segment(segment, waveform, sample_rate)
        adjusted[split_idx:split_idx] = replacement
        adjusted = sorted(adjusted, key=lambda seg: seg.start_s)
        if len(replacement) == 1:
            break
    return adjusted[:target_count]


def split_segment(segment: SegmentStats, waveform: np.ndarray, sample_rate: int) -> list[SegmentStats]:
    start = int(round(segment.start_s * sample_rate))
    end = int(round(segment.end_s * sample_rate))
    wave = waveform[start:end]
    if wave.size < int(round(0.80 * sample_rate)):
        return [segment]

    mid = wave.size // 2
    window = max(1, int(round(0.15 * sample_rate)))
    left = max(1, mid - window)
    right = min(wave.size - 1, mid + window)
    envelope = np.abs(wave)
    cut = left + int(np.argmin(envelope[left:right]))
    min_half = int(round(0.45 * sample_rate))
    if cut < min_half or (wave.size - cut) < min_half:
        cut = mid

    first = waveform[start : start + cut]
    second = waveform[start + cut : end]
    if first.size < min_half or second.size < min_half:
        return [segment]

    def build(seg_start: int, seg_end: int) -> SegmentStats:
        seg_wave = waveform[seg_start:seg_end]
        return SegmentStats(
            start_s=seg_start / sample_rate,
            end_s=seg_end / sample_rate,
            duration_s=(seg_end - seg_start) / sample_rate,
            score=float(((seg_end - seg_start) / sample_rate) * np.mean(np.abs(seg_wave), dtype=np.float64)),
        )

    return [build(start, start + cut), build(start + cut, end)]


def iter_input_audio(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in {".m4a", ".wav"}:
            yield path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Segment repeated voice recordings into per-utterance clips.")
    parser.add_argument("--input-root", default=str(REPO_ROOT / "data" / "audio"), help="Folder containing the long .m4a/.wav source files")
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data" / "audio" / "20260406" / "new"), help="Root in existing class/speaker dataset layout")
    parser.add_argument("--speaker-id", default="csy_20260410", help="Speaker folder name to write under each class")
    parser.add_argument("--target-count", type=int, default=20, help="Target number of clips per source file")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--report-json", default=None, help="Optional JSON report path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    input_files = list(iter_input_audio(input_root))
    if not input_files:
        raise RuntimeError(f"No top-level audio files found under {input_root}")

    report: dict[str, object] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "speaker_id": args.speaker_id,
        "target_count": args.target_count,
        "files": [],
    }

    for input_path in input_files:
        label = canonical_label_from_path(input_path)
        waveform = decode_audio_file(input_path, sample_rate=args.sample_rate)
        segments, threshold_db, gap_s, pad_s = choose_segment_plan(waveform, args.sample_rate, args.target_count)
        if len(segments) != args.target_count:
            raise RuntimeError(
                f"{input_path.name} produced {len(segments)} segments after adjustment; expected {args.target_count}"
            )

        speaker_dir = output_root / label / args.speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)

        file_report = {
            "input_path": str(input_path),
            "label": label,
            "threshold_db": threshold_db,
            "gap_s": gap_s,
            "pad_s": pad_s,
            "segments": [],
        }
        for idx, segment in enumerate(segments, start=1):
            start = int(round(segment.start_s * args.sample_rate))
            end = int(round(segment.end_s * args.sample_rate))
            clip = waveform[start:end].astype(np.float32)
            output_path = speaker_dir / f"{label}_{args.speaker_id}_{idx:02d}.wav"
            write_wav(output_path, clip, sample_rate=args.sample_rate)
            segment.output_path = str(output_path)
            file_report["segments"].append(asdict(segment))

        report["files"].append(file_report)
        print(
            f"{input_path.name} -> {label}: wrote {len(segments)} clips using threshold {threshold_db:.1f} dB, "
            f"gap {gap_s:.2f} s, pad {pad_s:.2f} s"
        )

    report_path = Path(args.report_json).resolve() if args.report_json else output_root / f"{args.speaker_id}_segmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

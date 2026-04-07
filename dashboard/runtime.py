from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = REPO_ROOT / "ultra96" / "deployment"
TOOLS_DIR = REPO_ROOT / "tools"
for search_path in [DEPLOYMENT_DIR, TOOLS_DIR]:
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from audio import VoicePreprocessor, load_feature_norm_stats, normalize_feature_matrix  # type: ignore  # noqa: E402
from imu import ImuPreprocessor, load_feature_norm_stats as load_gesture_norm_stats, normalize_window  # type: ignore  # noqa: E402
from voice_model_metadata import (  # type: ignore  # noqa: E402
    CANONICAL_VOICE_LABELS,
    VOICE_LABELS_FILENAME,
    VOICE_PREPROCESS_CONFIG_FILENAME,
    load_voice_labels,
    load_voice_preprocess_config,
)
from voice_cnn_training import build_voice_model  # type: ignore  # noqa: E402


GESTURE_LABELS = ["Raise", "Shake", "Chop", "Stir", "Swing", "Punch"]
VOICE_LABELS = list(CANONICAL_VOICE_LABELS)
VOICE_DASHBOARD_CHECKPOINT_FILENAME = "voice_dashboard_model.pt"

_ARRAY_RE = re.compile(
    r"static const data_t (?P<name>\w+)\[(?P<size>\d+)\] = \{(?P<body>.*?)\};",
    re.DOTALL,
)


class GestureCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 15, 32)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(32, len(GESTURE_LABELS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.relu3(self.fc1(x))
        return self.fc2(x)


class VoiceCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(40, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.drop1 = nn.Dropout(0.0)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool1d(1)
        self.drop2 = nn.Dropout(0.0)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.relu1(self.conv1(x))))
        x = self.drop2(self.pool2(self.relu2(self.conv2(x))))
        x = x.squeeze(-1)
        return self.fc(x)


def _read_weight_arrays(header_path: Path) -> dict[str, np.ndarray]:
    text = header_path.read_text(encoding="utf-8")
    arrays: dict[str, np.ndarray] = {}

    for match in _ARRAY_RE.finditer(text):
        name = match.group("name")
        size = int(match.group("size"))
        body = match.group("body").replace("\n", " ")
        values = np.fromstring(body, sep=",", dtype=np.float32)
        if values.size != size:
            raise ValueError(f"{header_path}: array {name} expected {size} values, found {values.size}")
        arrays[name] = values

    if not arrays:
        raise ValueError(f"No weight arrays found in {header_path}")
    return arrays


def _load_gesture_model(header_path: Path) -> GestureCNN:
    arrays = _read_weight_arrays(header_path)
    model = GestureCNN()
    state = model.state_dict()
    state["conv1.weight"] = torch.from_numpy(arrays["conv1_w"].reshape(16, 6, 3))
    state["conv1.bias"] = torch.from_numpy(arrays["conv1_b"])
    state["conv2.weight"] = torch.from_numpy(arrays["conv2_w"].reshape(32, 16, 3))
    state["conv2.bias"] = torch.from_numpy(arrays["conv2_b"])
    state["fc1.weight"] = torch.from_numpy(arrays["fc1_w"].reshape(32, 32 * 15))
    state["fc1.bias"] = torch.from_numpy(arrays["fc1_b"])
    state["fc2.weight"] = torch.from_numpy(arrays["fc2_w"].reshape(len(GESTURE_LABELS), 32))
    state["fc2.bias"] = torch.from_numpy(arrays["fc2_b"])
    model.load_state_dict(state)
    model.eval()
    return model


def _resolve_voice_labels(class_count: int, artifact_dir: Path | None = None) -> list[str]:
    if artifact_dir is not None:
        labels = load_voice_labels(artifact_dir / VOICE_LABELS_FILENAME)
        if labels is not None and len(labels) == class_count:
            return labels
    if class_count <= len(VOICE_LABELS):
        return VOICE_LABELS[:class_count]
    return [f"Voice {idx}" for idx in range(class_count)]


def _load_voice_model(header_path: Path) -> tuple[VoiceCNN, list[str]]:
    arrays = _read_weight_arrays(header_path)
    class_count = int(arrays["fc_b"].size)
    labels = _resolve_voice_labels(class_count, header_path.parent)
    model = VoiceCNN(num_classes=class_count)
    state = model.state_dict()
    state["conv1.weight"] = torch.from_numpy(arrays["conv1_w"].reshape(16, 40, 3))
    state["conv1.bias"] = torch.from_numpy(arrays["conv1_b"])
    state["conv2.weight"] = torch.from_numpy(arrays["conv2_w"].reshape(32, 16, 3))
    state["conv2.bias"] = torch.from_numpy(arrays["conv2_b"])
    state["fc.weight"] = torch.from_numpy(arrays["fc_w"].reshape(class_count, 32))
    state["fc.bias"] = torch.from_numpy(arrays["fc_b"])
    model.load_state_dict(state)
    model.eval()
    return model, labels


def _load_voice_checkpoint_model(checkpoint_path: Path) -> tuple[nn.Module, list[str]]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"{checkpoint_path}: expected checkpoint dict")
    variant = str(payload.get("variant", "deployed"))
    labels = payload.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{checkpoint_path}: labels missing from checkpoint")
    num_classes = int(payload.get("num_classes", len(labels)))
    dropout_p = float(payload.get("dropout_p", 0.0))
    bn_momentum = float(payload.get("bn_momentum", 0.05))
    bn_eps = float(payload.get("bn_eps", 1e-3))
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"{checkpoint_path}: state_dict missing from checkpoint")

    model = build_voice_model(
        num_classes=num_classes,
        variant=variant,
        dropout_p=dropout_p,
        bn_momentum=bn_momentum,
        bn_eps=bn_eps,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, [str(label) for label in labels]


def _softmax_result(logits: torch.Tensor, labels: list[str]) -> dict[str, Any]:
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "label_index": pred_idx,
        "label": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": [
            {"label": label, "index": idx, "probability": float(prob)}
            for idx, (label, prob) in enumerate(zip(labels, probs))
        ],
    }


def _downsample_1d(values: np.ndarray, max_points: int = 1200) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size <= max_points:
        return arr.astype(float).tolist()
    indices = np.linspace(0, arr.size - 1, num=max_points, dtype=np.int32)
    return arr[indices].astype(float).tolist()


def _encode_wav_bytes(values: np.ndarray, sample_rate: int = 16000) -> bytes:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pcm16 = np.clip(arr, -1.0, 1.0)
    pcm16 = np.round(pcm16 * 32767.0).astype("<i2")
    with io.BytesIO() as handle:
        with wave.open(handle, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(pcm16.tobytes())
        return handle.getvalue()


def _guess_suffix(filename: str | None, content_type: str | None) -> str:
    if filename:
        suffix = Path(filename).suffix.strip()
        if suffix:
            return suffix.lower()
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if guessed:
            return guessed.lower()
    return ".bin"


def _decode_wav_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(audio_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), int(sample_rate)


def _decode_audio_bytes(
    audio_bytes: bytes,
    suffix: str,
    sample_rate: int = 16000,
    ffmpeg_path: str = "ffmpeg",
) -> np.ndarray:
    if suffix == ".wav":
        waveform, wav_rate = _decode_wav_bytes(audio_bytes)
        if wav_rate != sample_rate:
            raise ValueError(f"Expected WAV sample rate {sample_rate}, got {wav_rate}")
        return waveform

    if shutil.which(ffmpeg_path) is None and not Path(ffmpeg_path).exists():
        raise FileNotFoundError(
            f"ffmpeg executable not found: {ffmpeg_path}. Install ffmpeg or upload a {sample_rate} Hz WAV file."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        temp_path = Path(handle.name)
        handle.write(audio_bytes)

    try:
        command = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(temp_path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-",
        ]
        result = subprocess.run(command, check=True, capture_output=True)
        waveform = np.frombuffer(result.stdout, dtype=np.float32)
        if waveform.size == 0:
            raise ValueError(f"Decoded empty waveform from {temp_path.name}")
        return waveform
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


class LocalAiRuntime:
    def __init__(
        self,
        gesture_weights: str | Path | None = None,
        voice_weights: str | Path | None = None,
        voice_checkpoint: str | Path | None = None,
        gesture_mean: str | Path | None = None,
        gesture_std: str | Path | None = None,
        voice_mean: str | Path | None = None,
        voice_std: str | Path | None = None,
    ) -> None:
        gesture_header = Path(gesture_weights) if gesture_weights is not None else REPO_ROOT / "hls" / "gesture" / "gesture_cnn_weights.h"
        voice_header = Path(voice_weights) if voice_weights is not None else REPO_ROOT / "hls" / "voice" / "voice_cnn_weights.h"
        self.gesture_weights_path = gesture_header.resolve()
        self.voice_weights_path = voice_header.resolve()
        self.voice_checkpoint_path = self._resolve_voice_checkpoint_path(voice_checkpoint)
        self.gesture_mean_path, self.gesture_std_path = self._resolve_gesture_norm_paths(gesture_mean, gesture_std)
        self.gesture_mean, self.gesture_std = load_gesture_norm_stats(self.gesture_mean_path, self.gesture_std_path)
        self.voice_mean_path, self.voice_std_path = self._resolve_voice_norm_paths(voice_mean, voice_std)
        self.voice_mean, self.voice_std = load_feature_norm_stats(self.voice_mean_path, self.voice_std_path)
        self.gesture_model = _load_gesture_model(self.gesture_weights_path)
        self.gesture_labels = list(GESTURE_LABELS)
        if self.voice_checkpoint_path is not None:
            self.voice_model, self.voice_labels = _load_voice_checkpoint_model(self.voice_checkpoint_path)
            self.voice_model_source = "checkpoint"
        else:
            self.voice_model, self.voice_labels = _load_voice_model(self.voice_weights_path)
            self.voice_model_source = "hls_header"
        self.gesture_preprocessor = ImuPreprocessor()
        self.voice_preprocess_config = self._resolve_voice_preprocess_config()
        self.voice_preprocessor = VoicePreprocessor(sample_rate=16000, **self.voice_preprocess_config)

    def _resolve_gesture_norm_paths(
        self,
        gesture_mean: str | Path | None,
        gesture_std: str | Path | None,
    ) -> tuple[Path, Path]:
        if (gesture_mean is None) != (gesture_std is None):
            raise ValueError("gesture_mean and gesture_std must be provided together")

        if gesture_mean is not None and gesture_std is not None:
            return Path(gesture_mean).resolve(), Path(gesture_std).resolve()

        candidates = [
            (self.gesture_weights_path.parent / "gesture_mean.npy", self.gesture_weights_path.parent / "gesture_std.npy"),
            (REPO_ROOT / "ultra96" / "deployment" / "gesture_mean.npy", REPO_ROOT / "ultra96" / "deployment" / "gesture_std.npy"),
            (REPO_ROOT / "data" / "gesture" / "20260328peer" / "mean.npy", REPO_ROOT / "data" / "gesture" / "20260328peer" / "std.npy"),
        ]
        for mean_path, std_path in candidates:
            if mean_path.exists() and std_path.exists():
                return mean_path.resolve(), std_path.resolve()
        raise FileNotFoundError("Could not locate gesture_mean.npy and gesture_std.npy for dashboard gesture normalization")

    def _resolve_voice_norm_paths(
        self,
        voice_mean: str | Path | None,
        voice_std: str | Path | None,
    ) -> tuple[Path, Path]:
        if (voice_mean is None) != (voice_std is None):
            raise ValueError("voice_mean and voice_std must be provided together")

        if voice_mean is not None and voice_std is not None:
            return Path(voice_mean).resolve(), Path(voice_std).resolve()

        candidates = []
        if self.voice_checkpoint_path is not None:
            candidates.append(
                (
                    self.voice_checkpoint_path.parent / "voice_mean.npy",
                    self.voice_checkpoint_path.parent / "voice_std.npy",
                )
            )
        candidates.extend([
            (self.voice_weights_path.parent / "voice_mean.npy", self.voice_weights_path.parent / "voice_std.npy"),
            (REPO_ROOT / "ultra96" / "deployment" / "voice_mean.npy", REPO_ROOT / "ultra96" / "deployment" / "voice_std.npy"),
            (REPO_ROOT / "data" / "audio" / "combined" / "voice_mean.npy", REPO_ROOT / "data" / "audio" / "combined" / "voice_std.npy"),
            (REPO_ROOT / "data" / "audio" / "20260321" / "voice_mean.npy", REPO_ROOT / "data" / "audio" / "20260321" / "voice_std.npy"),
        ])
        for mean_path, std_path in candidates:
            if mean_path.exists() and std_path.exists():
                return mean_path.resolve(), std_path.resolve()
        raise FileNotFoundError("Could not locate voice_mean.npy and voice_std.npy for dashboard voice normalization")

    def _resolve_voice_checkpoint_path(self, voice_checkpoint: str | Path | None) -> Path | None:
        if voice_checkpoint is not None:
            path = Path(voice_checkpoint).resolve()
            if not path.exists():
                raise FileNotFoundError(f"Voice checkpoint not found: {path}")
            return path

        candidates: list[Path] = []
        candidates.append(REPO_ROOT / "ultra96" / "deployment" / VOICE_DASHBOARD_CHECKPOINT_FILENAME)
        candidates.extend((REPO_ROOT / "data" / "audio").rglob(VOICE_DASHBOARD_CHECKPOINT_FILENAME))

        existing = []
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                existing.append(resolved)
        if not existing:
            return None
        return max(existing, key=lambda path: path.stat().st_mtime)

    def _resolve_voice_preprocess_config(self) -> dict[str, Any]:
        candidates = []
        if self.voice_checkpoint_path is not None:
            candidates.append(self.voice_checkpoint_path.parent / VOICE_PREPROCESS_CONFIG_FILENAME)
        candidates.extend([
            self.voice_weights_path.parent / VOICE_PREPROCESS_CONFIG_FILENAME,
            self.voice_mean_path.parent / VOICE_PREPROCESS_CONFIG_FILENAME,
        ])
        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            config = load_voice_preprocess_config(resolved)
            if config is not None:
                return config
        return {}

    def predict_gesture(self, samples: list[dict[str, float]] | list[list[float]]) -> dict[str, Any]:
        normalized_samples: list[dict[str, float]] | list[list[float]] = samples
        if samples and isinstance(samples[0], (list, tuple)):
            normalized_samples = [
                {
                    "gx": float(row[0]),
                    "gy": float(row[1]),
                    "gz": float(row[2]),
                    "ax": float(row[3]),
                    "ay": float(row[4]),
                    "az": float(row[5]),
                }
                for row in samples  # type: ignore[arg-type]
            ]

        data = self.gesture_preprocessor.preprocess(normalized_samples)
        processed_window = self.gesture_preprocessor.last_capture["resampled_window"].astype(np.float32)
        raw_window = self.gesture_preprocessor.last_capture["raw_window"].astype(np.float32)
        model_input = normalize_window(processed_window, self.gesture_mean, self.gesture_std)

        tensor = torch.tensor(model_input.T[None, :, :], dtype=torch.float32)
        logits = self.gesture_model(tensor)
        result = _softmax_result(logits, GESTURE_LABELS)
        result.update(
            {
                "raw_window": raw_window.astype(float).tolist(),
                "processed_window": processed_window.astype(float).tolist(),
                "model_input_window": model_input.astype(float).tolist(),
                "debug": self.gesture_preprocessor.last_debug,
                "sample_count": int(data.count),
            }
        )
        return result

    def predict_voice(
        self,
        audio_bytes: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        suffix = _guess_suffix(filename, content_type)
        waveform = _decode_audio_bytes(audio_bytes, suffix=suffix, sample_rate=self.voice_preprocessor.sample_rate)
        mfcc_raw = self.voice_preprocessor.process_waveform(waveform, self.voice_preprocessor.sample_rate)
        mfcc = normalize_feature_matrix(mfcc_raw, self.voice_mean, self.voice_std)
        tensor = torch.tensor(mfcc[None, :, :], dtype=torch.float32)
        logits = self.voice_model(tensor)
        result = _softmax_result(logits, self.voice_labels)
        capture = self.voice_preprocessor.last_capture or {}
        focused_waveform = np.asarray(
            capture.get("focused_waveform", capture.get("normalized_waveform", waveform)),
            dtype=np.float32,
        )
        result.update(
            {
                "raw_waveform": _downsample_1d(capture.get("raw_waveform", waveform)),
                "normalized_waveform": _downsample_1d(focused_waveform),
                "normalized_waveform_audio": focused_waveform.astype(float).tolist(),
                "normalized_waveform_wav_base64": base64.b64encode(
                    _encode_wav_bytes(focused_waveform, sample_rate=self.voice_preprocessor.sample_rate)
                ).decode("ascii"),
                "sample_rate": int(self.voice_preprocessor.sample_rate),
                "mfcc_raw": np.asarray(capture.get("mfcc", mfcc_raw), dtype=np.float32).astype(float).tolist(),
                "mfcc": mfcc.astype(float).tolist(),
                "debug": self.voice_preprocessor.last_debug,
                "filename": filename,
            }
        )
        return result

    @staticmethod
    def decode_base64_audio(payload: dict[str, Any]) -> tuple[bytes, str | None, str | None]:
        encoded = payload.get("audio_base64")
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("audio_base64 is required")
        try:
            audio_bytes = base64.b64decode(encoded)
        except Exception as exc:  # noqa: BLE001
            raise ValueError("Invalid base64 audio payload") from exc
        filename = payload.get("filename")
        content_type = payload.get("content_type")
        return audio_bytes, filename if isinstance(filename, str) else None, content_type if isinstance(content_type, str) else None

    @staticmethod
    def to_json(data: dict[str, Any]) -> bytes:
        return json.dumps(data, ensure_ascii=True).encode("utf-8")

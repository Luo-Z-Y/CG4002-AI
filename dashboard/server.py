from __future__ import annotations

import argparse
import base64
import json
import threading
import time
import uuid
import wave
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from runtime import GESTURE_LABELS, VOICE_LABELS, LocalAiRuntime, _decode_audio_bytes


STATIC_DIR = Path(__file__).resolve().parent / "static"
DATA_ROOT = Path(__file__).resolve().parent / "data"
RUNTIME: LocalAiRuntime | None = None
STORE: "DashboardStore | None" = None
STATE_LOCK = threading.Lock()
LATEST_STATE: dict[str, Any] = {
    "gesture": None,
    "voice": None,
    "updated_at": None,
    "session_dir": None,
}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _decode_base64_to_bytes(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def _write_audio_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _write_wav_from_float_list(path: Path, values: list[float], sample_rate: int) -> None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    pcm = np.clip(arr, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_out:
        wav_out.setnchannels(1)
        wav_out.setsampwidth(2)
        wav_out.setframerate(sample_rate)
        wav_out.writeframes(pcm.tobytes())


def _write_gesture_packet_file(path: Path, samples: list[list[float]]) -> None:
    lines = [
        "==== Gesture Packet ====",
        f"Sample count: {len(samples)}",
        "Idx | gx | gy | gz | ax | ay | az",
    ]
    for idx, row in enumerate(samples):
        values = [float(value) for value in row[:6]]
        lines.append(
            f"{idx} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
            f"{values[3]:.6f} | {values[4]:.6f} | {values[5]:.6f}"
        )
    lines.append("========================")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class DashboardStore:
    GESTURE_FILE_STEMS = {
        "Raise": "raise",
        "Shake": "shake",
        "Chop": "chop",
        "Stir": "stir",
        "Swing": "swing",
        "Punch": "punch",
    }

    def __init__(self, root: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        self.root = root.resolve()
        self.session_dir = (self.root / stamp).resolve()
        self.gesture_dir = self.session_dir / "gesture"
        self.voice_dir = self.session_dir / "voice"
        self.pending_gesture: dict[str, dict[str, Any]] = {}
        self.reviewed_gesture: dict[str, dict[str, Any]] = {}
        self.pending_voice: dict[str, dict[str, Any]] = {}
        self.reviewed_voice: dict[str, dict[str, Any]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self.gesture_dir.mkdir(parents=True, exist_ok=True)
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        for label in GESTURE_LABELS:
            (self.gesture_dir / self._label_slug(label)).mkdir(parents=True, exist_ok=True)
        for label in VOICE_LABELS:
            (self.voice_dir / self._label_slug(label)).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sample_id(kind: str) -> str:
        return f"{kind}-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _suffix(filename: str | None, content_type: str | None) -> str:
        if filename:
            suffix = Path(filename).suffix.strip()
            if suffix:
                return suffix.lower()
        if content_type and "wav" in content_type:
            return ".wav"
        if content_type and "webm" in content_type:
            return ".webm"
        if content_type and "m4a" in content_type:
            return ".m4a"
        return ".bin"

    @staticmethod
    def _label_slug(label: str) -> str:
        return label.strip().lower().replace(" ", "_")

    def _next_voice_path(self, label: str) -> Path:
        slug = self._label_slug(label)
        label_dir = self.voice_dir / slug
        existing = sorted(label_dir.glob(f"{slug}_*.wav"))
        max_idx = 0
        for path in existing:
            stem = path.stem
            suffix = stem.removeprefix(f"{slug}_")
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
        return label_dir / f"{slug}_{max_idx + 1}.wav"

    def _next_gesture_path(self, label: str) -> Path:
        stem = self.GESTURE_FILE_STEMS.get(label, self._label_slug(label))
        label_dir = self.gesture_dir / self._label_slug(label)
        existing = sorted(label_dir.glob(f"{stem}_*.txt"))
        max_idx = 0
        for path in existing:
            suffix = path.stem.removeprefix(f"{stem}_")
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
        return label_dir / f"{stem}_{max_idx + 1}.txt"

    @staticmethod
    def _review_stub(predicted_label: str) -> dict[str, Any]:
        return {
            "status": "pending",
            "is_correct": None,
            "predicted_label": predicted_label,
            "correct_label": None,
            "reviewed_at": None,
        }

    def create_gesture_record(self, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        sample_id = self._sample_id("gesture")
        review = self._review_stub(str(result.get("label", "")))
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("Gesture payload must include non-empty samples")
        self.pending_gesture[sample_id] = {
            "samples": samples,
            "predicted_label": str(result.get("label", "")),
            "source": result.get("source"),
            "created_at": time.time(),
        }
        return {
            "sample_id": sample_id,
            "sample_dir": str(self.gesture_dir),
            "review": review,
        }

    def create_voice_record(
        self,
        payload: dict[str, Any],
        result: dict[str, Any],
        audio_bytes: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> dict[str, Any]:
        sample_id = self._sample_id("voice")
        suffix = self._suffix(filename, content_type)
        sample_rate = int(result.get("sample_rate", 16000))
        waveform = _decode_audio_bytes(audio_bytes, suffix=suffix, sample_rate=sample_rate)
        review = self._review_stub(str(result.get("label", "")))
        self.pending_voice[sample_id] = {
            "waveform": waveform.astype(np.float32),
            "sample_rate": sample_rate,
            "filename": filename,
            "content_type": content_type,
            "predicted_label": str(result.get("label", "")),
            "source": result.get("source"),
            "created_at": time.time(),
        }
        return {
            "sample_id": sample_id,
            "sample_dir": str(self.voice_dir),
            "review": review,
        }

    def label_sample(self, kind: str, sample_id: str, is_correct: bool, correct_label: str | None) -> dict[str, Any]:
        if kind == "gesture":
            if sample_id in self.reviewed_gesture:
                return self.reviewed_gesture[sample_id]
            sample = self.pending_gesture.pop(sample_id, None)
            if sample is None:
                raise FileNotFoundError(f"Gesture sample not found: {sample_id}")
            predicted = str(sample.get("predicted_label", ""))
            resolved_label = predicted if is_correct else correct_label
            if not isinstance(resolved_label, str) or not resolved_label:
                raise ValueError("Gesture label could not be resolved")
            dst = self._next_gesture_path(resolved_label)
            _write_gesture_packet_file(dst, sample["samples"])
            review = {
                "status": "reviewed",
                "is_correct": bool(is_correct),
                "predicted_label": predicted,
                "correct_label": resolved_label,
                "reviewed_at": time.time(),
                "saved_path": str(dst),
            }
            self.reviewed_gesture[sample_id] = review
            return review

        if kind == "voice":
            if sample_id in self.reviewed_voice:
                return self.reviewed_voice[sample_id]
            sample = self.pending_voice.pop(sample_id, None)
            if sample is None:
                raise FileNotFoundError(f"Voice sample not found: {sample_id}")
            predicted = str(sample.get("predicted_label", ""))
            resolved_label = predicted if is_correct else correct_label
            if not isinstance(resolved_label, str) or not resolved_label:
                raise ValueError("Voice label could not be resolved")
            dst = self._next_voice_path(resolved_label)
            _write_wav_from_float_list(dst, sample["waveform"].tolist(), int(sample["sample_rate"]))
            review = {
                "status": "reviewed",
                "is_correct": bool(is_correct),
                "predicted_label": predicted,
                "correct_label": resolved_label,
                "reviewed_at": time.time(),
                "saved_path": str(dst),
            }
            self.reviewed_voice[sample_id] = review
            return review
        raise ValueError(f"Unsupported review kind: {kind}")


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CG4002Dashboard/0.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        self._send_bytes(_json_bytes(payload), "application/json; charset=utf-8", status=status)

    def _serve_static(self, rel_path: str) -> None:
        path = STATIC_DIR / rel_path
        if not path.exists() or not path.is_file():
            self._send_json({"error": f"Static file not found: {rel_path}"}, status=HTTPStatus.NOT_FOUND)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
        }.get(path.suffix, "application/octet-stream")
        self._send_bytes(path.read_bytes(), content_type)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._serve_static("index.html")
            return
        if self.path == "/app.js":
            self._serve_static("app.js")
            return
        if self.path == "/styles.css":
            self._serve_static("styles.css")
            return
        if self.path == "/api/state":
            with STATE_LOCK:
                payload = {
                    "gesture_labels": GESTURE_LABELS,
                    "voice_labels": VOICE_LABELS,
                    **LATEST_STATE,
                }
            self._send_json(payload)
            return
        self._send_json({"error": f"Unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/gesture/infer":
                payload = self._read_json()
                samples = payload.get("samples")
                if not isinstance(samples, list) or not samples:
                    raise ValueError("samples must be a non-empty array")

                if RUNTIME is None:
                    raise RuntimeError("Runtime not initialised")
                result = RUNTIME.predict_gesture(samples)
                result["source"] = payload.get("source", "gesture-client")
                result["updated_at"] = time.time()
                if STORE is None:
                    raise RuntimeError("Store not initialised")
                capture = STORE.create_gesture_record(payload, result)
                result.update(capture)
                with STATE_LOCK:
                    LATEST_STATE["gesture"] = result
                    LATEST_STATE["updated_at"] = result["updated_at"]
                self._send_json(result)
                return

            if self.path == "/api/voice/infer":
                payload = self._read_json()
                if RUNTIME is None:
                    raise RuntimeError("Runtime not initialised")
                audio_bytes, filename, content_type = RUNTIME.decode_base64_audio(payload)
                result = RUNTIME.predict_voice(audio_bytes, filename=filename, content_type=content_type)
                result["source"] = payload.get("source", "voice-web")
                result["updated_at"] = time.time()
                if STORE is None:
                    raise RuntimeError("Store not initialised")
                capture = STORE.create_voice_record(payload, result, audio_bytes, filename, content_type)
                result.update(capture)
                with STATE_LOCK:
                    LATEST_STATE["voice"] = result
                    LATEST_STATE["updated_at"] = result["updated_at"]
                self._send_json(result)
                return

            if self.path == "/api/review":
                payload = self._read_json()
                kind = payload.get("kind")
                sample_id = payload.get("sample_id")
                is_correct = payload.get("is_correct")
                correct_label = payload.get("correct_label")
                if kind not in {"gesture", "voice"}:
                    raise ValueError("kind must be 'gesture' or 'voice'")
                if not isinstance(sample_id, str) or not sample_id:
                    raise ValueError("sample_id is required")
                if not isinstance(is_correct, bool):
                    raise ValueError("is_correct must be boolean")
                valid_labels = GESTURE_LABELS if kind == "gesture" else VOICE_LABELS
                if not is_correct:
                    if not isinstance(correct_label, str) or correct_label not in valid_labels:
                        raise ValueError("correct_label is required when the prediction is wrong")
                else:
                    correct_label = None
                if STORE is None:
                    raise RuntimeError("Store not initialised")
                review = STORE.label_sample(kind, sample_id, is_correct, correct_label)
                with STATE_LOCK:
                    current = LATEST_STATE.get(kind)
                    if isinstance(current, dict) and current.get("sample_id") == sample_id:
                        current["review"] = review
                self._send_json({"ok": True, "review": review, "sample_id": sample_id, "kind": kind})
                return

            self._send_json({"error": f"Unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local CG4002 AI testing dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gesture-weights", default=None, help="Optional path to gesture_cnn_weights.h")
    parser.add_argument("--gesture-mean", default=None, help="Optional path to gesture_mean.npy for software IMU normalisation")
    parser.add_argument("--gesture-std", default=None, help="Optional path to gesture_std.npy for software IMU normalisation")
    parser.add_argument("--voice-weights", default=None, help="Optional path to voice_cnn_weights.h")
    parser.add_argument("--voice-mean", default=None, help="Optional path to voice_mean.npy for software MFCC normalisation")
    parser.add_argument("--voice-std", default=None, help="Optional path to voice_std.npy for software MFCC normalisation")
    parser.add_argument("--data-root", default=str(DATA_ROOT), help="Root folder for captured dashboard samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global RUNTIME, STORE
    STORE = DashboardStore(Path(args.data_root))
    RUNTIME = LocalAiRuntime(
        gesture_weights=args.gesture_weights,
        voice_weights=args.voice_weights,
        gesture_mean=args.gesture_mean,
        gesture_std=args.gesture_std,
        voice_mean=args.voice_mean,
        voice_std=args.voice_std,
    )
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    with STATE_LOCK:
        LATEST_STATE["session_dir"] = str(STORE.session_dir)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print(f"Dashboard data session: {STORE.session_dir}")
    print(f"Gesture weights: {RUNTIME.gesture_weights_path}")
    print(f"Gesture mean: {RUNTIME.gesture_mean_path}")
    print(f"Gesture std: {RUNTIME.gesture_std_path}")
    print(f"Voice weights: {RUNTIME.voice_weights_path}")
    print(f"Voice mean: {RUNTIME.voice_mean_path}")
    print(f"Voice std: {RUNTIME.voice_std_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

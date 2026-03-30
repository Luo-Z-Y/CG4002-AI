from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from runtime import GESTURE_LABELS, VOICE_LABELS, LocalAiRuntime


STATIC_DIR = Path(__file__).resolve().parent / "static"
RUNTIME: LocalAiRuntime | None = None
STATE_LOCK = threading.Lock()
LATEST_STATE: dict[str, Any] = {
    "gesture": None,
    "voice": None,
    "updated_at": None,
}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True).encode("utf-8")


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
                with STATE_LOCK:
                    LATEST_STATE["voice"] = result
                    LATEST_STATE["updated_at"] = result["updated_at"]
                self._send_json(result)
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
    parser.add_argument("--voice-weights", default=None, help="Optional path to voice_cnn_weights.h")
    parser.add_argument("--voice-mean", default=None, help="Optional path to voice_mean.npy for software MFCC normalisation")
    parser.add_argument("--voice-std", default=None, help="Optional path to voice_std.npy for software MFCC normalisation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global RUNTIME
    RUNTIME = LocalAiRuntime(
        gesture_weights=args.gesture_weights,
        voice_weights=args.voice_weights,
        voice_mean=args.voice_mean,
        voice_std=args.voice_std,
    )
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running at http://{args.host}:{args.port}")
    print(f"Gesture weights: {RUNTIME.gesture_weights_path}")
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

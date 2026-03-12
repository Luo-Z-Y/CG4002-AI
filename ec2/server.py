from __future__ import annotations

"""Minimal EC2 WebSocket server for the Ultra96 AI bridge."""

import argparse
import asyncio
import contextlib
import json
from pathlib import Path

try:
    import websockets
except ImportError as exc:
    raise SystemExit("The `websockets` package is required for ec2/server.py") from exc

from ai import AIReceiver, AISender


def _default_imu_window() -> list[dict[str, float]]:
    """Return a simple 60-sample IMU window matching the current Ultra96 expectation."""

    return [
        {
            "y": float(i % 6),
            "p": float((i % 5) * 0.1),
            "r": float((i % 4) * -0.1),
            "ax": 0.01 * i,
            "ay": -0.02 * i,
            "az": 9.81,
        }
        for i in range(60)
    ]


def _load_imu_samples(path: str | None) -> list[dict[str, float]]:
    if path is None:
        return _default_imu_window()

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("IMU JSON must be a top-level array of sample objects")
    return data


class EC2BridgeServer:
    def __init__(
        self,
        send_test_imu: bool,
        imu_json: str | None,
        m4a_path: str | None,
        ffmpeg_path: str,
        ws_path: str,
    ) -> None:
        self.send_test_imu = send_test_imu
        self.imu_json = imu_json
        self.m4a_path = m4a_path
        self.ffmpeg_path = ffmpeg_path
        self.ws_path = self._normalize_path(ws_path)

    @staticmethod
    def _normalize_path(path: str) -> str:
        normalized = path.strip() or "/"
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        if len(normalized) > 1 and normalized.endswith("/"):
            normalized = normalized[:-1]
        return normalized

    def _request_path(self, websocket, path: str | None) -> str:
        if path is not None:
            raw = path
        else:
            request = getattr(websocket, "request", None)
            raw = getattr(request, "path", None)
            if raw is None:
                raw = getattr(websocket, "path", "/")
        return self._normalize_path(str(raw).split("?", 1)[0])

    async def handle_client(self, websocket, path: str | None = None) -> None:
        request_path = self._request_path(websocket, path)
        if request_path != self.ws_path:
            remote = getattr(websocket, "remote_address", None)
            print(f"Rejected websocket from {remote}: path {request_path} does not match {self.ws_path}")
            await websocket.close(code=1008, reason=f"expected websocket path {self.ws_path}")
            return

        remote = getattr(websocket, "remote_address", None)
        print(f"Ultra96 connected: {remote} path={request_path}")

        sender = AISender(websocket)
        receiver = AIReceiver()

        receiver.on_action(self._on_action)
        receiver.on_pokemon(self._on_pokemon)

        if self.send_test_imu or self.imu_json is not None:
            samples = _load_imu_samples(self.imu_json)
            print(f"Sending IMU packet with {len(samples)} samples")
            await sender.send_imu(samples)

        if self.m4a_path:
            print(f"Sending voice_mfcc derived from {self.m4a_path}")
            await sender.send_m4a_as_mfcc(self.m4a_path, ffmpeg_path=self.ffmpeg_path)

        try:
            await receiver.listen_forever(websocket)
        except websockets.ConnectionClosed:
            print("Ultra96 disconnected")

    @staticmethod
    async def _on_action(_, data) -> None:
        print(f"ACTION label={data.label} confidence={data.confidence:.4f}")

    @staticmethod
    async def _on_pokemon(_, data) -> None:
        print(f"POKEMON label={data.label} confidence={data.confidence:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EC2 WebSocket server for the Ultra96 AI bridge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--path", default="/ai/ws", help="WebSocket path prefix for the AI bridge, e.g. /ai/ws")
    parser.add_argument(
        "--send-test-imu",
        action="store_true",
        help="Send one built-in 60-sample IMU packet immediately after Ultra96 connects.",
    )
    parser.add_argument(
        "--imu-json",
        default=None,
        help="Optional path to a JSON array of IMU samples. Used instead of the built-in IMU packet.",
    )
    parser.add_argument(
        "--m4a-path",
        default=None,
        help="Optional path to an .m4a clip. EC2 converts it to MFCC and sends one voice_mfcc packet on connect.",
    )
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    handler = EC2BridgeServer(
        send_test_imu=args.send_test_imu,
        imu_json=args.imu_json,
        m4a_path=args.m4a_path,
        ffmpeg_path=args.ffmpeg_path,
        ws_path=args.path,
    )

    async with websockets.serve(handler.handle_client, args.host, args.port, max_size=None):
        print(f"EC2 AI bridge listening on ws://{args.host}:{args.port}{handler.ws_path}")
        print("This port is dedicated to the Ultra96 AI WebSocket link.")
        await asyncio.Future()


def main() -> None:
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        with contextlib.suppress(Exception):
            print("\nServer stopped")


if __name__ == "__main__":
    main()
